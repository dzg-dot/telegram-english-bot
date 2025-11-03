import os
import re
import json
import time
import hmac
import hashlib
import logging
import threading
import asyncio
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv
from openai import OpenAI
from flask import Flask

from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup
)
from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler,
    ContextTypes, filters
)

# ========== LOGGING ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ========== STARTUP HOOKS ==========
async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.exception("Exception while handling an update:", exc_info=context.error)
    # best-effort error log
    try:
        uid = getattr(getattr(update, "effective_user", None), "id", "n/a")
        await log_event(context, "error", uid, {"error": str(context.error)})
    except Exception:
        pass

async def on_startup(app: Application):
    # Xóa webhook để dùng long-polling
    await app.bot.delete_webhook(drop_pending_updates=True)
    logger.info("Webhook deleted, switching to long-polling.")

# ========== ENV & CLIENTS ==========
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN missing")

USE_OPENROUTER = os.getenv("USE_OPENROUTER", "True").lower() == "true"
OR_KEY = os.getenv("OPENROUTER_API_KEY")
OA_KEY = os.getenv("OPENAI_API_KEY")

# NEW: Google Sheet webhook + salt
GSHEET_WEBHOOK = os.getenv("GSHEET_WEBHOOK", "").strip()
LOG_SALT = os.getenv("LOG_SALT", "").strip()

logger.info("DEBUG => USE_OPENROUTER=%s | OR_KEY? %s | OA_KEY? %s | GSHEET? %s",
            USE_OPENROUTER, bool(OR_KEY), bool(OA_KEY), bool(GSHEET_WEBHOOK))

httpx_client = httpx.Client(
    timeout=httpx.Timeout(connect=30.0, read=90.0, write=90.0, pool=90.0),
    limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
)

if USE_OPENROUTER:
    if not OR_KEY:
        raise RuntimeError("OPENROUTER_API_KEY missing")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OR_KEY,
        http_client=httpx_client,
        default_headers={
            "HTTP-Referer": "https://t.me/SearchVocabBot",
            "X-Title": "School English Bot",
        },
    )
    MODEL_NAME = "openai/gpt-4o-mini"
else:
    if not OA_KEY:
        raise RuntimeError("OPENAI_API_KEY missing")
    client = OpenAI(api_key=OA_KEY, http_client=httpx_client)
    MODEL_NAME = "gpt-3.5-turbo"

# ========== CONSTANTS & HELPERS ==========
DEFAULT_LANG = "auto"   # auto|en|ru
MAX_HISTORY = 10

ALLOWED_MODES = {"chat", "vocab", "reading", "grammar", "practice", "talk"}

BANNED_KEYWORDS = [
    r"\bsex\b", r"\bporn\b", r"\berotic\b",
    r"\bviolence\b", r"\bsuicide\b", r"\bself[- ]?harm\b",
    r"\bdrugs?\b", r"\balcohol\b", r"\bgamble\b",
    r"\bextremis(m|t)\b"
]

GRADE_TO_CEFR = {"6": "A2", "7": "A2+", "8": "B1-", "9": "B1"}
DEFAULT_DIALOGUE_LIMIT = 10  # talk mode turn limit

POLICY_CHAT = (
    "You are a safe, school-appropriate assistant for grades 6–9. "
    "No markdown bold or headings. English by default; if the user writes Russian, respond in Russian. "
    "Be friendly, concise, and helpful. If a request is unsafe/off-topic (adult/violent/etc.), refuse and steer to study topics. "
    "Level A2–B1."
)

POLICY_STUDY = (
    "You are an English study assistant for grades 6–9 (CEFR A2–B1). "
    "No markdown bold or headings. Keep answers short, safe, and age-appropriate."
)

CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")

def detect_lang(text: str) -> str:
    return "ru" if CYRILLIC_RE.search(text or "") else "en"

def trim(s: str, max_chars:  int = 1000) -> str:
    s = re.sub(r"\n{3,}", "\n\n", (s or "").strip())
    return s if len(s) <= max_chars else (s[:max_chars].rstrip() + "…")

def blocked(text: str) -> bool:
    for pat in BANNED_KEYWORDS:
        if re.search(pat, text or "", flags=re.IGNORECASE):
            return True
    return False

async def ask_openai(messages, max_tokens=500, temperature=0.3, model=None):
    model = model or MODEL_NAME
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                max_tokens=max_tokens, temperature=temperature
            )
            return resp.choices[0].message.content
        except Exception as e1:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            # fallback
            base_url = getattr(client, "base_url", "") or ""
            fb = "openai/gpt-3.5-turbo" if "openrouter.ai" in base_url else "gpt-3.5-turbo"
            try:
                resp = client.chat.completions.create(
                    model=fb, messages=messages,
                    max_tokens=max_tokens, temperature=temperature
                )
                return resp.choices[0].message.content
            except Exception:
                return f"[OpenAI error] {type(e1).__name__}: {e1}"

def extract_json(s: str):
    s = (s or "").strip()
    if "```" in s:
        parts = s.split("```")
        for i in range(len(parts)-1):
            block = parts[i+1]
            if block.lstrip().startswith("json"):
                try:
                    return json.loads(block.split("\n", 1)[1])
                except Exception:
                    pass
            try:
                return json.loads(block)
            except Exception:
                continue
    return json.loads(s)

# ========== USER PREFS / STATE ==========
user_prefs = {}  # per-user persistent (in RAM)
# context.user_data: per-chat session state (practice state, history...)

def get_prefs(user_id: int):
    if user_id not in user_prefs:
        user_prefs[user_id] = {
            "mode": "chat",
            "lang": DEFAULT_LANG,
            "grade": "7",
            "cefr": GRADE_TO_CEFR["7"],
            "dialogue_limit": DEFAULT_DIALOGUE_LIMIT,
        }
    return user_prefs[user_id]

# ========== GOOGLE SHEET LOGGING ==========
async def log_event(context: ContextTypes.DEFAULT_TYPE, event: str, user_id, extra: dict | None = None):
    """Fire-and-forget logging to Google Apps Script Web App."""
    if not GSHEET_WEBHOOK:
        return
    try:
        prefs = get_prefs(int(user_id)) if isinstance(user_id, int) else {}
        ts = datetime.now(timezone.utc).isoformat()
        payload = {
            "timestamp": ts,
            "user_id": str(user_id),
            "event": event,
            "mode": prefs.get("mode"),
            "lang": prefs.get("lang"),
            "grade": prefs.get("grade"),
            "cefr": prefs.get("cefr"),
            "extra": extra or {}
        }
        # optional signature
        sig_src = f"{payload['user_id']}|{payload['event']}|{payload['timestamp']}|{LOG_SALT}"
        signature = hmac.new(LOG_SALT.encode("utf-8"), sig_src.encode("utf-8"), hashlib.sha256).hexdigest() if LOG_SALT else ""
        headers = {"X-Log-Signature": signature} if signature else {}
        # run sync httpx in thread (non-blocking)
        await asyncio.to_thread(httpx_client.post, GSHEET_WEBHOOK, json=payload, headers=headers, timeout=10.0)
    except Exception as e:
        logger.warning("log_event failed: %s", e)

# ========== UI (INLINE MENUS) ==========
def root_menu(lang: str) -> InlineKeyboardMarkup:
    if lang == "ru":
        kb = [
            [InlineKeyboardButton("📚 Слова", callback_data="menu:mode:vocab"),
             InlineKeyboardButton("📖 Чтение", callback_data="menu:mode:reading")],
            [InlineKeyboardButton("⚙️ Грамматика", callback_data="menu:mode:grammar"),
             InlineKeyboardButton("📝 Практика", callback_data="menu:mode:practice")],
            [InlineKeyboardButton("💬 Разговор", callback_data="menu:mode:talk")],
            [InlineKeyboardButton("🏫 Класс", callback_data="menu:grade"),
             InlineKeyboardButton("🌐 Язык", callback_data="menu:lang")],
            [InlineKeyboardButton("📋 Меню", callback_data="menu:root")]  # self to refresh
        ]
    else:
        kb = [
            [InlineKeyboardButton("📚 Vocabulary", callback_data="menu:mode:vocab"),
             InlineKeyboardButton("📖 Reading", callback_data="menu:mode:reading")],
            [InlineKeyboardButton("⚙️ Grammar", callback_data="menu:mode:grammar"),
             InlineKeyboardButton("📝 Practice", callback_data="menu:mode:practice")],
            [InlineKeyboardButton("💬 Talk", callback_data="menu:mode:talk")],
            [InlineKeyboardButton("🏫 Grade", callback_data="menu:grade"),
             InlineKeyboardButton("🌐 Language", callback_data="menu:lang")],
            [InlineKeyboardButton("📋 Back to Menu", callback_data="menu:root")]
        ]
    return InlineKeyboardMarkup(kb)

def lang_menu() -> InlineKeyboardMarkup:
    kb = [
        [InlineKeyboardButton("English", callback_data="set_lang:en"),
         InlineKeyboardButton("Русский", callback_data="set_lang:ru")],
        [InlineKeyboardButton("⬅️ Back", callback_data="menu:root")]
    ]
    return InlineKeyboardMarkup(kb)

def grade_menu() -> InlineKeyboardMarkup:
    kb = [
        [InlineKeyboardButton("6", callback_data="set_grade:6"),
         InlineKeyboardButton("7", callback_data="set_grade:7"),
         InlineKeyboardButton("8", callback_data="set_grade:8"),
         InlineKeyboardButton("9", callback_data="set_grade:9")],
        [InlineKeyboardButton("⬅️ Back", callback_data="menu:root")]
    ]
    return InlineKeyboardMarkup(kb)

def practice_menu(lang="en") -> InlineKeyboardMarkup:
    if lang == "ru":
        text = ["Тест (A–D)", "Формы глагола", "Пропуски",
                "Словообразование", "Исправь ошибку", "Порядок слов"]
    else:
        text = ["Multiple Choice", "Verb Forms", "Gap Fill",
                "Word Formation", "Error Correction", "Sentence Ordering"]
    kb = [
        [InlineKeyboardButton(f"🅰 {text[0]}", callback_data="practice:type:mcq"),
         InlineKeyboardButton(f"🔤 {text[1]}", callback_data="practice:type:verb")],
        [InlineKeyboardButton(f"🕳 {text[2]}", callback_data="practice:type:gap"),
         InlineKeyboardButton(f"🧱 {text[3]}", callback_data="practice:type:wordform")],
        [InlineKeyboardButton(f"❌ {text[4]}", callback_data="practice:type:error"),
         InlineKeyboardButton(f"🔁 {text[5]}", callback_data="practice:type:order")],
        [InlineKeyboardButton("⬅️ Back", callback_data="menu:root")]
    ]
    return InlineKeyboardMarkup(kb)

def talk_topics_menu(lang="en") -> InlineKeyboardMarkup:
    if lang == "ru":
        lbl = ["Быт", "Школа", "Хобби", "Окружающая среда", "Праздники", "Семья"]
    else:
        lbl = ["Daily life", "School life", "Hobbies", "Environment", "Holidays", "Family"]
    kb = [
        [InlineKeyboardButton(lbl[0], callback_data="talk:topic:daily"),
         InlineKeyboardButton(lbl[1], callback_data="talk:topic:school")],
        [InlineKeyboardButton(lbl[2], callback_data="talk:topic:hobbies"),
         InlineKeyboardButton(lbl[3], callback_data="talk:topic:env")],
        [InlineKeyboardButton(lbl[4], callback_data="talk:topic:holidays"),
         InlineKeyboardButton(lbl[5], callback_data="talk:topic:family")],
        [InlineKeyboardButton("⬅️ Back", callback_data="menu:root")]
    ]
    return InlineKeyboardMarkup(kb)

def mcq_buttons(options):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"A) {options[0]}", callback_data="ans:A"),
         InlineKeyboardButton(f"B) {options[1]}", callback_data="ans:B")],
        [InlineKeyboardButton(f"C) {options[2]}", callback_data="ans:C"),
         InlineKeyboardButton(f"D) {options[3]}", callback_data="ans:D")]
    ])

# ========== START / HELP ==========
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text_in = (update.message.text or "").strip()
    greet = "Hi there! I’m your English study buddy. How can I help you today?"
    if detect_lang(text_in) == "ru":
        greet = "Привет! Я твой помощник по английскому. Чем помочь сегодня?"
    prefs = get_prefs(update.effective_user.id)
    prefs["mode"] = "chat"
    await update.message.reply_text(greet, reply_markup=root_menu(prefs.get("lang", "en")))
    await log_event(context, "start", update.effective_user.id, {"text": text_in[:200]})

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prefs = get_prefs(update.effective_user.id)
    lang = prefs.get("lang", "en")
    msg = "Choose from the menu below." if lang != "ru" else "Выберите пункт меню ниже."
    await update.message.reply_text(msg, reply_markup=root_menu(lang))
    await log_event(context, "help", update.effective_user.id, {"lang": lang})

# ========== VOCAB ==========
async def build_vocab_card(headword: str, prefs: dict, user_text: str) -> str:
    lang_for_examples = prefs.get("lang", "auto")
    if lang_for_examples == "auto":
        lang_for_examples = detect_lang(user_text or "")
    include_ru_examples = (lang_for_examples == "ru")
    prompt = (
        "You are an English-learning assistant for grades 6–9 (CEFR A2–B1). "
        "Make a compact vocabulary card. Do not use markdown bold. "
        "Definition must be in English with a short Russian translation in parentheses.\n\n"
        f"HEADWORD: {headword}\nTARGET LEVEL: {prefs['cefr']}\n\n"
        "Format exactly:\n"
        "Word: <headword> /<IPA>/\n"
        "POS: <part of speech>\n"
        "Definition: <short English definition> (<short Russian translation>)\n"
        "Examples:\n"
        f"1) <short English example>{' (Russian translation)' if include_ru_examples else ''}\n"
        f"2) <short English example>{' (Russian translation)' if include_ru_examples else ''}\n"
        f"3) <short English example>{' (optional Russian translation)' if include_ru_examples else ' (optional)'}\n"
        "Keep under 120 words."
    )
    msgs = [{"role": "system", "content": POLICY_STUDY},
            {"role": "user", "content": prompt}]
    return await ask_openai(msgs, max_tokens=320)

# ========== PRACTICE (GENERIC) ==========
"""
practice_state schema in context.user_data["practice"]:
{
  "type": "mcq|verb|gap|wordform|error|order",
  "topic": str,
  "items": [ ... 5 items ... ],
  "idx": 0,
  "attempts": 0,
  "score": 0,
  "ui_lang": "en|ru"
}
For MCQ item:
  {"id":1,"question":"...","options":["...","...","...","..."],"answer":"A",
   "explain_en":"...","explain_ru":"..."}
For text-based item (verb/gap/wordform/error/order):
  {"id":1,"prompt":"...", "answer":"...", "explain_en":"...", "explain_ru":"..."}
"""

def normalize_answer(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\s'-]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

async def build_mcq(topic: str, ui_lang: str, level: str):
    prompt = (
        f"Create a 5-question multiple-choice quiz (4 options A–D) on '{topic}', level {level}, grades 6–9.\n"
        "Return STRICT JSON only:\n"
        "{ \"questions\": [\n"
        "{\"id\":1,\"question\":\"...\",\"options\":[\"...\",\"...\",\"...\",\"...\"],"
        "\"answer\":\"A\",\"explain_en\":\"<=25 words\",\"explain_ru\":\"<=25 words\"},\n"
        "{\"id\":2,...},{\"id\":3,...},{\"id\":4,...},{\"id\":5,...}\n"
        "]}\n"
        f"Language for 'question' and 'options': {'Russian' if ui_lang=='ru' else 'English'} (A2–B1). "
        "School-safe. Do not leak answer hints in text."
    )
    msgs = [{"role": "system", "content": POLICY_STUDY},
            {"role": "user", "content": prompt}]
    raw = await ask_openai(msgs, max_tokens=800)
    data = extract_json(raw)
    items = []
    for q in data.get("questions", []):
        items.append({
            "id": q.get("id"),
            "question": q.get("question"),
            "options": q.get("options", ["", "", "", ""]),
            "answer": q.get("answer", "A"),
            "explain_en": q.get("explain_en", ""),
            "explain_ru": q.get("explain_ru", "")
        })
    return items

async def build_text_items(ptype: str, topic: str, ui_lang: str, level: str):
    task_desc = {
        "verb": "Conjugate the verb in brackets into the correct form.",
        "gap": "Fill in the blank with one suitable word.",
        "wordform": "Complete the sentence using the correct form of the word in parentheses.",
        "error": "Find and correct the mistake in the sentence (write the corrected version).",
        "order": "Reorder the words to make a correct sentence."
    }[ptype]

    prompt = (
        f"Create 5 short {ptype} exercises on '{topic}', level {level}, grades 6–9. "
        f"Task: {task_desc}\n"
        "Return STRICT JSON only:\n"
        "{ \"items\": [\n"
        "{\"id\":1,\"prompt\":\"...\",\"answer\":\"...\",\"explain_en\":\"<=25 words\",\"explain_ru\":\"<=25 words\"},\n"
        "{\"id\":2,...},{\"id\":3,...},{\"id\":4,...},{\"id\":5,...}\n"
        "]}\n"
        f"Language for 'prompt': {'Russian' if ui_lang=='ru' else 'English'} (A2–B1). "
        "Keep answers short. School-safe."
    )
    msgs = [{"role": "system", "content": POLICY_STUDY},
            {"role": "user", "content": prompt}]
    raw = await ask_openai(msgs, max_tokens=900)
    data = extract_json(raw)
    items = []
    for it in data.get("items", []):
        items.append({
            "id": it.get("id"),
            "prompt": it.get("prompt"),
            "answer": it.get("answer"),
            "explain_en": it.get("explain_en", ""),
            "explain_ru": it.get("explain_ru", "")
        })
    return items

async def send_practice_item(update_or_query, context: ContextTypes.DEFAULT_TYPE):
    st = context.user_data.get("practice")
    if not st:
        return
    ptype = st["type"]
    idx = st["idx"]
    total = len(st["items"])
    title = f"Q{idx+1}/{total}"
    if ptype == "mcq":
        q = st["items"][idx]
        text = f"{title}\n\n{q['question']}"
        kb = mcq_buttons(q["options"])
        if isinstance(update_or_query, Update):
            await update_or_query.message.reply_text(text, reply_markup=kb)
        else:
            await update_or_query.edit_message_text(text, reply_markup=kb)
    else:
        q = st["items"][idx]
        head = "Type your answer:" if st.get("ui_lang","en") != "ru" else "Напиши ответ:"
        text = f"{title}\n\n{q['prompt']}\n\n{head}"
        if isinstance(update_or_query, Update):
            await update_or_query.message.reply_text(text)
        else:
            await update_or_query.edit_message_text(text)

async def practice_summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    st = context.user_data.get("practice")
    if not st:
        return
    lang = st.get("ui_lang", "en")
    total = len(st["items"])
    score = st.get("score", 0)
    lines = []
    if lang == "ru":
        lines.append(f"Итоги: {score}/{total}")
        lines.append("Ответы и объяснения:")
    else:
        lines.append(f"Summary: {score}/{total}")
        lines.append("Answers and explanations:")
    if st["type"] == "mcq":
        for it in st["items"]:
            expl = it["explain_ru"] if lang == "ru" and it["explain_ru"] else it["explain_en"]
            lines.append(f"Q{it['id']}: {it['answer']} — {expl}")
    else:
        for it in st["items"]:
            expl = it["explain_ru"] if lang == "ru" and it["explain_ru"] else it["explain_en"]
            lines.append(f"Q{it['id']}: {it['answer']} — {expl}")
    await update.message.reply_text("\n".join(lines))
    # log summary
    await log_event(context, "practice_done", update.effective_user.id, {
        "type": st["type"], "topic": st.get("topic"), "score": score, "total": total
    })
    context.user_data.pop("practice", None)

# ========== TALK (CONVERSATION COACH) ==========
async def talk_reply(user_text: str, topic: str, ui_lang: str):
    prompt = (
        "You are a friendly English conversation coach for a middle-school student (A2–B1). "
        f"Topic: {topic}. Respond in 1–3 sentences. "
        "Encourage and, when helpful, suggest 1–2 useful words/phrases. "
        "Correct small mistakes implicitly by reformulating. "
        "If the student uses Russian, briefly translate key phrase and encourage continuing in English. "
        "School-safe, positive tone. No markdown bold."
    )
    msgs = [
        {"role": "system", "content": POLICY_STUDY},
        {"role": "user", "content": f"Student says: {user_text}"}
    ]
    return await ask_openai([{"role": "system", "content": prompt}, *msgs], max_tokens=180)

# ========== COMMANDS ==========
async def vocab_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prefs = get_prefs(update.effective_user.id)
    prefs["mode"] = "vocab"
    await update.message.reply_text("Vocabulary mode is ON. Send me a word.")
    await log_event(context, "mode_set", update.effective_user.id, {"mode": "vocab"})

# (NEW) quick logger test
async def logtest_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ok = {"ping": "pong", "note": "manual test"}
    await log_event(context, "logtest", update.effective_user.id, ok)
    await update.message.reply_text("Logtest sent (if GSHEET_WEBHOOK is set).")

# ========== CALLBACK HANDLER ==========
async def on_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    data = q.data or ""
    await q.answer()  # tắt quay vòng

    uid = update.effective_user.id
    prefs = get_prefs(uid)
    ui_lang = prefs.get("lang", "en")
    if ui_lang == "auto":
        ui_lang = "ru" if CYRILLIC_RE.search(q.message.text or "") else "en"

    # MENUS
    if data == "menu:root":
        msg = "Back to menu." if ui_lang != "ru" else "Возврат в меню."
        await q.edit_message_text(msg, reply_markup=root_menu(ui_lang))
        # reset to chat
        prefs["mode"] = "chat"
        await log_event(context, "menu_root", uid, {})
        return

    if data == "menu:lang":
        await q.edit_message_text("Choose language:" if ui_lang != "ru" else "Выберите язык:",
                                  reply_markup=lang_menu())
        return

    if data == "menu:grade":
        await q.edit_message_text("Choose grade:" if ui_lang != "ru" else "Выберите класс:",
                                  reply_markup=grade_menu())
        return

    if data.startswith("set_lang:"):
        lang = data.split(":", 1)[1]
        prefs["lang"] = lang
        await q.edit_message_text(f"Language set to {lang.upper()}." if lang!="ru" else "Язык: Русский.",
                                  reply_markup=root_menu(lang))
        await log_event(context, "lang_set", uid, {"lang": lang})
        return

    if data.startswith("set_grade:"):
        g = data.split(":", 1)[1]
        if g in GRADE_TO_CEFR:
            prefs["grade"] = g
            prefs["cefr"] = GRADE_TO_CEFR[g]
            txt = (f"Grade set to {g}. Target level: {prefs['cefr']}."
                   if ui_lang != "ru" else f"Класс: {g}. Уровень: {prefs['cefr']}.")
            await q.edit_message_text(txt, reply_markup=root_menu(ui_lang))
            await log_event(context, "grade_set", uid, {"grade": g, "cefr": prefs["cefr"]})
        else:
            await q.edit_message_text("Invalid grade.", reply_markup=root_menu(ui_lang))
        return

    # MODE ENTRIES
    if data.startswith("menu:mode:"):
        mode = data.split(":")[-1]  # vocab/reading/grammar/practice/talk
        prefs["mode"] = mode
        await log_event(context, "mode_set", uid, {"mode": mode})
        if mode == "vocab":
            txt = "Vocabulary mode is ON. Send a word." if ui_lang != "ru" else "Режим Слова. Отправь слово."
            await q.edit_message_text(txt, reply_markup=root_menu(ui_lang))
        elif mode == "reading":
            txt = "Reading mode is ON. Send a topic for a short passage." if ui_lang != "ru" else "Режим Чтение. Отправь тему."
            await q.edit_message_text(txt, reply_markup=root_menu(ui_lang))
        elif mode == "grammar":
            txt = "Grammar mode is ON. Send a grammar point (e.g., Present Simple)." if ui_lang != "ru" else "Режим Грамматика. Отправь тему (напр., Present Simple)."
            await q.edit_message_text(txt, reply_markup=root_menu(ui_lang))
        elif mode == "practice":
            txt = "Choose an exercise type:" if ui_lang != "ru" else "Выберите тип упражнения:"
            await q.edit_message_text(txt, reply_markup=practice_menu(ui_lang))
        elif mode == "talk":
            txt = "Choose a topic to talk about:" if ui_lang != "ru" else "Выберите тему для разговора:"
            await q.edit_message_reply_markup(reply_markup=talk_topics_menu(ui_lang))
            await q.edit_message_text(txt)
        return

    # PRACTICE TYPE SELECT
    if data.startswith("practice:type:"):
        ptype = data.split(":")[-1]  # mcq/verb/gap/wordform/error/order
        context.user_data["practice"] = {
            "type": ptype,
            "topic": None,
            "items": [],
            "idx": 0,
            "attempts": 0,
            "score": 0,
            "ui_lang": ui_lang
        }
        ask = "Send me a topic (e.g., pollution)." if ui_lang != "ru" else "Отправь тему (например, pollution)."
        await q.edit_message_text(ask)
        await log_event(context, "practice_type_set", uid, {"ptype": ptype})
        return

    # TALK TOPIC
    if data.startswith("talk:topic:"):
        topic_key = data.split(":")[-1]
        mapping = {
            "daily": "daily life", "school": "school life", "hobbies": "hobbies",
            "env": "environment", "holidays": "holidays", "family": "family"
        }
        topic = mapping.get(topic_key, "daily life")
        prefs["mode"] = "talk"
        context.user_data["talk"] = {"topic": topic, "turns": 0}
        opener = "Let’s talk! How are you today?" if ui_lang != "ru" else "Поговорим! Как твоё настроение сегодня?"
        await q.edit_message_text(f"Topic: {topic}\n\n{opener}", reply_markup=root_menu(ui_lang))
        await log_event(context, "talk_topic_set", uid, {"topic": topic})
        return

    # PRACTICE ANSWER (MCQ)
    if data.startswith("ans:"):
        st = context.user_data.get("practice")
        if not st or st.get("type") != "mcq":
            await q.edit_message_text("No active multiple-choice exercise.", reply_markup=root_menu(ui_lang))
            return
        choice = data.split(":", 1)[1]
        idx = st["idx"]
        qitem = st["items"][idx]
        correct = qitem["answer"]
        if choice == correct:
            st["score"] += 1
            st["attempts"] = 0
            expl = qitem["explain_ru"] if ui_lang == "ru" and qitem["explain_ru"] else qitem["explain_en"]
            ok = "Correct!" if ui_lang != "ru" else "Верно!"
            await q.edit_message_text(f"{ok}\n{expl}".strip())
            await log_event(context, "practice_answer", uid, {
                "ptype": "mcq", "qid": qitem.get("id"), "correct": True
            })
            st["idx"] += 1
        else:
            st["attempts"] += 1
            if st["attempts"] < 2:
                msg = "Not quite. Try again." if ui_lang != "ru" else "Почти. Попробуй еще раз."
                await q.edit_message_text(msg)
                await log_event(context, "practice_answer", uid, {
                    "ptype": "mcq", "qid": qitem.get("id"), "correct": False, "retry": True
                })
            else:
                st["attempts"] = 0
                ans = f"The correct answer is {correct}." if ui_lang != "ru" else f"Правильный ответ: {correct}."
                expl = qitem["explain_ru"] if ui_lang == "ru" and qitem["explain_ru"] else qitem["explain_en"]
                await q.edit_message_text(f"{ans}\n{expl}".strip())
                await log_event(context, "practice_answer", uid, {
                    "ptype": "mcq", "qid": qitem.get("id"), "correct": False, "revealed": True
                })
                st["idx"] += 1

        # next or summary
        if st["idx"] >= len(st["items"]):
            dummy_update = Update(update.update_id, message=q.message)  # reuse summary
            await practice_summary(dummy_update, context)
        else:
            await send_practice_item(q, context)
        return

# ========== FREE TEXT HANDLER ==========
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text or ""
    if blocked(user_message):
        return await update.message.reply_text(
            "⛔ That's outside our classroom scope. Please try vocabulary, reading, grammar, or a practice topic."
        )

    uid = update.effective_user.id
    prefs = get_prefs(uid)

    # Detect UI language
    lang = prefs.get("lang", "en")
    if lang == "auto":
        lang = detect_lang(user_message)

    # PRACTICE FLOW: nếu đã chọn type nhưng chưa tạo items -> coi message là topic
    st = context.user_data.get("practice")
    if st and not st.get("items"):
        topic = user_message.strip() or "school life"
        st["topic"] = topic
        level = prefs["cefr"]
        try:
            if st["type"] == "mcq":
                st["items"] = await build_mcq(topic, lang, level)
            else:
                st["items"] = await build_text_items(st["type"], topic, lang, level)
        except Exception:
            msg = "Failed to build exercises. Please try another topic." if lang != "ru" else "Не удалось создать задания. Попробуй другую тему."
            await update.message.reply_text(msg)
            await log_event(context, "practice_build_fail", uid, {"ptype": st["type"], "topic": topic})
            return
        if not st["items"]:
            msg = "No items generated. Try another topic." if lang != "ru" else "Задания не созданы. Попробуй другую тему."
            await update.message.reply_text(msg)
            await log_event(context, "practice_empty", uid, {"ptype": st["type"], "topic": topic})
            return
        st["idx"] = 0
        st["attempts"] = 0
        st["score"] = 0
        st["ui_lang"] = lang
        await log_event(context, "practice_built", uid, {"ptype": st["type"], "topic": topic, "count": len(st["items"])})
        return await send_practice_item(update, context)

    # PRACTICE FLOW (text input types) — nhận câu trả lời
    if st and st.get("items") and st["type"] != "mcq":
        idx = st["idx"]
        if idx < len(st["items"]):
            qitem = st["items"][idx]
            user_ans = normalize_answer(user_message)
            gold = normalize_answer(qitem["answer"])
            if user_ans == gold:
                st["score"] += 1
                st["attempts"] = 0
                expl = qitem["explain_ru"] if st["ui_lang"] == "ru" and qitem["explain_ru"] else qitem["explain_en"]
                ok = "Correct!" if st["ui_lang"] != "ru" else "Верно!"
                await update.message.reply_text(f"{ok}\n{expl}".strip())
                await log_event(context, "practice_answer", uid, {
                    "ptype": st["type"], "qid": qitem.get("id"), "correct": True
                })
                st["idx"] += 1
            else:
                st["attempts"] += 1
                if st["attempts"] < 2:
                    again = "Not quite. Try again." if st["ui_lang"] != "ru" else "Почти. Попробуй еще раз."
                    await update.message.reply_text(again)
                    await log_event(context, "practice_answer", uid, {
                        "ptype": st["type"], "qid": qitem.get("id"), "correct": False, "retry": True
                    })
                    return
                # reveal
                st["attempts"] = 0
                ans = f"The correct answer is: {qitem['answer']}" if st["ui_lang"] != "ru" else f"Правильный ответ: {qitem['answer']}"
                expl = qitem["explain_ru"] if st["ui_lang"] == "ru" and qitem["explain_ru"] else qitem["explain_en"]
                await update.message.reply_text(f"{ans}\n{expl}".strip())
                await log_event(context, "practice_answer", uid, {
                    "ptype": st["type"], "qid": qitem.get("id"), "correct": False, "revealed": True
                })
                st["idx"] += 1

            if st["idx"] >= len(st["items"]):
                return await practice_summary(update, context)
            else:
                return await send_practice_item(update, context)

    # VOCAB MODE
    if prefs["mode"] == "vocab":
        word = user_message.strip()
        if not word:
            return await update.message.reply_text("Send a word to look up." if lang != "ru" else "Отправь слово.")
        try:
            card = await build_vocab_card(word, prefs, update.message.text)
            await log_event(context, "vocab_lookup", uid, {"word": word})
            return await update.message.reply_text(trim(card))
        except Exception:
            await log_event(context, "vocab_fail", uid, {"word": word})
            return await update.message.reply_text("Failed to build the card. Try another word." if lang != "ru" else "Не удалось создать карточку. Попробуй другое слово.")

    # READING MODE
    if prefs["mode"] == "reading":
        topic = user_message.strip() or "school life"
        level = prefs["cefr"]
        passage_prompt = (
            f"Write a short reading passage (80–120 words) about '{topic}', level {level}, grades 6–9. "
            f"Language: {'Russian' if lang=='ru' else 'English'} (A2–B1). School-safe. No bold."
        )
        passage = await ask_openai(
            [{"role": "system", "content": POLICY_STUDY},
             {"role": "user", "content": passage_prompt}],
            max_tokens=220
        )
        await update.message.reply_text(trim(passage))
        await log_event(context, "reading_passage", uid, {"topic": topic})
        # then build 3 MCQs
        mcq_items = await build_mcq(topic, lang, level)
        mcq_items = mcq_items[:3] if len(mcq_items) > 3 else mcq_items
        context.user_data["practice"] = {
            "type": "mcq", "topic": topic, "items": mcq_items,
            "idx": 0, "attempts": 0, "score": 0, "ui_lang": lang
        }
        return await send_practice_item(update, context)

    # GRAMMAR MODE
    if prefs["mode"] == "grammar":
        text = user_message.strip()
        if re.search(r"\b(practice|exercises|tasks)\b", text, re.I) or (CYRILLIC_RE.search(text) and re.search(r"(упражн|практик)", text, re.I)):
            topic = context.user_data.get("last_grammar_topic", "general grammar")
            level = prefs["cefr"]
            items = await build_text_items("verb", topic, lang, level)
            context.user_data["practice"] = {
                "type": "verb","topic": topic,"items": items,
                "idx": 0,"attempts": 0,"score": 0,"ui_lang": lang
            }
            await log_event(context, "grammar_practice", uid, {"topic": topic, "count": len(items)})
            return await send_practice_item(update, context)

        context.user_data["last_grammar_topic"] = text or "Present Simple"
        g_prompt = (
            f"Explain briefly the grammar point: {context.user_data['last_grammar_topic']} "
            f"for level {prefs['cefr']} in 3–5 bullets with 1–2 examples. "
            f"Language: {'Russian' if lang=='ru' else 'English'}. No markdown bold."
        )
        exp = await ask_openai(
            [{"role": "system", "content": POLICY_STUDY},
             {"role": "user", "content": g_prompt}],
            max_tokens=260
        )
        await log_event(context, "grammar_explain", uid, {"topic": context.user_data['last_grammar_topic']})
        extra = "Type 'practice' to get exercises." if lang != "ru" else "Напиши 'practice', чтобы получить упражнения."
        return await update.message.reply_text(trim(exp) + "\n\n" + extra)

    # TALK MODE
    if prefs["mode"] == "talk":
        talk_state = context.user_data.get("talk") or {"topic": "daily life", "turns": 0}
        reply = await talk_reply(user_message, talk_state["topic"], lang)
        talk_state["turns"] = talk_state.get("turns", 0) + 1
        context.user_data["talk"] = talk_state
        await log_event(context, "talk_turn", uid, {"topic": talk_state["topic"], "turn": talk_state["turns"]})
        if talk_state["turns"] >= prefs.get("dialogue_limit", DEFAULT_DIALOGUE_LIMIT):
            wrap = ("Great chat! Want to practice? Try Vocabulary or Practice from the menu."
                    if lang != "ru" else
                    "Отличная беседа! Хочешь потренироваться? Выбери Слова или Практика в меню.")
            await update.message.reply_text(trim(reply))
            await update.message.reply_text(wrap, reply_markup=root_menu(lang))
            prefs["mode"] = "chat"
            context.user_data.pop("talk", None)
            return
        return await update.message.reply_text(trim(reply))

    # CHAT MODE (default)
    history = context.user_data.get("history", [])
    history.append({"role": "user", "content": user_message})
    history = history[-MAX_HISTORY:]
    context.user_data["history"] = history
    steer = (
        "Be helpful and concise. If the user asks about study tasks, you can suggest modes: "
        "Vocabulary, Reading, Grammar, Practice, Talk."
    )
    messages = [
        {"role": "system", "content": POLICY_CHAT},
        {"role": "user", "content": steer},
        *history
    ]
    text_out = await ask_openai(messages, max_tokens=400)
    await update.message.reply_text(trim(text_out))
    await log_event(context, "chat_message", uid, {"chars": len(user_message)})

# ========== FLASK (KEEP PORT OPEN) ==========
app = Flask(__name__)

@app.get("/")
def health():
    return "✅ Bot is alive", 200

def start_flask():
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)

# ========== MAIN ==========
def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_cmd))
    application.add_handler(CommandHandler("vocab", vocab_cmd))  # optional shortcut
    application.add_handler(CommandHandler("logtest", logtest_cmd))  # test logging

    application.add_handler(CallbackQueryHandler(on_cb))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.add_error_handler(on_error)
    application.post_init = on_startup

    threading.Thread(target=start_flask, daemon=True).start()
    logger.info("Bot is starting (Web Service + Flask)…")
    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == "__main__":
    main()
