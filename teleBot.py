# teleBot.py
# =========================================================
# 0) IMPORTS & GLOBAL SETUP
# =========================================================
import os, re, json, time, hmac, hashlib, logging, threading, asyncio, uuid, difflib
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv
from openai import OpenAI
from flask import Flask

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import BadRequest
from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler,
    ContextTypes, filters
)

# =========================================================
# 1) LOGGING & STARTUP HOOKS
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.exception("Exception while handling update:", exc_info=context.error)
    try:
        uid = getattr(getattr(update, "effective_user", None), "id", "n/a")
        await log_event(context, "error", uid, {"error": str(context.error)})
    except Exception:
        pass

async def on_startup(app: Application):
    await app.bot.delete_webhook(drop_pending_updates=True)
    logger.info("Webhook deleted, switching to long-polling.")

# =========================================================
# 2) ENV & API CLIENTS
# =========================================================
load_dotenv()

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN")
USE_OPENROUTER   = os.getenv("USE_OPENROUTER", "True").lower() == "true"
OR_KEY           = os.getenv("OPENROUTER_API_KEY")
OA_KEY           = os.getenv("OPENAI_API_KEY")
GSHEET_WEBHOOK   = os.getenv("GSHEET_WEBHOOK", "").strip()
LOG_SALT         = os.getenv("LOG_SALT", "").strip()

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN missing")

logger.info("DEBUG => USE_OPENROUTER=%s | OR=%s | OA=%s | GSHEET=%s",
            USE_OPENROUTER, bool(OR_KEY), bool(OA_KEY), bool(GSHEET_WEBHOOK))

httpx_client = httpx.Client(
    timeout=httpx.Timeout(connect=30.0, read=90.0, write=90.0, pool=90.0),
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

# =========================================================
# 3) CONSTANTS, HELPERS, POLICIES
# =========================================================
DEFAULT_LANG = "auto"
MAX_HISTORY  = 10
DEFAULT_DIALOGUE_LIMIT = 20

BANNED_KEYWORDS = [
    r"\bsex\b", r"\bporn\b", r"\berotic\b",
    r"\bviolence\b", r"\bsuicide\b", r"\bself[- ]?harm\b",
    r"\bdrugs?\b", r"\balcohol\b", r"\bgamble\b",
    r"\bextremis(m|t)\b"
]

GRADE_TO_CEFR = {"6": "A2", "7": "A2+", "8": "B1-", "9": "B1"}

POLICY_CHAT = (
    "You are a safe, school-appropriate assistant for grades 6–9. "
    "Focus only on English language learning topics (vocabulary, grammar, reading, speaking, writing). "
    "No other subjects such as math or science. "
    "Be friendly, concise, and helpful. Encourage learning and practice. "
    "No markdown bold or headings. Level A2–B1."
)
POLICY_STUDY = (
    "You are an English study assistant for grades 6–9 (CEFR A2–B1). "
    "Keep all responses school-appropriate, plain text (no bold), short and useful."
)

CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")

def detect_lang(text: str) -> str:
    return "ru" if CYRILLIC_RE.search(text or "") else "en"

def trim(s: str, max_chars: int = 1000) -> str:
    s = re.sub(r"\n{3,}", "\n\n", (s or "").strip())
    return s if len(s) <= max_chars else (s[:max_chars].rstrip() + "…")

def blocked(text: str) -> bool:
    for pat in BANNED_KEYWORDS:
        if re.search(pat, text or "", flags=re.IGNORECASE):
            return True
    return False

# =========================================================
# 4) USER PREFS / SESSION STATE
# =========================================================
user_prefs = {}  # RAM store

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

def remember_last_text(context: ContextTypes.DEFAULT_TYPE, text: str):
    text = (text or "").strip()
    if text and len(text) >= 8:
        context.user_data["last_text"] = text

def add_vocab_to_bank(context: ContextTypes.DEFAULT_TYPE, word: str):
    word = (word or "").strip()
    if not word:
        return
    bank = context.user_data.get("vocab_bank", [])
    if word.lower() not in [w.lower() for w in bank]:
        bank.append(word)
    context.user_data["vocab_bank"] = bank

# =========================================================
# 5) GOOGLE SHEET LOGGING (ẩn danh user)
# =========================================================
async def log_event(context: ContextTypes.DEFAULT_TYPE, event: str, user_id, extra: dict | None = None):
    if not GSHEET_WEBHOOK:
        return
    try:
        prefs = get_prefs(int(user_id)) if isinstance(user_id, int) else {}
        ts = datetime.now(timezone.utc).isoformat()
        anon_id = make_user_hash(user_id, LOG_SALT)

        payload = {
            "timestamp": ts,
            "user_hash": anon_id,
            "event": event,
            "mode": prefs.get("mode"),
            "lang": prefs.get("lang"),
            "grade": prefs.get("grade"),
            "cefr": prefs.get("cefr"),
            "extra": extra or {}
        }

        signature = ""
        if LOG_SALT:
            sig_src = f"{payload['user_hash']}|{payload['event']}|{payload['timestamp']}|{LOG_SALT}"
            signature = hmac.new(LOG_SALT.encode("utf-8"), sig_src.encode("utf-8"), hashlib.sha256).hexdigest()

        headers = {"X-Log-Signature": signature} if signature else {}

        await asyncio.to_thread(
            httpx_client.post,
            GSHEET_WEBHOOK,
            json=payload,
            headers=headers,
            timeout=10.0,
            follow_redirects=True,
        )
    except Exception as e:
        logger.warning("log_event failed: %s", e)

# =========================================================
# 6) SAFE SENDER HELPERS
# =========================================================
async def safe_reply_message(message, text: str, reply_markup=None):
    try:
        return await message.reply_text(text, reply_markup=reply_markup)
    except BadRequest as e:
        logger.error("sendMessage BadRequest: %s", e)
        try:
            return await message.reply_text(text)
        except Exception as e2:
            logger.exception("Fallback sendMessage failed: %s", e2)

async def safe_edit_text(query, text: str, reply_markup=None):
    try:
        if reply_markup is not None:
            return await query.edit_message_text(text, reply_markup=reply_markup)
        return await query.edit_message_text(text)
    except BadRequest as e:
        logger.error("editMessageText BadRequest: %s", e)
        try:
            msg = await query.edit_message_text(text)
            if reply_markup is not None:
                try:
                    await query.edit_message_reply_markup(reply_markup=reply_markup)
                except BadRequest as e2:
                    logger.error("editMessageReplyMarkup BadRequest: %s", e2)
            return msg
        except Exception as e3:
            logger.exception("Fallback edit failed: %s", e3)

# =========================================================
# 7) UI (INLINE MENUS)
# =========================================================
def root_menu(lang: str) -> InlineKeyboardMarkup:
    if lang == "ru":
        kb = [
            [InlineKeyboardButton("🏫 Класс", callback_data="menu:grade"),
             InlineKeyboardButton("🌐 Язык", callback_data="menu:lang")],
            [InlineKeyboardButton("💬 Разговор", callback_data="menu:mode:talk"),
             InlineKeyboardButton("📝 Практика", callback_data="menu:mode:practice")],
            [InlineKeyboardButton("❓ Помощь", callback_data="menu:help")]
        ]
    else:
        kb = [
            [InlineKeyboardButton("🏫 Grade", callback_data="menu:grade"),
             InlineKeyboardButton("🌐 Language", callback_data="menu:lang")],
            [InlineKeyboardButton("💬 Talk", callback_data="menu:mode:talk"),
             InlineKeyboardButton("📝 Practice", callback_data="menu:mode:practice")],
            [InlineKeyboardButton("❓ Help", callback_data="menu:help")]
        ]
    return InlineKeyboardMarkup(kb)

# =========================================================
# 8.5) HELP PROMPTS (for /help or menu:help)
# =========================================================
HELP_TEXT_EN = (
    "Prompt handbook:\n\n"
    "Vocabulary: Define the word 'set up' — IPA, part of speech, definition, short Russian translation, 3 examples.\n"
    "Reading: Write a short A2 text (100 words) about 'friendship'.\n"
    "Grammar: Explain Present Perfect with ✓/✗ examples.\n"
    "Gloss: Translate gloss for this text: <paste text>.\n"
    "Talk: Let's talk about hobbies."
)
HELP_TEXT_RU = (
    "Памятка промптов:\n\n"
    "Слова: дай определение 'set up' — IPA, часть речи, краткое объяснение, перевод на русский, 3 примера.\n"
    "Чтение: короткий текст уровня A2 (100 слов) на тему 'дружба'.\n"
    "Грамматика: объясни Present Perfect с примерами ✓/✗.\n"
    "Глоссы: переведи глоссы для текста.\n"
    "Разговор: начни короткий диалог о хобби."
)
# =========================================================
# 9) START / HELP COMMANDS
# =========================================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    greet = "Choose your language / Выбери язык:"
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("English", callback_data="set_lang:en"),
         InlineKeyboardButton("Русский", callback_data="set_lang:ru")]
    ])
    await safe_reply_message(update.message, greet, reply_markup=kb)
    await log_event(context, "start", update.effective_user.id, {})

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prefs = get_prefs(update.effective_user.id)
    lang = prefs.get("lang", "en")
    txt = HELP_TEXT_RU if lang == "ru" else HELP_TEXT_EN
    await safe_reply_message(update.message, txt, reply_markup=root_menu(lang))
    await log_event(context, "help_open", update.effective_user.id, {"lang": lang})

# =========================================================
# 10) VOCAB / READING / GRAMMAR BUILDERS
# =========================================================
async def build_vocab_card(headword: str, prefs: dict) -> str:
    lang = prefs.get("lang", "en")
    include_ru = (lang == "ru")
    prompt = (
        "You are an English-learning assistant for grades 6–9 (A2–B1). "
        "Create a clear vocabulary card with 3 examples (increasing difficulty). "
        "Do NOT use markdown or bold. "
        "Translate definition into Russian in parentheses if UI is Russian. "
        "Examples remain in English only.\n\n"
        f"HEADWORD: {headword}\nLEVEL: {prefs['cefr']}\n"
        "Output format:\n"
        "Word: <word> /<IPA>/\nPOS: <part of speech>\n"
        "Definition: <EN definition>(<RU short translation>)\n"
        "Synonyms: ...\nAntonyms: ...\n"
        "Examples:\n1) ...\n2) ...\n3) ..."
    )
    msgs = [{"role": "system", "content": POLICY_STUDY},
            {"role": "user", "content": prompt}]
    return await ask_openai(msgs, max_tokens=380)

async def build_reading_passage(topic: str, level: str, ui_lang: str):
    prompt = f"Write a short A2–B1 reading passage (80–120 words) about '{topic}'. School-safe, plain English."
    msgs = [{"role": "system", "content": POLICY_STUDY},
            {"role": "user", "content": prompt}]
    return await ask_openai(msgs, max_tokens=220)

async def build_reading_gloss(text: str, ui_lang: str):
    target = "Russian" if ui_lang != "ru" else "English"
    prompt = (
        "Gloss the text for A2–B1 learner:\n"
        "- Highlight 12–15 useful chunks (phrasal verbs, idioms, collocations).\n"
        "- Use _underscores_ for the chunk and give short "
        f"{target} meaning in parentheses right after.\n"
        "- Do not translate all text.\nTEXT:\n" + text
    )
    msgs = [{"role": "system", "content": POLICY_STUDY},
            {"role": "user", "content": prompt}]
    return await ask_openai(msgs, max_tokens=400)

# =========================================================
# 11) PRACTICE ENGINE (shared)
# =========================================================
async def send_practice_item(update_or_query, context: ContextTypes.DEFAULT_TYPE):
    st = context.user_data.get("practice")
    if not st:
        return
    idx = st["idx"]
    total = len(st["items"])
    q = st["items"][idx]
    title = f"Q{idx+1}/{total}"
    text = f"{title}\n\n{q['question']}"
    kb = mcq_buttons(q["options"])
    if isinstance(update_or_query, Update):
        await safe_reply_message(update_or_query.message, text, reply_markup=kb)
    else:
        await safe_edit_text(update_or_query, text, reply_markup=kb)

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
        lines.append("Ответы и пояснения:")
    else:
        lines.append(f"Summary: {score}/{total}")
        lines.append("Answers and explanations:")
    for it in st["items"]:
        expl = it["explain_ru"] if lang == "ru" and it["explain_ru"] else it["explain_en"]
        lines.append(f"Q{it['id']}: {it['answer']} — {expl}")
    await safe_reply_message(update.message, "\n".join(lines))
    await log_event(context, "practice_done", update.effective_user.id, {
        "type": st["type"], "topic": st.get("topic"), "score": score, "total": total
    })
    scope = st.get("scope", "free")
    await safe_reply_message(update.message, "—", reply_markup=practice_footer_kb(scope, lang))
    context.user_data.pop("practice", None)

# =========================================================
# 12) TALK COACH
# =========================================================
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

# =========================================================
# 13) OPTIONAL COMMANDS
# =========================================================
async def logtest_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ok = {"ping": "pong", "note": "manual test"}
    await log_event(context, "logtest", update.effective_user.id, ok)
    await safe_reply_message(update.message, "Logtest sent (if GSHEET_WEBHOOK is set).")
async def back_to_menu(update_or_query, context, lang="en"):
    msg = "Back to menu." if lang != "ru" else "Возврат в меню."
    await safe_reply_message(update_or_query.message if isinstance(update_or_query, Update) else update_or_query,
                             msg, reply_markup=root_menu(lang))

# =========================================================

# =========================================================
# 13) CALLBACK HANDLER (INLINE BUTTONS)
# =========================================================
async def on_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    data = q.data or ""
    await q.answer()

    uid = update.effective_user.id
    prefs = get_prefs(uid)
    lang = prefs.get("lang", "en")

    # --- menu root ---
    if data == "menu:root":
        await safe_edit_text(q,
            "Back to menu." if lang != "ru" else "Возврат в меню.",
            reply_markup=root_menu(lang))
        return

    # --- language menu ---
    if data == "menu:lang":
        await safe_edit_text(q, "Choose language / Выбери язык:",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("English", callback_data="set_lang:en"),
                 InlineKeyboardButton("Русский", callback_data="set_lang:ru")]
            ]))
        return

    if data.startswith("set_lang:"):
        lang_new = data.split(":")[1]
        prefs["lang"] = lang_new
        await log_event(context, "lang_set", uid, {"lang": lang_new})
        msg = "Language set to English." if lang_new == "en" else "Язык изменён на русский."
        await safe_edit_text(q, msg + "\n\nChoose an option below 👇",
                             reply_markup=root_menu(lang_new))
        return

    # --- help ---
    if data == "menu:help":
        txt = HELP_TEXT_RU if lang == "ru" else HELP_TEXT_EN
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("⬅️ Back", callback_data="menu:root")]
        ])
        await safe_edit_text(q, txt, reply_markup=kb)
        await log_event(context, "help_open", uid, {})
        return

    # --- grade selection ---
    if data == "menu:grade":
        txt = "Choose your grade:" if lang != "ru" else "Выбери свой класс:"
        await safe_edit_text(q, txt, reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("6", callback_data="set_grade:6"),
             InlineKeyboardButton("7", callback_data="set_grade:7"),
             InlineKeyboardButton("8", callback_data="set_grade:8"),
             InlineKeyboardButton("9", callback_data="set_grade:9")],
            [InlineKeyboardButton("⬅️ Back", callback_data="menu:root")]
        ]))
        return

    if data.startswith("set_grade:"):
        g = data.split(":")[1]
        if g in GRADE_TO_CEFR:
            prefs["grade"] = g
            prefs["cefr"] = GRADE_TO_CEFR[g]
            msg = (f"Grade set to {g} (level {prefs['cefr']})."
                   if lang != "ru" else f"Класс {g} (уровень {prefs['cefr']}).")
            await safe_edit_text(q, msg, reply_markup=root_menu(lang))
            await log_event(context, "grade_set", uid, {"grade": g})
        return

    # --- mode: practice ---
    if data == "menu:mode:practice":
        txt = "Send a topic or word to practice!" if lang != "ru" else "Отправь тему или слово для практики!"
        await safe_edit_text(q, txt, reply_markup=root_menu(lang))
        prefs["mode"] = "practice"
        await log_event(context, "mode_set", uid, {"mode": "practice"})
        return

    # --- mode: talk ---
    if data == "menu:mode:talk":
        prefs["mode"] = "talk"
        context.user_data["talk"] = {"topic": "daily life", "turns": 0}
        txt = "Let's talk! What do you like to do after school?" if lang != "ru" else "Поговорим! Чем ты любишь заниматься после школы?"
        await safe_edit_text(q, txt, reply_markup=root_menu(lang))
        await log_event(context, "talk_start", uid, {})
        return

    # --- answer MCQ ---
    if data.startswith("ans:"):
        st = context.user_data.get("practice")
        if not st:
            return await safe_edit_text(q, "No active quiz.")
        ch = data.split(":")[1]
        qitem = st["items"][st["idx"]]
        correct = qitem["answer"]

        if ch == correct:
            st["score"] += 1
            msg = "✅ Correct!" if lang != "ru" else "✅ Верно!"
        else:
            msg = f"❌ Correct answer: {correct}" if lang != "ru" else f"❌ Правильный ответ: {correct}"
        await safe_edit_text(q, msg)

        st["idx"] += 1
        if st["idx"] >= len(st["items"]):
            dummy = Update(update.update_id, message=q.message)
            await practice_summary(dummy, context)
        else:
            await send_practice_item(q, context)
        return

    # --- nudge mini quiz ---
    if data == "nudge:start":
        items = await build_mcq("English basics", lang, prefs["cefr"], flavor="generic")
        context.user_data["practice"] = {
            "type": "mcq", "topic": "nudge", "items": items[:2],
            "idx": 0, "score": 0, "ui_lang": lang, "scope": "free"
        }
        await log_event(context, "nudge_practice_start", uid, {"count": 2})
        return await send_practice_item(q, context)

    if data == "nudge:skip":
        await safe_edit_text(q, "No problem! Let's continue 😊" if lang != "ru" else "Хорошо! Продолжим 😊")
        return
# =========================================================


# =========================================================
# 14) NUDGE & REWARD HELPERS
# =========================================================
def increment_nudge_counter(context):
    c = context.user_data.get("nudge_count", 0) + 1
    context.user_data["nudge_count"] = c
    return c

def reset_nudge_counter(context):
    context.user_data["nudge_count"] = 0

async def maybe_send_nudge(update, context, lang="en"):
    c = increment_nudge_counter(context)
    if c >= 3:
        reset_nudge_counter(context)
        msg = "Do you want a quick 2-question mini-quiz? (≤1 min)" if lang != "ru" else "Хочешь мини-викторину из 2 вопросов?"
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("▶️ Start", callback_data="nudge:start"),
             InlineKeyboardButton("⏭ Skip", callback_data="nudge:skip")]
        ])
        await safe_reply_message(update.message, msg, reply_markup=kb)
        await log_event(context, "nudge_trigger", update.effective_user.id, {})

async def reward_message(update, context, score, total, lang="en"):
    if score == total:
        msg = "🌟 Perfect! All correct!" if lang != "ru" else "🌟 Отлично! Все правильно!"
    elif score / total >= 0.6:
        msg = "⭐ Great work!" if lang != "ru" else "⭐ Отличная работа!"
    else:
        msg = "👏 Nice effort!" if lang != "ru" else "👏 Хорошая попытка!"
    await safe_reply_message(update.message, msg,
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Menu", callback_data="menu:root")]]))
    await log_event(context, "reward_given", update.effective_user.id,
        {"score": score, "total": total})
# =========================================================


# =========================================================
# 15) FREE TEXT HANDLER (chat-first + intent detector)
# =========================================================
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    if not text:
        return

    if blocked(text):
        return await safe_reply_message(update.message,
            "⛔ Please keep messages about English learning only.")

    uid = update.effective_user.id
    prefs = get_prefs(uid)
    lang = prefs.get("lang", "en")
    if lang == "auto":
        lang = detect_lang(text)

    t_low = text.lower()

    # --- INTENT DETECTION ---
    if len(t_low.split()) <= 2:
        intent = "vocab"
        headword = t_low
    elif re.search(r"\bdefine\b|\bmeaning of\b", t_low):
        intent = "vocab"; headword = re.sub(r".*\b(?:define|meaning of)\b", "", t_low).strip()
    elif re.search(r"\bgrammar\b|\bexplain\b|\brule\b", t_low):
        intent = "grammar"
    elif re.search(r"\bwrite a short text\b|\bread(?:ing)?\b", t_low):
        intent = "reading"
    elif re.search(r"\btranslate\b|\bgloss\b", t_low):
        intent = "gloss"
    elif re.search(r"\bquiz\b|\bpractice\b|\bexercise\b", t_low):
        intent = "practice"
    elif re.search(r"\btalk\b|\bconversation\b|\bchat\b", t_low):
        intent = "talk"
    else:
        intent = "chat"

    # --- ROUTE ACTION ---
    if intent == "vocab":
        reset_nudge_counter(context)
        card = await build_vocab_card(headword, prefs)
        context.user_data["last_word"] = headword
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("✏️ Practice this word", callback_data="vocab:practice"),
             InlineKeyboardButton("➕ More examples", callback_data="vocab:more")],
            [InlineKeyboardButton("🏠 Menu", callback_data="menu:root")]
        ])
        await safe_reply_message(update.message, trim(card), reply_markup=kb)
        await log_event(context, "vocab_card", uid, {"word": headword})
        return await maybe_send_nudge(update, context, lang)

    if intent == "grammar":
        reset_nudge_counter(context)
        topic = text.strip()
        g_prompt = (
            f"Explain '{topic}' for level {prefs['cefr']} (A2–B1). "
            "5–7 bullets, ✓/✗ examples, signal words. "
            "If UI is Russian, add short hints in parentheses."
        )
        exp = await ask_openai(
            [{"role": "system", "content": POLICY_STUDY},
             {"role": "user", "content": g_prompt}],
            max_tokens=360)
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("✏️ Practice this rule", callback_data="grammar:practice"),
             InlineKeyboardButton("📚 Explain more", callback_data="footer:explain_more")],
            [InlineKeyboardButton("🏠 Menu", callback_data="menu:root")]
        ])
        await safe_reply_message(update.message, trim(exp), reply_markup=kb)
        await log_event(context, "grammar_explain", uid, {"topic": topic})
        return await maybe_send_nudge(update, context, lang)

    if intent == "reading":
        reset_nudge_counter(context)
        passage = await build_reading_passage(text, prefs["cefr"], lang)
        context.user_data["last_passage"] = passage
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("abc Translate (gloss)", callback_data="reading:gloss"),
             InlineKeyboardButton("📝 Practice from this text", callback_data="reading:practice")],
            [InlineKeyboardButton("🔁 Another text", callback_data="footer:new_text"),
             InlineKeyboardButton("🏠 Menu", callback_data="menu:root")]
        ])
        await safe_reply_message(update.message, trim(passage), reply_markup=kb)
        await log_event(context, "reading_passage", uid, {})
        return await maybe_send_nudge(update, context, lang)

    if intent == "gloss":
        passage = context.user_data.get("last_passage", text)
        glossed = await build_reading_gloss(passage, lang)
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("📝 Practice from this text", callback_data="reading:practice"),
             InlineKeyboardButton("🏠 Menu", callback_data="menu:root")]
        ])
        await safe_reply_message(update.message, trim(glossed), reply_markup=kb)
        await log_event(context, "reading_gloss", uid, {})
        return await maybe_send_nudge(update, context, lang)

    if intent == "talk":
        reset_nudge_counter(context)
        reply = await talk_reply(text, "school life", lang)
        await safe_reply_message(update.message, trim(reply),
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Menu", callback_data="menu:root")]]))
        await log_event(context, "talk_message", uid, {"chars": len(text)})
        return

    if intent == "practice":
        reset_nudge_counter(context)
        items = await build_mcq(text, lang, prefs["cefr"], flavor="generic")
        context.user_data["practice"] = {
            "type": "mcq", "topic": text, "items": items,
            "idx": 0, "score": 0, "ui_lang": lang, "scope": "free"
        }
        await log_event(context, "practice_start", uid, {"count": len(items)})
        return await send_practice_item(update, context)

    # default chat
    msg = [{"role": "system", "content": POLICY_CHAT},
           {"role": "user", "content": text}]
    reply = await ask_openai(msg, max_tokens=300)
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Menu", callback_data="menu:root")]])
    await safe_reply_message(update.message, trim(reply), reply_markup=kb)
    await log_event(context, "chat_message", uid, {"chars": len(text)})
    await maybe_send_nudge(update, context, lang)
# =========================================================

# =========================================================
# 16) FLASK HEALTHCHECK
# =========================================================
app = Flask(__name__)

@app.get("/")
def health():
    return "✅ Bot is alive", 200

def start_flask():
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)

# =========================================================
# 17) MAIN ENTRYPOINT
# =========================================================
def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_cmd))
    application.add_handler(CommandHandler("logtest", logtest_cmd))

    application.add_handler(CallbackQueryHandler(on_cb))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.add_error_handler(on_error)
    application.post_init = on_startup

    threading.Thread(target=start_flask, daemon=True).start()
    logger.info("Bot is starting (Web Service + Flask)…")
    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == "__main__":
    main()
