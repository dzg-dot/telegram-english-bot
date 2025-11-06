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
DEFAULT_LANG = "auto"          # auto|en|ru
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
    "Do not answer questions about other school subjects such as math, science, or history — politely redirect to English learning. "
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
def lang_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("English",  callback_data="set_lang:en"),
         InlineKeyboardButton("Русский",  callback_data="set_lang:ru")],
    ])

def root_menu(lang: str) -> InlineKeyboardMarkup:
    if lang == "ru":
        kb = [
            [InlineKeyboardButton("📚 Слова",        callback_data="menu:mode:vocab"),
             InlineKeyboardButton("📖 Чтение",       callback_data="menu:mode:reading")],
            [InlineKeyboardButton("⚙️ Грамматика",   callback_data="menu:mode:grammar"),
             InlineKeyboardButton("💬 Разговор",     callback_data="menu:mode:talk")],
            [InlineKeyboardButton("📝 Практика",     callback_data="menu:mode:practice")],
            [InlineKeyboardButton("🏫 Класс",        callback_data="menu:grade"),
             InlineKeyboardButton("🌐 Язык",         callback_data="menu:lang")],
            [InlineKeyboardButton("❓ Справка",       callback_data="menu:help")]
        ]
    else:
        kb = [
            [InlineKeyboardButton("📚 Vocabulary",   callback_data="menu:mode:vocab"),
             InlineKeyboardButton("📖 Reading",      callback_data="menu:mode:reading")],
            [InlineKeyboardButton("⚙️ Grammar",      callback_data="menu:mode:grammar"),
             InlineKeyboardButton("💬 Talk",         callback_data="menu:mode:talk")],
            [InlineKeyboardButton("📝 Practice",     callback_data="menu:mode:practice")],
            [InlineKeyboardButton("🏫 Grade",        callback_data="menu:grade"),
             InlineKeyboardButton("🌐 Language",     callback_data="menu:lang")],
            [InlineKeyboardButton("❓ Help",         callback_data="menu:help")]
        ]
    return InlineKeyboardMarkup(kb)
def reading_entry_menu(lang="en") -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📌 Topic",   callback_data="reading:input:topic"),
         InlineKeyboardButton("📝 My text", callback_data="reading:input:mytext")],
        [InlineKeyboardButton("⬅️ Back",    callback_data="menu:root")]
    ])

def reading_footer_kb(lang="en") -> InlineKeyboardMarkup:
    t = {
        "gloss": "abc Translate (gloss)",
        "prac":  "📝 Practice from this text",
        "again": "🔁 Another text",
        "back":  "⬅️ Back"
    } if lang != "ru" else {
        "gloss": "abc Перевод (глоссы)",
        "prac":  "📝 Упражнения по тексту",
        "again": "🔁 Ещё текст",
        "back":  "⬅️ Назад"
    }
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(t["gloss"], callback_data="reading:gloss"),
         InlineKeyboardButton(t["prac"],  callback_data="reading:practice")],
        [InlineKeyboardButton(t["again"], callback_data="footer:new_text")],
        [InlineKeyboardButton(t["back"],  callback_data="menu:root")]
    ])

def vocab_footer_kb(lang="en") -> InlineKeyboardMarkup:
    t = {
        "practice": "✏️ Practice this word",
        "more":     "➕ More examples",
        "back":     "⬅️ Back"
    } if lang != "ru" else {
        "practice": "✏️ Тренировка по слову",
        "more":     "➕ Ещё примеры",
        "back":     "⬅️ Назад"
    }
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(t["practice"], callback_data="vocab:practice"),
         InlineKeyboardButton(t["more"],     callback_data="vocab:more")],
        [InlineKeyboardButton(t["back"],     callback_data="menu:root")]
    ])

def grammar_footer_kb(lang="en") -> InlineKeyboardMarkup:
    t = {
        "prac": "✏️ Practice this rule",
        "more": "📚 Explain more",
        "back": "⬅️ Back"
    } if lang != "ru" else {
        "prac": "✏️ Тренировка по правилу",
        "more": "📚 Пояснить подробнее",
        "back": "⬅️ Назад"
    }
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(t["prac"], callback_data="grammar:practice"),
         InlineKeyboardButton(t["more"], callback_data="footer:explain_more")],
        [InlineKeyboardButton(t["back"], callback_data="menu:root")]
    ])

def practice_footer_kb(scope: str, lang="en") -> InlineKeyboardMarkup:
    t = {
        "again":    "🔁 Again",
        "new_word": "🔎 New word",
        "new_text": "🔁 Another text",
        "new_rule": "🔎 New rule",
        "menu":     "🏠 Menu"
    } if lang != "ru" else {
        "again":    "🔁 Ещё задание",
        "new_word": "🔎 Другое слово",
        "new_text": "🔁 Другой текст",
        "new_rule": "🔎 Другое правило",
        "menu":     "🏠 Меню"
    }
    rows = [[InlineKeyboardButton(t["again"], callback_data="footer:more_practice")]]
    if scope == "vocab":
        rows.append([InlineKeyboardButton(t["new_word"], callback_data="footer:new_word")])
    elif scope == "reading":
        rows.append([InlineKeyboardButton(t["new_text"], callback_data="footer:new_text")])
    elif scope == "grammar":
        rows.append([InlineKeyboardButton(t["new_rule"], callback_data="footer:new_rule")])
    rows.append([InlineKeyboardButton(t["menu"], callback_data="menu:root")])
    return InlineKeyboardMarkup(rows)

def mcq_buttons(options):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"A) {options[0]}", callback_data="ans:A"),
         InlineKeyboardButton(f"B) {options[1]}", callback_data="ans:B")],
        [InlineKeyboardButton(f"C) {options[2]}", callback_data="ans:C"),
         InlineKeyboardButton(f"D) {options[3]}", callback_data="ans:D")]
    ])

# =========================================================
# 8.5) HELP PROMPTS (for /help or menu:help)
# =========================================================
HELP_TEXT_EN = (
    "Prompt handbook:\n\n"
    "Vocabulary:\n"
    "- Define the word \"set up\" — include IPA, part of speech, short definition, Russian translation, and 2–3 examples.\n"
    "- Give me the meaning of \"take off\" as a phrasal verb, with synonyms, antonyms, and 3 examples.\n\n"
    "Reading:\n"
    "- Write a short A2-level reading passage (about 100 words) about \"friendship\".\n"
    "- Translate gloss for this text: <paste text>.\n\n"
    "Grammar:\n"
    "- Explain Present Perfect for A2–B1 with form, uses, and ✓/✗ examples.\n"
    "- Show me Conditional Type 1 with 3 correct and 3 wrong examples.\n\n"
    "Talk:\n"
    "- Let's talk about school life.\n"
    "- Start a short English conversation about hobbies at A2 level."
)
HELP_TEXT_RU = (
    "Памятка промптов:\n\n"
    "Слова:\n"
    "- Дай определение \"set up\" — IPA, часть речи, краткое объяснение, перевод на русский, 2–3 примера.\n"
    "- Значение фразового глагола \"take off\" + синонимы/антонимы и 3 примера.\n\n"
    "Чтение:\n"
    "- Короткий текст уровня A2 (~100 слов) на тему \"дружба\".\n"
    "- Глоссы для текста: <вставь текст>.\n\n"
    "Грамматика:\n"
    "- Объясни Present Perfect для A2–B1: форма, употребления, примеры ✓/✗.\n"
    "- Условные предложения типа 1: 3 правильных и 3 неправильных примера.\n\n"
    "Разговор:\n"
    "- Поговорим о школьной жизни.\n"
    "- Начни короткий диалог об увлечениях (уровень A2)."
)

# =========================================================
# 9) START / HELP COMMANDS
# =========================================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    greet = "Choose your language / Выбери язык:"
    await safe_reply_message(update.message, greet, reply_markup=lang_menu())
    prefs = get_prefs(update.effective_user.id)
    prefs["mode"] = "chat"
    context.user_data.pop("reading_input", None)
    await log_event(context, "start", update.effective_user.id, {"text": (update.message.text or "")[:200]})

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prefs = get_prefs(update.effective_user.id)
    lang = prefs.get("lang", "en")
    txt = HELP_TEXT_RU if lang == "ru" else HELP_TEXT_EN
    await safe_reply_message(update.message, txt, reply_markup=root_menu(lang))
    await log_event(context, "help", update.effective_user.id, {"lang": lang})

# =========================================================
# 10) VOCAB / READING / GRAMMAR BUILDERS
# =========================================================
async def build_vocab_card(headword: str, prefs: dict, user_text: str) -> str:
    lang_for_examples = prefs.get("lang", "auto")
    if lang_for_examples == "auto":
        lang_for_examples = detect_lang(user_text or "")
    include_ru_examples = (lang_for_examples == "ru")

    prompt = (
        "You are an English-learning assistant for grades 6–9 (CEFR A2–B1). "
        "Make a compact vocabulary card. Do not use markdown bold or any asterisks. "
        "Definition must be in English with a short Russian translation in parentheses. "
        "Add short Synonyms and Antonyms lists if natural.\n\n"
        f"HEADWORD: {headword}\nTARGET LEVEL: {prefs['cefr']}\n\n"
        "Format exactly (plain text):\n"
        "Word: <headword> /<IPA>/\n"
        "POS: <part of speech>\n"
        "Definition: <short English definition> (<short Russian translation>)\n"
        "Synonyms: x, y, z (if any)\n"
        "Antonyms: a, b (if any)\n"
        "Examples:\n"
        f"1) <short EN example>{' (RU translation)' if include_ru_examples else ''}\n"
        f"2) <short EN example>{' (RU translation)' if include_ru_examples else ''}\n"
        f"3) <short EN example>{' (optional RU translation)' if include_ru_examples else ' (optional)'}\n"
        "Keep under 140 words."
    )
    msgs = [{"role": "system", "content": POLICY_STUDY},
            {"role": "user", "content": prompt}]
    return await ask_openai(msgs, max_tokens=360)
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

# =========================================================
# 14) CALLBACK HANDLER (INLINE BUTTONS)
# =========================================================
async def on_cb(update:Update,context:ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    data = q.data or ""
    await q.answer()
    uid = update.effective_user.id
    prefs = get_prefs(uid)
    lang = prefs.get("lang","en")
    unlocked = context.user_data.get("unlocked_modes",{})

    # basic menus
    if data=="menu:root":
        await safe_edit_text(q,"Back to menu.",reply_markup=root_menu(lang,unlocked)); return
    if data.startswith("menu:mode:"):
        prefs["mode"]=data.split(":")[-1]
        await log_event(context,"mode_set",uid,{"mode":prefs["mode"]})
        txt=f"{prefs['mode'].capitalize()} mode unlocked. Send your request!"
        await safe_edit_text(q,txt,reply_markup=root_menu(lang,unlocked)); return

    # answer MCQ
    if data.startswith("ans:"):
        st=context.user_data.get("practice")
        if not st: return await safe_edit_text(q,"No active quiz.")
        ch=data.split(":")[1]
        qitem=st["items"][st["idx"]]
        correct=qitem["answer"]
        if ch==correct:
            st["score"]+=1; msg="✅ Correct!"
        else: msg=f"❌ Correct answer: {correct}"
        await safe_edit_text(q,msg)
        st["idx"]+=1
        if st["idx"]>=len(st["items"]):
            dummy=Update(update.update_id,message=q.message)
            await practice_summary(dummy,context)
        else: await send_practice_item(q,context)
        return

# =========================================================
# 15) FREE TEXT HANDLER (chat-first)
# =========================================================
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text or ""

    # remember last_text for translate/gloss
    if user_message and len(user_message) >= 8:
        remember_last_text(context, user_message)

    # --- safety filters ---
    if blocked(user_message):
        return await safe_reply_message(
            update.message,
            "⛔ That's outside our classroom scope. Please try vocabulary, reading, grammar, talk, or practice."
        )

    # --- English-only restriction (soft but strict) ---
    non_english_topics = [
        "math", "physics", "chemistry", "biology", "history",
        "geography", "geometry", "algebra", "calculus",
        "literature", "philosophy", "economics", "politics",
        "law", "astronomy", "programming", "python", "computer"
    ]
    if re.search(r"\b(" + "|".join(non_english_topics) + r")\b", user_message, re.I):
        return await safe_reply_message(
            update.message,
            "Let's stay focused on English learning 😊 Try vocabulary, grammar, reading, or conversation instead."
        )

    uid = update.effective_user.id
    prefs = get_prefs(uid)
    lang = prefs.get("lang", "en")
    if lang == "auto":
        lang = detect_lang(user_message)

    # (phần xử lý các mode vocab / reading / grammar / talk / practice giữ nguyên)
    # ...
    # cuối cùng chat mode default:
    history = context.user_data.get("history", [])
    history.append({"role": "user", "content": user_message})
    history = history[-MAX_HISTORY:]
    context.user_data["history"] = history
    steer = (
        "Be helpful and concise. If the user asks about study tasks, suggest: Vocabulary, Reading, Grammar, Talk, or Practice."
    )
    messages = [
        {"role": "system", "content": POLICY_CHAT},
        {"role": "user", "content": steer},
        *history
    ]
    text_out = await ask_openai(messages, max_tokens=400)
    await safe_reply_message(update.message, trim(text_out), reply_markup=root_menu(lang))
    await log_event(context, "chat_message", uid, {"chars": len(user_message)})

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
