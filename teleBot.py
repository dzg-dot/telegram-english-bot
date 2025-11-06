# =========================================================
# 0) IMPORTS & GLOBAL SETUP
# =========================================================
import os, re, json, time, hmac, hashlib, logging, asyncio, uuid, difflib
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
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OR_KEY = os.getenv("OPENROUTER_API_KEY")
GSHEET_WEBHOOK = os.getenv("GSHEET_WEBHOOK", "").strip()
LOG_SALT = os.getenv("LOG_SALT", "").strip()

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN missing")

httpx_client = httpx.Client(timeout=httpx.Timeout(connect=30.0, read=90.0))
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OR_KEY,
    http_client=httpx_client,
    default_headers={
        "HTTP-Referer": "https://t.me/EnglishStudyBuddy",
        "X-Title": "English Study Bot",
    },
)
MODEL_NAME = "openai/gpt-4o-mini"


# =========================================================
# 3) CONSTANTS, HELPERS, POLICIES
# =========================================================
DEFAULT_LANG = "auto"
MAX_HISTORY = 10
BANNED_KEYWORDS = [
    r"\bsex\b", r"\bporn\b", r"\berotic\b",
    r"\bviolence\b", r"\bsuicide\b", r"\bself[- ]?harm\b",
    r"\bdrugs?\b", r"\balcohol\b", r"\bgamble\b", r"\bextremis(m|t)\b"
]
GRADE_TO_CEFR = {"6": "A2", "7": "A2+", "8": "B1-", "9": "B1"}

POLICY_CHAT = (
    "You are a friendly English-learning tutor for grades 6–9 (CEFR A2–B1). "
    "Answer only about English language learning: vocabulary, grammar, reading, writing, and speaking. "
    "If the user asks about other school subjects (math, physics, etc.), refuse and steer back to English learning. "
    "No markdown bold, no headings. Be concise and encouraging."
)
POLICY_STUDY = (
    "You are an English study assistant for middle school (A2–B1). "
    "Use plain text, no markdown, keep concise and safe."
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
# 4) USER PREFS / STATE
# =========================================================
user_prefs = {}
def get_prefs(user_id: int):
    if user_id not in user_prefs:
        user_prefs[user_id] = {
            "lang": DEFAULT_LANG,
            "grade": "7",
            "cefr": "A2+"
        }
    return user_prefs[user_id]


# =========================================================
# 5) LOGGING TO GOOGLE SHEET
# =========================================================
async def log_event(context, event: str, user_id, extra=None):
    if not GSHEET_WEBHOOK:
        return
    try:
        ts = datetime.now(timezone.utc).isoformat()
        anon = hashlib.sha256(f"{user_id}|{LOG_SALT}".encode()).hexdigest()[:12]
        payload = {
            "timestamp": ts,
            "user_hash": anon,
            "event": event,
            "extra": extra or {}
        }
        await asyncio.to_thread(httpx_client.post, GSHEET_WEBHOOK, json=payload, timeout=8.0)
    except Exception as e:
        logger.warning("log_event failed: %s", e)


# =========================================================
# 6) SAFE SENDERS
# =========================================================
async def safe_reply_message(message, text: str, reply_markup=None):
    try:
        return await message.reply_text(text, reply_markup=reply_markup)
    except BadRequest:
        try:
            return await message.reply_text(text)
        except Exception as e:
            logger.warning("safe_reply failed: %s", e)

async def safe_edit_text(query, text: str, reply_markup=None):
    try:
        return await query.edit_message_text(text, reply_markup=reply_markup)
    except BadRequest:
        try:
            await query.edit_message_text(text)
        except Exception as e:
            logger.warning("safe_edit_text failed: %s", e)


# =========================================================
# 7) UI MENUS & HELP
# =========================================================
def root_menu(lang="en") -> InlineKeyboardMarkup:
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

HELP_TEXT_EN = (
    "Prompt handbook:\n\n"
    "- Define *set up* (IPA, part of speech, definition, RU translation, 3 examples)\n"
    "- Explain *Present Perfect* with ✓/✗ examples\n"
    "- Write a short A2 text about *friendship*\n"
    "- Translate gloss for this text: <your text>\n"
    "- Let's talk about *school life*"
)
HELP_TEXT_RU = (
    "Памятка промптов:\n\n"
    "- Дай определение *set up* — IPA, часть речи, краткое объяснение, перевод, 3 примера\n"
    "- Объясни *Present Perfect* с примерами ✓/✗\n"
    "- Короткий текст уровня A2 на тему *дружба*\n"
    "- Глоссы для текста: <вставь текст>\n"
    "- Поговорим о *школьной жизни*"
)


# =========================================================
# 8) START / HELP COMMANDS
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
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("⬅️ Back", callback_data="menu:root")]
    ])
    await safe_reply_message(update.message, txt, reply_markup=kb)
    await log_event(context, "help_open", update.effective_user.id, {})


# =========================================================
# 9) BUILDERS — VOCAB / READING / GLOSS / GRAMMAR
# =========================================================
async def ask_openai(messages, max_tokens=400, temperature=0.3):
    for _ in range(2):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME, messages=messages,
                max_tokens=max_tokens, temperature=temperature
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.warning("ask_openai fail: %s", e)
            time.sleep(0.8)
    return "[error contacting model]"

async def build_vocab_card(headword: str, prefs: dict) -> str:
    lang = prefs.get("lang", "en")
    include_ru = (lang == "ru")
    prompt = (
        "Create a vocabulary card for English learners (A2–B1). "
        "No markdown or bold. Definition in English with short Russian translation in parentheses. "
        "Give 3 short examples (increasing difficulty, English only). "
        f"Word: {headword}"
    )
    msgs = [{"role": "system", "content": POLICY_STUDY},
            {"role": "user", "content": prompt}]
    return await ask_openai(msgs, max_tokens=380)

async def build_reading_passage(topic: str, level: str):
    prompt = f"Write a short A2–B1 reading passage (80–120 words) about '{topic}'. School-safe, plain English."
    msgs = [{"role": "system", "content": POLICY_STUDY},
            {"role": "user", "content": prompt}]
    return await ask_openai(msgs, max_tokens=250)

async def build_reading_gloss(text: str, ui_lang: str):
    target = "Russian" if ui_lang != "ru" else "English"
    prompt = (
        "Gloss the text for A2–B1 learners:\n"
        "- Mark 12–15 useful chunks (phrasal verbs, idioms, collocations)\n"
        f"- Add short {target} translation in parentheses after each chunk\n"
        "- Do not translate everything.\nTEXT:\n" + text
    )
    msgs = [{"role": "system", "content": POLICY_STUDY},
            {"role": "user", "content": prompt}]
    return await ask_openai(msgs, max_tokens=400)

async def build_grammar_explain(topic: str, prefs: dict):
    lang = prefs.get("lang", "en")
    prompt = (
        f"Explain grammar topic '{topic}' for CEFR {prefs['cefr']}. "
        "5–7 bullet points (form, use, mistakes), include ✓/✗ examples and signal words. "
        "If UI is Russian, add short Russian hint in parentheses. No markdown."
    )
    msgs = [{"role": "system", "content": POLICY_STUDY},
            {"role": "user", "content": prompt}]
    return await ask_openai(msgs, max_tokens=400)
# =========================================================
# 11) PRACTICE ENGINE (MCQ + SUMMARY)
# =========================================================
def normalize_answer(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\s'-]", "", s)
    return re.sub(r"\s+", " ", s)

def fuzzy_equal(a: str, b: str, threshold: float = 0.85) -> bool:
    return difflib.SequenceMatcher(a=normalize_answer(a), b=normalize_answer(b)).ratio() >= threshold

async def send_practice_item(update_or_query, context: ContextTypes.DEFAULT_TYPE):
    st = context.user_data.get("practice")
    if not st: return
    idx = st["idx"]
    q = st["items"][idx]
    txt = f"Q{idx+1}/{len(st['items'])}\n\n{q['question']}"
    opts = q["options"]
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton(f"A) {opts[0]}", callback_data="ans:A"),
         InlineKeyboardButton(f"B) {opts[1]}", callback_data="ans:B")],
        [InlineKeyboardButton(f"C) {opts[2]}", callback_data="ans:C"),
         InlineKeyboardButton(f"D) {opts[3]}", callback_data="ans:D")]
    ])
    if isinstance(update_or_query, Update):
        await safe_reply_message(update_or_query.message, txt, reply_markup=kb)
    else:
        await safe_edit_text(update_or_query, txt, reply_markup=kb)

async def practice_summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    st = context.user_data.get("practice")
    if not st: return
    lang = st.get("ui_lang", "en")
    total = len(st["items"])
    score = st.get("score", 0)
    lines = ["Summary:" if lang != "ru" else "Итоги:"]
    for it in st["items"]:
        lines.append(f"Q{it['id']}: {it['answer']} – {it.get('explain_en','')}")
    await safe_reply_message(update.message, "\n".join(lines))
    await reward_message(update, context, score, total, lang)
    context.user_data.pop("practice", None)


# =========================================================
# 12) CALLBACK HANDLER (INLINE BUTTONS + FLUTTON)
# =========================================================
async def on_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    data = q.data or ""
    await q.answer()

    uid = update.effective_user.id
    prefs = get_prefs(uid)
    lang = prefs.get("lang", "en")

    # --- BASIC MENUS ---
    if data == "menu:root":
        await safe_edit_text(q, "Back to menu." if lang!="ru" else "Возврат в меню.", reply_markup=root_menu(lang))
        return

    if data == "menu:help":
        txt = HELP_TEXT_RU if lang == "ru" else HELP_TEXT_EN
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data="menu:root")]])
        await safe_edit_text(q, txt, reply_markup=kb)
        return

    if data == "menu:lang":
        await safe_edit_text(q, "Choose language / Выбери язык:", reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("English", callback_data="set_lang:en"),
             InlineKeyboardButton("Русский", callback_data="set_lang:ru")]
        ]))
        return

    if data.startswith("set_lang:"):
        lang_new = data.split(":")[1]
        prefs["lang"] = lang_new
        msg = "Language set! Choose an option below 👇" if lang_new=="en" else "Язык выбран! Выбери действие ниже 👇"
        await safe_edit_text(q, msg, reply_markup=root_menu(lang_new))
        await log_event(context, "lang_set", uid, {"lang": lang_new})
        return

    if data == "menu:grade":
        txt = "Choose your grade:" if lang!="ru" else "Выбери класс:"
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("6", callback_data="set_grade:6"),
             InlineKeyboardButton("7", callback_data="set_grade:7"),
             InlineKeyboardButton("8", callback_data="set_grade:8"),
             InlineKeyboardButton("9", callback_data="set_grade:9")],
            [InlineKeyboardButton("⬅️ Back", callback_data="menu:root")]
        ])
        await safe_edit_text(q, txt, reply_markup=kb)
        return

    if data.startswith("set_grade:"):
        g = data.split(":")[1]
        if g in GRADE_TO_CEFR:
            prefs["grade"] = g
            prefs["cefr"] = GRADE_TO_CEFR[g]
            txt = (f"Grade set to {g} (level {prefs['cefr']})."
                   if lang!="ru" else f"Класс {g} (уровень {prefs['cefr']}).")
            await safe_edit_text(q, txt, reply_markup=root_menu(lang))
            await log_event(context, "grade_set", uid, {"grade": g})
        return

    # --- ANSWERS ---
    if data.startswith("ans:"):
        st = context.user_data.get("practice")
        if not st: return await safe_edit_text(q, "No active quiz.")
        ch = data.split(":")[1]
        qitem = st["items"][st["idx"]]
        correct = qitem["answer"]
        if ch == correct:
            st["score"] += 1
            msg = "✅ Correct!" if lang!="ru" else "✅ Верно!"
        else:
            msg = f"❌ Correct answer: {correct}" if lang!="ru" else f"❌ Правильный ответ: {correct}"
        await safe_edit_text(q, msg)
        st["idx"] += 1
        if st["idx"] >= len(st["items"]):
            dummy = Update(update.update_id, message=q.message)
            await practice_summary(dummy, context)
        else:
            await send_practice_item(q, context)
        return

    # --- NUDGE MINI QUIZ ---
    if data == "nudge:start":
        items = await build_mcq("English basics", lang, prefs["cefr"], flavor="generic")
        context.user_data["practice"] = {"type": "mcq", "topic": "nudge", "items": items[:2],
                                         "idx": 0, "score": 0, "ui_lang": lang, "scope": "free"}
        await log_event(context, "nudge_quiz_start", uid, {"count": len(items[:2])})
        return await send_practice_item(q, context)

    if data == "nudge:skip":
        await safe_edit_text(q, "No problem! Let's continue 😊" if lang!="ru" else "Хорошо! Продолжим 😊")
        return


# =========================================================
# 13) NUDGE / REWARD HELPERS
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
        msg = ("Do you want a 2-question mini-quiz?" if lang!="ru"
               else "Хочешь мини-викторину из 2 вопросов?")
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("▶️ Start", callback_data="nudge:start"),
             InlineKeyboardButton("⏭ Skip", callback_data="nudge:skip")]
        ])
        await safe_reply_message(update.message, msg, reply_markup=kb)
        await log_event(context, "nudge_trigger", update.effective_user.id, {})

async def reward_message(update, context, score, total, lang="en"):
    if score == total:
        msg = "🌟 Perfect! All correct!" if lang!="ru" else "🌟 Отлично! Все правильно!"
    elif score / total >= 0.6:
        msg = "⭐ Great work!" if lang!="ru" else "⭐ Отличная работа!"
    else:
        msg = "👏 Nice effort!" if lang!="ru" else "👏 Хорошая попытка!"
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Menu", callback_data="menu:root")]])
    await safe_reply_message(update.message, msg, reply_markup=kb)
    await log_event(context, "reward_given", update.effective_user.id, {"score": score})


# =========================================================
# 14) CHAT-FIRST MESSAGE HANDLER (INTENT DETECTION)
# =========================================================
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    if not text: return

    if blocked(text):
        return await safe_reply_message(update.message, "⛔ Please keep it school-friendly.")

    uid = update.effective_user.id
    prefs = get_prefs(uid)
    lang = prefs.get("lang", "en")
    if lang == "auto": lang = detect_lang(text)

    # --- INTENT DETECTOR ---
    t = text.lower()
    intent = "chat"
    if len(t.split()) <= 2: intent = "vocab"
    if re.search(r"\bdefine\b|\bmeaning of\b", t): intent = "vocab"
    elif re.search(r"\bgrammar\b|\bexplain\b|\brule\b", t): intent = "grammar"
    elif re.search(r"\bread\b|\btext\b|\bwrite\b", t): intent = "reading"
    elif re.search(r"\btranslate\b|\bgloss\b", t): intent = "gloss"
    elif re.search(r"\bquiz\b|\bpractice\b|\bexercise\b", t): intent = "practice"
    elif re.search(r"\btalk\b|\bconversation\b|\bchat\b", t): intent = "talk"

    # --- VOCAB ---
    if intent == "vocab":
        reset_nudge_counter(context)
        headword = text
        card = await build_vocab_card(headword, prefs)
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("✏️ Practice this word", callback_data="vocab:practice"),
             InlineKeyboardButton("➕ More examples", callback_data="vocab:more")],
            [InlineKeyboardButton("🏠 Menu", callback_data="menu:root")]
        ])
        await safe_reply_message(update.message, trim(card), reply_markup=kb)
        await log_event(context, "vocab_card", uid, {"word": headword})
        return await maybe_send_nudge(update, context, lang)

    # --- GRAMMAR ---
    if intent == "grammar":
        reset_nudge_counter(context)
        exp = await build_grammar_explain(text, prefs)
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("✏️ Practice this rule", callback_data="grammar:practice"),
             InlineKeyboardButton("📚 Explain more", callback_data="footer:explain_more")],
            [InlineKeyboardButton("🏠 Menu", callback_data="menu:root")]
        ])
        await safe_reply_message(update.message, trim(exp), reply_markup=kb)
        await log_event(context, "grammar_explain", uid, {"topic": text})
        return await maybe_send_nudge(update, context, lang)

    # --- READING ---
    if intent == "reading":
        reset_nudge_counter(context)
        passage = await build_reading_passage(text, prefs["cefr"])
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("abc Gloss", callback_data="reading:gloss"),
             InlineKeyboardButton("📝 Practice from this text", callback_data="reading:practice")],
            [InlineKeyboardButton("🏠 Menu", callback_data="menu:root")]
        ])
        await safe_reply_message(update.message, trim(passage), reply_markup=kb)
        await log_event(context, "reading_passage", uid, {"topic": text})
        return await maybe_send_nudge(update, context, lang)

    # --- GLOSS ---
    if intent == "gloss":
        passage = context.user_data.get("last_passage", text)
        gloss = await build_reading_gloss(passage, lang)
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("📝 Practice", callback_data="reading:practice"),
             InlineKeyboardButton("🏠 Menu", callback_data="menu:root")]
        ])
        await safe_reply_message(update.message, trim(gloss), reply_markup=kb)
        await log_event(context, "reading_gloss", uid, {})
        return await maybe_send_nudge(update, context, lang)

    # --- PRACTICE ---
    if intent == "practice":
        reset_nudge_counter(context)
        items = await build_mcq(text, lang, prefs["cefr"], flavor="generic")
        context.user_data["practice"] = {"type": "mcq", "topic": text, "items": items,
                                         "idx": 0, "score": 0, "ui_lang": lang, "scope": "free"}
        await log_event(context, "practice_start", uid, {"count": len(items)})
        return await send_practice_item(update, context)

    # --- TALK ---
    if intent == "talk":
        reset_nudge_counter(context)
        msg = "Let's talk! What's your favorite subject?" if lang!="ru" else "Поговорим! Какой твой любимый предмет?"
        await safe_reply_message(update.message, msg, reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🏠 Menu", callback_data="menu:root")]
        ]))
        await log_event(context, "talk_start", uid, {})
        return

    # --- DEFAULT CHAT / PROMPT SUGGESTION ---
    examples = ("Try: Define 'friendship' / Explain 'Present Simple' / Write a short text about 'school life'"
                if lang!="ru"
                else "Попробуй: дай определение 'friendship' / объясни 'Present Simple' / напиши текст о 'school life'")
    steer = f"I can help you study English! {examples}"
    msgs = [{"role": "system", "content": POLICY_CHAT},
            {"role": "user", "content": steer}]
    reply = await ask_openai(msgs, max_tokens=200)
    await safe_reply_message(update.message, trim(reply), reply_markup=InlineKeyboardMarkup([
        [InlineKeyboardButton("🏠 Menu", callback_data="menu:root")]
    ]))
    await log_event(context, "chat_message", uid, {"chars": len(text)})
    await maybe_send_nudge(update, context, lang)


# =========================================================
# 15) FLASK HEALTHCHECK
# =========================================================
app = Flask(__name__)
@app.get("/")
def health(): return "✅ Bot is alive", 200
def start_flask():
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)


# =========================================================
# 16) MAIN ENTRYPOINT
# =========================================================
def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_cmd))
    application.add_handler(CallbackQueryHandler(on_cb))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_error_handler(on_error)
    application.post_init = on_startup
    asyncio.get_event_loop().create_task(asyncio.to_thread(start_flask))
    logger.info("Bot starting with chat-first mode…")
    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == "__main__":
    main()
