import os
import re
import time
import logging
import threading

import httpx
from dotenv import load_dotenv
from openai import OpenAI
from flask import Flask
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# ========== LOGGING ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ========== STARTUP HOOKS ==========
async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.exception("Exception while handling an update:", exc_info=context.error)

async def on_startup(app: Application):
    # Xóa webhook cũ để dùng long-polling
    await app.bot.delete_webhook(drop_pending_updates=True)
    logger.info("Webhook deleted, switching to long-polling.")

# ========== ENV & CLIENTS ==========
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise RuntimeError("❌ TELEGRAM_TOKEN not found in environment")

USE_OPENROUTER = os.getenv("USE_OPENROUTER", "True").lower() == "true"
OR_KEY = os.getenv("OPENROUTER_API_KEY")
OA_KEY = os.getenv("OPENAI_API_KEY")

logger.info("DEBUG => USE_OPENROUTER=%s | OR_KEY? %s | OA_KEY? %s",
            USE_OPENROUTER, bool(OR_KEY), bool(OA_KEY))

httpx_client = httpx.Client(
    timeout=httpx.Timeout(connect=30.0, read=90.0, write=90.0, pool=90.0),
    limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
)

if USE_OPENROUTER:
    if not OR_KEY:
        raise RuntimeError("OPENROUTER_API_KEY missing in env")
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
        raise RuntimeError("OPENAI_API_KEY missing in env")
    client = OpenAI(api_key=OA_KEY, http_client=httpx_client)
    MODEL_NAME = "gpt-3.5-turbo"

# ========== CLASSROOM DEFAULTS & HELPERS ==========
DEFAULT_LANG = "auto"   # auto | en | ru
MAX_HISTORY = 10

ALLOWED_MODES = {"vocab", "reading", "grammar", "quiz"}
BANNED_KEYWORDS = [
    r"\bsex\b", r"\bporn\b", r"\berotic\b",
    r"\bviolence\b", r"\bsuicide\b", r"\bself[- ]?harm\b",
    r"\bdrugs?\b", r"\balcohol\b", r"\bgamble\b",
    r"\bextremis(m|t)\b"
]

GRADE_TO_CEFR = {"6": "A2", "7": "A2+", "8": "B1-", "9": "B1"}

POLICY = (
    "You are a safe classroom teaching assistant for English learning (grades 6–9, ages 12–15).\n"
    "- Answer in ENGLISH by default. If the user's message is in Russian, respond in RUSSIAN.\n"
    "- Allowed scope: vocabulary, reading, grammar, short quizzes; school-safe topics only.\n"
    "- Target level: CEFR A2–B1 (depending on grade). Keep explanations simple and age-appropriate.\n"
    "- If a request is off-topic or unsafe, refuse briefly and redirect back to study tasks.\n"
    "- Keep answers concise (<= 150 words). Vocabulary: include IPA and 2–3 short examples.\n"
)

user_prefs = {}
CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")

def detect_lang(text: str) -> str:
    return "ru" if CYRILLIC_RE.search(text or "") else "en"

def get_prefs(user_id: int):
    if user_id not in user_prefs:
        user_prefs[user_id] = {
            "mode": "vocab",
            "lang": DEFAULT_LANG,
            "grade": "7",
            "cefr": GRADE_TO_CEFR["7"]
        }
    return user_prefs[user_id]

def blocked(text: str) -> bool:
    for pat in BANNED_KEYWORDS:
        if re.search(pat, text or "", flags=re.IGNORECASE):
            return True
    return False

def trim(s: str, max_chars: int = 900) -> str:
    s = re.sub(r"\n{3,}", "\n\n", (s or "").strip())
    return s if len(s) <= max_chars else (s[:max_chars].rstrip() + "…")

async def ask_openai(messages, max_tokens=500):
    """Gọi model với retry + fallback."""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.3,
            )
            return resp.choices[0].message.content
        except Exception as e1:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            # fallback
            base_url = getattr(client, "base_url", "") or ""
            fallback_model = "openai/gpt-3.5-turbo" if "openrouter.ai" in base_url else "gpt-3.5-turbo"
            try:
                resp = client.chat.completions.create(
                    model=fallback_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.3,
                )
                return resp.choices[0].message.content
            except Exception as e2:
                return f"[OpenAI error] {type(e1).__name__}: {e1}"

# ========== COMMANDS ==========
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hi! I’m your English study bot for grades 6–9.\n"
        "Default reply language: English. If you write in Russian, I’ll answer in Russian.\n"
        "Commands: /help, /grade, /mode, /lang, /vocab, /quiz, /clear_history."
    )

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Commands:\n"
        "/grade 6|7|8|9 – set school grade (CEFR will adjust)\n"
        "/mode vocab|reading|grammar|quiz – choose study mode\n"
        "/lang auto|en|ru – response language (auto = detect)\n"
        "/vocab <word> – IPA + short meaning + 2–3 examples\n"
        "/quiz [topic] [A2|B1] – 5 MCQs with answer key\n"
        "/clear_history – clear chat context"
    )

async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["history"] = []
    await update.message.reply_text("Context cleared. Let's start fresh!")

async def grade_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prefs = get_prefs(update.effective_user.id)
    if not context.args or context.args[0] not in {"6", "7", "8", "9"}:
        return await update.message.reply_text("Use: /grade 6|7|8|9")
    g = context.args[0]
    prefs["grade"] = g
    prefs["cefr"] = GRADE_TO_CEFR[g]
    await update.message.reply_text(f"Grade set to {g}. Target level: {prefs['cefr']}.")

async def mode_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prefs = get_prefs(update.effective_user.id)
    if not context.args or context.args[0] not in ALLOWED_MODES:
        return await update.message.reply_text("Use: /mode vocab|reading|grammar|quiz")
    prefs["mode"] = context.args[0]
    await update.message.reply_text(f"Mode: {prefs['mode']}")

async def lang_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prefs = get_prefs(update.effective_user.id)
    if not context.args or context.args[0] not in {"auto", "en", "ru"}:
        return await update.message.reply_text("Use: /lang auto|en|ru")
    prefs["lang"] = context.args[0]
    await update.message.reply_text(f"Response language: {prefs['lang']}")

async def ping_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        msg = [{"role": "user", "content": "Say 'pong' in one word."}]
        text = await ask_openai(msg, max_tokens=5)
        await update.message.reply_text(f"✅ OpenAI connected: {text}")
    except Exception as e:
        await update.message.reply_text(f"⚠️ OpenAI error: {e}")

async def vocab_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prefs = get_prefs(update.effective_user.id)
    if not context.args:
        return await update.message.reply_text("Use: /vocab <word>")
    word = " ".join(context.args)
    if blocked(word):
        return await update.message.reply_text("⛔ Off-topic. Please ask study-related content.")

    lang = prefs["lang"]
    if lang == "auto":
        lang = detect_lang(update.message.text)

    prompt_user = (
        f"User language: {lang}\n"
        f"Grade: {prefs['grade']} (target {prefs['cefr']})\n"
        f"Task: VOCAB card for '{word}'. Include IPA, concise meaning "
        f"in user's language, and 2–3 short, school-safe examples at {prefs['cefr']} level."
    )
    messages = [
        {"role": "system", "content": POLICY},
        {"role": "user", "content": prompt_user},
    ]
    try:
        text = await ask_openai(messages, max_tokens=350)
        await update.message.reply_text(trim(text))
    except Exception as e:
        await update.message.reply_text(f"⚠️ OpenAI error: {e}")

async def quiz_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prefs = get_prefs(update.effective_user.id)
    topic = context.args[0] if len(context.args) >= 1 else "school life"
    level = context.args[1] if len(context.args) >= 2 else prefs["cefr"]
    if blocked(topic):
        return await update.message.reply_text("⛔ Off-topic. Please ask study-related content.")

    lang = prefs["lang"]
    if lang == "auto":
        lang = detect_lang(update.message.text)

    prompt_user = (
        f"User language: {lang}\n"
        f"Grade: {prefs['grade']} (target {prefs['cefr']})\n"
        f"Task: Create a 5-question multiple-choice quiz (4 options each) on '{topic}', "
        f"level {level}. Use Russian instructions if user language is Russian, "
        f"otherwise English. Keep explanations short and include an answer key."
    )
    messages = [
        {"role": "system", "content": POLICY},
        {"role": "user", "content": prompt_user},
    ]
    try:
        text = await ask_openai(messages, max_tokens=600)
        await update.message.reply_text(trim(text))
    except Exception as e:
        await update.message.reply_text(f"⚠️ OpenAI error: {e}")

# ========== FREE CHAT ==========
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text or ""
    if blocked(user_message):
        return await update.message.reply_text(
            "⛔ That's outside our classroom scope. "
            "Try vocabulary, reading, grammar, or a quiz topic."
        )

    await update.message.reply_text("Thinking…")

    prefs = get_prefs(update.effective_user.id)
    lang = prefs["lang"]
    if lang == "auto":
        lang = detect_lang(user_message)

    history = context.user_data.get("history", [])
    history.append({"role": "user", "content": user_message})
    history = history[-MAX_HISTORY:]
    context.user_data["history"] = history

    mode_instruction = {
        "vocab":   "Behave as VOCAB helper: IPA, brief meaning in user's language, 2–3 short examples (A2–B1).",
        "reading": "Provide a short reading (80–120 words) on a school-safe topic + 2–3 comprehension questions.",
        "grammar": "Explain the grammar point (A2–B1) in 3–5 concise bullets + 1–2 examples.",
        "quiz":    "Create a 5-question MCQ quiz (4 options) for the user's topic; include answer key."
    }[prefs["mode"]]

    steer = (
        f"User language: {lang}\n"
        f"Grade: {prefs['grade']} (target {prefs['cefr']})\n"
        f"Mode: {prefs['mode']}\nInstruction: {mode_instruction}\n"
        "If the user's request is off-topic, briefly refuse and redirect to the current mode."
    )

    messages = [
        {"role": "system", "content": POLICY},
        {"role": "user", "content": steer},
        *history
    ]

    try:
        text = await ask_openai(messages, max_tokens=500)
        context.user_data["history"].append({"role": "assistant", "content": text})
        await update.message.reply_text(trim(text))
    except Exception as e:
        await update.message.reply_text(f"Error: {str(e)}. Please try again.")

# ========== FLASK (KEEP PORT OPEN FOR RENDER) ==========
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

    # Commands
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_cmd))
    application.add_handler(CommandHandler("clear_history", clear_history))
    application.add_handler(CommandHandler("grade", grade_cmd))
    application.add_handler(CommandHandler("mode", mode_cmd))
    application.add_handler(CommandHandler("lang", lang_cmd))
    application.add_handler(CommandHandler("vocab", vocab_cmd))
    application.add_handler(CommandHandler("quiz", quiz_cmd))
    application.add_handler(CommandHandler("ping", ping_cmd))

    # Free text
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Hooks
    application.add_error_handler(on_error)
    application.post_init = on_startup

    # Run Flask + polling
    threading.Thread(target=start_flask, daemon=True).start()
    logger.info("Bot is starting (Web Service + Flask)…")
    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == "__main__":
    main()
