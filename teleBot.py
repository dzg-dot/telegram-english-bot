import os
import re
import time
import json
import logging
import threading
import httpx
from dotenv import load_dotenv
from openai import OpenAI
from flask import Flask
from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup
)
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters
)

# ===== LOGGING =====
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ===== STARTUP HOOKS =====
async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.exception("Exception:", exc_info=context.error)

async def on_startup(app: Application):
    await app.bot.delete_webhook(drop_pending_updates=True)
    logger.info("Webhook cleared.")

# ===== LOAD ENV =====
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TOKEN:
    raise RuntimeError("❌ TELEGRAM_TOKEN not found")
USE_OR = os.getenv("USE_OPENROUTER", "True").lower() == "true"

OR_KEY = os.getenv("OPENROUTER_API_KEY")
OA_KEY = os.getenv("OPENAI_API_KEY")

httpx_client = httpx.Client(
    timeout=httpx.Timeout(30.0, read=90.0),
    limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
)

if USE_OR:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OR_KEY,
        http_client=httpx_client,
        default_headers={"HTTP-Referer": "https://t.me/SearchVocabBot"}
    )
    MODEL = "openai/gpt-4o-mini"
else:
    client = OpenAI(api_key=OA_KEY, http_client=httpx_client)
    MODEL = "gpt-3.5-turbo"

# ===== CONSTANTS =====
DEFAULT_LANG = "auto"
ALLOWED_MODES = {"vocab", "reading", "grammar", "quiz", "dialogue"}
GRADE_TO_CEFR = {"6": "A2", "7": "A2+", "8": "B1-", "9": "B1"}
DEFAULT_DIALOGUE_LIMIT = 10
MAX_HISTORY = 10

POLICY = (
    "You are a classroom English tutor for grades 6–9 (A2–B1). "
    "Avoid unsafe topics. Answer briefly, clearly, and age-appropriately."
)

# ===== MEMORY =====
user_prefs = {}
CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")

def detect_lang(text): return "ru" if CYRILLIC_RE.search(text or "") else "en"
def get_prefs(uid):
    if uid not in user_prefs:
        user_prefs[uid] = {"mode": "vocab", "lang": "auto", "grade": "7",
                           "cefr": GRADE_TO_CEFR["7"], "dialogue_limit": DEFAULT_DIALOGUE_LIMIT}
    return user_prefs[uid]

def trim(t, n=900): return (t or "").strip()[:n]

async def ask_openai(msgs, max_tokens=500):
    for _ in range(2):
        try:
            r = client.chat.completions.create(model=MODEL, messages=msgs, max_tokens=max_tokens)
            return r.choices[0].message.content
        except Exception as e:
            err = str(e)
            await asyncio.sleep(1)
    return f"[Error] {err}"

# ===== UTIL FOR BUTTONS =====
def t(uid, en, ru): return ru if get_prefs(uid)["lang"] == "ru" else en
def kb(rows): return InlineKeyboardMarkup([[InlineKeyboardButton(l, callback_data=d) for l, d in r] for r in rows])

# ===== COMMANDS =====
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    text = t(uid, "Hi! How can I help you today?", "Привет! Чем могу помочь?")
    menu = kb([
        [(t(uid, "Mode", "Режим"), "menu:mode"),
         (t(uid, "Language", "Язык"), "menu:lang")],
        [(t(uid, "Grade", "Класс"), "menu:grade"),
         (t(uid, "Help", "Помощь"), "menu:tips")]
    ])
    await update.message.reply_text(text, reply_markup=menu)

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    text = t(uid, "Choose what you want to do:", "Выберите действие:")
    menu = kb([
        [(t(uid, "Mode", "Режим"), "menu:mode"),
         (t(uid, "Language", "Язык"), "menu:lang")],
        [(t(uid, "Grade", "Класс"), "menu:grade")]
    ])
    await update.message.reply_text(text, reply_markup=menu)

async def lang_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    await update.message.reply_text(
        t(uid, "Select language:", "Выберите язык:"),
        reply_markup=kb([[("Auto", "set_lang:auto"), ("EN", "set_lang:en"), ("RU", "set_lang:ru")]])
    )

async def grade_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    await update.message.reply_text(
        t(uid, "Choose grade:", "Выберите класс:"),
        reply_markup=kb([[("6", "set_grade:6"), ("7", "set_grade:7"), ("8", "set_grade:8"), ("9", "set_grade:9")]])
    )

async def mode_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    await update.message.reply_text(
        t(uid, "Select mode:", "Выберите режим:"),
        reply_markup=kb([
            [(t(uid, "Vocab", "Словарь"), "set_mode:vocab"),
             (t(uid, "Quiz", "Викторина"), "set_mode:quiz")],
            [(t(uid, "Reading", "Чтение"), "set_mode:reading"),
             (t(uid, "Grammar", "Грамматика"), "set_mode:grammar")],
            [(t(uid, "Dialogue", "Диалог"), "set_mode:dialogue")]
        ])
    )

# ===== CALLBACKS =====
async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    uid = update.effective_user.id
    prefs = get_prefs(uid)
    data = q.data or ""

    if data.startswith("menu:"):
        if data.endswith("mode"): return await mode_cmd(update, context)
        if data.endswith("lang"): return await lang_cmd(update, context)
        if data.endswith("grade"): return await grade_cmd(update, context)
        if data.endswith("tips"):
            return await q.edit_message_text(
                t(uid, "Tip: after choosing a mode, just type your content.",
                  "Подсказка: после выбора режима просто пишите сообщение.")
            )

    if data.startswith("set_lang:"):
        val = data.split(":")[1]
        prefs["lang"] = val
        return await q.edit_message_text(t(uid, f"Language set to {val}.", f"Язык установлен: {val}."))

    if data.startswith("set_grade:"):
        g = data.split(":")[1]
        prefs["grade"] = g
        prefs["cefr"] = GRADE_TO_CEFR[g]
        return await q.edit_message_text(t(uid, f"Grade {g} (level {prefs['cefr']}) set.",
                                           f"Класс {g} (уровень {prefs['cefr']}) установлен."))

    if data.startswith("set_mode:"):
        m = data.split(":")[1]
        prefs["mode"] = m
        if m == "quiz":
            return await q.edit_message_text(t(uid, "Quiz mode ON. Type a topic (e.g. Pollution).",
                                               "Викторина включена. Введите тему (например, Pollution)."))
        if m == "vocab":
            return await q.edit_message_text(t(uid, "Vocab mode ON. Type a word.", "Режим словаря. Введите слово."))
        if m == "dialogue":
            return await q.edit_message_text(t(uid, "Dialogue mode ON. Say hi to start.",
                                               "Режим диалога. Скажите привет, чтобы начать."))
        if m == "reading":
            return await q.edit_message_text(t(uid, "Reading mode ON. Type a topic.", "Режим чтения. Введите тему."))
        if m == "grammar":
            return await q.edit_message_text(t(uid, "Grammar mode ON. Type a point.",
                                               "Режим грамматики. Введите тему."))

# ===== MAIN HANDLE (vocab + quiz + dialogue, etc.) =====
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    msg = update.message.text.strip()
    prefs = get_prefs(uid)
    mode = prefs["mode"]

    if mode == "vocab":
        prompt = (
            f"Create a simple vocabulary card for word '{msg}' (A2–B1). "
            "Show word, IPA, part of speech, English definition with Russian translation in parentheses, and 2–3 examples."
        )
        r = await ask_openai([{"role":"system","content":POLICY},{"role":"user","content":prompt}],300)
        return await update.message.reply_text(trim(r))

    if mode == "quiz":
        await update.message.reply_text("Thinking...")
        topic = msg
        prompt = (
            f"Create a 5-question multiple-choice quiz about '{topic}'. "
            "Each question has: question, 4 options, correct answer, short explanation_en, explanation_ru. "
            "Return JSON only."
        )
        raw = await ask_openai([{"role":"system","content":POLICY},{"role":"user","content":prompt}],800)
        try:
            data = json.loads(re.search(r"\{.*\}", raw, re.S).group())
        except:
            return await update.message.reply_text("Format error.")
        context.user_data["quiz"] = {"topic": topic, "i": 0, "q": data["questions"], "tries": {}}
        return await send_quiz_q(update, context)

    if mode == "dialogue":
        hist = context.user_data.get("h", [])
        hist.append({"role":"user","content":msg})
        hist = hist[-MAX_HISTORY:]
        context.user_data["h"]=hist
        steer = "You're a friendly teacher. Ask back short questions (A2–B1)."
        r = await ask_openai([{"role":"system","content":steer}]+hist,150)
        hist.append({"role":"assistant","content":r})
        context.user_data["h"]=hist
        return await update.message.reply_text(trim(r))

# ===== QUIZ FLOW =====
async def send_quiz_q(update, context):
    qd = context.user_data.get("quiz")
    if not qd: return
    i = qd["i"]
    if i >= len(qd["q"]):
        return await update.message.reply_text(f"🎉 Great job! You finished the quiz on {qd['topic']}!")
    q = qd["q"][i]
    opts = q["options"]
    rows = [[(opt, f"ans:{i}:{opt}")] for opt in opts]
    markup = kb(rows)
    text = f"Topic: {qd['topic']} | Q{i+1}/{len(qd['q'])}\n{q['question']}"
    await update.message.reply_text(text, reply_markup=markup)

async def on_quiz_answer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data.split(":")
    _, idx, choice = data
    idx = int(idx)
    qd = context.user_data["quiz"]
    question = qd["q"][idx]
    tries = qd["tries"].get(idx, 0)
    correct = question["correct"]
    if choice == correct:
        await q.edit_message_text(f"✅ Correct! {question['explain_en']}")
        qd["i"] += 1
        return await send_quiz_q(update, context)
    else:
        if tries >= 1:
            await q.edit_message_text(f"❌ Wrong twice. Answer: {correct}. {question['explain_en']}")
            qd["i"] += 1
            return await send_quiz_q(update, context)
        else:
            qd["tries"][idx] = tries + 1
            await q.edit_message_text("❌ Try again!")
            return await send_quiz_q(update, context)

# ===== FLASK KEEP-ALIVE =====
app = Flask(__name__)
@app.get("/") 
def health(): return "✅ Bot alive", 200
def start_flask(): app.run(host="0.0.0.0", port=int(os.getenv("PORT","10000")))

# ===== MAIN =====
def main():
    app_ = Application.builder().token(TOKEN).build()
    app_.add_handler(CommandHandler("start", start))
    app_.add_handler(CommandHandler("help", help_cmd))
    app_.add_handler(CommandHandler("mode", mode_cmd))
    app_.add_handler(CommandHandler("lang", lang_cmd))
    app_.add_handler(CommandHandler("grade", grade_cmd))
    app_.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app_.add_handler(CallbackQueryHandler(on_button, pattern="^(menu|set)_"))
    app_.add_handler(CallbackQueryHandler(on_quiz_answer, pattern="^ans:"))
    app_.add_error_handler(on_error)
    app_.post_init = on_startup
    threading.Thread(target=start_flask, daemon=True).start()
    logger.info("Bot is running…")
    app_.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
