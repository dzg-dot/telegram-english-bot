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
    Application, CommandHandler, MessageHandler, CallbackQueryHandler,
    ContextTypes, filters
)

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

# --------------- STARTUP HOOKS ---------------
async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.exception("Exception while handling an update:", exc_info=context.error)

async def on_startup(app: Application):
    await app.bot.delete_webhook(drop_pending_updates=True)
    logger.info("Webhook deleted, switching to long-polling.")

# --------------- ENV & CLIENTS ---------------
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN missing")

USE_OPENROUTER = os.getenv("USE_OPENROUTER", "True").lower() == "true"
OR_KEY = os.getenv("OPENROUTER_API_KEY")
OA_KEY = os.getenv("OPENAI_API_KEY")

logger.info("DEBUG => USE_OPENROUTER=%s | OR_KEY? %s | OA_KEY? %s",
            USE_OPENROUTER, bool(OR_KEY), bool(OA_KEY))

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

# --------------- CLASSROOM DEFAULTS ---------------
DEFAULT_LANG = "auto"   # auto|en|ru
MAX_HISTORY = 10
ALLOWED_MODES = {"vocab", "reading", "grammar", "quiz", "dialogue"}

BANNED_KEYWORDS = [
    r"\bsex\b", r"\bporn\b", r"\berotic\b",
    r"\bviolence\b", r"\bsuicide\b", r"\bself[- ]?harm\b",
    r"\bdrugs?\b", r"\balcohol\b", r"\bgamble\b",
    r"\bextremis(m|t)\b"
]

DEFAULT_DIALOGUE_LIMIT = 10
GRADE_TO_CEFR = {"6": "A2", "7": "A2+", "8": "B1-", "9": "B1"}

POLICY = (
    "You are a safe classroom teaching assistant for English learning (grades 6–9, ages 12–15). "
    "Do not use markdown bold or headings. "
    "Answer in English by default; if the user writes in Russian, respond in Russian. "
    "Allowed scope: vocabulary, reading, grammar, short quizzes; school-safe topics only. "
    "Target level: CEFR A2–B1. Keep explanations simple and concise (<=150 words). "
    "If a request is off-topic or unsafe, refuse briefly and redirect to learning tasks."
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
            "cefr": GRADE_TO_CEFR["7"],
            "dialogue_limit": DEFAULT_DIALOGUE_LIMIT,
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
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME, messages=messages,
                max_tokens=max_tokens, temperature=0.3
            )
            return resp.choices[0].message.content
        except Exception as e1:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            base_url = getattr(client, "base_url", "") or ""
            fb = "openai/gpt-3.5-turbo" if "openrouter.ai" in base_url else "gpt-3.5-turbo"
            try:
                resp = client.chat.completions.create(
                    model=fb, messages=messages,
                    max_tokens=max_tokens, temperature=0.3
                )
                return resp.choices[0].message.content
            except Exception:
                return f"[OpenAI error] {type(e1).__name__}: {e1}"

# --------------- UI HELPERS ---------------
def root_menu(lang: str) -> InlineKeyboardMarkup:
    if lang == "ru":
        txt = [["Режим", "menu:mode"], ["Язык", "menu:lang"]],
    kb = [
        [InlineKeyboardButton("Mode", callback_data="menu:mode"),
         InlineKeyboardButton("Language", callback_data="menu:lang")],
        [InlineKeyboardButton("Grade", callback_data="menu:grade"),
         InlineKeyboardButton("Help", callback_data="menu:help")],
    ]
    return InlineKeyboardMarkup(kb)

def mode_menu() -> InlineKeyboardMarkup:
    kb = [
        [InlineKeyboardButton("Vocab",   callback_data="set_mode:vocab"),
         InlineKeyboardButton("Reading", callback_data="set_mode:reading")],
        [InlineKeyboardButton("Grammar", callback_data="set_mode:grammar"),
         InlineKeyboardButton("Quiz",    callback_data="set_mode:quiz")],
        [InlineKeyboardButton("Back", callback_data="menu:root")]
    ]
    return InlineKeyboardMarkup(kb)

def lang_menu() -> InlineKeyboardMarkup:
    kb = [
        [InlineKeyboardButton("English", callback_data="set_lang:en"),
         InlineKeyboardButton("Русский", callback_data="set_lang:ru")],
        [InlineKeyboardButton("Back", callback_data="menu:root")]
    ]
    return InlineKeyboardMarkup(kb)

def grade_menu() -> InlineKeyboardMarkup:
    kb = [
        [InlineKeyboardButton("6", callback_data="set_grade:6"),
         InlineKeyboardButton("7", callback_data="set_grade:7"),
         InlineKeyboardButton("8", callback_data="set_grade:8"),
         InlineKeyboardButton("9", callback_data="set_grade:9")],
        [InlineKeyboardButton("Back", callback_data="menu:root")]
    ]
    return InlineKeyboardMarkup(kb)

# --------------- START / HELP ---------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # greeting
    txt = "Hi! How can I help you today?"
    # default EN; if user typed RU, greet in RU
    if detect_lang(update.message.text or "") == "ru":
        txt = "Привет! Чем я могу помочь сегодня?"
    await update.message.reply_text(txt, reply_markup=root_menu("en"))

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prefs = get_prefs(update.effective_user.id)
    lang = prefs.get("lang", "en")
    text = (
        "Commands:\n"
        "/start – open menu\n"
        "/help – show menu and tips\n"
        "/grade 6|7|8|9 – set grade\n"
        "/mode vocab|reading|grammar|quiz – choose study mode\n"
        "/lang auto|en|ru – response language\n\n"
        "Tip: after you choose a mode, just type content.\n"
        "In Vocab: send a word. In Quiz: send a topic."
    )
    if lang == "ru":
        text = (
            "Команды:\n"
            "/start – открыть меню\n"
            "/help – меню и подсказки\n"
            "/grade 6|7|8|9 – выбрать класс\n"
            "/mode vocab|reading|grammar|quiz – режим\n"
            "/lang auto|en|ru – язык ответа\n\n"
            "Подсказка: после выбора режима просто пишите сообщение.\n"
            "Vocab: слово. Quiz: тема."
        )
    await update.message.reply_text(text, reply_markup=root_menu(lang))

# --------------- VOCAB ---------------
async def vocab_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # giữ command cho ai thích, nhưng không bắt buộc
    prefs = get_prefs(update.effective_user.id)
    prefs["mode"] = "vocab"
    await update.message.reply_text("Vocab mode is ON. Send me a word.")

async def build_vocab_card(headword: str, prefs: dict, user_text: str) -> str:
    # show EN def + short RU in parentheses; POS + IPA; no bold
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
    msgs = [{"role": "system", "content": POLICY},
            {"role": "user", "content": prompt}]
    return await ask_openai(msgs, max_tokens=320)

# --------------- QUIZ STATE ---------------
# context.user_data["quiz"] = {
#   "topic": str, "level": str, "questions": [ {id, q, options[A..D], correct, explain_en, explain_ru}... ],
#   "idx": 0, "attempts": 0
# }
def quiz_buttons(options):
    # Each button text includes the option letter and text.
    letters = ["A", "B", "C", "D"]
    row1 = [
        InlineKeyboardButton(f"A) {options[0]}", callback_data="qa:A"),
        InlineKeyboardButton(f"B) {options[1]}", callback_data="qa:B"),
    ]
    row2 = [
        InlineKeyboardButton(f"C) {options[2]}", callback_data="qa:C"),
        InlineKeyboardButton(f"D) {options[3]}", callback_data="qa:D"),
    ]
    return InlineKeyboardMarkup([row1, row2])

async def send_quiz_question(update_or_query, context: ContextTypes.DEFAULT_TYPE):
    """Send current question (by index) with buttons."""
    data = context.user_data.get("quiz")
    if not data:
        return
    i = data["idx"]
    q = data["questions"][i]
    title = f"Topic: {data['topic'].title()} | Q{i+1}/{len(data['questions'])}"
    text = f"{title}\n\n{q['question']}"
    buttons = quiz_buttons(q["options"])
    if isinstance(update_or_query, Update):
        await update_or_query.message.reply_text(text, reply_markup=buttons)
    else:
        # CallbackQuery
        await update_or_query.edit_message_text(text, reply_markup=buttons)

async def build_quiz(topic: str, prefs: dict, user_text: str, lang_detected: str):
    level = prefs["cefr"]
    ui_lang = prefs.get("lang", "auto")
    if ui_lang == "auto":
        ui_lang = lang_detected

    prompt = (
        f"Create a 5-question MCQ quiz on '{topic}', level {level}, grades 6–9. "
        "Return STRICT JSON only (no prose, no markdown):\n"
        "{ \"questions\": [\n"
        "{\"id\":1,\"question\":\"...\",\"options\":[\"...\",\"...\",\"...\",\"...\"],"
        "\"correct\":\"A\",\"explain_en\":\"<=25 words\",\"explain_ru\":\"<=25 words\"},\n"
        "{\"id\":2,...},{\"id\":3,...},{\"id\":4,...},{\"id\":5,...}\n"
        "]}\n"
        f"Language for 'question' and 'options': {'Russian' if ui_lang=='ru' else 'English'} "
        "at A2–B1 simplicity. Keep school-safe. Do not include answer hints in text."
    )
    msgs = [{"role": "system", "content": POLICY},
            {"role": "user", "content": prompt}]
    raw = await ask_openai(msgs, max_tokens=800)

    def extract_json(s: str):
        s = s.strip()
        if "```" in s:
            parts = s.split("```")
            for i in range(len(parts)-1):
                block = parts[i+1]
                if block.lstrip().startswith("json"):
                    return json.loads(block.split("\n", 1)[1])
                try:
                    return json.loads(block)
                except Exception:
                    continue
        return json.loads(s)

    data = extract_json(raw)
    qs = []
    for q in data.get("questions", []):
        qs.append({
            "id": q.get("id"),
            "question": q.get("question"),
            "options": q.get("options", ["", "", "", ""]),
            "correct": q.get("correct", "A"),
            "explain_en": q.get("explain_en", ""),
            "explain_ru": q.get("explain_ru", "")
        })
    return qs

# --------------- COMMANDS ---------------
async def quiz_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prefs = get_prefs(update.effective_user.id)
    prefs["mode"] = "quiz"
    prefs["dialogue_turns"] = 0
    await update.message.reply_text("Quiz mode is ON. Send me a topic (e.g., pollution).")

# --------------- CALLBACKS ---------------
async def on_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    data = q.data or ""
    await q.answer()  # stop the spinner fast
    uid = update.effective_user.id
    prefs = get_prefs(uid)

    # --- MENUS ---
    if data == "menu:root":
        await q.edit_message_text("Hi! How can I help you today?", reply_markup=root_menu(prefs.get("lang","en")))
        return

    if data == "menu:mode":
        await q.edit_message_text("Choose a mode:", reply_markup=mode_menu())
        return

    if data == "menu:lang":
        await q.edit_message_text("Choose language:", reply_markup=lang_menu())
        return

    if data == "menu:grade":
        await q.edit_message_text("Choose grade:", reply_markup=grade_menu())
        return

    if data.startswith("set_mode:"):
        mode = data.split(":", 1)[1]
        prefs["mode"] = mode
        txt = f"Mode set to {mode}."
        if mode == "quiz":
            txt += " Send me a topic."
        elif mode == "vocab":
            txt += " Send me a word."
        await q.edit_message_text(txt, reply_markup=root_menu(prefs.get("lang","en")))
        return

    if data.startswith("set_lang:"):
        lang = data.split(":", 1)[1]
        prefs["lang"] = lang
        await q.edit_message_text(f"Language set to {lang.upper()}.", reply_markup=root_menu(lang))
        return

    if data.startswith("set_grade:"):
        g = data.split(":", 1)[1]
        if g in GRADE_TO_CEFR:
            prefs["grade"] = g
            prefs["cefr"] = GRADE_TO_CEFR[g]
            await q.edit_message_text(f"Grade set to {g}. Target level: {prefs['cefr']}.",
                                      reply_markup=root_menu(prefs.get("lang","en")))
        else:
            await q.edit_message_text("Invalid grade.", reply_markup=root_menu(prefs.get("lang","en")))
        return

    # --- QUIZ ANSWER BUTTONS ---
    if data.startswith("qa:"):
        choice = data.split(":", 1)[1]  # A/B/C/D
        pack = context.user_data.get("quiz")
        if not pack:
            await q.edit_message_text("No active quiz. Choose Quiz mode and send a topic.", reply_markup=root_menu(prefs.get("lang","en")))
            return
        i = pack["idx"]
        question = pack["questions"][i]
        correct = question["correct"]
        ui_lang = prefs.get("lang", "auto")
        if ui_lang == "auto":
            ui_lang = "ru" if CYRILLIC_RE.search(q.message.text or "") else "en"

        if choice == correct:
            # correct: show explanation and next
            expl = question["explain_ru"] if ui_lang == "ru" and question["explain_ru"] else question["explain_en"]
            ok = "Верно!" if ui_lang == "ru" else "Correct!"
            msg = f"{ok}\n{expl}".strip()
            await q.edit_message_text(msg)
            # next
            pack["idx"] += 1
            pack["attempts"] = 0
            if pack["idx"] >= len(pack["questions"]):
                done = "Quiz finished. Great job!" if ui_lang != "ru" else "Тест завершен. Отличная работа!"
                await q.message.reply_text(done)
                context.user_data.pop("quiz", None)
            else:
                # small guiding hint for Q2 (open-ended nudge)
                if pack["idx"] == 1:
                    hint = "Tip: think about cause and effect." if ui_lang != "ru" else "Подсказка: подумай о причине и следствии."
                    await q.message.reply_text(hint)
                await send_quiz_question(q, context)
            return
        else:
            # wrong
            pack["attempts"] += 1
            if pack["attempts"] < 2:
                msg = "Not quite. Try again." if ui_lang != "ru" else "Почти. Попробуй еще раз."
                await q.edit_message_text(msg)
                await send_quiz_question(q, context)
                return
            # second wrong -> reveal
            expl = question["explain_ru"] if ui_lang == "ru" and question["explain_ru"] else question["explain_en"]
            ans = f"The correct answer is {correct}." if ui_lang != "ru" else f"Правильный ответ: {correct}."
            await q.edit_message_text(f"{ans}\n{expl}".strip())
            pack["idx"] += 1
            pack["attempts"] = 0
            if pack["idx"] >= len(pack["questions"]):
                done = "Quiz finished. Keep practicing!" if ui_lang != "ru" else "Тест завершен. Продолжай тренироваться!"
                await q.message.reply_text(done)
                context.user_data.pop("quiz", None)
            else:
                await send_quiz_question(q, context)
            return

# --------------- FREE TEXT HANDLER ---------------
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text or ""
    if blocked(user_message):
        return await update.message.reply_text(
            "⛔ That's outside our classroom scope. Try vocabulary, reading, grammar, or a quiz topic."
        )

    prefs = get_prefs(update.effective_user.id)
    lang = prefs["lang"]
    if lang == "auto":
        lang = detect_lang(user_message)

    # Mode shortcuts
    if prefs["mode"] == "vocab":
        word = user_message.strip()
        if not word:
            return await update.message.reply_text("Send a word to look up.")
        await update.message.reply_text("Thinking…")
        card = await build_vocab_card(word, prefs, update.message.text)
        return await update.message.reply_text(trim(card))

    if prefs["mode"] == "quiz":
        # Treat free text as topic to build a new quiz
        topic = user_message.strip() or "school life"
        await update.message.reply_text("Thinking…")
        try:
            qs = await build_quiz(topic, prefs, update.message.text, lang)
        except Exception:
            return await update.message.reply_text("Sorry, quiz building failed. Please try again.")
        if not qs:
            return await update.message.reply_text("Couldn't build the quiz. Try another topic.")
        context.user_data["quiz"] = {"topic": topic, "level": prefs["cefr"],
                                     "questions": qs, "idx": 0, "attempts": 0}
        return await send_quiz_question(update, context)

    # other modes: simple steering chat (reading/grammar/dialogue)
    await update.message.reply_text("Thinking…")
    history = context.user_data.get("history", [])
    history.append({"role": "user", "content": user_message})
    history = history[-MAX_HISTORY:]
    context.user_data["history"] = history

    mode_instruction = {
        "reading": "Provide a short reading (80–120 words) on a school-safe topic plus 2–3 comprehension questions. No bold.",
        "grammar": "Explain the grammar point (A2–B1) in 3–5 short bullets with 1–2 examples. No bold.",
        "dialogue": (
            "You are a friendly English conversation tutor for grades 6–9. "
            "Reply in 1–3 short sentences, safe topics only. If off-topic, redirect to learning."
        ),
        "vocab": "Vocabulary helper.",
        "quiz": "Quiz builder."
    }.get(prefs["mode"], "General helper for English study.")

    steer = (
        f"User language: {lang}\n"
        f"Grade: {prefs['grade']} (target {prefs['cefr']})\n"
        f"Mode: {prefs['mode']}\nInstruction: {mode_instruction}"
    )

    messages = [
        {"role": "system", "content": POLICY},
        {"role": "user", "content": steer},
        *history
    ]
    text = await ask_openai(messages, max_tokens=500)
    context.user_data["history"].append({"role": "assistant", "content": text})
    await update.message.reply_text(trim(text))

# --------------- FLASK (KEEP PORT OPEN) ---------------
app = Flask(__name__)

@app.get("/")
def health():
    return "✅ Bot is alive", 200

def start_flask():
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)

# --------------- MAIN ---------------
def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_cmd))
    application.add_handler(CommandHandler("grade", lambda u,c: None))  # kept for compatibility
    application.add_handler(CommandHandler("mode",  lambda u,c: None))
    application.add_handler(CommandHandler("lang",  lambda u,c: None))
    application.add_handler(CommandHandler("vocab", vocab_cmd))
    application.add_handler(CommandHandler("quiz",  quiz_cmd))

    application.add_handler(CallbackQueryHandler(on_cb))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.add_error_handler(on_error)
    application.post_init = on_startup

    threading.Thread(target=start_flask, daemon=True).start()
    logger.info("Bot is starting (Web Service + Flask)…")
    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == "__main__":
    main()
