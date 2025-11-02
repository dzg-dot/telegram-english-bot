import os
import re
import time
import logging
import threading
import json

import httpx
from dotenv import load_dotenv
from openai import OpenAI
from flask import Flask
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler,
    filters, ContextTypes
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

ALLOWED_MODES = {"vocab", "reading", "grammar", "quiz", "dialogue"}
BANNED_KEYWORDS = [
    r"\bsex\b", r"\bporn\b", r"\berotic\b",
    r"\bviolence\b", r"\bsuicide\b", r"\bself[- ]?harm\b",
    r"\bdrugs?\b", r"\balcohol\b", r"\bgamble\b",
    r"\bextremis(m|t)\b"
]

# === DIALOGUE CONFIG ===
DEFAULT_DIALOGUE_LIMIT = 10

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
GREET_RE = re.compile(r"^(hi|hello|hey|привет|здравств|добрый|hola)\b", re.I)

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

# (giữ lại – không còn dùng trong quiz mới, nhưng để sẵn nếu bạn cần sau này)
EN_ASK_ANS = re.compile(r"\b(give me answer|show answer|answer please)\b", re.I)
RU_ASK_ANS = re.compile(r"\b(дай\s+ответ|покажи\s+ответ|ответ\s+пожалуйста)\b", re.I)
def is_answer_request(text: str) -> bool:
    t = (text or "").strip()
    return bool(EN_ASK_ANS.search(t) or RU_ASK_ANS.search(t))

# prefix callback
CBQ_QUIZ_PREFIX = "QUIZ:"
CBQ_HELP_PREFIX = "HELP:"

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
            except Exception:
                return f"[OpenAI error] {type(e1).__name__}: {e1}"

# ========== INLINE HELP ==========
async def on_help_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    key = q.data.replace(CBQ_HELP_PREFIX, "")
    mapping = {
        "grade": "Use: /grade 6|7|8|9",
        "mode": "Use: /mode vocab|reading|grammar|quiz",
        "lang": "Use: /lang auto|en|ru",
        "vocab": "Use: /vocab <word>",
        "quiz": "Type /quiz then send a topic. I’ll ask one question at a time.",
        "talk": "Use: /talk [number] to start a short dialogue.",
        "clear": "Use: /clear_history to reset context."
    }
    try:
        await q.edit_message_reply_markup(reply_markup=None)
    except Exception:
        pass
    if key in mapping:
        await q.message.reply_text(mapping[key])

# ========== QUIZ HELPERS ==========
def _quiz_keyboard(i: int):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("A", callback_data=f"{CBQ_QUIZ_PREFIX}A:{i}"),
         InlineKeyboardButton("B", callback_data=f"{CBQ_QUIZ_PREFIX}B:{i}")],
        [InlineKeyboardButton("C", callback_data=f"{CBQ_QUIZ_PREFIX}C:{i}"),
         InlineKeyboardButton("D", callback_data=f"{CBQ_QUIZ_PREFIX}D:{i}")],
        [InlineKeyboardButton("⏭ Skip", callback_data=f"{CBQ_QUIZ_PREFIX}SKIP:{i}")]
    ])

async def send_quiz_question(context: ContextTypes.DEFAULT_TYPE, chat_id: int):
    qstate = context.user_data.get("quiz")
    if not qstate:
        return
    i = qstate["i"]
    qs = qstate["questions"]
    if i >= len(qs):
        topic = qstate["topic"]; level = qstate["level"]
        lang = qstate["lang"]
        done = "Викторина завершена!" if lang=="ru" else "Quiz finished!"
        await context.bot.send_message(chat_id=chat_id, text=f"{done} Topic: {topic} (level {level}).")
        context.user_data.pop("quiz", None)
        return

    q = qs[i]
    header = f"Topic: {qstate['topic']} | Q{i+1}/{len(qs)}"
    text = f"{header}\n\n{q.get('question')}\n"
    await context.bot.send_message(chat_id=chat_id, text=text, reply_markup=_quiz_keyboard(i))

async def on_quiz_answer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    if not q.data.startswith(CBQ_QUIZ_PREFIX):
        return
    data = q.data.replace(CBQ_QUIZ_PREFIX, "")  # e.g. "A:0" or "SKIP:0"
    try:
        choice, idx_str = data.split(":")
        idx = int(idx_str)
    except Exception:
        return

    qstate = context.user_data.get("quiz")
    if not qstate or idx != qstate.get("i", -1):
        return  # bấm muộn / trạng thái không khớp

    curr = qstate["questions"][idx]
    correct = (curr.get("correct") or "A").strip().upper()
    lang = qstate["lang"]
    explanation = curr.get("explain_ru") if lang=="ru" else curr.get("explain_en")
    explanation = explanation or ""
    chat_id = q.message.chat_id

    try:
        await q.edit_message_reply_markup(reply_markup=None)
    except Exception:
        pass

    if choice == "SKIP":
        msg = "Пропустим." if lang=="ru" else "Skipped."
        if explanation:
            msg += " " + explanation
        await q.message.reply_text(msg)
        qstate["i"] += 1
        qstate["tries_left"] = 2
        await send_quiz_question(context, chat_id)
        return

    if choice == correct:
        good = "Правильно! " if lang=="ru" else "Correct! "
        await q.message.reply_text(good + (explanation if explanation else ""))
        qstate["i"] += 1
        qstate["tries_left"] = 2
        await send_quiz_question(context, chat_id)
    else:
        qstate["tries_left"] -= 1
        if qstate["tries_left"] > 0:
            try_again = "Неверно. Попробуй ещё раз." if lang=="ru" else "Not quite. Try again."
            # gửi lại phím cho cùng câu hỏi
            await q.message.reply_text(try_again, reply_markup=_quiz_keyboard(idx))
        else:
            ans = "Ответ: " if lang=="ru" else "Answer: "
            await q.message.reply_text(f"{ans}{correct}. {explanation}")
            qstate["i"] += 1
            qstate["tries_left"] = 2
            await send_quiz_question(context, chat_id)

# ========== COMMANDS ==========
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prefs = get_prefs(update.effective_user.id)
    lang = prefs.get("lang", "auto")
    if lang == "auto":
        lang = detect_lang(update.message.text or "")

    if lang == "ru":
        text = ("Привет! Я школьный бот по английскому (6–9 классы). "
                "Отвечаю по-английски, а если ты пишешь по-русски — по-русски. "
                "Чем помочь сегодня?")
        btns = [
            [InlineKeyboardButton("📘 Словарь /vocab", callback_data=CBQ_HELP_PREFIX+"vocab"),
             InlineKeyboardButton("🧩 Викторина /quiz", callback_data=CBQ_HELP_PREFIX+"quiz")],
            [InlineKeyboardButton("🎯 Режим /mode", callback_data=CBQ_HELP_PREFIX+"mode"),
             InlineKeyboardButton("🌐 Язык /lang", callback_data=CBQ_HELP_PREFIX+"lang")],
            [InlineKeyboardButton("🧹 /clear_history", callback_data=CBQ_HELP_PREFIX+"clear")]
        ]
    else:
        text = ("Hi! I’m your English study bot for grades 6–9. "
                "I answer in English; if you write in Russian, I’ll answer in Russian. "
                "How can I help you today?")
        btns = [
            [InlineKeyboardButton("📘 Vocab /vocab", callback_data=CBQ_HELP_PREFIX+"vocab"),
             InlineKeyboardButton("🧩 Quiz /quiz", callback_data=CBQ_HELP_PREFIX+"quiz")],
            [InlineKeyboardButton("🎯 Mode /mode", callback_data=CBQ_HELP_PREFIX+"mode"),
             InlineKeyboardButton("🌐 Lang /lang", callback_data=CBQ_HELP_PREFIX+"lang")],
            [InlineKeyboardButton("🧹 /clear_history", callback_data=CBQ_HELP_PREFIX+"clear")]
        ]

    await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(btns))

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prefs = get_prefs(update.effective_user.id)
    lang = prefs.get("lang", "auto")
    if lang == "auto":
        lang = detect_lang(update.message.text or "")

    if lang == "ru":
        text = (
            "Команды:\n"
            "/start – приветствие\n"
            "/grade 6|7|8|9 – выбрать класс\n"
            "/mode vocab|reading|grammar|quiz – режим обучения\n"
            "/lang auto|en|ru – язык ответа\n"
            "/vocab <слово> – IPA, часть речи (POS), определение EN (RU), 2–3 примера\n"
            "/quiz – вопросы по одному, 2 попытки, Skip\n"
            "/clear_history – очистить контекст\n"
            "/talk [число] – диалог (по умолчанию 10)\n"
            "/endtalk – завершить диалог\n"
        )
        btns = [
            [InlineKeyboardButton("➕ /grade", callback_data=CBQ_HELP_PREFIX+"grade"),
             InlineKeyboardButton("🎯 /mode", callback_data=CBQ_HELP_PREFIX+"mode")],
            [InlineKeyboardButton("🌐 /lang", callback_data=CBQ_HELP_PREFIX+"lang"),
             InlineKeyboardButton("📘 /vocab", callback_data=CBQ_HELP_PREFIX+"vocab")],
            [InlineKeyboardButton("🧩 /quiz", callback_data=CBQ_HELP_PREFIX+"quiz"),
             InlineKeyboardButton("💬 /talk", callback_data=CBQ_HELP_PREFIX+"talk")],
            [InlineKeyboardButton("🧹 /clear_history", callback_data=CBQ_HELP_PREFIX+"clear")]
        ]
    else:
        text = (
            "Commands:\n"
            "/start – introduction\n"
            "/grade 6|7|8|9 – set grade\n"
            "/mode vocab|reading|grammar|quiz – choose mode\n"
            "/lang auto|en|ru – response language\n"
            "/vocab <word> – IPA, POS, definition EN (RU), 2–3 examples\n"
            "/quiz – one question at a time, 2 tries, Skip\n"
            "/clear_history – clear context\n"
            "/talk [number] – dialogue (default 10)\n"
            "/endtalk – end dialogue\n"
        )
        btns = [
            [InlineKeyboardButton("➕ /grade", callback_data=CBQ_HELP_PREFIX+"grade"),
             InlineKeyboardButton("🎯 /mode", callback_data=CBQ_HELP_PREFIX+"mode")],
            [InlineKeyboardButton("🌐 /lang", callback_data=CBQ_HELP_PREFIX+"lang"),
             InlineKeyboardButton("📘 /vocab", callback_data=CBQ_HELP_PREFIX+"vocab")],
            [InlineKeyboardButton("🧩 /quiz", callback_data=CBQ_HELP_PREFIX+"quiz"),
             InlineKeyboardButton("💬 /talk", callback_data=CBQ_HELP_PREFIX+"talk")],
            [InlineKeyboardButton("🧹 /clear_history", callback_data=CBQ_HELP_PREFIX+"clear")]
        ]
    await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(btns))

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

    headword = " ".join(context.args).strip()
    if blocked(headword):
        return await update.message.reply_text("⛔ Off-topic. Please ask study-related content.")

    # Luôn hiển thị định nghĩa tiếng Anh + tiếng Nga trong ngoặc
    lang_for_examples = prefs.get("lang", "auto")
    if lang_for_examples == "auto":
        lang_for_examples = detect_lang(update.message.text or "")
    include_ru_examples = (lang_for_examples == "ru")

    prompt = (
        "You are an English-learning assistant for grades 6–9 (CEFR A2–B1). "
        "Create a concise, school-safe vocabulary card for the given word. "
        "The definition must ALWAYS be in English with a short Russian translation in parentheses. "
        "Follow the EXACT format and rules.\n\n"
        f"HEADWORD: {headword}\n"
        f"TARGET LEVEL: {prefs['cefr']}\n\n"
        "FORMAT EXACTLY:\n"
        "Word: <headword> /<IPA>/\n"
        "POS: <part of speech>\n"
        "Definition: <short English definition> (<short Russian translation>)\n"
        "Examples:\n"
        f"1) <short English sentence at A2–B1 level>{' (Russian translation)' if include_ru_examples else ''}\n"
        f"2) <short English sentence>{' (Russian translation)' if include_ru_examples else ''}\n"
        f"3) <short English sentence>{' (Russian translation, optional)' if include_ru_examples else ' (optional)'}\n\n"
        "RULES:\n"
        "- Keep total under 120 words; one-line definition.\n"
        "- The definition MUST always include both English and Russian (in parentheses).\n"
        "- Examples are in English; if user's language is Russian, add Russian translations in parentheses.\n"
        "- Use common, school-safe sense of the word.\n"
        "- Do not add commentary or headings beyond the specified format."
    )

    messages = [
        {"role": "system", "content": POLICY},
        {"role": "user", "content": prompt},
    ]

    try:
        text = await ask_openai(messages, max_tokens=350)
        await update.message.reply_text(trim(text))
    except Exception as e:
        await update.message.reply_text(f"⚠️ Vocab error: {e}")

async def quiz_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prefs = get_prefs(update.effective_user.id)
    prefs["mode"] = "quiz"
    prefs["dialogue_turns"] = 0
    lang = prefs.get("lang", "auto")
    if lang == "auto":
        lang = detect_lang(update.message.text or "")

    if lang == "ru":
        msg = "Режим викторины включён. Отправь тему (например, pollution), и я буду задавать вопросы по одному."
    else:
        msg = "Quiz mode is ON. Send me a topic (e.g., pollution). I’ll ask one question at a time."
    await update.message.reply_text(msg)

# === TALK MODE COMMANDS ===
async def talk_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Bắt đầu chế độ hội thoại học tập."""
    prefs = get_prefs(update.effective_user.id)
    prefs["mode"] = "dialogue"
    prefs["dialogue_turns"] = 0

    if context.args and context.args[0].isdigit():
        prefs["dialogue_limit"] = max(4, min(int(context.args[0]), 40))
    else:
        prefs["dialogue_limit"] = DEFAULT_DIALOGUE_LIMIT

    lang = prefs.get("lang", "auto")
    if lang == "auto":
        lang = detect_lang(update.message.text or "")
    if lang == "ru":
        opener = f"Привет! Давай немного поболтаем на английском. Как ты сегодня? (≈{prefs['dialogue_limit']} реплик)"
    else:
        opener = f"Hi! Let’s have a short English chat. How are you today? (≈{prefs['dialogue_limit']} turns)"
    await update.message.reply_text(opener)

async def endtalk_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Thoát chế độ hội thoại."""
    prefs = get_prefs(update.effective_user.id)
    prefs["mode"] = "vocab"
    prefs.pop("dialogue_turns", None)
    await update.message.reply_text("Dialogue ended. Back to study mode (vocab).")

# ========== FREE CHAT ==========
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text or ""
    if blocked(user_message):
        return await update.message.reply_text(
            "⛔ That's outside our classroom scope. "
            "Try vocabulary, reading, grammar, or a quiz topic."
        )

    # Chào tự nhiên nếu người dùng gõ "hi/hello/привет..."
    if GREET_RE.match(user_message.strip()):
        prefs = get_prefs(update.effective_user.id)
        lang = prefs.get("lang","auto")
        if lang == "auto":
            lang = detect_lang(user_message)
        if lang == "ru":
            return await update.message.reply_text("Привет! Как ты сегодня? Чем помочь?")
        else:
            return await update.message.reply_text("Hi! How are you today? How can I help?")

    await update.message.reply_text("Thinking…")

    prefs = get_prefs(update.effective_user.id)
    lang = prefs["lang"]
    if lang == "auto":
        lang = detect_lang(user_message)

    # --- QUIZ: sinh đề khi người dùng gửi topic ---
    if prefs["mode"] == "quiz" and "quiz" not in context.user_data:
        topic = user_message.strip() or "school life"
        level = prefs["cefr"]
        lang_ui = lang  # 'en' or 'ru'

        prompt_user = (
            f"Create a 5-question multiple-choice quiz (4 options each) on '{topic}', "
            f"level {level}, for grades 6–9.\n"
            "Return STRICT JSON only, no prose, no markdown.\n"
            "{ \"questions\": ["
            "{\"id\":1,\"question\":\"...\",\"options\":[\"...\",\"...\",\"...\",\"...\"],"
            "\"correct\":\"A\",\"explain_en\":\"<=25 words\",\"explain_ru\":\"<=25 words\"},"
            "{\"id\":2,...},{\"id\":3,...},{\"id\":4,...},{\"id\":5,...}"
            "]}\n"
            f"Language for 'question' and 'options': "
            f"{'Russian' if lang_ui=='ru' else 'English'} at A2–B1 simplicity.\n"
            "Keep content school-safe."
        )
        messages = [
            {"role": "system", "content": POLICY},
            {"role": "user", "content": prompt_user},
        ]
        raw = await ask_openai(messages, max_tokens=800)

        def extract_json(s: str):
            s = s.strip()
            if "```" in s:
                parts = s.split("```")
                for i in range(len(parts)-1):
                    block = parts[i+1]
                    if block.lstrip().startswith("json"):
                        return json.loads(block.split("\n",1)[1])
                    try:
                        return json.loads(block)
                    except Exception:
                        continue
            return json.loads(s)

        try:
            data = extract_json(raw)
        except Exception:
            return await update.message.reply_text("Sorry, the quiz format failed. Please try again.")

        qs = data.get("questions", [])
        if not qs:
            return await update.message.reply_text("No questions generated. Try another topic.")

        context.user_data["quiz"] = {
            "topic": topic, "level": level, "lang": lang_ui,
            "questions": qs, "i": 0, "tries_left": 2
        }
        await send_quiz_question(context, update.effective_chat.id)
        return

    # (các mode khác): free chat/reading/grammar/dialogue
    history = context.user_data.get("history", [])
    history.append({"role": "user", "content": user_message})
    history = history[-MAX_HISTORY:]
    context.user_data["history"] = history

    mode_instruction = {
        "vocab":   "Behave as VOCAB helper: IPA, brief meaning, and 2–3 short examples.",
        "reading": "Provide a short reading (80–120 words) + comprehension questions.",
        "grammar": "Explain a grammar point in 3–5 short bullets + examples.",
        "quiz":    "Create a 5-question quiz (4 options each) with answers.",
        "dialogue": (
            "You are a friendly English conversation tutor for grades 6–9. "
            "Engage in short, safe, simple dialogues (A2–B1). "
            "Allowed topics: greetings, school, hobbies, weather, family, daily life. "
            "Each reply should be 1–3 sentences. "
            "If the student goes off-topic, politely redirect to learning. "
            "Keep tone positive and age-appropriate."
        ),
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

        # === DIALOGUE TURN COUNTING ===
        prefs = get_prefs(update.effective_user.id)
        if prefs["mode"] == "dialogue":
            prefs["dialogue_turns"] = prefs.get("dialogue_turns", 0) + 1
            limit = prefs.get("dialogue_limit", DEFAULT_DIALOGUE_LIMIT)

            if prefs["dialogue_turns"] >= limit:
                lang2 = prefs.get("lang", "auto")
                if lang2 == "auto":
                    lang2 = detect_lang(user_message)
                if lang2 == "ru":
                    msg = ("Отличная беседа! Хочешь немного потренироваться? "
                           "Попробуй /vocab <слово> или /quiz по теме нашей беседы. "
                           "Если хочешь продолжить — используй /talk <число> чтобы задать новый лимит.")
                else:
                    msg = ("Great chat! Want to learn a bit more? "
                           "Try /vocab <word> or /quiz about our topic. "
                           "If you'd like to keep talking, use /talk <number> to set a new limit.")
                await update.message.reply_text(msg)
                prefs["dialogue_turns"] = 0

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
    application.add_handler(CommandHandler("talk", talk_cmd))
    application.add_handler(CommandHandler("endtalk", endtalk_cmd))

    # CallbackQuery (inline buttons)
    application.add_handler(CallbackQueryHandler(on_quiz_answer, pattern=f"^{CBQ_QUIZ_PREFIX}"))
    application.add_handler(CallbackQueryHandler(on_help_button, pattern=f"^{CBQ_HELP_PREFIX}"))

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
