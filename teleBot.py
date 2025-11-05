# teleBot.py
# =========================================================
# 0) IMPORTS & GLOBAL SETUP
# =========================================================
import os, re, json, time, hmac, hashlib, logging, threading, asyncio, uuid
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
    logger.exception("Exception while handling an update:", exc_info=context.error)
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
if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN missing")

USE_OPENROUTER = os.getenv("USE_OPENROUTER", "True").lower() == "true"
OR_KEY = os.getenv("OPENROUTER_API_KEY")
OA_KEY = os.getenv("OPENAI_API_KEY")

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

# =========================================================
# 3) CONSTANTS, HELPERS, POLICIES
# =========================================================
DEFAULT_LANG = "auto"   # auto|en|ru
MAX_HISTORY = 10
ALLOWED_MODES = {"chat", "vocab", "reading", "grammar", "talk"}
BANNED_KEYWORDS = [
    r"\bsex\b", r"\bporn\b", r"\berotic\b",
    r"\bviolence\b", r"\bsuicide\b", r"\bself[- ]?harm\b",
    r"\bdrugs?\b", r"\balcohol\b", r"\bgamble\b",
    r"\bextremis(m|t)\b"
]
GRADE_TO_CEFR = {"6": "A2", "7": "A2+", "8": "B1-", "9": "B1"}
DEFAULT_DIALOGUE_LIMIT = 20  # Talk limit

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

def trim(s: str, max_chars: int = 1000) -> str:
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

def make_user_hash(user_id: object, salt: str) -> str:
    try:
        raw = (str(user_id) + "|" + (salt or "")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:12]
    except Exception:
        return "unknown"

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
# 7) INTENT LAYER (SMART OVERRIDES)
# =========================================================
INTENT_PATTERNS = {
    "translate": re.compile(r"\btranslate\b|\bdịch\b|\bпереведи\b", re.I),
    "more_examples": re.compile(r"\bmore examples\b|\bví dụ thêm\b|\bещё примеры\b", re.I),
    "new_questions": re.compile(r"\banother (quiz|questions?)\b|\bcâu hỏi khác\b|\bещё вопросы\b", re.I),
    "summary_request": re.compile(r"\bhow were my answers\b|\btổng kết\b|\bитоги\b", re.I),
    "define_word": re.compile(r"^define\s+\w+|^what does .* mean\??", re.I),
}

def detect_intent(text: str):
    t = (text or "").strip()
    for k, rx in INTENT_PATTERNS.items():
        if rx.search(t):
            return k
    if re.search(r"\bexample(s)?\b", t, re.I) and len(t.split()) <= 6:
        return "more_examples"
    if re.search(r"\b(correct|check)\b.*\banswer(s)?\b", t, re.I):
        return "summary_request"
    if re.search(r"\btranslate\b|\bdịch\b|\bперевод\b", t, re.I):
        return "translate"
    return None

_TRANSLATE_HINTS = {
    r"\bsang tiếng nga\b": "ru", r"\btiếng nga\b": "ru",
    r"\bsang tiếng anh\b": "en", r"\btiếng anh\b": "en",
    r"\bto russian\b": "ru", r"\binto russian\b": "ru",
    r"\bto english\b": "en", r"\binto english\b": "en",
    r"\bна русский\b": "ru", r"\bна английский\b": "en",
}

def _guess_translate_target(text: str, ui_lang: str) -> str:
    t = text.lower()
    for pat, tgt in _TRANSLATE_HINTS.items():
        if re.search(pat, t):
            return tgt
    return "en" if ui_lang == "ru" else "ru"

def _extract_translate_content(text: str) -> str:
    t = text.strip()
    m = re.search(r"(?::|–|-)\s*(.+)$", t)
    if m and len(m.group(1).strip()) >= 2:
        return m.group(1).strip()
    m = re.match(r"^(translate|dịch|переведи)\s+(.*)$", t, flags=re.I)
    if m and len(m.group(2).strip()) >= 2:
        return m.group(2).strip()
    return t

# =========================================================
# 8) UI (INLINE MENUS)
# =========================================================
def root_menu(lang: str) -> InlineKeyboardMarkup:
    # Practice mode has been removed; practice is inline per mode
    if lang == "ru":
        kb = [
            [InlineKeyboardButton("📚 Слова", callback_data="menu:mode:vocab"),
             InlineKeyboardButton("📖 Чтение", callback_data="menu:mode:reading")],
            [InlineKeyboardButton("⚙️ Грамматика", callback_data="menu:mode:grammar")],
            [InlineKeyboardButton("💬 Разговор", callback_data="menu:mode:talk")],
            [InlineKeyboardButton("🏫 Класс", callback_data="menu:grade"),
             InlineKeyboardButton("🌐 Язык", callback_data="menu:lang")],
            [InlineKeyboardButton("📋 Меню", callback_data="menu:root")]
        ]
    else:
        kb = [
            [InlineKeyboardButton("📚 Vocabulary", callback_data="menu:mode:vocab"),
             InlineKeyboardButton("📖 Reading", callback_data="menu:mode:reading")],
            [InlineKeyboardButton("⚙️ Grammar", callback_data="menu:mode:grammar")],
            [InlineKeyboardButton("💬 Talk", callback_data="menu:mode:talk")],
            [InlineKeyboardButton("🏫 Grade", callback_data="menu:grade"),
             InlineKeyboardButton("🌐 Language", callback_data="menu:lang")],
            [InlineKeyboardButton("📋 Back to Menu", callback_data="menu:root")]
        ]
    return InlineKeyboardMarkup(kb)

def reading_entry_menu(lang="en") -> InlineKeyboardMarkup:
    t1 = "Choose input:" if lang != "ru" else "Выбери источник:"
    # We'll send t1 as text; buttons are:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📌 Topic", callback_data="reading:input:topic"),
         InlineKeyboardButton("📝 My text", callback_data="reading:input:mytext")],
        [InlineKeyboardButton("⬅️ Back", callback_data="menu:root")]
    ])

def reading_post_menu(lang="en") -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔤 Translate (gloss)", callback_data="reading:gloss"),
         InlineKeyboardButton("📝 Practice from this text", callback_data="reading:menu")],
        [InlineKeyboardButton("🔁 Another text", callback_data="reading:another")],
        [InlineKeyboardButton("⬅️ Back", callback_data="menu:root")]
    ])

def reading_practice_menu(lang="en") -> InlineKeyboardMarkup:
    # main idea, detail, cloze, vocab
    if lang == "ru":
        labels = ["Главная идея", "Подробности (тест)", "Пропуски", "Слова в тексте", "Смешанное"]
    else:
        labels = ["Main idea", "Detail MCQ", "Cloze", "Vocab-in-context", "Mix"]
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(labels[0], callback_data="reading:do:main"),
         InlineKeyboardButton(labels[1], callback_data="reading:do:detail")],
        [InlineKeyboardButton(labels[2], callback_data="reading:do:cloze"),
         InlineKeyboardButton(labels[3], callback_data="reading:do:vocab")],
        [InlineKeyboardButton(labels[4], callback_data="reading:do:mix")],
        [InlineKeyboardButton("⬅️ Back", callback_data="menu:root")]
    ])

def vocab_post_menu(lang="en") -> InlineKeyboardMarkup:
    if lang == "ru":
        a, b = "Практика по слову", "Больше примеров"
    else:
        a, b = "Practice this word", "More examples"
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"🧪 {a}", callback_data="vocab:menu"),
         InlineKeyboardButton(f"➕ {b}", callback_data="vocab:more")],
        [InlineKeyboardButton("⬅️ Back", callback_data="menu:root")]
    ])

def vocab_practice_menu(lang="en") -> InlineKeyboardMarkup:
    if lang == "ru":
        labels = ["Синонимы", "Антонимы", "Часть речи", "Пропуски", "Смешанное"]
    else:
        labels = ["Synonyms", "Antonyms", "Part-of-speech", "Cloze", "Mix"]
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(labels[0], callback_data="vocab:do:syn"),
         InlineKeyboardButton(labels[1], callback_data="vocab:do:ant")],
        [InlineKeyboardButton(labels[2], callback_data="vocab:do:pos"),
         InlineKeyboardButton(labels[3], callback_data="vocab:do:cloze")],
        [InlineKeyboardButton(labels[4], callback_data="vocab:do:mix")],
        [InlineKeyboardButton("⬅️ Back", callback_data="menu:root")]
    ])

def grammar_post_menu(lang="en") -> InlineKeyboardMarkup:
    if lang == "ru":
        lab = "Практика по правилу"
    else:
        lab = "Practice this rule"
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"🧪 {lab}", callback_data="grammar:menu")],
        [InlineKeyboardButton("⬅️ Back", callback_data="menu:root")]
    ])

def grammar_practice_menu(lang="en") -> InlineKeyboardMarkup:
    if lang == "ru":
        labels = ["Формы глагола", "Исправь ошибку", "Порядок слов", "Смешанное"]
    else:
        labels = ["Verb forms", "Error fix", "Sentence order", "Mix"]
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(labels[0], callback_data="grammar:do:verb"),
         InlineKeyboardButton(labels[1], callback_data="grammar:do:error")],
        [InlineKeyboardButton(labels[2], callback_data="grammar:do:order")],
        [InlineKeyboardButton(labels[3], callback_data="grammar:do:mix")],
        [InlineKeyboardButton("⬅️ Back", callback_data="menu:root")]
    ])

def mcq_buttons(options):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"A) {options[0]}", callback_data="ans:A"),
         InlineKeyboardButton(f"B) {options[1]}", callback_data="ans:B")],
        [InlineKeyboardButton(f"C) {options[2]}", callback_data="ans:C"),
         InlineKeyboardButton(f"D) {options[3]}", callback_data="ans:D")]
    ])

# =========================================================
# 9) START / HELP COMMANDS
# =========================================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text_in = (update.message.text or "").strip()
    greet = "Hi there! I’m your English study buddy. How can I help you today?"
    if detect_lang(text_in) == "ru":
        greet = "Привет! Я твой помощник по английскому. Чем помочь сегодня?"
    prefs = get_prefs(update.effective_user.id)
    prefs["mode"] = "chat"
    await safe_reply_message(update.message, greet, reply_markup=root_menu(prefs.get("lang", "en")))
    await log_event(context, "start", update.effective_user.id, {"text": text_in[:200]})

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prefs = get_prefs(update.effective_user.id)
    lang = prefs.get("lang", "en")
    msg = "Choose from the menu below." if lang != "ru" else "Выберите пункт меню ниже."
    await safe_reply_message(update.message, msg, reply_markup=root_menu(lang))
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
        "Make a compact vocabulary card. No markdown bold. "
        "Include synonyms and antonyms lists (1–4 items each) if natural. "
        "Definition must be in English with a short Russian translation in parentheses.\n\n"
        f"HEADWORD: {headword}\nTARGET LEVEL: {prefs['cefr']}\n\n"
        "Format exactly:\n"
        "Word: <headword> /<IPA>/\n"
        "POS: <part of speech>\n"
        "Definition: <short English definition> (<short Russian translation>)\n"
        "Synonyms: x, y, z (if any)\n"
        "Antonyms: a, b (if any)\n"
        "Examples:\n"
        f"1) <short English example>{' (Russian translation)' if include_ru_examples else ''}\n"
        f"2) <short English example>{' (Russian translation)' if include_ru_examples else ''}\n"
        f"3) <short English example>{' (optional Russian translation)' if include_ru_examples else ' (optional)'}\n"
        "Keep under 140 words."
    )
    msgs = [{"role": "system", "content": POLICY_STUDY},
            {"role": "user", "content": prompt}]
    return await ask_openai(msgs, max_tokens=360)

async def build_reading_passage(topic: str, level: str, ui_lang: str):
    prompt = (
        f"Write a short reading passage (80–120 words) about '{topic}', level {level}, grades 6–9. "
        f"Language: {'Russian' if ui_lang=='ru' else 'English'} (A2–B1). School-safe. No bold."
    )
    msgs = [{"role": "system", "content": POLICY_STUDY},
            {"role": "user", "content": prompt}]
    return await ask_openai(msgs, max_tokens=220)

async def build_reading_gloss(text: str, ui_lang: str):
    # produce glossed English text: underline (simulate with _..._) a limited set of phrases and add short RU meaning in parentheses
    target = "Russian" if ui_lang != "ru" else "English"
    prompt = (
        "Gloss the given English reading for a middle-school learner (A2–B1):\n"
        "- Keep original English sentences.\n"
        "- Select 10–15 useful words/phrases (including phrasal verbs/idioms as chunks) and mark them as _like this_.\n"
        f"- Immediately after each underlined chunk, add a {target} hint in parentheses, 1–3 words.\n"
        "- Do NOT translate everything. Do NOT use markdown bold."
        "\n\nTEXT:\n" + text
    )
    msgs = [{"role": "system", "content": POLICY_STUDY},
            {"role": "user", "content": prompt}]
    return await ask_openai(msgs, max_tokens=380)

# Generic practice builders powered by LLM
def normalize_answer(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\s'-]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

async def build_mcq(topic_or_text: str, ui_lang: str, level: str, flavor: str = "generic"):
    """
    flavor: generic | reading_main | reading_detail | reading_vocab | reading_cloze
            | vocab_syn | vocab_ant | vocab_pos | vocab_cloze
            | grammar_verb | grammar_error | grammar_order
    """
    # Set base instruction per flavor
    task = "A general topic quiz."
    if flavor == "reading_main":
        task = "Based on the passage, write 3 MCQs on the main idea (A–D)."
    elif flavor == "reading_detail":
        task = "Based on the passage, write 3 detail MCQs (A–D), avoid trivial facts."
    elif flavor == "reading_vocab":
        task = "From the passage, write 3 vocab-in-context MCQs (ask meaning/closest synonym)."
    elif flavor == "reading_cloze":
        task = "From the passage, write 3 cloze MCQs: remove a word/phrase, 4 options."
    elif flavor == "vocab_syn":
        task = "For the headword, write 3 synonym-choice MCQs (A–D)."
    elif flavor == "vocab_ant":
        task = "For the headword, write 3 antonym-choice MCQs (A–D)."
    elif flavor == "vocab_pos":
        task = "For the headword, write 3 part-of-speech MCQs (A–D)."
    elif flavor == "vocab_cloze":
        task = "For the headword, write 3 cloze MCQs where the headword or derivative fits best."
    elif flavor == "grammar_verb":
        task = "Write 5 MCQs on correct verb forms for the given rule/context (A–D)."
    elif flavor == "grammar_error":
        task = "Write 5 MCQs choosing the corrected sentence (A–D) for the given rule."
    elif flavor == "grammar_order":
        task = "Write 5 MCQs selecting correct word order (A–D) for the given rule."

    base = (
        f"{task}\n"
        "Return STRICT JSON only:\n"
        "{ \"questions\": [\n"
        "{\"id\":1,\"question\":\"...\",\"options\":[\"...\",\"...\",\"...\",\"...\"],"
        "\"answer\":\"A\",\"explain_en\":\"<=25 words\",\"explain_ru\":\"<=25 words\"},\n"
        "{\"id\":2,...},{\"id\":3,...}\n"
        "]}\n"
        f"Language for 'question' and 'options': {'Russian' if ui_lang=='ru' else 'English'} (A2–B1). "
        "School-safe; do not leak answer hints."
    )
    if flavor.startswith("grammar_"):
        count_hint = "Create 5 items."  # more practice for grammar
        base = base.replace("\"{\\\"id\\\":3,...}\\n\"]", "")  # keep spec clean (not necessary strictly)
    else:
        count_hint = "Create 3 items."

    # Supply text or headword/topic depending on flavor
    if flavor.startswith("reading_"):
        user_payload = f"PASSAGE:\n{topic_or_text}\n\nLevel: {level}. {count_hint}"
    elif flavor.startswith("vocab_"):
        user_payload = f"HEADWORD: {topic_or_text}\nLevel: {level}. {count_hint}"
    elif flavor.startswith("grammar_"):
        user_payload = f"RULE: {topic_or_text}\nLevel: {level}. {count_hint}"
    else:
        user_payload = f"TOPIC: {topic_or_text}\nLevel: {level}. {count_hint}"

    msgs = [{"role": "system", "content": POLICY_STUDY},
            {"role": "user", "content": base + "\n\n" + user_payload}]
    raw = await ask_openai(msgs, max_tokens=900)
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

# =========================================================
# 11) PRACTICE ENGINE (shared)
# =========================================================
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
            await safe_reply_message(update_or_query.message, text, reply_markup=kb)
        else:
            await safe_edit_text(update_or_query, text, reply_markup=kb)
    else:
        q = st["items"][idx]
        head = "Type your answer:" if st.get("ui_lang","en") != "ru" else "Напиши ответ:"
        text = f"{title}\n\n{q['prompt']}\n\n{head}"
        if isinstance(update_or_query, Update):
            await safe_reply_message(update_or_query.message, text)
        else:
            await safe_edit_text(update_or_query, text)

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
    for it in st["items"]:
        expl = it["explain_ru"] if lang == "ru" and it["explain_ru"] else it["explain_en"]
        ans = it.get("answer") or ""
        lines.append(f"Q{it['id']}: {ans} — {expl}")
    await safe_reply_message(update.message, "\n".join(lines))
    await log_event(context, "practice_done", update.effective_user.id, {
        "type": st["type"], "topic": st.get("topic"), "score": score, "total": total
    })
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
async def vocab_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prefs = get_prefs(update.effective_user.id)
    prefs["mode"] = "vocab"
    await safe_reply_message(update.message, "Vocabulary mode is ON. Send me a word.")
    await log_event(context, "mode_set", update.effective_user.id, {"mode": "vocab"})

async def logtest_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ok = {"ping": "pong", "note": "manual test"}
    await log_event(context, "logtest", update.effective_user.id, ok)
    await safe_reply_message(update.message, "Logtest sent (if GSHEET_WEBHOOK is set).")

# =========================================================
# 14) CALLBACK HANDLER (INLINE BUTTONS)
# =========================================================
async def on_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    data = q.data or ""
    await q.answer()

    uid = update.effective_user.id
    prefs = get_prefs(uid)
    ui_lang = prefs.get("lang", "en")
    if ui_lang == "auto":
        ui_lang = "ru" if CYRILLIC_RE.search(q.message.text or "") else "en"

    # ----- Common menus -----
    if data == "menu:root":
        msg = "Back to menu." if ui_lang != "ru" else "Возврат в меню."
        await safe_edit_text(q, msg, reply_markup=root_menu(ui_lang))
        prefs["mode"] = "chat"
        await log_event(context, "menu_root", uid, {})
        return

    if data == "menu:lang":
        await safe_edit_text(q, "Choose language:" if ui_lang != "ru" else "Выберите язык:",
                             reply_markup=InlineKeyboardMarkup([
                                 [InlineKeyboardButton("English", callback_data="set_lang:en"),
                                  InlineKeyboardButton("Русский", callback_data="set_lang:ru")],
                                 [InlineKeyboardButton("⬅️ Back", callback_data="menu:root")]
                             ]))
        return

    if data == "menu:grade":
        await safe_edit_text(q, "Choose grade:" if ui_lang != "ru" else "Выберите класс:",
                             reply_markup=InlineKeyboardMarkup([
                                 [InlineKeyboardButton("6", callback_data="set_grade:6"),
                                  InlineKeyboardButton("7", callback_data="set_grade:7"),
                                  InlineKeyboardButton("8", callback_data="set_grade:8"),
                                  InlineKeyboardButton("9", callback_data="set_grade:9")],
                                 [InlineKeyboardButton("⬅️ Back", callback_data="menu:root")]
                             ]))
        return

    if data.startswith("set_lang:"):
        lang = data.split(":", 1)[1]
        prefs["lang"] = lang
        txt = f"Language set to {lang.upper()}." if lang!="ru" else "Язык: Русский."
        await safe_edit_text(q, txt, reply_markup=root_menu(lang))
        await log_event(context, "lang_set", uid, {"lang": lang})
        return

    if data.startswith("set_grade:"):
        g = data.split(":", 1)[1]
        if g in GRADE_TO_CEFR:
            prefs["grade"] = g
            prefs["cefr"] = GRADE_TO_CEFR[g]
            txt = (f"Grade set to {g}. Target level: {prefs['cefr']}."
                   if ui_lang != "ru" else f"Класс: {g}. Уровень: {prefs['cefr']}.")
            await safe_edit_text(q, txt, reply_markup=root_menu(ui_lang))
            await log_event(context, "grade_set", uid, {"grade": g, "cefr": prefs["cefr"]})
        else:
            await safe_edit_text(q, "Invalid grade.", reply_markup=root_menu(ui_lang))
        return

    # ----- Mode entries -----
    if data.startswith("menu:mode:"):
        mode = data.split(":")[-1]  # vocab/reading/grammar/talk
        prefs["mode"] = mode
        await log_event(context, "mode_set", uid, {"mode": mode})
        if mode == "vocab":
            txt = "Vocabulary mode is ON. Send a word." if ui_lang != "ru" else "Режим Слова. Отправь слово."
            await safe_edit_text(q, txt, reply_markup=root_menu(ui_lang))
        elif mode == "reading":
            txt = "Reading mode is ON. Choose: Topic or My text." if ui_lang != "ru" else "Режим Чтение. Выбери: Тема или Мой текст."
            await safe_edit_text(q, txt, reply_markup=reading_entry_menu(ui_lang))
            context.user_data["reading_input"] = None
        elif mode == "grammar":
            txt = "Grammar mode is ON. Send a grammar point (e.g., Present Simple)." if ui_lang != "ru" else "Режим Грамматика. Отправь тему (напр., Present Simple)."
            await safe_edit_text(q, txt, reply_markup=root_menu(ui_lang))
        elif mode == "talk":
            txt = "Choose a topic to talk about:" if ui_lang != "ru" else "Выберите тему для разговора:"
            # Reuse old topics
            await safe_edit_text(q, txt, reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("Daily life", callback_data="talk:topic:daily"),
                 InlineKeyboardButton("School life", callback_data="talk:topic:school")],
                [InlineKeyboardButton("Hobbies", callback_data="talk:topic:hobbies"),
                 InlineKeyboardButton("Environment", callback_data="talk:topic:env")],
                [InlineKeyboardButton("Holidays", callback_data="talk:topic:holidays"),
                 InlineKeyboardButton("Family", callback_data="talk:topic:family")],
                [InlineKeyboardButton("⬅️ Back", callback_data="menu:root")]
            ]))
        return

    # ----- Reading flow: choose input -----
    if data.startswith("reading:input:"):
        kind = data.split(":")[-1]  # topic | mytext
        context.user_data["reading_input"] = kind
        if kind == "topic":
            ask = "Send me a topic (e.g., animals)." if ui_lang != "ru" else "Отправь тему (например, animals)."
        else:
            ask = "Paste your text (80–150 words)." if ui_lang != "ru" else "Вставь свой текст (80–150 слов)."
        await safe_edit_text(q, ask)
        return

    # ----- Reading: gloss / menu / another -----
    if data == "reading:gloss":
        rp = context.user_data.get("reading", {})
        passage = rp.get("last_passage", "")
        if not passage:
            txt = "No passage yet." if ui_lang != "ru" else "Пока нет текста."
            return await safe_edit_text(q, txt)
        glossed = await build_reading_gloss(passage, ui_lang)
        await safe_edit_text(q, trim(glossed), reply_markup=reading_post_menu(ui_lang))
        await log_event(context, "reading_gloss", uid, {"chars": len(passage)})
        return

    if data == "reading:menu":
        await safe_edit_text(q, "Choose a practice type:" if ui_lang!="ru" else "Выбери тип практики:",
                             reply_markup=reading_practice_menu(ui_lang))
        return

    if data == "reading:another":
        rp = context.user_data.get("reading", {})
        topic = rp.get("topic") or "school life"
        level = get_prefs(uid)["cefr"]
        passage = await build_reading_passage(topic, level, ui_lang)
        context.user_data["reading"] = {"topic": topic, "last_passage": passage}
        remember_last_text(context, passage)
        await safe_edit_text(q, trim(passage), reply_markup=reading_post_menu(ui_lang))
        await log_event(context, "reading_passage", uid, {"topic": topic, "another": True})
        return

    # ----- Reading practice do: xxx -----
    if data.startswith("reading:do:"):
        typ = data.split(":")[-1]  # main/detail/cloze/vocab/mix
        rp = context.user_data.get("reading", {})
        passage = rp.get("last_passage", "")
        topic = rp.get("topic", "reading")
        if not passage:
            return await safe_edit_text(q, "No passage yet." if ui_lang!="ru" else "Пока нет текста.")
        flavor_map = {
            "main": "reading_main",
            "detail": "reading_detail",
            "cloze": "reading_cloze",
            "vocab": "reading_vocab",
            "mix": "reading_detail"  # simple choice; could randomize multiple
        }
        flavor = flavor_map.get(typ, "reading_detail")
        items = await build_mcq(passage, ui_lang, get_prefs(uid)["cefr"], flavor=flavor)
        if not items:
            return await safe_edit_text(q, "Failed to build practice." if ui_lang!="ru" else "Не удалось создать задания.")
        context.user_data["practice"] = {
            "type": "mcq", "topic": topic, "items": items,
            "idx": 0, "attempts": 0, "score": 0, "ui_lang": ui_lang
        }
        await log_event(context, "practice_built", uid, {"ptype": flavor, "topic": topic, "count": len(items)})
        return await send_practice_item(q, context)

    # ----- Vocab menus -----
    if data == "vocab:menu":
        await safe_edit_text(q, "Choose a practice type:" if ui_lang!="ru" else "Выбери тип практики:",
                             reply_markup=vocab_practice_menu(ui_lang))
        return

    if data == "vocab:more":
        head = context.user_data.get("last_word", "")
        if not head:
            return await safe_edit_text(q, "No word yet." if ui_lang!="ru" else "Слова пока нет.")
        p = (
            f"Give 3 short extra examples for the word '{head}' (A2–B1). "
            f"Language: {'Russian' if ui_lang=='ru' else 'English'}. Keep compact. No bold."
        )
        out = await ask_openai(
            [{"role": "system", "content": POLICY_STUDY},
             {"role": "user", "content": p}],
            max_tokens=160
        )
        await safe_edit_text(q, trim(out), reply_markup=vocab_post_menu(ui_lang))
        await log_event(context, "vocab_more_examples", uid, {"word": head})
        return

    if data.startswith("vocab:do:"):
        kind = data.split(":")[-1]  # syn/ant/pos/cloze/mix
        head = context.user_data.get("last_word", "")
        if not head:
            return await safe_edit_text(q, "No word yet." if ui_lang!="ru" else "Слова пока нет.")
        flavor_map = {
            "syn": "vocab_syn",
            "ant": "vocab_ant",
            "pos": "vocab_pos",
            "cloze": "vocab_cloze",
            "mix": "vocab_syn",
        }
        flavor = flavor_map.get(kind, "vocab_syn")
        items = await build_mcq(head, ui_lang, get_prefs(uid)["cefr"], flavor=flavor)
        if not items:
            return await safe_edit_text(q, "Failed to build practice." if ui_lang!="ru" else "Не удалось создать задания.")
        context.user_data["practice"] = {
            "type": "mcq", "topic": head, "items": items,
            "idx": 0, "attempts": 0, "score": 0, "ui_lang": ui_lang
        }
        await log_event(context, "practice_built", uid, {"ptype": flavor, "word": head, "count": len(items)})
        return await send_practice_item(q, context)

    # ----- Grammar menus -----
    if data == "grammar:menu":
        await safe_edit_text(q, "Choose a practice type:" if ui_lang!="ru" else "Выбери тип практики:",
                             reply_markup=grammar_practice_menu(ui_lang))
        return

    if data.startswith("grammar:do:"):
        kind = data.split(":")[-1]  # verb/error/order/mix
        rule = context.user_data.get("last_grammar_topic") or "Present Simple"
        flavor_map = {
            "verb": "grammar_verb",
            "error": "grammar_error",
            "order": "grammar_order",
            "mix": "grammar_verb",
        }
        flavor = flavor_map.get(kind, "grammar_verb")
        items = await build_mcq(rule, ui_lang, get_prefs(uid)["cefr"], flavor=flavor)
        if not items:
            return await safe_edit_text(q, "Failed to build practice." if ui_lang!="ru" else "Не удалось создать задания.")
        context.user_data["practice"] = {
            "type": "mcq", "topic": rule, "items": items,
            "idx": 0, "attempts": 0, "score": 0, "ui_lang": ui_lang
        }
        await log_event(context, "practice_built", uid, {"ptype": flavor, "rule": rule, "count": len(items)})
        return await send_practice_item(q, context)

    # ----- Talk topics (unchanged) -----
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
        await safe_edit_text(q, f"Topic: {topic}\n\n{opener}", reply_markup=root_menu(ui_lang))
        await log_event(context, "talk_topic_set", uid, {"topic": topic})
        return

    # ----- Practice answers (MCQ) -----
    if data.startswith("ans:"):
        st = context.user_data.get("practice")
        if not st or st.get("type") != "mcq":
            await safe_edit_text(q, "No active multiple-choice exercise.", reply_markup=root_menu(ui_lang))
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
            await safe_edit_text(q, f"{ok}\n{expl}".strip())
            await log_event(context, "practice_answer", uid, {"ptype": "mcq", "qid": qitem.get("id"), "correct": True})
            st["idx"] += 1
        else:
            st["attempts"] += 1
            if st["attempts"] < 2:
                msg = "Not quite. Try again." if ui_lang != "ru" else "Почти. Попробуй еще раз."
                await safe_edit_text(q, msg)
                await log_event(context, "practice_answer", uid, {"ptype": "mcq", "qid": qitem.get("id"), "correct": False, "retry": True})
            else:
                st["attempts"] = 0
                ans = f"The correct answer is {correct}." if ui_lang != "ru" else f"Правильный ответ: {correct}."
                expl = qitem["explain_ru"] if ui_lang == "ru" and qitem["explain_ru"] else qitem["explain_en"]
                await safe_edit_text(q, f"{ans}\n{expl}".strip())
                await log_event(context, "practice_answer", uid, {"ptype": "mcq", "qid": qitem.get("id"), "correct": False, "revealed": True})
                st["idx"] += 1

        if st["idx"] >= len(st["items"]):
            dummy_update = Update(update.update_id, message=q.message)
            await practice_summary(dummy_update, context)
        else:
            await send_practice_item(q, context)
        return

# =========================================================
# 15) FREE TEXT HANDLER (INTENT-FIRST ⇒ MODE FALLBACK)
# =========================================================
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text or ""

    # remember last_text for translate
    if user_message and len(user_message) >= 8:
        remember_last_text(context, user_message)

    if blocked(user_message):
        return await safe_reply_message(
            update.message,
            "⛔ That's outside our classroom scope. Please try vocabulary, reading, grammar, or a talk topic."
        )

    uid = update.effective_user.id
    prefs = get_prefs(uid)
    lang = prefs.get("lang", "en")
    if lang == "auto":
        lang = detect_lang(user_message)

    # ---- INTENT OVERRIDES (in any mode) ----
    intent = detect_intent(user_message)

    if intent == "translate":
        only_kw = re.fullmatch(r"\s*(translate|dịch|переведи)\s*\.?", user_message, flags=re.I) is not None
        if only_kw:
            text_for_tr = (context.user_data.get("last_text")
                           or context.user_data.get("reading", {}).get("last_passage", ""))
        else:
            text_for_tr = _extract_translate_content(user_message)
            remember_last_text(context, text_for_tr)

        if not text_for_tr:
            return await safe_reply_message(
                update.message,
                "I can translate, but I need some text. Please paste it or ask me again after sending a passage."
            )

        tgt = _guess_translate_target(text_for_tr, lang)
        target_name = "Russian" if tgt == "ru" else "English"
        prompt = (
            f"Translate this into {target_name} (A2–B1, natural for a middle-schooler). "
            f"If helpful, add one short note:\n\n{text_for_tr}"
        )
        out = await ask_openai(
            [{"role": "system", "content": POLICY_CHAT},
             {"role": "user", "content": prompt}],
            max_tokens=220
        )
        await log_event(context, "chat_message", uid, {"chars": len(user_message)})
        return await safe_reply_message(update.message, trim(out))

    if intent == "define_word":
        m = re.search(r"define\s+(\w+)", user_message, re.I)
        head = m.group(1) if m else user_message
        card = await build_vocab_card(head, prefs, user_message)
        add_vocab_to_bank(context, head)
        context.user_data["last_word"] = head
        await log_event(context, "chat_message", uid, {"chars": len(user_message)})
        return await safe_reply_message(update.message, trim(card), reply_markup=vocab_post_menu(lang))

    if intent == "more_examples":
        topic = context.user_data.get("last_grammar_topic") or context.user_data.get("reading", {}).get("topic") or "daily life"
        p = (
            f"Give 3 short example sentences (A2–B1) about '{topic}'. "
            f"Language: {'Russian' if lang=='ru' else 'English'}. No bold. Keep it compact."
        )
        out = await ask_openai(
            [{"role": "system", "content": POLICY_STUDY},
             {"role": "user", "content": p}],
            max_tokens=150
        )
        await log_event(context, "chat_message", uid, {"chars": len(user_message)})
        return await safe_reply_message(update.message, trim(out))

    if intent == "new_questions":
        # keep legacy quick 3 MCQ on nearest topic (compat)
        topic = context.user_data.get("reading", {}).get("topic") or "school life"
        items = await build_mcq(topic, lang, prefs["cefr"], flavor="generic")
        items = items[:3] if len(items) > 3 else items
        context.user_data["practice"] = {
            "type": "mcq","topic": topic,"items": items,
            "idx": 0,"attempts": 0,"score": 0,"ui_lang": lang
        }
        await log_event(context, "practice_built", uid, {"ptype": "generic", "topic": topic, "count": len(items)})
        return await send_practice_item(update, context)

    if intent == "summary_request":
        st = context.user_data.get("practice")
        if st and st.get("items"):
            await log_event(context, "chat_message", uid, {"chars": len(user_message)})
            return await practice_summary(update, context)
        else:
            msg = "No active exercise yet." if lang != "ru" else "Активных заданий пока нет."
            return await safe_reply_message(update.message, msg)

    # ---- MODE-SPECIFIC ----
    if prefs["mode"] == "vocab":
        word = user_message.strip()
        if not word:
            return await safe_reply_message(update.message, "Send a word to look up." if lang != "ru" else "Отправь слово.")
        try:
            card = await build_vocab_card(word, prefs, update.message.text)
            add_vocab_to_bank(context, word)
            context.user_data["last_word"] = word
            await log_event(context, "chat_message", uid, {"chars": len(user_message)})
            return await safe_reply_message(update.message, trim(card), reply_markup=vocab_post_menu(lang))
        except Exception:
            await log_event(context, "chat_message", uid, {"chars": len(user_message)})
            return await safe_reply_message(update.message, "Failed to build the card. Try another word." if lang != "ru" else "Не удалось создать карточку. Попробуй другое слово.")

    if prefs["mode"] == "reading":
        input_kind = context.user_data.get("reading_input")  # topic | mytext | None
        level = prefs["cefr"]
        if input_kind == "mytext":
            # Treat user message as the passage
            passage = (user_message or "").strip()
            if len(passage) < 40:
                ask = "Please send a longer text (>= 80 words) or choose Topic." if lang!="ru" else "Отправь текст подлиннее (>= 80 слов) или выбери Тему."
                return await safe_reply_message(update.message, ask, reply_markup=reading_entry_menu(lang))
            context.user_data["reading"] = {"topic": "my text", "last_passage": passage}
            remember_last_text(context, passage)
            await safe_reply_message(update.message, trim(passage), reply_markup=reading_post_menu(lang))
            await log_event(context, "reading_passage", uid, {"topic": "my_text"})
            return
        else:
            # default to topic flow
            topic = user_message.strip() or context.user_data.get("last_reading_topic") or "school life"
            passage = await build_reading_passage(topic, level, lang)
            context.user_data["reading"] = {"topic": topic, "last_passage": passage}
            context.user_data["last_reading_topic"] = topic
            remember_last_text(context, passage)
            await safe_reply_message(update.message, trim(passage), reply_markup=reading_post_menu(lang))
            await log_event(context, "reading_passage", uid, {"topic": topic})
            return

    if prefs["mode"] == "grammar":
        text = user_message.strip()
        context.user_data["last_grammar_topic"] = text or context.user_data.get("last_grammar_topic") or "Present Simple"
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
        await log_event(context, "chat_message", uid, {"chars": len(user_message)})
        return await safe_reply_message(update.message, trim(exp), reply_markup=grammar_post_menu(lang))

    if prefs["mode"] == "talk":
        talk_state = context.user_data.get("talk") or {"topic": "daily life", "turns": 0}
        reply = await talk_reply(user_message, talk_state["topic"], lang)
        talk_state["turns"] = talk_state.get("turns", 0) + 1
        context.user_data["talk"] = talk_state
        await log_event(context, "chat_message", uid, {"chars": len(user_message)})
        if talk_state["turns"] >= prefs.get("dialogue_limit", DEFAULT_DIALOGUE_LIMIT):
            wrap = ("Great chat! Want to study something next? Try Vocabulary or Reading from the menu."
                    if lang != "ru" else "Отличная беседа! Хочешь потренироваться дальше? Выбери Слова или Чтение в меню.")
            await safe_reply_message(update.message, trim(reply))
            await safe_reply_message(update.message, wrap, reply_markup=root_menu(lang))
            prefs["mode"] = "chat"
            context.user_data.pop("talk", None)
            return
        return await safe_reply_message(update.message, trim(reply))

    # ---- CHAT MODE (default) ----
    history = context.user_data.get("history", [])
    history.append({"role": "user", "content": user_message})
    history = history[-MAX_HISTORY:]
    context.user_data["history"] = history
    steer = (
        "Be helpful and concise. If the user asks about study tasks, suggest modes: Vocabulary, Reading, Grammar, Talk."
    )
    messages = [
        {"role": "system", "content": POLICY_CHAT},
        {"role": "user", "content": steer},
        *history
    ]
    text_out = await ask_openai(messages, max_tokens=400)
    await safe_reply_message(update.message, trim(text_out))
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
    application.add_handler(CommandHandler("vocab", vocab_cmd))
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
