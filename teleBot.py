# =========================================================
# teleBot_v2_full.py
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

# =========================================================
# 1) LOGGING
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("teleBot_v2")

async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.exception("Error:", exc_info=context.error)
    try:
        uid = getattr(getattr(update, "effective_user", None), "id", "n/a")
        await log_event(context, "error", uid, {"error": str(context.error)})
    except Exception:
        pass

async def on_startup(app: Application):
    await app.bot.delete_webhook(drop_pending_updates=True)
    logger.info("Webhook deleted, bot ready for polling.")


# =========================================================
# 2) ENV & CLIENT SETUP
# =========================================================
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OR_KEY = os.getenv("OPENROUTER_API_KEY")
GSHEET_WEBHOOK = os.getenv("GSHEET_WEBHOOK", "").strip()
LOG_SALT = os.getenv("LOG_SALT", "").strip()

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN missing")

httpx_client = httpx.Client(
    timeout=httpx.Timeout(connect=30.0, read=90.0, write=90.0, pool=90.0)
)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OR_KEY,
    http_client=httpx_client,
    default_headers={
        "HTTP-Referer": "https://t.me/EnglishClassBot",
        "X-Title": "AI English Tutor",
    },
)
MODEL_NAME = "openai/gpt-4o-mini"


# =========================================================
# 3) CONSTANTS, HELPERS, POLICIES
# =========================================================
DEFAULT_LANG = "en"
MAX_HISTORY = 10

BANNED_KEYWORDS = [
    r"\bsex\b", r"\bporn\b", r"\berotic\b",
    r"\bviolence\b", r"\bsuicide\b", r"\bself[- ]?harm\b",
    r"\bdrugs?\b", r"\balcohol\b", r"\bgamble\b",
    r"\bextremis(m|t)\b"
]
GRADE_TO_CEFR = {"6": "A2", "7": "A2+", "8": "B1-", "9": "B1", "10": "B1+"}

POLICY_CHAT = (
    "You are a friendly English-learning tutor for grades 6–9 (CEFR A2–B1). "
    "Answer only about English language learning: vocabulary, grammar, reading, writing, and speaking. "
    "If the user asks about other school subjects (math, physics, chemistry, etc.), politely refuse and guide them to English topics. "
    "Use plain text only (no markdown, no bold). Be concise and positive."
)
POLICY_STUDY = (
    "You are an English teacher for middle-school students (A2–B1). "
    "Use simple English, safe and age-appropriate content. "
    "No markdown or special formatting."
)

CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")

def detect_lang(text: str) -> str:
    return "ru" if CYRILLIC_RE.search(text or "") else "en"

def trim(s: str, max_chars=1200) -> str:
    s = re.sub(r"\n{3,}", "\n\n", (s or "").strip())
    return s if len(s) <= max_chars else (s[:max_chars].rstrip() + "…")

def blocked(text: str) -> bool:
    for pat in BANNED_KEYWORDS:
        if re.search(pat, text or "", flags=re.IGNORECASE):
            return True
    return False


# =========================================================
# 4) STATE & PREFS
# =========================================================
user_prefs = {}

def get_prefs(user_id: int):
    if user_id not in user_prefs:
        user_prefs[user_id] = {
            "lang": DEFAULT_LANG,
            "grade": "7",
            "cefr": "A2+",
            "mode": "chat"
        }
    return user_prefs[user_id]

def make_user_hash(uid, salt):
    try:
        return hashlib.sha256(f"{uid}|{salt}".encode()).hexdigest()[:12]
    except Exception:
        return "anon"


# =========================================================
# 5) LOGGING TO GOOGLE SHEET (ANONYMOUS)
# =========================================================
async def log_event(context, event, user_id, extra=None):
    if not GSHEET_WEBHOOK: return
    try:
        ts = datetime.now(timezone.utc).isoformat()
        anon = make_user_hash(user_id, LOG_SALT)
        payload = {
            "timestamp": ts,
            "user_hash": anon,
            "event": event,
            "extra": extra or {}
        }
        await asyncio.to_thread(
            httpx_client.post, GSHEET_WEBHOOK, json=payload, timeout=10.0
        )
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
            return await message.reply_text(trim(text))
        except Exception as e:
            logger.warning("safe_reply failed: %s", e)

async def safe_edit_text(query, text: str, reply_markup=None):
    try:
        return await query.edit_message_text(text, reply_markup=reply_markup)
    except BadRequest:
        try:
            return await query.edit_message_text(trim(text))
        except Exception as e:
            logger.warning("safe_edit_text failed: %s", e)
# =========================================================
# PATCH 1: UNIVERSAL BACK TO MENU
# =========================================================
async def back_to_menu(update_or_query, context: ContextTypes.DEFAULT_TYPE, lang="en"):
    """Reset session state and return to main menu safely."""
    prefs = get_prefs(update_or_query.effective_user.id if hasattr(update_or_query, "effective_user") else 0)
    prefs["mode"] = "chat"
    context.user_data.pop("reading_input", None)
    context.user_data.pop("practice", None)
    context.user_data.pop("talk", None)
    msg = "Back to main menu." if lang != "ru" else "Возврат в меню."
    try:
        if hasattr(update_or_query, "callback_query"):
            q = update_or_query.callback_query
            await safe_edit_text(q, msg, reply_markup=main_menu(lang))
        else:
            await safe_reply_message(update_or_query.message, msg, reply_markup=main_menu(lang))
    except Exception as e:
        logger.warning(f"back_to_menu failed: {e}")
        try:
            await safe_reply_message(update_or_query.message, msg, reply_markup=main_menu(lang))
        except Exception:
            pass
    await log_event(context, "menu_return", update_or_query.effective_user.id if hasattr(update_or_query, "effective_user") else "n/a", {"lang": lang})



# =========================================================
# 7) UI MENUS & HELP
# =========================================================
# =========================================================
# MAIN MENU (UNIFIED)
# =========================================================
def main_menu(lang="en") -> InlineKeyboardMarkup:
    if lang == "ru":
        kb = [
            [InlineKeyboardButton("💬 Разговор", callback_data="menu:talk"),
             InlineKeyboardButton("📝 Практика", callback_data="menu:practice")],
            [InlineKeyboardButton("🏫 Класс", callback_data="menu:grade"),
             InlineKeyboardButton("🌐 Язык", callback_data="menu:lang")],
            [InlineKeyboardButton("❓ Помощь", callback_data="menu:help")]
        ]
    else:
        kb = [
            [InlineKeyboardButton("💬 Talk", callback_data="menu:talk"),
             InlineKeyboardButton("📝 Practice", callback_data="menu:practice")],
            [InlineKeyboardButton("🏫 Grade", callback_data="menu:grade"),
             InlineKeyboardButton("🌐 Language", callback_data="menu:lang")],
            [InlineKeyboardButton("❓ Help", callback_data="menu:help")]
        ]
    return InlineKeyboardMarkup(kb)


def practice_menu(lang="en") -> InlineKeyboardMarkup:
    if lang == "ru":
        kb = [
            [InlineKeyboardButton("Multiple Choice", callback_data="grammar:type:mcq")],
            [InlineKeyboardButton("Fill in the blanks", callback_data="grammar:type:fill")],
            [InlineKeyboardButton("Verb Form", callback_data="grammar:type:verb")],
            [InlineKeyboardButton("Error Correction", callback_data="grammar:type:error")],
            [InlineKeyboardButton("⬅️ Назад", callback_data="menu:root")]
        ]
    else:
        kb = [
            [InlineKeyboardButton("Multiple Choice", callback_data="grammar:type:mcq")],
            [InlineKeyboardButton("Fill in the blanks", callback_data="grammar:type:fill")],
            [InlineKeyboardButton("Verb Form", callback_data="grammar:type:verb")],
            [InlineKeyboardButton("Error Correction", callback_data="grammar:type:error")],
            [InlineKeyboardButton("⬅️ Back", callback_data="menu:root")]
        ]
    return InlineKeyboardMarkup(kb)

HELP_TEXT_EN = (
    "💡 Prompt Examples:\n\n"
    "🟢 Vocabulary:\n"
    "- Define 'set up' (IPA, part of speech, short definition, RU translation, 3 examples)\n"
    "🟢 Grammar:\n"
    "- Explain 'Present Perfect' with ✓/✗ examples\n"
    "🟢 Reading:\n"
    "- Write a short A2 text about 'friendship'\n"
    "- Translate gloss for this text: <your text>\n"
    "🟢 Talk:\n"
    "- Let's talk about school life\n"
)
HELP_TEXT_RU = (
    "💡 Примеры промптов:\n\n"
    "🟢 Словарь:\n"
    "- Дай определение 'set up' — IPA, часть речи, краткое объяснение, перевод, 3 примера\n"
    "🟢 Грамматика:\n"
    "- Объясни 'Present Perfect' с примерами ✓/✗\n"
    "🟢 Чтение:\n"
    "- Короткий текст уровня A2 на тему 'дружба'\n"
    "- Глоссы для текста: <вставь текст>\n"
    "🟢 Разговор:\n"
    "- Поговорим о школьной жизни\n"
)


# =========================================================
# 8) START / HELP COMMANDS
# =========================================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    greet = "Choose your language / Выберите язык:"
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

    # footer chỉ có nút Back to menu
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("🏠 Back to menu" if lang!="ru" else "🏠 В меню", callback_data="menu:root")]
    ])

    await safe_reply_message(update.message, txt, reply_markup=kb)
    await log_event(context, "help_open", update.effective_user.id, {"lang": lang})

# =========================================================
# 9) ASK OPENAI WRAPPER
# =========================================================
async def ask_openai(messages, max_tokens=450, temperature=0.4):
    for _ in range(2):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME, messages=messages,
                max_tokens=max_tokens, temperature=temperature
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.warning("ask_openai fail: %s", e)
            await asyncio.sleep(0.8)
    return "[Error: model not responding]"


# =========================================================
# 10) CONTENT BUILDERS
# =========================================================

# --- Vocabulary Builder ---
# =========================================================
# BUILD VOCAB CARD (improved)
# =========================================================
async def build_vocab_card(word: str, prefs: dict) -> str:
    """
    Trả về thẻ từ vựng có:
      - Word + IPA
      - POS (nhận biết: phrasal verb, idiom, noun, adj, adv, verb, phrase…)
      - Definition (EN + RU ngắn)
      - 3 ví dụ tăng dần độ khó (không dịch ví dụ)
      - Synonyms / Antonyms nếu có
    """
    lang = prefs.get("lang", "en")
    include_ru = "(short Russian translation)" if lang != "ru" else "(краткий перевод на русский)"
    level = prefs.get("cefr", "B1+")

    prompt = (
        f"You are an English vocabulary teacher for secondary school students (A2–B1 level).\n"
        f"Create a clear vocabulary card for the word or phrase: '{word}'.\n"
        "Identify the correct part of speech precisely — e.g. phrasal verb, idiom, noun, adjective, verb, adverb, phrase, expression, etc.\n"
        "Include:\n"
        "• Word and IPA transcription\n"
        "• Part of speech (use the exact POS label)\n"
        f"• Short English definition {include_ru}\n"
        "• 3 short example sentences (A2–B1, increasing difficulty; no translation)\n"
        "• Synonyms and Antonyms if naturally relevant\n\n"
        "Strictly follow this plain-text format (no markdown, no bold):\n"
        "Word: <word>\n"
        "IPA: /.../\n"
        "Part of speech: ...\n"
        "Definition: ...\n"
        "Synonyms: ...\n"
        "Antonyms: ...\n"
        "Examples:\n"
        "1) ...\n"
        "2) ...\n"
        "3) ...\n"
        f"Keep concise and under 160 words. Target level: {level}."
    )

    msgs = [{"role": "system", "content": POLICY_STUDY},
            {"role": "user", "content": prompt}]
    return await ask_openai(msgs, max_tokens=400)


# --- Grammar Explain Builder ---
async def build_grammar_explain(topic: str, prefs: dict) -> str:
    lang = prefs.get("lang", "en")
    ru_hint = "Add short Russian hints in parentheses." if lang == "ru" else ""
    prompt = (
        f"Explain grammar topic '{topic}' for level {prefs['cefr']} (A2–B1). "
        "Include 5–7 concise bullet points: form, use, and common mistakes. "
        "Add 2–3 example pairs (✓ correct / ✗ wrong) and 3–5 signal words. "
        f"{ru_hint} No markdown, no bold."
    )
    msgs = [{"role": "system", "content": POLICY_STUDY},
            {"role": "user", "content": prompt}]
    return await ask_openai(msgs, max_tokens=400)


# --- Reading Passage Builder ---
async def build_reading_passage(topic: str, prefs: dict) -> str:
    prompt = (
        f"Write a short reading passage (80–120 words) about '{topic}'. "
        f"Level: {prefs['cefr']} (A2–B1). School-safe and positive. "
        "Plain English only. No bold."
    )
    msgs = [{"role": "system", "content": POLICY_STUDY},
            {"role": "user", "content": prompt}]
    return await ask_openai(msgs, max_tokens=260)


# --- Reading Gloss Builder ---
async def build_reading_gloss(text: str, ui_lang: str):
    target = "Russian" if ui_lang != "ru" else "English"
    prompt = (
        "Gloss the given English text for A2–B1+ learners:\n"
        "- Keep the original English sentences.\n"
        "- Select 12–15 useful English words or phrases (phrasal verbs, idioms, collocations).\n"
        "- Enclose each English chunk in angle brackets <like this> and immediately add a short "
        f"{target} translation in parentheses (1–3 words).\n"
        "- Example: She <set up> (организовала) a small company.\n"
        "- Do NOT gloss every word, and do NOT use markdown.\n\n"
        "- If some Russian text is included, ignore it and gloss only English parts.\n"
        "TEXT:\n" + text
    )
    msgs = [{"role": "system", "content": POLICY_STUDY},
            {"role": "user", "content": prompt}]
    return await ask_openai(msgs, max_tokens=420)


# --- Talk Coach Builder ---
async def talk_reply(user_text: str, topic: str, ui_lang: str):
    prompt = (
        f"You are a friendly English conversation coach for middle school students (A2–B1). "
        f"Topic: {topic}. Respond naturally in 1–3 sentences. "
        "Reformulate small mistakes gently. Suggest 1–2 useful phrases if relevant. "
        "Encourage continuing the conversation. Use plain English only. "
        "No markdown, no bold."
    )
    msgs = [{"role": "system", "content": POLICY_STUDY},
            {"role": "user", "content": f"Student says: {user_text}"}]
    return await ask_openai([{"role": "system", "content": prompt}, *msgs], max_tokens=180)
# =========================================================
# 11) PRACTICE ENGINE (MCQ + RETRY + SUMMARY)
# =========================================================
def normalize_answer(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\s'-]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def fuzzy_equal(a: str, b: str, threshold: float = 0.85) -> bool:
    return difflib.SequenceMatcher(a=normalize_answer(a), b=normalize_answer(b)).ratio() >= threshold


async def build_mcq(topic_or_text: str, ui_lang: str, level: str, flavor: str = "generic"):
    """
    Create 5-question MCQ set based on grade and practice type.
    Each flavor has its own prompt. Level (A2–B1) controls difficulty.
    """
    # Map practice flavor -> task description
    task_map = {
        # --- Vocabulary ---
        "vocab_syn": "Write 5 synonym-choice MCQs for the word below. Each question tests a synonym in context.",
        "vocab_ant": "Write 5 antonym-choice MCQs for the word below. Include short example sentences.",
        "vocab_cloze": "Write 5 fill-in-the-blank MCQs where the correct word or its derivative fits best.",
        # --- Grammar ---
        "grammar_rule": "Write 5 mixed grammar MCQs testing the rule below (2 verb-form, 2 error-correction, 1 word-order).",
        "grammar_verb": "Write 5 MCQs where students choose the correct verb form based on the rule.",
        "grammar_error": "Write 5 MCQs where students identify the corrected sentence for the rule.",
        "grammar_order": "Write 5 MCQs selecting correct word order.",
        # --- Reading ---
        "reading_main": "Write 5 MCQs about the main idea of the passage.",
        "reading_detail": "Write 5 MCQs about details in the passage, avoiding trivial facts.",
        "reading_vocab": "Write 5 MCQs about vocabulary meaning in context.",
        "reading_cloze": "Write 5 fill-in-the-blank MCQs based on the passage.",
        # --- Generic ---
        "generic": "Write 5 general English MCQs suitable for school-level learners."
    }

    # Select prompt text by flavor
    task = task_map.get(flavor, task_map["generic"])

    # CEFR explanation tag for the model
    if level in ("A2", "A2+"):
        diff_note = "Use simple sentences and common words. Avoid abstract grammar."
    elif level == "B1":
        diff_note = "Include 1–2 slightly challenging structures or idioms."
    else:
        diff_note = "Use neutral A2–B1 school-level language."

    # Core prompt to model
    prompt = (
        f"{task}\n\n"
        "Return STRICT JSON only in this format:\n"
        "{ \"questions\": ["
        "{\"id\":1,\"question\":\"...\",\"options\":[\"...\",\"...\",\"...\",\"...\"],"
        "\"answer\":\"A\",\"explain_en\":\"<=25 words\",\"explain_ru\":\"<=25 words\"},"
        "{\"id\":2,...},...,{\"id\":5,...}]}\n\n"
        f"LEVEL: {level} | {diff_note}\n"
        f"TOPIC or INPUT:\n{topic_or_text}\n\n"
        f"Language for question and options: {'Russian' if ui_lang=='ru' else 'English'}."
    )

    msgs = [{"role": "system", "content": POLICY_STUDY},
            {"role": "user", "content": prompt}]

    # Request model
    raw = await ask_openai(msgs, max_tokens=950)
    try:
        # Extract JSON portion
        data = json.loads(re.search(r"\{.*\}", raw, re.S).group())
        questions = data.get("questions", [])
    except Exception as e:
        logger.warning(f"MCQ parse fail: {e} | raw={raw}")
        questions = []

    # Validate questions (ensure 4 options)
    valid = []
    for q in questions:
        opts = q.get("options", [])
        if len(opts) != 4:
            continue
        ans = str(q.get("answer", "A")).strip().upper()
        if ans not in ("A", "B", "C", "D"):
            ans = "A"
        valid.append({
            "id": q.get("id", len(valid)+1),
            "question": q.get("question", ""),
            "options": opts,
            "answer": ans,
            "explain_en": q.get("explain_en", ""),
            "explain_ru": q.get("explain_ru", "")
        })

    return valid



async def send_practice_item(update_or_query, context: ContextTypes.DEFAULT_TYPE):
    """Send current question with 4 button options."""
    st = context.user_data.get("practice")
    if not st: return
    idx = st["idx"]
    q = st["items"][idx]
    txt = f"Q{idx+1}/5\n\n{q['question']}"
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
    # Footer theo nguồn
    scope = st.get("scope", "free")
    await safe_reply_message(update.message, "—", reply_markup=practice_footer_kb(scope, lang))
    # cho phép học sinh quay lại menu trực tiếp
    await safe_reply_message(update.message, "🏠 Back to menu", reply_markup=main_menu(lang))



# =========================================================
# 12) REWARD HELPERS
# =========================================================
async def reward_message(update, context, score, total, lang="en"):
    rate = score / max(total, 1)
    if rate >= 1.0:
        msg = "🌟 Perfect! All correct!" if lang != "ru" else "🌟 Отлично! Все правильно!"
    elif rate >= 0.6:
        msg = "⭐ Great work!" if lang != "ru" else "⭐ Отличная работа!"
    else:
        msg = "👏 Nice try!" if lang != "ru" else "👏 Хорошая попытка!"
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Menu", callback_data="menu:root")]])
    await safe_reply_message(update.message, msg, reply_markup=kb)
    await log_event(context, "reward_given", update.effective_user.id, {"score": score})


# =========================================================
# 13) CALLBACK HANDLER
# =========================================================
# =========================================================
# 13) CALLBACK HANDLER (FINAL VERSION)
# =========================================================
async def on_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    data = q.data or ""
    await q.answer()
    uid = update.effective_user.id
    prefs = get_prefs(uid)
    lang = prefs.get("lang", "en")

    # === MENU ROOT ===
    if data == "menu:root":
        prefs["mode"] = "chat"
        context.user_data.clear()
        msg = "Back to main menu." if lang != "ru" else "Возврат в меню."
        await safe_edit_text(q, msg, reply_markup=main_menu(lang))
        await log_event(context, "menu_root", uid, {})
        return

    # === LANGUAGE SELECT ===
    if data == "menu:lang":
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("English", callback_data="set_lang:en"),
             InlineKeyboardButton("Русский", callback_data="set_lang:ru")]
        ])
        await safe_edit_text(q, "Choose language / Выберите язык:", reply_markup=kb)
        return

    if data.startswith("set_lang:"):
        lang = data.split(":")[1]
        prefs["lang"] = lang
        txt = "Language set to English." if lang == "en" else "Язык: Русский."
        await safe_edit_text(q, txt, reply_markup=main_menu(lang))
        await log_event(context, "lang_set", uid, {"lang": lang})
        return

    # === GRADE SELECT ===
    if data == "menu:grade":
        txt = "Select your grade:" if lang != "ru" else "Выберите класс:"
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("6", callback_data="set_grade:6"),
             InlineKeyboardButton("7", callback_data="set_grade:7"),
             InlineKeyboardButton("8", callback_data="set_grade:8"),
             InlineKeyboardButton("9", callback_data="set_grade:9"),
             InlineKeyboardButton("10", callback_data="set_grade:10")],
            [InlineKeyboardButton("⬅️ Back", callback_data="menu:root")]
        ])
        await safe_edit_text(q, txt, reply_markup=kb)
        return

    if data.startswith("set_grade:"):
        g = data.split(":")[1]
        if g in GRADE_TO_CEFR:
            prefs["grade"] = g
            prefs["cefr"] = GRADE_TO_CEFR[g]
            prefs["mode"] = "chat"
            context.user_data.clear()
            txt = (f"Grade set to {g} (level {prefs['cefr']})."
                   if lang != "ru" else f"Класс {g} (уровень {prefs['cefr']}).")
            await safe_edit_text(q, txt, reply_markup=main_menu(lang))
            await log_event(context, "grade_set", uid, {"grade": g, "cefr": prefs["cefr"]})
        return

    # === HELP MENU ===
    if data == "menu:help":
        txt = HELP_TEXT_RU if lang == "ru" else HELP_TEXT_EN
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Back to menu", callback_data="menu:root")]])
        await safe_edit_text(q, txt, reply_markup=kb)
        await log_event(context, "help_open", uid, {})
        return

    # === MAIN PRACTICE MENU ===
    if data == "menu:practice":
        txt = "Choose practice type:" if lang != "ru" else "Выберите тип практики:"
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("Multiple Choice", callback_data="practice:type:mcq"),
             InlineKeyboardButton("Fill in the Blanks", callback_data="practice:type:fill")],
            [InlineKeyboardButton("Verb Forms", callback_data="practice:type:verb"),
             InlineKeyboardButton("Error Correction", callback_data="practice:type:error")],
            [InlineKeyboardButton("⬅️ Back", callback_data="menu:root")]
        ])
        await safe_edit_text(q, txt, reply_markup=kb)
        return

    # === PRACTICE TYPE HANDLER (GRAMMAR MODE) ===
    if data.startswith("practice:type:"):
        typ = data.split(":")[2]
        flavor_map = {
            "mcq": "grammar_mcq",
            "fill": "grammar_fill",
            "verb": "verb_forms",
            "error": "error_fix"
        }
        flavor = flavor_map.get(typ, "grammar_mcq")
        topic = context.user_data.get("last_grammar_topic", "English Grammar")
        items = await build_mcq(topic, lang, prefs["cefr"], flavor=flavor)
        items = items[:5]
        if not items:
            return await safe_edit_text(q, "⚠️ No questions found.", reply_markup=main_menu(lang))
        context.user_data["practice"] = {
            "type": typ, "topic": topic, "items": items,
            "idx": 0, "score": 0, "ui_lang": lang, "scope": "grammar"
        }
        await send_practice_item(q, context)
        await log_event(context, "grammar_practice_start", uid, {"type": typ, "count": len(items)})
        return

    # === VOCAB PRACTICE ===
    if data == "vocab:practice":
        word = context.user_data.get("last_word", "").strip()
        if not word:
            return await safe_edit_text(q, "Please define a word first.", reply_markup=main_menu(lang))
        flavors = ["vocab_syn", "vocab_ant", "vocab_cloze"]
        all_items = []
        for f in flavors:
            sub = await build_mcq(word, lang, prefs["cefr"], flavor=f)
            all_items.extend(sub[:1])
        items = all_items[:3]
        if not items:
            return await safe_edit_text(q, "⚠️ No quiz available.", reply_markup=main_menu(lang))
        context.user_data["practice"] = {
            "type": "vocab", "topic": word, "items": items,
            "idx": 0, "score": 0, "ui_lang": lang, "scope": "vocab"
        }
        await send_practice_item(q, context)
        await log_event(context, "vocab_practice", uid, {"word": word})
        return

        # === VOCAB MORE EXAMPLES (B1+ level) ===
    if data == "vocab:more":
        word = (context.user_data.get("last_word") or "").strip()
        if not word:
            return await safe_edit_text(
                q,
                "Please define or search a word first."
                if lang != "ru" else "Сначала найди слово.",
                reply_markup=main_menu(lang)
            )

        prompt = (
            f"Give 3 additional example sentences for the word or phrase '{word}'.\n"
            "• Level: B1+ (upper-intermediate)\n"
            "• Each sentence 6–12 words.\n"
            "• Increase difficulty slightly each time.\n"
            "• English only. No translation. No markdown.\n"
            "Format:\n1) ...\n2) ...\n3) ..."
        )

        msgs = [
            {"role": "system", "content": POLICY_STUDY},
            {"role": "user", "content": prompt}
        ]

        out = await ask_openai(msgs, max_tokens=180)
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("✏️ Practice this word", callback_data="vocab:practice"),
             InlineKeyboardButton("🏠 Menu", callback_data="menu:root")]
        ])

        await safe_edit_text(q, trim(out), reply_markup=kb)
        await log_event(context, "vocab_more_examples", uid, {"word": word})
        return


        # === GRAMMAR PRACTICE (3 câu theo last_grammar_topic) ===
    if data == "grammar:practice":
        topic = context.user_data.get("last_grammar_topic", "Present Simple")
        if not topic:
            return await safe_edit_text(q, "No grammar topic found.", reply_markup=main_menu(lang))
        flavors = ["grammar_verb", "grammar_error", "grammar_order"]
        all_items = []
        for f in flavors:
            sub = await build_mcq(topic, lang, prefs["cefr"], flavor=f)
            all_items.extend(sub[:1])  # mỗi loại lấy 1 câu
        items = all_items[:3]  # tổng 3 câu
        if not items:
            return await safe_edit_text(q, "⚠️ No questions found.", reply_markup=main_menu(lang))
        context.user_data["practice"] = {
            "type": "grammar", "topic": topic, "items": items,
            "idx": 0, "score": 0, "ui_lang": lang, "scope": "grammar"
        }
        await send_practice_item(q, context)
        await log_event(context, "grammar_practice_start", uid, {"topic": topic, "count": len(items)})
        return

    # === EXPLAIN MORE CALLBACK ===
    if data == "footer:explain_more":
        topic = context.user_data.get("last_grammar_topic", "Present Simple")
        prompt = (
            f"Add more details and pitfalls for '{topic}' (level {prefs['cefr']}). "
            "Include 3 new examples and short explanations. No markdown."
        )
        msgs = [{"role": "system", "content": POLICY_STUDY},
                {"role": "user", "content": prompt}]
        out = await ask_openai(msgs, max_tokens=300)
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("✏️ Practice this rule", callback_data="grammar:practice")],
            [InlineKeyboardButton("🏠 Menu", callback_data="menu:root")]
        ])
        await safe_edit_text(q, trim(out), reply_markup=kb)
        await log_event(context, "grammar_explain_more", uid, {"topic": topic})
        return


    # === READING GLOSS (text) ===
    if data == "reading:gloss":
        passage = context.user_data.get("last_passage", "")
        if not passage:
            return await safe_edit_text(q, "No passage found." if lang != "ru" else "Нет текста.")
        glossed = await build_reading_gloss(passage, lang)
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("📝 Practice this text", callback_data="reading:practice")],
            [InlineKeyboardButton("🏠 Menu", callback_data="menu:root")]
        ])
        await safe_edit_text(q, trim(glossed), reply_markup=kb)
        await log_event(context, "reading_gloss", uid, {"chars": len(passage)})
        return

    # === READING GLOSS (from OCR image) ===
    if data == "reading:gloss_from_image":
        text = context.user_data.get("image_text", "")
        if not text:
            return await safe_edit_text(q, "No text found from image.")
        glossed = await build_reading_gloss(text, lang)
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Menu", callback_data="menu:root")]])
        await safe_edit_text(q, trim(glossed), reply_markup=kb)
        await log_event(context, "image_gloss", uid, {"chars": len(text)})
        return

    # === AUTO-GLOSS CALLBACK ===
    if data == "reading:auto_gloss":
        text = context.user_data.get("auto_gloss_text", "")
        if not text:
            return await safe_edit_text(q, "No text found to gloss.")
        glossed = await build_reading_gloss(text, prefs["lang"])
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("📝 Practice this text", callback_data="reading:practice")],
            [InlineKeyboardButton("🏠 Menu", callback_data="menu:root")]
        ])
        await safe_edit_text(q, trim(glossed), reply_markup=kb)
        await log_event(context, "auto_gloss_done", uid, {"chars": len(text)})
        return

    # === READING PRACTICE ===
    if data == "reading:practice":
        passage = context.user_data.get("last_passage", "")
        topic = context.user_data.get("reading_topic", "reading")
        if not passage:
            return await safe_edit_text(q, "No passage found.", reply_markup=main_menu(lang))
        items = await build_mcq(passage, lang, prefs["cefr"], flavor="reading_detail")
        items = items[:5]
        if not items:
            return await safe_edit_text(q, "⚠️ Failed to build questions.", reply_markup=main_menu(lang))
        context.user_data["practice"] = {
            "type": "reading", "topic": topic, "items": items,
            "idx": 0, "score": 0, "ui_lang": lang, "scope": "reading"
        }
        await send_practice_item(q, context)
        await log_event(context, "reading_practice", uid, {"topic": topic})
        return

    # === ANSWER HANDLING ===
    if data.startswith("ans:"):
        st = context.user_data.get("practice")
        if not st:
            return await safe_edit_text(q, "No active quiz.")
        ch = data.split(":")[1]
        idx = st["idx"]
        qitem = st["items"][idx]
        correct = qitem["answer"]
        ui_lang = st.get("ui_lang", "en")

        # check answer
        if ch == correct:
            st["score"] += 1
            msg = "✅ Correct!" if ui_lang != "ru" else "✅ Верно!"
        else:
            msg = f"❌ The correct answer: {correct}" if ui_lang != "ru" else f"❌ Правильный ответ: {correct}"
        await safe_edit_text(q, msg)
        await asyncio.sleep(1)

        # next or summary
        st["idx"] += 1
        if st["idx"] >= len(st["items"]):
            dummy = Update(update.update_id, message=q.message)
            await practice_summary(dummy, context)
        else:
            await send_practice_item(q, context)
        return

    # === FOOTER AGAIN CALLBACK ===
    if data == "footer:again":
        st = context.user_data.get("practice")
        if not st:
            return await safe_edit_text(q, "No previous quiz found.", reply_markup=main_menu(lang))
        topic = st.get("topic", "English")
        flavor = st.get("flavor", "generic")
        items = await build_mcq(topic, lang, prefs["cefr"], flavor=flavor)
        st.update({"items": items[:5], "idx": 0, "score": 0})
        await safe_edit_text(q, "New quiz starting…")
        await send_practice_item(q, context)
        return


    # === TALK TOPIC SELECT ===
    if data.startswith("talk:topic:"):
        topic = data.split(":")[-1]
        context.user_data["talk"] = {"topic": topic, "turns": 0}
        greet = f"Let's talk about {topic}! You start 😊" if lang != "ru" else f"Поговорим о {topic}! Начинай 😊"
        await safe_edit_text(q, greet, reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🏠 Menu", callback_data="menu:root")]
        ]))
        await log_event(context, "talk_topic", uid, {"topic": topic})
        return


    # --- HELP MENU CALLBACK ---
    if data == "menu:help":
        txt = HELP_TEXT_RU if lang == "ru" else HELP_TEXT_EN
        await safe_edit_text(q, txt, reply_markup=main_menu(lang))
        await log_event(context, "help_open", uid, {})
        return

# =========================================================
async def extract_text_from_image(file_obj):
    """Use OCR.Space API instead of pytesseract for Render compatibility."""
    try:
        bio = io.BytesIO()
        await file_obj.download_to_memory(out=bio)
        bio.seek(0)
        files = {'file': ('image.jpg', bio, 'image/jpeg')}
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                "https://api.ocr.space/parse/image",
                data={"language": "eng", "isOverlayRequired": False},
                files=files
            )
        data = r.json()
        text = data.get("ParsedResults", [{}])[0].get("ParsedText", "")
        return text.strip()
    except Exception as e:
        logger.warning(f"OCR API failed: {e}")
        return ""

# =========================================================
# 15) TALK COACH & NUDGE SYSTEM
# =========================================================
async def talk_coach(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prefs = get_prefs(update.effective_user.id)
    lang = prefs.get("lang", "en")
    state = context.user_data.get("talk", {"topic": "school life", "turns": 0})
    topic = state.get("topic", "school life")
    user_text = update.message.text or ""
    reply = await talk_reply(user_text, topic, lang)
    state["turns"] = state.get("turns", 0) + 1
    context.user_data["talk"] = state
    await safe_reply_message(update.message, trim(reply), reply_markup=InlineKeyboardMarkup([
        [InlineKeyboardButton("💬 More ideas", callback_data="talk:more"),
         InlineKeyboardButton("🏠 Menu", callback_data="menu:root")]
    ]))
    if state["turns"] >= 20:
        await safe_reply_message(update.message,
            "Great chat! Try Vocabulary or Grammar practice next." if lang!="ru"
            else "Отличная беседа! Попробуй упражнения по словарю или грамматике.",
            reply_markup=main_menu(lang))
        context.user_data.pop("talk", None)
    await log_event(context, "talk_message", update.effective_user.id,
                    {"topic": topic, "turns": state["turns"]})


# --- Nudge mini-quiz ---
def increment_nudge(context):
    c = context.user_data.get("nudge", 0) + 1
    context.user_data["nudge"] = c
    return c

def reset_nudge(context):
    context.user_data["nudge"] = 0

async def maybe_nudge(update, context, lang):
    c = increment_nudge(context)
    if c >= 3:
        reset_nudge(context)
        msg = "Do a quick 2-question mini-quiz?" if lang!="ru" else "Хочешь мини-викторину из 2 вопросов?"
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("▶️ Start", callback_data="nudge:start"),
             InlineKeyboardButton("⏭ Skip", callback_data="nudge:skip")]
        ])
        await safe_reply_message(update.message, msg, reply_markup=kb)
        await log_event(context, "nudge_offer", update.effective_user.id, {})


# =========================================================
# 16) HANDLE MESSAGE (CHAT-FIRST LOGIC)
# =========================================================
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    if not text: return
# --- AUTO GRAMMAR HINT (Patch 11) ---
    grammar_hints = [
        (r"\b(am|is|are)\s+\w+ing\b", "Present Continuous — be + V-ing for actions happening now."),
        (r"\b(was|were)\s+\w+ing\b", "Past Continuous — was/were + V-ing for actions in progress in the past."),
        (r"\b(has|have)\s+\w+(ed|en)\b", "Present Perfect — have/has + V3 for experiences or recent results."),
        (r"\bhad\s+\w+(ed|en)\b", "Past Perfect — had + V3 for actions before another past."),
        (r"\bwill\s+\w+\b", "Future Simple — will + base verb for future predictions."),
        (r"\b(am|is|are|was|were|been|be)\s+\w+(ed|en)\b", "Passive Voice — be + V3 (object focus)."),
        (r"\b(should|must|can|could|may|might|shall|will|would)\b", "Modal verbs — use base form after modal."),
        (r"\bif\b.*\bwill\b", "First Conditional — If + Present, will + V."),
        (r"\bif\b.*\bwould\b", "Second Conditional — If + Past, would + V."),
        (r"\bif\b.*\bhad\b", "Third Conditional — If + Past Perfect, would have + V3."),
        (r"\b(er than|more .+ than)\b", "Comparatives — adjective + than."),
        (r"\b(the .+est|the most)\b", "Superlatives — the + adj-est / the most + adjective."),
    ]
    for pattern, hint in grammar_hints:
        if re.search(pattern, text, re.I):
            await safe_reply_message(update.message, f"💡 Grammar hint: {hint}")
            await log_event(context, "grammar_hint", update.effective_user.id, {"hint": hint})
            break

# =========================================================
# PATCH 10: AUTO-GLOSS & SMART GRAMMAR GUIDANCE
# =========================================================
    # 1️⃣ Auto Gloss trigger for long English text
    word_count = len(re.findall(r"[A-Za-z]+", text))
    if word_count >= 60 and not re.search(r"\b(translate|gloss)\b", text, re.I):
        msg = ("This looks like a reading passage. Would you like me to gloss it?"
               if detect_lang(text) == "en"
               else "Похоже, это английский текст. Сделать глоссы?")
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("📝 Gloss it", callback_data="reading:auto_gloss"),
             InlineKeyboardButton("⏭ Skip", callback_data="nudge:skip")]
        ])
        context.user_data["auto_gloss_text"] = text
        await safe_reply_message(update.message, msg, reply_markup=kb)
        await log_event(context, "auto_gloss_offer", update.effective_user.id, {"words": word_count})
        return

    # 2️⃣ Smart Grammar detector for textbook-style exercises
    if re.search(r"\b(fill in|underline|choose|complete|correct)\b", text.lower()):
        msg = ("It looks like a grammar exercise. "
               "I can guide you step by step instead of giving direct answers. "
               "What grammar topic is this about?")
        await safe_reply_message(update.message, msg)
        await log_event(context, "textbook_ex_detected", update.effective_user.id, {"text": text[:80]})
        return

  # --- TALK CONTEXT CONTINUE ---
    if "talk" in context.user_data:
        user_text = (update.message.text or "").strip().lower()

        # --- TALK EXIT HANDLER (Patch 12.5) ---
        if user_text in ("exit", "quit", "menu", "back", "stop", "меню", "выход"):
            prefs = get_prefs(update.effective_user.id)
            lang = prefs.get("lang", "en")
            context.user_data.pop("talk", None)
            msg = "Exited talk mode. Back to main menu." if lang != "ru" else "Выход из режима разговора. Главное меню."
            await safe_reply_message(update.message, msg, reply_markup=main_menu(lang))
            await log_event(context, "talk_exit", update.effective_user.id, {})
            return

        # --- NORMAL TALK FLOW ---
        return await talk_coach(update, context)

    # --- GENERAL FILTERS & SETUP ---
    if blocked(text):
        return await safe_reply_message(update.message,
            "⛔ Please keep it school-appropriate. Try an English topic.")

    uid = update.effective_user.id
    prefs = get_prefs(uid)
    lang = prefs.get("lang", "en")
    if lang == "auto":
        lang = detect_lang(text)


    # GREETING DETECTION
    if re.fullmatch(r"hi|hello|hey|привет|здравствуй", text.lower()):
        msg = ("Hello! I'm your English tutor. Ask me anything about English learning!"
               if lang!="ru" else "Привет! Я твой помощник по английскому. Задай вопрос о языке!")
        return await safe_reply_message(update.message, msg, reply_markup=main_menu(lang))

    # --- INTENT DETECTION ---
    t = text.lower()
    intent = "chat"
    if re.search(r"\bdefine\b|\bmeaning of\b", t): intent = "vocab"
    elif re.search(r"\bgrammar\b|\btense\b|\bexplain\b|\brule\b", t): intent = "grammar"
    elif re.search(r"\btalk\b|\bconversation\b|\bspeak\b", t): intent = "talk"        # 👈 move up
    elif re.search(r"\bread\b|\btext\b|\bwrite\b|\btranslate\b|\bgloss\b", t): intent = "reading"
    elif re.search(r"\bquiz\b|\bpractice\b|\bexercise\b", t): intent = "practice"


    # --- VOCABULARY ---
        # --- VOCABULARY (chat-first) ---
    if intent == "vocab":
        reset_nudge(context)

        # 🧩 Làm sạch từ khóa và kiểm tra hợp lệ
        word = re.sub(r"[^A-Za-z' -]", "", text).strip()
        if not word or len(word) < 2:
            return await safe_reply_message(
                update.message,
                "Please type a valid English word or phrase (e.g., 'define look after')."
                if lang != "ru" else
                "Пожалуйста, напиши корректное английское слово или фразу (например, 'define look after')."
            )

        # 🧠 Sinh vocabulary card (IPA + POS + nghĩa EN + nghĩa RU ngắn)
        card = await build_vocab_card(word, prefs)

        # 💾 Lưu lại để practice hoạt động
        context.user_data["last_word"] = word
        add_vocab_to_bank(context, word)
        prefs["mode"] = "vocab"

        # 📘 Gửi kết quả + nút tương tác
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("✏️ Practice this word", callback_data="vocab:practice"),
             InlineKeyboardButton("➕ More examples", callback_data="vocab:more")],
            [InlineKeyboardButton("🏠 Menu", callback_data="menu:root")]
        ])

        await safe_reply_message(update.message, trim(card), reply_markup=kb)

        # 🧾 Ghi log
        await log_event(context, "vocab_card", uid, {"word": word})
        return await maybe_nudge(update, context, lang)

    # --- GRAMMAR ---
        # --- GRAMMAR ---
    if intent == "grammar":
        reset_nudge(context)

        # ✅ Sinh phần giải thích ngữ pháp
        exp = await build_grammar_explain(text, prefs)

        # ✅ Lưu lại topic để practice / explain more dùng
        context.user_data["last_grammar_topic"] = text
        prefs["mode"] = "grammar"

        # ✅ Gửi phản hồi + nút tương tác
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("✏️ Practice this rule", callback_data="grammar:practice"),
             InlineKeyboardButton("📚 Explain more", callback_data="footer:explain_more")],
            [InlineKeyboardButton("🏠 Menu", callback_data="menu:root")]
        ])

        await safe_reply_message(update.message, trim(exp), reply_markup=kb)
        await log_event(context, "grammar_explain", uid, {"topic": text})
        return await maybe_nudge(update, context, lang)

    # --- READING ---
    if intent == "reading":
        reset_nudge(context)
        level = prefs["cefr"]

        # 🧩 Kiểm tra độ dài để đoán: My text hay Topic
        word_count = len(text.split())

        # Nếu đoạn quá ngắn → hiểu là Topic
        if word_count < 50:
            topic = text.strip().capitalize()
            passage = await build_reading_passage(topic, prefs)
            context.user_data["last_passage"] = passage
            context.user_data["reading_topic"] = topic

            await safe_reply_message(update.message, trim(passage), reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📘 Gloss this text", callback_data="reading:gloss"),
                 InlineKeyboardButton("📝 Practice this text", callback_data="reading:practice")],
                [InlineKeyboardButton("🏠 Menu", callback_data="menu:root")]
            ]))
            await log_event(context, "reading_passage", uid, {"topic": topic, "mode": "auto_topic"})
            return

        # Nếu đoạn dài (≥ 50 từ) → hiểu là My text
        passage = text.strip()
        context.user_data["last_passage"] = passage
        context.user_data["reading_topic"] = "mytext"

        await safe_reply_message(update.message,
            "Got it! Here’s your text:" if lang != "ru" else "Понял! Вот твой текст:")
        await safe_reply_message(update.message, trim(passage), reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("📘 Gloss this text", callback_data="reading:gloss"),
             InlineKeyboardButton("📝 Practice this text", callback_data="reading:practice")],
            [InlineKeyboardButton("🏠 Menu", callback_data="menu:root")]
        ]))
        await log_event(context, "reading_passage", uid, {"topic": "mytext", "words": word_count})
        return


    # --- TALK ---
    if intent == "talk":
        reset_nudge(context)
        context.user_data["talk"] = {"topic": "school life", "turns": 0}
        greet = "Let's talk! What's your favorite subject?" if lang!="ru" else "Поговорим! Какая твоя любимая тема?"
        await safe_reply_message(update.message, greet, reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🏠 Menu", callback_data="menu:root")]
        ]))
        await log_event(context, "talk_start", uid, {})
        return

    # --- PRACTICE ---
    if intent == "practice":
        reset_nudge(context)

        # ✅ Tạo 5 câu hỏi hỗn hợp: multiple choice, fill, verb form, error fix
        flavors = ["grammar_mcq", "grammar_fill", "verb_forms", "error_fix"]
        all_items = []
        for f in flavors:
            try:
                sub = await build_mcq(text, lang, prefs["cefr"], flavor=f)
                all_items.extend(sub[:1])   # lấy 1 câu từ mỗi loại
            except Exception as e:
                logger.warning(f"build_mcq failed for {f}: {e}")
                continue

        items = all_items[:5]  # tổng cộng 5 câu

        if not items:
            return await safe_reply_message(
                update.message,
                "⚠️ I couldn't create practice questions. Try another topic."
            )

        context.user_data["practice"] = {
            "type": "mcq",
            "topic": text,
            "items": items,
            "idx": 0,
            "score": 0,
            "ui_lang": lang,
            "scope": "free"
        }

        await send_practice_item(update, context)
        await log_event(context, "practice_start", uid, {"topic": text, "count": len(items)})
        return


    # --- DEFAULT CHAT ---
    msgs = [
        {"role": "system", "content": POLICY_CHAT},
        {"role": "user", "content": text}
    ]
    reply = await ask_openai(msgs, max_tokens=350)
    await safe_reply_message(update.message, trim(reply), reply_markup=main_menu(lang))
    await log_event(context, "chat_message", uid, {"chars": len(text)})
    await maybe_nudge(update, context, lang)
# =========================================================
# PATCH 9B: HANDLE IMAGE INPUT
# =========================================================
async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle photo messages: detect if it's text, grammar exercise, or unrelated."""
    photo = update.message.photo[-1]
    file = await photo.get_file()
    text = await extract_text_from_image(file)

    if not text:
        return await safe_reply_message(update.message, "I couldn't read the image clearly. Try again.")

    # Basic classification
        # =========================================================
    # PATCH 12: SMART GRAMMAR HINT FROM IMAGE
    # =========================================================
    if re.search(r"(exercise|fill|underline|choose|correct|complete)", text, re.I):
        # Step 1 — phản hồi cơ bản
        await safe_reply_message(update.message,
            "This looks like a grammar exercise 📘. Let me check the patterns...")
        await asyncio.sleep(0.5)

        # Step 2 — nhận diện cấu trúc ngữ pháp giống Patch 11
        grammar_hints = [
            (r"\b(am|is|are)\s+\w+ing\b", "Present Continuous — be + V-ing for actions happening now."),
            (r"\b(was|were)\s+\w+ing\b", "Past Continuous — was/were + V-ing for actions in progress in the past."),
            (r"\b(has|have)\s+\w+(ed|en)\b", "Present Perfect — have/has + V3 for experiences or recent results."),
            (r"\bhad\s+\w+(ed|en)\b", "Past Perfect — had + V3 for actions before another past."),
            (r"\bwill\s+\w+\b", "Future Simple — will + base verb for predictions."),
            (r"\b(am|is|are|was|were|been|be)\s+\w+(ed|en)\b", "Passive Voice — be + V3 (object focus)."),
            (r"\bif\b.*\bwill\b", "First Conditional — If + Present, will + V."),
            (r"\bif\b.*\bwould\b", "Second Conditional — If + Past, would + V."),
            (r"\bif\b.*\bhad\b", "Third Conditional — If + Past Perfect, would have + V3."),
            (r"\b(er than|more .+ than)\b", "Comparatives — adjective + than."),
        ]

        matched = False
        for pattern, hint in grammar_hints:
            if re.search(pattern, text, re.I):
                await safe_reply_message(update.message, f"💡 Grammar hint: {hint}")
                await log_event(context, "image_grammar_hint", update.effective_user.id, {"hint": hint})
                matched = True
                break

        # Step 3 — nếu không nhận ra gì cụ thể
        if not matched:
            msg = ("It seems to be a grammar task, but I can't identify the rule yet. "
                   "Can you tell me which topic this is about?")
            await safe_reply_message(update.message, msg)
        return



# =========================================================
# 17) FLASK HEALTHCHECK & MAIN ENTRYPOINT
# =========================================================
app = Flask(__name__)

@app.get("/")
def health():
    return "✅ AI English Tutor v2 is alive", 200

def start_flask():
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)

def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_cmd))
    application.add_handler(CallbackQueryHandler(on_cb))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))
    application.add_error_handler(on_error)
    application.post_init = on_startup
    threading = asyncio.get_event_loop().create_task(asyncio.to_thread(start_flask))
    logger.info("🚀 Bot starting: English Tutor v2 ready for class!")
    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == "__main__":
    main()
