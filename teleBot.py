# =========================================================
# teleBot_v2_full.py
# =========================================================
# 0) IMPORTS & GLOBAL SETUP
# =========================================================
import os, re, json, time, hmac, hashlib, logging, asyncio, uuid, difflib
from datetime import datetime, timezone
import threading

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

# --- SIMPLE VOCAB BANK HANDLER ---
def add_vocab_to_bank(context, word: str):
    """Lưu từ vựng vào bộ nhớ tạm (per-user)."""
    if not word:
        return
    bank = context.user_data.get("vocab_bank", [])
    if word not in bank:
        bank.append(word)
        context.user_data["vocab_bank"] = bank
    logger.info(f"VOCAB BANK UPDATED: {context.user_data['vocab_bank']}")
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
    "You are a friendly and flexible English-learning assistant for students in grades 6–9 (CEFR A2–B1+). "
    "Your role is to help them improve their English through natural conversation and interactive learning. "
    "You may discuss any topic (school, hobbies, science, math, daily life, technology, current events, etc.) "
    "as long as you use English that matches their level. "
    "You can briefly use the student's native language (Russian) for short clarifications or translations, "
    "but most of your reply should remain in simple English."
    "If the student asks for an explanation, dialogue, or story — respond fully and clearly. "
    "If the message sounds like casual chat, reply briefly and naturally. "
    "You can discuss academic topics *in English* for learning purposes, "
    "but do not perform calculations, write code, or complete homework tasks. "
    "Keep your tone friendly, supportive, and age-appropriate. "
    "Use plain English only (no markdown, no bold)."
    "Never use **, *, _, or other formatting markers. Output plain text only."
)

POLICY_STUDY = (
    "You are an English teacher for middle-school students (CEFR A2–B1+). "
    "Use clear, simple English that matches their level. "
    "Keep content safe, encouraging, and age-appropriate. "
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
    """Gửi tin nhắn an toàn (fallback khi Telegram từ chối)."""
    try:
        msg = await message.reply_text(text, reply_markup=reply_markup)
        return msg
    except BadRequest:
        try:
            msg = await message.reply_text(trim(text))
            return msg
        except Exception as e:
            logger.warning("safe_reply failed: %s", e)
            return None


async def safe_edit_text(query, text: str, reply_markup=None):
    try:
        return await query.edit_message_text(text, reply_markup=reply_markup)
    except BadRequest:
        try:
            return await query.edit_message_text(trim(text))
        except Exception as e:
            logger.warning("safe_edit_text failed: %s", e)

def mcq_buttons(options):
        """Tạo nút A/B/C/D cho câu hỏi hiện tại."""
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("A", callback_data="ans:A"),
             InlineKeyboardButton("B", callback_data="ans:B"),
             InlineKeyboardButton("C", callback_data="ans:C"),
             InlineKeyboardButton("D", callback_data="ans:D")]
        ])


# =========================================================
# CLEAR CHAT COMMAND
async def clear_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Xóa các tin nhắn cũ của bot."""
    try:
        chat_id = update.effective_chat.id
        messages = context.user_data.get("messages_to_delete", [])
        for mid in messages:
            try:
                await context.bot.delete_message(chat_id, mid)
            except Exception:
                continue
        context.user_data["messages_to_delete"] = []
        await update.message.reply_text("🧹 Chat cleared!")
    except Exception as e:
        logger.warning(f"Clear chat failed: {e}")
        await update.message.reply_text("⚠️ Failed to clear chat.")

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
# 8) START / HELP / MENU COMMANDS
# =========================================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    greet = "Choose your language / Выберите язык:"
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("English", callback_data="set_lang:en"),
         InlineKeyboardButton("Русский", callback_data="set_lang:ru")]
    ])
    await safe_reply_message(update.message, greet, reply_markup=kb)
    await log_event(context, "start", update.effective_user.id, {})


# --- MENU COMMAND HANDLER ---
async def handle_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Hiển thị lại menu chính khi người dùng gõ /menu"""
    prefs = get_prefs(update.effective_user.id)
    prefs["mode"] = "chat"          # 🟢 Reset mode ngay
    context.user_data.clear()
    reset_nudge(context)            # 🟢 Reset bộ đếm quiz mini
    
    lang = prefs.get("lang", "en")
    await safe_reply_message(update.message, "📋 Main menu:", reply_markup=main_menu(lang))
    await log_event(context, "menu_command", update.effective_user.id, {})

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
        f"You are an English vocabulary teacher for secondary school students (A2–B1+ level).\n"
        f"Adjust difficulty based on CEFR level {level}."
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
    try:
        result = await ask_openai(msgs, max_tokens=400)
        return result or f"[No response for word '{word}']"
    except Exception as e:
        logger.warning(f"⚠️ build_vocab_card failed for '{word}': {e}")
        return f"[Error generating card for '{word}']"


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


# --- Reading Gloss Builder (supports translated gloss) ---
async def build_reading_gloss(text: str, ui_lang: str, translate_mode: bool = True):
    """Always produce translated gloss (song ngữ) for A2–B1 learners."""
    gloss_lang = "English" if (translate_mode and ui_lang == "ru") else "Russian"

    prompt = (
        f"Gloss the given English text for A2–B1 learners:\n"
        f"- Keep the original English sentences.\n"
        f"- Select 12–15 useful or challenging English words and phrases.\n"
        f"- Include verbs, adjectives, and nouns that carry key meaning.\n"
        f"- Prefer idioms, phrasal verbs, collocations, or academic words.\n"
        f"- Enclose each English chunk in <angle brackets> and immediately add a short {gloss_lang} translation in parentheses.\n"
        "- Example: She <set up> (организовала) a small company.\n"
        "- Do NOT gloss every word, and do NOT use markdown.\n\n"
        "TEXT:\n" + text
    )

    msgs = [
        {"role": "system", "content": POLICY_STUDY},
        {"role": "user", "content": prompt}
    ]
    return await ask_openai(msgs, max_tokens=420)
 
# --- Talk Coach Builder ---
async def talk_reply(user_text: str, topic: str, ui_lang: str):
    """Friendly English coach — corrects lightly and gives short tips."""
    lang_note = (
        "If the student uses Russian or another language, respond mostly in English but briefly explain one key word in that language."
        if ui_lang == "ru" else
        "Keep the whole reply in English."
    )

    prompt = (
        f"You are an encouraging English speaking coach for students (A2–B1+). "
        f"Topic: {topic}. The student said: '{user_text}'. "
        "1️⃣ Respond naturally in 1–3 sentences of conversational English.\n"
        "2️⃣ Correct grammar or vocabulary mistakes implicitly (reformulate naturally).\n"
        "3️⃣ Add 1–2 short useful phrases, words, or sentence patterns that fit the topic, marked with '[Tip:]'.\n"
        "4️⃣ End your reply with one friendly question to keep the talk going.\n"
        f"5️⃣ {lang_note}\n"
        "Output plain text only. No markdown, no bold, no lists."
    )

    msgs = [
        {"role": "system", "content": POLICY_STUDY},
        {"role": "user", "content": prompt}
    ]

    try:
        return await ask_openai(msgs, max_tokens=200)
    except Exception as e:
        logger.warning(f"talk_reply failed: {e}")
        return "Sorry, I didn’t catch that. Could you say it again?"

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
    Create a 5-question MCQ set based on grade, topic, and exercise flavor.
    Supports: vocab_*, grammar_*, reading_*.
    """
    # =========================
    # 1️⃣ Define task map
    # =========================
    task_map = {
        # --- VOCABULARY TYPES ---
        "vocab_synonyms": (
            "Write 5 multiple-choice questions testing SYNONYMS for the given word or phrase. "
            "Each question should have:\n"
            "• A short instruction like 'Choose the synonym for ...'.\n"
            "• 4 options (A–D), one correct synonym.\n"
            "• A brief explanation (≤20 words) why it’s correct."
        ),
        "vocab_antonyms": (
            "Write 5 MCQs testing ANTONYMS of the given word or phrase. "
            "Include short sentences where helpful. Each question: 4 options (A–D), one correct opposite meaning."
        ),
        "vocab_context": (
            "Write 5 MCQs asking students to choose the best word IN CONTEXT. "
            "Each question shows one short sentence with a blank '____'. "
            "Provide 4 options, 1 correct, 3 distractors. Add short explanations."
        ),
        "vocab_formation": (
            "Write 5 MCQs testing WORD FORMATION (noun, verb, adjective, adverb forms). "
            "Each question should include a sentence with a blank, e.g. 'She was very ____ (beauty)'."
        ),
        "vocab_collocations": (
            "Write 5 MCQs testing common COLLOCATIONS with the given word. "
            "Each question gives a phrase with a missing word (e.g., 'make ___', 'heavy ___'). "
            "4 options (1 correct), short explanations."
        ),
        "vocab_phrasal": (
            "Write 5 MCQs testing PHRASAL VERBS with the given base verb. "
            "Each question uses a natural short sentence and 4 options (e.g. 'give up', 'take off'). "
            "Include correct answer and a 1-sentence explanation."
        ),

        # --- GRAMMAR TYPES ---
        "grammar_verbs": (
            "Write 5 MCQs where students choose the correct VERB FORM (tense, agreement, or aspect). "
            "Each question has 4 options and a short explanation. Level: {level}."
        ),
        "grammar_errors": (
            "Write 5 MCQs for ERROR CORRECTION. Each shows one incorrect sentence. "
            "Provide 4 corrected versions (A–D). Explain briefly why the correct form is right."
        ),
        "grammar_order": (
            "Write 5 MCQs testing WORD ORDER. "
            "Each question gives 4 jumbled options (A–D), one grammatically correct."
        ),
        "grammar_conditionals": (
            "Write 5 MCQs about CONDITIONALS (0, 1st, 2nd, 3rd). "
            "Include mixed examples with 4 options. Add 1-sentence explanation."
        ),
        "grammar_modals": (
            "Write 5 MCQs about MODAL VERBS (can, must, should, might, etc.). "
            "Ask for correct usage or meaning. Provide 4 options, short explanation."
        ),
        "grammar_mixed": (
            "Write 5 mixed GRAMMAR MCQs combining tenses, modals, and prepositions. "
            "Level: {level}. Each question has 4 clear options."
        ),

        # --- READING TYPES ---
        "reading_mainidea": (
            "Write 5 READING COMPREHENSION questions about the MAIN IDEA of the passage. "
            "Avoid details; focus on topic and purpose."
        ),
        "reading_details": (
            "Write 5 MCQs about SPECIFIC DETAILS or facts from the passage. "
            "Each question should have 4 options and 1-sentence explanation."
        ),
        "reading_inference": (
            "Write 5 MCQs testing INFERENCE — what can be understood but not directly stated. "
            "Each question has 4 choices and a short explanation."
        ),
        "reading_vocabcontext": (
            "Write 5 MCQs about VOCABULARY IN CONTEXT. "
            "Each question quotes a short sentence and asks the meaning of one word or phrase."
        ),
        "reading_cloze": (
            "Write 5 CLOZE TEST questions (fill in missing word in passage). "
            "Each blank should have 4 possible options (A–D)."
        ),

        # --- FALLBACK / GENERIC ---
        "generic": (
            "Write 5 general English MCQs (A2–B1+). "
            "Mix grammar, vocabulary, and reading comprehension. 4 options per question."
        ),
    }

    # =========================
    # 2️⃣ Select task prompt
    # =========================
    task = task_map.get(flavor, task_map["generic"])

    # Difficulty tag
    if level in ("A2", "A2+"):
        diff_note = "Use simple sentences and everyday words."
    elif level == "B1":
        diff_note = "Include 1–2 slightly more advanced structures or idioms."
    else:
        diff_note = "Keep within A2–B1 school-level range."

    # =========================
    # 3️⃣ Construct model prompt
    # =========================
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

    # =========================
    # 4️⃣ Request from model
    # =========================
    raw = await ask_openai(msgs, max_tokens=950)
    try:
        data = json.loads(re.search(r"\{.*\}", raw, re.S).group())
        questions = data.get("questions", [])
    except Exception as e:
        logger.warning(f"MCQ parse fail: {e} | raw={raw}")
        questions = []

    # =========================
    # 5️⃣ Validate questions
    # =========================
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


# =========================================================
async def send_practice_item(update_or_query, context: ContextTypes.DEFAULT_TYPE):
    """Gửi 1 câu hỏi trắc nghiệm (MCQ) có 4 lựa chọn A–D, hiển thị gọn và an toàn."""
    st = context.user_data.get("practice")
    if not st:
        return

    idx = st["idx"]
    q = st["items"][idx]
    total = len(st["items"])

    # --- Build question text safely ---
    question = q.get("question", "").strip()
    options = q.get("options", [])
    if not options:
        msg_target = (
            update_or_query.message
            if isinstance(update_or_query, Update)
            else update_or_query.message
        )
        return await safe_reply_message(msg_target, "⚠️ This question has no options.")

    # --- Format hiển thị ---
    header = f"📘 Q{idx+1}/{total}\n\n"
    wrapped_q = (question[:3800] + "..." if len(question) > 3800 else question)
    txt = header + wrapped_q + "\n\n"

   # --- Thêm các lựa chọn ---
    letters = ["A", "B", "C", "D"]
    for i, opt in enumerate(options):
        clean_opt = opt.strip().replace("\n", " ")
        if len(clean_opt) > 300:
            clean_opt = clean_opt[:300] + "..."
        txt += f"{letters[i]}) {clean_opt}\n"

    # --- Nút chọn đáp án (2 hàng) ---
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("A", callback_data="ans:A"),
         InlineKeyboardButton("B", callback_data="ans:B")],
        [InlineKeyboardButton("C", callback_data="ans:C"),
         InlineKeyboardButton("D", callback_data="ans:D")]
    ])

    # --- Gửi hoặc chỉnh sửa tin nhắn (chỉ 1 lần duy nhất) ---
    if isinstance(update_or_query, Update):
        await safe_reply_message(update_or_query.message, txt, reply_markup=kb)
    else:
        await safe_edit_text(update_or_query, txt, reply_markup=kb)


  
# =========================================================
async def practice_summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show practice results with explanations, reward line, and next-step buttons."""
    st = context.user_data.get("practice")
    if not st:
        return

    lang = st.get("ui_lang", "en")
    total = len(st["items"])
    score = st.get("score", 0)
    ptype = st.get("type", "generic")
    scope = st.get("scope", "free")

    # --- Header ---
    lines = []
    if lang == "ru":
        lines.append(f"Итоги: {score}/{total}")
        lines.append("Ответы и пояснения:")
    else:
        lines.append(f"Summary: {score}/{total}")
        lines.append("Answers and explanations:")

    # --- Item explanations ---
    for it in st["items"]:
        expl = it.get("explain_ru") if lang == "ru" else it.get("explain_en")
        if not expl:
            expl = "(no explanation)"
        lines.append(f"Q{it.get('id', '?')}: {it.get('answer', '')} — {expl}")

    # --- Inline reward text ---
    rate = score / max(total, 1)
    if rate >= 1.0:
        reward_text = "🌟 Perfect! All correct!" if lang != "ru" else "🌟 Отлично! Все правильно!"
    elif rate >= 0.6:
        reward_text = "⭐ Great work!" if lang != "ru" else "⭐ Отличная работа!"
    else:
        reward_text = "👏 Nice try!" if lang != "ru" else "👏 Хорошая попытка!"
    lines.append("")  # add space
    lines.append(reward_text)

    # --- Determine “continue” action ---
    if scope == "vocab":
        again_callback = "vocab:practice"
    elif scope == "grammar":
        again_callback = "grammar:practice"
    elif scope == "reading":
        again_callback = "reading:practice"
    else:
        again_callback = "footer:again"

        # --- Build footer keyboard (simplified & smart back) ---
    again_callback = "footer:again"
    layer = context.user_data.get("menu_layer", "root")
    scope = st.get("scope", "")
    if scope == "vocab_direct":
        back_target = "menu:root"
    elif layer == "exercise":
        back_target = "menu:practice"
    else:
        back_target = "menu:root"

    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("🔁 Again", callback_data=again_callback)],
        [InlineKeyboardButton("⬅️ Back to menu", callback_data=back_target)]
    ])


    # --- Reset nudge if mini-quiz ---
    if st.get("type") == "nudge_quiz":
        reset_nudge(context)

    # --- Send summary + reward together ---
    await safe_reply_message(update.message, trim("\n".join(lines)), reply_markup=kb)

    # --- Log result ---
    await log_event(context, "practice_done", update.effective_user.id, {
        "type": ptype,
        "topic": st.get("topic"),
        "scope": scope,
        "score": score,
        "total": total
    })
   

# =========================================================
# 12) CALLBACK HANDLER

async def on_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    data = q.data or ""
    await q.answer()
    uid = update.effective_user.id
    prefs = get_prefs(uid)
    lang = prefs.get("lang", "en")


    if data == "clear:chat":
        try:
            await clear_chat(update, context)
        except Exception as e:
            logger.warning(f"Callback clear_chat failed: {e}")
            await safe_edit_text(q, "⚠️ Couldn't clear chat history.", reply_markup=main_menu(lang))
        return

    # === MENU ROOT ===
    if data == "menu:root":
        prefs["mode"] = "chat"
        context.user_data.clear()     # 🧹 Xóa mọi state cũ (practice, talk, vocab_bank,…)
        reset_nudge(context)
        msg = "📋 Back to main menu." if lang != "ru" else "Возврат в меню."
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


# === ENTER PRACTICE MENU FROM MAIN MENU ===
    if data == "menu:practice":
        txt = "Choose a practice category:" if lang != "ru" else "Выберите категорию практики:"
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("🧠 Vocabulary", callback_data="practice:vocab_menu")],
            [InlineKeyboardButton("⚙️ Grammar", callback_data="practice:grammar_menu")],
            [InlineKeyboardButton("📖 Reading", callback_data="practice:reading_menu")],
            [InlineKeyboardButton("🏠 Back to menu", callback_data="menu:root")]
        ])
        await safe_edit_text(q, txt, reply_markup=kb)
        await log_event(context, "menu_practice_enter", uid, {})
        return

    # === MAIN PRACTICE MENU ===
    if data == "practice:menu":
        txt = "Choose a practice category:" if lang != "ru" else "Выберите категорию практики:"
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("🧠 Vocabulary", callback_data="practice:vocab_menu")],
            [InlineKeyboardButton("⚙️ Grammar", callback_data="practice:grammar_menu")],
            [InlineKeyboardButton("📖 Reading", callback_data="practice:reading_menu")],
            [InlineKeyboardButton("🏠 Back to menu", callback_data="menu:root")]
        ])
        await safe_edit_text(q, txt, reply_markup=kb)
        return

    if data == "practice:vocab_menu":
        txt = "Choose a vocabulary exercise type:" if lang != "ru" else "Выберите тип словарной практики:"
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔤 Synonyms", callback_data="practice:vocab:synonyms"),
             InlineKeyboardButton("❌ Antonyms", callback_data="practice:vocab:antonyms")],
            [InlineKeyboardButton("📘 Word in Context", callback_data="practice:vocab:context"),
             InlineKeyboardButton("🧩 Word Formation", callback_data="practice:vocab:formation")],
            [InlineKeyboardButton("🪄 Collocations", callback_data="practice:vocab:collocations"),
             InlineKeyboardButton("🌀 Phrasal Verbs", callback_data="practice:vocab:phrasal")],
            [InlineKeyboardButton("🔙 Back", callback_data="practice:menu")]
        ])
        await safe_edit_text(q, txt, reply_markup=kb)
        return

    if data == "practice:grammar_menu":
        txt = "Choose a grammar exercise type:" if lang != "ru" else "Выберите тип грамматической практики:"
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("🧾 Verb Forms", callback_data="practice:grammar:verbs"),
             InlineKeyboardButton("🧹 Error Correction", callback_data="practice:grammar:errors")],
            [InlineKeyboardButton("🔀 Word Order", callback_data="practice:grammar:order"),
             InlineKeyboardButton("⛓ Conditionals", callback_data="practice:grammar:conditionals")],
            [InlineKeyboardButton("🗣 Modal Verbs", callback_data="practice:grammar:modals"),
             InlineKeyboardButton("📚 Mixed Grammar", callback_data="practice:grammar:mixed")],
            [InlineKeyboardButton("🔙 Back", callback_data="practice:menu")]
        ])
        await safe_edit_text(q, txt, reply_markup=kb)
        return

    if data == "practice:reading_menu":
        txt = "Choose a reading exercise type:" if lang != "ru" else "Выберите тип чтения:"
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("🌟 Main Idea", callback_data="practice:reading:mainidea"),
             InlineKeyboardButton("🔍 Details", callback_data="practice:reading:details")],
            [InlineKeyboardButton("💭 Inference", callback_data="practice:reading:inference"),
             InlineKeyboardButton("🧠 Vocabulary in Context", callback_data="practice:reading:vocabcontext")],
            [InlineKeyboardButton("✏️ Cloze Passage", callback_data="practice:reading:cloze")],
            [InlineKeyboardButton("🔙 Back", callback_data="practice:menu")]
        ])
        await safe_edit_text(q, txt, reply_markup=kb)
        return


    # === PRACTICE TYPE HANDLER ===
    if data.startswith("practice:vocab:") or data.startswith("practice:grammar:") or data.startswith("practice:reading:"):
        try:
            _, group, flavor = data.split(":")
        except ValueError:
            return await safe_edit_text(
                q,
                "⚠️ Invalid exercise type.",
                reply_markup=main_menu(lang)
            )

        prefs = get_prefs(uid)
        lang = prefs.get("lang", "en")
        level = prefs.get("cefr", "A2")

        await safe_edit_text(
            q,
            f"📝 Generating {group.capitalize()} - {flavor.capitalize()} practice... Please wait."
        )

        # 🧩 Lấy topic hoặc đoạn text gần nhất (nếu có)
        topic_or_text = context.user_data.get("last_passage", "general English")

        # 🧠 Map nhóm + flavor thành flavor_key chuẩn cho build_mcq
        flavor_key = f"{group}_{flavor}"

        try:
            items = await build_mcq(topic_or_text, lang, level, flavor=flavor_key)
        except Exception as e:
            logger.warning(f"build_mcq error ({flavor_key}): {e}")
            return await safe_edit_text(
                q,
                "❌ Failed to create practice questions. Try again later.",
                reply_markup=main_menu(lang)
            )

        if not items:
            return await safe_edit_text(
                q,
                "⚠️ No questions generated.",
                reply_markup=main_menu(lang)
            )

        # 🔍 Lọc trùng câu hỏi nếu có
        seen = set()
        unique_items = []
        for qu in items:
            q_text = qu.get("question", "").strip().lower()
            if q_text and q_text not in seen:
                seen.add(q_text)
                unique_items.append(qu)

        # 🔢 Gán lại ID theo thứ tự
        for i, qu in enumerate(unique_items, start=1):
            qu["id"] = i

        # 🎯 Chỉ giữ tối đa 5 câu hỏi
        items = unique_items[:5]

        # 💾 Lưu trạng thái bài tập
        context.user_data["practice"] = {
            "type": "practice",
            "scope": group,
            "flavor": flavor_key,
            "items": items,
            "idx": 0,
            "score": 0,
            "ui_lang": lang
        }

        # 📍 Đánh dấu đang ở layer bài tập chính thức
        context.user_data["menu_layer"] = "exercise"

        # 🚀 Gửi câu hỏi đầu tiên
        await send_practice_item(update.callback_query, context)
        await log_event(context, "practice_start", uid, {"group": group, "flavor": flavor})
        return



    # === VOCAB PRACTICE ===
       # === VOCABULARY QUICK QUIZ (Practice this word) ===
    if data == "vocab:practice":
        word = context.user_data.get("last_word", "").strip()
        if not word:
            return await safe_edit_text(
                q,
                "Please define a word first.",
                reply_markup=main_menu(lang)
            )

        # 🧩 Sinh 3 câu hỏi nhỏ về từ vừa tra
        flavors = ["vocab_synonyms", "vocab_antonyms", "vocab_context"]
        items = await build_mcq(word, lang, prefs["cefr"], flavor=flavor)
        items = items[:3] 

        # 🔍 Lọc trùng câu hỏi nếu có
        seen = set()
        unique_items = []
        for qu in all_items:
            text = qu.get("question", "").strip().lower()
            if text and text not in seen:
                seen.add(text)
                unique_items.append(qu)

        # 🔢 Gán lại ID theo thứ tự
        for i, qu in enumerate(unique_items, start=1):
            qu["id"] = i

        # 🎯 Giới hạn 3 câu hỏi
        items = unique_items[:3]

        if not items:
            return await safe_edit_text(
                q,
                "⚠️ No quiz available.",
                reply_markup=main_menu(lang)
            )

        # 💾 Lưu trạng thái quiz
        context.user_data["practice"] = {
            "type": "vocab",
            "topic": word,
            "items": items,
            "idx": 0,
            "score": 0,
            "ui_lang": lang,
            "scope": "vocab_direct"
        }

        # 📍 Đánh dấu đang ở layer bài tập độc lập
        context.user_data["menu_layer"] = "exercise"

        # 🚀 Gửi câu hỏi đầu tiên
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


        # === GRAMMAR PRACTICE (with retry & summary footer) ===
    if data == "grammar:practice":
        topic = context.user_data.get("last_grammar_topic", "Present Simple")
        if not topic:
            return await safe_edit_text(q, "No grammar topic found.", reply_markup=main_menu(lang))

        # 🔹 Sinh 3 câu hỏi (mỗi loại 1 câu)
        flavors = ["grammar_verbs", "grammar_errors", "grammar_order"]
        all_items = []
        for f in flavors:
            sub = await build_mcq(topic, lang, prefs["cefr"], flavor=f)
            all_items.extend(sub[:1])
        items = all_items[:3]

        # Sau khi tạo all_items
        seen = set()
        unique_items = []
        for q in all_items:
            q_text = q.get("question", "").strip().lower()
            if q_text and q_text not in seen:
                seen.add(q_text)
                unique_items.append(q)
        items = unique_items[:3]

        if not items:
            return await safe_edit_text(q, "⚠️ No questions found.", reply_markup=main_menu(lang))

        # 🔹 Lưu trạng thái luyện tập
        context.user_data["practice"] = {
            "type": "grammar",
            "topic": topic,
            "items": items,
            "idx": 0,
            "score": 0,
            "ui_lang": lang,
            "scope": "grammar",
            "retry": False
        }
        context.user_data["menu_layer"] = "exercise"

        # 🔹 Gửi câu hỏi đầu tiên
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
        passage = (context.user_data.get("last_passage") or "").strip()
        if not passage:
            return await safe_edit_text(
                q,
                "⚠️ No passage found. Please send or generate a text first."
                if lang != "ru" else "⚠️ Нет текста. Сначала отправь или сгенерируй текст.",
                reply_markup=main_menu(lang)
            )

        await safe_edit_text(q, "🔎 Creating gloss version, please wait...")
        try:
            glossed = await build_reading_gloss(passage, lang, translate_mode=True)
        except Exception as e:
            logger.warning(f"Gloss build failed: {e}")
            return await safe_edit_text(
                q,
                "❌ Failed to generate gloss. Try again or shorten the text."
                if lang != "ru" else "❌ Не удалось создать глоссу. Попробуй снова.",
                reply_markup=main_menu(lang)
            )

        chunks = [glossed[i:i+3500] for i in range(0, len(glossed), 3500)]
        for i, chunk in enumerate(chunks):
            header = f"📘 Glossed text (part {i+1}/{len(chunks)}):\n\n" if len(chunks) > 1 else "📘 Glossed text:\n\n"
            await safe_reply_message(update.callback_query.message, trim(header + chunk))

        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("📝 Practice this text", callback_data="reading:practice")],
            [InlineKeyboardButton("🏠 Back to menu", callback_data="menu:root")]
        ])
        await safe_reply_message(update.callback_query.message, "—", reply_markup=kb)
        await log_event(context, "reading_gloss_done", uid, {"chars": len(passage)})
        return


    # === READING GLOSS (from OCR image) ===
    if data == "reading:gloss_from_image":
        text = context.user_data.get("image_text", "")
        if not text:
            return await safe_edit_text(q, "No text found from image.")
        glossed = await build_reading_gloss(passage, lang, translate_mode=True)
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Menu", callback_data="menu:root")]])
        await safe_edit_text(q, trim(glossed), reply_markup=kb)
        await log_event(context, "image_gloss", uid, {"chars": len(text)})
        return


    # === READING PRACTICE ===
    if data == "reading:practice":
        passage = (context.user_data.get("last_passage") or "").strip()
        topic = context.user_data.get("reading_topic", "reading")

        if not passage:
            return await safe_edit_text(q, "⚠️ No passage found.", reply_markup=main_menu(lang))

        # Gửi lại đoạn passage cho học sinh đọc
        await safe_edit_text(q, f"📖 Text:\n\n{trim(passage[:1800])}")
        await asyncio.sleep(1)

        # Sinh 5 câu hỏi chi tiết dựa theo đoạn đọc
        items = await build_mcq(passage, lang, prefs["cefr"], flavor="reading_detail")
        items = items[:5]

        if not items:
            return await safe_reply_message(
                update.callback_query.message,
                "⚠️ Could not generate reading questions.",
                reply_markup=main_menu(lang)
            )

        context.user_data["practice"] = {
            "type": "reading", "topic": topic, "items": items,
            "idx": 0, "score": 0, "ui_lang": lang, "scope": "reading"
        }
        context.user_data["menu_layer"] = "exercise"

        await send_practice_item(update.callback_query, context)
        await log_event(context, "reading_practice_start", uid, {"topic": topic, "count": len(items)})
        return



# === NUDGE MINI-QUIZ CALLBACK ===
    if data == "nudge:start":
        reset_nudge(context)

        # 📘 Xác định chủ đề và loại bài học gần nhất
        last_practice = context.user_data.get("practice", {})
        vocab_bank = context.user_data.get("vocab_bank", [])
        topic = "general English"
        flavor = "vocab_syn"  # mặc định nếu không xác định được

        if last_practice:
            # Nếu đang học grammar
            if "grammar" in last_practice.get("type", ""):
                topic = last_practice.get("topic", "grammar practice")
                flavor = random.choice(["grammar_verb", "grammar_error", "grammar_order"])
            # Nếu đang học reading
            elif "reading" in last_practice.get("type", ""):
                topic = last_practice.get("topic", "reading comprehension")
                flavor = "reading_detail"
            # Nếu đang học vocab
            elif "vocab" in last_practice.get("type", "") or vocab_bank:
                topic = vocab_bank[-1] if vocab_bank else "vocabulary"
                flavor = random.choice(["vocab_syn", "vocab_cloze", "vocab_ant"])

        await safe_edit_text(q, f"🧠 Starting a quick mini-quiz on {topic}!")

        # 🧩 Sinh 2 câu hỏi mini
        items = await build_mcq(topic, lang, prefs["cefr"], flavor=flavor)
        items = items[:2]

        if not items:
            return await safe_reply_message(
                update.callback_query.message,
                "⚠️ Couldn't build the quiz. Try again later.",
                reply_markup=main_menu(lang)
            )

        context.user_data["practice"] = {
            "type": "nudge_quiz",
            "topic": topic,
            "items": items,
            "idx": 0,
            "score": 0,
            "ui_lang": lang,
            "scope": "mini"
        }

        await send_practice_item(update.callback_query, context)
        await log_event(context, "nudge_quiz_start", uid, {"topic": topic, "flavor": flavor})
        return

    if data == "nudge:skip":
        reset_nudge(context)
        msg = (
            "⏭ Okay, we’ll skip the mini-quiz this time."
            if lang != "ru" else
            "⏭ Хорошо, пропустим мини-викторину."
        )
        await safe_edit_text(q, msg, reply_markup=main_menu(lang))
        await log_event(context, "nudge_skip", uid, {})
        return




        # === ANSWER HANDLING ===
    if data.startswith("ans:"):
        st = context.user_data.get("practice")
        if not st:
            return await safe_edit_text(q, "No active quiz.", reply_markup=main_menu(lang))

        choice = data.split(":")[1]
        idx = st["idx"]
        qitem = st["items"][idx]
        correct = qitem["answer"]
        ui_lang = st.get("ui_lang", "en")

        # --- ✅ Trả lời đúng ---
        if choice == correct:
            st["score"] += 1
            st["retry"] = False
            msg = "✅ Correct!" if ui_lang != "ru" else "✅ Верно!"
            await safe_edit_text(q, msg)
            await asyncio.sleep(1)

            st["idx"] += 1
            if st["idx"] >= len(st["items"]):
                dummy = Update(update.update_id, message=q.message)
                await practice_summary(dummy, context)
            else:
                await send_practice_item(q, context)
            return

        # --- ❌ Sai lần đầu ---
        if not st.get("retry"):
            st["retry"] = True
            msg = "❌ Try again!" if ui_lang != "ru" else "❌ Попробуй ещё раз!"
            # Hiển thị lại câu hỏi hiện tại
            await safe_edit_text(q, msg)
            await asyncio.sleep(0.6)
            return await send_practice_item(q, context) 

        # --- ❌ Sai lần 2 ---
        st["retry"] = False
        msg = (f"❌ Correct answer: {correct}"
               if ui_lang != "ru"
               else f"❌ Правильный ответ: {correct}")
        await safe_edit_text(q, msg)
        await asyncio.sleep(1)

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
            return await safe_edit_text(q, "⚠️ No previous practice found.", reply_markup=main_menu(lang))

        scope = st.get("scope", "free")
        topic = st.get("topic", "English")
        lang = st.get("ui_lang", "en")

        await safe_edit_text(q, "🔁 Creating a new practice set, please wait...")

        try:
            if scope == "vocab":
                # --- Vocabulary practice regeneration ---
                flavors = ["vocab_syn", "vocab_ant", "vocab_cloze"]
                all_items = []
                for f in flavors:
                    sub = await build_mcq(topic, lang, prefs["cefr"], flavor=f)
                    all_items.extend(sub[:1])
                items = all_items[:3]

            elif scope == "grammar":
                # --- Grammar practice regeneration ---
                flavors = ["grammar_verb", "grammar_error", "grammar_order"]
                all_items = []
                for f in flavors:
                    sub = await build_mcq(topic, lang, prefs["cefr"], flavor=f)
                    all_items.extend(sub[:1])
                items = all_items[:3]

            elif scope == "reading":
                # --- Reading practice regeneration ---
                passage = context.user_data.get("last_passage", "")
                items = await build_mcq(passage, lang, prefs["cefr"], flavor="reading_detail")
                items = items[:5]

            else:
                # --- Default generic practice ---
                items = await build_mcq(topic, lang, prefs["cefr"], flavor="generic")
                items = items[:5]

            if not items:
                return await safe_edit_text(q, "⚠️ Could not create new questions.", reply_markup=main_menu(lang))

            # --- Reset state ---
            context.user_data["menu_layer"] = "exercise"
            st.update({"items": items, "idx": 0, "score": 0})
            await send_practice_item(q, context)
            await log_event(context, "practice_regenerated", uid, {"scope": scope, "topic": topic, "count": len(items)})

        except Exception as e:
            logger.warning(f"footer:again error: {e}")
            await safe_edit_text(
                q,
                "❌ Failed to restart practice. Please try again or go back to menu.",
                reply_markup=main_menu(lang)
            )
        return


        # === TALK MODE ENTRY ===
    if data == "menu:talk":
        prefs["mode"] = "talk"
        context.user_data["talk"] = {"topic": "general", "turns": 0}

        # 💬 Lời chào khi vào Talk Mode
        msg = (
            "🗣 Let's practice speaking English!\n"
            "You can start by talking about your school, family, hobbies or future plans.\n"
            "I'll listen and help you with light corrections and useful phrases."
            if lang != "ru" else
            "🗣 Потренируемся говорить по-английски!\n"
            "Ты можешь начать рассказывать о школе, семье, хобби или планах на будущее.\n"
            "Я помогу с исправлениями и полезными фразами."
        )

        # 💡 Gợi ý ngẫu nhiên mẫu câu mở đầu
        talk_tips = [
            "You can start with: 'My name is ...', 'I'm from ...', or 'I like ... because ...'",
            "Try: 'At school, I usually ...', 'My favorite subject is ...'",
            "Try: 'In my free time, I ...', 'My hobby is ...'",
            "Try: 'My family is ...', 'We often ... together.'",
            "You can say: 'In the future, I want to ...', 'I hope to visit ... someday.'"
        ]
        import random
        tip = random.choice(talk_tips)

        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("💬 More ideas", callback_data="talk:more"),
             InlineKeyboardButton("🏠 Back to menu", callback_data="menu:root")]
        ])

        await safe_edit_text(q, msg, reply_markup=kb)
        await safe_reply_message(update.callback_query.message, f"💡 Tip: {tip}")

        await log_event(context, "talk_mode_started", uid, {})
        return

    # === TALK: MORE IDEAS ===
    if data == "talk:more":
        topic = (context.user_data.get("talk") or {}).get("topic", "daily life")
        prompt = (
            f"Give 3 short example sentences or ideas about {topic}. "
            "Each 5–10 words, level A2–B1+, plain English. No markdown."
        )
        msgs = [
            {"role": "system", "content": POLICY_STUDY},
            {"role": "user", "content": prompt}
        ]
        out = await ask_openai(msgs, max_tokens=150)
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("🏠 Menu", callback_data="menu:root")]
        ])
        await safe_edit_text(q, trim(out), reply_markup=kb)
        await log_event(context, "talk_more_ideas", uid, {"topic": topic})
        return


    # --- HELP MENU CALLBACK ---
    if data == "menu:help":
        txt = HELP_TEXT_RU if lang == "ru" else HELP_TEXT_EN
        await safe_edit_text(q, txt, reply_markup=main_menu(lang))
        await log_event(context, "help_open", uid, {})
        return



# =========================================================
# 13) TALK COACH & NUDGE SYSTEM
# =========================================================
async def talk_coach(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """English speaking coach — responds supportively and keeps dialogue going."""
    prefs = get_prefs(update.effective_user.id)
    lang = prefs.get("lang", "en")

    # Chỉ hoạt động khi đang ở Talk Mode
    if prefs.get("mode") != "talk":
        return

    state = context.user_data.get("talk", {"topic": "general", "turns": 0})
    topic = state.get("topic", "general")
    user_text = update.message.text or ""

    # Gọi AI tạo phản hồi
    try:
        reply = await talk_reply(user_text, topic, lang)
    except Exception as e:
        logger.warning(f"talk_reply failed: {e}")
        reply = "Sorry, I didn’t catch that. Could you say it again?"

    # Cập nhật lượt trò chuyện
    state["turns"] = state.get("turns", 0) + 1
    context.user_data["talk"] = state

    # Hiển thị phản hồi + footer
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("💬 More ideas", callback_data="talk:more"),
         InlineKeyboardButton("🏠 Menu", callback_data="menu:root")]
    ])
    await safe_reply_message(update.message, trim(reply), reply_markup=kb)

    # --- Khen nhẹ mỗi 5 lượt ---
    if state["turns"] % 5 == 0:
        encouragement = random.choice([
            "You're doing great! Keep going!",
            "Nice! Could you give me an example?",
            "That’s interesting — tell me more!",
            "Great effort! I like your sentences!"
        ])
        await safe_reply_message(update.message, encouragement)

    # --- Nhắc nhở nhỏ mỗi 10 lượt ---
    if state["turns"] == 10:
        msg_warn = (
            "⚠️ Reminder: I'm an AI tutor and may make mistakes. "
            "Please double-check important information."
            if lang != "ru" else
            "⚠️ Напоминание: я искусственный интеллект и могу ошибаться. "
            "Проверяй важные сведения."
        )
        await safe_reply_message(update.message, msg_warn)

    # --- Nếu đủ 20 lượt, gợi ý kết thúc ---
    if state["turns"] >= 20:
        end_msg = (
            "That was a great talk! Would you like to practice vocabulary or grammar next?"
            if lang != "ru" else
            "Отличная беседа! Хочешь потренироваться со словами или грамматикой?"
        )
        kb_end = InlineKeyboardMarkup([
            [InlineKeyboardButton("📚 Practice", callback_data="menu:practice"),
             InlineKeyboardButton("🏠 Menu", callback_data="menu:root")]
        ])
        await safe_reply_message(update.message, trim(reply))
        await safe_reply_message(update.message, end_msg, reply_markup=kb_end)
        prefs["mode"] = "chat"
        context.user_data.pop("talk", None)
        return

    await log_event(context, "talk_message", update.effective_user.id,
                    {"topic": topic, "turns": state["turns"]})



# --- Nudge mini-quiz ---
def increment_nudge(context):
    """Tăng bộ đếm nudge mỗi khi học sinh hoàn thành 1 lượt học."""
    c = context.user_data.get("nudge", 0) + 1
    context.user_data["nudge"] = c
    return c

def reset_nudge(context):
    """Đặt lại bộ đếm nudge về 0."""
    context.user_data["nudge"] = 0

async def maybe_nudge(update, context, lang):
    """Chỉ gợi ý mini-quiz trong các chế độ học (vocab, grammar, reading)."""
    prefs = get_prefs(update.effective_user.id)
    mode = prefs.get("mode", "chat")
    st = context.user_data.get("practice", {})
    scope = st.get("scope", "")

    # Chỉ kích hoạt trong các mode học
    allowed_scopes = {"vocab", "grammar", "reading", "practice"}

    if mode in {"chat", "talk"}:
        return
    if not any(scope.startswith(a) for a in allowed_scopes):
        return

    c = increment_nudge(context)
    if c >= 4:  # 👉 xuất hiện sau 4 lượt học
        reset_nudge(context)
        msg = (
            "Do a quick 2-question mini-quiz?" if lang != "ru"
            else "Хочешь мини-викторину из 2 вопросов?"
        )
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("▶️ Start", callback_data="nudge:start"),
             InlineKeyboardButton("⏭ Skip", callback_data="nudge:skip")]
        ])
        await safe_reply_message(update.message, msg, reply_markup=kb)
        await log_event(context, "nudge_offer", update.effective_user.id, {})


# =========================================================
# 14) HANDLE MESSAGE (CHAT-FIRST LOGIC)
# =========================================================
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    if not text:
        return

    # ✅ 1. Luôn khởi tạo prefs + lang sớm để tránh lỗi UnboundLocalError
    uid = update.effective_user.id
    prefs = get_prefs(uid)
    lang = prefs.get("lang", "en")
    if lang == "auto":
        lang = detect_lang(text)


# ✅ 2. Xác định intent sớm, trước khi xử lý grammar hint
    t = text.lower()
    intent = "chat"
    if re.search(r"\bread\b|\btext\b|\bwrite\b|\btranslate\b|\bgloss\b", t):
        intent = "reading"
    elif re.search(r"\bdefine\b|\bmeaning of\b", t):
        intent = "vocab"
    elif re.search(r"\bgrammar\b|\btense\b|\bexplain\b|\brule\b", t):
        intent = "grammar"
    elif re.search(r"\btalk\b|\bconversation\b|\bspeak\b", t):
        # Chỉ kích hoạt Talk Mode nếu học sinh đã vào mode talk qua menu
        if prefs.get("mode") == "talk":
            intent = "talk"
        else:
            intent = "chat"   # vẫn coi là chat bình thường
    elif re.search(r"\bquiz\b|\bpractice\b|\bexercise\b", t):
        intent = "practice"



        # --- OUT-OF-SCOPE FILTER (Math, Science, etc.) ---
    out_of_scope = [
        r"\bsolve\b", r"\bcalculate\b", r"\bhow much\b", r"\bformula\b",
        r"\bphysics\b", r"\bchemistry\b", r"\bmath\b", r"\bgeometry\b",
        r"\bequation\b", r"\bsquare root\b", r"\btriangle\b", r"\bvolume\b",
        r"\bmolecule\b", r"\bchemical\b", r"\bderive\b", r"\bproof\b",
        r"\bintegral\b", r"\bderivative\b", r"\blogarithm\b", r"\btheorem\b"
    ]
    for pattern in out_of_scope:
        if re.search(pattern, text.lower()):
            msg = (
                "I'm your English learning assistant. 😊 "
                "I can help with vocabulary, grammar, reading, or speaking — "
                "but not with math or science tasks."
                if lang != "ru" else
                "Я помогаю изучать английский 😊 — словарь, грамматика, чтение, разговор, "
                "но не решаю задачи по математике или физике."
            )
            await safe_reply_message(update.message, msg)
            await log_event(context, "out_of_scope", uid, {"query": text})
            return

    # --- AUTO GRAMMAR HINT  ---
    word_count = len(re.findall(r"[A-Za-z]+", text))

    # ❌ Không bật grammar hint cho text dài hoặc các mode không học ngữ pháp
    if word_count < 40 and prefs.get("mode") not in ("talk", "chat"):
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

	

       # --- TALK CONTEXT CONTINUE ---
    if prefs.get("mode") == "talk" or ("talk" in context.user_data):
        talk_state = context.user_data.get("talk", {"topic": "general", "turns": 0})
        topic = talk_state.get("topic", "daily life")
        user_text = (update.message.text or "").strip()

        # --- Lệnh thoát hội thoại ---
        if user_text.lower() in ("exit", "quit", "menu", "back", "stop", "меню", "выход"):
            context.user_data.pop("talk", None)
            prefs["mode"] = "chat"
            msg = "Exited talk mode. Back to main menu." if lang != "ru" else "Выход из разговора. Главное меню."
            await safe_reply_message(update.message, msg, reply_markup=main_menu(lang))
            await log_event(context, "talk_exit", uid, {})
            return

        # --- Trả lời hội thoại ---
        try:
            reply = await talk_reply(user_text, topic, lang)
        except Exception as e:
            logger.warning(f"talk_reply failed: {e}")
            reply = "Hmm, could you repeat that?"

        talk_state["turns"] += 1
        prefs["mode"] = "talk"  # đảm bảo vẫn ở chế độ hội thoại
        context.user_data["talk"] = talk_state

        # --- Lời khen nhẹ mỗi 5 lượt ---
        if talk_state["turns"] % 5 == 0:
            encouragement = random.choice([
                "You're doing great! Tell me more!",
                "Nice! Could you give an example?",
                "That’s interesting — keep going!",
                "Great effort! Keep speaking English!",
            ])
            await safe_reply_message(update.message, encouragement)

        # --- Nhắc nhở nhẹ sau 10 lượt ---
        if talk_state["turns"] == 10:
            msg_warn = (
                "⚠️ Reminder: I'm an AI tutor and may make mistakes. "
                "Please double-check important information."
                if lang != "ru" else
                "⚠️ Напоминание: я искусственный интеллект и могу ошибаться. "
                "Проверяй важные сведения."
            )
            await safe_reply_message(update.message, msg_warn)


        # --- Nếu trò chuyện đủ dài, gợi ý kết thúc ---
        if talk_state["turns"] >= 20:
            end_msg = ("That was a nice talk! Want to study something next?"
                       if lang != "ru" else "Отличный разговор! Хочешь потренироваться дальше?")
            kb = InlineKeyboardMarkup([
                [InlineKeyboardButton("📚 Practice", callback_data="menu:practice"),
                 InlineKeyboardButton("🏠 Menu", callback_data="menu:root")]
            ])
            await safe_reply_message(update.message, trim(reply))
            await safe_reply_message(update.message, end_msg, reply_markup=kb)
            prefs["mode"] = "chat"
            context.user_data.pop("talk", None)
            return

        # --- Gửi phản hồi bình thường ---
        await safe_reply_message(update.message, trim(reply))
        await log_event(context, "talk_message", uid, {"topic": topic, "turns": talk_state["turns"]})
        return


    # --- GENERAL FILTERS & SETUP ---
    if blocked(text):
        return await safe_reply_message(update.message,
            "⛔ Please keep it school-appropriate. Try an English topic.")


    # GREETING DETECTION
    if re.fullmatch(r"hi|hello|hey|привет|здравствуй", text.lower()):
        msg = ("Hello! I'm your English tutor. Ask me anything about English learning!"
               if lang!="ru" else "Привет! Я твой помощник по английскому. Задай вопрос о языке!")
        return await safe_reply_message(update.message, msg, reply_markup=main_menu(lang))

  
        # --- LONG TEXT SAFEGUARD ---
    word_count = len(re.findall(r"[A-Za-z]+", text))
    if word_count >= 50 and intent == "vocab":
        intent = "chat"  # chuyển về chat để hỏi ý người dùng
        msg = (
            "I see a long text. Would you like me to summarize, gloss, or check grammar?"
            if lang != "ru" else
            "Я вижу длинный текст. Хочешь, я помогу с кратким изложением, глоссой или грамматикой?"
        )
        await safe_reply_message(update.message, msg)
        await log_event(context, "long_text_redirected", uid, {"words": word_count})
        # Không return để bot vẫn có thể phản hồi tiếp


    # --- VOCABULARY ---
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
        await maybe_nudge(update, context, lang)
        return await maybe_nudge(update, context, lang) 


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

   

        # --- READING INTENT ---
    if intent == "reading":
        reset_nudge(context)
        level = prefs["cefr"]
        word_count = len(text.split())
        lower = text.lower()

        # 1️⃣ Nếu học sinh ra lệnh translate/gloss this text → gloss dịch song ngữ
        if re.search(r"\b(translate|gloss)\b", lower):
            passage = re.sub(r"\b(translate|gloss|this text)\b", "", text, flags=re.I).strip()
            if not passage:
                return await safe_reply_message(
                    update.message,
                    "Please include a text after your command."
                    if lang != "ru"
                    else "Пожалуйста, добавь текст после команды."
                )

            context.user_data["last_passage"] = passage
            context.user_data["reading_topic"] = "user_text"

            await safe_reply_message(update.message, "🔎 Translating and glossing your text, please wait...")

            try:
                glossed = await build_reading_gloss(passage, lang, translate_mode=True)
            except Exception as e:
                logger.warning(f"Gloss error: {e}")
                return await safe_reply_message(
                    update.message,
                    "❌ Could not generate gloss. Try again or shorten the text."
                    if lang != "ru"
                    else "❌ Не удалось создать глоссу. Попробуй снова."
                )

            # Nếu gloss dài, chia nhỏ để gửi từng phần
            chunks = [glossed[i:i+3500] for i in range(0, len(glossed), 3500)]
            for i, chunk in enumerate(chunks):
                header = (
                    f"📘 Translated gloss (part {i+1}/{len(chunks)}):\n\n"
                    if len(chunks) > 1
                    else "📘 Translated gloss:\n\n"
                )
                await safe_reply_message(update.message, trim(header + chunk))

            kb = InlineKeyboardMarkup([
                [InlineKeyboardButton("📝 Practice this text", callback_data="reading:practice")],
                [InlineKeyboardButton("🏠 Back to menu", callback_data="menu:root")]
            ])
            await safe_reply_message(update.message, "—", reply_markup=kb)
            await log_event(context, "reading_translate_gloss", uid, {"chars": len(passage)})
            return

        # 2️⃣ Nếu học sinh gửi text dài nhưng KHÔNG ra lệnh gì rõ ràng
        if word_count >= 50 and not re.search(r"\b(write|translate|gloss)\b", lower):
            # Hỏi lại xem học sinh muốn làm gì với đoạn văn
            msg = (
                "I see a long text. Would you like me to summarize, check grammar, or explain it?"
                if lang != "ru"
                else "Я вижу длинный текст. Хочешь, я помогу с кратким изложением, грамматикой или объяснением?"
            )
            await safe_reply_message(update.message, msg)
            await log_event(context, "reading_unclear_text", uid, {"words": word_count})
            # ❗ Không return — cho phép Chat Mode phản hồi tự nhiên sau đó

        # 3️⃣ Nếu học sinh chỉ gửi topic ngắn (ví dụ: 'animals', 'friendship')
        topic = text.strip().capitalize()
        passage = await build_reading_passage(topic, prefs)
        context.user_data["last_passage"] = passage
        context.user_data["reading_topic"] = topic

        await safe_reply_message(
            update.message,
            trim(passage),
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📘 Gloss this text", callback_data="reading:gloss"),
                 InlineKeyboardButton("📝 Practice this text", callback_data="reading:practice")],
                [InlineKeyboardButton("🏠 Menu", callback_data="menu:root")]
            ])
        )
        await log_event(context, "reading_passage", uid, {"topic": topic, "mode": "auto_topic"})
        return await maybe_nudge(update, context, lang)



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



   
    # --- DEFAULT CHAT MODE ---
    if intent == "chat":
        word_count = len(re.findall(r"[A-Za-z]+", text))
 
        # 🧩 Nếu học sinh gửi đoạn văn dài, gợi ý hành động
        if word_count >= 60 and not re.search(r"\b(translate|gloss|summarize|explain|correct|question)\b", text, re.I):
            msg = (
                "I see a long text. Would you like me to summarize, check grammar, or explain it?"
                if lang != "ru" else
                "Я вижу длинный текст. Хочешь, я помогу с кратким изложением, грамматикой или объяснением?"
            )
            await safe_reply_message(update.message, msg)
            await log_event(context, "long_text_detected", update.effective_user.id, {"words": word_count})
            # ❗ Không return — để bot vẫn phản hồi như chat bình thường

        # 🧠 Chat tự nhiên
        msgs = [
            {"role": "system", "content": POLICY_CHAT},
            {"role": "user", "content": text}
        ]
        reply = await ask_openai(msgs, max_tokens=350)

        # 💬 Chỉ phản hồi text — không thêm nút menu
        await safe_reply_message(update.message, trim(reply))

        await log_event(context, "chat_message", uid, {"chars": len(text)})
        await maybe_nudge(update, context, lang)
        return



# =========================================================
   

    # 2️⃣ Smart Grammar detector for textbook-style exercises
    if re.search(r"\b(fill in|underline|choose|complete|correct)\b", text.lower()):
        msg = ("It looks like a grammar exercise. "
               "I can guide you step by step instead of giving direct answers. "
               "What grammar topic is this about?")
        await safe_reply_message(update.message, msg)
        await log_event(context, "textbook_ex_detected", update.effective_user.id, {"text": text[:80]})
        return


 
    reply = await ask_openai(msgs, max_tokens=350)
    await safe_reply_message(update.message, trim(reply), reply_markup=main_menu(lang))
    await log_event(context, "chat_message", uid, {"chars": len(text)})
    await maybe_nudge(update, context, lang)

           
# --- Nhắc nhở định kỳ trong chế độ chat ---
    chat_turns = context.user_data.get("chat_turns", 0) + 1
    context.user_data["chat_turns"] = chat_turns

    if chat_turns == 10:
        warn_msg = (
            "⚠️ Reminder: I'm an AI tutor and may make mistakes. "
            "Please double-check important information."
            if lang != "ru" else
            "⚠️ Напоминание: я искусственный интеллект и могу ошибаться. "
            "Проверяй важные сведения."
        )
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("🏠 Menu", callback_data="menu:root")]
        ])
        await safe_reply_message(update.message, warn_msg, reply_markup=kb)
        context.user_data["chat_turns"] = 0  # reset sau khi nhắc


   # --- DEFAULT CHAT ---
    msgs = [
        {"role": "system", "content": POLICY_CHAT},
        {"role": "user", "content": text}
    ]

# =========================================================
 # HANDLE IMAGE INPUT

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle photo messages: detect if it's text, grammar exercise, or unrelated."""
    photo = update.message.photo[-1]
    file = await photo.get_file()
    text = await extract_text_from_image(file)

    if not text:
        return await safe_reply_message(update.message, "I couldn't read the image clearly. Try again.")

    # Basic classification

 # =========================================================
 #SMART GRAMMAR HINT FROM IMAGE

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
import pytesseract
from PIL import Image
import io

async def extract_text_from_image(file_obj):
    """Extract English text from uploaded image using pytesseract."""
    try:
        bio = io.BytesIO()
        await file_obj.download_to_memory(out=bio)
        bio.seek(0)
        image = Image.open(bio)
        image = image.convert("L")  # grayscale improves OCR accuracy
        text = pytesseract.image_to_string(image, lang="eng")
        return text.strip()
    except Exception as e:
        logger.warning(f"OCR failed: {e}")
        return ""

# =========================================================
# 15) FLASK HEALTHCHECK & MAIN ENTRYPOINT
# =========================================================
app = Flask(__name__)

@app.get("/")
def health():
    return "✅ AI English Tutor v2 is alive", 200

def start_flask():
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)

import asyncio
def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # --- Add handlers ---
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("menu", handle_menu))
    application.add_handler(CommandHandler("help", help_cmd))
    application.add_handler(CommandHandler("clear", clear_chat))
    application.add_handler(CallbackQueryHandler(on_cb))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))
    application.add_error_handler(on_error)

    # --- 🔹 Gọi hàm on_startup để xóa webhook cũ ---
    asyncio.run(on_startup(application))

    # --- 🔹 Chạy Flask song song trong thread riêng (Render friendly) ---
    threading.Thread(target=start_flask, daemon=True).start()

    logger.info("🚀 Bot starting: English Tutor v2 ready for class!")
    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == "__main__":
    main()
