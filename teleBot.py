# =========================================================
# teleBot_v2_full.py
# =========================================================
# 0) IMPORTS & GLOBAL SETUP
# =========================================================
import os, re, json, time, hmac, hashlib, logging, asyncio, uuid, difflib, random

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

import threading, requests

def keep_alive():
    while True:
        try:
            requests.get("https://telegram-english-bot-1.onrender.com")
        except Exception:
            pass
        time.sleep(300)

def remove_markdown(text: str) -> str:
    if not isinstance(text, str):
        return text
    # Loại bỏ **bold**
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    # Loại bỏ *italic*
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    # Loại bỏ __bold__
    text = re.sub(r"__(.*?)__", r"\1", text)
    # Loại bỏ _italic_
    text = re.sub(r"_(.*?)_", r"\1", text)
    # Loại bỏ inline code `...`
    text = re.sub(r"`(.*?)`", r"\1", text)
    # Loại bỏ link dạng [title](url)
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    return text
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
    """
    Gọi 1 lần khi bot khởi động.
    Dùng để xóa webhook cũ (nếu còn) để tránh lỗi 409 conflict.
    """
    try:
        await app.bot.delete_webhook(drop_pending_updates=True)
        logger.info("Webhook deleted, bot ready for polling.")
    except Exception as e:
        logger.warning(f"on_startup failed: {e}")

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
    " If the student gives a list of words or phrases, you can help by creating short sentences, questions, or a short paragraph using them."
    " Always keep vocabulary and grammar at A2–B1+ level and explain briefly if needed."
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
            [InlineKeyboardButton("🪞 Рефлексия", callback_data="menu:reflect"),
             InlineKeyboardButton("❓ Помощь", callback_data="menu:help")]
        ]
    else:
        kb = [
            [InlineKeyboardButton("💬 Talk", callback_data="menu:talk"),
             InlineKeyboardButton("📝 Practice", callback_data="menu:practice")],
            [InlineKeyboardButton("🏫 Grade", callback_data="menu:grade"),
             InlineKeyboardButton("🌐 Language", callback_data="menu:lang")],
            [InlineKeyboardButton("🪞 Reflect", callback_data="menu:reflect"),
             InlineKeyboardButton("❓ Help", callback_data="menu:help")]
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
            raw = resp.choices[0].message.content
            return remove_markdown(raw)
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


async def build_reading_passage(topic, prefs):
    """
    Generate a short reading passage (A2–B1) based on topic.
    Used in Reading Practice to create context for comprehension questions.
    """
    lang = prefs.get("lang", "en")
    level = prefs.get("cefr", "A2")

    # 🧠 Prompt hướng dẫn model tạo đoạn text
    msgs = [
        {"role": "system", "content": (
            "You are an English teacher for middle school students (CEFR A2–B1). "
            "Write short, interesting reading passages about everyday topics."
        )},
        {"role": "user", "content": (
            f"Write a {level} level English reading passage of about 100–120 words. "
            f"Topic: {topic}. "
            "Use clear sentences, familiar vocabulary, and one main idea. "
            "Do not include questions or bullet points."
        )}
    ]

    try:
        raw = await ask_openai(msgs, max_tokens=350)
        passage = raw.strip()
        if len(passage) < 60:
            # fallback nếu đoạn quá ngắn
            msgs[1]["content"] = (
                f"Write a simple short story about {topic} (A2–B1, 100 words). "
                "Include details students can answer questions about later."
            )
            raw = await ask_openai(msgs, max_tokens=350)
            passage = raw.strip()
        return passage
    except Exception as e:
        logger.warning(f"build_reading_passage error for topic={topic}: {e}")
        return ""


 # =========================
async def build_mcq(topic_or_text: str, ui_lang: str, level: str, flavor: str = "generic"):
    """
    Create a 5-question MCQ set based on grade, topic, and exercise flavor.
    Supports: vocab_*, grammar_*, reading_*.
    """
    # =========================
    # 1️⃣ Define task map
    # =========================
    task_map = {
        # =====================
        # --- VOCABULARY TYPES ---
        # =====================

        "vocab_synonyms": (
            "Write 5 multiple-choice questions (A–D) testing SYNONYMS (similar meaning words). "
            "Each question should:\n"
            "• Ask: 'Which word is closest in meaning to ...?'\n"
            "• Include a short example sentence if needed.\n"
            "• Provide 4 clear options (A–D), one correct synonym and three distractors.\n"
            "• Keep vocabulary at CEFR A2–B1 level.\n"
            "• Add a short explanation (≤20 words)."
        ),

        "vocab_antonyms": (
            "Write 5 multiple-choice questions (A–D) testing ANTONYMS (opposite meaning words). "
            "Each question should:\n"
            "• Ask: 'Which word has the opposite meaning to ...?'\n"
            "• Include a short example sentence when possible.\n"
            "• Provide 4 short options (A–D), one correct antonym and three distractors.\n"
            "• Keep vocabulary suitable for CEFR A2–B1 students.\n"
            "• Include a 1-sentence explanation."
        ),

        "vocab_context": (
            "Write 5 MCQs asking students to choose the correct word IN CONTEXT. "
            "Each question should:\n"
            "• Include a short sentence with a blank '____'.\n"
            "• Provide 4 possible words (A–D), one that fits grammatically and logically.\n"
            "• Avoid using overly advanced or idiomatic phrases.\n"
            "• Add a short explanation of why the correct word fits best."
        ),

        "vocab_formation": (
            "Write 5 MCQs testing WORD FORMATION (noun, verb, adjective, adverb forms). "
            "Each question should:\n"
            "• Include a sentence with a blank and a base word in parentheses, e.g. 'She was very ____ (beauty)'.\n"
            "• Ask which form fits grammatically.\n"
            "• Provide 4 choices (A–D) with different word forms.\n"
            "• Include short explanation (≤20 words)."
        ),

        "vocab_collocations": (
            "Write 5 MCQs testing COLLOCATIONS (natural word combinations). "
            "Each question should:\n"
            "• Contain a sentence with a missing word, e.g. 'He made a ____ mistake.'\n"
            "• Provide 4 possible collocations (A–D), one correct and three wrong.\n"
            "• Keep words common for A2–B1 learners.\n"
            "• Add a short explanation."
        ),

        "vocab_phrasal": (
            "Write 5 MCQs testing PHRASAL VERBS. "
            "Each question should:\n"
            "• Use a short natural sentence with a blank.\n"
            "• Provide 4 phrasal verbs (A–D) formed from the same base verb (e.g. take off, take up, take in, take over).\n"
            "• Include one correct and three distractors.\n"
            "• Add a short explanation (≤20 words)."
        ),


        # =====================
        # --- GRAMMAR TYPES ---
        # =====================

        "grammar_verbs": (
            "Write 5 multiple-choice questions (A–D) testing correct verb forms. "
            "Each question should:\n"
            "• Have one blank space for the verb.\n"
            "• Provide 4 verb forms (A–D) covering tenses and aspects (present, past, perfect, continuous).\n"
            "• Ensure natural grammar for CEFR A2–B1.\n"
            "• Add a short explanation (≤20 words)."
        ),

        "grammar_errors": (
            "Write 5 MCQs testing grammar error correction. "
            "Each question should:\n"
            "• Show one incorrect sentence.\n"
            "• Ask: 'Which is the correct sentence?'\n"
            "• Provide 4 corrected options (A–D).\n"
            "• Use grammar points such as subject-verb agreement, articles, or prepositions.\n"
            "• Include a brief explanation of the correction."
        ),

        "grammar_order": (
            "Write 5 MCQs that test correct English word order. "
            "Each question should:\n"
            "• Present a jumbled sentence (e.g. 'every / plays / Saturday / she / soccer').\n"
            "• Ask: 'Choose the correct order.'\n"
            "• Provide 4 possible orders (A–D), only one correct.\n"
            "• Keep sentences short and clear for A2–B1.\n"
            "• Add a brief explanation."
        ),

        "grammar_conditionals": (
            "Write 5 MCQs testing CONDITIONAL SENTENCES (Type 0–3). "
            "Each question should:\n"
            "• Include one conditional sentence with a blank.\n"
            "• Provide 4 choices (A–D) — one correct form of the verb or clause.\n"
            "• Include a short explanation of the grammar rule."
        ),

        "grammar_modals": (
            "Write 5 MCQs testing MODAL VERBS (can, must, should, may, might, etc.). "
            "Each question should:\n"
            "• Ask about correct meaning or usage in context.\n"
            "• Provide 4 options (A–D), one correct.\n"
            "• Include short explanation (≤20 words)."
        ),

        "grammar_mixed": (
            "Write 5 mixed grammar MCQs combining different grammar areas (tenses, prepositions, articles, modals). "
            "Each question should:\n"
            "• Be one clear sentence with a blank.\n"
            "• Provide 4 options (A–D), one correct.\n"
            "• Add a short explanation of the grammar point."
        ),


        # =====================
        # --- READING TYPES ---
        # =====================

        "reading_mainidea": (
            "Write 5 READING COMPREHENSION questions testing MAIN IDEA. "
            "Each question should:\n"
            "• Focus on the general meaning, topic, or purpose of the passage.\n"
            "• Avoid factual or detail-based questions.\n"
            "• Provide 4 options (A–D) and a short explanation."
        ),

        "reading_details": (
            "Write 5 READING COMPREHENSION questions testing DETAILS or FACTS. "
            "Each question should:\n"
            "• Ask about specific information mentioned in the passage.\n"
            "• Avoid trivial numbers or dates.\n"
            "• Provide 4 options (A–D), one correct, with a short explanation."
        ),

        "reading_inference": (
            "Write 5 READING COMPREHENSION questions testing INFERENCE. "
            "Each question should:\n"
            "• Require students to understand meaning that is not directly stated.\n"
            "• Provide 4 options (A–D) with one logical answer.\n"
            "• Include a short explanation."
        ),

        "reading_vocabcontext": (
            "Write 5 READING COMPREHENSION questions testing VOCABULARY IN CONTEXT. "
            "Each question should:\n"
            "• Quote a short sentence from the passage.\n"
            "• Ask: 'What does the word ___ mean here?'\n"
            "• Provide 4 meanings (A–D), one correct.\n"
            "• Include a short explanation."
        ),

        "reading_cloze": (
            "Write 5 CLOZE TEST questions (fill in the blanks) based on the passage. "
            "Each question should:\n"
            "• Omit one key word.\n"
            "• Provide 4 possible options (A–D).\n"
            "• Indicate one correct answer."
        ),


        # =====================
        # --- FALLBACK / GENERIC ---
        # =====================

        "generic": (
            "Write 5 general English MCQs (A2–B1+). "
            "Mix grammar, vocabulary, and comprehension. "
            "Each question should have 4 options and one correct answer with a short explanation."
        ),
    }
    # =========================
    # 2️⃣ Select task prompt
    # =========================
    task = task_map.get(flavor, task_map["generic"])

    # Difficulty tag
    if level in ("A2", "A2+"):
        diff_note = "Use simple sentences and everyday words."
    elif level == "B1+":
        diff_note = "Include 1–2 slightly more advanced structures or idioms."
    else:
        diff_note = "Keep within A2–B1 school-level range."

 
        # =========================
    # 3️⃣ Construct model prompt (TỐI ƯU TOKEN)
    # =========================

    prompt = f"""
Generate exactly 5 English MCQs (A–D).

Output STRICT JSON only, in this exact structure:
{{
  "questions": [
    {{
      "id": 1,
      "question": "text",
      "options": ["A","B","C","D"],
      "answer": "A",
      "explain_en": "short"
    }}
  ]
}}

Rules:
- Output ONLY JSON. No markdown.
- No explanations outside JSON.
- Each explanation <= 20 words.
- Level: {level}
- Focus: {flavor}
- Topic: {topic_or_text}
- Language: {"Russian" if ui_lang=='ru' else "English"}.
"""

    msgs = [
        {"role": "system", "content": "You must output STRICT JSON only."},
        {"role": "user", "content": prompt}
    ]

    logger.info(f"🧠 MCQ | {flavor} | Level={level} | Lang={ui_lang}")



    # =========================
    # 4️⃣ Request from model
    # =========================
    raw = await ask_openai(msgs, max_tokens=450)
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
            # attempt to detect correct option from explanation
            expl = q.get("explain_en","") + q.get("question","")
            for letter, opt in zip(["A","B","C","D"], opts):
                if opt.lower() in expl.lower():
                    ans = letter
                    break
            if ans not in ["A","B","C","D"]:
                ans = random.choice(["A","B","C","D"])
        valid.append({
                "id": q.get("id", 0),
                "question": q.get("question", ""),
                "options": opts,
                "answer": ans,
                "explain_en": q.get("explain_en", ""),
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
    scope = st.get("scope", "")
    lang = st.get("ui_lang", "en")

    # --- Nếu là bài Reading, hiển thị lại đoạn passage trước câu hỏi ---
    passage_text = ""
    if scope == "reading":
        passage = context.user_data.get("last_passage", "")
        if passage:
            passage_preview = trim(passage[:800])
            passage_text = f"📖 Passage:\n{passage_preview}\n\n"

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

    # --- Shuffle options (random vị trí đáp án đúng) ---
    correct_answer = q.get("answer", "A").strip().upper()
    letters = ["A", "B", "C", "D"]
    if len(options) == 4:
        # xác định đáp án đúng trước khi xáo
        correct_index = letters.index(correct_answer) if correct_answer in letters else 0
        correct_text = options[correct_index] if correct_index < len(options) else options[0]
        random.shuffle(options)
        correct_answer = letters[options.index(correct_text)]
        q["options"] = options
        q["answer"] = correct_answer

    # --- Gắn header câu hỏi ---
    header = f"{passage_text}📘 Q{idx + 1}/{total}\n\n"
    wrapped_q = question[:3800] + "..." if len(question) > 3800 else question
    txt = header + wrapped_q + "\n\n"

    # --- Thêm các lựa chọn (đã shuffle) ---
    for i, opt in enumerate(options):
        label = chr(65 + i)  # 65 = 'A'

        # ❗❗ FIX DUPLICATE LABELS (A) A) ...)
        clean_opt = opt.strip()
        clean_opt = re.sub(r"^[A-D][\)\.\:\-\s]+", "", clean_opt)  # Xoá nhãn do model thêm

        clean_opt = clean_opt.replace("\n", " ")
        if len(clean_opt) > 300:
            clean_opt = clean_opt[:300] + "..."

        txt += f"{label}) {clean_opt}\n"

    # --- Nút chọn đáp án (2 hàng, gọn gàng) ---
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("A", callback_data="ans:A"),
         InlineKeyboardButton("B", callback_data="ans:B")],
        [InlineKeyboardButton("C", callback_data="ans:C"),
         InlineKeyboardButton("D", callback_data="ans:D")]
    ])

    # --- Gửi hoặc chỉnh sửa tin nhắn ---
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
    # --- Determine “continue” action ---
    if scope in ("vocab", "vocab_direct"):
        again_callback = "vocab:quiz"
    elif scope == "grammar":
        again_callback = "grammar:quiz"
    elif scope == "reading":
        again_callback = "reading:quiz"
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
# 11.5 REFLECT MODE — 7-Question Self-Assessment (FIXED)
# =========================================================

# ---------- 1) QUESTION DATA ----------
REFLECT_Q = {
    "en": [
        {"id": 1, "text": "1. Did you review the material before class?",
         "options": ["Yes", "No"]},

        {"id": 2, "text": "2. Did you check your mistakes after finishing your tasks?",
         "options": ["Yes, using the chatbot", "Yes, by myself", "No"]},

        {"id": 3, "text": "3. Which AI tool did you use most often?",
         "options": ["Chatbot", "Video", "Quiz", "I didn't use anything"]},

        {"id": 4, "text": "4. Was this topic clear to you?",
         "options": ["Yes, completely", "Partly", "No, it was difficult"]},

        {"id": 5, "text": "5. Rate your responsibility for learning this week (1–5):",
         "options": [
             "1 — I did not feel responsible",
             "2 — I felt a little responsible",
             "3 — I felt somewhat responsible",
             "4 — I felt quite responsible",
             "5 — I felt very responsible"
         ]},

        {"id": 6, "text": "6. What went best for you this week?", "options": []},
        {"id": 7, "text": "7. What was the most difficult and why?", "options": []},
    ],

    "ru": [
        {"id": 1, "text": "1. Вы пересматривали материал перед уроком?",
         "options": ["Да", "Нет"]},

        {"id": 2, "text": "2. Вы проверяли свои ошибки после выполнения задания?",
         "options": ["Да, с помощью чат-бота", "Да, самостоятельно", "Нет"]},

        {"id": 3, "text": "3. Какой ИИ-инструмент вы использовали чаще всего?",
         "options": ["Чат-бот", "Видео", "Викторина", "Ничего не использовал(а)"]},

        {"id": 4, "text": "4. Был ли вам понятен материал этой темы?",
         "options": ["Да, полностью", "Частично", "Нет, было сложно"]},

        {"id": 5, "text": "5. Оцените свою ответственность за обучение (1–5):",
         "options": [
             "1 — совсем не чувствовал(а)",
             "2 — немного чувствовал(а)",
             "3 — средний уровень",
             "4 — довольно сильно чувствовал(а)",
             "5 — очень сильно чувствовал(а)"
         ]},

        {"id": 6, "text": "6. Что у вас получилось лучше всего на этой неделе?", "options": []},
        {"id": 7, "text": "7. Что было самым трудным и почему?", "options": []},
    ]
}


# ---------- 2) KEYBOARD BUILDER ----------
def reflect_keyboard(qid, options):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(opt, callback_data=f"reflect:ans:{qid}:{opt}")]
        for opt in options
    ])


# ---------- 3) START REFLECTION ----------
async def reflect_start(update_or_query, context, lang):
    uid = update_or_query.effective_user.id
    prefs = get_prefs(update_or_query.effective_user.id)
    prefs["mode"] = "reflect"
    
    context.user_data["reflect"] = {"step": 1, "answers": []}
    q = REFLECT_Q[lang][0]
    await send_reflect_question(update_or_query, q)

# ---------- 4) SEND QUESTION ----------
async def send_reflect_question(update_or_query, q):
    if q["options"]:
        kb = reflect_keyboard(q["id"], q["options"])
    else:
        kb = None  # Q6–7 không có nút

    if getattr(update_or_query, "callback_query", None):
        await safe_edit_text(update_or_query.callback_query, q["text"], reply_markup=kb)
    else:
        await safe_reply_message(update_or_query.message, q["text"], reply_markup=kb)


# ---------- 5) HANDLE TEXT ANSWERS (Q6–7) ----------
async def reflect_handle_text(update, context):
    st = context.user_data["reflect"]
    step = st["step"]            # 6 hoặc 7
    lang = get_prefs(update.effective_user.id)["lang"]

    st["answers"].append(update.message.text)

    if step == 7:
        return await reflect_finalize(update, context)

    # next question = step + 1 → index = (step+1)-1
    st["step"] = step + 1
    next_q = REFLECT_Q[lang][st["step"] - 1]
    return await send_reflect_question(update, next_q)


# ---------- 6) HANDLE MULTIPLE CHOICE ANSWERS (Q1–Q5) ----------
async def reflect_handle_choice(update_or_query, context, qid, choice):
    st = context.user_data["reflect"]
    lang = get_prefs(update_or_query.effective_user.id)["lang"]

    st["answers"].append(choice)

    if qid == 5:
        # Sau Q5 → Q6 (text mode)
        st["step"] = 6
        q = REFLECT_Q[lang][5]   # index 5 = Q6
        return await send_reflect_question(update_or_query, q)

    if qid >= 7:
        return await reflect_finalize(update_or_query, context)

    # next step
    st["step"] = qid + 1
    q = REFLECT_Q[lang][st["step"] - 1]
    await send_reflect_question(update_or_query, q)


# ---------- 7) FINALIZE REFLECTION (with AI advice) ----------
async def reflect_finalize(update_or_query, context):
    st = context.user_data.get("reflect")
    if not st:
        return

    answers = st["answers"]

    # --- Validate đủ 7 câu ---
    target_msg = None
    if hasattr(update_or_query, "message") and update_or_query.message:
        target_msg = update_or_query.message
    elif hasattr(update_or_query, "callback_query") and update_or_query.callback_query:
        target_msg = update_or_query.callback_query.message

    if len(answers) < 7:
        if target_msg:
            await safe_reply_message(target_msg, "Reflection incomplete. Please try again.")
        return

    # --- Extract data ---
    lang = get_prefs(update_or_query.effective_user.id)["lang"]

    a1, a2, a3, a4, a5 = answers[:5]     # MCQ answers
    a6 = answers[5]                      # Strengths
    a7 = answers[6]                      # Difficulties

    try:
        score = int(a5)
    except:
        score = 3  # fallback

    # ============================================================
    # 🔥 AI-generated personalized advice
    # ============================================================

    # Prompt xây dựng lời khuyên từ AI
    advice_prompt = (
        f"The student completed a 7-question reflection.\n\n"
        f"1) Reviewed before class: {a1}\n"
        f"2) Checked mistakes: {a2}\n"
        f"3) AI tools used: {a3}\n"
        f"4) Topic clarity: {a4}\n"
        f"5) Responsibility (1–5): {score}\n"
        f"6) Strengths: {a6}\n"
        f"7) Difficulties: {a7}\n\n"
        f"Write a short, warm, motivating advice (2–3 sentences) "
        f"for a middle-school student. "
        f"Use simple { 'English' if lang=='en' else 'Russian' }. "
        f"Be encouraging and practical."
    )

    try:
        advice = await ask_openai([
            {"role": "system", "content": "You are a friendly and supportive school teacher."},
            {"role": "user", "content": advice_prompt}
        ], max_tokens=120)
        advice = advice.strip()
    except:
        # fallback nếu AI không trả lời
        advice = (
            "Keep practicing a little every day — consistent effort helps you grow!"
            if lang == "en" else
            "Продолжай заниматься понемногу каждый день — постоянство принесёт результат!"
        )

    # ============================================================
    # 🔥 Build final result message
    # ============================================================

    if lang == "en":
        txt = (
            f"📝 Your Reflection Results:\n\n"
            f"⭐️ Strengths:\n• {a6}\n\n"
            f"⚠️ Difficulties:\n• {a7}\n\n"
            f"💡 Personalized Advice:\n• {advice}"
        )
    else:
        txt = (
            f"📝 Ваши результаты рефлексии:\n\n"
            f"⭐️ Сильные стороны:\n• {a6}\n\n"
            f"⚠️ Трудности:\n• {a7}\n\n"
            f"💡 Персональная рекомендация:\n• {advice}"
        )

    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("🏠 Menu" if lang=="en" else "🏠 Меню", callback_data="menu:root")]
    ])

    # --- SEND OUTPUT SAFELY ---
    if hasattr(update_or_query, "callback_query") and update_or_query.callback_query:
        await safe_edit_text(update_or_query.callback_query, txt, reply_markup=kb)
    else:
        await safe_reply_message(update_or_query.message, txt, reply_markup=kb)

    # --- LOG EVENT ---
    try:
        await log_event(context, "reflect", update_or_query.effective_user.id, {"answers": answers})
    except:
        pass

    # --- CLEAR STATE ---
    context.user_data.pop("reflect", None)
    prefs = get_prefs(update_or_query.effective_user.id)
    prefs["mode"] = "chat"
    
# ---------- 8) COMMAND WRAPPER ----------
async def start_reflect(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = get_prefs(update.effective_user.id)["lang"]
    context.user_data.pop("reflect", None)
    return await reflect_start(update, context, lang)


# =========================================================
# 12) CALLBACK HANDLER

async def on_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    data = q.data or ""
    await q.answer()
    uid = update.effective_user.id
    prefs = get_prefs(uid)
    lang = prefs.get("lang", "en")

# === REFLECT CALLBACKS (PHẢI ĐỂ TRÊN CÙNG) ===
    if data == "menu:reflect":
        lang = prefs["lang"]
        prefs["mode"] = "reflect"   # 🟢 BẮT BUỘC
        context.user_data.pop("reflect", None)
        return await reflect_start(update, context, lang)

    if data.startswith("reflect:ans:"):
        _, _, qid, choice = data.split(":", 3)
        return await reflect_handle_choice(update, context, int(qid), choice)

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
        layer = context.user_data.get("menu_layer", "")
        reset_nudge(context)

        # Nếu đang ở exercise (practice mode) → quay về menu practice
        if layer == "exercise":
            txt = "📘 Back to practice menu." if lang != "ru" else "📘 Возврат в меню практики."
            kb = InlineKeyboardMarkup([
                [InlineKeyboardButton("🧠 Vocabulary", callback_data="practice:vocab_menu")],
                [InlineKeyboardButton("⚙️ Grammar", callback_data="practice:grammar_menu")],
                [InlineKeyboardButton("📖 Reading", callback_data="practice:reading_menu")],
                [InlineKeyboardButton("🏠 Main menu", callback_data="menu:root_force")]
            ])
            await safe_edit_text(q, txt, reply_markup=kb)
            await log_event(context, "menu_back_to_practice", uid, {"lang": lang})
            return

        # Còn nếu đang ở quiz hoặc ở bất kỳ layer nào khác → về main menu
        context.user_data.clear()
        msg = "📋 Back to main menu." if lang != "ru" else "📋 Возврат в главное меню."
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

     
        # 🧠 Map nhóm + flavor thành flavor_key chuẩn cho build_mcq
        flavor_key = f"{group}_{flavor}"

        try:
            # --- Tách riêng Reading mode ---
            if group == "reading":
                # 📝 Random topic + sinh đoạn passage
                topic = random.choice(["daily life", "friendship", "school life", "animals", "family", "hobbies", "technology"])
                passage = await build_reading_passage(topic, prefs)

                # ⚙️ Nếu passage trống hoặc lỗi → thử lại 1 lần
                if not passage or len(passage.strip()) < 40:
                    passage = await build_reading_passage("general topic", prefs)

                # 🔐 Lưu passage để gloss / lại dùng sau
                context.user_data["last_passage"] = passage
                context.user_data["reading_topic"] = topic

                # 🧠 Gọi model tạo câu hỏi
                items = await build_mcq(passage, lang, level, flavor=flavor_key)

                # ⚙️ Nếu vẫn không có câu hỏi → thử fallback generic
                if not items:
                    logger.warning(f"Reading MCQ failed for {flavor_key}, retrying generic")
                    items = await build_mcq(passage, lang, level, flavor="reading_details")

            else:
                # --- Grammar & Vocab dùng nội dung gần nhất hoặc general ---
                topic_or_text = context.user_data.get("last_passage", "general English")
                items = await build_mcq(topic_or_text, lang, level, flavor=flavor_key)

        except Exception as e:
            logger.warning(f"build_mcq error ({flavor_key}): {e}")
            return await safe_edit_text(
                q,
                "❌ Failed to create practice questions. Try again later.",
                reply_markup=main_menu(lang)
            )

        # --- Không tạo được câu hỏi ---
        if not items:
            logger.warning(f"build_mcq returned empty for flavor_key={flavor_key}, passage_len={len(passage) if 'passage' in locals() else 0}")
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

          # === VOCABULARY QUICK QUIZ (Practice this word) ===
    if data == "vocab:quiz":
        word = context.user_data.get("last_word", "").strip()
        if not word:
            return await safe_edit_text(
                q,
                "Please define a word first.",
                reply_markup=main_menu(lang)
            )

        # 🔹 Gọi 1 lần build_mcq → tránh timeout
        sub = await build_mcq(word, lang, prefs["cefr"], flavor="vocab_mixed")
        items = sub[:3]

        if not items:
            return await safe_edit_text(
                q,
                "⚠️ No quiz available.",
                reply_markup=main_menu(lang)
            )

        # 🔢 Gán lại ID
        for i, qu in enumerate(items, start=1):
            qu["id"] = i

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

        context.user_data["menu_layer"] = "quiz"

        # 🚀 Gửi câu hỏi đầu tiên
        await send_practice_item(q, context)
        await log_event(context, "vocab_quiz", uid, {"word": word})
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
            [InlineKeyboardButton("✏️ Practice this word", callback_data="vocab:quiz"),
             InlineKeyboardButton("🏠 Menu", callback_data="menu:root")]
        ])

        await safe_edit_text(q, trim(out), reply_markup=kb)
        await log_event(context, "vocab_more_examples", uid, {"word": word})
        return


        # === GRAMMAR PRACTICE (with retry & summary footer) ===
    if data == "grammar:quiz":
        topic = context.user_data.get("last_grammar_topic", "").strip()
        if not topic:
            return await safe_edit_text(q, "No grammar topic found.", reply_markup=main_menu(lang))

        # 🔹 Gọi 1 lần build_mcq
        sub = await build_mcq(topic, lang, prefs["cefr"], flavor="grammar_mixed")
        items = sub[:3]

        if not items:
            return await safe_edit_text(
                q,
                "⚠️ No questions found.",
                reply_markup=main_menu(lang)
            )

        # Gán lại ID
        for i, qu in enumerate(items, start=1):
            qu["id"] = i

        # Lưu trạng thái luyện tập
        context.user_data["practice"] = {
            "type": "grammar",
            "topic": topic,
            "items": items,
            "idx": 0,
            "score": 0,
            "ui_lang": lang,
            "scope": "grammar"
        }

        context.user_data["menu_layer"] = "quiz"

        # 🚀 Gửi câu 1
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
            [InlineKeyboardButton("✏️ Practice this rule", callback_data="grammar:quiz")],
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
            [InlineKeyboardButton("📝 Practice this text", callback_data="reading:quiz")],
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


    if data == "reading:quiz":
        passage = context.user_data.get("last_passage", "").strip()
        topic = context.user_data.get("reading_topic", "reading")

        if not passage:
            return await safe_edit_text(q, "⚠️ No passage found.", reply_markup=main_menu(lang))

        # Gửi passage cho học sinh đọc (edit message)
        await safe_edit_text(q, f"📖 Text:\n\n{trim(passage[:1800])}")
        await asyncio.sleep(0.8)

        # Gọi 1 lần build_mcq
        sub = await build_mcq(passage, lang, prefs["cefr"], flavor="reading_mixed")
        items = sub[:5]

        if not items:
            return await safe_edit_text(
                q,
                "⚠️ Could not generate reading questions.",
                reply_markup=main_menu(lang)
            )

        for i, qu in enumerate(items, start=1):
            qu["id"] = i

        context.user_data["practice"] = {
            "type": "reading",
            "topic": topic,
            "items": items,
            "idx": 0,
            "score": 0,
            "ui_lang": lang,
            "scope": "reading"
        }
        context.user_data["menu_layer"] = "quiz"

        await send_practice_item(q, context)
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
        level = prefs.get("cefr", "A2")

        await safe_edit_text(q, "🔁 Creating a new practice set, please wait...")

        try:
            # ==========================
            # 🔹 VOCABULARY
            # ==========================
            if scope == "vocab" or scope == "vocab_direct":
                word = st.get("topic", "").strip()
                sub = await build_mcq(word, lang, level, flavor="vocab_mixed")
                items = sub[:3]

            # ==========================
            # 🔹 GRAMMAR
            # ==========================
            elif scope == "grammar":
                sub = await build_mcq(topic, lang, level, flavor="grammar_mixed")
                items = sub[:3]

            # ==========================
            # 🔹 READING
            # ==========================
            elif scope == "reading":
                passage = context.user_data.get("last_passage", "")
                sub = await build_mcq(passage, lang, level, flavor="reading_details")
                items = sub[:5]

            # ==========================
            # 🔹 DEFAULT / GENERIC
            # ==========================
            else:
                sub = await build_mcq(topic, lang, level, flavor="generic")
                items = sub[:3]
  
            # ==========================
            # 🔹 Validate
            # ==========================
            if not items:
                return await safe_edit_text(
                    q,
                    "⚠️ No questions found.",
                    reply_markup=main_menu(lang)
                )

            # Gán lại ID cho items
            for i, qu in enumerate(items, start=1):
                qu["id"] = i

            # ==========================
            # 🔹 Reset state
            # ==========================
            st.update({"items": items, "idx": 0, "score": 0})
            context.user_data["practice"] = st
            context.user_data["menu_layer"] = "exercise"

            await send_practice_item(q, context)
            await log_event(context, "practice_regenerated", uid, {"scope": scope, "topic": topic, "count": len(items)})

        except Exception as e:
            logger.warning(f"footer:again error: {e}")
            return await safe_edit_text(
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

    # === REFLECT MODE OVERRIDE ===
    if prefs.get("mode") == "reflect" and "reflect" in context.user_data:
        if "reflect" in context.user_data:
            st = context.user_data["reflect"]
            step = st.get("step", 1)

            # Q6–Q7 nhận text
            if step >= 6:
                return await reflect_handle_text(update, context)

            # Q1–Q5: text không hợp lệ → hướng dẫn học sinh bấm nút
            return 

    # === INTENT DETECTION ===
# ✅ 2. Xác định intent sớm, trước khi xử lý grammar hint

    # ✅ 2️⃣ Prompt-locked intent detection
    t = text.lower()
    intent = "chat"

     # --- VOCABULARY ---
    if re.fullmatch(r"define\s+['\"]?.+['\"]?", t.strip()):
        intent = "vocab"

    # --- GRAMMAR ---
    elif re.fullmatch(r"explain\s+['\"]?.+['\"]?", t.strip()):
        intent = "grammar"

    # --- READING ---
    elif re.fullmatch(r"write\s+(a\s+short\s+)?(a1|a2|b1|b1\+)?\s*text\s+about\s+['\"]?.+['\"]?", t.strip()) \
        or re.fullmatch(r"translate\s+gloss\s+for\s+this\s+text[:\-]?\s*.+", t.strip()):
        intent = "reading"

    # --- TALK ---
    elif re.fullmatch(r"let'?s\s+talk\s+about\s+.+", t.strip()):
        intent = "talk"

    logger.info(f"🎯 Prompt-locked intent: {intent}")


        # --- OUT-OF-SCOPE FILTER (Math, Science, etc.) ---
    out_of_scope_patterns = [
        r"\bsolve\s+\d",            # solve 2x+5=10
        r"\bcalculate\s+\d",        # calculate 45/3
        r"\btriangle\s+area",       # geometry
        r"\bvolume\s+of",           # physics/math
        r"\bderivative\s+of",       # calculus
        r"\bintegral\s+of",         # calculus
        r"\bchemical\s+equation",   # chemistry
        r"\bperiodic\s+table",      # chemistry
        r"\bphysics\b",             # explicit mentions
        r"\bchemistry\b"
    ]
    for pattern in out_of_scope_patterns:
        if re.search(pattern, text.lower()):
            msg = (
                "I’m here to help with *English learning only* 😊 "
                "I can explain vocabulary, grammar, reading texts, or conversation — "
                "but I can't solve math/physics tasks."
                if lang != "ru" else
                "Я помогаю только с английским 😊 "
                "могу объяснить слова, грамматику, чтение или разговор — "
                "но не решаю задачи по математике/физике."
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
            [InlineKeyboardButton("✏️ Practice this word", callback_data="vocab:quiz"),
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
            [InlineKeyboardButton("✏️ Practice this rule", callback_data="grammar:quiz"),
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
                [InlineKeyboardButton("📝 Practice this text", callback_data="reading:quiz")],
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
                 InlineKeyboardButton("📝 Practice this text", callback_data="reading:quiz")],
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
   
        # =========================================================
    # 🌐 DEFAULT CHAT MODE (with memory)
    # =========================================================
    if intent == "chat":

        # 1) Đếm từ để detect long text
        word_count = len(re.findall(r"[A-Za-z]+", text))

        if word_count >= 60 and not re.search(r"\b(translate|gloss|summarize|explain|correct|question)\b", text, re.I):
            msg = (
                "I see a long text. Would you like me to summarize, check grammar, or explain it?"
                if lang != "ru" else
                "Я вижу длинный текст. Хочешь, я помогу с кратким изложением, грамматикой или объяснением?"
            )
            await safe_reply_message(update.message, msg)
            await log_event(context, "long_text_detected", uid, {"words": word_count})

        # =========================================================
        # 2) MEMORY — lưu history 8 lượt gần nhất
        # =========================================================
        history = context.user_data.get("chat_history", [])

        # Thêm message hiện tại
        history.append({"role": "user", "content": text})

        # Giới hạn 8 message cuối
        history = history[-8:]
        context.user_data["chat_history"] = history

        # =========================================================
        # 3) Chuẩn bị messages gửi OpenAI
        # =========================================================
        msgs = [{"role": "system", "content": POLICY_CHAT}]
        msgs.extend(history)

        # =========================================================
        # 4) Gửi request OpenAI
        # =========================================================
        reply = await ask_openai(msgs, max_tokens=350)

        # Lưu reply vào memory để giữ ngữ cảnh
        context.user_data["chat_history"].append({"role": "assistant", "content": reply})
        context.user_data["chat_history"] = context.user_data["chat_history"][-8:]

        # =========================================================
        # 5) Trả lời
        # =========================================================
        reply = remove_markdown(await ask_openai(msgs, max_tokens=350))

        await safe_reply_message(update.message, trim(reply))
        await log_event(context, "chat_message", uid, {"chars": len(text)})

        # =========================================================
        # 6) Nhắc nhở định kỳ sau 10 lượt
        # =========================================================
        chat_turns = context.user_data.get("chat_turns", 0) + 1
        context.user_data["chat_turns"] = chat_turns

        if chat_turns >= 10:
            warn_msg = (
                "⚠️ Reminder: I'm an AI tutor and may make mistakes. Please double-check important information."
                if lang != "ru" else
                "⚠️ Напоминание: я искусственный интеллект и могу ошибаться. Проверяй важные сведения."
            )
            await safe_reply_message(update.message, warn_msg)         
            context.user_data["chat_turns"] = 0  # reset

        return


    # =========================================================
    # 📘 SMART GRAMMAR DETECTION (before CHAT MODE)
    # =========================================================
    if re.search(r"\b(fill in|underline|choose|complete|correct)\b", text.lower()):
        msg = (
            "It looks like a grammar exercise. "
            "I can help you understand the rule step-by-step instead of giving direct answers. "
            "What grammar topic is this about?"
            if lang != "ru" else
            "Похоже на задание по грамматике. "
            "Я могу помочь тебе понять правило шаг за шагом. "
            "О какой грамматике идёт речь?"
        )
        await safe_reply_message(update.message, msg)
        await log_event(context, "textbook_ex_detected", uid, {"text": text[:80]})
        return

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
    return "✅ Bot alive", 200

# --- Start Flask in background ---
def start_flask():
    app.run(host="0.0.0.0", port=10000)

# --- Async polling runner ---
async def run_bot():
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("menu", handle_menu))
    application.add_handler(CommandHandler("reflect_mode", start_reflect))
    application.add_handler(CommandHandler("help", help_cmd))
    application.add_handler(CommandHandler("clear", clear_chat))
    application.add_handler(CallbackQueryHandler(on_cb))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))
    application.add_error_handler(on_error)

    # Gỡ webhook trước khi polling
    await application.bot.delete_webhook(drop_pending_updates=True)
    print("✅ Webhook deleted, ready for polling.")
    print("🚀 Starting async polling loop...")

    # Khởi động theo cách "thủ công" an toàn với Python 3.13
    await application.initialize()
    await application.start()
    await application.updater.start_polling(
        allowed_updates=Update.ALL_TYPES, drop_pending_updates=True
    )
    print("✅ Polling started.")

    # Block vòng lặp (thay cho Updater.wait())
    stop_event = asyncio.Event()
    try:
        await stop_event.wait()   # chặn mãi cho tới khi service bị stop
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        # shutdown gọn gàng
        await application.updater.stop()
        await application.stop()
        await application.shutdown()

def main():
    # Start Flask + keep-alive in background
    threading.Thread(target=start_flask, daemon=True).start()
    threading.Thread(target=keep_alive, daemon=True).start()

    # Run bot asynchronously
    asyncio.run(run_bot())

if __name__ == "__main__":
    main()