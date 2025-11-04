# teleBot.py  (smart intents + fixed talk + clear stale practice)
import os, re, json, time, hmac, hashlib, logging, threading, asyncio
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv
from openai import OpenAI
from flask import Flask

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler,
    ContextTypes, filters
)

# ---------- LOGGING ----------
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
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

# ---------- ENV & CLIENTS ----------
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN: raise RuntimeError("TELEGRAM_TOKEN missing")

USE_OPENROUTER = os.getenv("USE_OPENROUTER", "True").lower() == "true"
OR_KEY = os.getenv("OPENROUTER_API_KEY")
OA_KEY = os.getenv("OPENAI_API_KEY")

GSHEET_WEBHOOK = os.getenv("GSHEET_WEBHOOK", "").strip()
LOG_SALT = os.getenv("LOG_SALT", "").strip()

httpx_client = httpx.Client(timeout=httpx.Timeout(connect=30, read=90, write=90, pool=90),
                            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20))

if USE_OPENROUTER:
    if not OR_KEY: raise RuntimeError("OPENROUTER_API_KEY missing")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OR_KEY, http_client=httpx_client,
                    default_headers={"HTTP-Referer":"https://t.me/SearchVocabBot","X-Title":"School English Bot"})
    MODEL_NAME = "openai/gpt-4o-mini"
else:
    if not OA_KEY: raise RuntimeError("OPENAI_API_KEY missing")
    client = OpenAI(api_key=OA_KEY, http_client=httpx_client)
    MODEL_NAME = "gpt-3.5-turbo"

# ---------- CONSTANTS ----------
DEFAULT_LANG = "auto"  # auto|en|ru
MAX_HISTORY = 10
ALLOWED_MODES = {"chat","vocab","reading","grammar","practice","talk"}
BANNED_KEYWORDS = [r"\bsex\b", r"\bporn\b", r"\berotic\b", r"\bviolence\b", r"\bsuicide\b",
                   r"\bself[- ]?harm\b", r"\bdrugs?\b", r"\balcohol\b", r"\bgamble\b", r"\bextremis(m|t)\b"]
GRADE_TO_CEFR = {"6":"A2","7":"A2+","8":"B1-","9":"B1"}
DEFAULT_DIALOGUE_LIMIT = 10

POLICY_CHAT = ("You are a safe, school-appropriate assistant for grades 6–9. "
               "No markdown bold or headings. English by default; if the user writes Russian, respond in Russian. "
               "Be friendly, concise, and helpful. If a request is unsafe/off-topic, refuse and steer to study topics. "
               "Level A2–B1.")
POLICY_STUDY = ("You are an English study assistant for grades 6–9 (CEFR A2–B1). "
                "No markdown bold or headings. Keep answers short, safe, and age-appropriate.")

CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")

def detect_lang(text: str) -> str:
    return "ru" if CYRILLIC_RE.search(text or "") else "en"

def trim(s: str, max_chars: int = 1000) -> str:
    s = re.sub(r"\n{3,}", "\n\n", (s or "").strip())
    return s if len(s) <= max_chars else (s[:max_chars].rstrip()+"…")

def blocked(text: str) -> bool:
    for pat in BANNED_KEYWORDS:
        if re.search(pat, text or "", flags=re.IGNORECASE): return True
    return False

async def ask_openai(messages, max_tokens=500, temperature=0.3, model=None):
    model = model or MODEL_NAME
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(model=model, messages=messages,
                                                  max_tokens=max_tokens, temperature=temperature)
            return resp.choices[0].message.content
        except Exception as e1:
            if attempt < 2: time.sleep(2**attempt); continue
            base_url = getattr(client, "base_url", "") or ""
            fb = "openai/gpt-3.5-turbo" if "openrouter.ai" in base_url else "gpt-3.5-turbo"
            try:
                resp = client.chat.completions.create(model=fb, messages=messages,
                                                      max_tokens=max_tokens, temperature=temperature)
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
                try: return json.loads(block.split("\n",1)[1])
                except Exception: pass
            try: return json.loads(block)
            except Exception: continue
    return json.loads(s)

# ---------- PREFS / STATE ----------
user_prefs = {}
def get_prefs(user_id: int):
    if user_id not in user_prefs:
        user_prefs[user_id] = {"mode":"chat","lang":DEFAULT_LANG,"grade":"7",
                               "cefr":GRADE_TO_CEFR["7"],"dialogue_limit":DEFAULT_DIALOGUE_LIMIT}
    return user_prefs[user_id]

# ---------- METRICS ----------
def _post_json(url, data, timeout=5.0, headers=None):
    if not url: return
    try: httpx.post(url, json=data, headers=headers or {}, timeout=timeout)
    except Exception: pass

async def log_event(context: ContextTypes.DEFAULT_TYPE, action: str, user_id,
                    extra: dict|None=None, chat_id: int|None=None,
                    started_ms: float|None=None, text: str|None=None,
                    message_len: int|None=None, mode: str|None=None):
    if not GSHEET_WEBHOOK: return
    now = time.time()
    delta_ms = int((now - started_ms)*1000) if started_ms else None
    sid = f"{chat_id}:{int(now//86400)}" if chat_id is not None else ""
    payload = {
        "ts": int(now*1000),
        "user_id": str(user_id),
        "chat_id": str(chat_id) if chat_id is not None else "",
        "action": action,
        "mode": mode or (get_prefs(int(user_id))["mode"] if isinstance(user_id,int) else ""),
        "session_id": sid,
        "delta_ms": delta_ms,
        "message_len": message_len if message_len is not None else (len(text) if text else None),
        "text": (text or "")[:200]
    }
    headers = {}
    if LOG_SALT:
        sig_src = f"{payload['user_id']}|{payload['action']}|{payload['ts']}"
        signature = hmac.new(LOG_SALT.encode("utf-8"), sig_src.encode("utf-8"), hashlib.sha256).hexdigest()
        headers["X-Log-Signature"] = signature
    try:
        await asyncio.to_thread(_post_json, GSHEET_WEBHOOK, payload, 10.0, headers)
    except Exception as e:
        logger.warning("log_event failed: %s", e)

# ---------- UI ----------
def root_menu(lang: str) -> InlineKeyboardMarkup:
    if lang == "ru":
        kb = [
            [InlineKeyboardButton("📚 Слова","menu:mode:vocab"),
             InlineKeyboardButton("📖 Чтение","menu:mode:reading")],
            [InlineKeyboardButton("⚙️ Грамматика","menu:mode:grammar"),
             InlineKeyboardButton("📝 Практика","menu:mode:practice")],
            [InlineKeyboardButton("💬 Разговор","menu:mode:talk")],
            [InlineKeyboardButton("🏫 Класс","menu:grade"),
             InlineKeyboardButton("🌐 Язык","menu:lang")],
            [InlineKeyboardButton("📋 Меню","menu:root")]
        ]
    else:
        kb = [
            [InlineKeyboardButton("📚 Vocabulary","menu:mode:vocab"),
             InlineKeyboardButton("📖 Reading","menu:mode:reading")],
            [InlineKeyboardButton("⚙️ Grammar","menu:mode:grammar"),
             InlineKeyboardButton("📝 Practice","menu:mode:practice")],
            [InlineKeyboardButton("💬 Talk","menu:mode:talk")],
            [InlineKeyboardButton("🏫 Grade","menu:grade"),
             InlineKeyboardButton("🌐 Language","menu:lang")],
            [InlineKeyboardButton("📋 Back to Menu","menu:root")]
        ]
    # telegram lib requires callback_data set via kw; shortcut above not allowed → fix:
    rows=[]
    for row in kb:
        rows.append([InlineKeyboardButton(text=b.text, callback_data=b.callback_data) for b in row])
    return InlineKeyboardMarkup(rows)

def lang_menu():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("English","set_lang:en"),
         InlineKeyboardButton("Русский","set_lang:ru")],
        [InlineKeyboardButton("⬅️ Back","menu:root")]
    ])

def grade_menu():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("6","set_grade:6"),
         InlineKeyboardButton("7","set_grade:7"),
         InlineKeyboardButton("8","set_grade:8"),
         InlineKeyboardButton("9","set_grade:9")],
        [InlineKeyboardButton("⬅️ Back","menu:root")]
    ])

def practice_menu(lang="en"):
    text = (["Тест (A–D)","Формы глагола","Пропуски","Словообразование","Исправь ошибку","Порядок слов"]
            if lang=="ru" else
            ["Multiple Choice","Verb Forms","Gap Fill","Word Formation","Error Correction","Sentence Ordering"])
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"🅰 {text[0]}","practice:type:mcq"),
         InlineKeyboardButton(f"🔤 {text[1]}","practice:type:verb")],
        [InlineKeyboardButton(f"🕳 {text[2]}","practice:type:gap"),
         InlineKeyboardButton(f"🧱 {text[3]}","practice:type:wordform")],
        [InlineKeyboardButton(f"❌ {text[4]}","practice:type:error"),
         InlineKeyboardButton(f"🔁 {text[5]}","practice:type:order")],
        [InlineKeyboardButton("⬅️ Back","menu:root")]
    ])

def talk_topics_menu(lang="en"):
    lbl = (["Быт","Школа","Хобби","Окружающая среда","Праздники","Семья"]
           if lang=="ru" else
           ["Daily life","School life","Hobbies","Environment","Holidays","Family"])
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(lbl[0],"talk:topic:daily"),
         InlineKeyboardButton(lbl[1],"talk:topic:school")],
        [InlineKeyboardButton(lbl[2],"talk:topic:hobbies"),
         InlineKeyboardButton(lbl[3],"talk:topic:env")],
        [InlineKeyboardButton(lbl[4],"talk:topic:holidays"),
         InlineKeyboardButton(lbl[5],"talk:topic:family")],
        [InlineKeyboardButton("⬅️ Back","menu:root")]
    ])

def mcq_buttons(options):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"A) {options[0]}","ans:A"),
         InlineKeyboardButton(f"B) {options[1]}","ans:B")],
        [InlineKeyboardButton(f"C) {options[2]}","ans:C"),
         InlineKeyboardButton(f"D) {options[3]}","ans:D")]
    ])

# ---------- START / HELP ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text_in = (update.message.text or "").strip()
    greet = "Hi there! I’m your English study buddy. How can I help you today?"
    if detect_lang(text_in) == "ru": greet = "Привет! Я твой помощник по английскому. Чем помочь сегодня?"
    prefs = get_prefs(update.effective_user.id); prefs["mode"]="chat"
    await update.message.reply_text(greet, reply_markup=root_menu(prefs.get("lang","en")))
    await log_event(context,"start",update.effective_user.id,{"text":text_in[:200]},
                    chat_id=update.effective_chat.id, text=text_in, mode="chat")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prefs = get_prefs(update.effective_user.id)
    lang = prefs.get("lang","en")
    msg = "Choose from the menu below." if lang!="ru" else "Выберите пункт меню ниже."
    await update.message.reply_text(msg, reply_markup=root_menu(lang))
    await log_event(context,"help",update.effective_user.id,{"lang":lang},
                    chat_id=update.effective_chat.id, text=msg, mode=prefs.get("mode"))

# ---------- VOCAB CARD ----------
async def build_vocab_card(headword: str, prefs: dict, user_text: str) -> str:
    lang_for_examples = prefs.get("lang","auto")
    if lang_for_examples=="auto": lang_for_examples = detect_lang(user_text or "")
    include_ru = (lang_for_examples=="ru")
    prompt = (
        "You are an English-learning assistant for grades 6–9 (CEFR A2–B1). "
        "Make a compact vocabulary card. Do not use markdown bold. "
        "Definition must be in English with a short Russian translation in parentheses.\n\n"
        f"HEADWORD: {headword}\nTARGET LEVEL: {prefs['cefr']}\n\n"
        "Format exactly:\n"
        "Word: <headword> /<IPA>/\nPOS: <part of speech>\n"
        "Definition: <short English definition> (<short Russian translation>)\n"
        "Examples:\n"
        f"1) <A2–B1 English example>{' (Russian translation)' if include_ru else ''}\n"
        f"2) <A2–B1 English example>{' (Russian translation)' if include_ru else ''}\n"
        f"3) <A2–B1 English example>{' (optional Russian translation)' if include_ru else ' (optional)'}\n"
        "Keep under 120 words."
    )
    return await ask_openai([{"role":"system","content":POLICY_STUDY},{"role":"user","content":prompt}], max_tokens=320)

# ---------- PRACTICE BUILDERS ----------
def normalize_answer(s:str)->str:
    s=(s or "").strip().lower(); s=re.sub(r"[^\w\s'-]","",s); s=re.sub(r"\s+"," ",s); return s

async def build_mcq(topic:str, ui_lang:str, level:str):
    prompt=(f"Create a 5-question multiple-choice quiz (4 options A–D) on '{topic}', level {level}, grades 6–9.\n"
            "Return STRICT JSON only:\n{ \"questions\": ["
            "{\"id\":1,\"question\":\"...\",\"options\":[\"...\",\"...\",\"...\",\"...\"],"
            "\"answer\":\"A\",\"explain_en\":\"<=25 words\",\"explain_ru\":\"<=25 words\"},"
            "{\"id\":2,...},{\"id\":3,...},{\"id\":4,...},{\"id\":5,...}]}\n"
            f"Language for 'question' and 'options': {'Russian' if ui_lang=='ru' else 'English'} (A2–B1). ")
    raw=await ask_openai([{"role":"system","content":POLICY_STUDY},{"role":"user","content":prompt}],max_tokens=800)
    data=extract_json(raw); items=[]
    for q in data.get("questions",[]): items.append({"id":q.get("id"),"question":q.get("question"),
        "options":q.get("options",["","","",""]),"answer":q.get("answer","A"),
        "explain_en":q.get("explain_en",""),"explain_ru":q.get("explain_ru","")})
    return items

async def build_text_items(ptype:str, topic:str, ui_lang:str, level:str):
    task_desc={"verb":"Conjugate the verb in brackets into the correct form.",
               "gap":"Fill in the blank with one suitable word.",
               "wordform":"Complete the sentence using the correct form of the word in parentheses.",
               "error":"Find and correct the mistake in the sentence (write the corrected version).",
               "order":"Reorder the words to make a correct sentence."}[ptype]
    prompt=(f"Create 5 short {ptype} exercises on '{topic}', level {level}, grades 6–9. Task: {task_desc}\n"
            "Return STRICT JSON only:\n{ \"items\": ["
            "{\"id\":1,\"prompt\":\"...\",\"answer\":\"...\",\"explain_en\":\"<=25 words\",\"explain_ru\":\"<=25 words\"},"
            "{\"id\":2,...},{\"id\":3,...},{\"id\":4,...},{\"id\":5,...}]}\n"
            f"Language for 'prompt': {'Russian' if ui_lang=='ru' else 'English'} (A2–B1). ")
    raw=await ask_openai([{"role":"system","content":POLICY_STUDY},{"role":"user","content":prompt}],max_tokens=900)
    data=extract_json(raw); items=[]
    for it in data.get("items",[]): items.append({"id":it.get("id"),"prompt":it.get("prompt"),
        "answer":it.get("answer"),"explain_en":it.get("explain_en",""),"explain_ru":it.get("explain_ru","")})
    return items

async def send_practice_item(update_or_query, context: ContextTypes.DEFAULT_TYPE):
    st=context.user_data.get("practice"); 
    if not st: 
        return
    idx=st["idx"]; total=len(st["items"]); title=f"Q{idx+1}/{total}"
    if st["type"]=="mcq":
        q=st["items"][idx]; text=f"{title}\n\n{q['question']}"; kb=mcq_buttons(q["options"])
        if isinstance(update_or_query,Update): await update_or_query.message.reply_text(text, reply_markup=kb)
        else: await update_or_query.edit_message_text(text, reply_markup=kb)
    else:
        q=st["items"][idx]; head="Type your answer:" if st.get("ui_lang","en")!="ru" else "Напиши ответ:"
        text=f"{title}\n\n{q['prompt']}\n\n{head}"
        if isinstance(update_or_query,Update): await update_or_query.message.reply_text(text)
        else: await update_or_query.edit_message_text(text)
    # log show_question
    try:
        if isinstance(update_or_query,Update):
            chat_id=update_or_query.effective_chat.id; uid=update_or_query.effective_user.id
        else:
            chat_id=update_or_query.message.chat.id; uid=update_or_query.from_user.id
        await log_event(context,"show_question",uid,chat_id=chat_id,text=str(st["items"][idx].get("id")),mode="practice")
    except Exception: pass

async def practice_summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    st=context.user_data.get("practice"); 
    if not st: 
        return
    lang=st.get("ui_lang","en"); total=len(st["items"]); score=st.get("score",0)
    lines=[(f"Итоги: {score}/{total}" if lang=="ru" else f"Summary: {score}/{total}"),
           ("Ответы и объяснения:" if lang=="ru" else "Answers and explanations:")]
    for it in st["items"]:
        expl = it["explain_ru"] if lang=="ru" and it["explain_ru"] else it["explain_en"]
        lines.append(f"Q{it['id']}: {it['answer']} — {expl}")
    await update.message.reply_text("\n".join(lines))
    await log_event(context,"practice_done",update.effective_user.id,
                    {"type":st["type"],"topic":st.get("topic"),"score":score,"total":total},
                    chat_id=update.effective_chat.id, text="summary", mode="practice")
    context.user_data.pop("practice",None)

# ---------- TALK COACH ----------
async def talk_reply(user_text:str, topic:str, ui_lang:str):
    prompt=("You are a friendly English conversation coach for a middle-school student (A2–B1). "
            f"Topic: {topic}. Respond in 1–3 sentences. Encourage and suggest 1–2 useful words/phrases. "
            "Correct small mistakes implicitly by reformulating. School-safe, positive. No markdown bold.")
    return await ask_openai([{"role":"system","content":prompt},
                             {"role":"user","content":f"Student says: {user_text}"}], max_tokens=180)

# ---------- INTENT LAYER ----------
INTENTS = ["translate","examples_more","questions_more","explain_more","new_topic","continue","general_chat"]

def quick_heuristic(txt:str):
    t=txt.lower()
    if any(k in t for k in ["dịch","translate","переведи","перевод"]): return "translate"
    if any(k in t for k in ["more examples","another example","ví dụ khác","ещё пример"]): return "examples_more"
    if any(k in t for k in ["more questions","another question","ra thêm câu hỏi","ещё вопросы"]): return "questions_more"
    if any(k in t for k in ["explain more","giải thích thêm","подробнее"]): return "explain_more"
    if any(k in t for k in ["new topic","đổi chủ đề","change topic","другая тема"]): return "new_topic"
    if any(k in t for k in ["continue","tiếp tục","продолжить"]): return "continue"
    return None

async def classify_intent(txt:str, ui_lang:str):
    h=quick_heuristic(txt)
    if h: return h
    # very small model call to be safe; return one label
    prompt=(f"Classify the user's message into one of {INTENTS}. "
            "Reply with only the label. Message: "+txt[:300])
    out=await ask_openai([{"role":"system","content":"Return only a label from the list."},
                          {"role":"user","content":prompt}], max_tokens=3, temperature=0)
    out=(out or "").strip().lower()
    return out if out in INTENTS else "general_chat"

async def handle_intent(update:Update, context:ContextTypes.DEFAULT_TYPE, intent:str, ui_lang:str):
    uid=update.effective_user.id; chat_id=update.effective_chat.id
    msg=update.message.text or ""
    # If practicing and user asks off-track, answer shortly then hint to continue
    practicing = context.user_data.get("practice") is not None
    hint = ("\n\nType 'continue' to go back to your exercise." if practicing and ui_lang!="ru"
            else ("\n\nНапиши 'continue', чтобы вернуться к упражнению." if practicing else ""))

    if intent=="translate":
        # translate to UI language
        dest = "Russian" if ui_lang=="ru" else "English"
        trans = await ask_openai(
            [{"role":"system","content":"You are a careful translator for short texts. No bold."},
             {"role":"user","content":f"Translate to {dest}:\n{msg}"}],
            max_tokens=300)
        await update.message.reply_text(trim(trans)+hint)
        await log_event(context,"bot_reply",uid,chat_id=chat_id,text=trans,mode=get_prefs(uid)["mode"])
        return True

    if intent=="examples_more":
        ex = await ask_openai(
            [{"role":"system","content":POLICY_STUDY},
             {"role":"user","content":"Give 3 short A2–B1 examples about this topic or the last concept. No bold."}],
            max_tokens=200)
        await update.message.reply_text(trim(ex)+hint); return True

    if intent=="questions_more":
        qs = await ask_openai(
            [{"role":"system","content":POLICY_STUDY},
             {"role":"user","content":"Create 3 quick comprehension/practice questions (A2–B1) about our last topic. No answers."}],
            max_tokens=180)
        await update.message.reply_text(trim(qs)+hint); return True

    if intent=="explain_more":
        ex = await ask_openai(
            [{"role":"system","content":POLICY_STUDY},
             {"role":"user","content":"Explain the last concept again in 3–5 simple bullets with 1–2 examples. No bold."}],
            max_tokens=220)
        await update.message.reply_text(trim(ex)+hint); return True

    if intent=="new_topic":
        txt = "Sure — tell me the new topic." if ui_lang!="ru" else "Хорошо — напиши новую тему."
        await update.message.reply_text(txt); return True

    if intent=="continue" and practicing:
        await send_practice_item(update, context); return True

    if intent=="general_chat":
        return False
    return False

# ---------- COMMANDS ----------
async def vocab_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prefs=get_prefs(update.effective_user.id); prefs["mode"]="vocab"
    # clear other states when entering vocab
    context.user_data.pop("practice",None); context.user_data.pop("talk",None)
    await update.message.reply_text("Vocabulary mode is ON. Send me a word.")
    await log_event(context,"mode_set",update.effective_user.id,{"mode":"vocab"},
                    chat_id=update.effective_chat.id, mode="vocab")

# ---------- CALLBACKS ----------
async def on_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q=update.callback_query; data=q.data or ""; await q.answer()
    uid=update.effective_user.id; prefs=get_prefs(uid)
    ui_lang=prefs.get("lang","en"); 
    if ui_lang=="auto": ui_lang = "ru" if CYRILLIC_RE.search(q.message.text or "") else "en"
    chat_id=q.message.chat.id

    await log_event(context,"button",uid,chat_id=chat_id,text=data,mode=prefs.get("mode","chat"))

    if data=="menu:root":
        prefs["mode"]="chat"; context.user_data.pop("practice",None); context.user_data.pop("talk",None)
        await q.edit_message_text("Back to menu." if ui_lang!="ru" else "Возврат в меню.",
                                  reply_markup=root_menu(ui_lang))
        return

    if data=="menu:lang":
        await q.edit_message_text("Choose language:" if ui_lang!="ru" else "Выберите язык:",
                                  reply_markup=lang_menu()); return

    if data=="menu:grade":
        await q.edit_message_text("Choose grade:" if ui_lang!="ru" else "Выберите класс:",
                                  reply_markup=grade_menu()); return

    if data.startswith("set_lang:"):
        lang=data.split(":",1)[1]; prefs["lang"]=lang
        await q.edit_message_text(("Language set to "+lang.upper()) if lang!="ru" else "Язык: Русский.",
                                  reply_markup=root_menu(lang))
        await log_event(context,"lang_set",uid,{"lang":lang},chat_id=chat_id,mode=prefs.get("mode")); return

    if data.startswith("set_grade:"):
        g=data.split(":",1)[1]
        if g in GRADE_TO_CEFR:
            prefs["grade"]=g; prefs["cefr"]=GRADE_TO_CEFR[g]
            txt=(f"Grade set to {g}. Target level: {prefs['cefr']}."
                 if ui_lang!="ru" else f"Класс: {g}. Уровень: {prefs['cefr']}.")
            await q.edit_message_text(txt, reply_markup=root_menu(ui_lang))
            await log_event(context,"grade_set",uid,{"grade":g,"cefr":prefs["cefr"]},chat_id=chat_id,mode=prefs.get("mode"))
        else:
            await q.edit_message_text("Invalid grade.", reply_markup=root_menu(ui_lang))
        return

    if data.startswith("menu:mode:"):
        mode=data.split(":")[-1]
        prefs["mode"]=mode
        # clear stale states when switching mode
        if mode!="practice": context.user_data.pop("practice",None)
        if mode!="talk": context.user_data.pop("talk",None)
        await log_event(context,"mode_set",uid,{"mode":mode},chat_id=chat_id,mode=mode)
        if mode=="vocab":
            txt="Vocabulary mode is ON. Send a word." if ui_lang!="ru" else "Режим Слова. Отправь слово."
            await q.edit_message_text(txt, reply_markup=root_menu(ui_lang))
        elif mode=="reading":
            txt="Reading mode is ON. Send a topic for a short passage." if ui_lang!="ru" else "Режим Чтение. Отправь тему."
            await q.edit_message_text(txt, reply_markup=root_menu(ui_lang))
        elif mode=="grammar":
            txt="Grammar mode is ON. Send a grammar point (e.g., Present Simple)." if ui_lang!="ru" else "Режим Грамматика. Отправь тему."
            await q.edit_message_text(txt, reply_markup=root_menu(ui_lang))
        elif mode=="practice":
            await q.edit_message_text("Choose an exercise type:" if ui_lang!="ru" else "Выберите тип упражнения:",
                                      reply_markup=practice_menu(ui_lang))
        elif mode=="talk":
            await q.edit_message_text("Choose a topic to talk about:" if ui_lang!="ru" else "Выберите тему для разговора:",
                                      reply_markup=talk_topics_menu(ui_lang))
        return

    if data.startswith("practice:type:"):
        ptype=data.split(":")[-1]
        context.user_data["practice"]={"type":ptype,"topic":None,"items":[],
                                       "idx":0,"attempts":0,"score":0,"ui_lang":ui_lang}
        await q.edit_message_text("Send me a topic (e.g., pollution)." if ui_lang!="ru" else "Отправь тему (например, pollution).")
        await log_event(context,"practice_type_set",uid,{"ptype":ptype},chat_id=chat_id,mode="practice"); return

    if data.startswith("talk:topic:"):
        topic_key=data.split(":")[-1]
        mapping={"daily":"daily life","school":"school life","hobbies":"hobbies","env":"environment","holidays":"holidays","family":"family"}
        topic=mapping.get(topic_key,"daily life")
        prefs["mode"]="talk"; context.user_data["talk"]={"topic":topic,"turns":0}
        opener="Let’s talk! How are you today?" if ui_lang!="ru" else "Поговорим! Как твоё настроение сегодня?"
        await q.edit_message_text(f"Topic: {topic}\n\n{opener}", reply_markup=root_menu(ui_lang))
        await log_event(context,"talk_topic_set",uid,{"topic":topic},chat_id=chat_id,mode="talk"); return

    if data.startswith("ans:"):
        st=context.user_data.get("practice")
        if not st or st.get("type")!="mcq":
            await q.edit_message_text("No active multiple-choice exercise.", reply_markup=root_menu(ui_lang)); return
        choice=data.split(":",1)[1]; idx=st["idx"]; qitem=st["items"][idx]; correct=qitem["answer"]
        if choice==correct:
            st["score"]+=1; st["attempts"]=0
            expl=qitem["explain_ru"] if ui_lang=="ru" and qitem["explain_ru"] else qitem["explain_en"]
            await q.edit_message_text(("Correct!" if ui_lang!="ru" else "Верно!")+"\n"+expl)
            await log_event(context,"practice_answer",uid,{"ptype":"mcq","qid":qitem.get("id"),"correct":True},
                            chat_id=chat_id,mode="practice")
            st["idx"]+=1
        else:
            st["attempts"]+=1
            if st["attempts"]<2:
                await q.edit_message_text("Not quite. Try again." if ui_lang!="ru" else "Почти. Попробуй еще раз.")
                await log_event(context,"practice_answer",uid,{"ptype":"mcq","qid":qitem.get("id"),"correct":False,"retry":True},
                                chat_id=chat_id,mode="practice")
            else:
                st["attempts"]=0
                expl=qitem["explain_ru"] if ui_lang=="ru" and qitem["explain_ru"] else qitem["explain_en"]
                await q.edit_message_text((f"The correct answer is {correct}." if ui_lang!="ru" else f"Правильный ответ: {correct}.")+"\n"+expl)
                await log_event(context,"practice_answer",uid,{"ptype":"mcq","qid":qitem.get("id"),"correct":False,"revealed":True},
                                chat_id=chat_id,mode="practice")
                st["idx"]+=1
        if st["idx"]>=len(st["items"]):
            dummy_update=Update(update.update_id, message=q.message)
            await practice_summary(dummy_update, context)
        else:
            await send_practice_item(q, context)
        return

# ---------- FREE TEXT ----------
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text or ""
    if blocked(user_message):
        return await update.message.reply_text("⛔ That's outside our classroom scope. Please try vocabulary, reading, grammar, or a practice topic.")
    uid=update.effective_user.id; chat_id=update.effective_chat.id
    prefs=get_prefs(uid); lang=prefs.get("lang","en"); 
    if lang=="auto": lang=detect_lang(user_message)
    mode_now=prefs.get("mode","chat")
    t0=time.time()

    # 1) Log user message
    await log_event(context,"user_msg",uid,chat_id=chat_id,text=user_message,message_len=len(user_message),mode=mode_now)

    # 2) Smart intent layer first (works in ANY mode)
    intent = await classify_intent(user_message, lang)
    handled = await handle_intent(update, context, intent, lang)
    if handled: return

    # 3) Mode-specific (unchanged behaviour)
    st=context.user_data.get("practice")
    # If user previously selected practice type but not built yet → treat as topic
    if st and not st.get("items"):
        topic=user_message.strip() or "school life"; st["topic"]=topic; level=prefs["cefr"]
        try:
            st["items"] = await (build_mcq(topic,lang,level) if st["type"]=="mcq"
                                 else build_text_items(st["type"],topic,lang,level))
        except Exception:
            msg="Failed to build exercises. Please try another topic." if lang!="ru" else "Не удалось создать задания. Попробуй другую тему."
            await update.message.reply_text(msg)
            await log_event(context,"practice_build_fail",uid,{"ptype":st["type"],"topic":topic},chat_id=chat_id,mode="practice")
            return
        if not st["items"]:
            msg="No items generated. Try another topic." if lang!="ru" else "Задания не созданы. Попробуй другую тему."
            await update.message.reply_text(msg)
            await log_event(context,"practice_empty",uid,{"ptype":st["type"],"topic":topic},chat_id=chat_id,mode="practice")
            return
        st.update({"idx":0,"attempts":0,"score":0,"ui_lang":lang})
        await log_event(context,"practice_built",uid,{"ptype":st["type"],"topic":topic,"count":len(st["items"])},
                        chat_id=chat_id,mode="practice")
        return await send_practice_item(update, context)

    # PRACTICE answers (text types)
    if st and st.get("items") and st["type"]!="mcq":
        idx=st["idx"]
        if idx < len(st["items"]):
            qitem=st["items"][idx]; user_ans=normalize_answer(user_message); gold=normalize_answer(qitem["answer"])
            if user_ans==gold:
                st["score"]+=1; st["attempts"]=0
                expl=qitem["explain_ru"] if st["ui_lang"]=="ru" and qitem["explain_ru"] else qitem["explain_en"]
                await update.message.reply_text(("Correct!" if st["ui_lang"]!="ru" else "Верно!")+"\n"+expl)
                await log_event(context,"practice_answer",uid,{"ptype":st["type"],"qid":qitem.get("id"),"correct":True},
                                chat_id=chat_id,mode="practice")
                st["idx"]+=1
            else:
                st["attempts"]+=1
                if st["attempts"]<2:
                    await update.message.reply_text("Not quite. Try again." if st["ui_lang"]!="ru" else "Почти. Попробуй еще раз.")
                    await log_event(context,"practice_answer",uid,{"ptype":st["type"],"qid":qitem.get("id"),"correct":False,"retry":True},
                                    chat_id=chat_id,mode="practice")
                    return
                st["attempts"]=0
                expl=qitem["explain_ru"] if st["ui_lang"]=="ru" and qitem["explain_ru"] else qitem["explain_en"]
                await update.message.reply_text((f"The correct answer is: {qitem['answer']}" if st["ui_lang"]!="ru" else f"Правильный ответ: {qitem['answer']}")+"\n"+expl)
                await log_event(context,"practice_answer",uid,{"ptype":st["type"],"qid":qitem.get("id"),"correct":False,"revealed":True},
                                chat_id=chat_id,mode="practice")
                st["idx"]+=1
            if st["idx"]>=len(st["items"]): return await practice_summary(update, context)
            else: return await send_practice_item(update, context)

    # VOCAB
    if prefs["mode"]=="vocab":
        word=user_message.strip()
        if not word: return await update.message.reply_text("Send a word to look up." if lang!="ru" else "Отправь слово.")
        try:
            card=await build_vocab_card(word,prefs,update.message.text)
            await update.message.reply_text(trim(card))
            await log_event(context,"bot_reply",uid,chat_id=chat_id,started_ms=t0,text=card,mode="vocab"); return
        except Exception:
            await update.message.reply_text("Failed to build the card. Try another word." if lang!="ru" else "Не удалось создать карточку. Попробуй другое слово.")
            await log_event(context,"vocab_fail",uid,{"word":word},chat_id=chat_id,mode="vocab"); return

    # READING
    if prefs["mode"]=="reading":
        topic=user_message.strip() or "school life"; level=prefs["cefr"]
        passage=await ask_openai([{"role":"system","content":POLICY_STUDY},
                                  {"role":"user","content":f"Write a short reading (80–120 words) about '{topic}', level {level}. Language: {'Russian' if lang=='ru' else 'English'}. No bold."}],
                                 max_tokens=220)
        await update.message.reply_text(trim(passage))
        await log_event(context,"bot_reply",uid,chat_id=chat_id,started_ms=t0,text=passage,mode="reading")
        mcq_items= await build_mcq(topic,lang,level)
        mcq_items=mcq_items[:3] if len(mcq_items)>3 else mcq_items
        context.user_data["practice"]={"type":"mcq","topic":topic,"items":mcq_items,"idx":0,"attempts":0,"score":0,"ui_lang":lang}
        return await send_practice_item(update, context)

    # GRAMMAR
    if prefs["mode"]=="grammar":
        text=user_message.strip()
        if re.search(r"\b(practice|exercises|tasks)\b", text, re.I) or (CYRILLIC_RE.search(text) and re.search(r"(упражн|практик)", text, re.I)):
            topic=context.user_data.get("last_grammar_topic","general grammar"); level=prefs["cefr"]
            items=await build_text_items("verb",topic,lang,level)
            context.user_data["practice"]={"type":"verb","topic":topic,"items":items,"idx":0,"attempts":0,"score":0,"ui_lang":lang}
            await log_event(context,"grammar_practice",uid,{"topic":topic,"count":len(items)},chat_id=chat_id,mode="practice")
            return await send_practice_item(update, context)
        context.user_data["last_grammar_topic"]=text or "Present Simple"
        g_prompt=(f"Explain briefly the grammar point: {context.user_data['last_grammar_topic']} for level {prefs['cefr']} "
                  f"in 3–5 bullets with 1–2 examples. Language: {'Russian' if lang=='ru' else 'English'}. No markdown bold.")
        exp=await ask_openai([{"role":"system","content":POLICY_STUDY},{"role":"user","content":g_prompt}],max_tokens=260)
        extra="Type 'practice' to get exercises." if lang!="ru" else "Напиши 'practice', чтобы получить упражнения."
        await update.message.reply_text(trim(exp)+"\n\n"+extra)
        await log_event(context,"bot_reply",uid,chat_id=chat_id,started_ms=t0,text=exp,mode="grammar"); return

    # TALK
    if prefs["mode"]=="talk":
        talk_state=context.user_data.get("talk") or {"topic":"daily life","turns":0}
        reply=await talk_reply(user_message, talk_state["topic"], lang)
        talk_state["turns"]=talk_state.get("turns",0)+1; context.user_data["talk"]=talk_state
        await update.message.reply_text(trim(reply))
        await log_event(context,"bot_reply",uid,chat_id=chat_id,started_ms=t0,text=reply,mode="talk")
        if talk_state["turns"]>=prefs.get("dialogue_limit",DEFAULT_DIALOGUE_LIMIT):
            wrap=("Great chat! Want to practice? Try Vocabulary or Practice from the menu."
                  if lang!="ru" else "Отличная беседа! Хочешь потренироваться? Выбери Слова или Практика в меню.")
            await update.message.reply_text(wrap, reply_markup=root_menu(lang))
            prefs["mode"]="chat"; context.user_data.pop("talk",None)
        return

    # Default CHAT
    history=context.user_data.get("history",[]); history.append({"role":"user","content":user_message})
    history=history[-MAX_HISTORY:]; context.user_data["history"]=history
    steer=("Be helpful and concise. If the user asks about study tasks, you can suggest modes: Vocabulary, Reading, Grammar, Practice, Talk.")
    text_out=await ask_openai([{"role":"system","content":POLICY_CHAT},{"role":"user","content":steer},*history], max_tokens=400)
    await update.message.reply_text(trim(text_out))
    await log_event(context,"bot_reply",uid,chat_id=chat_id,started_ms=t0,text=text_out,mode="chat")

# ---------- FLASK ----------
app = Flask(__name__)
@app.get("/")
def health(): return "✅ Bot is alive", 200
def start_flask():
    port=int(os.getenv("PORT","10000")); app.run(host="0.0.0.0", port=port)

# ---------- MAIN ----------
def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_cmd))
    application.add_handler(CommandHandler("vocab", vocab_cmd))
    application.add_handler(CallbackQueryHandler(on_cb))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_error_handler(on_error); application.post_init = on_startup
    threading.Thread(target=start_flask, daemon=True).start()
    logger.info("Bot is starting (Web Service + Flask)…")
    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__=="__main__": main()
