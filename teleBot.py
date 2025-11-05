# teleBot.py
# =========================================================
# 0) IMPORTS & GLOBAL SETUP
# =========================================================
import os, re, json, time, hmac, hashlib, logging, threading, asyncio, uuid, difflib
from datetime import datetime, timezone
import httpx
from dotenv import load_dotenv
from openai import OpenAI
from flask import Flask
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import BadRequest
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters

# =========================================================
# 1) LOGGING & STARTUP HOOKS
# =========================================================
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
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
USE_OPENROUTER = os.getenv("USE_OPENROUTER", "True").lower() == "true"
OR_KEY = os.getenv("OPENROUTER_API_KEY")
OA_KEY = os.getenv("OPENAI_API_KEY")
GSHEET_WEBHOOK = os.getenv("GSHEET_WEBHOOK", "").strip()
LOG_SALT = os.getenv("LOG_SALT", "").strip()

logger.info("DEBUG => OR=%s | OA=%s | GSHEET=%s", bool(OR_KEY), bool(OA_KEY), bool(GSHEET_WEBHOOK))
httpx_client = httpx.Client(timeout=httpx.Timeout(connect=30, read=90, write=90, pool=90))

if USE_OPENROUTER:
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OR_KEY, http_client=httpx_client,
        default_headers={"HTTP-Referer": "https://t.me/SearchVocabBot", "X-Title": "School English Bot"})
    MODEL_NAME = "openai/gpt-4o-mini"
else:
    client = OpenAI(api_key=OA_KEY, http_client=httpx_client)
    MODEL_NAME = "gpt-3.5-turbo"

# =========================================================
# 3) CONSTANTS, HELPERS, POLICIES
# =========================================================
DEFAULT_LANG = "auto"
LOCKED_MODES = True   # chỉ hiện mode khi học sinh unlock
GRADE_TO_CEFR = {"6": "A2", "7": "A2+", "8": "B1-", "9": "B1"}
DEFAULT_DIALOGUE_LIMIT = 20
MAX_HISTORY = 10
BANNED_KEYWORDS = [r"\bsex\b", r"\bporn\b", r"\bsuicide\b", r"\bviolence\b"]
POLICY_CHAT = "You are a safe, school-appropriate assistant for grades 6–9. No markdown bold or headings."
POLICY_STUDY = "You are an English-learning assistant for grades 6–9 (A2–B1). Keep output short, safe, clear."
CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")

def detect_lang(text: str) -> str:
    return "ru" if CYRILLIC_RE.search(text or "") else "en"

def trim(s: str, max_chars=1000): 
    s = re.sub(r"\n{3,}", "\n\n", (s or "").strip())
    return s if len(s) <= max_chars else s[:max_chars]+"…"

def blocked(text: str) -> bool:
    return any(re.search(p, text or "", re.I) for p in BANNED_KEYWORDS)

async def ask_openai(messages, max_tokens=500, temperature=0.4, model=None):
    model = model or MODEL_NAME
    for _ in range(2):
        try:
            res = client.chat.completions.create(model=model, messages=messages,
                                                 max_tokens=max_tokens, temperature=temperature)
            return res.choices[0].message.content
        except Exception as e: err = e
    return f"[LLM error] {err}"

def extract_json(s: str):
    s = (s or "").strip()
    if "```" in s:
        parts = s.split("```")
        for block in parts[1:]:
            if block.lstrip().startswith("json"):
                try: return json.loads(block.split("\n",1)[1])
                except: pass
    try: return json.loads(s)
    except: return {}

def make_user_hash(user_id: object, salt: str) -> str:
    try: return hashlib.sha256(f"{user_id}|{salt}".encode()).hexdigest()[:12]
    except: return "unknown"

# =========================================================
# 4) USER PREFS / SESSION STATE
# =========================================================
user_prefs = {}
def get_prefs(uid: int):
    if uid not in user_prefs:
        user_prefs[uid] = {"mode":"chat","lang":DEFAULT_LANG,"grade":"7",
                           "cefr":GRADE_TO_CEFR["7"],"dialogue_limit":DEFAULT_DIALOGUE_LIMIT}
    return user_prefs[uid]

def remember_last_text(context,text:str):
    if text and len(text)>8: context.user_data["last_text"]=text.strip()

# =========================================================
# 5) GOOGLE SHEET LOGGING (ẩn danh)
# =========================================================
async def log_event(context,event,user_id,extra=None):
    if not GSHEET_WEBHOOK: return
    try:
        prefs=get_prefs(int(user_id))
        payload={"timestamp":datetime.now(timezone.utc).isoformat(),
                 "user_hash":make_user_hash(user_id,LOG_SALT),"event":event,
                 "mode":prefs.get("mode"),"lang":prefs.get("lang"),
                 "grade":prefs.get("grade"),"cefr":prefs.get("cefr"),"extra":extra or {}}
        sig=""
        if LOG_SALT:
            src=f"{payload['user_hash']}|{payload['event']}|{payload['timestamp']}|{LOG_SALT}"
            sig=hmac.new(LOG_SALT.encode(),src.encode(),hashlib.sha256).hexdigest()
        await asyncio.to_thread(httpx_client.post,GSHEET_WEBHOOK,json=payload,
            headers={"X-Log-Signature":sig},timeout=10,follow_redirects=True)
    except Exception as e: logger.warning("log_event failed: %s",e)

# =========================================================
# 6) SAFE SENDERS
# =========================================================
async def safe_reply_message(msg,text,reply_markup=None):
    try: return await msg.reply_text(text,reply_markup=reply_markup)
    except: return await msg.reply_text(text)

async def safe_edit_text(q,text,reply_markup=None):
    try:
        return await q.edit_message_text(text,reply_markup=reply_markup)
    except BadRequest:
        try: return await q.message.reply_text(text,reply_markup=reply_markup)
        except: return None

# =========================================================
# 7) INTENT LAYER (Unlock prompt detect)
# =========================================================
PROMPT_UNLOCK = {
    "vocab": re.compile(r"define the word|give me the meaning of", re.I),
    "reading": re.compile(r"write a short a2.*reading passage|translate this text into", re.I),
    "grammar": re.compile(r"explain|show me the grammar rule", re.I),
    "talk": re.compile(r"let'?s talk about|start a short english conversation", re.I)
}

def detect_unlock_prompt(text:str):
    for m,rx in PROMPT_UNLOCK.items():
        if rx.search(text): return m
    return None

# =========================================================
# 8) UI MENUS (root dynamic)
# =========================================================
def root_menu(lang="en", unlocked=None):
    unlocked = unlocked or {}
    rows=[]
    if not LOCKED_MODES or unlocked.get("vocab"):
        rows.append([InlineKeyboardButton("📚 Vocabulary", callback_data="menu:mode:vocab")])
    if not LOCKED_MODES or unlocked.get("reading"):
        rows.append([InlineKeyboardButton("📖 Reading", callback_data="menu:mode:reading")])
    if not LOCKED_MODES or unlocked.get("grammar"):
        rows.append([InlineKeyboardButton("⚙️ Grammar", callback_data="menu:mode:grammar")])
    if not LOCKED_MODES or unlocked.get("talk"):
        rows.append([InlineKeyboardButton("💬 Talk", callback_data="menu:mode:talk")])
    rows.append([InlineKeyboardButton("🏫 Grade", callback_data="menu:grade"),
                 InlineKeyboardButton("🌐 Language", callback_data="menu:lang")])
    rows.append([InlineKeyboardButton("📋 Help", callback_data="menu:roothelp")])
    return InlineKeyboardMarkup(rows)

# =========================================================
# 9) START / HELP
# =========================================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prefs=get_prefs(update.effective_user.id)
    prefs["mode"]="chat"
    context.user_data["unlocked_modes"]={}
    greet="Hi there! 👋 I’m your English study buddy. Use prompts to unlock study modes!"
    await safe_reply_message(update.message,greet,reply_markup=root_menu(prefs.get("lang","en")))
    await log_event(context,"start",update.effective_user.id,{"text":(update.message.text or "")[:200]})

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt=("Try prompts like:\n• Define the word 'set up'…\n• Write a short A2 reading passage…\n• Explain Present Perfect…\n• Let's talk about hobbies."
         "\nThese unlock learning modes!") 
    await safe_reply_message(update.message,txt,reply_markup=root_menu())
    await log_event(context,"help",update.effective_user.id,{})

# =========================================================
# 10) VOCAB / READING / GRAMMAR BUILDERS
# =========================================================
async def build_vocab_card(headword,prefs,user_text):
    prompt=(f"You are an English-learning assistant for grades 6–9 (A2–B1). "
            f"Make a compact vocabulary card with synonyms, antonyms, and 3 examples. "
            f"Word: {headword}")
    msgs=[{"role":"system","content":POLICY_STUDY},{"role":"user","content":prompt}]
    return await ask_openai(msgs,300)

async def build_reading_passage(topic,level,lang):
    prompt=(f"Write a short reading passage (80–120 words) about '{topic}', level {level}, "
            f"Language: {'Russian' if lang=='ru' else 'English'} (A2–B1).")
    return await ask_openai([{"role":"system","content":POLICY_STUDY},{"role":"user","content":prompt}],220)

async def build_reading_gloss(passage,lang):
    tgt="Russian" if lang!="ru" else "English"
    prompt=(f"Return a glossed version: underline 10–15 phrases with _underscores_ and add short {tgt} hints. Do not translate everything.\n\n{passage}")
    return await ask_openai([{"role":"system","content":POLICY_STUDY},{"role":"user","content":prompt}],350)

def normalize_answer(s:str):
    s=re.sub(r"[^\w\s'-]","",s.lower().strip())
    return re.sub(r"\s+"," ",s)

def normalize_answer_multi(s:str):
    parts=re.split(r"\s*(?:/|;| or )\s*",s,flags=re.I)
    return [normalize_answer(p) for p in parts if p.strip()]

def fuzzy_equal(a,b,t=0.85): 
    return difflib.SequenceMatcher(a=normalize_answer(a),b=normalize_answer(b)).ratio()>=t
# =========================================================
# 11) PRACTICE ENGINE (Shared)
# =========================================================
def fix_mcq_item(q: dict):
    opts = q.get("options", ["", "", "", ""])
    ans = str(q.get("answer", "A")).strip().upper()
    if ans not in ("A", "B", "C", "D"):
        try:
            idx = [normalize_answer(o) for o in opts].index(normalize_answer(ans))
            ans = "ABCD"[idx]
        except ValueError:
            ans = "A"
    q["answer"] = ans
    return q

async def send_practice_item(update_or_query, context: ContextTypes.DEFAULT_TYPE):
    st = context.user_data.get("practice")
    if not st: return
    qitem = st["items"][st["idx"]]
    qnum = f"Q{st['idx']+1}/{len(st['items'])}"
    text = f"{qnum}\n\n{qitem['question']}"
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton(f"A) {qitem['options'][0]}", callback_data="ans:A"),
         InlineKeyboardButton(f"B) {qitem['options'][1]}", callback_data="ans:B")],
        [InlineKeyboardButton(f"C) {qitem['options'][2]}", callback_data="ans:C"),
         InlineKeyboardButton(f"D) {qitem['options'][3]}", callback_data="ans:D")]
    ])
    if isinstance(update_or_query, Update):
        await safe_reply_message(update_or_query.message, text, reply_markup=kb)
    else:
        await safe_edit_text(update_or_query, text, reply_markup=kb)

async def practice_summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    st = context.user_data.get("practice")
    if not st: return
    lang = st.get("ui_lang","en")
    total = len(st["items"])
    score = st.get("score",0)
    lines = [f"{'Summary' if lang!='ru' else 'Итоги'}: {score}/{total}",
             "Answers and explanations:" if lang!='ru' else "Ответы и пояснения:"]
    for it in st["items"]:
        expl = it["explain_ru"] if lang=="ru" and it["explain_ru"] else it["explain_en"]
        lines.append(f"Q{it['id']}: {it['answer']} — {expl}")
    await safe_reply_message(update.message,"\n".join(lines))
    await log_event(context,"practice_done",update.effective_user.id,{
        "type":st["type"],"topic":st.get("topic"),"score":score,"total":total})
    # Footer
    scope = st.get("scope","free")
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("🔁 Again", callback_data="footer:more_practice")],
        [InlineKeyboardButton("🏠 Menu", callback_data="menu:root")]
    ])
    await safe_reply_message(update.message,"—",reply_markup=kb)
    context.user_data.pop("practice",None)

# =========================================================
# 12) TALK COACH
# =========================================================
async def talk_reply(user_text:str, topic:str, ui_lang:str):
    prompt = (
        f"You are a friendly English conversation coach (A2–B1). Topic: {topic}. "
        "Respond in 1–3 sentences; correct gently; suggest 1–2 useful words."
    )
    msgs = [{"role":"system","content":POLICY_STUDY},
            {"role":"user","content":f"Student: {user_text}"}]
    return await ask_openai([{"role":"system","content":prompt},*msgs],180)

# =========================================================
# 13) OPTIONAL COMMANDS
# =========================================================
async def logtest_cmd(update:Update,context:ContextTypes.DEFAULT_TYPE):
    await log_event(context,"logtest",update.effective_user.id,{"ping":"pong"})
    await safe_reply_message(update.message,"Logtest sent.")

# =========================================================
# 14) CALLBACK HANDLER (INLINE BUTTONS)
# =========================================================
async def on_cb(update:Update,context:ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    data = q.data or ""
    await q.answer()
    uid = update.effective_user.id
    prefs = get_prefs(uid)
    lang = prefs.get("lang","en")
    unlocked = context.user_data.get("unlocked_modes",{})

    # basic menus
    if data=="menu:root":
        await safe_edit_text(q,"Back to menu.",reply_markup=root_menu(lang,unlocked)); return
    if data.startswith("menu:mode:"):
        prefs["mode"]=data.split(":")[-1]
        await log_event(context,"mode_set",uid,{"mode":prefs["mode"]})
        txt=f"{prefs['mode'].capitalize()} mode unlocked. Send your request!"
        await safe_edit_text(q,txt,reply_markup=root_menu(lang,unlocked)); return

    # answer MCQ
    if data.startswith("ans:"):
        st=context.user_data.get("practice")
        if not st: return await safe_edit_text(q,"No active quiz.")
        ch=data.split(":")[1]
        qitem=st["items"][st["idx"]]
        correct=qitem["answer"]
        if ch==correct:
            st["score"]+=1; msg="✅ Correct!"
        else: msg=f"❌ Correct answer: {correct}"
        await safe_edit_text(q,msg)
        st["idx"]+=1
        if st["idx"]>=len(st["items"]):
            dummy=Update(update.update_id,message=q.message)
            await practice_summary(dummy,context)
        else: await send_practice_item(q,context)
        return

# =========================================================
# 15) FREE TEXT HANDLER (Unlock + Logic)
# =========================================================
async def handle_message(update:Update,context:ContextTypes.DEFAULT_TYPE):
    text=update.message.text or ""
    if blocked(text): return await safe_reply_message(update.message,"⛔ Inappropriate topic.")
    uid=update.effective_user.id
    prefs=get_prefs(uid)
    unlocked=context.user_data.get("unlocked_modes",{})

    # check unlock prompt
    unlock=detect_unlock_prompt(text)
    if unlock and (unlock not in unlocked):
        unlocked[unlock]=True
        context.user_data["unlocked_modes"]=unlocked
        await safe_reply_message(update.message,f"✨ You’ve unlocked {unlock.capitalize()} mode!",
                                 reply_markup=root_menu(prefs.get('lang','en'),unlocked))
        await log_event(context,"mode_unlocked",uid,{"mode":unlock,"prompt":text[:200]})
        return

    # MODE: vocab
    if prefs["mode"]=="vocab" or unlocked.get("vocab") and re.search(r"\bdefine\b",text,re.I):
        word=re.sub(r"[^A-Za-z\s'-]","",text).strip().split()[-1]
        card=await build_vocab_card(word,prefs,text)
        context.user_data["last_word"]=word
        await log_event(context,"chat_message",uid,{"chars":len(text)})
        return await safe_reply_message(update.message,trim(card))

    # MODE: reading
    if prefs["mode"]=="reading" or unlocked.get("reading") and "reading" in text.lower():
        topic=text.strip() or "school life"
        passage=await build_reading_passage(topic,prefs["cefr"],prefs.get("lang","en"))
        context.user_data["reading"]={"topic":topic,"last_passage":passage}
        await log_event(context,"reading_passage",uid,{"topic":topic})
        return await safe_reply_message(update.message,trim(passage))

    # MODE: grammar
    if prefs["mode"]=="grammar" or unlocked.get("grammar") and re.search(r"grammar|tense|rule",text,re.I):
        topic=text.strip() or "Present Simple"
        g_prompt=(f"Explain grammar point '{topic}' for A2–B1 with 5–7 bullets, ✓✗ examples, signal words.")
        exp=await ask_openai([{"role":"system","content":POLICY_STUDY},{"role":"user","content":g_prompt}],350)
        context.user_data["last_grammar_topic"]=topic
        await log_event(context,"grammar_explain",uid,{"topic":topic})
        return await safe_reply_message(update.message,trim(exp))

    # MODE: talk
    if prefs["mode"]=="talk" or unlocked.get("talk") and re.search(r"talk|conversation|chat",text,re.I):
        tstate=context.user_data.get("talk",{"topic":"daily life","turns":0})
        reply=await talk_reply(text,tstate["topic"],prefs.get("lang","en"))
        tstate["turns"]+=1; context.user_data["talk"]=tstate
        await log_event(context,"chat_message",uid,{"chars":len(text)})
        return await safe_reply_message(update.message,trim(reply))

    # default chat
    msgs=[{"role":"system","content":POLICY_CHAT},{"role":"user","content":text}]
    out=await ask_openai(msgs,300)
    await safe_reply_message(update.message,trim(out))
    await log_event(context,"chat_message",uid,{"chars":len(text)})

# =========================================================
# 16) FLASK HEALTHCHECK
# =========================================================
app = Flask(__name__)
@app.get("/")
def health(): return "✅ Bot is alive",200
def start_flask():
    port=int(os.getenv("PORT","10000"))
    app.run(host="0.0.0.0",port=port)

# =========================================================
# 17) MAIN ENTRYPOINT
# =========================================================
def main():
    application=Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start",start))
    application.add_handler(CommandHandler("help",help_cmd))
    application.add_handler(CommandHandler("logtest",logtest_cmd))
    application.add_handler(CallbackQueryHandler(on_cb))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND,handle_message))
    application.add_error_handler(on_error)
    application.post_init=on_startup
    threading.Thread(target=start_flask,daemon=True).start()
    logger.info("Bot is starting…")
    application.run_polling(allowed_updates=Update.ALL_TYPES,drop_pending_updates=True)

if __name__=="__main__": main()
