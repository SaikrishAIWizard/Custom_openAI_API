import os
import json
import asyncio
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from crewai import LLM, Agent, Task, Crew
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from telegram.constants import ChatAction

# ---------------- ENV SETUP ----------------
load_dotenv()
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
APP_URL = os.getenv("APP_URL") 
user_sessions = {}

# ---------------- LLM INIT ----------------
llm = LLM(
    model="openai/openai/gpt-oss-120b",
    api_key=os.getenv("Custom_OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_COMPATIBLE_ENDPOINT"),
)

# ---------------- KEEP ALIVE HEARTBEAT ----------------
async def keep_alive_task():
    if not APP_URL: return
    async with httpx.AsyncClient() as client:
        while True:
            await asyncio.sleep(600)
            try:
                await client.get(f"{APP_URL}/health")
            except: pass

# ---------------- AGENTS & LOGIC ----------------

def run_single_formatter(product_input: str):
    agent = Agent(
        role="E-commerce Single Product Formatter",
        goal=(
            "Extract and normalize ONLY the single-piece version of a product "
            "into a strict, database-ready format."
        ),
        backstory=(
            "You are a precision-driven e-commerce catalog specialist. "
            "You clean raw seller inputs and convert them into structured, "
            "professional product listings suitable for websites and apps. "
            "You follow formatting rules exactly and never add extra text."
        ),
        llm=llm
    )

    task = Task(
        description=(
            "Convert the product information below into **ONE SINGLE LINE** "
            "using the EXACT pipe-separated format:\n\n"
            "Name | Price | Description | URLs | Sizes\n\n"

            "STRICT & NON-NEGOTIABLE RULES:\n"
            "1. OUTPUT must be exactly ONE line only.\n"
            "2. Extract ONLY the SINGLE-PIECE version. Ignore combo/bulk offers.\n\n"

            "FIELD RULES:\n"
            "NAME:\n"
            "- Format strictly as:\n"
            "  '1 pc <Brand> <Product Type> <Fabric>'\n"
            "- Example: '1 pc MIX Cotton Casual Shirt'\n\n"

            "PRICE:\n"
            "- Identify the single-piece price from input\n"
            "- Increase it by 15%\n"
            "- Round to the nearest whole number\n"
            "- Output digits only (no ₹, no symbols, no text)\n\n"

            "DESCRIPTION:\n"
            "- Write a clean, professional, customer-friendly description\n"
            "- 2–3 short lines MAX (use real line breaks)\n"
            "- ABSOLUTELY DO NOT include:\n"
            "  prices, currency symbols, shipping info, offers, emojis, hashtags, sizes\n\n"

            "URLS:\n"
            "- Include ONLY image or video URLs\n"
            "- Raw links, comma-separated, no spaces\n\n"

            "SIZES:\n"
            "- List each size as a separate value\n"
            "- Example: 'M38' → 'M,38'\n"
            "- Example: 'M38 L40 XL42' → 'M,38,L,40,XL,42'\n"
            "- Output comma-separated with no spaces\n\n"

            "ABSOLUTE PROHIBITIONS:\n"
            "- No headings\n"
            "- No markdown\n"
            "- No quotes\n"
            "- No explanations\n"
            "- No multiple lines outside the Description field\n\n"

            f"INPUT DATA:\n{product_input}"
        ),
        agent=agent,
        expected_output="One single pipe-separated product line."
    )

    result = Crew(
        agents=[agent],
        tasks=[task]
    ).kickoff()

    return (
        str(result)
        .replace("\\n", " ")
        .replace('"', "")
        .replace("Output:", "")
        .replace("Result:", "")
        .strip()
    )

def run_combo_formatter(product_input: str):
    agent = Agent(
        role="E-commerce Combo Product Formatter",
        goal=(
            "Extract and format ONLY the combo/pack version of a product "
            "into a strict, database-ready single-line format."
        ),
        backstory=(
            "You are a precision-focused e-commerce catalog specialist. "
            "You convert messy seller inputs into structured combo listings. "
            "You follow formatting rules exactly and never add extra text."
        ),
        llm=llm
    )

    task = Task(
        description=(
            "Convert the product information below into **ONE SINGLE LINE** "
            "using the EXACT pipe-separated format:\n\n"
            "Name | Price | Description | URLs | Sizes\n\n"

            "STRICT & NON-NEGOTIABLE RULES:\n"
            "1. OUTPUT must be exactly ONE line (no bullets, no explanations).\n"
            "2. Use ONLY combo/pack details. Ignore single-piece offers except where instructed.\n\n"

            "FIELD RULES:\n"
            "NAME:\n"
            "- Format strictly as:\n"
            "  '<Quantity> pcs COMBO <Brand> <Product Type> <Fabric>'\n"
            "- Example: '3 pcs COMBO MIX Cotton Casual Shirts'\n\n"

            "PRICE:\n"
            "- Identify the combo price from input\n"
            "- Increase it by 15%\n"
            "- Round to the nearest whole number\n"
            "- Output digits only (no ₹, no text)\n\n"

            "DESCRIPTION:\n"
            "- Write a premium, professional description (2–3 short lines max)\n"
            "- MUST include exactly one sentence:\n"
            "  'Single piece also available at ₹<Single Price + 15%>'\n"
            "- Do NOT mention any other prices or calculations\n\n"

            "URLS:\n"
            "- Include ONLY product image or video URLs\n"
            "- Comma-separated, no spaces\n\n"

            "SIZES:\n"
            "- Comma-separated (example: M,L,XL,XXL)\n"
            "- If missing, infer from brand standard sizing\n\n"

            "ABSOLUTE PROHIBITIONS:\n"
            "- No headings\n"
            "- No quotes\n"
            "- No markdown\n"
            "- No extra commentary\n"
            "- No multiple lines\n\n"

            f"INPUT DATA:\n{product_input}"
        ),
        agent=agent,
        expected_output="One single pipe-separated combo product line."
    )

    result = Crew(
        agents=[agent],
        tasks=[task]
    ).kickoff()

    return (
        str(result)
        .replace("\\n", " ")
        .replace('"', "")
        .replace("Output:", "")
        .replace("Result:", "")
        .strip()
    )

def run_instagram_crew(product_input: str):
    agent = Agent(
        role="Senior Instagram Marketing Copywriter",
        goal=(
            "Write high-conversion Instagram captions that increase reach, saves, "
            "and orders for fashion products."
        ),
        backstory=(
            "You are an expert social media copywriter specializing in Instagram "
            "fashion marketing for Indian brands. You understand reels, hooks, "
            "emojis, scarcity tactics, and CTAs that drive DMs and orders. "
            "You write concise, catchy, and scroll-stopping captions."
        ),
        llm=llm
    )

    task = Task(
        description=(
            "Create a creative, sales-focused Instagram caption for "
            "Premalatha Collections using the product details below.\n\n"
            f"{product_input}\n\n"
            "Rules:\n"
            "- Start with a strong hook (first line must grab attention)\n"
            "- Use relevant emojis (not excessive)\n"
            "- Highlight fabric, quality, price, sizes, and colors\n"
            "- Add urgency or offer (if applicable)\n"
            "- End with a clear CTA (DM to order / Limited stock / Free shipping)\n"
            "- Add 5–8 relevant fashion hashtags\n"
            "- Use line breaks for readability\n"
            "- Do NOT include explanations, quotes, or headings\n"
            "- Output ONLY the final caption text"
        ),
        agent=agent,
        expected_output="A polished Instagram caption ready to post."
    )

    result = Crew(
        agents=[agent],
        tasks=[task]
    ).kickoff()

    return (
        str(result)
        .replace("\\n", "\n")
        .replace('"', '')
        .strip()
    )


# ---------------- LIFESPAN MANAGER ----------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    await application.bot.set_my_commands([
        BotCommand("start", "🚀 Dashboard"),
        BotCommand("menu", "📋 Main Menu"),
        BotCommand("help", "❓ How to use")
    ])
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("menu", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CallbackQueryHandler(handle_callback))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    await application.initialize()
    await application.start()
    await application.updater.start_polling()
    asyncio.create_task(keep_alive_task())
    yield
    await application.updater.stop()
    await application.stop()

app = FastAPI(title="Premalatha API", lifespan=lifespan)

# ---------------- UI & HANDLERS ----------------

def get_main_menu():
    keyboard = [
        [InlineKeyboardButton("👕 Single PC (Pipe)", callback_data="mode_single")],
        [InlineKeyboardButton("📦 Combo Pack (Pipe)", callback_data="mode_combo")],
        [InlineKeyboardButton("📸 Instagram Post", callback_data="mode_instagram")],
        [InlineKeyboardButton("🧹 Reset", callback_data="reset")]
    ]
    return InlineKeyboardMarkup(keyboard)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_sessions[user_id] = {"step": None, "mode": None}
    await update.message.reply_text("🌟 **Premalatha Collections Dashboard**", reply_markup=get_main_menu())

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "❓ **How to use Premalatha Bot:**\n\n"
        "👕 **Single PC Mode:**\n"
        "1. Send product details.\n"
        "2. Send URLs.\n"
        "Result: Pipe format for 1pc listing (+15% price).\n\n"
        "📦 **Combo Pack Mode:**\n"
        "1. Send product details.\n"
        "2. Send URLs.\n"
        "Result: Pipe format for Combo listing. Description will automatically mention the single piece price.\n\n"
        "📸 **Instagram Mode:**\n"
        "Send all info in ONE message.\n"
        "Result: A creative, emoji-rich caption ready for Instagram."
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    await query.answer()

    if query.data in ["mode_single", "mode_combo"]:
        mode = "single" if query.data == "mode_single" else "combo"
        user_sessions[user_id] = {"description": "", "urls": "", "step": "desc", "mode": mode}
        await query.edit_message_text(f"🛠 **{mode.upper()} Mode**\nStep 1/2: Paste Product Details:")
    elif query.data == "mode_instagram":
        user_sessions[user_id] = {"step": "insta_single", "mode": "instagram"}
        await query.edit_message_text("📸 **Instagram Mode**\nPaste all details here:")
    elif query.data == "reset":
        user_sessions[user_id] = {"step": None, "mode": None}
        await query.edit_message_text("Select a tool:", reply_markup=get_main_menu())

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    session = user_sessions.get(user_id)
    if not session or session.get("step") is None:
        return

    text = update.message.text
    if session["mode"] in ["single", "combo"]:
        if session["step"] == "desc":
            session["description"] = text
            session["step"] = "urls"
            await update.message.reply_text("✅ Details saved.\nStep 2/2: Paste URLs (or 'none'):")
        elif session["step"] == "urls":
            session["urls"] = text
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
            combined = f"DATA: {session['description']}\nURLS: {session['urls']}"
            try:
                res = run_single_formatter(combined) if session["mode"] == "single" else run_combo_formatter(combined)
                await update.message.reply_text(res)
            except Exception as e:
                await update.message.reply_text(f"Error: {e}")
            session["step"] = None
            await update.message.reply_text("Done!", reply_markup=get_main_menu())
    elif session["mode"] == "instagram":
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
        try:
            res = run_instagram_crew(text)
            await update.message.reply_text(res)
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")
        session["step"] = None
        await update.message.reply_text("Done!", reply_markup=get_main_menu())

@app.get("/health")
def health(): return {"status": "ok"}