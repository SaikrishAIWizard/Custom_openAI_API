import os
import json
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from crewai import LLM, Agent, Task, Crew
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from telegram.constants import ChatAction

# ---------------- ENV SETUP ----------------
load_dotenv()
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

# ---------------- APP CONFIG ----------------
app = FastAPI(title="Product Multi-Agent API & Bot")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

user_sessions = {}

# ---------------- LLM INIT ----------------
llm = LLM(
    model="openai/openai/gpt-oss-120b",
    api_key=os.getenv("Custom_OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_COMPATIBLE_ENDPOINT"),
)

# ---------------- AGENTS & LOGIC ----------------

def clean_output(text: str):
    """Ensures no literal slash-n characters exist and removes surrounding quotes"""
    return text.strip().replace("\\n", "\n").replace('"', '')

def run_formatter_crew(product_input: str):
    agent = Agent(
        role="Professional Product Formatter",
        goal="Extract data into a clean pipe-separated format with physical line breaks.",
        backstory="You generate text meant for direct copy-pasting into web databases.",
        llm=llm
    )
    task = Task(
        description=(
            "Extract data into this EXACT pipe-separated format:\n"
            "Name | Price | Description | URLs | Sizes\n\n"
            "RULES:\n"
            "- NAME: '<Quantity> <Brand> <Product Type> <Fabric>'.\n"
            "- PRICE: Base price + 20%, whole number only.\n"
            "- DESCRIPTION: Use physical Enter/Return keys for line breaks. NO '\\n' text.\n"
            "- SIZES: Separate components (e.g., M, 38).\n"
            "- OUTPUT: Provide ONLY the pipe-separated line. No labels like 'Result:' or 'Output:'.\n"
            f"INPUT DATA:\n{product_input}"
        ),
        agent=agent,
        expected_output="A single pipe-separated line with no extra text."
    )
    result = str(Crew(agents=[agent], tasks=[task]).kickoff())
    return clean_output(result)

def run_instagram_crew(product_input: str):
    agent = Agent(
        role="Instagram Content Creator",
        goal="Create viral captions for Premalatha Collections.",
        backstory="You provide the final caption text ready for Instagram. No intro/outro text.",
        llm=llm
    )
    task = Task(
        description=(
            "Convert input into a creative Instagram caption.\n"
            "Header: ✨ PREMALATHA COLLECTIONS: NEW ARRIVALS ✨\n"
            "Body: Emoji-rich, urgent, using physical line breaks for spacing.\n"
            "Footer: Relevant hashtags and 'DM to order!'.\n"
            "OUTPUT: Provide ONLY the caption. Do not include any meta-talk or labels.\n"
            f"INPUT DATA:\n{product_input}"
        ),
        agent=agent,
        expected_output="The final Instagram caption only."
    )
    result = str(Crew(agents=[agent], tasks=[task]).kickoff())
    return clean_output(result)

# ---------------- TELEGRAM UI ----------------

def get_main_menu():
    keyboard = [
        [InlineKeyboardButton("📊 Format Product (Pipe)", callback_data="mode_format")],
        [InlineKeyboardButton("📸 Create Instagram Post", callback_data="mode_instagram")],
        [InlineKeyboardButton("🧹 Reset", callback_data="reset")]
    ]
    return InlineKeyboardMarkup(keyboard)

# ---------------- TELEGRAM HANDLERS ----------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_sessions[user_id] = {"description": "", "urls": "", "step": None, "mode": None}
    await update.message.reply_text("Choose a tool:", reply_markup=get_main_menu())

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    await query.answer()

    if query.data == "mode_format":
        user_sessions[user_id] = {"description": "", "urls": "", "step": "desc", "mode": "format"}
        await query.edit_message_text("1/2: Paste Description")
    elif query.data == "mode_instagram":
        user_sessions[user_id] = {"description": "", "urls": "", "step": "desc", "mode": "instagram"}
        await query.edit_message_text("1/2: Paste Details")
    elif query.data == "reset":
        user_sessions[user_id] = {"description": "", "urls": "", "step": None, "mode": None}
        await query.edit_message_text("Choose a tool:", reply_markup=get_main_menu())

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text
    session = user_sessions.get(user_id)

    if not session or session["step"] is None:
        await update.message.reply_text("Select a mode:", reply_markup=get_main_menu())
        return

    if session["step"] == "desc":
        session["description"] = text
        session["step"] = "urls"
        await update.message.reply_text("2/2: Paste URLs (or 'none')")
    
    elif session["step"] == "urls":
        session["urls"] = text
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
        
        combined_data = f"DESCRIPTION:\n{session['description']}\n\nURLS:\n{session['urls']}"
        
        try:
            if session["mode"] == "format":
                final_text = run_formatter_crew(combined_data)
            else:
                final_text = run_instagram_crew(combined_data)
            
            # Sending ONLY the content for easy copy-paste
            await update.message.reply_text(final_text)
        except Exception as e:
            await update.message.reply_text(f"Error: {str(e)}")
        
        session["step"] = None
        # Send menu separately so it doesn't get copied with the text
        await update.message.reply_text("--- Processed ---", reply_markup=get_main_menu())

# ---------------- STARTUP ----------------

@app.on_event("startup")
async def start_bot():
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(handle_callback))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    await application.initialize()
    await application.start()
    await application.updater.start_polling()

@app.get("/health")
def health(): return {"status": "ok"}