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

# user_sessions = { user_id: {"description": "", "urls": "", "step": None, "mode": "format" | "instagram"} }
user_sessions = {}

# ---------------- LLM INIT ----------------
llm = LLM(
    model="openai/openai/gpt-oss-120b",
    api_key=os.getenv("Custom_OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_COMPATIBLE_ENDPOINT"),
)

# ---------------- AGENTS & LOGIC ----------------

def run_formatter_crew(product_input: str):
    """Agent for the pipe-separated professional format"""
    agent = Agent(
        role="Professional Product Formatter",
        goal="Extract data into an exact pipe-separated format.",
        backstory="High-speed e-commerce data specialist. Strictly follows formatting rules.",
        llm=llm
    )
    task = Task(
        description=(
            "You are a professional product formatter. Extract data into this EXACT pipe-separated format:\n"
            "Name | Price | Description | URLs | Sizes\n\n"
            "RULES:\n"
            "- NAME: '<Quantity> <Brand> <Product Type> <Fabric>'.\n"
            "- PRICE: Base price + 20%, whole number only.\n"
            "- DESCRIPTION: Engaging, emojis, '\\n' for line breaks.\n"
            "- SIZES: Separate components (e.g., M38 -> 'M, 38').\n"
            f"INPUT DATA:\n{product_input}"
        ),
        agent=agent,
        expected_output="A single pipe-separated line only."
    )
    return str(Crew(agents=[agent], tasks=[task]).kickoff()).strip()

def run_instagram_crew(product_input: str):
    """Agent for Creative Instagram Captions"""
    agent = Agent(
        role="Instagram Content Creator",
        goal="Convert product lists into viral, high-energy Instagram captions.",
        backstory="Social media manager for top fashion brands. Expert in emojis and call-to-actions.",
        llm=llm
    )
    task = Task(
        description=(
            "Transform the following product data into a professional and creative Instagram caption.\n\n"
            "STYLE RULES:\n"
            "- Use '✨ PREMALATHA COLLECTIONS: NEW ARRIVALS ✨' as the header.\n"
            "- Use stylish bullet points (🛍️, 🔥, ⚡) for each product.\n"
            "- Highlight key features (like 'Premium Suede', '450 GSM', 'Bell Bottom').\n"
            "- Keep pricing clear with the ₹ symbol.\n"
            "- Add a strong Call to Action (CTA) like 'DM to order' or 'Link in bio'.\n"
            "- Include 5-8 relevant fashion hashtags at the end.\n"
            f"INPUT DATA:\n{product_input}"
        ),
        agent=agent,
        expected_output="A full creative Instagram caption with emojis and hashtags."
    )
    return str(Crew(agents=[agent], tasks=[task]).kickoff()).strip()

# ---------------- TELEGRAM UI ----------------

def get_main_menu():
    keyboard = [
        [InlineKeyboardButton("📊 Format Product (Pipe-Separated)", callback_data="mode_format")],
        [InlineKeyboardButton("📸 Create Instagram Post", callback_data="mode_instagram")],
        [InlineKeyboardButton("🧹 Clear/Reset Session", callback_data="reset")]
    ]
    return InlineKeyboardMarkup(keyboard)

# ---------------- TELEGRAM HANDLERS ----------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_sessions[user_id] = {"description": "", "urls": "", "step": None, "mode": None}
    await update.message.reply_text(
        "Welcome to the Product Management Bot! 🚀\nWhat would you like to do today?",
        reply_markup=get_main_menu()
    )

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    await query.answer()

    if query.data == "mode_format":
        user_sessions[user_id] = {"description": "", "urls": "", "step": "desc", "mode": "format"}
        await query.edit_message_text("🛠 **Pipe-Formatter Mode**\nStep 1: Paste the Product Description.")
    
    elif query.data == "mode_instagram":
        user_sessions[user_id] = {"description": "", "urls": "", "step": "desc", "mode": "instagram"}
        await query.edit_message_text("🎨 **Instagram Post Mode**\nStep 1: Paste the Product Details/Description.")
    
    elif query.data == "reset":
        user_sessions[user_id] = {"description": "", "urls": "", "step": None, "mode": None}
        await query.edit_message_text("Session cleared. Choose a mode:", reply_markup=get_main_menu())

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text
    session = user_sessions.get(user_id)

    if not session or session["step"] is None:
        await update.message.reply_text("Please select a mode first:", reply_markup=get_main_menu())
        return

    if session["step"] == "desc":
        session["description"] = text
        session["step"] = "urls"
        await update.message.reply_text("✅ Description received.\nStep 2: Paste the **URLs/Links** (or type 'none').")
    
    elif session["step"] == "urls":
        session["urls"] = text
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
        
        combined_data = f"DESCRIPTION:\n{session['description']}\n\nURLS:\n{session['urls']}"
        
        try:
            if session["mode"] == "format":
                result = run_formatter_crew(combined_data)
            else:
                result = run_instagram_crew(combined_data)
            
            await update.message.reply_text(f"✨ **Output:**\n\n{result}")
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {str(e)}")
        
        # Reset and show menu again
        session["step"] = None
        await update.message.reply_text("Process complete!", reply_markup=get_main_menu())

# ---------------- BOT & API STARTUP ----------------

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