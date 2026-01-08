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
from telegram import BotCommand

# ---------------- ENV SETUP ----------------
load_dotenv()
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

app = FastAPI(title="Product Multi-Agent API & Bot")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
user_sessions = {}

llm = LLM(
    model="openai/openai/gpt-oss-120b",
    api_key=os.getenv("Custom_OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_COMPATIBLE_ENDPOINT"),
)

# ---------------- AGENTS & LOGIC ----------------

def run_formatter_crew(product_input: str):
    agent = Agent(
        role="E-commerce Catalog Specialist",
        goal="Extract product data into a professional format while filtering out junk.",
        backstory="You are an expert at creating clean website listings. You strictly separate data fields and never mix price/size into the description.",
        llm=llm
    )
    
    task = Task(
        description=(
            "You are a professional product formatter. Convert the input into this EXACT pipe-separated format:\n"
            "Name | Price | Description | URLs | Sizes\n\n"
            "STRICT RULES:\n"
            "1. NAME: Format as '<Quantity> <Brand> <Product Type> <Fabric>'. Use 1 for Quantity if not specified.\n"
            "2. PRICE: Find the numerical price, add 15%, and output ONLY the whole number. No symbols.\n"
            "3. DESCRIPTION: Rewrite into a premium, engaging description with emojis. \n"
            "   ⚠️ ABSOLUTE FORBIDDEN: Do NOT include prices, currency, sizes, or shipping info in the description. \n"
            "   Use actual physical line breaks for each sentence. Do NOT use '\\n' text.\n"
            "4. URLs: List raw links separated by commas. If none, write 'none'.\n"
            "5. SIZES: List individually (e.g., M, L, XL, XXL). Split 'M38' into 'M, 38'.\n"
            "6. OUTPUT: One single pipe-separated line only.\n\n"
            f"INPUT DATA:\n{product_input}"
        ),
        agent=agent,
        expected_output="A single pipe-separated line only."
    )
    
    result = str(Crew(agents=[agent], tasks=[task]).kickoff()).strip()
    return result.replace("\\n", "\n").replace('"', '').replace("Output:", "").replace("Result:", "").strip()

def run_instagram_crew(product_input: str):
    agent = Agent(
        role="Social Media Copywriter",
        goal="Create high-conversion Instagram captions for Premalatha Collections.",
        backstory="You are a creative mastermind. You take raw product info and turn it into a high-energy, emoji-rich post that drives sales.",
        llm=llm
    )
    task = Task(
        description=(
            "Transform the input into a creative Instagram caption.\n"
            "Header: ✨ PREMALATHA COLLECTIONS: NEW ARRIVALS ✨\n"
            "Style: High-energy, emojis (🛍️, 🔥, 🧵, 💎). Use physical line breaks for readability.\n"
            "Price: Include the final calculated price (Original).\n"
            "Branding: Mention 'Premium Quality Guaranteed' and 'Stock Moving Fast!'.\n"
            "Footer: DM to order! 📥 #PremalathaCollections #PremiumFashion #NewArrivals\n"
            "⚠️ OUTPUT: Final caption ONLY. No intro or internal notes.\n\n"
            f"INPUT DATA:\n{product_input}"
        ),
        agent=agent,
        expected_output="The final polished Instagram caption only."
    )
    result = str(Crew(agents=[agent], tasks=[task]).kickoff()).strip()
    return result.replace("\\n", "\n").replace('"', '').strip()

# ---------------- TELEGRAM UI & HANDLERS ----------------

def get_main_menu():
    keyboard = [[InlineKeyboardButton("📊 Format (Pipe)", callback_data="mode_format")],
                [InlineKeyboardButton("📸 Instagram Post", callback_data="mode_instagram")],
                [InlineKeyboardButton("🧹 Reset", callback_data="reset")]]
    return InlineKeyboardMarkup(keyboard)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_sessions[user_id] = {"description": "", "urls": "", "step": None, "mode": None}
    await update.message.reply_text("🌟 **Premalatha Collections Bot**\nSelect a tool:", reply_markup=get_main_menu())

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    await query.answer()

    if query.data == "mode_format":
        user_sessions[user_id] = {"description": "", "urls": "", "step": "desc", "mode": "format"}
        await query.edit_message_text("🛠 **Pipe Formatter**\nStep 1/2: Paste Product Details")
    
    elif query.data == "mode_instagram":
        # SINGLE STEP for Instagram
        user_sessions[user_id] = {"description": "", "urls": "", "step": "insta_single", "mode": "instagram"}
        await query.edit_message_text("📸 **Instagram Post Creator**\nPaste all product info here:")
    
    elif query.data == "reset":
        user_sessions[user_id] = {"description": "", "urls": "", "step": None, "mode": None}
        await query.edit_message_text("Reset complete. Select tool:", reply_markup=get_main_menu())

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text
    session = user_sessions.get(user_id)

    if not session or session["step"] is None:
        await update.message.reply_text("Please select a mode:", reply_markup=get_main_menu())
        return

    # --- FORMATTER MODE (2 Steps) ---
    if session["mode"] == "format":
        if session["step"] == "desc":
            session["description"] = text
            session["step"] = "urls"
            await update.message.reply_text("✅ Details saved. Step 2/2: Paste URLs (or 'none')")
        elif session["step"] == "urls":
            session["urls"] = text
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
            combined_data = f"DATA:\n{session['description']}\nURLS:\n{session['urls']}"
            try:
                final_text = run_formatter_crew(combined_data)
                await update.message.reply_text(final_text)
            except Exception as e:
                await update.message.reply_text(f"Error: {str(e)}")
            session["step"] = None
            await update.message.reply_text("--- ✅ Done ---", reply_markup=get_main_menu())

    # --- INSTAGRAM MODE (1 Step) ---
    elif session["mode"] == "instagram":
        if session["step"] == "insta_single":
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
            try:
                final_text = run_instagram_crew(text)
                await update.message.reply_text(final_text)
            except Exception as e:
                await update.message.reply_text(f"Error: {str(e)}")
            session["step"] = None
            await update.message.reply_text("--- ✨ Done ---", reply_markup=get_main_menu())

# ---------------- STARTUP ----------------
@app.on_event("startup")
async def start_bot():

    # Initialize the application
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    
    # 1. SET BOT COMMANDS (The part you requested)
    # This makes the "Menu" button appear in the bottom left of the chat
    await application.bot.set_my_commands([
        BotCommand("start", "🚀 Start the Bot"),
        #BotCommand("start", "📋 Dashboard")
    ])

    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(handle_callback))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    await application.initialize()
    await application.start()
    await application.updater.start_polling()

@app.get("/health")
def health(): return {"status": "ok"}


# # ... (rest of your imports and agent logic remain the same) ...

# # ---------------- STARTUP ----------------
# @app.on_event("startup")
# async def start_bot():
#     # Initialize the application
#     application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    
#     # 1. SET BOT COMMANDS (The part you requested)
#     # This makes the "Menu" button appear in the bottom left of the chat
#     from telegram import BotCommand
#     await application.bot.set_my_commands([
#         BotCommand("start", "🚀 Start the Bot"),
#         BotCommand("menu", "📋 Dashboard")
#     ])

#     # 2. ADD HANDLERS
#     application.add_handler(CommandHandler("start", start))
#     application.add_handler(CommandHandler("menu", start)) # Link /menu to the dashboard
#     application.add_handler(CallbackQueryHandler(handle_callback))
#     application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    
#     # 3. START BOT
#     await application.initialize()
#     await application.start()
#     await application.updater.start_polling()
    
#     print("🚀 Bot is running with /menu command active.")

# @app.get("/health")
# def health(): 
#     return {"status": "ok", "commands_active": True}