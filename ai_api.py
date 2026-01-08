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
    """Background task to ping the server health endpoint every 10 mins"""
    if not APP_URL:
        print("⚠️ APP_URL not set. Keep-alive heartbeat is disabled.")
        return
    
    async with httpx.AsyncClient() as client:
        while True:
            await asyncio.sleep(600) # 10 minutes
            try:
                response = await client.get(f"{APP_URL}/health")
                print(f"📡 Heartbeat: {APP_URL} - Status: {response.status_code}")
            except Exception as e:
                print(f"❌ Heartbeat Failed: {e}")

# ---------------- AGENTS & LOGIC ----------------

def run_single_formatter(product_input: str):
    """Agent for Standard 1pc Pipe Format"""
    agent = Agent(
        role="E-commerce Catalog Specialist",
        goal="Extract product data into a professional format while filtering out junk.",
        backstory="You are an expert at creating clean website listings. You strictly separate data fields.",
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

def run_combo_formatter(product_input: str):
    """Agent for Combo/Bulk Pipe Format"""
    agent = Agent(
        role="Combo Deal Specialist",
        goal="Extract only the combo/pack version of a product.",
        backstory="Expert at formatting bulk and combo deals for e-commerce listings.",
        llm=llm
    )
    task = Task(
        description=(
            "You are a professional product formatter. Convert the input into this EXACT pipe-separated format:\n"
            "Name | Price | Description | URLs | Sizes\n\n"
            "STRICT RULES:\n"
            "1. NAME: Format as '<Quantity> pcs COMBO <Brand> <Product Type> <Fabric>'.\n"
            "2. PRICE: Find the 'combo' or 'pcs pack' price, add 15%, output whole number only.\n"
            "3. DESCRIPTION: Premium rewrite. ⚠️ YOU MUST INCLUDE a line: 'Single piece also available at ₹[Calculated 1pc Price + 15%]'.\n"
            "   No other prices or sizes. Use actual physical line breaks.\n"
            "4. URLs: List raw links separated by commas.\n"
            "5. SIZES: List individually (e.g., M, L, XL, XXL).\n"
            "6. OUTPUT: One single pipe-separated line only.\n\n"
            f"INPUT DATA:\n{product_input}"
        ),
        agent=agent,
        expected_output="A single pipe-separated line for COMBO version."
    )
    result = str(Crew(agents=[agent], tasks=[task]).kickoff()).strip()
    return result.replace("\\n", "\n").replace('"', '').replace("Output:", "").replace("Result:", "").strip()

def run_instagram_crew(product_input: str):
    """Agent for Creative Instagram Captions"""
    agent = Agent(
        role="Social Media Copywriter",
        goal="Create high-conversion Instagram captions for Premalatha Collections.",
        backstory="Social media manager who turns product info into viral sales posts.",
        llm=llm
    )
    task = Task(
        description=(
            "Transform the input into a creative Instagram caption.\n"
            "Header: ✨ PREMALATHA COLLECTIONS: NEW ARRIVALS ✨\n"
            "Content: Use emojis, high energy, and physical line breaks. Mention all available prices.\n"
            "Footer: DM to order! 📥 #PremalathaCollections #PremiumFashion\n\n"
            f"INPUT DATA:\n{product_input}"
        ),
        agent=agent,
        expected_output="The final Instagram caption only."
    )
    result = str(Crew(agents=[agent], tasks=[task]).kickoff()).strip()
    return result.replace("\\n", "\n").replace('"', '').strip()

# ---------------- LIFESPAN MANAGER ----------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    
    # Configure Bot Commands Menu
    await application.bot.set_my_commands([
        BotCommand("start", "🚀 Start Bot / Dashboard"),
        BotCommand("menu", "📋 Main Menu")
    ])
    
    # Register Handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("menu", start))
    application.add_handler(CallbackQueryHandler(handle_callback))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    
    await application.initialize()
    await application.start()
    await application.updater.start_polling()
    
    # Start Keep-Alive loop
    keep_alive_loop = asyncio.create_task(keep_alive_task())
    
    print("🚀 Premalatha Bot Online")
    yield
    # --- Shutdown ---
    keep_alive_loop.cancel()
    await application.updater.stop()
    await application.stop()
    await application.shutdown()

app = FastAPI(title="Premalatha Collections API", lifespan=lifespan)

# ---------------- UI HANDLERS ----------------

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
    await update.message.reply_text("🌟 **Premalatha Collections Bot**\nSelect a tool below:", reply_markup=get_main_menu())

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    await query.answer()

    if query.data == "mode_single":
        user_sessions[user_id] = {"step": "input", "mode": "single"}
        await query.edit_message_text("🛠 **Single Formatter**\nPaste raw product details (1 pc focus):")
    elif query.data == "mode_combo":
        user_sessions[user_id] = {"step": "input", "mode": "combo"}
        await query.edit_message_text("📦 **Combo Formatter**\nPaste details (Combo focus + Single rate in desc):")
    elif query.data == "mode_instagram":
        user_sessions[user_id] = {"step": "input", "mode": "instagram"}
        await query.edit_message_text("📸 **Instagram Post Creator**\nPaste your product info here:")
    elif query.data == "reset":
        user_sessions[user_id] = {"step": None, "mode": None}
        await query.edit_message_text("Ready. Select a tool:", reply_markup=get_main_menu())

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    session = user_sessions.get(user_id)
    
    if not session or session.get("step") is None:
        await update.message.reply_text("Please select a mode from the menu:", reply_markup=get_main_menu())
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    
    try:
        user_input = update.message.text
        if session["mode"] == "single":
            res = run_single_formatter(user_input)
        elif session["mode"] == "combo":
            res = run_combo_formatter(user_input)
        else:
            res = run_instagram_crew(user_input)
        
        await update.message.reply_text(res)
        
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")
    
    # Reset session step and show menu
    session["step"] = None
    await update.message.reply_text("--- ✅ Done ---", reply_markup=get_main_menu())

@app.get("/health")
def health():
    return {"status": "ok", "message": "Server is awake"}