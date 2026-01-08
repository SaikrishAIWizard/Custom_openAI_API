import os
import json
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from crewai import LLM, Agent, Task, Crew
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ChatAction

# ---------------- ENV SETUP ----------------
load_dotenv()
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

# ---------------- APP CONFIG ----------------
app = FastAPI(title="Product Formatter API & Bot")
# Replace with your token in .env or paste directly here for testing
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# ---------------- LLM INIT ----------------
llm = LLM(
    model="openai/openai/gpt-oss-120b",
    api_key=os.getenv("Custom_OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_COMPATIBLE_ENDPOINT"),
)

# ---------------- CORE LOGIC ----------------
def process_product_logic(product_input: str):
    agent = Agent(
        role="Professional Product Formatter",
        goal="Extract data into an exact pipe-separated format with specific naming and pricing rules.",
        backstory="You are a high-speed e-commerce data specialist. You never deviate from formatting rules.",
        llm=llm
    )

    task = Task(
        description=(
            "You are a professional product formatter. Extract data into this EXACT pipe-separated format:\n"
            "Name | Price | Description | URLs | Sizes\n\n"
            "RULES:\n"
            "- NAME: '<Quantity> <Brand> <Product Type> <Fabric>'. NO Price/Sizes.\n"
            "  * Type = Pants ONLY if input contains: pants/trackpant/trouser/pyjama\n"
            "  * Type = Shirt for ALL OTHER products (even with numbers)\n"
            "- PRICE: Base price + 20%, output whole number only.\n"
            "- DESCRIPTION: Engaging, emojis, '\\n' for line breaks. No mentions of links/videos/photos.\n"
            "- URLs: Comma-separated raw links.\n"
            "- SIZES: Separate all components (e.g., M38 becomes 'M, 38'). List individually.\n"
            "  * IF Type is 'Pants' -> 28,30,32,34,36,38,40\n"
            "  * IF Type is 'Shirt' -> M,L,XL,XXL\n"
            "- OUTPUT: One single pipe-separated line only.\n\n"
            f"INPUT DATA:\n{product_input}"
        ),
        agent=agent,
        expected_output="A single pipe-separated line: Name | Price | Description | URLs | Sizes"
    )

    crew = Crew(agents=[agent], tasks=[task])
    result = crew.kickoff()
    return str(result).strip()

# ---------------- TELEGRAM HANDLERS ----------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello! Send me product details and I'll format them according to your specific rules.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    # Show "typing..." in Telegram
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    
    try:
        # Run the CrewAI logic
        formatted_result = process_product_logic(user_text)
        await update.message.reply_text(formatted_result)
    except Exception as e:
        await update.message.reply_text(f"Error: {str(e)}")

# ---------------- API ENDPOINTS ----------------
@app.post("/format-product")
async def format_product(request: Request):
    raw_body = await request.body()
    body_text = raw_body.decode("utf-8", errors="ignore").strip()

    try:
        parsed = json.loads(body_text)
        product_input = parsed.get("text", body_text)
    except json.JSONDecodeError:
        product_input = body_text

    if not product_input:
        return JSONResponse(status_code=400, content={"error": "Empty input"})

    result = process_product_logic(product_input)
    return {"formatted_text": result}

# ---------------- BOT LIFECYCLE ----------------
@app.on_event("startup")
async def start_bot():
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    
    # Initialize and start polling in the background
    await application.initialize()
    await application.start()
    await application.updater.start_polling()
    print("Telegram Bot is active at t.me/PLCProductFormatter_bot")

@app.get("/health")
def health():
    return {"status": "ok"}