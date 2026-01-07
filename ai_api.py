import os
import json
from typing import Optional

from fastapi import FastAPI, Request, HTTPException, Body
from crewai import LLM, Agent, Task, Crew
from dotenv import load_dotenv

# --------------------------------------------------
# Environment setup
# --------------------------------------------------
load_dotenv()
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

app = FastAPI(
    title="Product Formatter API",
    version="1.0.0"
)

# --------------------------------------------------
# Initialize LLM (ONCE)
# --------------------------------------------------
llm = LLM(
    model="openai/openai/gpt-oss-120b",
    api_key=os.getenv("Custom_OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_COMPATIBLE_ENDPOINT"),
)

# --------------------------------------------------
# Multiline-safe + Swagger-visible endpoint
# --------------------------------------------------
@app.post("/format-product")
async def format_product(
    request: Request,
    payload: Optional[dict] = Body(
        default=None,
        example={
            "text": "Brand -MIX\nPURE COTTON\nPRICE - 399\nSIZE M L XL"
        }
    )
):
    try:
        # 1️⃣ Prefer Swagger / proper JSON
        if payload and isinstance(payload, dict):
            product_input = payload.get("text", "")
        else:
            # 2️⃣ Fallback: raw body (broken JSON / copy-paste)
            raw_body = await request.body()
            product_input = raw_body.decode("utf-8", errors="ignore")

        # Normalize input
        product_input = product_input.replace("\r\n", "\n").strip()

        if not product_input:
            raise HTTPException(status_code=400, detail="No input provided")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")

    # --------------------------------------------------
    # CrewAI logic
    # --------------------------------------------------
    agent = Agent(
        role="Product Formatter",
        goal="Convert product listings into strict pipe-separated format",
        backstory="Expert at clean product formatting without sizes or price mistakes.",
        llm=llm
    )

    task = Task(
        description=(
            "CRITICAL: Follow ALL rules strictly.\n\n"

            "OUTPUT FORMAT (ONE LINE ONLY):\n"
            "Name | Price | Description | URLs | Sizes\n\n"

            "TITLE RULES (STRICT):\n"
            "- NEVER include sizes (M, L, XL, XXL, 36, 38, 40, etc.) in title\n"
            "- Title MUST be: '1 pc MIX <Product Type> <Fabric>'\n"
            "- Brand is ALWAYS: MIX\n"
            "- Detect product type: shirt / tshirt / top / kurta / pants\n"
            "- Fabric: Cotton or Pure Cotton\n\n"

            "PRICE RULE:\n"
            "- Increase mentioned price by 20%\n"
            "- Return final numeric price only\n\n"

            "DESCRIPTION RULES:\n"
            "- Multi-line description\n"
            "- DO NOT mention price or sizes\n\n"

            "SIZES RULE:\n"
            "- If product type is pants → 36,38,40\n"
            "- Else → M,L,XL,XXL\n\n"

            "OUTPUT RULES:\n"
            "- One single line only\n"
            "- Pipe-separated fields\n"
            "- No explanations\n\n"

            "PRODUCT INPUT:\n"
            f"{product_input}"
        ),
        agent=agent,
        expected_output="Single pipe-separated line"
    )

    crew = Crew(
        agents=[agent],
        tasks=[task]
    )

    result = crew.kickoff()

    return {
        "formatted_text": str(result).strip()
    }

# --------------------------------------------------
# Health Check
# --------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}
