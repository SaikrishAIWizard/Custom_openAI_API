import os
import json
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from crewai import LLM, Agent, Task, Crew
from dotenv import load_dotenv

# ---------------- ENV SETUP ----------------
load_dotenv()
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

# ---------------- APP ----------------
app = FastAPI(
    title="Product Formatter API",
    version="1.0.0",
    description="Handles multiline product text safely without 422 errors"
)

# ---------------- LLM INIT (ONCE) ----------------
llm = LLM(
    model="openai/openai/gpt-oss-120b",
    api_key=os.getenv("Custom_OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_COMPATIBLE_ENDPOINT"),
)

# ---------------- MAIN ENDPOINT ----------------
@app.post("/format-product")
async def format_product(request: Request):
    """
    Accepts:
    - JSON: { "text": "multi line content" }
    - OR raw text body (text/plain)
    NEVER throws 422
    """

    try:
        raw_body = await request.body()

        if not raw_body:
            return JSONResponse(
                status_code=400,
                content={"error": "Empty request body"}
            )

        body_text = raw_body.decode("utf-8", errors="ignore").strip()

        # Try JSON parsing first
        try:
            parsed = json.loads(body_text)
            product_input = parsed.get("text", body_text)
        except json.JSONDecodeError:
            # If not JSON, treat whole body as text
            product_input = body_text

        product_input = product_input.replace("\r\n", "\n").strip()

        if not product_input:
            return JSONResponse(
                status_code=400,
                content={"error": "No input text provided"}
            )

    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid input: {str(e)}"}
        )

    # -------- CrewAI Logic --------
    agent = Agent(
    role="Product Formatter",
    goal="Format products accurately by product type",
    backstory="E-commerce expert - shirts get letters, pants get numbers",
    llm=llm)

    task = Task(
        description=(
            f"Output ONLY: Name | Price | Description | URLs | Sizes\n\n"
            f"TITLE: '1 pc MIX [Type] Cotton'\n"
            f"Type = Pants ONLY if input contains: pants/trackpant/trouser/pyjama\n"
            f"Type = Shirt for ALL OTHER products (even with numbers)\n\n"
            f"Price: ×1.2 rounded\n"
            f"Description: Multi-line, NO price/sizes\n\n"
            f"SIZES:\n"
            f"IF 'pants/trackpant/trouser/pyjama' in text → 28,30,32,34,36,38,40\n"
            f"ELSE → M,L,XL,XXL (numbers in shirts = ignore)\n\n"
            f"{product_input}"
        ),
        agent=agent,
        expected_output="Pipe-separated line only"
    )



    crew = Crew(agents=[agent], tasks=[task])
    result = crew.kickoff()

    return {
        "formatted_text": str(result).strip()
    }

# ---------------- HEALTH CHECK ----------------
@app.get("/health")
def health():
    return {"status": "ok"}
