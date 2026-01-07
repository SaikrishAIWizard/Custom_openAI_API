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
        goal="Convert product listings into exact pipe-separated format",
        backstory="Expert at clean product formatting without price/sizes in wrong places.",
        llm=llm)

    task = Task(
        description=(
            "Convert the product details into EXACTLY this format:\n"
            "Name | Price | Description | URLs | Sizes\n\n"

            "TITLE RULES (STRICT):\n"
            "- NEVER include sizes (M, L, XL, XXL, 36, 38, 40, etc.) in the title\n"
            "- Title format MUST be:\n"
            "  '1 pc <Brand> <Product Type> <Fabric>'\n"
            "- Brand is ALWAYS: MIX\n"
            "- Detect product type from input (shirt / tshirt / top / kurta / pants)\n"
            "- Fabric must be Cotton or Pure Cotton\n\n"

            "PRICE RULE:\n"
            "- Increase given price by 20% and round reasonably\n\n"

            "DESCRIPTION RULES:\n"
            "- Multi-line description\n"
            "- Do NOT mention price or sizes\n\n"

            "SIZES RULE (BASED ON PRODUCT TYPE):\n"
            "- If product type is pants → Sizes: 28,30,32,34,36,38,40,42\n"
            "- Otherwise → Sizes: M,L,XL,XXL\n\n"

            "OUTPUT RULES:\n"
            "- Output ONLY one single line\n"
            "- Fields must be pipe-separated (|)\n"
            "- No extra text, no explanations\n\n"

            "Product Input:\n"
            f"{product_input}"
        ),
        agent=agent,
        expected_output="Single pipe-separated line"
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
