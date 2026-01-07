import os
import json
from fastapi import FastAPI, Request, HTTPException
from crewai import LLM, Agent, Task, Crew
from dotenv import load_dotenv

# Load env vars
load_dotenv()
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

app = FastAPI(
    title="Product Formatter API",
    version="1.0.0"
)

# -------- Initialize LLM (once) --------
llm = LLM(
    model="openai/openai/gpt-oss-120b",
    api_key=os.getenv("Custom_OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_COMPATIBLE_ENDPOINT"),
)

# -------- Multiline-Safe Endpoint --------
@app.post("/format-product")
async def format_product(request: Request):
    try:
        raw_body = await request.body()
        body_text = raw_body.decode("utf-8", errors="ignore")

        # Try JSON parse first
        try:
            data = json.loads(body_text)
            product_input = data.get("text", "")
        except json.JSONDecodeError:
            # Fallback: treat entire body as text
            product_input = body_text

        product_input = product_input.replace("\r\n", "\n").strip()

        if not product_input:
            raise HTTPException(status_code=400, detail="No input provided")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")

    agent = Agent(
        role="Product Formatter",
        goal="Convert product listings into exact pipe-separated format",
        backstory="Expert at clean product formatting without price/sizes in wrong places.",
        llm=llm
)

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
            "- If product type is pants → Sizes: 36,38,40\n"
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

# -------- Health Check --------
@app.get("/health")
def health():
    return {"status": "ok"}
