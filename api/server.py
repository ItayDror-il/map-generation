"""
Sequence Map Generator API

FastAPI server that generates financial automation maps using Claude.
Simplified version - no ML models required.
"""

import os
import json
import httpx
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import anthropic


# ============================================================================
# SEQUENCE PLAYGROUND API
# ============================================================================

SEQUENCE_GRAPHQL_URL = "https://app.getsequence.io/api/admin-graphql"
SEQUENCE_ADMIN_KEY = os.environ.get("SEQUENCE_ADMIN_KEY", "Admin better_luck_next_time")


async def create_playground_map(map_data: dict, name: str = None) -> str:
    """Create a playground map via Sequence Admin GraphQL API."""
    if name is None:
        name = f"Generated Map {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    query = """
    mutation CreateMap($name: String!, $mapJson: String!, $fingerprint: String!) {
        adminCreatePlaygroundMapFromJson(
            name: $name
            mapJson: $mapJson
            ownerFingerprint: $fingerprint
        )
    }
    """

    variables = {
        "name": name,
        "mapJson": json.dumps(map_data),
        "fingerprint": "map-generator-api"
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                SEQUENCE_GRAPHQL_URL,
                json={"query": query, "variables": variables},
                headers={
                    "Content-Type": "application/json",
                    "x-admin-key": SEQUENCE_ADMIN_KEY
                },
                timeout=30.0
            )

            response.raise_for_status()
            data = response.json()

            if "errors" in data:
                error_msg = data["errors"][0].get("message", "GraphQL error")
                raise HTTPException(status_code=500, detail=f"Sequence API error: {error_msg}")

            playground_id = data.get("data", {}).get("adminCreatePlaygroundMapFromJson")
            if not playground_id:
                raise HTTPException(status_code=500, detail="No playground ID returned")

            return playground_id

        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Sequence API unreachable: {str(e)}")


# ============================================================================
# LLM GENERATION (Simplified - no ML models)
# ============================================================================

SYSTEM_PROMPT = """You are generating JSON for Sequence, a financial automation platform. Users describe their money management goals, and you create a "map" that automates fund movements between accounts.

Return ONLY valid JSON matching this exact structure:

{
  "nodes": [
    {
      "id": "UUID",
      "type": "POD|PORT|DEPOSITORY_ACCOUNT|LIABILITY_ACCOUNT",
      "subtype": null|"CHECKING"|"SAVINGS"|"CREDIT_CARD"|"LOAN"|"LINE_OF_CREDIT",
      "name": "string",
      "balance": 0,
      "icon": "emoji",
      "position": {"x": number, "y": number}
    }
  ],
  "rules": [
    {
      "id": "UUID",
      "sourceId": "node-id",
      "trigger": {
        "type": "INCOMING_FUNDS|SCHEDULED|BALANCE_THRESHOLD",
        "sourceId": "node-id",
        "cron": null|"0 0 1 * *"
      },
      "steps": [{
        "actions": [{
          "type": "PERCENTAGE|FIXED|TOP_UP|AVALANCHE|SNOWBALL",
          "sourceId": "node-id",
          "destinationId": "node-id",
          "amountInCents": 0,
          "amountInPercentage": 0,
          "groupIndex": 0,
          "limit": null,
          "upToEnabled": null
        }]
      }]
    }
  ],
  "viewport": {"x": 300, "y": 100, "zoom": 0.9}
}

Node types:
- PORT: Income entry point (salary, deposits)
- POD: Virtual envelope for budgeting
- DEPOSITORY_ACCOUNT: Bank account (CHECKING/SAVINGS)
- LIABILITY_ACCOUNT: Debt (CREDIT_CARD/LOAN/LINE_OF_CREDIT)

Action types:
- PERCENTAGE: amountInPercentage 0-100 (use 100 with higher groupIndex for "remainder")
- FIXED: amountInCents (dollars × 100)
- AVALANCHE/SNOWBALL: Debt payoff strategies (only for LIABILITY_ACCOUNT)

CRITICAL - "Remainder" implementation:
- Use PERCENTAGE with amountInPercentage: 100 and HIGHER groupIndex
- Actions execute in groupIndex order: lower first
- Each action operates on what's LEFT after previous actions

Financial Guidelines:
- Emergency fund: 3-6 months expenses
- Savings rate: 15-20% of income
- Debt priority: High interest first (avalanche) or smallest balance (snowball)
- Business: Reserve 25-35% for taxes

CONSTRAINTS:
1. Return ONLY JSON - no explanations, no markdown
2. All IDs must be valid UUIDs (8-4-4-4-12 format)
3. All sourceId/destinationId must reference existing node IDs
4. Position nodes left-to-right: income (x~100) → processing (x~400) → destinations (x~700)
5. Icons: 📥 PORT, 💰 POD, 🏦 DEPOSITORY, 💳 LIABILITY"""


def generate_map_with_llm(prompt: str, profile: dict) -> dict:
    """Generate a map using Claude API."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")

    client = anthropic.Anthropic(api_key=api_key)

    # Build user message with profile context
    user_type = profile.get('USER_TYPE', 'INDIVIDUAL')
    income = profile.get('ANNUALINCOME', 'Unknown')
    age = profile.get('AGE_GROUP', 'Unknown')
    occupation = profile.get('OCCUPATION', 'Unknown')
    goals = profile.get('PRODUCTGOAL', [])

    user_message = f"""User Request: "{prompt}"

Profile:
- Type: {user_type}
- Income: {income}
- Age: {age}
- Occupation: {occupation}
- Goals: {', '.join(goals) if goals else 'General budgeting'}

Generate a Sequence map JSON for this user. Return ONLY valid JSON."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}]
        )

        response_text = response.content[0].text.strip()

        # Handle markdown code blocks
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()

        return json.loads(response_text)

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Invalid JSON from LLM: {e}")
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Claude API error: {e}")


def generate_explanation(map_data: dict, profile: dict) -> str:
    """Generate human-readable explanation of the map."""
    nodes = map_data.get("nodes", [])
    rules = map_data.get("rules", [])
    node_map = {n["id"]: n for n in nodes}

    lines = []
    user_type = profile.get("USER_TYPE", "INDIVIDUAL")

    if user_type == "BUSINESS":
        lines.append("Here's your business automation map:\n")
    else:
        lines.append("Here's your personalized financial automation map:\n")

    # Summarize nodes
    income_nodes = [n for n in nodes if n["type"] == "PORT"]
    pods = [n for n in nodes if n["type"] == "POD"]
    accounts = [n for n in nodes if n["type"] == "DEPOSITORY_ACCOUNT"]
    debts = [n for n in nodes if n["type"] == "LIABILITY_ACCOUNT"]

    if income_nodes:
        lines.append(f"**Income Sources:** {', '.join(n['name'] for n in income_nodes)}\n")

    if pods:
        lines.append(f"**Budget Categories:** {', '.join(n['name'] for n in pods)}\n")

    if accounts:
        lines.append(f"**Accounts:** {', '.join(n['name'] for n in accounts)}\n")

    if debts:
        lines.append(f"**Debts:** {', '.join(n['name'] for n in debts)}\n")

    lines.append(f"\n**{len(rules)} automation rules** will move your money automatically.")

    return "\n".join(lines)


# ============================================================================
# API MODELS
# ============================================================================

class ProfileInput(BaseModel):
    USER_TYPE: str = Field(default="INDIVIDUAL")
    ANNUALINCOME: str = Field(default="")
    AGE_GROUP: str = Field(default="")
    OCCUPATION: str = Field(default="")
    PRODUCTGOAL: list[str] = Field(default_factory=list)


class GenerateRequest(BaseModel):
    profile: ProfileInput
    prompt: str = Field(..., min_length=20, max_length=1000)


class GenerateResponse(BaseModel):
    id: str
    explanation: str
    map: dict


# ============================================================================
# APP SETUP
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Map Generator API starting...")
    yield
    print("Map Generator API shutting down...")


app = FastAPI(
    title="Sequence Map Generator API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS - allow all origins (public marketing tool)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/api/generate-map", response_model=GenerateResponse)
async def generate_map(request: GenerateRequest):
    """Generate a financial automation map."""
    try:
        profile = {
            "USER_TYPE": request.profile.USER_TYPE,
            "ANNUALINCOME": request.profile.ANNUALINCOME,
            "AGE_GROUP": request.profile.AGE_GROUP,
            "OCCUPATION": request.profile.OCCUPATION,
            "PRODUCTGOAL": request.profile.PRODUCTGOAL,
        }

        # Generate map using LLM
        map_data = generate_map_with_llm(request.prompt, profile)

        # Generate explanation
        explanation = generate_explanation(map_data, profile)

        # Create playground map via Sequence API
        playground_id = await create_playground_map(
            map_data=map_data,
            name=f"Map Generator: {request.prompt[:50]}..."
        )

        return GenerateResponse(
            id=playground_id,
            explanation=explanation,
            map=map_data
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting server on port {port}...")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
