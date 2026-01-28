"""
Sequence Map Generator API

FastAPI server that generates financial automation maps using Claude + ML.
Uses ML models to find similar users and enrich the prompt with patterns.
"""

import os
import sys
import json
import httpx
from pathlib import Path
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent dir to path so we can import llm_generator
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import ML-enabled generator, fall back to simple mode if not available
try:
    from llm_generator import LLMGenerator, validate_sequence_json
    ML_AVAILABLE = True
    print("ML models available - using enriched generation")
except Exception as e:
    ML_AVAILABLE = False
    print(f"ML models not available ({e}) - using simple generation")
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
# SIMPLE GENERATION (Fallback when ML not available)
# ============================================================================

SIMPLE_SYSTEM_PROMPT = """You are generating JSON for Sequence, a financial automation platform.

Return ONLY valid JSON matching this structure:
{
  "nodes": [{"id": "UUID", "type": "POD|PORT|DEPOSITORY_ACCOUNT|LIABILITY_ACCOUNT", "subtype": null|"CHECKING"|"SAVINGS"|"CREDIT_CARD"|"LOAN", "name": "string", "balance": 0, "icon": "emoji", "position": {"x": number, "y": number}}],
  "rules": [{"id": "UUID", "sourceId": "node-id", "trigger": {"type": "INCOMING_FUNDS|SCHEDULED|BALANCE_THRESHOLD", "sourceId": "node-id", "cron": null}, "steps": [{"actions": [{"type": "PERCENTAGE|FIXED|AVALANCHE|SNOWBALL", "sourceId": "node-id", "destinationId": "node-id", "amountInCents": 0, "amountInPercentage": 0, "groupIndex": 0, "limit": null, "upToEnabled": null}]}]}],
  "viewport": {"x": 300, "y": 100, "zoom": 0.9}
}

Node types: PORT (income), POD (envelope), DEPOSITORY_ACCOUNT (bank), LIABILITY_ACCOUNT (debt)
Action types: PERCENTAGE (0-100), FIXED (cents), AVALANCHE/SNOWBALL (debt strategies)
Use PERCENTAGE 100 with higher groupIndex for "remainder".
Position: income (x~100) → processing (x~400) → destinations (x~700)
Return ONLY JSON - no explanations."""


def generate_simple(prompt: str, profile: dict) -> dict:
    """Simple generation without ML context."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")

    client = anthropic.Anthropic(api_key=api_key)

    user_message = f"""User: "{prompt}"
Profile: {profile.get('USER_TYPE', 'INDIVIDUAL')}, Income: {profile.get('ANNUALINCOME', 'Unknown')}, Age: {profile.get('AGE_GROUP', 'Unknown')}
Goals: {', '.join(profile.get('PRODUCTGOAL', [])) or 'General budgeting'}

Generate a Sequence map JSON."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=SIMPLE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}]
    )

    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    return json.loads(text)


# ============================================================================
# EXPLANATION GENERATOR
# ============================================================================

def generate_explanation(map_data: dict, profile: dict, prompt: str) -> str:
    """Generate human-readable explanation of the map."""
    nodes = map_data.get("nodes", [])
    rules = map_data.get("rules", [])
    node_map = {n["id"]: n for n in nodes}

    lines = []
    user_type = profile.get("USER_TYPE", "INDIVIDUAL")
    age = profile.get("AGE_GROUP", "")
    income = profile.get("ANNUALINCOME", "")

    # Intro
    if user_type == "BUSINESS":
        lines.append("Based on your business profile, here's your personalized automation map:\n")
    else:
        intro = "Based on your profile"
        if age:
            intro += f" as a {age} year old"
        if income:
            income_readable = income.replace("BETWEEN_", "").replace("_AND_", "-").replace("_", " ").lower()
            if "over" in income_readable.lower():
                intro += f" earning over $250k"
            elif "under" in income_readable.lower():
                intro += f" earning under $25k"
            else:
                intro += f" earning ${income_readable}"
        lines.append(intro + ", here's your personalized automation map:\n")

    # Categorize nodes
    income_nodes = [n for n in nodes if n["type"] == "PORT"]
    pods = [n for n in nodes if n["type"] == "POD"]
    accounts = [n for n in nodes if n["type"] == "DEPOSITORY_ACCOUNT"]
    debts = [n for n in nodes if n["type"] == "LIABILITY_ACCOUNT"]

    # Income sources
    if income_nodes:
        for node in income_nodes:
            lines.append(f"**{node['icon']} {node['name']}**")
            lines.append(f"Your money flows in here first. Every time funds arrive, they automatically route to the right places.\n")

    # Processing pods with rule details
    if pods:
        for pod in pods:
            lines.append(f"**{pod['icon']} {pod['name']}**")
            pod_rules = [r for r in rules if r.get("sourceId") == pod["id"]]
            if pod_rules:
                actions = []
                for rule in pod_rules:
                    for step in rule.get("steps", []):
                        for action in step.get("actions", []):
                            dest = node_map.get(action.get("destinationId"), {})
                            if action["type"] == "PERCENTAGE" and action.get("amountInPercentage"):
                                pct = action["amountInPercentage"]
                                if pct == 100 and action.get("groupIndex", 0) > 0:
                                    actions.append(f"Remaining → {dest.get('name', 'destination')}")
                                else:
                                    actions.append(f"{pct}% → {dest.get('name', 'destination')}")
                            elif action["type"] == "FIXED" and action.get("amountInCents"):
                                amt = action["amountInCents"] / 100
                                actions.append(f"${amt:,.0f} → {dest.get('name', 'destination')}")
                            elif action["type"] in ["AVALANCHE", "SNOWBALL"]:
                                actions.append(f"{action['type'].title()} → {dest.get('name', 'destination')}")
                if actions:
                    lines.append("Splits your income:")
                    for a in actions[:5]:
                        lines.append(f"  • {a}")
            lines.append("")

    # Accounts
    if accounts:
        lines.append("**Destination Accounts:**")
        for acc in accounts:
            subtype = acc.get("subtype", "").replace("_", " ").title()
            lines.append(f"  • {acc['icon']} {acc['name']} ({subtype})")
        lines.append("")

    # Debts with strategy detection
    if debts:
        has_avalanche = any(
            action.get("type") == "AVALANCHE"
            for rule in rules for step in rule.get("steps", []) for action in step.get("actions", [])
        )
        has_snowball = any(
            action.get("type") == "SNOWBALL"
            for rule in rules for step in rule.get("steps", []) for action in step.get("actions", [])
        )

        if has_avalanche:
            lines.append("**💳 Debt Payoff (Avalanche Method)**")
            lines.append("Targeting highest-interest debt first:")
        elif has_snowball:
            lines.append("**💳 Debt Payoff (Snowball Method)**")
            lines.append("Targeting smallest balance first:")
        else:
            lines.append("**💳 Debt Payments:**")

        for i, debt in enumerate(debts, 1):
            balance = debt.get("balance", 0) / 100 if debt.get("balance") else 0
            if balance > 0:
                lines.append(f"  {i}. {debt['name']} (${balance:,.0f} balance)")
            else:
                lines.append(f"  {i}. {debt['name']}")

        if has_avalanche or has_snowball:
            lines.append("\nOnce the first debt is paid off, those payments automatically roll into the next.")

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

# Global generator instance
_generator: Optional[LLMGenerator] = None if not ML_AVAILABLE else None


def get_generator() -> Optional[LLMGenerator]:
    """Get or create the LLM generator with ML context."""
    global _generator
    if not ML_AVAILABLE:
        return None
    if _generator is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")

        # Use paths relative to project root
        project_root = Path(__file__).parent.parent
        _generator = LLMGenerator(
            api_key=api_key,
            models_dir=str(project_root / "models"),
            data_dir=str(project_root)
        )
    return _generator


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Map Generator API starting...")
    if ML_AVAILABLE:
        try:
            get_generator()
            print("ML generator initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize ML generator: {e}")
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
    return {
        "status": "ok",
        "ml_available": ML_AVAILABLE
    }


@app.post("/api/generate-map", response_model=GenerateResponse)
async def generate_map(request: GenerateRequest):
    """Generate a financial automation map using ML + LLM."""
    try:
        profile = {
            "USER_TYPE": request.profile.USER_TYPE,
            "ANNUALINCOME": request.profile.ANNUALINCOME,
            "AGE_GROUP": request.profile.AGE_GROUP,
            "OCCUPATION": request.profile.OCCUPATION,
            "PRODUCTGOAL": request.profile.PRODUCTGOAL,
        }

        # Try ML-enriched generation first
        generator = get_generator()
        if generator:
            print("Using ML-enriched generation")
            result = generator.generate(
                prompt=request.prompt,
                profile=profile
            )

            if result["error"]:
                raise HTTPException(status_code=500, detail=result["error"])

            map_data = result["json_map"]
        else:
            print("Using simple generation (no ML)")
            map_data = generate_simple(request.prompt, profile)

        # Generate explanation
        explanation = generate_explanation(map_data, profile, request.prompt)

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
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Invalid JSON from LLM: {e}")
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
