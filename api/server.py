"""
Sequence Map Generator API

Minimal FastAPI server wrapping the LLM generator.
Single endpoint, full separation from frontend.
"""

import os
import sys
import json
import httpx
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent dir to path so we can import llm_generator
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_generator import LLMGenerator, validate_sequence_json


# ============================================================================
# SEQUENCE PLAYGROUND API
# ============================================================================

SEQUENCE_GRAPHQL_URL = "https://app.getsequence.io/api/admin-graphql"
SEQUENCE_ADMIN_KEY = os.environ.get("SEQUENCE_ADMIN_KEY", "Admin better_luck_next_time")


async def create_playground_map(map_data: dict, name: str = None) -> str:
    """
    Create a playground map via Sequence Admin GraphQL API.

    Args:
        map_data: The map JSON to save
        name: Optional name for the map (defaults to timestamp)

    Returns:
        Playground map ID (UUID)

    Raises:
        HTTPException if API call fails
    """
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
# EXPLANATION GENERATOR
# ============================================================================

def generate_explanation(map_data: dict, profile: dict, prompt: str) -> str:
    """
    Generate human-readable explanation of the map.

    This is separate from LLM generation - it's a deterministic summary
    of what was created, so frontend changes don't affect it.
    """
    nodes = map_data.get("nodes", [])
    rules = map_data.get("rules", [])

    # Build node lookup
    node_map = {n["id"]: n for n in nodes}

    # Categorize nodes
    income_nodes = [n for n in nodes if n["type"] == "PORT"]
    pods = [n for n in nodes if n["type"] == "POD"]
    accounts = [n for n in nodes if n["type"] == "DEPOSITORY_ACCOUNT"]
    debts = [n for n in nodes if n["type"] == "LIABILITY_ACCOUNT"]

    # Build explanation
    lines = []

    # Intro based on profile
    user_type = profile.get("USER_TYPE", "INDIVIDUAL")
    age = profile.get("AGE_GROUP", "")
    income = profile.get("ANNUALINCOME", "")

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

    # Income sources
    if income_nodes:
        for node in income_nodes:
            lines.append(f"**{node['icon']} {node['name']}**")
            lines.append(f"Your money flows in here first. Every time funds arrive, they automatically route to the right places.\n")

    # Processing pods
    if pods:
        for pod in pods:
            lines.append(f"**{pod['icon']} {pod['name']}**")

            # Find rules that send money FROM this pod
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
                    for a in actions[:5]:  # Limit to 5
                        lines.append(f"  • {a}")
            lines.append("")

    # Accounts
    if accounts:
        lines.append("**Destination Accounts:**")
        for acc in accounts:
            subtype = acc.get("subtype", "").replace("_", " ").title()
            lines.append(f"  • {acc['icon']} {acc['name']} ({subtype})")
        lines.append("")

    # Debts with strategy
    if debts:
        # Check for avalanche/snowball strategy
        has_avalanche = any(
            action.get("type") == "AVALANCHE"
            for rule in rules
            for step in rule.get("steps", [])
            for action in step.get("actions", [])
        )
        has_snowball = any(
            action.get("type") == "SNOWBALL"
            for rule in rules
            for step in rule.get("steps", [])
            for action in step.get("actions", [])
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

# Initialize generator on startup (lazy load)
_generator: Optional[LLMGenerator] = None


def get_generator() -> LLMGenerator:
    global _generator
    if _generator is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="ANTHROPIC_API_KEY not configured"
            )
        _generator = LLMGenerator(api_key=api_key)
    return _generator


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Map Generator API starting...")
    yield
    # Shutdown
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
    """
    Generate a financial automation map.

    Takes user profile and free-text goal description.
    Returns playground map ID, human-readable explanation, and the map JSON.

    The map is automatically saved to Sequence playground via GraphQL API.
    """
    try:
        generator = get_generator()

        # Convert profile to dict for llm_generator
        profile = {
            "USER_TYPE": request.profile.USER_TYPE,
            "ANNUALINCOME": request.profile.ANNUALINCOME,
            "AGE_GROUP": request.profile.AGE_GROUP,
            "OCCUPATION": request.profile.OCCUPATION,
            "PRODUCTGOAL": request.profile.PRODUCTGOAL,
        }

        # Generate map using LLM
        result = generator.generate(
            prompt=request.prompt,
            profile=profile
        )

        if result["error"]:
            raise HTTPException(
                status_code=500,
                detail=result["error"]
            )

        map_data = result["json_map"]

        # Generate explanation
        explanation = generate_explanation(map_data, profile, request.prompt)

        # Create playground map via Sequence API
        # This returns the real playground ID that works with playground.getsequence.io
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
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}"
        )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))

    print(f"Starting server on port {port}...")
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )
