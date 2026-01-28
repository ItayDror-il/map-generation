"""
Sequence Map Generator API

FastAPI server that generates financial automation maps using Claude + ML.
Uses ML models to find similar users and enrich the prompt with patterns.
"""

import os
import sys
import json
import httpx
import random
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
# JSON VALIDATION (Based on PLAYGROUND_JSON_SPECIFICATION.md)
# ============================================================================

VALID_NODE_TYPES = {"PORT", "POD", "DEPOSITORY_ACCOUNT", "LIABILITY_ACCOUNT", "INVESTMENT_ACCOUNT", "DESTINATION_ACCOUNT"}
VALID_NODE_SUBTYPES = {
    "DEPOSITORY_ACCOUNT": {"SAVINGS", "CHECKING"},
    "LIABILITY_ACCOUNT": {"CREDIT_CARD", "MORTGAGE", "PERSONAL_LOAN", "HOME_LOAN", "STUDENT_LOAN", "AUTO_LOAN", "BUSINESS_LOAN"},
    "INVESTMENT_ACCOUNT": {"PENSION", "IRA", "BROKERAGE", "EDUCATION_SAVINGS"},
}
VALID_TRIGGER_TYPES = {"INCOMING_FUNDS", "SCHEDULED"}
VALID_ACTION_TYPES = {
    "FIXED", "PERCENTAGE", "TOP_UP", "ROUND_DOWN",
    "NEXT_PAYMENT_MINIMUM", "TOTAL_AMOUNT_DUE", "PERCENTAGE_LIABILITY_BALANCE",
    "SNOWBALL", "AVALANCHE"
}
LIABILITY_ONLY_ACTIONS = {"NEXT_PAYMENT_MINIMUM", "TOTAL_AMOUNT_DUE", "PERCENTAGE_LIABILITY_BALANCE", "SNOWBALL", "AVALANCHE"}
TOP_UP_VALID_DESTINATIONS = {"POD", "DEPOSITORY_ACCOUNT"}


def validate_playground_json(data: dict) -> tuple[bool, list[str], list[str]]:
    """
    Validate JSON against Playground specification.
    Returns (is_valid, errors, warnings)
    """
    errors = []
    warnings = []

    # Check top-level structure
    if "nodes" not in data:
        errors.append("Missing required 'nodes' array")
    if "rules" not in data:
        errors.append("Missing required 'rules' array")
    if "viewport" not in data:
        warnings.append("Missing 'viewport' - will use default")

    if errors:
        return False, errors, warnings

    # Collect node IDs for reference validation
    node_ids = set()
    node_types = {}  # id -> type

    # Validate nodes
    for i, node in enumerate(data.get("nodes", [])):
        prefix = f"nodes[{i}]"

        # Required fields
        for field in ["id", "name", "icon", "type", "position"]:
            if field not in node:
                errors.append(f"{prefix}: Missing required field '{field}'")

        if "id" in node:
            if node["id"] in node_ids:
                errors.append(f"{prefix}: Duplicate node ID '{node['id']}'")
            node_ids.add(node["id"])
            node_types[node["id"]] = node.get("type")

        # Type validation
        node_type = node.get("type")
        if node_type and node_type not in VALID_NODE_TYPES:
            errors.append(f"{prefix}: Invalid type '{node_type}'. Must be one of {VALID_NODE_TYPES}")

        # Subtype validation
        subtype = node.get("subtype")
        if subtype and node_type in VALID_NODE_SUBTYPES:
            valid_subtypes = VALID_NODE_SUBTYPES[node_type]
            if subtype not in valid_subtypes:
                errors.append(f"{prefix}: Invalid subtype '{subtype}' for {node_type}. Must be one of {valid_subtypes}")

        # Position validation
        pos = node.get("position", {})
        if not isinstance(pos.get("x"), (int, float)) or not isinstance(pos.get("y"), (int, float)):
            errors.append(f"{prefix}: position must have numeric 'x' and 'y'")

        # Balance should be integer (cents)
        balance = node.get("balance")
        if balance is not None and not isinstance(balance, int):
            warnings.append(f"{prefix}: balance should be integer (cents), got {type(balance).__name__}")

    # Validate rules
    for i, rule in enumerate(data.get("rules", [])):
        prefix = f"rules[{i}]"

        # Required fields
        for field in ["id", "sourceId", "trigger", "steps"]:
            if field not in rule:
                errors.append(f"{prefix}: Missing required field '{field}'")

        # Source reference validation
        source_id = rule.get("sourceId")
        if source_id and source_id not in node_ids:
            errors.append(f"{prefix}: sourceId '{source_id}' references non-existent node")

        # Trigger validation
        trigger = rule.get("trigger", {})
        trigger_type = trigger.get("type")
        if trigger_type not in VALID_TRIGGER_TYPES:
            errors.append(f"{prefix}.trigger: Invalid type '{trigger_type}'. Must be one of {VALID_TRIGGER_TYPES}")

        if trigger_type == "SCHEDULED" and not trigger.get("cron"):
            errors.append(f"{prefix}.trigger: SCHEDULED trigger requires 'cron' expression")

        trigger_source = trigger.get("sourceId")
        if trigger_source and trigger_source not in node_ids:
            errors.append(f"{prefix}.trigger: sourceId '{trigger_source}' references non-existent node")

        # Steps validation
        steps = rule.get("steps", [])
        if not steps:
            errors.append(f"{prefix}: Must have at least one step")

        for j, step in enumerate(steps):
            step_prefix = f"{prefix}.steps[{j}]"

            actions = step.get("actions", [])
            if not actions:
                errors.append(f"{step_prefix}: Must have at least one action")

            # Track group percentages
            group_percentages = {}

            for k, action in enumerate(actions):
                action_prefix = f"{step_prefix}.actions[{k}]"

                # Required fields
                for field in ["type", "sourceId", "destinationId", "groupIndex"]:
                    if field not in action:
                        errors.append(f"{action_prefix}: Missing required field '{field}'")

                action_type = action.get("type")
                if action_type and action_type not in VALID_ACTION_TYPES:
                    errors.append(f"{action_prefix}: Invalid type '{action_type}'. Must be one of {VALID_ACTION_TYPES}")

                # Reference validation
                action_source = action.get("sourceId")
                action_dest = action.get("destinationId")
                if action_source and action_source not in node_ids:
                    errors.append(f"{action_prefix}: sourceId '{action_source}' references non-existent node")
                if action_dest and action_dest not in node_ids:
                    errors.append(f"{action_prefix}: destinationId '{action_dest}' references non-existent node")

                # Destination-specific action validation
                if action_type in LIABILITY_ONLY_ACTIONS:
                    dest_type = node_types.get(action_dest)
                    if dest_type and dest_type != "LIABILITY_ACCOUNT":
                        errors.append(f"{action_prefix}: {action_type} can only target LIABILITY_ACCOUNT, got {dest_type}")

                if action_type == "TOP_UP":
                    dest_type = node_types.get(action_dest)
                    if dest_type and dest_type not in TOP_UP_VALID_DESTINATIONS:
                        errors.append(f"{action_prefix}: TOP_UP can only target POD or DEPOSITORY_ACCOUNT, got {dest_type}")

                # Amount field validation
                amount_cents = action.get("amountInCents")
                amount_pct = action.get("amountInPercentage")
                group_idx = action.get("groupIndex", 0)

                if action_type == "FIXED":
                    if amount_cents is None or amount_cents <= 0:
                        warnings.append(f"{action_prefix}: FIXED action should have positive amountInCents")
                elif action_type == "PERCENTAGE":
                    if amount_pct is None:
                        warnings.append(f"{action_prefix}: PERCENTAGE action should have amountInPercentage")
                    elif not (0 <= amount_pct <= 100):
                        errors.append(f"{action_prefix}: amountInPercentage must be 0-100, got {amount_pct}")

                    # Track percentages per group
                    if group_idx not in group_percentages:
                        group_percentages[group_idx] = 0
                    group_percentages[group_idx] += amount_pct or 0
                elif action_type in {"TOP_UP", "ROUND_DOWN"}:
                    if amount_cents is None:
                        warnings.append(f"{action_prefix}: {action_type} action should have amountInCents")

            # Check group percentages don't exceed 100
            for group_idx, total_pct in group_percentages.items():
                if total_pct > 100:
                    warnings.append(f"{step_prefix}: Group {group_idx} percentages sum to {total_pct}%, which exceeds 100%")

    # Validate viewport
    viewport = data.get("viewport", {})
    if viewport:
        for field in ["x", "y", "zoom"]:
            if field not in viewport:
                warnings.append(f"viewport: Missing field '{field}'")
            elif not isinstance(viewport[field], (int, float)):
                warnings.append(f"viewport.{field}: Should be numeric")

    return len(errors) == 0, errors, warnings


def fix_common_json_issues(data: dict, income_bracket: str = "BETWEEN_50K_AND_100K") -> dict:
    """
    Auto-fix common JSON issues to ensure valid output.
    This is the safety net - fixes LLM mistakes before validation.
    """
    # Income bracket to bi-weekly pay (in cents)
    INCOME_TO_BALANCE = {
        "UNDER_25K": 83300,
        "BETWEEN_25K_AND_50K": 156200,
        "BETWEEN_50K_AND_100K": 312500,
        "BETWEEN_100K_AND_250K": 729100,
        "OVER_250K": 1458300,
    }
    port_balance = INCOME_TO_BALANCE.get(income_bracket, 312500)

    # Ensure top-level structure
    if "viewport" not in data:
        data["viewport"] = {"x": 300, "y": 100, "zoom": 0.9}
    if "nodes" not in data:
        data["nodes"] = []
    if "rules" not in data:
        data["rules"] = []
    if "name" not in data or not data["name"]:
        data["name"] = "My Financial Map"

    # Build node type lookup for action fixes
    node_types = {}

    # Fix node issues
    for node in data.get("nodes", []):
        node_id = node.get("id")
        node_type = node.get("type")

        # Fix invalid node types
        if node_type and node_type not in VALID_NODE_TYPES:
            # Try to map common mistakes
            type_fixes = {
                "INCOME": "PORT",
                "SAVINGS": "DEPOSITORY_ACCOUNT",
                "CHECKING": "DEPOSITORY_ACCOUNT",
                "CREDIT": "LIABILITY_ACCOUNT",
                "DEBT": "LIABILITY_ACCOUNT",
                "INVESTMENT": "INVESTMENT_ACCOUNT",
                "ENVELOPE": "POD",
                "BUCKET": "POD",
            }
            node["type"] = type_fixes.get(node_type.upper(), "POD")
            node_type = node["type"]

        if node_id:
            node_types[node_id] = node_type

        # Fix balance - generate random if zero or missing
        balance = node.get("balance")
        if balance is None or balance == 0:
            if node_type == "PORT":
                # Use income-based balance for PORT nodes
                node["balance"] = port_balance
            elif node_type == "LIABILITY_ACCOUNT":
                # Higher random range for debt: $500-$2000
                node["balance"] = random.randint(50, 200) * 1000
            else:
                # Random $200-$1000 for other nodes (in cents, increments of $10)
                node["balance"] = random.randint(20, 100) * 1000
        elif not isinstance(balance, int):
            try:
                node["balance"] = int(balance)
            except (ValueError, TypeError):
                node["balance"] = random.randint(20, 100) * 1000

        # Ensure position exists
        if "position" not in node or not isinstance(node.get("position"), dict):
            node["position"] = {"x": 300, "y": 300}
        else:
            pos = node["position"]
            if "x" not in pos or not isinstance(pos.get("x"), (int, float)):
                pos["x"] = 300
            if "y" not in pos or not isinstance(pos.get("y"), (int, float)):
                pos["y"] = 300

    # Fix rule issues
    for rule in data.get("rules", []):
        # Fix trigger
        trigger = rule.get("trigger", {})
        if not isinstance(trigger, dict):
            trigger = {}
            rule["trigger"] = trigger

        trigger_type = trigger.get("type")

        # Fix invalid trigger types - CRITICAL FIX
        if trigger_type not in VALID_TRIGGER_TYPES:
            # Map common mistakes to valid types
            trigger_fixes = {
                "BALANCE_THRESHOLD": "INCOMING_FUNDS",
                "BALANCE": "INCOMING_FUNDS",
                "THRESHOLD": "INCOMING_FUNDS",
                "ON_DEPOSIT": "INCOMING_FUNDS",
                "DEPOSIT": "INCOMING_FUNDS",
                "RECURRING": "SCHEDULED",
                "TIMER": "SCHEDULED",
                "CRON": "SCHEDULED",
            }
            fixed_type = trigger_fixes.get(trigger_type, "INCOMING_FUNDS") if trigger_type else "INCOMING_FUNDS"
            trigger["type"] = fixed_type
            trigger_type = fixed_type

        # Ensure trigger has sourceId
        if "sourceId" not in trigger:
            trigger["sourceId"] = rule.get("sourceId")

        # SCHEDULED requires cron
        if trigger_type == "SCHEDULED" and not trigger.get("cron"):
            trigger["cron"] = "0 0 1 * *"  # Default: 1st of month

        # Ensure steps exist
        if "steps" not in rule or not rule["steps"]:
            rule["steps"] = [{"actions": []}]

        for step in rule.get("steps", []):
            if "actions" not in step:
                step["actions"] = []

            for action in step.get("actions", []):
                action_type = action.get("type")

                # Fix invalid action types
                if action_type and action_type not in VALID_ACTION_TYPES:
                    action_fixes = {
                        "REMAINDER": "PERCENTAGE",  # Use 100% with higher groupIndex
                        "TRANSFER": "FIXED",
                        "SPLIT": "PERCENTAGE",
                        "PAY_MINIMUM": "NEXT_PAYMENT_MINIMUM",
                        "PAY_FULL": "TOTAL_AMOUNT_DUE",
                    }
                    action["type"] = action_fixes.get(action_type, "PERCENTAGE")

                # Ensure required fields have defaults
                if "amountInCents" not in action:
                    action["amountInCents"] = 0
                if "amountInPercentage" not in action:
                    action["amountInPercentage"] = 0
                if "groupIndex" not in action:
                    action["groupIndex"] = 0
                if "limit" not in action:
                    action["limit"] = None
                if "upToEnabled" not in action:
                    action["upToEnabled"] = None

                # Fix liability-only actions targeting wrong node types
                dest_id = action.get("destinationId")
                dest_type = node_types.get(dest_id)
                if action.get("type") in LIABILITY_ONLY_ACTIONS and dest_type and dest_type != "LIABILITY_ACCOUNT":
                    # Convert to PERCENTAGE instead
                    action["type"] = "PERCENTAGE"
                    if not action.get("amountInPercentage"):
                        action["amountInPercentage"] = 100

    return data


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
  "name": "Short Name (4 words max)",
  "nodes": [{"id": "UUID", "type": "POD|PORT|DEPOSITORY_ACCOUNT|LIABILITY_ACCOUNT", "subtype": null|"CHECKING"|"SAVINGS"|"CREDIT_CARD"|"LOAN", "name": "string", "balance": 50000, "icon": "emoji", "position": {"x": number, "y": number}}],
  "rules": [{"id": "UUID", "sourceId": "node-id", "trigger": {"type": "INCOMING_FUNDS|SCHEDULED", "sourceId": "node-id", "cron": null}, "steps": [{"actions": [{"type": "PERCENTAGE|FIXED|AVALANCHE|SNOWBALL", "sourceId": "node-id", "destinationId": "node-id", "amountInCents": 0, "amountInPercentage": 0, "groupIndex": 0, "limit": null, "upToEnabled": null}]}]}],
  "viewport": {"x": 300, "y": 100, "zoom": 0.9}
}

IMPORTANT RULES:
- Trigger types: ONLY "INCOMING_FUNDS" or "SCHEDULED" (nothing else!)
- Node types: PORT (income), POD (envelope), DEPOSITORY_ACCOUNT (bank), LIABILITY_ACCOUNT (debt)
- Action types: PERCENTAGE (0-100), FIXED (cents), AVALANCHE/SNOWBALL (debt, LIABILITY_ACCOUNT only)
- Use PERCENTAGE 100 with higher groupIndex for "remainder"
- Position: income (x~100) → processing (x~400) → destinations (x~700)
- All balances must be non-zero (use values 20000-100000 cents)
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

        # Fix common issues and validate JSON before sending to Sequence
        map_data = fix_common_json_issues(map_data, income_bracket=profile.get("ANNUALINCOME", ""))
        is_valid, validation_errors, validation_warnings = validate_playground_json(map_data)

        if validation_warnings:
            print(f"JSON validation warnings: {validation_warnings}")

        if not is_valid:
            print(f"JSON validation errors: {validation_errors}")
            raise HTTPException(
                status_code=500,
                detail=f"Generated JSON failed validation: {'; '.join(validation_errors[:3])}"
            )

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
