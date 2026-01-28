"""
LLM-Based Sequence Map Generator

Uses Anthropic Claude API to generate maps, informed by:
- User's prompt
- Similar user patterns from ML model
- Schema specification
- Pattern examples from real users

The ML work provides CONTEXT, the LLM does the GENERATION.
"""

import json
import os
from typing import Optional
from pathlib import Path
from dataclasses import dataclass

import anthropic

# Import our ML modules
import train_models
from inference import PatternMatcher
from rule_parser import RuleParser, get_user_structure, extract_patterns_from_users
import polars as pl


# ============================================================================
# JSON SCHEMA FOR VALIDATION
# ============================================================================

SEQUENCE_SCHEMA = {
    "node_types": ["POD", "PORT", "DEPOSITORY_ACCOUNT", "LIABILITY_ACCOUNT"],
    "node_subtypes": {
        "DEPOSITORY_ACCOUNT": ["CHECKING", "SAVINGS"],
        "LIABILITY_ACCOUNT": ["CREDIT_CARD", "LOAN", "LINE_OF_CREDIT"]
    },
    "trigger_types": ["INCOMING_FUNDS", "SCHEDULED"],
    "action_types": [
        "PERCENTAGE", "FIXED", "TOP_UP", "ROUND_DOWN",
        "AVALANCHE", "SNOWBALL", "NEXT_PAYMENT_MINIMUM",
        "TOTAL_AMOUNT_DUE"
    ],
    "required_node_fields": ["id", "type", "name", "balance", "icon", "position"],
    "required_rule_fields": ["id", "sourceId", "trigger", "steps"],
    "required_action_fields": ["type", "sourceId", "destinationId", "amountInCents", "amountInPercentage", "groupIndex"]
}


@dataclass
class ValidationResult:
    valid: bool
    errors: list
    warnings: list


def validate_sequence_json(data: dict) -> ValidationResult:
    """Validate that JSON matches Sequence schema exactly."""
    errors = []
    warnings = []

    # Check top-level structure
    if "nodes" not in data:
        errors.append("Missing 'nodes' array")
    if "rules" not in data:
        errors.append("Missing 'rules' array")
    if "viewport" not in data:
        warnings.append("Missing 'viewport' (will use default)")

    if errors:
        return ValidationResult(False, errors, warnings)

    # Collect node IDs for reference validation
    node_ids = set()

    # Validate nodes
    for i, node in enumerate(data.get("nodes", [])):
        prefix = f"nodes[{i}]"

        # Required fields
        for field in SEQUENCE_SCHEMA["required_node_fields"]:
            if field not in node:
                errors.append(f"{prefix}: Missing required field '{field}'")

        # Type validation
        if node.get("type") not in SEQUENCE_SCHEMA["node_types"]:
            errors.append(f"{prefix}: Invalid type '{node.get('type')}'. Must be one of {SEQUENCE_SCHEMA['node_types']}")

        # Subtype validation
        node_type = node.get("type")
        subtype = node.get("subtype")
        if node_type in SEQUENCE_SCHEMA["node_subtypes"]:
            valid_subtypes = SEQUENCE_SCHEMA["node_subtypes"][node_type]
            if subtype and subtype not in valid_subtypes:
                errors.append(f"{prefix}: Invalid subtype '{subtype}' for {node_type}. Must be one of {valid_subtypes}")
            if node_type == "LIABILITY_ACCOUNT" and not subtype:
                warnings.append(f"{prefix}: LIABILITY_ACCOUNT should have subtype")

        # Position validation
        pos = node.get("position", {})
        if not isinstance(pos.get("x"), (int, float)) or not isinstance(pos.get("y"), (int, float)):
            errors.append(f"{prefix}: position must have numeric x and y")

        # Collect ID
        if node.get("id"):
            node_ids.add(node["id"])

    # Validate rules
    for i, rule in enumerate(data.get("rules", [])):
        prefix = f"rules[{i}]"

        # Required fields
        for field in SEQUENCE_SCHEMA["required_rule_fields"]:
            if field not in rule:
                errors.append(f"{prefix}: Missing required field '{field}'")

        # Source reference validation
        if rule.get("sourceId") and rule["sourceId"] not in node_ids:
            errors.append(f"{prefix}: sourceId '{rule['sourceId']}' references non-existent node")

        # Trigger validation
        trigger = rule.get("trigger", {})
        if trigger.get("type") not in SEQUENCE_SCHEMA["trigger_types"]:
            errors.append(f"{prefix}.trigger: Invalid type '{trigger.get('type')}'")

        if trigger.get("type") == "SCHEDULED" and not trigger.get("cron"):
            warnings.append(f"{prefix}.trigger: SCHEDULED trigger should have cron")

        # Steps/Actions validation
        for j, step in enumerate(rule.get("steps", [])):
            for k, action in enumerate(step.get("actions", [])):
                action_prefix = f"{prefix}.steps[{j}].actions[{k}]"

                # Required fields
                for field in SEQUENCE_SCHEMA["required_action_fields"]:
                    if field not in action:
                        errors.append(f"{action_prefix}: Missing required field '{field}'")

                # Action type validation
                if action.get("type") not in SEQUENCE_SCHEMA["action_types"]:
                    errors.append(f"{action_prefix}: Invalid type '{action.get('type')}'")

                # Reference validation
                if action.get("sourceId") and action["sourceId"] not in node_ids:
                    errors.append(f"{action_prefix}: sourceId references non-existent node")
                if action.get("destinationId") and action["destinationId"] not in node_ids:
                    errors.append(f"{action_prefix}: destinationId references non-existent node")

                # Amount validation
                if action.get("type") == "PERCENTAGE":
                    pct = action.get("amountInPercentage", 0)
                    if not (0 <= pct <= 100):
                        warnings.append(f"{action_prefix}: PERCENTAGE should be 0-100, got {pct}")

                # Liability-only action validation
                liability_only_actions = ["AVALANCHE", "SNOWBALL", "NEXT_PAYMENT_MINIMUM", "TOTAL_AMOUNT_DUE"]
                if action.get("type") in liability_only_actions:
                    dest_id = action.get("destinationId")
                    dest_node = next((n for n in data["nodes"] if n["id"] == dest_id), None)
                    if dest_node and dest_node.get("type") != "LIABILITY_ACCOUNT":
                        errors.append(f"{action_prefix}: {action['type']} can only target LIABILITY_ACCOUNT, got {dest_node.get('type')}")

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )


# ============================================================================
# CONTEXT BUILDER
# ============================================================================

class ContextBuilder:
    """Builds context from ML models for LLM prompt."""

    def __init__(self, models_dir: str = None, data_dir: str = None):
        if models_dir is None:
            models_dir = str(Path(__file__).parent / 'models')
        if data_dir is None:
            data_dir = str(Path(__file__).parent)

        self.parser = RuleParser()
        self.matcher = PatternMatcher(models_dir=models_dir)
        self.rules_df = pl.read_csv(Path(data_dir) / 'Itaytestfinal.csv')

    def build_context(self, prompt: str, profile: dict, num_examples: int = 3) -> dict:
        """
        Build context for LLM from ML models.

        Returns:
            dict with:
            - predicted_goal: ML-predicted goal
            - similar_users: list of similar user summaries
            - example_structures: parsed structures from similar users
            - pattern_stats: common patterns in similar users
        """

        # Get ML predictions
        match_result = self.matcher.match_user(profile, rules_text=prompt)

        similar_users = match_result.get('similar_users', [])
        predicted_goal = match_result.get('predicted_goal', 'AUTOMATE_MY_BUDGETING')
        goal_group = match_result.get('goal_group', 'budget')

        # Get example structures from similar users with good setups
        examples = []
        for user in similar_users[:10]:
            parsed = get_user_structure(self.rules_df, user['user_id'], self.parser)
            if parsed and len(parsed.nodes) >= 3 and len(parsed.rules) >= 2:
                user_goal = user.get('metadata', {}).get('PRODUCTGOAL') or 'Unknown'
                examples.append({
                    'similarity': user['similarity'],
                    'goal': user_goal,
                    'num_nodes': len(parsed.nodes),
                    'num_rules': len(parsed.rules),
                    'nodes': dict(parsed.nodes),
                    'rules_summary': self._summarize_rules(parsed)
                })
                if len(examples) >= num_examples:
                    break

        # Get pattern statistics
        user_ids = [u['user_id'] for u in similar_users[:10]]
        pattern_stats = extract_patterns_from_users(self.rules_df, user_ids, self.parser)

        return {
            'predicted_goal': predicted_goal,
            'goal_group': goal_group,
            'similar_users_count': len(similar_users),
            'examples': examples,
            'common_action_types': dict(pattern_stats['action_types']),
            'common_trigger_types': dict(pattern_stats['trigger_types']),
            'avg_complexity': {
                'rules': pattern_stats['avg_rules_per_user'],
                'nodes': pattern_stats['avg_nodes_per_user']
            }
        }

    def _summarize_rules(self, parsed) -> list:
        """Create human-readable summary of rules."""
        summaries = []
        for rule in parsed.rules[:5]:  # Limit to 5 rules
            actions_desc = []
            for a in rule.actions[:3]:  # Limit actions
                if a.is_percentage:
                    actions_desc.append(f"{a.amount}% to {a.destination}")
                elif a.amount:
                    actions_desc.append(f"${a.amount} to {a.destination}")
                else:
                    actions_desc.append(f"{a.action_type} to {a.destination}")

            summaries.append({
                'trigger': f"{rule.trigger_type} at {rule.trigger_source}",
                'actions': actions_desc
            })
        return summaries


# ============================================================================
# FINANCIAL PLANNING CONTEXT
# ============================================================================

def get_financial_advice(profile: dict) -> dict:
    """Generate profile-specific financial planning advice."""

    user_type = profile.get('USER_TYPE', 'INDIVIDUAL')
    income = profile.get('ANNUALINCOME', '')
    age_group = profile.get('AGE_GROUP', '')
    occupation = profile.get('OCCUPATION', '')

    advice = {
        'allocations': {},
        'priorities': [],
        'notes': []
    }

    # Income bracket mapping
    income_map = {
        'UNDER_25K': 25000,
        'BETWEEN_25K_AND_50K': 37500,
        'BETWEEN_50K_AND_100K': 75000,
        'BETWEEN_100K_AND_250K': 175000,
        'OVER_250K': 350000
    }
    income_val = income_map.get(income, 75000)

    # Business vs Individual allocation
    if user_type == 'BUSINESS':
        # Profit First allocation
        if income_val < 250000:
            advice['allocations'] = {'profit': 5, 'owner_pay': 50, 'tax': 15, 'operating': 30}
        elif income_val < 500000:
            advice['allocations'] = {'profit': 10, 'owner_pay': 35, 'tax': 15, 'operating': 40}
        else:
            advice['allocations'] = {'profit': 15, 'owner_pay': 20, 'tax': 15, 'operating': 50}

        advice['priorities'] = [
            'Tax reserve (25-35% of income)',
            'Operating expenses buffer',
            'Owner pay (consistent monthly)',
            'Profit reserve (quarterly distribution)'
        ]
        advice['notes'].append('Business: Use Profit First method - pay yourself first')

        # Tax reserve rate
        if income_val < 50000:
            advice['tax_reserve'] = '20-25%'
        elif income_val < 100000:
            advice['tax_reserve'] = '25-30%'
        elif income_val < 250000:
            advice['tax_reserve'] = '30-35%'
        else:
            advice['tax_reserve'] = '35-40%'
    else:
        # Individual allocation by age
        age_allocations = {
            '18-24': {'savings': 10, 'debt': 15, 'investments': 5, 'emergency_months': 3},
            '25-34': {'savings': 15, 'debt': 15, 'investments': 15, 'emergency_months': 4},
            '35-44': {'savings': 15, 'debt': 10, 'investments': 20, 'emergency_months': 4},
            '45-54': {'savings': 10, 'debt': 5, 'investments': 25, 'emergency_months': 5},
            '55-64': {'savings': 10, 'debt': 0, 'investments': 20, 'emergency_months': 6},
            '65+': {'savings': 5, 'debt': 0, 'investments': 15, 'emergency_months': 6}
        }

        alloc = age_allocations.get(age_group, {'savings': 15, 'debt': 10, 'investments': 15, 'emergency_months': 4})
        advice['allocations'] = alloc

        advice['priorities'] = [
            'Emergency fund (3-6 months expenses)',
            'High-interest debt payoff (>7% APR)',
            'Retirement contributions',
            'Other savings goals'
        ]

    # Occupation stability affects emergency fund
    high_stability = ['EXECUTIVE_OR_MANAGER', 'ARCHITECT_OR_ENGINEER', 'SCIENTIST_OR_TECHNOLOGIST',
                      'DOCTOR_OR_HEALTHCARE', 'GOVERNMENT']
    variable_income = ['SELF_EMPLOYED', 'FREELANCER', 'GIG_WORKER', 'SALES', 'REAL_ESTATE']

    if occupation in high_stability:
        advice['notes'].append('Stable income: 3-month emergency fund sufficient')
        advice['income_stability'] = 'high'
    elif occupation in variable_income:
        advice['notes'].append('Variable income: Build 6-month emergency fund, smooth income through reserves')
        advice['income_stability'] = 'variable'
    else:
        advice['income_stability'] = 'medium'

    return advice


# ============================================================================
# LLM GENERATOR
# ============================================================================

SYSTEM_PROMPT = """# CONTEXT
You are generating JSON for Sequence, a financial automation platform. Users describe their money management goals, and you create a "map" that automates fund movements between accounts.

# OUTPUT FORMAT
Return ONLY valid JSON matching this exact structure:

```json
{
  "name": "Short Map Name (max 4 words)",
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
        "type": "INCOMING_FUNDS|SCHEDULED",
        "sourceId": "node-id",
        "cron": null|"0 0 1 * *"
      },
      "steps": [{
        "actions": [{
          "type": "PERCENTAGE|FIXED|REMAINDER|TOP_UP|AVALANCHE|SNOWBALL",
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
```

Node types:
- PORT: Income entry point (salary, deposits)
- POD: Virtual envelope for budgeting
- DEPOSITORY_ACCOUNT: Bank account (CHECKING/SAVINGS)
- LIABILITY_ACCOUNT: Debt (CREDIT_CARD/LOAN/LINE_OF_CREDIT)

# FINANCIAL PLANNING RULES

## Allocation Priority (fund in order):
1. Minimum debt payments - always first
2. Emergency buffer ($1,000 starter)
3. Employer 401k match (if applicable)
4. High-interest debt (>7% APR)
5. Full emergency fund (3-6 months)
6. Retirement/investments
7. Other goals

## Age-Based Guidelines:
| Age | Savings | Debt Priority | Investments |
|-----|---------|---------------|-------------|
| 18-24 | 10-15% | Aggressive | 5-10% |
| 25-34 | 15-20% | High | 15-20% |
| 35-44 | 15% | Moderate | 20-25% |
| 45-54 | 10% | Low | 25-30% |
| 55+ | 5-10% | Minimal | 15-20% |

## Business Owner (Profit First):
| Revenue | Profit | Owner Pay | Tax | Operating |
|---------|--------|-----------|-----|-----------|
| <$250k | 5% | 50% | 15% | 30% |
| $250k-$500k | 10% | 35% | 15% | 40% |
| $500k+ | 15% | 20% | 15% | 50% |

## Tax Reserve (Self-Employed):
- Under $50k income: 20-25%
- $50k-$100k: 25-30%
- $100k-$250k: 30-35%
- $250k+: 35-40%

## Emergency Fund by Income Stability:
- High stability (salaried, govt): 3 months
- Medium (professional): 4 months
- Variable (freelance, sales): 6 months
- Seasonal: 6+ months

# CRITICAL: BALANCED MAP REQUIREMENT

NEVER send 100% of income from a PORT to a single destination. Real financial planning requires distributing income across multiple buckets.

## Design Principle: Cover Essentials While Accounting for the Goal
Think holistically about the user's financial life. Their stated goal should be prominent, but not the ONLY destination.

## Goal-Aware Distribution:
| User's Goal | Suggested Split |
|-------------|-----------------|
| Debt payoff | 60% debt, 25% bills/expenses, 15% small savings buffer |
| Savings | 40% savings, 15% investments, 45% bills/expenses |
| Investing | 35% investments, 15% savings, 50% bills/expenses |
| Budgeting | 50% needs, 30% wants, 20% savings |
| Emergency fund | 30% emergency, 20% general savings, 50% expenses |

## When User Focuses on One Thing:
- Make it the LARGEST allocation, not the ONLY allocation
- Include at least 2-3 destinations from income
- Bills/expenses POD covers the "rest of life" - always useful to include

## Bad vs Good:
BAD: User says "avalanche my debt" → 100% to credit card
GOOD: User says "avalanche my debt" → 60% debt (AVALANCHE), 25% bills, 15% savings

Action types:
- PERCENTAGE: amountInPercentage 0-100 (use 100 with higher groupIndex for "remainder")
- FIXED: amountInCents (dollars × 100)

CRITICAL - How to implement "remainder" (send everything left):
- There is NO "REMAINDER" type
- Use PERCENTAGE with amountInPercentage: 100 and a HIGHER groupIndex
- Actions execute in groupIndex order: lower indexes first
- Each action operates on what's LEFT after previous actions
- Example: groupIndex 0 takes 10%, groupIndex 1 with 100% takes all remaining 90%

# EXAMPLES

## Example 1: Simple Budget
Input: "I get paid bi-weekly and want to save 20% and pay my rent" (Income: BETWEEN_50K_AND_100K)
Output:
{"name":"Save & Pay Rent","nodes":[{"id":"a1b2c3d4-e5f6-7890-abcd-ef1234567890","type":"PORT","subtype":null,"name":"Paycheck","balance":312500,"icon":"📥","position":{"x":100,"y":300}},{"id":"b2c3d4e5-f6a7-8901-bcde-f12345678901","type":"POD","subtype":null,"name":"Main Hub","balance":45000,"icon":"💰","position":{"x":400,"y":300}},{"id":"c3d4e5f6-a7b8-9012-cdef-123456789012","type":"DEPOSITORY_ACCOUNT","subtype":"SAVINGS","name":"Savings","balance":75000,"icon":"🏦","position":{"x":700,"y":200}},{"id":"d4e5f6a7-b8c9-0123-def0-234567890123","type":"POD","subtype":null,"name":"Rent","balance":62000,"icon":"🏠","position":{"x":700,"y":400}}],"rules":[{"id":"e5f6a7b8-c9d0-1234-ef01-345678901234","sourceId":"a1b2c3d4-e5f6-7890-abcd-ef1234567890","trigger":{"type":"INCOMING_FUNDS","sourceId":"a1b2c3d4-e5f6-7890-abcd-ef1234567890","cron":null},"steps":[{"actions":[{"type":"PERCENTAGE","sourceId":"a1b2c3d4-e5f6-7890-abcd-ef1234567890","destinationId":"b2c3d4e5-f6a7-8901-bcde-f12345678901","amountInCents":0,"amountInPercentage":100,"groupIndex":0,"limit":null,"upToEnabled":null}]}]},{"id":"f6a7b8c9-d0e1-2345-f012-456789012345","sourceId":"b2c3d4e5-f6a7-8901-bcde-f12345678901","trigger":{"type":"INCOMING_FUNDS","sourceId":"b2c3d4e5-f6a7-8901-bcde-f12345678901","cron":null},"steps":[{"actions":[{"type":"PERCENTAGE","sourceId":"b2c3d4e5-f6a7-8901-bcde-f12345678901","destinationId":"c3d4e5f6-a7b8-9012-cdef-123456789012","amountInCents":0,"amountInPercentage":20,"groupIndex":0,"limit":null,"upToEnabled":null},{"type":"FIXED","sourceId":"b2c3d4e5-f6a7-8901-bcde-f12345678901","destinationId":"d4e5f6a7-b8c9-0123-def0-234567890123","amountInCents":150000,"amountInPercentage":0,"groupIndex":1,"limit":null,"upToEnabled":null}]}]}],"viewport":{"x":300,"y":100,"zoom":0.9}}

## Example 2: Business Owner with Tax Reserve
Input: "Small business owner, need to set aside 30% for taxes and pay myself a salary" (Income: BETWEEN_100K_AND_250K)
Output:
{"name":"Business Tax Setup","nodes":[{"id":"11111111-1111-1111-1111-111111111111","type":"PORT","subtype":null,"name":"Business Income","balance":729100,"icon":"📥","position":{"x":100,"y":300}},{"id":"22222222-2222-2222-2222-222222222222","type":"POD","subtype":null,"name":"Tax Reserve","balance":58000,"icon":"📋","position":{"x":400,"y":150}},{"id":"33333333-3333-3333-3333-333333333333","type":"POD","subtype":null,"name":"Owner Pay","balance":41000,"icon":"💵","position":{"x":400,"y":450}},{"id":"44444444-4444-4444-4444-444444444444","type":"DEPOSITORY_ACCOUNT","subtype":"CHECKING","name":"Business Checking","balance":93000,"icon":"🏦","position":{"x":700,"y":300}}],"rules":[{"id":"55555555-5555-5555-5555-555555555555","sourceId":"11111111-1111-1111-1111-111111111111","trigger":{"type":"INCOMING_FUNDS","sourceId":"11111111-1111-1111-1111-111111111111","cron":null},"steps":[{"actions":[{"type":"PERCENTAGE","sourceId":"11111111-1111-1111-1111-111111111111","destinationId":"22222222-2222-2222-2222-222222222222","amountInCents":0,"amountInPercentage":30,"groupIndex":0,"limit":null,"upToEnabled":null},{"type":"FIXED","sourceId":"11111111-1111-1111-1111-111111111111","destinationId":"33333333-3333-3333-3333-333333333333","amountInCents":500000,"amountInPercentage":0,"groupIndex":1,"limit":null,"upToEnabled":null},{"type":"PERCENTAGE","sourceId":"11111111-1111-1111-1111-111111111111","destinationId":"44444444-4444-4444-4444-444444444444","amountInCents":0,"amountInPercentage":100,"groupIndex":2,"limit":null,"upToEnabled":null}]}]}],"viewport":{"x":300,"y":100,"zoom":0.9}}

## Example 3: Debt-Focused with Balanced Distribution
Input: "I want to aggressively pay down my credit card debt using avalanche method" (Income: BETWEEN_25K_AND_50K)
Output:
{"name":"Debt Freedom Plan","nodes":[{"id":"aaaa1111-bbbb-cccc-dddd-eeee11111111","type":"PORT","subtype":null,"name":"Income","balance":156200,"icon":"📥","position":{"x":100,"y":300}},{"id":"aaaa2222-bbbb-cccc-dddd-eeee22222222","type":"POD","subtype":null,"name":"Bills & Expenses","balance":72000,"icon":"🏠","position":{"x":400,"y":100}},{"id":"aaaa3333-bbbb-cccc-dddd-eeee33333333","type":"POD","subtype":null,"name":"Safety Buffer","balance":34000,"icon":"🛡️","position":{"x":400,"y":300}},{"id":"aaaa4444-bbbb-cccc-dddd-eeee44444444","type":"LIABILITY_ACCOUNT","subtype":"CREDIT_CARD","name":"Credit Card","balance":142000,"icon":"💳","position":{"x":400,"y":500}}],"rules":[{"id":"aaaa5555-bbbb-cccc-dddd-eeee55555555","sourceId":"aaaa1111-bbbb-cccc-dddd-eeee11111111","trigger":{"type":"INCOMING_FUNDS","sourceId":"aaaa1111-bbbb-cccc-dddd-eeee11111111","cron":null},"steps":[{"actions":[{"type":"PERCENTAGE","sourceId":"aaaa1111-bbbb-cccc-dddd-eeee11111111","destinationId":"aaaa2222-bbbb-cccc-dddd-eeee22222222","amountInCents":0,"amountInPercentage":25,"groupIndex":0,"limit":null,"upToEnabled":null},{"type":"PERCENTAGE","sourceId":"aaaa1111-bbbb-cccc-dddd-eeee11111111","destinationId":"aaaa3333-bbbb-cccc-dddd-eeee33333333","amountInCents":0,"amountInPercentage":15,"groupIndex":0,"limit":null,"upToEnabled":null},{"type":"AVALANCHE","sourceId":"aaaa1111-bbbb-cccc-dddd-eeee11111111","destinationId":"aaaa4444-bbbb-cccc-dddd-eeee44444444","amountInCents":0,"amountInPercentage":100,"groupIndex":1,"limit":null,"upToEnabled":null}]}]}],"viewport":{"x":300,"y":100,"zoom":0.9}}

## Example 4: Multiple Income Sources
Input: "I have a day job and freelance on the side. Keep freelance money separate for taxes." (Income: BETWEEN_50K_AND_100K)
Output:
{"name":"Dual Income Setup","nodes":[{"id":"job11111-2222-3333-4444-555566667777","type":"PORT","subtype":null,"name":"Salary","balance":250000,"icon":"📥","position":{"x":100,"y":200}},{"id":"free1111-2222-3333-4444-555566667777","type":"PORT","subtype":null,"name":"Freelance","balance":62500,"icon":"💼","position":{"x":100,"y":500}},{"id":"main1111-2222-3333-4444-555566667777","type":"DEPOSITORY_ACCOUNT","subtype":"CHECKING","name":"Main Checking","balance":85000,"icon":"🏦","position":{"x":500,"y":200}},{"id":"tax11111-2222-3333-4444-555566667777","type":"POD","subtype":null,"name":"Freelance Taxes","balance":47000,"icon":"📋","position":{"x":500,"y":400}},{"id":"bus11111-2222-3333-4444-555566667777","type":"DEPOSITORY_ACCOUNT","subtype":"CHECKING","name":"Business Account","balance":63000,"icon":"🏦","position":{"x":500,"y":600}}],"rules":[{"id":"rule1111-2222-3333-4444-555566667777","sourceId":"job11111-2222-3333-4444-555566667777","trigger":{"type":"INCOMING_FUNDS","sourceId":"job11111-2222-3333-4444-555566667777","cron":null},"steps":[{"actions":[{"type":"PERCENTAGE","sourceId":"job11111-2222-3333-4444-555566667777","destinationId":"main1111-2222-3333-4444-555566667777","amountInCents":0,"amountInPercentage":100,"groupIndex":0,"limit":null,"upToEnabled":null}]}]},{"id":"rule2222-2222-3333-4444-555566667777","sourceId":"free1111-2222-3333-4444-555566667777","trigger":{"type":"INCOMING_FUNDS","sourceId":"free1111-2222-3333-4444-555566667777","cron":null},"steps":[{"actions":[{"type":"PERCENTAGE","sourceId":"free1111-2222-3333-4444-555566667777","destinationId":"tax11111-2222-3333-4444-555566667777","amountInCents":0,"amountInPercentage":30,"groupIndex":0,"limit":null,"upToEnabled":null},{"type":"PERCENTAGE","sourceId":"free1111-2222-3333-4444-555566667777","destinationId":"bus11111-2222-3333-4444-555566667777","amountInCents":0,"amountInPercentage":100,"groupIndex":1,"limit":null,"upToEnabled":null}]}]}],"viewport":{"x":300,"y":100,"zoom":0.9}}

# CONSTRAINTS
1. Generate ONLY the JSON object - no explanations, no markdown code blocks
2. All IDs must be valid UUIDs (8-4-4-4-12 format)
3. All sourceId/destinationId must reference existing node IDs
4. PERCENTAGE: amountInPercentage 0-100, amountInCents=0
5. FIXED: amountInCents in cents (e.g., $500 = 50000), amountInPercentage=0
6. Position nodes left-to-right: income (x~100) → processing (x~400) → destinations (x~700)
7. Icons: 📥 PORT, 💰 POD, 🏦 DEPOSITORY, 💳 LIABILITY
8. Every action needs all required fields: type, sourceId, destinationId, amountInCents, amountInPercentage, groupIndex, limit, upToEnabled

# MAP NAME
- Create a short, descriptive name (max 4 words)
- Examples: "Debt Freedom Plan", "Smart Budget", "Business Tax Setup", "Save & Invest"
- DO NOT use the user's full prompt as the name

# REALISTIC BALANCES (in cents)
IMPORTANT: ALL nodes MUST have non-zero balances. Never use 0.

For PORT (income) nodes, use bi-weekly pay based on income bracket:
| Income Bracket | Balance (cents) |
|----------------|-----------------|
| UNDER_25K | 83300 |
| BETWEEN_25K_AND_50K | 156200 |
| BETWEEN_50K_AND_100K | 312500 |
| BETWEEN_100K_AND_250K | 729100 |
| OVER_250K | 1458300 |
| Unknown/not provided | 312500 |

For ALL other nodes, generate a random balance between 20000-100000 cents ($200-$1000) in increments of 1000:
- POD nodes: Random 20000-100000 (e.g., 45000, 67000, 82000)
- DEPOSITORY_ACCOUNT: Random 20000-100000 (e.g., 53000, 78000)
- LIABILITY_ACCOUNT: Random 50000-200000 for realistic debt (e.g., 85000, 142000)

NEVER generate balance: 0 for any node."""


class LLMGenerator:
    """Generate Sequence maps using Claude API with ML context."""

    def __init__(self, api_key: str = None, models_dir: str = None, data_dir: str = None):
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.context_builder = ContextBuilder(models_dir=models_dir, data_dir=data_dir)

    def generate(self, prompt: str, profile: dict = None, max_retries: int = 2) -> dict:
        """
        Generate a Sequence map from user prompt.

        Args:
            prompt: User's description of what they want
            profile: Optional user profile (USER_TYPE, ANNUALINCOME, etc.)
            max_retries: Number of retries if validation fails

        Returns:
            dict with:
            - json_map: The generated map (if valid)
            - validation: ValidationResult
            - context: ML context used
            - error: Error message (if failed)
        """
        profile = profile or {}

        # Build context from ML models
        print("Building context from ML models...")
        context = self.context_builder.build_context(prompt, profile)

        # Build user message with context
        user_message = self._build_user_message(prompt, profile, context)

        # Try generation with retries
        last_error = None
        for attempt in range(max_retries + 1):
            print(f"Calling Claude API (attempt {attempt + 1})...")

            try:
                response = self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=4096,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_message}]
                )

                # Extract JSON from response
                response_text = response.content[0].text.strip()

                # Try to parse JSON
                try:
                    # Handle potential markdown code blocks
                    if response_text.startswith("```"):
                        response_text = response_text.split("```")[1]
                        if response_text.startswith("json"):
                            response_text = response_text[4:]
                        response_text = response_text.strip()

                    json_map = json.loads(response_text)
                except json.JSONDecodeError as e:
                    last_error = f"Invalid JSON: {e}"
                    user_message += f"\n\nYour previous response was not valid JSON. Error: {e}\nPlease return ONLY valid JSON."
                    continue

                # Validate
                validation = validate_sequence_json(json_map)

                if validation.valid:
                    print(f"Generated valid map with {len(json_map['nodes'])} nodes, {len(json_map['rules'])} rules")
                    if validation.warnings:
                        print(f"Warnings: {validation.warnings}")

                    return {
                        'json_map': json_map,
                        'validation': validation,
                        'context': context,
                        'error': None
                    }
                else:
                    last_error = f"Validation errors: {validation.errors}"
                    user_message += f"\n\nYour previous response had validation errors:\n{validation.errors}\nPlease fix and return valid JSON."

            except Exception as e:
                last_error = str(e)
                print(f"API error: {e}")

        return {
            'json_map': None,
            'validation': None,
            'context': context,
            'error': last_error
        }

    def _build_user_message(self, prompt: str, profile: dict, context: dict) -> str:
        """Build the user message with all context."""

        # Build ML context section
        ml_context = []

        # User profile
        user_type = profile.get('USER_TYPE', 'INDIVIDUAL')
        income = profile.get('ANNUALINCOME', 'Unknown')
        occupation = profile.get('OCCUPATION', 'Unknown')
        age_group = profile.get('AGE_GROUP', 'Unknown')

        ml_context.append(f"User type: {user_type}")
        if income != 'Unknown':
            ml_context.append(f"Income bracket: {income}")
        if occupation != 'Unknown':
            ml_context.append(f"Occupation: {occupation}")
        if age_group != 'Unknown':
            ml_context.append(f"Age group: {age_group}")

        # ML predictions
        ml_context.append(f"Predicted goal: {context['predicted_goal']}")
        ml_context.append(f"Goal category: {context['goal_group']}")

        # Pattern insights
        if context['common_action_types']:
            top_actions = sorted(context['common_action_types'].items(), key=lambda x: -x[1])[:3]
            action_summary = ", ".join([f"{a[0]}({a[1]})" for a in top_actions])
            ml_context.append(f"Popular actions among similar users: {action_summary}")

        avg_nodes = context['avg_complexity']['nodes']
        avg_rules = context['avg_complexity']['rules']
        if avg_nodes > 0:
            ml_context.append(f"Typical complexity: {avg_nodes:.0f} nodes, {avg_rules:.0f} rules")

        # Get financial planning advice based on profile
        fin_advice = get_financial_advice(profile)

        # Build financial advice section
        fin_context = []
        if fin_advice['allocations']:
            alloc = fin_advice['allocations']
            if 'profit' in alloc:  # Business
                fin_context.append(f"Recommended split: {alloc['profit']}% profit, {alloc['owner_pay']}% owner pay, {alloc['tax']}% tax, {alloc['operating']}% operating")
            else:  # Individual
                fin_context.append(f"Recommended: {alloc.get('savings', 15)}% savings, {alloc.get('investments', 15)}% investments")
                if alloc.get('emergency_months'):
                    fin_context.append(f"Emergency fund target: {alloc['emergency_months']} months expenses")

        if fin_advice.get('tax_reserve'):
            fin_context.append(f"Tax reserve: {fin_advice['tax_reserve']} of income")

        for note in fin_advice.get('notes', []):
            fin_context.append(note)

        if fin_advice['priorities']:
            fin_context.append(f"Priority order: {' → '.join(fin_advice['priorities'][:3])}")

        # Build message
        msg = f"""# USER REQUEST
"{prompt}"

# ML ANALYSIS (use as guidance)
{chr(10).join('- ' + item for item in ml_context)}

# FINANCIAL PLANNING ADVICE (based on profile)
{chr(10).join('- ' + item for item in fin_context) if fin_context else '- Use standard 50/30/20 allocation'}

"""

        # Add example patterns from similar users (if available and useful)
        if context['examples']:
            msg += "# PATTERNS FROM SIMILAR USERS\n"
            for i, ex in enumerate(context['examples'][:2], 1):  # Limit to 2 examples
                msg += f"\nPattern {i} (Goal: {ex['goal']}):\n"
                msg += f"- Nodes used: {', '.join(ex['nodes'].keys())}\n"
                if ex['rules_summary']:
                    rule = ex['rules_summary'][0]
                    msg += f"- Flow example: {rule['trigger']} → {', '.join(rule['actions'][:2])}\n"

        msg += """
# TASK
Generate a Sequence map JSON for this user's request.

Requirements:
1. Create nodes for all accounts/categories mentioned or implied
2. Create rules that automate the money flow the user described
3. Use appropriate action types (PERCENTAGE for ratios, FIXED for amounts, PERCENTAGE 100% with higher groupIndex for "everything else")
4. Return ONLY valid JSON - no explanations, no code blocks

Output:"""

        return msg


# ============================================================================
# MAIN / CLI
# ============================================================================

def generate_map(prompt: str, profile: dict = None, api_key: str = None) -> dict:
    """Convenience function to generate a map."""
    generator = LLMGenerator(api_key=api_key)
    return generator.generate(prompt, profile)


if __name__ == '__main__':
    import sys

    # Check for API key
    if not os.environ.get('ANTHROPIC_API_KEY'):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Set it with: export ANTHROPIC_API_KEY='your-key'")
        sys.exit(1)

    # Demo
    print("="*70)
    print("LLM-Based Sequence Map Generator")
    print("="*70)

    result = generate_map(
        prompt="I'm a small business owner making $190,000 and looking to organize my finances with tax reserves and profit allocation",
        profile={
            'USER_TYPE': 'BUSINESS',
            'ANNUALINCOME': 'BETWEEN_100K_AND_250K'
        }
    )

    if result['error']:
        print(f"\nError: {result['error']}")
    else:
        print("\n" + "="*70)
        print("GENERATED MAP")
        print("="*70)
        print(json.dumps(result['json_map'], indent=2))

        print("\n" + "="*70)
        print("VALIDATION")
        print("="*70)
        print(f"Valid: {result['validation'].valid}")
        if result['validation'].warnings:
            print(f"Warnings: {result['validation'].warnings}")
