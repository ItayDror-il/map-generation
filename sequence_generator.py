"""
Sequence Map Generator - End-to-End Pipeline
Transforms user prompts into structured plans and Sequence-compatible JSON.

Usage:
    from sequence_generator import SequenceMapGenerator

    generator = SequenceMapGenerator()
    result = generator.generate(
        prompt="Help me pay off my credit cards faster",
        profile={'USER_TYPE': 'INDIVIDUAL', 'ANNUALINCOME': 'BETWEEN_50K_AND_100K'}
    )
    print(result['plan'])       # Human-readable plan
    print(result['json_map'])   # Sequence-compatible JSON
"""

import json
import uuid
import re
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

# Import the pattern matcher (handle import errors gracefully)
try:
    from inference import PatternMatcher
    MATCHER_AVAILABLE = True
except ImportError:
    PatternMatcher = None
    MATCHER_AVAILABLE = False


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Node:
    """Represents a node in the Sequence map."""
    id: str
    type: str  # POD, PORT, DEPOSITORY_ACCOUNT, LIABILITY_ACCOUNT
    name: str
    subtype: Optional[str] = None  # CHECKING, SAVINGS, CREDIT_CARD, LOAN
    balance: int = 0  # In cents
    icon: str = "💰"
    position: dict = field(default_factory=lambda: {"x": 0, "y": 0})
    target_amount: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "subtype": self.subtype,
            "name": self.name,
            "balance": self.balance,
            "icon": self.icon,
            "position": self.position
        }


@dataclass
class Action:
    """Represents an action in a rule."""
    type: str  # PERCENTAGE, FIXED, REMAINDER, TOP_UP, AVALANCHE, etc.
    source_id: str
    destination_id: str
    amount_cents: int = 0
    amount_percentage: float = 0
    group_index: int = 0
    limit: Optional[int] = None
    up_to_enabled: Optional[bool] = None

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "sourceId": self.source_id,
            "destinationId": self.destination_id,
            "amountInCents": self.amount_cents,
            "amountInPercentage": self.amount_percentage,
            "groupIndex": self.group_index,
            "limit": self.limit,
            "upToEnabled": self.up_to_enabled
        }


@dataclass
class Rule:
    """Represents a rule in the Sequence map."""
    id: str
    source_id: str
    trigger_type: str  # INCOMING_FUNDS, SCHEDULED, BALANCE_THRESHOLD
    actions: list
    cron: Optional[str] = None
    condition: Optional[dict] = None
    name: Optional[str] = None

    def to_dict(self) -> dict:
        trigger = {
            "type": self.trigger_type,
            "sourceId": self.source_id,
            "cron": self.cron
        }
        if self.condition:
            trigger["condition"] = self.condition

        return {
            "id": self.id,
            "sourceId": self.source_id,
            "trigger": trigger,
            "steps": [{"actions": [a.to_dict() for a in self.actions]}]
        }


# ============================================================================
# GOAL DETECTION
# ============================================================================

GOAL_KEYWORDS = {
    'debt': {
        'keywords': ['debt', 'credit card', 'loan', 'pay off', 'payoff', 'owe', 'balance', 'interest'],
        'goal': 'PAY_OFF_DEBT',
        'pattern': 'debt_payoff'
    },
    'budget': {
        'keywords': ['budget', 'allocate', 'spend', 'category', 'envelope', '50/30/20', 'needs', 'wants'],
        'goal': 'AUTOMATE_MY_BUDGETING',
        'pattern': 'budget_allocation'
    },
    'savings': {
        'keywords': ['save', 'savings', 'emergency fund', 'goal', 'vacation', 'house', 'down payment'],
        'goal': 'MAXIMIZE_SAVINGS',
        'pattern': 'savings_first'
    },
    'bills': {
        'keywords': ['bill', 'rent', 'utilities', 'automatic', 'due date', 'never miss'],
        'goal': 'AUTOMATE_MY_BILLS',
        'pattern': 'bill_automation'
    },
    'business': {
        'keywords': ['business', 'profit first', 'owner pay', 'operating', 'payroll', 'tax reserve', 'opex'],
        'goal': 'PROFIT_FIRST',
        'pattern': 'profit_first'
    },
    'taxes': {
        'keywords': ['tax', 'taxes', 'irs', 'quarterly', 'estimated', 'withhold'],
        'goal': 'SAVE_FOR_TAXES',
        'pattern': 'tax_reserve'
    },
    'cash_flow': {
        'keywords': ['cash flow', 'sweep', 'consolidate', 'overflow', 'threshold'],
        'goal': 'OPTIMIZE_CASH_FLOW',
        'pattern': 'sweep_pattern'
    }
}


def detect_goal(prompt: str) -> dict:
    """Detect the primary financial goal from user prompt."""
    prompt_lower = prompt.lower()

    scores = {}
    for goal_type, config in GOAL_KEYWORDS.items():
        score = sum(1 for kw in config['keywords'] if kw in prompt_lower)
        if score > 0:
            scores[goal_type] = score

    if not scores:
        return {'goal': 'AUTOMATE_MY_BUDGETING', 'pattern': 'budget_allocation', 'type': 'budget'}

    best_match = max(scores.items(), key=lambda x: x[1])
    config = GOAL_KEYWORDS[best_match[0]]
    return {
        'goal': config['goal'],
        'pattern': config['pattern'],
        'type': best_match[0]
    }


def extract_amounts(prompt: str) -> dict:
    """Extract dollar amounts and percentages from prompt."""
    amounts = {
        'dollars': [],
        'percentages': []
    }

    # Find dollar amounts ($1,000 or $1000 or $1000.00)
    dollar_pattern = r'\$[\d,]+(?:\.\d{2})?'
    for match in re.findall(dollar_pattern, prompt):
        value = float(match.replace('$', '').replace(',', ''))
        amounts['dollars'].append(value)

    # Find percentages
    pct_pattern = r'(\d+(?:\.\d+)?)\s*%'
    for match in re.findall(pct_pattern, prompt):
        amounts['percentages'].append(float(match))

    return amounts


def extract_accounts(prompt: str) -> dict:
    """Extract mentioned accounts from prompt."""
    accounts = {
        'income_sources': [],
        'credit_cards': [],
        'banks': [],
        'goals': []
    }

    prompt_lower = prompt.lower()

    # Common credit card names
    cc_names = ['chase', 'amex', 'american express', 'discover', 'capital one',
                'citi', 'bank of america', 'wells fargo', 'barclays']
    for name in cc_names:
        if name in prompt_lower:
            accounts['credit_cards'].append(name.title())

    # Income sources
    income_sources = ['paycheck', 'salary', 'direct deposit', 'freelance',
                      'side gig', 'rental', 'doordash', 'uber']
    for source in income_sources:
        if source in prompt_lower:
            accounts['income_sources'].append(source.title())

    # Savings goals
    goals = ['emergency', 'vacation', 'house', 'car', 'wedding', 'retirement']
    for goal in goals:
        if goal in prompt_lower:
            accounts['goals'].append(goal.title())

    return accounts


# ============================================================================
# PATTERN TEMPLATES
# ============================================================================

class PatternTemplates:
    """Templates for different automation patterns."""

    @staticmethod
    def debt_payoff(profile: dict, extracted: dict) -> dict:
        """Generate debt payoff pattern."""
        nodes = [
            Node(str(uuid.uuid4()), "PORT", "Income", icon="📥"),
            Node(str(uuid.uuid4()), "POD", "Router", icon="🔀"),
            Node(str(uuid.uuid4()), "POD", "Bills & Essentials", icon="📋"),
            Node(str(uuid.uuid4()), "POD", "Debt Payment", icon="💳"),
            Node(str(uuid.uuid4()), "POD", "Emergency Fund", icon="🛡️", target_amount=100000),  # $1000 starter
        ]

        # Add credit cards
        cc_names = extracted.get('accounts', {}).get('credit_cards', [])
        if not cc_names:
            cc_names = ['Credit Card 1', 'Credit Card 2']

        for name in cc_names[:3]:  # Max 3 cards
            nodes.append(Node(
                str(uuid.uuid4()),
                "LIABILITY_ACCOUNT",
                name,
                subtype="CREDIT_CARD",
                icon="💳"
            ))

        liabilities = [n for n in nodes if n.type == "LIABILITY_ACCOUNT"]
        income = nodes[0]
        router = nodes[1]
        bills = nodes[2]
        debt_pod = nodes[3]
        emergency = nodes[4]

        # Determine percentages based on income
        income_level = profile.get('ANNUALINCOME', 'BETWEEN_50K_AND_100K')
        if income_level in ['UP_TO_10K', 'BETWEEN_10K_AND_25K']:
            debt_pct = 20
            bills_pct = 70
        elif income_level in ['BETWEEN_25K_AND_50K', 'BETWEEN_50K_AND_100K']:
            debt_pct = 30
            bills_pct = 60
        else:
            debt_pct = 40
            bills_pct = 50

        rules = [
            Rule(
                str(uuid.uuid4()), income.id, "INCOMING_FUNDS",
                [Action("PERCENTAGE", income.id, router.id, amount_percentage=100)],
                name="Route all income"
            ),
            Rule(
                str(uuid.uuid4()), router.id, "INCOMING_FUNDS",
                [
                    Action("TOP_UP", router.id, emergency.id, group_index=0, limit=100000, up_to_enabled=True),
                    Action("PERCENTAGE", router.id, bills.id, amount_percentage=bills_pct, group_index=1),
                    Action("PERCENTAGE", router.id, debt_pod.id, amount_percentage=debt_pct, group_index=1),
                ],
                name="Allocate to bills and debt"
            ),
            Rule(
                str(uuid.uuid4()), debt_pod.id, "INCOMING_FUNDS",
                [Action("AVALANCHE", debt_pod.id, l.id, amount_percentage=100, group_index=0)
                 for l in liabilities],
                name="Pay debt using avalanche method"
            )
        ]

        return {
            'nodes': nodes,
            'rules': rules,
            'description': f"Debt payoff using avalanche method. {bills_pct}% to bills, {debt_pct}% to debt."
        }

    @staticmethod
    def budget_allocation(profile: dict, extracted: dict) -> dict:
        """Generate budget allocation pattern (50/30/20 style)."""
        nodes = [
            Node(str(uuid.uuid4()), "PORT", "Income", icon="📥"),
            Node(str(uuid.uuid4()), "POD", "Router", icon="🔀"),
            Node(str(uuid.uuid4()), "POD", "Needs", icon="🏠"),
            Node(str(uuid.uuid4()), "POD", "Wants", icon="🎉"),
            Node(str(uuid.uuid4()), "POD", "Savings", icon="💰"),
        ]

        # Use custom percentages if provided
        percentages = extracted.get('amounts', {}).get('percentages', [])
        if len(percentages) >= 3:
            needs_pct, wants_pct, savings_pct = percentages[:3]
        else:
            needs_pct, wants_pct, savings_pct = 50, 30, 20

        income = nodes[0]
        router = nodes[1]
        needs = nodes[2]
        wants = nodes[3]
        savings = nodes[4]

        rules = [
            Rule(
                str(uuid.uuid4()), income.id, "INCOMING_FUNDS",
                [Action("PERCENTAGE", income.id, router.id, amount_percentage=100)],
                name="Route all income"
            ),
            Rule(
                str(uuid.uuid4()), router.id, "INCOMING_FUNDS",
                [
                    Action("PERCENTAGE", router.id, needs.id, amount_percentage=needs_pct, group_index=0),
                    Action("PERCENTAGE", router.id, wants.id, amount_percentage=wants_pct, group_index=0),
                    Action("PERCENTAGE", router.id, savings.id, amount_percentage=savings_pct, group_index=0),
                ],
                name=f"Allocate {needs_pct}/{wants_pct}/{savings_pct}"
            )
        ]

        return {
            'nodes': nodes,
            'rules': rules,
            'description': f"Budget allocation: {needs_pct}% needs, {wants_pct}% wants, {savings_pct}% savings"
        }

    @staticmethod
    def savings_first(profile: dict, extracted: dict) -> dict:
        """Generate pay-yourself-first savings pattern."""
        nodes = [
            Node(str(uuid.uuid4()), "PORT", "Income", icon="📥"),
            Node(str(uuid.uuid4()), "POD", "Savings", icon="💰"),
            Node(str(uuid.uuid4()), "POD", "Spending", icon="💵"),
        ]

        # Add goal-specific pods
        goals = extracted.get('accounts', {}).get('goals', [])
        for goal in goals[:3]:
            nodes.append(Node(str(uuid.uuid4()), "POD", f"{goal} Fund", icon="🎯"))

        # Determine savings rate
        percentages = extracted.get('amounts', {}).get('percentages', [])
        savings_pct = percentages[0] if percentages else 20

        income = nodes[0]
        savings = nodes[1]
        spending = nodes[2]

        rules = [
            Rule(
                str(uuid.uuid4()), income.id, "INCOMING_FUNDS",
                [
                    Action("PERCENTAGE", income.id, savings.id, amount_percentage=savings_pct, group_index=0),
                    Action("REMAINDER", income.id, spending.id, group_index=1),
                ],
                name=f"Save {savings_pct}% first, rest to spending"
            )
        ]

        return {
            'nodes': nodes,
            'rules': rules,
            'description': f"Pay yourself first: {savings_pct}% to savings before anything else"
        }

    @staticmethod
    def profit_first(profile: dict, extracted: dict) -> dict:
        """Generate Profit First business pattern."""
        nodes = [
            Node(str(uuid.uuid4()), "PORT", "Revenue", icon="📥"),
            Node(str(uuid.uuid4()), "DEPOSITORY_ACCOUNT", "Operating Account", subtype="CHECKING", icon="🏦"),
            Node(str(uuid.uuid4()), "POD", "Profit", icon="📈"),
            Node(str(uuid.uuid4()), "POD", "Owner Pay", icon="👤"),
            Node(str(uuid.uuid4()), "POD", "Tax Reserve", icon="🏛️"),
            Node(str(uuid.uuid4()), "POD", "Operating Expenses", icon="⚙️"),
        ]

        revenue = nodes[0]
        operating = nodes[1]
        profit = nodes[2]
        owner = nodes[3]
        tax = nodes[4]
        opex = nodes[5]

        rules = [
            Rule(
                str(uuid.uuid4()), revenue.id, "INCOMING_FUNDS",
                [Action("PERCENTAGE", revenue.id, operating.id, amount_percentage=100)],
                name="All revenue to operating"
            ),
            Rule(
                str(uuid.uuid4()), operating.id, "INCOMING_FUNDS",
                [
                    Action("PERCENTAGE", operating.id, profit.id, amount_percentage=5, group_index=0),
                    Action("PERCENTAGE", operating.id, owner.id, amount_percentage=50, group_index=0),
                    Action("PERCENTAGE", operating.id, tax.id, amount_percentage=15, group_index=0),
                    Action("REMAINDER", operating.id, opex.id, group_index=1),
                ],
                name="Profit First allocation"
            )
        ]

        return {
            'nodes': nodes,
            'rules': rules,
            'description': "Profit First: 5% profit, 50% owner pay, 15% taxes, 30% operating"
        }

    @staticmethod
    def bill_automation(profile: dict, extracted: dict) -> dict:
        """Generate bill automation pattern."""
        nodes = [
            Node(str(uuid.uuid4()), "PORT", "Income", icon="📥"),
            Node(str(uuid.uuid4()), "POD", "Bills Holding", icon="📋"),
            Node(str(uuid.uuid4()), "POD", "Spending", icon="💵"),
        ]

        # Add common bills
        bills = [
            ("Rent", 1500),
            ("Utilities", 200),
            ("Phone", 100),
        ]

        for name, amount in bills:
            nodes.append(Node(str(uuid.uuid4()), "POD", name, icon="📝"))

        income = nodes[0]
        bills_holding = nodes[1]
        spending = nodes[2]
        bill_pods = nodes[3:]

        # Calculate total bills
        total_bills = sum(b[1] for b in bills)

        rules = [
            Rule(
                str(uuid.uuid4()), income.id, "INCOMING_FUNDS",
                [
                    Action("FIXED", income.id, bills_holding.id, amount_cents=total_bills*100, group_index=0),
                    Action("REMAINDER", income.id, spending.id, group_index=1),
                ],
                name="Set aside bill money"
            ),
            Rule(
                str(uuid.uuid4()), bills_holding.id, "INCOMING_FUNDS",
                [Action("FIXED", bills_holding.id, pod.id, amount_cents=bills[i][1]*100, group_index=i)
                 for i, pod in enumerate(bill_pods)],
                name="Distribute to bill categories"
            )
        ]

        return {
            'nodes': nodes,
            'rules': rules,
            'description': f"Bill automation: ${total_bills}/month set aside for bills"
        }

    @staticmethod
    def tax_reserve(profile: dict, extracted: dict) -> dict:
        """Generate tax reserve pattern."""
        nodes = [
            Node(str(uuid.uuid4()), "PORT", "Income", icon="📥"),
            Node(str(uuid.uuid4()), "POD", "Tax Reserve", icon="🏛️"),
            Node(str(uuid.uuid4()), "POD", "Take Home", icon="💵"),
        ]

        # Determine tax rate based on income
        income_level = profile.get('ANNUALINCOME', 'BETWEEN_50K_AND_100K')
        tax_rates = {
            'UP_TO_10K': 10,
            'BETWEEN_10K_AND_25K': 12,
            'BETWEEN_25K_AND_50K': 15,
            'BETWEEN_50K_AND_100K': 20,
            'BETWEEN_100K_AND_250K': 25,
            'OVER_250K': 30
        }
        tax_pct = tax_rates.get(income_level, 20)

        income = nodes[0]
        tax = nodes[1]
        take_home = nodes[2]

        rules = [
            Rule(
                str(uuid.uuid4()), income.id, "INCOMING_FUNDS",
                [
                    Action("PERCENTAGE", income.id, tax.id, amount_percentage=tax_pct, group_index=0),
                    Action("REMAINDER", income.id, take_home.id, group_index=1),
                ],
                name=f"Reserve {tax_pct}% for taxes"
            )
        ]

        return {
            'nodes': nodes,
            'rules': rules,
            'description': f"Tax reserve: {tax_pct}% set aside for taxes"
        }

    @staticmethod
    def sweep_pattern(profile: dict, extracted: dict) -> dict:
        """Generate overflow/sweep pattern."""
        nodes = [
            Node(str(uuid.uuid4()), "DEPOSITORY_ACCOUNT", "Checking", subtype="CHECKING", icon="🏦"),
            Node(str(uuid.uuid4()), "DEPOSITORY_ACCOUNT", "Savings", subtype="SAVINGS", icon="💰"),
        ]

        # Determine threshold
        amounts = extracted.get('amounts', {}).get('dollars', [])
        threshold = int(amounts[0] * 100) if amounts else 500000  # Default $5,000

        checking = nodes[0]
        savings = nodes[1]

        rules = [
            Rule(
                str(uuid.uuid4()), checking.id, "BALANCE_THRESHOLD",
                [Action("REMAINDER", checking.id, savings.id)],
                condition={"operator": "greaterThan", "amountInCents": threshold},
                name=f"Sweep excess above ${threshold/100:.0f}"
            )
        ]

        return {
            'nodes': nodes,
            'rules': rules,
            'description': f"Sweep: Move excess above ${threshold/100:.0f} to savings"
        }


# ============================================================================
# LAYOUT CALCULATOR
# ============================================================================

def calculate_positions(nodes: list) -> None:
    """Calculate node positions for visual layout."""
    x_start, y_start = 200, 200
    spacing_x, spacing_y = 250, 150

    # Group by type
    ports = [n for n in nodes if n.type == 'PORT']
    pods = [n for n in nodes if n.type == 'POD']
    accounts = [n for n in nodes if n.type == 'DEPOSITORY_ACCOUNT']
    liabilities = [n for n in nodes if n.type == 'LIABILITY_ACCOUNT']

    # Position ports on left
    for i, node in enumerate(ports):
        node.position = {'x': x_start, 'y': y_start + i * spacing_y}

    # Position pods in middle columns
    for i, node in enumerate(pods):
        col = i // 4
        row = i % 4
        node.position = {
            'x': x_start + spacing_x + col * spacing_x,
            'y': y_start + row * spacing_y
        }

    # Position accounts on right
    for i, node in enumerate(accounts):
        node.position = {
            'x': x_start + spacing_x * 3,
            'y': y_start + i * spacing_y
        }

    # Position liabilities at bottom
    for i, node in enumerate(liabilities):
        node.position = {
            'x': x_start + i * spacing_x,
            'y': y_start + spacing_y * 5
        }


# ============================================================================
# PLAN GENERATOR
# ============================================================================

def generate_human_readable_plan(goal_info: dict, pattern_result: dict,
                                  similar_users: list, profile: dict) -> str:
    """Generate human-readable plan from pattern result."""

    plan = f"""## Financial Automation Plan

### Goal
{goal_info['goal'].replace('_', ' ').title()}

### Recommended Structure

**Income Sources (Ports):**
"""

    for node in pattern_result['nodes']:
        if node.type == 'PORT':
            plan += f"- {node.name}\n"

    plan += "\n**Allocation Pods:**\n"
    for node in pattern_result['nodes']:
        if node.type == 'POD':
            plan += f"- {node.name}"
            if node.target_amount:
                plan += f" (target: ${node.target_amount/100:,.0f})"
            plan += "\n"

    accounts = [n for n in pattern_result['nodes'] if n.type == 'DEPOSITORY_ACCOUNT']
    if accounts:
        plan += "\n**Bank Accounts:**\n"
        for node in accounts:
            plan += f"- {node.name} ({node.subtype})\n"

    liabilities = [n for n in pattern_result['nodes'] if n.type == 'LIABILITY_ACCOUNT']
    if liabilities:
        plan += "\n**Liabilities:**\n"
        for node in liabilities:
            plan += f"- {node.name} ({node.subtype})\n"

    plan += "\n### Automation Rules\n\n"

    for i, rule in enumerate(pattern_result['rules'], 1):
        source_name = next((n.name for n in pattern_result['nodes'] if n.id == rule.source_id), "Unknown")

        plan += f"**{i}. {rule.name or 'Rule'}**\n"
        plan += f"- Trigger: {rule.trigger_type.replace('_', ' ').title()} at {source_name}\n"
        plan += "- Actions:\n"

        for action in rule.actions:
            dest_name = next((n.name for n in pattern_result['nodes'] if n.id == action.destination_id), "Unknown")

            if action.type == 'PERCENTAGE':
                plan += f"  - Move {action.amount_percentage}% to {dest_name}\n"
            elif action.type == 'FIXED':
                plan += f"  - Move ${action.amount_cents/100:.2f} to {dest_name}\n"
            elif action.type == 'REMAINDER':
                plan += f"  - Move remaining funds to {dest_name}\n"
            elif action.type == 'TOP_UP':
                plan += f"  - Fill {dest_name} up to ${action.limit/100:.2f}\n"
            elif action.type in ['AVALANCHE', 'SNOWBALL']:
                plan += f"  - {action.type.title()} payment to {dest_name}\n"

        plan += "\n"

    plan += f"### Implementation Notes\n{pattern_result['description']}\n"

    # Add similar user insights
    if similar_users:
        users_with_goals = [u for u in similar_users[:5] if u.get('metadata', {}).get('PRODUCTGOAL')]
        if users_with_goals:
            plan += "\n### Similar Users\n"
            plan += "Based on your profile, here are similar successful users:\n"
            for u in users_with_goals[:3]:
                goal = u['metadata'].get('PRODUCTGOAL', 'N/A')
                plan += f"- User with goal: {goal.replace('_', ' ').title()}\n"

    return plan


# ============================================================================
# MAIN GENERATOR CLASS
# ============================================================================

class SequenceMapGenerator:
    """
    Main generator that transforms user prompts into Sequence maps.
    """

    def __init__(self, models_dir: str = None, use_matcher: bool = True):
        """Initialize with optional pattern matcher."""
        if models_dir is None:
            models_dir = str(Path(__file__).parent / 'models')

        self.matcher = None
        self.use_matcher = False

        if use_matcher and MATCHER_AVAILABLE and PatternMatcher is not None:
            try:
                # Import train_models first to ensure classes are available for unpickling
                import train_models  # noqa: F401
                self.matcher = PatternMatcher(models_dir=models_dir)
                self.use_matcher = True
                print("Pattern matcher loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load pattern matcher: {e}")

        # Pattern templates
        self.templates = {
            'debt_payoff': PatternTemplates.debt_payoff,
            'budget_allocation': PatternTemplates.budget_allocation,
            'savings_first': PatternTemplates.savings_first,
            'profit_first': PatternTemplates.profit_first,
            'bill_automation': PatternTemplates.bill_automation,
            'tax_reserve': PatternTemplates.tax_reserve,
            'sweep_pattern': PatternTemplates.sweep_pattern,
        }

    def generate(self, prompt: str, profile: dict = None) -> dict:
        """
        Generate a complete Sequence map from a user prompt.

        Args:
            prompt: User's description of what they want to achieve
            profile: Optional user profile dict with keys like:
                     USER_TYPE, ANNUALINCOME, OCCUPATION, etc.

        Returns:
            dict with:
            - plan: Human-readable plan text
            - json_map: Sequence-compatible JSON
            - goal: Detected goal
            - similar_users: List of similar users (if matcher available)
        """

        profile = profile or {}

        # Step 1: Detect goal from prompt
        goal_info = detect_goal(prompt)

        # Step 2: Extract amounts and account mentions
        extracted = {
            'amounts': extract_amounts(prompt),
            'accounts': extract_accounts(prompt)
        }

        # Step 3: Find similar users (if matcher available)
        similar_users = []
        if self.use_matcher and self.matcher:
            try:
                match_result = self.matcher.match_user(profile, rules_text=prompt)
                similar_users = match_result.get('similar_users', [])

                # If goal not detected well, use predicted goal
                if match_result.get('predicted_goal'):
                    profile['_predicted_goal'] = match_result['predicted_goal']
            except Exception as e:
                print(f"Warning: Pattern matching failed: {e}")

        # Step 4: Generate pattern based on goal
        pattern_name = goal_info['pattern']
        template_fn = self.templates.get(pattern_name, PatternTemplates.budget_allocation)
        pattern_result = template_fn(profile, extracted)

        # Step 5: Calculate node positions
        calculate_positions(pattern_result['nodes'])

        # Step 6: Generate human-readable plan
        plan_text = generate_human_readable_plan(
            goal_info, pattern_result, similar_users, profile
        )

        # Step 7: Convert to JSON
        json_map = {
            "nodes": [n.to_dict() for n in pattern_result['nodes']],
            "rules": [r.to_dict() for r in pattern_result['rules']],
            "viewport": {
                "x": 321.5,
                "y": -198.4,
                "zoom": 0.9
            }
        }

        return {
            'plan': plan_text,
            'json_map': json_map,
            'goal': goal_info,
            'similar_users': similar_users[:5],
            'extracted': extracted
        }

    def to_json_file(self, result: dict, filepath: str) -> None:
        """Save the JSON map to a file."""
        with open(filepath, 'w') as f:
            json.dump(result['json_map'], f, indent=2)


# ============================================================================
# CLI / DEMO
# ============================================================================

def demo():
    """Demonstrate the generator with example prompts."""

    generator = SequenceMapGenerator()

    examples = [
        {
            'prompt': "Help me pay off my Chase and Amex credit cards faster using the avalanche method",
            'profile': {
                'USER_TYPE': 'INDIVIDUAL',
                'ANNUALINCOME': 'BETWEEN_50K_AND_100K',
                'OCCUPATION': 'SCIENTIST_OR_TECHNOLOGIST'
            }
        },
        {
            'prompt': "I want to set up a 50/30/20 budget automatically",
            'profile': {
                'USER_TYPE': 'INDIVIDUAL',
                'ANNUALINCOME': 'BETWEEN_25K_AND_50K'
            }
        },
        {
            'prompt': "Set up Profit First for my small business",
            'profile': {
                'USER_TYPE': 'BUSINESS',
                'ANNUALINCOME': 'BETWEEN_100K_AND_250K'
            }
        }
    ]

    for i, example in enumerate(examples, 1):
        print(f"\n{'='*70}")
        print(f"EXAMPLE {i}")
        print(f"{'='*70}")
        print(f"Prompt: {example['prompt']}")
        print(f"Profile: {example['profile']}")

        result = generator.generate(example['prompt'], example['profile'])

        print("\n--- PLAN ---")
        print(result['plan'])

        print("\n--- JSON (first 50 lines) ---")
        json_str = json.dumps(result['json_map'], indent=2)
        lines = json_str.split('\n')[:50]
        print('\n'.join(lines))
        if len(json_str.split('\n')) > 50:
            print("... (truncated)")

        # Save to file
        output_file = f"/Users/itaydror/Map generator/output_example_{i}.json"
        generator.to_json_file(result, output_file)
        print(f"\nSaved to: {output_file}")


if __name__ == '__main__':
    demo()
