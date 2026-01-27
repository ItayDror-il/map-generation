"""
Rule Parser - Extract structured patterns from human-readable rule descriptions.
No embeddings - pure regex/pattern matching.
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict


@dataclass
class ParsedAction:
    """A single parsed action from a rule."""
    action_type: str  # PERCENTAGE, FIXED, REMAINDER, OVERFLOW
    source: str
    destination: str
    amount: Optional[float] = None  # Dollar amount or percentage
    is_percentage: bool = False

    def to_dict(self):
        return {
            'action_type': self.action_type,
            'source': self.source,
            'destination': self.destination,
            'amount': self.amount,
            'is_percentage': self.is_percentage
        }


@dataclass
class ParsedRule:
    """A single parsed rule with trigger and actions."""
    trigger_type: str  # INCOMING_FUNDS, BALANCE_THRESHOLD, SCHEDULED
    trigger_source: str
    actions: list
    threshold_amount: Optional[float] = None
    threshold_operator: Optional[str] = None  # greaterThan, lessThan
    transfer_condition: Optional[dict] = None  # For conditional rules

    def to_dict(self):
        return {
            'trigger_type': self.trigger_type,
            'trigger_source': self.trigger_source,
            'threshold_amount': self.threshold_amount,
            'threshold_operator': self.threshold_operator,
            'transfer_condition': self.transfer_condition,
            'actions': [a.to_dict() for a in self.actions]
        }


@dataclass
class ParsedMap:
    """Complete parsed map structure from rules."""
    nodes: dict = field(default_factory=dict)  # name -> inferred type
    rules: list = field(default_factory=list)

    def to_dict(self):
        return {
            'nodes': self.nodes,
            'rules': [r.to_dict() for r in self.rules]
        }


class RuleParser:
    """Parse human-readable rule descriptions into structured data."""

    # Patterns for different rule components
    TRIGGER_PATTERNS = {
        'incoming_funds': r'When funds are received:',
        'balance_threshold': r'When ([^:]+?) balance is at least \$([\d,]+(?:\.\d{2})?):',
        'balance_less_than': r'When balance lessThan (\d+):',
        'transfer_condition': r'When transferAmount greaterThan (\d+):',
    }

    ACTION_PATTERNS = {
        # "100.00% of incoming funds goes from X to Y"
        'percentage': r'([\d.]+)% of incoming funds goes from ([^;.]+?) to ([^;.\n]+)',
        # "$500.00 moves from X to Y"
        'fixed': r'\$([\d,]+(?:\.\d{2})?) moves from ([^;.]+?) to ([^;.\n]+)',
        # "Anything above $500.00 moves from X to Y"
        'overflow': r'Anything above \$([\d,]+(?:\.\d{2})?) moves from ([^;.]+?) to ([^;.\n]+)',
        # "Funds move from X to Y" (remainder)
        'remainder': r'Funds move from ([^;.]+?) to ([^;.\n]+)',
    }

    # Keywords that suggest node types
    NODE_TYPE_HINTS = {
        'PORT': ['deposit', 'income', 'paycheck', 'salary', 'doordash', 'uber',
                 'square deposits', 'stripe', 'paypal', 'venmo incoming'],
        'LIABILITY_ACCOUNT': ['credit card', 'card', 'loan', 'debt', 'cc ',
                              'amex', 'chase sapphire', 'discover', 'citi'],
        'DEPOSITORY_ACCOUNT': ['checking', 'savings', 'bank', 'account'],
        'POD': []  # Default
    }

    def __init__(self):
        pass

    def parse_description(self, description: str) -> ParsedMap:
        """Parse a full rule description into structured data."""
        parsed_map = ParsedMap()

        # Split into individual rule blocks
        rule_blocks = self._split_into_rules(description)

        for block in rule_blocks:
            parsed_rule = self._parse_rule_block(block)
            if parsed_rule:
                parsed_map.rules.append(parsed_rule)

                # Collect nodes from rule
                self._collect_nodes(parsed_rule, parsed_map.nodes)

        return parsed_map

    def _split_into_rules(self, description: str) -> list:
        """Split description into individual rule blocks."""
        # Split on "When" but keep the delimiter
        blocks = re.split(r'(?=When )', description)
        return [b.strip() for b in blocks if b.strip() and b.strip().startswith('When')]

    def _parse_rule_block(self, block: str) -> Optional[ParsedRule]:
        """Parse a single rule block."""

        # Determine trigger type
        trigger_type = 'INCOMING_FUNDS'
        trigger_source = None
        threshold_amount = None
        threshold_operator = None
        transfer_condition = None

        # Check for balance threshold trigger
        balance_match = re.search(self.TRIGGER_PATTERNS['balance_threshold'], block)
        if balance_match:
            trigger_type = 'BALANCE_THRESHOLD'
            trigger_source = balance_match.group(1).strip()
            threshold_amount = self._parse_amount(balance_match.group(2))
            threshold_operator = 'greaterThan'

        # Check for balance less than
        less_than_match = re.search(self.TRIGGER_PATTERNS['balance_less_than'], block)
        if less_than_match:
            trigger_type = 'BALANCE_THRESHOLD'
            threshold_amount = float(less_than_match.group(1))
            threshold_operator = 'lessThan'

        # Check for transfer condition
        transfer_match = re.search(self.TRIGGER_PATTERNS['transfer_condition'], block)
        if transfer_match:
            transfer_condition = {
                'operator': 'greaterThan',
                'amountInCents': int(float(transfer_match.group(1)))
            }

        # Parse actions
        actions = self._parse_actions(block)

        if not actions:
            return None

        # Infer trigger source from first action if not set
        if trigger_source is None and actions:
            trigger_source = actions[0].source

        return ParsedRule(
            trigger_type=trigger_type,
            trigger_source=trigger_source,
            actions=actions,
            threshold_amount=threshold_amount,
            threshold_operator=threshold_operator,
            transfer_condition=transfer_condition
        )

    def _parse_actions(self, block: str) -> list:
        """Parse all actions from a rule block."""
        actions = []

        # Find percentage actions
        for match in re.finditer(self.ACTION_PATTERNS['percentage'], block):
            pct = float(match.group(1))
            source = match.group(2).strip()
            dest = match.group(3).strip()
            actions.append(ParsedAction(
                action_type='PERCENTAGE',
                source=source,
                destination=dest,
                amount=pct,
                is_percentage=True
            ))

        # Find fixed amount actions (but not overflow which also has $ amount)
        for match in re.finditer(self.ACTION_PATTERNS['fixed'], block):
            # Check this isn't part of an overflow pattern
            match_start = match.start()
            preceding_text = block[max(0, match_start-20):match_start]
            if 'above' in preceding_text.lower():
                continue  # Skip, this is an overflow action

            amount = self._parse_amount(match.group(1))
            source = match.group(2).strip()
            dest = match.group(3).strip()
            actions.append(ParsedAction(
                action_type='FIXED',
                source=source,
                destination=dest,
                amount=amount,
                is_percentage=False
            ))

        # Find overflow actions
        for match in re.finditer(self.ACTION_PATTERNS['overflow'], block):
            threshold = self._parse_amount(match.group(1))
            source = match.group(2).strip()
            dest = match.group(3).strip()
            actions.append(ParsedAction(
                action_type='OVERFLOW',
                source=source,
                destination=dest,
                amount=threshold,
                is_percentage=False
            ))

        # Find remainder actions (only if no other actions matched this pattern)
        for match in re.finditer(self.ACTION_PATTERNS['remainder'], block):
            source = match.group(1).strip()
            dest = match.group(2).strip()

            # Check if this wasn't already captured by another pattern
            already_captured = any(
                a.source == source and a.destination == dest
                for a in actions
            )
            if not already_captured:
                actions.append(ParsedAction(
                    action_type='REMAINDER',
                    source=source,
                    destination=dest
                ))

        return actions

    def _parse_amount(self, amount_str: str) -> float:
        """Parse dollar amount string to float."""
        return float(amount_str.replace(',', ''))

    def _collect_nodes(self, rule: ParsedRule, nodes: dict):
        """Collect unique nodes from a rule and infer their types."""

        # Add trigger source
        if rule.trigger_source and rule.trigger_source not in nodes:
            nodes[rule.trigger_source] = self._infer_node_type(rule.trigger_source)

        # Add action sources and destinations
        for action in rule.actions:
            if action.source not in nodes:
                nodes[action.source] = self._infer_node_type(action.source)
            if action.destination not in nodes:
                nodes[action.destination] = self._infer_node_type(action.destination)

    def _infer_node_type(self, name: str) -> str:
        """Infer node type from name using keyword hints."""
        name_lower = name.lower()

        for node_type, keywords in self.NODE_TYPE_HINTS.items():
            if any(kw in name_lower for kw in keywords):
                return node_type

        # Default to POD
        return 'POD'

    def infer_liability_subtype(self, name: str) -> Optional[str]:
        """Infer liability subtype from name."""
        name_lower = name.lower()

        if any(kw in name_lower for kw in ['credit card', 'card', 'cc ', 'amex', 'visa', 'mastercard', 'discover']):
            return 'CREDIT_CARD'
        elif any(kw in name_lower for kw in ['loan', 'mortgage', 'auto loan', 'student loan']):
            return 'LOAN'

        return 'CREDIT_CARD'  # Default for liabilities

    def infer_depository_subtype(self, name: str) -> Optional[str]:
        """Infer depository account subtype from name."""
        name_lower = name.lower()

        if 'saving' in name_lower:
            return 'SAVINGS'
        return 'CHECKING'


def extract_patterns_from_users(rules_df, user_ids: list, parser: RuleParser = None) -> dict:
    """
    Extract patterns from multiple users' rules.

    Returns aggregated pattern statistics.
    """
    if parser is None:
        parser = RuleParser()

    all_patterns = {
        'node_types': defaultdict(int),
        'trigger_types': defaultdict(int),
        'action_types': defaultdict(int),
        'common_nodes': defaultdict(int),
        'avg_rules_per_user': 0,
        'avg_nodes_per_user': 0,
        'sample_structures': []
    }

    total_rules = 0
    total_nodes = 0

    for user_id in user_ids:
        # Get user's rules
        user_rows = rules_df.filter(rules_df['organization_id'] == user_id)
        if len(user_rows) == 0:
            continue

        description = user_rows.row(0, named=True).get('description', '')
        if not description:
            continue

        # Parse rules
        parsed = parser.parse_description(description)

        total_rules += len(parsed.rules)
        total_nodes += len(parsed.nodes)

        # Aggregate statistics
        for node_name, node_type in parsed.nodes.items():
            all_patterns['node_types'][node_type] += 1
            all_patterns['common_nodes'][node_name] += 1

        for rule in parsed.rules:
            all_patterns['trigger_types'][rule.trigger_type] += 1
            for action in rule.actions:
                all_patterns['action_types'][action.action_type] += 1

        # Store sample structure (first 3)
        if len(all_patterns['sample_structures']) < 3:
            all_patterns['sample_structures'].append(parsed.to_dict())

    if user_ids:
        all_patterns['avg_rules_per_user'] = total_rules / len(user_ids)
        all_patterns['avg_nodes_per_user'] = total_nodes / len(user_ids)

    return all_patterns


def get_user_structure(rules_df, user_id: str, parser: RuleParser = None) -> Optional[ParsedMap]:
    """Get the parsed structure for a specific user."""
    if parser is None:
        parser = RuleParser()

    user_rows = rules_df.filter(rules_df['organization_id'] == user_id)
    if len(user_rows) == 0:
        return None

    description = user_rows.row(0, named=True).get('description', '')
    if not description:
        return None

    return parser.parse_description(description)


# Demo/test
if __name__ == '__main__':
    parser = RuleParser()

    # Test with sample rules
    test_rules = """When funds are received: 100.00% of incoming funds goes from eBay Pod to Sequence Router Pod.

When funds are received: Anything above $500.00 moves from Sequence Router Pod to Mercury Personal Inbox.

When funds are received: 100.00% of incoming funds goes from Cash App Pod to Sequence Router Pod.

When Chase Pod balance is at least $1,500.00: Funds move from Sapphire Checking to Sequence Router Pod.

When funds are received: $125.00 moves from Needs to 13th- SDGE- $150.

When funds are received: Funds move from Needs to 27TH - AMEX Gold Card."""

    parsed = parser.parse_description(test_rules)

    print("Parsed Nodes:")
    for name, node_type in parsed.nodes.items():
        print(f"  {name}: {node_type}")

    print("\nParsed Rules:")
    for i, rule in enumerate(parsed.rules, 1):
        print(f"\n  Rule {i}:")
        print(f"    Trigger: {rule.trigger_type} at {rule.trigger_source}")
        if rule.threshold_amount:
            print(f"    Threshold: {rule.threshold_operator} ${rule.threshold_amount}")
        for action in rule.actions:
            if action.is_percentage:
                print(f"    Action: {action.amount}% from {action.source} to {action.destination}")
            elif action.amount:
                print(f"    Action: ${action.amount} from {action.source} to {action.destination}")
            else:
                print(f"    Action: {action.action_type} from {action.source} to {action.destination}")
