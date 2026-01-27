"""
Dynamic Sequence Map Generator
Generates maps based on ACTUAL patterns from similar users - no hardcoded templates.
Uses structured features (no embeddings).
"""

import json
import uuid
import re
import polars as pl
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from collections import defaultdict

# Import our modules
from rule_parser import RuleParser, ParsedMap, get_user_structure
import train_models  # Ensure classes available for unpickling
from inference import PatternMatcher


# ============================================================================
# JSON CONVERSION
# ============================================================================

def parsed_to_sequence_json(parsed_map: ParsedMap, parser: RuleParser) -> dict:
    """Convert a ParsedMap to Sequence-compatible JSON."""

    nodes = []
    node_id_map = {}  # name -> uuid

    # Create nodes
    for name, node_type in parsed_map.nodes.items():
        node_id = str(uuid.uuid4())
        node_id_map[name] = node_id

        # Determine subtype
        subtype = None
        if node_type == 'LIABILITY_ACCOUNT':
            subtype = parser.infer_liability_subtype(name)
        elif node_type == 'DEPOSITORY_ACCOUNT':
            subtype = parser.infer_depository_subtype(name)

        # Icon based on type
        icons = {
            'PORT': '📥',
            'POD': '💰',
            'DEPOSITORY_ACCOUNT': '🏦',
            'LIABILITY_ACCOUNT': '💳'
        }

        nodes.append({
            'id': node_id,
            'type': node_type,
            'subtype': subtype,
            'name': name,
            'balance': 0,
            'icon': icons.get(node_type, '💰'),
            'position': {'x': 0, 'y': 0}
        })

    # Calculate positions
    _calculate_positions(nodes)

    # Create rules
    rules = []
    for parsed_rule in parsed_map.rules:
        source_id = node_id_map.get(parsed_rule.trigger_source)
        if not source_id:
            continue

        # Build trigger
        trigger = {
            'type': parsed_rule.trigger_type,
            'sourceId': source_id,
            'cron': None
        }
        if parsed_rule.threshold_amount and parsed_rule.threshold_operator:
            trigger['condition'] = {
                'operator': parsed_rule.threshold_operator,
                'amountInCents': int(parsed_rule.threshold_amount * 100)
            }

        # Build actions
        actions = []
        for i, parsed_action in enumerate(parsed_rule.actions):
            action_source_id = node_id_map.get(parsed_action.source)
            action_dest_id = node_id_map.get(parsed_action.destination)

            if not action_source_id or not action_dest_id:
                continue

            # Map action types
            action_type_map = {
                'PERCENTAGE': 'PERCENTAGE',
                'FIXED': 'FIXED',
                'REMAINDER': 'REMAINDER',
                'OVERFLOW': 'TOP_UP'  # Overflow becomes TOP_UP with limit
            }
            action_type = action_type_map.get(parsed_action.action_type, parsed_action.action_type)

            action = {
                'type': action_type,
                'sourceId': action_source_id,
                'destinationId': action_dest_id,
                'amountInCents': 0,
                'amountInPercentage': 0,
                'groupIndex': i,
                'limit': None,
                'upToEnabled': None
            }

            if parsed_action.is_percentage:
                action['amountInPercentage'] = parsed_action.amount
            elif parsed_action.amount:
                if action_type == 'TOP_UP':
                    action['limit'] = int(parsed_action.amount * 100)
                    action['upToEnabled'] = True
                else:
                    action['amountInCents'] = int(parsed_action.amount * 100)

            actions.append(action)

        if actions:
            # Add step condition if present
            steps = [{'actions': actions}]
            if parsed_rule.transfer_condition:
                steps[0]['condition'] = {
                    'type': 'transferAmount',
                    **parsed_rule.transfer_condition
                }

            rules.append({
                'id': str(uuid.uuid4()),
                'sourceId': source_id,
                'trigger': trigger,
                'steps': steps
            })

    return {
        'nodes': nodes,
        'rules': rules,
        'viewport': {'x': 321.5, 'y': -198.4, 'zoom': 0.9}
    }


def _calculate_positions(nodes: list):
    """Calculate node positions for visual layout."""
    x_start, y_start = 200, 200
    spacing_x, spacing_y = 250, 150

    ports = [n for n in nodes if n['type'] == 'PORT']
    pods = [n for n in nodes if n['type'] == 'POD']
    accounts = [n for n in nodes if n['type'] == 'DEPOSITORY_ACCOUNT']
    liabilities = [n for n in nodes if n['type'] == 'LIABILITY_ACCOUNT']

    for i, node in enumerate(ports):
        node['position'] = {'x': x_start, 'y': y_start + i * spacing_y}

    for i, node in enumerate(pods):
        col = i // 5
        row = i % 5
        node['position'] = {'x': x_start + spacing_x + col * spacing_x, 'y': y_start + row * spacing_y}

    for i, node in enumerate(accounts):
        node['position'] = {'x': x_start + spacing_x * 4, 'y': y_start + i * spacing_y}

    for i, node in enumerate(liabilities):
        node['position'] = {'x': x_start + i * spacing_x, 'y': y_start + spacing_y * 6}


# ============================================================================
# TEMPLATE SYNTHESIS FROM SIMILAR USERS
# ============================================================================

def synthesize_template_from_users(similar_users: list, rules_df: pl.DataFrame,
                                    parser: RuleParser, max_users: int = 10,
                                    min_rules: int = 3, min_nodes: int = 4) -> ParsedMap:
    """
    Synthesize a template from multiple similar users' rules.

    Strategy:
    - Find similar users with sufficiently complex rule sets
    - Prefer users with more nodes and rules (more developed setups)
    - Use the best match with adequate complexity
    """

    # Collect parsed structures from similar users
    user_structures = []
    for user in similar_users[:max_users]:
        user_id = user['user_id']
        parsed = get_user_structure(rules_df, user_id, parser)
        if parsed and parsed.rules:
            # Score by complexity (more rules/nodes = better template)
            complexity = len(parsed.rules) + len(parsed.nodes) * 0.5
            user_structures.append({
                'user_id': user_id,
                'similarity': user['similarity'],
                'parsed': parsed,
                'complexity': complexity,
                'num_rules': len(parsed.rules),
                'num_nodes': len(parsed.nodes)
            })

    if not user_structures:
        return None

    # Filter to users with adequate complexity
    adequate = [u for u in user_structures
                if u['num_rules'] >= min_rules and u['num_nodes'] >= min_nodes]

    # If none meet minimum, relax requirements
    if not adequate:
        adequate = [u for u in user_structures if u['num_rules'] >= 2]

    if not adequate:
        adequate = user_structures

    # Score by similarity * complexity (balance both)
    for u in adequate:
        u['score'] = u['similarity'] * 0.6 + (u['complexity'] / 30) * 0.4

    # Sort by combined score
    adequate.sort(key=lambda x: x['score'], reverse=True)

    best_match = adequate[0]
    print(f"Selected template from user {best_match['user_id'][:8]}... "
          f"(sim={best_match['similarity']:.2f}, rules={best_match['num_rules']}, nodes={best_match['num_nodes']})")

    return best_match['parsed']


def adapt_template_to_user(template: ParsedMap, user_profile: dict,
                           user_prompt: str, parser: RuleParser) -> ParsedMap:
    """
    Adapt a template to a specific user's context.

    - Rename nodes based on user's mentioned accounts
    - Adjust amounts based on income level
    - Keep the rule structure but personalize
    """

    # Extract user's mentioned accounts from prompt
    mentioned_accounts = _extract_mentioned_accounts(user_prompt)

    # Create adapted map
    adapted_nodes = {}
    node_remap = {}  # old_name -> new_name

    for name, node_type in template.nodes.items():
        new_name = name

        # Try to match with user's mentioned accounts
        if node_type == 'LIABILITY_ACCOUNT' and mentioned_accounts['credit_cards']:
            # Replace generic card names with user's cards
            if any(kw in name.lower() for kw in ['card', 'credit', 'amex', 'chase', 'citi']):
                if mentioned_accounts['credit_cards']:
                    new_name = mentioned_accounts['credit_cards'].pop(0)

        node_remap[name] = new_name
        adapted_nodes[new_name] = node_type

    # Adapt rules with new node names
    adapted_rules = []
    for rule in template.rules:
        new_source = node_remap.get(rule.trigger_source, rule.trigger_source)

        new_actions = []
        for action in rule.actions:
            new_action_source = node_remap.get(action.source, action.source)
            new_action_dest = node_remap.get(action.destination, action.destination)

            # Adjust amounts based on income (optional)
            adjusted_amount = action.amount
            if action.amount and not action.is_percentage:
                adjusted_amount = _adjust_amount_for_income(action.amount, user_profile)

            from rule_parser import ParsedAction
            new_actions.append(ParsedAction(
                action_type=action.action_type,
                source=new_action_source,
                destination=new_action_dest,
                amount=adjusted_amount,
                is_percentage=action.is_percentage
            ))

        from rule_parser import ParsedRule
        adapted_rules.append(ParsedRule(
            trigger_type=rule.trigger_type,
            trigger_source=new_source,
            actions=new_actions,
            threshold_amount=rule.threshold_amount,
            threshold_operator=rule.threshold_operator,
            transfer_condition=rule.transfer_condition
        ))

    return ParsedMap(nodes=adapted_nodes, rules=adapted_rules)


def _extract_mentioned_accounts(prompt: str) -> dict:
    """Extract mentioned account names from user prompt."""
    accounts = {
        'credit_cards': [],
        'banks': [],
        'income_sources': []
    }

    prompt_lower = prompt.lower()

    # Credit cards
    cc_patterns = [
        (r'chase\s*(?:sapphire|freedom|slate)?', 'Chase'),
        (r'amex\s*(?:gold|platinum|blue)?', 'Amex'),
        (r'american express', 'American Express'),
        (r'discover\s*(?:it)?', 'Discover'),
        (r'capital one', 'Capital One'),
        (r'citi\s*(?:card)?', 'Citi'),
        (r'bank of america', 'Bank of America'),
    ]

    for pattern, name in cc_patterns:
        if re.search(pattern, prompt_lower):
            accounts['credit_cards'].append(name)

    return accounts


def _adjust_amount_for_income(amount: float, profile: dict) -> float:
    """Adjust dollar amounts based on user's income level."""
    income = profile.get('ANNUALINCOME', 'BETWEEN_50K_AND_100K')

    multipliers = {
        'UP_TO_10K': 0.3,
        'BETWEEN_10K_AND_25K': 0.5,
        'BETWEEN_25K_AND_50K': 0.7,
        'BETWEEN_50K_AND_100K': 1.0,
        'BETWEEN_100K_AND_250K': 1.5,
        'OVER_250K': 2.0
    }

    multiplier = multipliers.get(income, 1.0)
    return round(amount * multiplier, 2)


# ============================================================================
# HUMAN READABLE PLAN
# ============================================================================

def generate_plan_text(parsed_map: ParsedMap, goal: str, similar_users: list) -> str:
    """Generate human-readable plan from parsed map."""

    plan = f"""## Financial Automation Plan

### Goal
{goal.replace('_', ' ').title()}

### Structure (based on {len(similar_users)} similar successful users)

**Nodes:**
"""

    # Group nodes by type
    by_type = defaultdict(list)
    for name, node_type in parsed_map.nodes.items():
        by_type[node_type].append(name)

    type_labels = {
        'PORT': 'Income Sources',
        'POD': 'Allocation Pods',
        'DEPOSITORY_ACCOUNT': 'Bank Accounts',
        'LIABILITY_ACCOUNT': 'Liabilities'
    }

    for node_type, label in type_labels.items():
        if by_type[node_type]:
            plan += f"\n**{label}:**\n"
            for name in by_type[node_type]:
                plan += f"- {name}\n"

    plan += "\n### Automation Rules\n"

    for i, rule in enumerate(parsed_map.rules, 1):
        plan += f"\n**Rule {i}:** When "

        if rule.trigger_type == 'INCOMING_FUNDS':
            plan += f"funds arrive at {rule.trigger_source}"
        elif rule.trigger_type == 'BALANCE_THRESHOLD':
            plan += f"{rule.trigger_source} balance is {rule.threshold_operator} ${rule.threshold_amount:.0f}"

        plan += "\n"

        for action in rule.actions:
            if action.is_percentage:
                plan += f"  - Move {action.amount:.0f}% to {action.destination}\n"
            elif action.action_type == 'FIXED':
                plan += f"  - Move ${action.amount:.2f} to {action.destination}\n"
            elif action.action_type == 'OVERFLOW':
                plan += f"  - Move anything above ${action.amount:.2f} to {action.destination}\n"
            elif action.action_type == 'REMAINDER':
                plan += f"  - Move remaining funds to {action.destination}\n"

    # Add similar user context
    if similar_users:
        plan += "\n### Based On\n"
        plan += f"Patterns from {len(similar_users)} similar users:\n"
        for u in similar_users[:3]:
            goal = u.get('metadata', {}).get('PRODUCTGOAL') or 'Unknown'
            plan += f"- User with goal: {goal.replace('_', ' ').title()} (similarity: {u['similarity']:.2f})\n"

    return plan


# ============================================================================
# MAIN GENERATOR CLASS
# ============================================================================

class DynamicSequenceGenerator:
    """
    Generate Sequence maps using actual patterns from similar users.
    No hardcoded templates - learns from real user data.
    """

    def __init__(self, models_dir: str = None, data_dir: str = None):
        if models_dir is None:
            models_dir = str(Path(__file__).parent / 'models')
        if data_dir is None:
            data_dir = str(Path(__file__).parent)

        self.parser = RuleParser()

        # Load pattern matcher
        self.matcher = PatternMatcher(models_dir=models_dir)

        # Load rules data
        self.rules_df = pl.read_csv(Path(data_dir) / 'Itaytestfinal.csv')
        print(f"Loaded {len(self.rules_df)} user rules")

    def generate(self, prompt: str, profile: dict = None) -> dict:
        """
        Generate a Sequence map from user prompt.

        Uses ML to find similar users, then uses their actual rules as template.
        """
        profile = profile or {}

        # Step 1: Find similar users using ML
        match_result = self.matcher.match_user(profile, rules_text=prompt)
        similar_users = match_result.get('similar_users', [])
        predicted_goal = match_result.get('predicted_goal', 'AUTOMATE_MY_BUDGETING')

        if not similar_users:
            return {
                'error': 'No similar users found',
                'plan': None,
                'json_map': None
            }

        # Step 2: Get template from most similar user
        template = synthesize_template_from_users(
            similar_users, self.rules_df, self.parser, max_users=3
        )

        if not template:
            return {
                'error': 'Could not extract patterns from similar users',
                'plan': None,
                'json_map': None
            }

        # Step 3: Adapt template to this user
        adapted = adapt_template_to_user(template, profile, prompt, self.parser)

        # Step 4: Generate outputs
        plan_text = generate_plan_text(adapted, predicted_goal, similar_users)
        json_map = parsed_to_sequence_json(adapted, self.parser)

        return {
            'plan': plan_text,
            'json_map': json_map,
            'predicted_goal': predicted_goal,
            'similar_users': similar_users[:5],
            'template_source': similar_users[0]['user_id'] if similar_users else None
        }

    def generate_from_user(self, user_id: str) -> dict:
        """
        Generate JSON from an existing user's rules (for testing/validation).
        """
        parsed = get_user_structure(self.rules_df, user_id, self.parser)
        if not parsed:
            return {'error': f'User {user_id} not found'}

        json_map = parsed_to_sequence_json(parsed, self.parser)
        plan_text = generate_plan_text(parsed, 'EXISTING_USER', [])

        return {
            'plan': plan_text,
            'json_map': json_map,
            'user_id': user_id
        }


# ============================================================================
# CLI / DEMO
# ============================================================================

def demo():
    print("Initializing Dynamic Sequence Generator...")
    generator = DynamicSequenceGenerator()

    print("\n" + "="*70)
    print("TEST 1: Generate from user prompt")
    print("="*70)

    result = generator.generate(
        prompt="I want to pay off my Chase and Amex credit cards aggressively",
        profile={
            'USER_TYPE': 'INDIVIDUAL',
            'ANNUALINCOME': 'BETWEEN_50K_AND_100K'
        }
    )

    if result.get('error'):
        print(f"Error: {result['error']}")
    else:
        print("\n--- PLAN ---")
        print(result['plan'])

        print("\n--- JSON STRUCTURE ---")
        json_map = result['json_map']
        print(f"Nodes: {len(json_map['nodes'])}")
        for node in json_map['nodes'][:10]:
            print(f"  - {node['name']} ({node['type']})")
        if len(json_map['nodes']) > 10:
            print(f"  ... and {len(json_map['nodes'])-10} more")

        print(f"Rules: {len(json_map['rules'])}")

        # Save output
        with open('/Users/itaydror/Map generator/dynamic_output.json', 'w') as f:
            json.dump(json_map, f, indent=2)
        print("\nSaved to dynamic_output.json")

    print("\n" + "="*70)
    print("TEST 2: Parse existing user's rules")
    print("="*70)

    # Get a sample user
    sample_user = generator.rules_df.row(0, named=True)['organization_id']
    result2 = generator.generate_from_user(sample_user)

    if result2.get('error'):
        print(f"Error: {result2['error']}")
    else:
        print(f"\nParsed user: {sample_user[:8]}...")
        print(f"Nodes extracted: {len(result2['json_map']['nodes'])}")
        print(f"Rules extracted: {len(result2['json_map']['rules'])}")


if __name__ == '__main__':
    demo()
