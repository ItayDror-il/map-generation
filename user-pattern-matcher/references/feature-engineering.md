# Feature Engineering Guide

## Profile Features

### Income Encoding
```python
INCOME_MAP = {
    'UP_TO_10K': 0,
    'BETWEEN_10K_AND_25K': 1,
    'BETWEEN_25K_AND_50K': 2,
    'BETWEEN_50K_AND_100K': 3,
    'BETWEEN_100K_AND_250K': 4,
    'OVER_250K': 5
}
```

### Goal Encoding
```python
GOALS = [
    'DEBT_PAYMENT',
    'PAY_OFF_DEBT',
    'AUTOMATE_MY_BUDGETING',
    'ORGANIZE_FINANCING',
    'PROFIT_FIRST',
    'MAXIMIZE_SAVINGS',
    'MAXIMIZE_MY_SAVINGS',
    'VISUALIZE_AND_TRACK_FINANCES',
    'OPTIMIZE_CASH_FLOW',
    'SAVE_FOR_TAXES',
    'AUTOMATE_MY_BILLS',
    'AUTOMATE_MY_INVESTMENTS',
    'MAINTAIN_CONTROL',
    'MANAGE_BUSINESS_PAYMENTS'
]

# Group similar goals
GOAL_GROUPS = {
    'debt': ['DEBT_PAYMENT', 'PAY_OFF_DEBT'],
    'budget': ['AUTOMATE_MY_BUDGETING', 'ORGANIZE_FINANCING'],
    'savings': ['MAXIMIZE_SAVINGS', 'MAXIMIZE_MY_SAVINGS', 'SAVE_FOR_TAXES'],
    'business': ['PROFIT_FIRST', 'MANAGE_BUSINESS_PAYMENTS', 'OPTIMIZE_CASH_FLOW'],
    'tracking': ['VISUALIZE_AND_TRACK_FINANCES', 'MAINTAIN_CONTROL'],
    'automation': ['AUTOMATE_MY_BILLS', 'AUTOMATE_MY_INVESTMENTS']
}
```

### Occupation Encoding
```python
OCCUPATIONS = [
    'EXECUTIVE_OR_MANAGER',
    'ARCHITECT_OR_ENGINEER',
    'SCIENTIST_OR_TECHNOLOGIST',
    'DOCTOR',
    'SALES_REPRESENTATIVE_BROKER_AGENT',
    'BUSINESS_ANALYST_ACCOUNTANT_OR_FINANCIAL_ADVISOR',
    'ENTERTAINMENT_SPORTS_ARTS_OR_MEDIA',
    'EDUCATOR',
    'PILOT_DRIVER_OPERATOR',
    'CONSTRUCTION_MECHANIC_OR_MAINTENANCE_WORKER',
    'MANUFACTURING_OR_PRODUCTION_WORKER',
    'PERSONAL_CARE_OR_SERVICE_WORKER',
    'HOSPITALITY_OFFICE_OR_ADMINISTRATIVE_SUPPORT_WORKER',
    'FOOD_SERVICE_WORKER',
    'MILITARY_OR_PUBLIC_SAFETY',
    'GIG_WORKER',
    'STUDENT'
]

# Group by income stability/type
OCCUPATION_GROUPS = {
    'high_income_stable': ['EXECUTIVE_OR_MANAGER', 'DOCTOR', 'ARCHITECT_OR_ENGINEER'],
    'professional': ['SCIENTIST_OR_TECHNOLOGIST', 'BUSINESS_ANALYST_ACCOUNTANT_OR_FINANCIAL_ADVISOR', 'EDUCATOR'],
    'variable_income': ['SALES_REPRESENTATIVE_BROKER_AGENT', 'GIG_WORKER', 'ENTERTAINMENT_SPORTS_ARTS_OR_MEDIA'],
    'trade': ['CONSTRUCTION_MECHANIC_OR_MAINTENANCE_WORKER', 'MANUFACTURING_OR_PRODUCTION_WORKER'],
    'service': ['PERSONAL_CARE_OR_SERVICE_WORKER', 'HOSPITALITY_OFFICE_OR_ADMINISTRATIVE_SUPPORT_WORKER', 'FOOD_SERVICE_WORKER']
}
```

### Age Group Encoding
```python
AGE_MAP = {
    '18-24': 0,
    '25-35': 1,
    '36-44': 2,
    '45-54': 3,
    '55-64': 4,
    '65+': 5
}
```

## Rule Pattern Features

### Extracting Patterns from Rule Text

```python
import re

def extract_rule_features(rule_text):
    features = {
        # Trigger types
        'has_incoming_funds': 'When funds are received' in rule_text,
        'has_balance_threshold': 'balance is at least' in rule_text.lower(),
        'has_scheduled': bool(re.search(r'When.*\d{1,2}(st|nd|rd|th)', rule_text)),

        # Action types
        'has_percentage': '%' in rule_text,
        'has_fixed_amount': bool(re.search(r'\$[\d,]+\.\d{2} moves', rule_text)),
        'has_overflow': 'Anything above' in rule_text,
        'has_remainder': 'Funds move from' in rule_text and '%' not in rule_text,

        # Complexity
        'rule_count': rule_text.count('When '),
        'unique_destinations': len(set(re.findall(r'to ([^.;]+)', rule_text))),
        'unique_sources': len(set(re.findall(r'from ([^.;]+)', rule_text))),

        # Debt indicators
        'has_debt_keywords': any(kw in rule_text.lower() for kw in ['credit', 'loan', 'debt', 'card']),
        'liability_count': len(re.findall(r'(credit|card|loan)', rule_text.lower())),

        # Business indicators
        'has_business_keywords': any(kw in rule_text.lower() for kw in ['opex', 'payroll', 'tax', 'profit', 'operating']),
        'has_entity_prefix': bool(re.search(r'\([A-Z]+\)', rule_text))
    }
    return features
```

### Pattern Classification

```python
def classify_pattern(features):
    """Classify user's primary automation pattern"""

    if features['has_debt_keywords'] and features['liability_count'] >= 2:
        return 'debt_payoff'

    if features['has_business_keywords'] or features['has_entity_prefix']:
        return 'business_ops'

    if features['has_percentage'] and features['unique_destinations'] >= 3:
        return 'budget_allocation'

    if features['has_overflow'] or features['has_balance_threshold']:
        return 'savings_overflow'

    if features['has_scheduled']:
        return 'bill_automation'

    return 'basic_routing'
```

## Combined Feature Vector

```python
def create_feature_vector(profile, rules_text):
    """Create complete feature vector for a user"""

    # Profile features
    profile_features = [
        INCOME_MAP.get(profile['ANNUALINCOME'], 2),  # Default to middle
        1 if profile['USER_TYPE'] == 'BUSINESS' else 0,
        1 if profile['DEBIT_CARD_SPENDER'] == 'true' else 0,
        min(profile['ACTIVATED_PORTS'] / 10, 1),  # Normalize
        min(profile['ACCOUNTS_CONNECTED'] / 100, 1),  # Normalize
        AGE_MAP.get(profile['AGE_GROUP'], 2)
    ]

    # Goal one-hot (simplified to groups)
    goal = profile.get('PRODUCTGOAL', '')
    goal_features = [
        1 if goal in GOAL_GROUPS['debt'] else 0,
        1 if goal in GOAL_GROUPS['budget'] else 0,
        1 if goal in GOAL_GROUPS['savings'] else 0,
        1 if goal in GOAL_GROUPS['business'] else 0,
    ]

    # Rule pattern features
    rule_features = extract_rule_features(rules_text)
    pattern_features = [
        rule_features['has_percentage'],
        rule_features['has_fixed_amount'],
        rule_features['has_debt_keywords'],
        rule_features['has_business_keywords'],
        min(rule_features['rule_count'] / 20, 1),  # Normalize
        min(rule_features['unique_destinations'] / 15, 1)
    ]

    return profile_features + goal_features + [int(f) for f in pattern_features]
```

## Similarity Calculation

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def find_similar_users(target_vector, all_vectors, user_ids, top_k=10):
    """Find most similar users by cosine similarity"""

    target = np.array(target_vector).reshape(1, -1)
    all_vecs = np.array(all_vectors)

    similarities = cosine_similarity(target, all_vecs)[0]

    top_indices = np.argsort(similarities)[::-1][:top_k]

    return [
        {'user_id': user_ids[i], 'similarity': similarities[i]}
        for i in top_indices
    ]
```

## Clustering Users

```python
from sklearn.cluster import KMeans

def cluster_users(feature_vectors, n_clusters=8):
    """Cluster users into archetypes"""

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(feature_vectors)

    return clusters, kmeans

# Suggested cluster interpretation
CLUSTER_NAMES = {
    0: 'debt_focused_individual',
    1: 'budget_optimizer',
    2: 'small_business_owner',
    3: 'savings_builder',
    4: 'high_income_complex',
    5: 'gig_worker_variable',
    6: 'bill_automator',
    7: 'passive_tracker'
}
```

## Handling Missing Data

```python
def impute_missing(profile):
    """Handle missing profile data"""

    defaults = {
        'ANNUALINCOME': 'BETWEEN_50K_AND_100K',
        'OCCUPATION': None,  # Don't impute, use as null feature
        'PRODUCTGOAL': None,  # Infer from rules if possible
        'USER_TYPE': 'INDIVIDUAL',
        'DEBIT_CARD_SPENDER': 'false',
        'ACTIVATED_PORTS': 1,
        'ACCOUNTS_CONNECTED': 5,
        'AGE_GROUP': '25-35'
    }

    for key, default in defaults.items():
        if not profile.get(key):
            profile[key] = default

    return profile
```
