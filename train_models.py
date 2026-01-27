"""
Sequence Pattern Matcher - ML Training Pipeline
Trains similarity matching, goal classification, and user clustering models.
"""

import polars as pl
import numpy as np
import re
import json
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, silhouette_score
from sklearn.cluster import KMeans
import lightgbm as lgb
import joblib

# ============================================================================
# FEATURE ENCODINGS
# ============================================================================

INCOME_ORDINAL = {
    'UP_TO_10K': 0, 'BETWEEN_10K_AND_25K': 1, 'BETWEEN_25K_AND_50K': 2,
    'BETWEEN_50K_AND_100K': 3, 'BETWEEN_100K_AND_250K': 4, 'OVER_250K': 5
}

AGE_ORDINAL = {
    '18-24': 0, '25-35': 1, '36-44': 2, '45-54': 3, '55-64': 4, '65+': 5
}

USER_TYPE_MAP = {'INDIVIDUAL': 0, 'BUSINESS': 1, 'ADVISOR': 2}

SUBSCRIPTION_MAP = {'Starter': 0, 'Pro': 1, 'Business': 2, 'Growth': 3}

# Occupation groupings for derived feature
OCCUPATION_GROUPS = {
    'high_income_stable': ['EXECUTIVE_OR_MANAGER', 'DOCTOR', 'ARCHITECT_OR_ENGINEER'],
    'professional': ['SCIENTIST_OR_TECHNOLOGIST', 'BUSINESS_ANALYST_ACCOUNTANT_OR_FINANCIAL_ADVISOR', 'EDUCATOR'],
    'variable_income': ['SALES_REPRESENTATIVE_BROKER_AGENT', 'GIG_WORKER', 'ENTERTAINMENT_SPORTS_ARTS_OR_MEDIA'],
    'trade': ['CONSTRUCTION_MECHANIC_OR_MAINTENANCE_WORKER', 'MANUFACTURING_OR_PRODUCTION_WORKER'],
    'service': ['PERSONAL_CARE_OR_SERVICE_WORKER', 'HOSPITALITY_OFFICE_OR_ADMINISTRATIVE_SUPPORT_WORKER', 'FOOD_SERVICE_WORKER']
}

GOAL_GROUPS = {
    'debt': ['DEBT_PAYMENT', 'PAY_OFF_DEBT'],
    'budget': ['AUTOMATE_MY_BUDGETING', 'ORGANIZE_FINANCING'],
    'savings': ['MAXIMIZE_SAVINGS', 'MAXIMIZE_MY_SAVINGS', 'SAVE_FOR_TAXES'],
    'business': ['PROFIT_FIRST', 'OPTIMIZE_CASH_FLOW', 'MANAGE_BUSINESS_PAYMENTS'],
    'tracking': ['VISUALIZE_AND_TRACK_FINANCES', 'MAINTAIN_CONTROL'],
    'automation': ['AUTOMATE_MY_BILLS', 'AUTOMATE_MY_INVESTMENTS']
}

# Valid goal categories (filter out freeform "OTHER-" entries)
VALID_GOALS = [
    'DEBT_PAYMENT', 'PAY_OFF_DEBT',
    'AUTOMATE_MY_BUDGETING', 'ORGANIZE_FINANCING',
    'MAXIMIZE_SAVINGS', 'MAXIMIZE_MY_SAVINGS', 'SAVE_FOR_TAXES',
    'PROFIT_FIRST', 'OPTIMIZE_CASH_FLOW', 'MANAGE_BUSINESS_PAYMENTS',
    'VISUALIZE_AND_TRACK_FINANCES', 'MAINTAIN_CONTROL',
    'AUTOMATE_MY_BILLS', 'AUTOMATE_MY_INVESTMENTS',
    'MAXIMIZE_BANK_REWARDS', 'STOP_PAYING_LATE_FEES'
]

# Feature names in order (for weight application and interpretability)
PROFILE_FEATURE_NAMES = [
    'income_level', 'user_type', 'is_business', 'debit_card_spender',
    'activated_ports_norm', 'accounts_connected_norm', 'age_level',
    'subscription_level', 'complexity_score', 'occupation_stability'
]

RULE_FEATURE_NAMES = [
    'incoming_funds_count', 'balance_threshold_count', 'scheduled_trigger_count',
    'has_percentage', 'has_fixed_amount', 'has_overflow', 'has_remainder',
    'rule_count', 'destination_count', 'source_count', 'total_actions',
    'has_debt_keywords', 'has_business_keywords', 'has_savings_keywords',
    'has_budget_keywords', 'liability_mention_count', 'multiple_liabilities',
    'has_router_pattern', 'has_pod_structure', 'percentage_count',
    'uses_100_percent', 'fixed_amount_count'
]

ALL_FEATURE_NAMES = PROFILE_FEATURE_NAMES + RULE_FEATURE_NAMES

# Feature weights for similarity matching
FEATURE_WEIGHTS = {
    'income_level': 1.5,
    'user_type': 2.0,
    'is_business': 2.0,
    'debit_card_spender': 0.5,
    'activated_ports_norm': 1.0,
    'accounts_connected_norm': 1.0,
    'age_level': 0.5,
    'subscription_level': 0.5,
    'complexity_score': 1.5,
    'occupation_stability': 1.0,
    'has_debt_keywords': 2.0,
    'has_business_keywords': 2.0,
    'has_savings_keywords': 1.5,
    'has_budget_keywords': 1.5,
    'rule_count': 1.0,
    'multiple_liabilities': 1.5,
}


def get_occupation_stability(occupation: str) -> float:
    """Map occupation to income stability score (0-1)."""
    if occupation in OCCUPATION_GROUPS.get('high_income_stable', []):
        return 1.0
    elif occupation in OCCUPATION_GROUPS.get('professional', []):
        return 0.8
    elif occupation in OCCUPATION_GROUPS.get('trade', []):
        return 0.6
    elif occupation in OCCUPATION_GROUPS.get('service', []):
        return 0.4
    elif occupation in OCCUPATION_GROUPS.get('variable_income', []):
        return 0.2
    return 0.5  # Unknown


def get_goal_group(goal: str) -> str:
    """Map specific goal to goal group."""
    if goal is None:
        return 'unknown'
    for group, goals in GOAL_GROUPS.items():
        if goal in goals:
            return group
    return 'unknown'


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def encode_profile(profile: dict) -> dict:
    """Extract profile features from a user profile."""

    # Handle potential None values
    income = profile.get('ANNUALINCOME') or 'BETWEEN_50K_AND_100K'
    user_type = profile.get('USER_TYPE') or 'INDIVIDUAL'
    age_group = profile.get('AGE_GROUP') or '25-35'
    subscription = profile.get('CURRENT_SUBSCRIPTION_NAME') or 'Starter'
    debit_spender = profile.get('DEBIT_CARD_SPENDER')
    activated_ports = profile.get('ACTIVATED_PORTS') or 1
    accounts_connected = profile.get('ACCOUNTS_CONNECTED') or 5
    occupation = profile.get('OCCUPATION') or ''

    # Handle debit_card_spender (can be string 'true'/'false' or boolean)
    if isinstance(debit_spender, str):
        debit_spender = 1 if debit_spender.lower() == 'true' else 0
    elif isinstance(debit_spender, bool):
        debit_spender = 1 if debit_spender else 0
    else:
        debit_spender = 0

    activated_ports_norm = min(activated_ports / 10, 1) if activated_ports else 0
    accounts_connected_norm = min(accounts_connected / 100, 1) if accounts_connected else 0

    return {
        'income_level': INCOME_ORDINAL.get(income, 3),
        'user_type': USER_TYPE_MAP.get(user_type, 0),
        'is_business': 1 if user_type == 'BUSINESS' else 0,
        'debit_card_spender': debit_spender,
        'activated_ports_norm': activated_ports_norm,
        'accounts_connected_norm': accounts_connected_norm,
        'age_level': AGE_ORDINAL.get(age_group, 2),
        'subscription_level': SUBSCRIPTION_MAP.get(subscription, 0),
        'complexity_score': activated_ports_norm * accounts_connected_norm,
        'occupation_stability': get_occupation_stability(occupation)
    }


def extract_rule_patterns(rule_text: str) -> dict:
    """Extract structured features from rule text without embeddings."""

    if not rule_text:
        # Return zeros for empty rule text
        return {name: 0 for name in RULE_FEATURE_NAMES}

    text_lower = rule_text.lower()

    return {
        # Trigger Types
        'incoming_funds_count': rule_text.count('When funds are received'),
        'balance_threshold_count': len(re.findall(r'balance is at least', text_lower)),
        # Improved regex to catch patterns like "27TH-", "13th-", etc.
        'scheduled_trigger_count': len(re.findall(r'\d{1,2}(?:st|nd|rd|th)[\s-]', text_lower)),

        # Action Types
        'has_percentage': 1 if '%' in rule_text else 0,
        'has_fixed_amount': 1 if re.search(r'\$[\d,]+\.\d{2} moves', rule_text) else 0,
        'has_overflow': 1 if 'Anything above' in rule_text else 0,
        'has_remainder': 1 if ('Funds move from' in rule_text and '%' not in rule_text) else 0,

        # Complexity Metrics
        'rule_count': rule_text.count('When '),
        'destination_count': len(set(re.findall(r'to ([^.;\n]+)', rule_text))),
        'source_count': len(set(re.findall(r'from ([^.;\n]+)', rule_text))),
        'total_actions': rule_text.count('moves') + rule_text.count('goes'),

        # Financial Domain Indicators
        'has_debt_keywords': 1 if any(kw in text_lower for kw in ['credit', 'loan', 'debt', 'card', 'payment']) else 0,
        'has_business_keywords': 1 if any(kw in text_lower for kw in ['opex', 'payroll', 'tax', 'profit', 'operating', 'cogs']) else 0,
        'has_savings_keywords': 1 if any(kw in text_lower for kw in ['savings', 'save', 'emergency', 'reserve']) else 0,
        'has_budget_keywords': 1 if any(kw in text_lower for kw in ['budget', 'needs', 'wants', 'groceries', 'bills']) else 0,

        # Debt Strategy Indicators
        'liability_mention_count': len(re.findall(r'(credit card|loan|debt|liability)', text_lower)),
        'multiple_liabilities': 1 if len(re.findall(r'(credit card|loan|cc |card)', text_lower)) > 1 else 0,

        # Structure Indicators
        'has_router_pattern': 1 if any(kw in text_lower for kw in ['router', 'sweep', 'hub']) else 0,
        'has_pod_structure': 1 if 'pod' in text_lower else 0,

        # Percentage Extraction
        'percentage_count': len(re.findall(r'\d+\.?\d*%', rule_text)),
        'uses_100_percent': 1 if '100.00%' in rule_text or '100%' in rule_text else 0,

        # Dollar Amount Stats
        'fixed_amount_count': len(re.findall(r'\$[\d,]+\.\d{2}', rule_text)),
    }


def create_user_vector(profile: dict, rule_text: str) -> np.ndarray:
    """Create feature vector from structured features only."""

    profile_features = encode_profile(profile)
    rule_patterns = extract_rule_patterns(rule_text)

    # Build vector in consistent order
    vector = []
    for name in PROFILE_FEATURE_NAMES:
        vector.append(profile_features.get(name, 0))
    for name in RULE_FEATURE_NAMES:
        vector.append(rule_patterns.get(name, 0))

    return np.array(vector, dtype=np.float64)


# ============================================================================
# SIMILARITY MATCHER
# ============================================================================

class StructuredSimilarityMatcher:
    """Weighted cosine similarity matcher on structured features."""

    def __init__(self, feature_names=None):
        self.scaler = StandardScaler()
        self.user_vectors = None
        self.user_ids = None
        self.user_metadata = None  # Store additional info for retrieval
        self.feature_names = feature_names or ALL_FEATURE_NAMES
        self.weight_vector = self._build_weight_vector()

    def _build_weight_vector(self) -> np.ndarray:
        """Build weight vector from FEATURE_WEIGHTS dict."""
        weights = []
        for name in self.feature_names:
            weights.append(FEATURE_WEIGHTS.get(name, 1.0))
        return np.array(weights, dtype=np.float64)

    def fit(self, user_vectors: np.ndarray, user_ids: list, user_metadata: list = None):
        """Fit scaler and store user data."""
        self.user_ids = user_ids
        self.user_metadata = user_metadata

        # Scale features
        scaled = self.scaler.fit_transform(user_vectors)

        # Apply feature weights
        self.user_vectors = scaled * self.weight_vector

    def find_similar(self, query_vector: np.ndarray, k: int = 10,
                     filter_user_type: str = None) -> list:
        """Find k most similar users."""

        # Scale and weight query
        query_scaled = self.scaler.transform([query_vector])
        query_weighted = query_scaled * self.weight_vector

        # Calculate similarities
        similarities = cosine_similarity(query_weighted, self.user_vectors)[0]

        # Get all indices sorted by similarity
        sorted_indices = np.argsort(similarities)[::-1]

        # Filter and collect top-k
        results = []
        for idx in sorted_indices:
            if len(results) >= k:
                break

            # Optional: filter by user type
            if filter_user_type and self.user_metadata:
                user_type = self.user_metadata[idx].get('USER_TYPE', '')
                if user_type != filter_user_type:
                    continue

            results.append({
                'user_id': self.user_ids[idx],
                'similarity': float(similarities[idx]),
                'metadata': self.user_metadata[idx] if self.user_metadata else None
            })

        return results


# ============================================================================
# DATA LOADING
# ============================================================================

def load_and_prepare_data(profiles_path: str, rules_path: str) -> pl.DataFrame:
    """Load and join profile and rules data."""

    print("Loading data...")
    profiles = pl.read_csv(profiles_path)
    rules = pl.read_csv(rules_path)

    print(f"  Profiles: {len(profiles)} rows")
    print(f"  Rules: {len(rules)} rows")

    # Join on organization_id
    df = profiles.join(
        rules,
        left_on='ORGANIZATION_ID',
        right_on='organization_id',
        how='inner'
    )

    print(f"  Joined: {len(df)} rows")

    # Handle missing values
    df = df.with_columns([
        pl.col('ANNUALINCOME').fill_null('BETWEEN_50K_AND_100K'),
        pl.col('OCCUPATION').fill_null('UNKNOWN'),
        pl.col('AGE_GROUP').fill_null('25-35'),
        pl.col('CURRENT_SUBSCRIPTION_NAME').fill_null('Starter'),
        pl.col('ACTIVATED_PORTS').fill_null(1),
        pl.col('ACCOUNTS_CONNECTED').fill_null(5),
        pl.col('USER_TYPE').fill_null('INDIVIDUAL'),
        pl.col('DEBIT_CARD_SPENDER').fill_null('false'),
    ])

    # Report goal coverage
    with_goals = df.filter(pl.col('PRODUCTGOAL').is_not_null())
    print(f"  Users with PRODUCTGOAL: {len(with_goals)} ({len(with_goals)/len(df)*100:.1f}%)")

    return df


# ============================================================================
# TRAINING PIPELINE
# ============================================================================

def train_similarity_matcher(df: pl.DataFrame, output_dir: Path) -> StructuredSimilarityMatcher:
    """Train and save the similarity matcher."""

    print("\n=== Training Similarity Matcher ===")

    # Extract features for all users
    user_vectors = []
    user_ids = []
    user_metadata = []

    for row in df.iter_rows(named=True):
        vector = create_user_vector(row, row['description'])
        user_vectors.append(vector)
        user_ids.append(row['ORGANIZATION_ID'])
        user_metadata.append({
            'USER_TYPE': row.get('USER_TYPE'),
            'PRODUCTGOAL': row.get('PRODUCTGOAL'),
            'ANNUALINCOME': row.get('ANNUALINCOME'),
            'OCCUPATION': row.get('OCCUPATION')
        })

    user_vectors = np.array(user_vectors)
    print(f"Feature matrix shape: {user_vectors.shape}")

    # Train matcher
    matcher = StructuredSimilarityMatcher(feature_names=ALL_FEATURE_NAMES)
    matcher.fit(user_vectors, user_ids, user_metadata)

    # Save
    joblib.dump(matcher, output_dir / 'similarity_matcher.pkl')
    np.save(output_dir / 'user_vectors.npy', user_vectors)

    with open(output_dir / 'user_id_mapping.json', 'w') as f:
        json.dump(user_ids, f)

    print(f"Saved similarity_matcher.pkl, user_vectors.npy, user_id_mapping.json")

    return matcher, user_vectors, user_ids, user_metadata


def train_goal_classifier(df: pl.DataFrame, output_dir: Path):
    """Train and evaluate goal classification model."""

    print("\n=== Training Goal Classifier ===")

    # Filter to users WITH valid goals (exclude freeform "OTHER-" entries)
    df_with_goals = df.filter(
        pl.col('PRODUCTGOAL').is_not_null() &
        pl.col('PRODUCTGOAL').is_in(VALID_GOALS)
    )
    print(f"Users with valid goals: {len(df_with_goals)}")

    # Check class distribution
    goal_counts = df_with_goals.group_by('PRODUCTGOAL').len().sort('len', descending=True)
    print("\nGoal distribution:")
    for row in goal_counts.iter_rows(named=True):
        print(f"  {row['PRODUCTGOAL']}: {row['len']}")

    # Extract features
    X = np.array([
        create_user_vector(row, row['description'])
        for row in df_with_goals.iter_rows(named=True)
    ])
    y = df_with_goals['PRODUCTGOAL'].to_numpy()

    # Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

    # Train LightGBM
    clf = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        class_weight='balanced',
        random_state=42,
        verbose=-1
    )

    # Cross-validation
    print("\nRunning 5-fold cross-validation...")
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='f1_macro')
    print(f"CV F1 Macro: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")

    # Final training
    clf.fit(X_train, y_train)

    # Evaluation
    y_pred = clf.predict(X_test)
    print("\n=== Test Set Classification Report ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Goal-group level accuracy
    y_test_groups = [get_goal_group(g) for g in y_test]
    y_pred_groups = [get_goal_group(g) for g in y_pred]
    group_accuracy = sum(1 for t, p in zip(y_test_groups, y_pred_groups) if t == p) / len(y_test)
    print(f"Goal-Group Accuracy: {group_accuracy:.3f}")

    # Top-3 accuracy
    y_proba = clf.predict_proba(X_test)
    top3_correct = 0
    for i, true_label in enumerate(y_test):
        top3_idx = np.argsort(y_proba[i])[::-1][:3]
        top3_labels = [clf.classes_[j] for j in top3_idx]
        if true_label in top3_labels:
            top3_correct += 1
    print(f"Top-3 Accuracy: {top3_correct / len(y_test):.3f}")

    # Feature importance
    print("\nTop 10 Feature Importances:")
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    for i in indices:
        print(f"  {ALL_FEATURE_NAMES[i]}: {importances[i]:.3f}")

    # Save
    joblib.dump(clf, output_dir / 'goal_classifier.pkl')
    print(f"\nSaved goal_classifier.pkl")

    return clf


def train_user_clusters(df: pl.DataFrame, user_vectors: np.ndarray, output_dir: Path):
    """Train user clustering model."""

    print("\n=== Training User Clustering ===")

    # Use full feature vectors (profile + rules) instead of just rule patterns
    # This addresses the critique that clustering should include profile features
    scaler = StandardScaler()
    scaled_vectors = scaler.fit_transform(user_vectors)

    # Find optimal k using elbow method
    print("Finding optimal k...")
    inertias = []
    silhouettes = []
    k_range = range(4, 15)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_vectors)
        inertias.append(kmeans.inertia_)
        sil = silhouette_score(scaled_vectors, clusters)
        silhouettes.append(sil)
        print(f"  k={k}: inertia={kmeans.inertia_:.0f}, silhouette={sil:.3f}")

    # Choose k with best silhouette (simple heuristic)
    best_k = k_range[np.argmax(silhouettes)]
    print(f"\nBest k by silhouette: {best_k}")

    # Train final model with k=8 (as specified in plan, but can adjust)
    final_k = 8
    print(f"Training final model with k={final_k}")
    kmeans = KMeans(n_clusters=final_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_vectors)

    # Analyze clusters
    print("\nCluster Analysis:")
    df_with_clusters = df.with_columns(pl.Series('cluster', clusters))

    for cluster_id in range(final_k):
        cluster_df = df_with_clusters.filter(pl.col('cluster') == cluster_id)
        size = len(cluster_df)

        # Goal distribution
        goals = cluster_df.filter(pl.col('PRODUCTGOAL').is_not_null())['PRODUCTGOAL'].to_list()
        goal_dist = {}
        for g in goals:
            goal_dist[g] = goal_dist.get(g, 0) + 1
        top_goals = sorted(goal_dist.items(), key=lambda x: -x[1])[:2]

        # User type distribution
        user_types = cluster_df['USER_TYPE'].to_list()
        business_pct = sum(1 for u in user_types if u == 'BUSINESS') / len(user_types) * 100

        print(f"\nCluster {cluster_id} ({size} users, {business_pct:.0f}% business):")
        for goal, count in top_goals:
            print(f"  {goal}: {count}")

    # Save
    joblib.dump({
        'kmeans': kmeans,
        'scaler': scaler,
        'n_clusters': final_k
    }, output_dir / 'user_clusters.pkl')

    print(f"\nSaved user_clusters.pkl")

    return kmeans, clusters


def evaluate_similarity_matcher(matcher: StructuredSimilarityMatcher,
                                df: pl.DataFrame,
                                user_vectors: np.ndarray,
                                user_ids: list):
    """Evaluate similarity matcher quality."""

    print("\n=== Evaluating Similarity Matcher ===")

    # Hold-out evaluation: for each user, find similar users and check goal alignment
    df_with_goals = df.filter(pl.col('PRODUCTGOAL').is_not_null())

    # Create lookup dict for goals
    goal_lookup = {}
    for row in df.iter_rows(named=True):
        goal_lookup[row['ORGANIZATION_ID']] = {
            'goal': row.get('PRODUCTGOAL'),
            'goal_group': get_goal_group(row.get('PRODUCTGOAL')),
            'user_type': row.get('USER_TYPE')
        }

    # Sample 100 users for evaluation
    sample_size = min(100, len(df_with_goals))
    sample_indices = np.random.choice(len(df_with_goals), sample_size, replace=False)

    goal_group_matches = []
    user_type_matches = []

    for idx in sample_indices:
        row = df_with_goals.row(idx, named=True)
        query_id = row['ORGANIZATION_ID']
        query_goal_group = get_goal_group(row['PRODUCTGOAL'])
        query_user_type = row['USER_TYPE']

        # Find query vector index
        query_idx = user_ids.index(query_id)
        query_vector = user_vectors[query_idx]

        # Find top 6 (will skip self, so get 5 actual matches)
        similar = matcher.find_similar(query_vector, k=6)

        # Skip self
        similar = [s for s in similar if s['user_id'] != query_id][:5]

        # Check goal group match
        group_match_count = sum(
            1 for s in similar
            if goal_lookup.get(s['user_id'], {}).get('goal_group') == query_goal_group
        )
        goal_group_matches.append(group_match_count)

        # Check user type match
        type_match_count = sum(
            1 for s in similar
            if goal_lookup.get(s['user_id'], {}).get('user_type') == query_user_type
        )
        user_type_matches.append(type_match_count)

    avg_goal_group_match = np.mean(goal_group_matches) / 5
    avg_user_type_match = np.mean(user_type_matches) / 5

    print(f"Goal-Group Match@5: {avg_goal_group_match:.3f} (target: >0.6)")
    print(f"User-Type Match@5: {avg_user_type_match:.3f} (target: >0.8)")

    # Show example matches
    print("\nExample Similarity Matches:")
    for i in range(3):
        row = df_with_goals.row(sample_indices[i], named=True)
        query_idx = user_ids.index(row['ORGANIZATION_ID'])
        query_vector = user_vectors[query_idx]
        similar = matcher.find_similar(query_vector, k=4)
        similar = [s for s in similar if s['user_id'] != row['ORGANIZATION_ID']][:3]

        print(f"\n  Query: {row['PRODUCTGOAL']} ({row['USER_TYPE']})")
        for s in similar:
            info = goal_lookup.get(s['user_id'], {})
            print(f"    Match: {info.get('goal', 'N/A')} ({info.get('user_type', 'N/A')}) - sim: {s['similarity']:.3f}")


def save_feature_config(output_dir: Path):
    """Save feature configuration for inference."""

    config = {
        'feature_names': ALL_FEATURE_NAMES,
        'profile_feature_names': PROFILE_FEATURE_NAMES,
        'rule_feature_names': RULE_FEATURE_NAMES,
        'feature_weights': FEATURE_WEIGHTS,
        'encodings': {
            'income': INCOME_ORDINAL,
            'age': AGE_ORDINAL,
            'user_type': USER_TYPE_MAP,
            'subscription': SUBSCRIPTION_MAP
        },
        'goal_groups': GOAL_GROUPS,
        'occupation_groups': OCCUPATION_GROUPS
    }

    with open(output_dir / 'feature_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Saved feature_config.json")


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Paths
    base_dir = Path('/Users/itaydror/Map generator')
    profiles_path = base_dir / 'enrichment_data.csv'
    rules_path = base_dir / 'Itaytestfinal.csv'
    output_dir = base_dir / 'models'

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Load data
    df = load_and_prepare_data(str(profiles_path), str(rules_path))

    # Train models
    matcher, user_vectors, user_ids, user_metadata = train_similarity_matcher(df, output_dir)
    clf = train_goal_classifier(df, output_dir)
    kmeans, clusters = train_user_clusters(df, user_vectors, output_dir)

    # Evaluate similarity matcher
    evaluate_similarity_matcher(matcher, df, user_vectors, user_ids)

    # Save config
    save_feature_config(output_dir)

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nOutput artifacts in {output_dir}:")
    for f in output_dir.iterdir():
        size = f.stat().st_size
        print(f"  {f.name}: {size/1024:.1f} KB")


if __name__ == '__main__':
    main()
