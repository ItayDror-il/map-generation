# Sequence Pattern Matcher - ML Training Plan

## Executive Summary

Train a pattern matching system to recommend financial automations based on user profiles and successful user patterns from 2,293 existing Sequence users.

**Approach: Structured Features First, Embeddings Optional**

---

## 1. Problem Definition

### Primary Task: User Similarity Matching
- **Input:** New user profile (income, occupation, goal, user_type, etc.)
- **Output:** Top-K most similar existing users + their automation patterns
- **Approach:** Weighted similarity on structured features (no embeddings initially)

### Secondary Task: Goal Prediction
- **Input:** User profile + extracted rule patterns
- **Output:** Predicted PRODUCTGOAL (15 categories)
- **Purpose:** Infer goals for the 40% of users with missing PRODUCTGOAL
- **Approach:** Multi-class classification with LightGBM

### Tertiary Task: Rule Pattern Clustering
- **Input:** Extracted rule pattern features
- **Output:** User archetype cluster assignment
- **Purpose:** Discover implicit user segments beyond stated goals
- **Approach:** K-Means on structured pattern features

---

## 2. Data Overview

| Dataset | Rows | Key Columns |
|---------|------|-------------|
| enrichment_data.csv | 2,522 | ORGANIZATION_ID, ANNUALINCOME, OCCUPATION, PRODUCTGOAL, USER_TYPE, ACCOUNTS_CONNECTED, ACTIVATED_PORTS, AGE_GROUP |
| itaytestfinal.csv | 2,293 | organization_id, description (rule text) |

### Data Quality Issues
- **PRODUCTGOAL:** 40% missing (1,013 nulls) - primary target for classification
- **ANNUALINCOME/OCCUPATION:** ~4% missing - impute with mode or "UNKNOWN"
- **Join coverage:** 100% of rules have matching profiles

---

## 3. Feature Engineering (Structured Only)

### 3.1 Profile Features

```python
import numpy as np

# Categorical encoding
INCOME_ORDINAL = {
    'UP_TO_10K': 0, 'BETWEEN_10K_AND_25K': 1, 'BETWEEN_25K_AND_50K': 2,
    'BETWEEN_50K_AND_100K': 3, 'BETWEEN_100K_AND_250K': 4, 'OVER_250K': 5
}

AGE_ORDINAL = {
    '18-24': 0, '25-35': 1, '36-44': 2, '45-54': 3, '55-64': 4, '65+': 5
}

USER_TYPE_MAP = {'INDIVIDUAL': 0, 'BUSINESS': 1, 'ADVISOR': 2}

SUBSCRIPTION_MAP = {'Starter': 0, 'Pro': 1, 'Business': 2, 'Growth': 3}

def encode_profile(profile):
    return {
        'income_level': INCOME_ORDINAL.get(profile['ANNUALINCOME'], 3),
        'user_type': USER_TYPE_MAP.get(profile['USER_TYPE'], 0),
        'is_business': 1 if profile['USER_TYPE'] == 'BUSINESS' else 0,
        'debit_card_spender': 1 if profile['DEBIT_CARD_SPENDER'] == 'true' else 0,
        'activated_ports_norm': min(profile['ACTIVATED_PORTS'] / 10, 1),
        'accounts_connected_norm': min(profile['ACCOUNTS_CONNECTED'] / 100, 1),
        'age_level': AGE_ORDINAL.get(profile['AGE_GROUP'], 2),
        'subscription_level': SUBSCRIPTION_MAP.get(profile['CURRENT_SUBSCRIPTION_NAME'], 0),
        'complexity_score': min(profile['ACTIVATED_PORTS'] / 10, 1) * min(profile['ACCOUNTS_CONNECTED'] / 100, 1)
    }
```

### 3.2 Rule Pattern Features (Extracted from Text)

Parse rule text to extract structured patterns - **no embeddings, pure regex/counting:**

```python
import re

def extract_rule_patterns(rule_text):
    """Extract structured features from rule text without embeddings"""

    text_lower = rule_text.lower()

    return {
        # === Trigger Types ===
        'incoming_funds_count': rule_text.count('When funds are received'),
        'balance_threshold_count': len(re.findall(r'balance is at least', text_lower)),
        'scheduled_trigger_count': len(re.findall(r'when.*\d{1,2}(st|nd|rd|th)', text_lower)),

        # === Action Types ===
        'has_percentage': 1 if '%' in rule_text else 0,
        'has_fixed_amount': 1 if re.search(r'\$[\d,]+\.\d{2} moves', rule_text) else 0,
        'has_overflow': 1 if 'Anything above' in rule_text else 0,
        'has_remainder': 1 if ('Funds move from' in rule_text and '%' not in rule_text) else 0,

        # === Complexity Metrics ===
        'rule_count': rule_text.count('When '),
        'destination_count': len(set(re.findall(r'to ([^.;\n]+)', rule_text))),
        'source_count': len(set(re.findall(r'from ([^.;\n]+)', rule_text))),
        'total_actions': rule_text.count('moves') + rule_text.count('goes'),

        # === Financial Domain Indicators ===
        'has_debt_keywords': 1 if any(kw in text_lower for kw in ['credit', 'loan', 'debt', 'card', 'payment']) else 0,
        'has_business_keywords': 1 if any(kw in text_lower for kw in ['opex', 'payroll', 'tax', 'profit', 'operating', 'cogs']) else 0,
        'has_savings_keywords': 1 if any(kw in text_lower for kw in ['savings', 'save', 'emergency', 'reserve']) else 0,
        'has_budget_keywords': 1 if any(kw in text_lower for kw in ['budget', 'needs', 'wants', 'groceries', 'bills']) else 0,

        # === Debt Strategy Indicators ===
        'liability_mention_count': len(re.findall(r'(credit card|loan|debt|liability)', text_lower)),
        'multiple_liabilities': 1 if len(re.findall(r'(credit card|loan|cc |card)', text_lower)) > 1 else 0,

        # === Structure Indicators ===
        'has_router_pattern': 1 if any(kw in text_lower for kw in ['router', 'sweep', 'hub']) else 0,
        'has_pod_structure': 1 if 'pod' in text_lower else 0,

        # === Percentage Extraction ===
        'percentage_count': len(re.findall(r'\d+\.?\d*%', rule_text)),
        'uses_100_percent': 1 if '100.00%' in rule_text or '100%' in rule_text else 0,

        # === Dollar Amount Stats ===
        'fixed_amount_count': len(re.findall(r'\$[\d,]+\.\d{2}', rule_text)),
    }
```

### 3.3 Goal One-Hot Encoding (for users with goals)

```python
GOAL_CATEGORIES = [
    'DEBT_PAYMENT', 'PAY_OFF_DEBT',  # Debt-focused
    'AUTOMATE_MY_BUDGETING', 'ORGANIZE_FINANCING',  # Budget-focused
    'MAXIMIZE_SAVINGS', 'MAXIMIZE_MY_SAVINGS', 'SAVE_FOR_TAXES',  # Savings-focused
    'PROFIT_FIRST', 'OPTIMIZE_CASH_FLOW', 'MANAGE_BUSINESS_PAYMENTS',  # Business-focused
    'VISUALIZE_AND_TRACK_FINANCES', 'MAINTAIN_CONTROL',  # Tracking-focused
    'AUTOMATE_MY_BILLS', 'AUTOMATE_MY_INVESTMENTS'  # Automation-focused
]

# Simplified goal groups for matching
GOAL_GROUPS = {
    'debt': ['DEBT_PAYMENT', 'PAY_OFF_DEBT'],
    'budget': ['AUTOMATE_MY_BUDGETING', 'ORGANIZE_FINANCING'],
    'savings': ['MAXIMIZE_SAVINGS', 'MAXIMIZE_MY_SAVINGS', 'SAVE_FOR_TAXES'],
    'business': ['PROFIT_FIRST', 'OPTIMIZE_CASH_FLOW', 'MANAGE_BUSINESS_PAYMENTS'],
    'tracking': ['VISUALIZE_AND_TRACK_FINANCES', 'MAINTAIN_CONTROL'],
    'automation': ['AUTOMATE_MY_BILLS', 'AUTOMATE_MY_INVESTMENTS']
}

def get_goal_group(goal):
    for group, goals in GOAL_GROUPS.items():
        if goal in goals:
            return group
    return 'unknown'
```

### 3.4 Combined Feature Vector (No Embeddings)

```python
def create_user_vector(profile, rule_text):
    """Create feature vector from structured features only - NO embeddings"""

    profile_features = encode_profile(profile)
    rule_patterns = extract_rule_patterns(rule_text)

    # Combine all features into a single vector
    # Profile: 9 features + Rule patterns: ~22 features = ~31 total dimensions

    feature_vector = list(profile_features.values()) + list(rule_patterns.values())

    return np.array(feature_vector, dtype=np.float32)
```

**Total feature dimensions: ~31** (vs 402 with embeddings)

---

## 4. Model Architecture (Structured-First)

### 4.1 Similarity Search (Primary) - Weighted Matching

**Approach:** Weighted cosine similarity on normalized structured features

```python
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class StructuredSimilarityMatcher:
    def __init__(self):
        self.scaler = StandardScaler()
        self.user_vectors = None
        self.user_ids = None

        # Feature weights - tune based on what matters most
        self.feature_weights = {
            # Profile weights (higher = more important for matching)
            'income_level': 1.5,
            'user_type': 2.0,        # Business vs Individual is critical
            'is_business': 2.0,
            'debit_card_spender': 0.5,
            'activated_ports_norm': 1.0,
            'accounts_connected_norm': 1.0,
            'age_level': 0.5,
            'subscription_level': 0.5,
            'complexity_score': 1.5,

            # Rule pattern weights
            'has_debt_keywords': 2.0,      # Strong signal
            'has_business_keywords': 2.0,  # Strong signal
            'has_savings_keywords': 1.5,
            'has_budget_keywords': 1.5,
            'rule_count': 1.0,
            'multiple_liabilities': 1.5,
            # ... other pattern features get weight 1.0 by default
        }

    def fit(self, user_vectors, user_ids):
        """Fit scaler and store user data"""
        self.user_ids = user_ids
        self.user_vectors = self.scaler.fit_transform(user_vectors)

        # Apply feature weights
        weights = self._get_weight_vector(user_vectors.shape[1])
        self.user_vectors = self.user_vectors * weights

    def _get_weight_vector(self, n_features):
        """Create weight vector matching feature order"""
        # Default weight is 1.0
        return np.ones(n_features)  # Customize based on feature order

    def find_similar(self, query_vector, k=10, filter_same_type=True):
        """Find k most similar users"""
        query_scaled = self.scaler.transform([query_vector])

        # Calculate similarities
        similarities = cosine_similarity(query_scaled, self.user_vectors)[0]

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:k]

        return [
            {'user_id': self.user_ids[i], 'similarity': similarities[i]}
            for i in top_indices
        ]
```

**Storage:** Save with joblib to `models/similarity_matcher.pkl`

### 4.2 Goal Classification (Secondary)

**Approach:** LightGBM on structured features only

```python
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score

# Filter to users WITH goals (60% of data)
users_with_goals = df[df['PRODUCTGOAL'].notna()]

# Features: profile + rule patterns (NO embeddings)
X = np.array([create_user_vector(row, row['description'])
              for row in users_with_goals.iter_rows(named=True)])
y = users_with_goals['PRODUCTGOAL'].to_numpy()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train with class weights (imbalanced classes)
clf = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=6,
    class_weight='balanced',
    random_state=42,
    verbose=-1
)

# Cross-validation first
cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='f1_macro')
print(f"CV F1 Macro: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")

# Final training
clf.fit(X_train, y_train)
```

**Storage:** Save to `models/goal_classifier.pkl`

### 4.3 User Clustering (Tertiary)

**Approach:** K-Means on structured rule pattern features

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Extract just rule pattern features for clustering
rule_pattern_features = np.array([
    list(extract_rule_patterns(desc).values())
    for desc in df['description'].to_list()
])

# Scale features
scaler = StandardScaler()
scaled_patterns = scaler.fit_transform(rule_pattern_features)

# Find optimal k using elbow method
inertias = []
for k in range(2, 15):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_patterns)
    inertias.append(kmeans.inertia_)

# Train final model (start with k=8, adjust based on elbow)
kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
clusters = kmeans.fit_predict(scaled_patterns)

# Analyze cluster characteristics
for cluster_id in range(8):
    cluster_mask = clusters == cluster_id
    cluster_goals = df[cluster_mask]['PRODUCTGOAL'].value_counts()
    print(f"\nCluster {cluster_id} ({cluster_mask.sum()} users):")
    print(cluster_goals.head(3))
```

**Storage:** Save to `models/user_clusters.pkl`

---

## 5. Training Pipeline

### Step 1: Data Loading & Preprocessing

```python
import polars as pl
import numpy as np

# Load data
profiles = pl.read_csv('enrichment_data.csv')
rules = pl.read_csv('itaytestfinal.csv')

# Join on organization_id
df = profiles.join(
    rules,
    left_on='ORGANIZATION_ID',
    right_on='organization_id',
    how='inner'
)

print(f"Joined dataset: {len(df)} rows")

# Handle missing values
df = df.with_columns([
    pl.col('ANNUALINCOME').fill_null('BETWEEN_50K_AND_100K'),
    pl.col('OCCUPATION').fill_null('UNKNOWN'),
    pl.col('AGE_GROUP').fill_null('25-35'),
    pl.col('CURRENT_SUBSCRIPTION_NAME').fill_null('Starter'),
    pl.col('ACTIVATED_PORTS').fill_null(1),
    pl.col('ACCOUNTS_CONNECTED').fill_null(5),
])
```

### Step 2: Feature Extraction

```python
# Extract features for all users
user_vectors = []
user_ids = []

for row in df.iter_rows(named=True):
    vector = create_user_vector(row, row['description'])
    user_vectors.append(vector)
    user_ids.append(row['ORGANIZATION_ID'])

user_vectors = np.array(user_vectors)
print(f"Feature matrix shape: {user_vectors.shape}")  # Should be (2293, ~31)
```

### Step 3: Train Models

```python
import joblib
import json

# Create models directory
import os
os.makedirs('models', exist_ok=True)

# 3a. Train similarity matcher
matcher = StructuredSimilarityMatcher()
matcher.fit(user_vectors, user_ids)
joblib.dump(matcher, 'models/similarity_matcher.pkl')

# 3b. Train goal classifier (on users with goals)
df_with_goals = df.filter(pl.col('PRODUCTGOAL').is_not_null())
X_goals = np.array([create_user_vector(row, row['description'])
                    for row in df_with_goals.iter_rows(named=True)])
y_goals = df_with_goals['PRODUCTGOAL'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X_goals, y_goals, test_size=0.2, stratify=y_goals, random_state=42
)

clf = lgb.LGBMClassifier(
    n_estimators=200, learning_rate=0.05, num_leaves=31,
    class_weight='balanced', random_state=42, verbose=-1
)
clf.fit(X_train, y_train)
joblib.dump(clf, 'models/goal_classifier.pkl')

# 3c. Train clustering
rule_patterns = np.array([list(extract_rule_patterns(d).values()) for d in df['description'].to_list()])
pattern_scaler = StandardScaler()
scaled_patterns = pattern_scaler.fit_transform(rule_patterns)

kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
clusters = kmeans.fit_predict(scaled_patterns)
joblib.dump({'kmeans': kmeans, 'scaler': pattern_scaler}, 'models/user_clusters.pkl')

# Save metadata
with open('models/user_id_mapping.json', 'w') as f:
    json.dump(user_ids, f)

# Save feature vectors for later use
np.save('models/user_vectors.npy', user_vectors)
```

### Step 4: Evaluation

```python
from sklearn.metrics import classification_report, confusion_matrix

# Goal classification evaluation
y_pred = clf.predict(X_test)
print("Goal Classification Results:")
print(classification_report(y_test, y_pred))

# Similarity evaluation - manual inspection
print("\nSimilarity Search Evaluation:")
test_idx = 0  # Pick a test user
test_user = df_with_goals.row(test_idx, named=True)
test_vector = create_user_vector(test_user, test_user['description'])

similar = matcher.find_similar(test_vector, k=5)
print(f"\nTest user goal: {test_user['PRODUCTGOAL']}")
print("Similar users:")
for match in similar:
    matched_user = df.filter(pl.col('ORGANIZATION_ID') == match['user_id']).row(0, named=True)
    print(f"  - {match['user_id'][:8]}... | Goal: {matched_user.get('PRODUCTGOAL', 'N/A')} | Sim: {match['similarity']:.3f}")
```

---

## 6. Evaluation Metrics

### Similarity Search
| Metric | Target | Description |
|--------|--------|-------------|
| Goal-Group Match@5 | >0.6 | 3+ of top 5 users share same goal group |
| User-Type Match@5 | >0.8 | 4+ of top 5 users share same user type |
| Manual Review | Pass | 20 spot-checks return sensible results |

### Goal Classification
| Metric | Target | Description |
|--------|--------|-------------|
| Macro F1 | >0.4 | Balanced across 15 classes (hard task) |
| Balanced Accuracy | >0.35 | Account for class imbalance |
| Top-3 Accuracy | >0.6 | Correct goal in top 3 predictions |
| Goal-Group Accuracy | >0.6 | Correct goal GROUP (6 groups) |

### Clustering
| Metric | Target | Description |
|--------|--------|-------------|
| Silhouette | >0.2 | Cluster separation (lower bar without embeddings) |
| Interpretability | Pass | Each cluster has identifiable characteristics |

---

## 7. Output Artifacts

```
models/
├── similarity_matcher.pkl    # Weighted similarity matcher
├── goal_classifier.pkl       # LightGBM goal predictor
├── user_clusters.pkl         # KMeans + scaler
├── user_vectors.npy          # Pre-computed feature vectors (2293 x 31)
├── user_id_mapping.json      # Index position → organization_id
└── feature_config.json       # Feature names, weights, encodings
```

---

## 8. Integration with Skills

### user-pattern-matcher skill usage:

```python
import joblib
import json
import numpy as np

# Load models
matcher = joblib.load('models/similarity_matcher.pkl')
clf = joblib.load('models/goal_classifier.pkl')
user_ids = json.load(open('models/user_id_mapping.json'))

def match_user(new_profile, new_rules_text=None):
    """
    Find similar users for a new user.

    Args:
        new_profile: dict with keys like ANNUALINCOME, USER_TYPE, etc.
        new_rules_text: optional rule description (for existing users)

    Returns:
        similar_users: list of similar user IDs with scores
        predicted_goal: predicted goal if not provided
    """

    # If no rules yet, use empty/default pattern
    if new_rules_text is None:
        new_rules_text = ""

    # Create feature vector
    vector = create_user_vector(new_profile, new_rules_text)

    # Find similar users
    similar = matcher.find_similar(vector, k=10)

    # Predict goal if not provided
    predicted_goal = None
    if not new_profile.get('PRODUCTGOAL'):
        predicted_goal = clf.predict([vector])[0]
        # Also get top-3 predictions with probabilities
        proba = clf.predict_proba([vector])[0]
        top3_idx = np.argsort(proba)[::-1][:3]
        top3_goals = [(clf.classes_[i], proba[i]) for i in top3_idx]

    return {
        'similar_users': similar,
        'predicted_goal': predicted_goal,
        'top3_goals': top3_goals if predicted_goal else None
    }
```

---

## 9. Recommended Libraries

```
polars>=0.20.0          # Fast dataframes
lightgbm               # Classification
scikit-learn           # Similarity, clustering, preprocessing
joblib                 # Model serialization
numpy                  # Arrays
```

**NOT required initially:**
- sentence-transformers (embeddings)
- faiss (vector search)
- torch (deep learning)

---

## 10. Optional: Adding Embeddings Later

If structured features don't meet targets, add embeddings as a **weighted component**:

```python
# Only if structured matching isn't good enough
from sentence_transformers import SentenceTransformer

def create_hybrid_vector(profile, rule_text, use_embeddings=False, embedding_weight=0.2):
    """Create feature vector with optional embeddings"""

    structured_vector = create_user_vector(profile, rule_text)  # ~31 dims

    if not use_embeddings:
        return structured_vector

    # Add embeddings as weighted component
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding = model.encode(rule_text, normalize_embeddings=True)  # 384 dims

    # Weight embeddings lower than structured features
    weighted_embedding = embedding * embedding_weight

    return np.concatenate([structured_vector, weighted_embedding])
```

**When to add embeddings:**
- Structured Macro F1 < 0.3 after tuning
- Similar users frequently have mismatched goals
- New users use vocabulary not captured by keyword matching

---

## 11. Next Steps

1. **Implement** the training pipeline above
2. **Evaluate** against targets in Section 6
3. **Tune** feature weights based on error analysis
4. **Iterate** on rule pattern extraction if needed
5. **Add embeddings** only if structured approach doesn't meet targets
6. **Deploy** to user-pattern-matcher skill
