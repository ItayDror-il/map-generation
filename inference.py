"""
Sequence Pattern Matcher - Inference Module
Load trained models and make predictions for new users.
"""

import json
import sys
import numpy as np
import joblib
from pathlib import Path

# Import feature extraction functions from training module
# Also import StructuredSimilarityMatcher to ensure it's available for unpickling
import train_models
from train_models import (
    create_user_vector,
    get_goal_group,
    GOAL_GROUPS,
    ALL_FEATURE_NAMES,
    StructuredSimilarityMatcher
)

# Fix pickle module resolution: ensure StructuredSimilarityMatcher can be found
# when unpickling models saved from __main__
sys.modules['__main__'].StructuredSimilarityMatcher = StructuredSimilarityMatcher


class PatternMatcher:
    """
    Main interface for pattern matching.
    Combines similarity search, goal prediction, and clustering.
    """

    def __init__(self, models_dir: str = 'models'):
        self.models_dir = Path(models_dir)
        self._load_models()

    def _load_models(self):
        """Load all trained models and artifacts."""

        print("Loading models...")

        # Similarity matcher
        self.matcher = joblib.load(self.models_dir / 'similarity_matcher.pkl')

        # Goal classifier
        self.goal_clf = joblib.load(self.models_dir / 'goal_classifier.pkl')

        # Clustering
        cluster_data = joblib.load(self.models_dir / 'user_clusters.pkl')
        self.kmeans = cluster_data['kmeans']
        self.cluster_scaler = cluster_data['scaler']

        # User ID mapping
        with open(self.models_dir / 'user_id_mapping.json') as f:
            self.user_ids = json.load(f)

        # User vectors for lookup
        self.user_vectors = np.load(self.models_dir / 'user_vectors.npy')

        # Feature config
        with open(self.models_dir / 'feature_config.json') as f:
            self.feature_config = json.load(f)

        print(f"Loaded {len(self.user_ids)} users")

    def match_user(self, profile: dict, rules_text: str = None, k: int = 10,
                   filter_same_type: bool = True) -> dict:
        """
        Find similar users and predict goal for a new user.

        Args:
            profile: dict with keys like ANNUALINCOME, USER_TYPE, OCCUPATION, etc.
            rules_text: optional rule description text (empty string for new users)
            k: number of similar users to return
            filter_same_type: if True, only return users of same USER_TYPE

        Returns:
            dict with:
            - similar_users: list of similar user matches with similarity scores
            - predicted_goal: predicted PRODUCTGOAL (if not already set)
            - top3_goals: top 3 goal predictions with probabilities
            - cluster: assigned user archetype cluster
            - goal_group: predicted goal group (debt, budget, savings, etc.)
        """

        # Create feature vector (use float64 for sklearn compatibility)
        if rules_text is None:
            rules_text = ""
        vector = create_user_vector(profile, rules_text).astype(np.float64)

        # Find similar users
        filter_type = profile.get('USER_TYPE') if filter_same_type else None
        similar = self.matcher.find_similar(vector, k=k, filter_user_type=filter_type)

        # Predict goal
        predicted_goal = self.goal_clf.predict([vector])[0]
        proba = self.goal_clf.predict_proba([vector])[0]
        top3_idx = np.argsort(proba)[::-1][:3]
        top3_goals = [
            {'goal': self.goal_clf.classes_[i], 'probability': float(proba[i])}
            for i in top3_idx
        ]

        # Assign cluster
        scaled_vector = self.cluster_scaler.transform([vector])
        cluster = int(self.kmeans.predict(scaled_vector)[0])

        # Determine goal group
        goal_group = get_goal_group(predicted_goal)

        return {
            'similar_users': similar,
            'predicted_goal': predicted_goal,
            'top3_goals': top3_goals,
            'cluster': cluster,
            'goal_group': goal_group
        }

    def get_user_rules(self, user_id: str, df=None) -> str:
        """
        Get rule descriptions for a specific user ID.
        Requires the original dataframe to be passed.
        """
        if df is None:
            raise ValueError("DataFrame required to lookup user rules")

        user_rows = df.filter(df['ORGANIZATION_ID'] == user_id)
        if len(user_rows) == 0:
            return None

        return user_rows.row(0, named=True).get('description', '')

    def explain_cluster(self, cluster_id: int) -> dict:
        """
        Get interpretable description of a cluster.
        """

        # Cluster descriptions based on training analysis
        CLUSTER_DESCRIPTIONS = {
            0: "High-complexity business users with multiple income sources",
            1: "Individual users with budget-focused goals",
            2: "Mixed personal/business users focused on visualization",
            3: "Business users focused on budget automation",
            4: "Debt-focused users with moderate complexity",
            5: "Profit-first business users",
            6: "Small segment with debt focus",
            7: "Mixed users with debt and budgeting goals"
        }

        return {
            'cluster_id': cluster_id,
            'description': CLUSTER_DESCRIPTIONS.get(cluster_id, "Unknown cluster")
        }


def demo():
    """Demonstrate the pattern matcher with example profiles."""

    matcher = PatternMatcher(models_dir='/Users/itaydror/Map generator/models')

    # Example 1: New business user
    print("\n" + "="*60)
    print("Example 1: Business user with high income")
    print("="*60)

    profile1 = {
        'ANNUALINCOME': 'BETWEEN_100K_AND_250K',
        'OCCUPATION': 'EXECUTIVE_OR_MANAGER',
        'USER_TYPE': 'BUSINESS',
        'DEBIT_CARD_SPENDER': 'true',
        'ACTIVATED_PORTS': 3,
        'ACCOUNTS_CONNECTED': 25,
        'AGE_GROUP': '36-44',
        'CURRENT_SUBSCRIPTION_NAME': 'Pro'
    }

    result1 = matcher.match_user(profile1)

    print(f"\nPredicted Goal: {result1['predicted_goal']}")
    print(f"Goal Group: {result1['goal_group']}")
    print(f"Cluster: {result1['cluster']}")
    print(f"\nTop 3 Goals:")
    for g in result1['top3_goals']:
        print(f"  {g['goal']}: {g['probability']:.3f}")

    print(f"\nTop 5 Similar Users:")
    for i, user in enumerate(result1['similar_users'][:5]):
        print(f"  {i+1}. {user['user_id'][:8]}... (sim: {user['similarity']:.3f})")
        if user['metadata']:
            print(f"      Goal: {user['metadata'].get('PRODUCTGOAL', 'N/A')}")

    # Example 2: Individual focused on debt
    print("\n" + "="*60)
    print("Example 2: Individual with debt focus")
    print("="*60)

    profile2 = {
        'ANNUALINCOME': 'BETWEEN_50K_AND_100K',
        'OCCUPATION': 'SCIENTIST_OR_TECHNOLOGIST',
        'USER_TYPE': 'INDIVIDUAL',
        'DEBIT_CARD_SPENDER': 'false',
        'ACTIVATED_PORTS': 1,
        'ACCOUNTS_CONNECTED': 10,
        'AGE_GROUP': '25-35',
        'CURRENT_SUBSCRIPTION_NAME': 'Starter'
    }

    # With some rule text that mentions debt
    rules2 = """When funds are received: $200.00 moves from Checking to Credit Card Payment.
When funds are received: $150.00 moves from Checking to Student Loan.
When funds are received: 10% of incoming funds goes to Emergency Fund."""

    result2 = matcher.match_user(profile2, rules_text=rules2)

    print(f"\nPredicted Goal: {result2['predicted_goal']}")
    print(f"Goal Group: {result2['goal_group']}")
    print(f"Cluster: {result2['cluster']}")
    print(f"\nTop 3 Goals:")
    for g in result2['top3_goals']:
        print(f"  {g['goal']}: {g['probability']:.3f}")


if __name__ == '__main__':
    demo()
