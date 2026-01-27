---
name: user-pattern-matcher
description: Match users to successful automation patterns using ML. Use when: (1) Finding similar users from the database to inform recommendations, (2) Validating a proposed plan against successful user patterns, (3) Extracting intent/goals from existing user rules, (4) Training or updating the pattern matching model. This skill handles the ML/data science aspects of map generation.
---

# User Pattern Matcher

Find similar successful users and extract patterns to inform automation recommendations.

## Overview

This skill uses machine learning to match new users to patterns from 2,500+ existing Sequence users. It analyzes user profiles and their automation rules to find relevant templates and validate proposed plans.

## Data Sources

### User Profiles (enrichment_data.csv)
Features available:
- `ANNUALINCOME` - Income bracket (UP_TO_10K, BETWEEN_10K_AND_25K, BETWEEN_25K_AND_50K, BETWEEN_50K_AND_100K, BETWEEN_100K_AND_250K, OVER_250K)
- `OCCUPATION` - Job category
- `PRODUCTGOAL` - Stated primary goal
- `USER_TYPE` - INDIVIDUAL or BUSINESS
- `CURRENT_SUBSCRIPTION_NAME` - Starter, Pro, Business, Growth
- `DEBIT_CARD_SPENDER` - true/false
- `ACTIVATED_PORTS` - Number of income sources connected
- `ACCOUNTS_CONNECTED` - Total connected accounts
- `AGE_GROUP` - Age bracket

### User Rules (itaytestfinal.csv)
- `organization_id` - Links to profile
- `description` - Human-readable rules (parseable for pattern extraction)

## Matching Process

### 1. Feature Engineering

**Categorical encoding:**
- Income bracket → ordinal (0-5)
- User type → binary
- Occupation → one-hot or embedding
- Goal → one-hot or embedding

**Numerical features:**
- Accounts connected (normalize)
- Activated ports (normalize)

**Derived features:**
- Complexity score = accounts_connected * activated_ports
- Business indicator = USER_TYPE == BUSINESS
- Goal alignment score (if comparing to a specific pattern)

### 2. Similarity Matching

**For finding similar users:**
```
Input: New user profile
Output: Top-K most similar existing users

Approach options:
1. KNN on encoded features (cosine similarity)
2. Embedding-based similarity
3. Rule-based filtering + similarity
```

**Feature weights (suggested starting point):**
- PRODUCTGOAL: High weight (primary intent signal)
- USER_TYPE: High weight (personal vs business very different)
- ANNUALINCOME: Medium weight (affects scale of automations)
- ACCOUNTS_CONNECTED: Medium weight (complexity indicator)
- OCCUPATION: Low weight (weak signal)

### 3. Pattern Extraction from Rules

Parse human-readable rules to extract:

**Trigger patterns:**
- "When funds are received" → INCOMING_FUNDS
- "When [Pod] balance is at least $X" → BALANCE_THRESHOLD
- Scheduled patterns → SCHEDULED

**Action patterns:**
- "X% of incoming funds" → PERCENTAGE
- "$X moves from" → FIXED
- "Anything above $X moves" → TOP_UP/overflow
- "Funds move from" → REMAINDER

**Strategy patterns:**
- Multiple liabilities with same source → debt payoff strategy
- Percentage splits from income → budget allocation
- Threshold-based moves → savings/reserve building

## Use Cases

### Find Similar Users
```
Input:
- User profile (income, occupation, goal, type)

Process:
1. Encode profile features
2. Calculate similarity to all users in database
3. Filter by same USER_TYPE
4. Return top 10 most similar

Output:
- List of similar user IDs
- Their rule descriptions
- Common patterns among them
```

### Validate Proposed Plan
```
Input:
- Proposed automation plan
- User profile

Process:
1. Find similar successful users
2. Extract patterns from their rules
3. Compare proposed plan to common patterns
4. Flag deviations or missing elements

Output:
- Validation score
- Suggestions based on what similar users do
- Missing patterns to consider
```

### Extract Intent from Rules
```
Input:
- Raw rule descriptions (from itaytestfinal.csv)

Process:
1. Parse trigger types
2. Parse action types
3. Identify node relationships
4. Classify into goal categories

Output:
- Inferred PRODUCTGOAL
- Automation complexity score
- Primary patterns used
```

## Model Training Approach

### Supervised: Goal Prediction
Train model to predict PRODUCTGOAL from rules.
- Input: Rule descriptions (text)
- Output: Goal category
- Use for: Inferring intent when goal not stated

### Unsupervised: User Clustering
Cluster users by rule patterns + profile.
- Features: Profile + rule pattern features
- Method: K-means or hierarchical
- Use for: Finding user archetypes

### Similarity: User Matching
Build similarity index for fast lookup.
- Encode all users
- Index with FAISS or similar
- Query with new user profile

## Integration Points

### With sequence-map-generator
1. Generator requests similar users for a profile
2. Matcher returns top matches + their patterns
3. Generator incorporates patterns into plan

### With map-json-converter
1. Converter can request validation
2. Matcher checks if JSON structure matches successful patterns

## References

- [feature-engineering.md](references/feature-engineering.md) - Detailed feature encoding guide
