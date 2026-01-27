---
name: financial-planner
description: Financial planning domain expertise for money allocation decisions. Use when determining: (1) What percentage of income should go to savings, debt, or investments, (2) How to prioritize financial goals based on age, income, occupation, or family status, (3) Emergency fund sizing, (4) Debt payoff strategies (avalanche vs snowball), (5) Tax reserve calculations for self-employed/business owners. Provides evidence-based allocation rules and heuristics.
---

# Financial Planner

Provides financial planning logic to inform automated money allocation.

## Core Framework: The Allocation Hierarchy

Priority order (fund in sequence):
1. **Minimum debt payments** - Always first
2. **Small emergency buffer** - $1,000 starter
3. **Employer 401k match** - Free money, never skip
4. **High-interest debt** - >7% APR
5. **Full emergency fund** - 3-6 months expenses
6. **Retirement accounts** - Max tax-advantaged
7. **Other goals** - House, education, etc.

## Age-Based Allocation Guidelines

See [references/age-allocations.md](references/age-allocations.md) for detailed breakdowns.

**Quick reference:**
| Age | Savings | Debt | Investments | Notes |
|-----|---------|------|-------------|-------|
| 18-24 | 10-15% | Aggressive | 5-10% | Build habits, starter emergency fund |
| 25-34 | 15-20% | High priority | 15-20% | Peak debt payoff years, start retirement |
| 35-44 | 15% | Moderate | 20-25% | Balance growth and security |
| 45-54 | 10% | Low | 25-30% | Catch-up contributions, reduce risk |
| 55+ | 5-10% | Minimal | 15-20% | Preserve capital, prepare distribution |

## Income-Based Rules

See [references/income-rules.md](references/income-rules.md) for detailed breakdowns.

**Emergency fund sizing:**
- Under $50k income: 3 months expenses
- $50k-$100k: 4 months expenses
- $100k-$250k: 5 months expenses
- Over $250k: 6 months expenses

**Tax reserve (self-employed/business):**
- Base rate: 25-30% of gross income
- High earners ($250k+): 35-40%
- Adjust for state taxes, deductions

## Occupation-Based Strategies

See [references/occupation-strategies.md](references/occupation-strategies.md) for details.

**Income stability affects allocation:**
| Stability | Emergency Fund | Debt Aggression | Investment Risk |
|-----------|----------------|-----------------|-----------------|
| High (salaried, govt) | 3 months | Aggressive | Higher |
| Medium (professional) | 4 months | Moderate | Moderate |
| Variable (freelance, sales) | 6 months | Conservative | Lower |
| Seasonal (construction, tourism) | 6+ months | Very conservative | Lower |

## Family Status Adjustments

- **Single, no dependents**: Higher risk tolerance, aggressive debt payoff
- **Married, no kids**: Coordinate benefits, joint emergency fund
- **With children**: Increase emergency fund +1 month per child, add education savings
- **Single parent**: Conservative approach, larger emergency buffer

## Business Owner Specifics

See [references/business-allocation.md](references/business-allocation.md) for Profit First and other frameworks.

**Recommended allocation (Profit First method):**
| Revenue | Profit | Owner Pay | Tax | Operating |
|---------|--------|-----------|-----|-----------|
| <$250k | 5% | 50% | 15% | 30% |
| $250k-$500k | 10% | 35% | 15% | 40% |
| $500k-$1M | 15% | 20% | 15% | 50% |
| >$1M | 20% | 10% | 15% | 55% |

## Debt Payoff Strategies

**Avalanche (mathematically optimal):**
- Pay minimums on all debts
- Extra payments to highest interest rate first
- Best for: disciplined, motivated by math

**Snowball (psychologically effective):**
- Pay minimums on all debts
- Extra payments to smallest balance first
- Best for: need quick wins, multiple small debts

**When to use which:**
- High-interest debt (>15% APR): Always avalanche
- Similar rates (<3% difference): Snowball for motivation
- Large balances with high rates: Hybrid approach
