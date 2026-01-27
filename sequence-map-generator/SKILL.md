---
name: sequence-map-generator
description: Generate financial automation plans for Sequence users. Use when: (1) A user provides a prompt describing their financial goals, (2) Creating a new automation map from scratch, (3) Suggesting automations based on user profile (income, occupation, goals), (4) Explaining what automations would help achieve a specific financial outcome. This is the main orchestrator that produces human-readable plans.
---

# Sequence Map Generator

Generate personalized financial automation plans based on user goals and profile.

## Overview

Sequence is a financial automation platform ("Zapier for money"). Users connect accounts and create "maps" - visual flows of automated money movements between accounts, pods (virtual envelopes), and liabilities.

This skill takes a user's prompt and profile, then generates a human-readable automation plan that can later be converted to the Sequence JSON format.

## Input Requirements

To generate a plan, gather:

1. **User prompt** - What they want to achieve (e.g., "Help me pay off my credit cards faster")
2. **Profile data** (if available):
   - Annual income bracket
   - Occupation type
   - Primary goal (from: DEBT_PAYMENT, AUTOMATE_MY_BUDGETING, ORGANIZE_FINANCING, PAY_OFF_DEBT, PROFIT_FIRST, MAXIMIZE_SAVINGS, VISUALIZE_AND_TRACK_FINANCES, OPTIMIZE_CASH_FLOW, SAVE_FOR_TAXES, AUTOMATE_MY_BILLS, AUTOMATE_MY_INVESTMENTS, MAINTAIN_CONTROL, MANAGE_BUSINESS_PAYMENTS)
   - User type: INDIVIDUAL or BUSINESS
   - Connected accounts count

## Plan Generation Process

1. **Identify primary goal** from user prompt
2. **Match to automation patterns** - See [patterns.md](references/patterns.md)
3. **Consider user profile** for personalization
4. **Use pattern matcher** - Invoke user-pattern-matcher skill to find similar successful users
5. **Generate human-readable plan** with:
   - Recommended node structure (accounts, pods)
   - Rules with triggers and actions
   - Explanation of why each automation helps

## Output Format

Generate plans in this structure:

```
## Financial Automation Plan

### Goal
[Primary goal derived from user prompt]

### Recommended Structure

**Income Sources (Ports/Accounts):**
- [List entry points for money]

**Allocation Pods:**
- [List virtual envelopes needed]

**Liabilities:**
- [List debts/bills to automate]

### Automation Rules

1. **[Rule Name]**
   - Trigger: [When this happens]
   - Action: [What money moves where]
   - Why: [Brief explanation]

2. **[Rule Name]**
   ...

### Implementation Notes
[Any special considerations, order of setup, etc.]
```

## Core Concepts

### Node Types
- **PORT** - Entry point for external income (deposits, payments received)
- **POD** - Virtual envelope for allocating funds (like digital envelopes)
- **DEPOSITORY_ACCOUNT** - Connected bank account (checking, savings)
- **LIABILITY_ACCOUNT** - Debt account (credit card, loan)

### Trigger Types
- **INCOMING_FUNDS** - When money arrives at a node
- **SCHEDULED** - Cron-based timing (e.g., "0 0 1 * *" for 1st of month)
- **BALANCE_THRESHOLD** - When balance reaches a condition

### Action Types
- **PERCENTAGE** - Move X% of funds
- **FIXED** - Move exact dollar amount
- **REMAINDER** - Move everything left after other actions
- **TOP_UP** - Fill destination up to target amount
- **AVALANCHE** - Debt payoff prioritizing highest interest
- **SNOWBALL** - Debt payoff prioritizing smallest balance
- **NEXT_PAYMENT_MINIMUM** - Pay minimum due on liability
- **TOTAL_AMOUNT_DUE** - Pay full statement balance

## Goal-to-Pattern Quick Reference

| User Goal | Primary Pattern | Key Actions |
|-----------|-----------------|-------------|
| Pay off debt | Debt Payoff | AVALANCHE or SNOWBALL |
| Budget automation | Income Allocation | PERCENTAGE splits to pods |
| Bill automation | Bill Pay | FIXED amounts, SCHEDULED triggers |
| Save for taxes | Tax Reserve | PERCENTAGE to tax pod |
| Maximize savings | Savings First | PERCENTAGE to savings, REMAINDER to spending |
| Business cash flow | Profit First | PERCENTAGE splits: profit, tax, opex |

## References

- [patterns.md](references/patterns.md) - Detailed automation patterns by goal
- [node-types.md](references/node-types.md) - Complete node type reference

## Integration with Other Skills

After generating a human-readable plan:
1. Use **user-pattern-matcher** to validate against successful user patterns
2. Use **map-json-converter** to transform plan into Sequence JSON format
