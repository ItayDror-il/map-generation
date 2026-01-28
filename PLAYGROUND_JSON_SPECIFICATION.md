# Playground JSON Specification

Technical documentation for structuring JSON output for the Sequence Playground system.

## Table of Contents

1. [Overview](#overview)
2. [Root Structure: PlaygroundMapState](#root-structure-playgroundmapstate)
3. [Nodes](#nodes)
4. [Rules](#rules)
5. [Triggers](#triggers)
6. [Conditions](#conditions)
7. [Actions](#actions)
8. [Limits](#limits)
9. [Viewport](#viewport)
10. [Validation Rules & Business Logic](#validation-rules--business-logic)
11. [Complete JSON Examples](#complete-json-examples)

---

## Overview

The Playground JSON structure represents a financial automation map that defines:
- **Nodes**: Financial accounts (checking, savings, credit cards, investments, etc.)
- **Rules**: Automation rules that transfer money between nodes based on triggers and conditions
- **Viewport**: Visual positioning information for the map display

---

## Root Structure: PlaygroundMapState

The top-level structure for a playground map state.

```json
{
  "id": "string (optional - auto-generated if not provided)",
  "name": "string (required)",
  "ownerFingerprint": "string (required)",
  "rules": [PlaygroundRule],
  "nodes": [PlaygroundNode],
  "viewport": ViewportMapState
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | No | Unique identifier. Auto-generated (UUID) if not provided when creating |
| `name` | string | Yes | Display name of the playground map |
| `ownerFingerprint` | string | Yes | Identifier for the owner of this map |
| `rules` | PlaygroundRule[] | Yes | Array of automation rules (can be empty) |
| `nodes` | PlaygroundNode[] | Yes | Array of account nodes (can be empty) |
| `viewport` | ViewportMapState | Yes | Viewport positioning and zoom |

---

## Nodes

Nodes represent financial accounts or containers in the playground.

### PlaygroundNode Structure

```json
{
  "id": "string (required)",
  "name": "string (required)",
  "icon": "string (required)",
  "balance": 0,
  "type": "DEPOSITORY_ACCOUNT",
  "subtype": "CHECKING",
  "position": {
    "x": 0,
    "y": 0
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Unique identifier for the node |
| `name` | string | Yes | Display name (e.g., "Chase Checking") |
| `icon` | string | Yes | Icon identifier/name for display |
| `balance` | integer | Yes | Balance in **cents** (e.g., 10000 = $100.00) |
| `type` | PlaygroundNodeType | Yes | Primary type of the node |
| `subtype` | PlaygroundNodeSubtype | No | More specific categorization |
| `position` | NodePosition | Yes | X/Y coordinates for visual placement |

### PlaygroundNodeType (Enum)

| Value | Description | Use Case |
|-------|-------------|----------|
| `PORT` | Income source/port | Represents where money comes from (paycheck, etc.) |
| `POD` | Savings container | Internal savings buckets |
| `DEPOSITORY_ACCOUNT` | Bank deposit account | Checking/savings accounts |
| `LIABILITY_ACCOUNT` | Debt/credit account | Credit cards, loans, mortgages |
| `INVESTMENT_ACCOUNT` | Investment account | 401k, IRA, brokerage accounts |
| `DESTINATION_ACCOUNT` | External destination | Accounts for outgoing transfers |

### PlaygroundNodeSubtype (Enum)

| Value | Compatible Type | Description |
|-------|-----------------|-------------|
| `SAVINGS` | DEPOSITORY_ACCOUNT | Savings account |
| `CHECKING` | DEPOSITORY_ACCOUNT | Checking account |
| `CREDIT_CARD` | LIABILITY_ACCOUNT | Credit card |
| `MORTGAGE` | LIABILITY_ACCOUNT | Mortgage loan |
| `PERSONAL_LOAN` | LIABILITY_ACCOUNT | Personal loan |
| `HOME_LOAN` | LIABILITY_ACCOUNT | Home equity loan |
| `STUDENT_LOAN` | LIABILITY_ACCOUNT | Student loan |
| `AUTO_LOAN` | LIABILITY_ACCOUNT | Auto loan |
| `BUSINESS_LOAN` | LIABILITY_ACCOUNT | Business loan |
| `PENSION` | INVESTMENT_ACCOUNT | Pension account |
| `IRA` | INVESTMENT_ACCOUNT | Individual retirement account |
| `BROKERAGE` | INVESTMENT_ACCOUNT | Brokerage account |
| `EDUCATION_SAVINGS` | INVESTMENT_ACCOUNT | 529 or education savings |
| `DESTINATION` | DESTINATION_ACCOUNT | Generic external destination |

### NodePosition Structure

```json
{
  "x": 150.5,
  "y": 200.0
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `x` | float | Yes | Horizontal position on canvas |
| `y` | float | Yes | Vertical position on canvas |

---

## Rules

Rules define automated money movement logic.

### PlaygroundRule Structure

```json
{
  "id": "string (required)",
  "sourceId": "string (required)",
  "trigger": PlaygroundTrigger,
  "steps": [PlaygroundRuleStep]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Unique identifier for the rule |
| `sourceId` | string | Yes | ID of the source node (must match a node ID) |
| `trigger` | PlaygroundTrigger | Yes | What triggers this rule |
| `steps` | PlaygroundRuleStep[] | Yes | Array of execution steps (at least 1) |

### PlaygroundRuleStep Structure

```json
{
  "conditions": ChainableRuleCondition,
  "actions": [PlaygroundRuleAction]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `conditions` | ChainableRuleCondition | No | Optional conditions for this step |
| `actions` | PlaygroundRuleAction[] | Yes | Actions to execute (at least 1) |

---

## Triggers

Triggers define when a rule should execute.

### PlaygroundTrigger Structure

```json
{
  "type": "INCOMING_FUNDS",
  "sourceId": "string (required)",
  "cron": "string (optional)"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | PlaygroundTriggerType | Yes | Type of trigger |
| `sourceId` | string | Yes | ID of the source node |
| `cron` | string | Conditional | Cron expression (required for SCHEDULED) |

### PlaygroundTriggerType (Enum)

| Value | Description | Cron Required |
|-------|-------------|---------------|
| `INCOMING_FUNDS` | Triggers when money arrives at source | No |
| `SCHEDULED` | Triggers on a schedule | Yes |

### Cron Expression Examples

| Schedule | Cron Expression |
|----------|-----------------|
| Daily at midnight | `0 0 * * *` |
| Weekly on Monday | `0 0 * * MON` |
| Weekly on Friday | `0 0 * * FRI` |
| Monthly on 1st | `0 0 1 * *` |
| Monthly on 15th | `0 0 15 * *` |
| 1st and 15th | `00 0 1,15 * *` |
| Last day of month | `0 0 L * *` |
| Custom day (e.g., 5th) | `0 0 5 * *` |

---

## Conditions

Conditions control whether a rule step executes.

### ChainableRuleCondition Structure

Conditions use a recursive structure that supports logical AND/OR chaining.

**Option 1: Single Condition**
```json
{
  "condition": {
    "fact": "BALANCE",
    "operator": "GREATER_THAN",
    "value": "10000",
    "valueFact": null,
    "valueFactParams": null,
    "params": {
      "accountId": "account-123"
    }
  }
}
```

**Option 2: AND Conditions (all must be true)**
```json
{
  "all": [
    { "condition": { ... } },
    { "condition": { ... } }
  ]
}
```

**Option 3: OR Conditions (any must be true)**
```json
{
  "any": [
    { "condition": { ... } },
    { "condition": { ... } }
  ]
}
```

**Option 4: Empty (no conditions)**
```json
{
  "all": []
}
```

### RuleCondition Structure

```json
{
  "fact": "BALANCE",
  "operator": "GREATER_THAN",
  "value": "10000",
  "valueFact": "BALANCE",
  "valueFactParams": {
    "accountId": "other-account-id"
  },
  "params": {
    "accountId": "source-account-id"
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `fact` | RuleAvailableFact | Yes | What to evaluate |
| `operator` | RuleOperator | Yes | Comparison operator |
| `value` | string | Conditional | Static value to compare against (in cents for money) |
| `valueFact` | RuleAvailableFact | Conditional | Dynamic value - compare against another fact |
| `valueFactParams` | RuleConditionParams | Conditional | Parameters for valueFact |
| `params` | RuleConditionParams | Conditional | Parameters for the main fact |

**Note:** Either `value` OR `valueFact` must be provided, not both.

### RuleAvailableFact (Enum)

| Value | Description | Requires params | Notes |
|-------|-------------|-----------------|-------|
| `TRANSFER_AMOUNT` | Amount of incoming transfer | No | Only valid for INCOMING_FUNDS trigger |
| `BALANCE` | Account balance | Yes | Needs accountId, portId, or podId |
| `DATE` | Day of month (1-31) | No | Value is day number as string |
| `LAST_DAY_OF_MONTH` | Is it the last day | No | Boolean check |
| `NEXT_PAYMENT_MINIMUM_AMOUNT` | Minimum payment due | Yes | For liability accounts |
| `LAST_STATEMENT_BALANCE` | Last statement balance | Yes | For liability accounts |

### RuleOperator (Enum)

| Value | Symbol | Description |
|-------|--------|-------------|
| `EQUALS` | `=` | Equal to |
| `NOT_EQUALS` | `!=` | Not equal to |
| `GREATER_THAN` | `>` | Greater than |
| `LESS_THAN` | `<` | Less than |
| `GREATER_THAN_OR_EQUAL` | `>=` | Greater than or equal |
| `LESS_THAN_OR_EQUAL` | `<=` | Less than or equal |
| `CONTAINS` | - | Contains (for strings) |
| `NOT_CONTAINS` | - | Does not contain |

### RuleConditionParams Structure

Only **one** of these should be provided:

```json
{
  "portId": "string",
  "accountId": "string",
  "podId": "string"
}
```

---

## Actions

Actions define what happens when a rule executes.

### PlaygroundRuleAction Structure

```json
{
  "type": "FIXED",
  "amountInCents": 10000,
  "amountInPercentage": null,
  "sourceId": "source-node-id",
  "destinationId": "destination-node-id",
  "groupIndex": 0,
  "upToEnabled": false,
  "limit": null
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | PlaygroundRuleActionType | Yes | Type of action |
| `amountInCents` | integer | Conditional | Amount in cents (for FIXED, TOP_UP, ROUND_DOWN) |
| `amountInPercentage` | float | Conditional | Percentage 0-100 (for PERCENTAGE types) |
| `sourceId` | string | Yes | Source node ID |
| `destinationId` | string | Yes | Destination node ID |
| `groupIndex` | integer | Yes | Action group index (0-based) |
| `upToEnabled` | boolean | No | Enable "up to" constraint |
| `limit` | RuleActionLimit | No | Optional spending limit |

### PlaygroundRuleActionType (Enum)

| Type | Amount Field | Valid Destinations | Description |
|------|--------------|-------------------|-------------|
| `FIXED` | amountInCents | All | Transfer exact dollar amount |
| `PERCENTAGE` | amountInPercentage | All | Transfer % of incoming (INCOMING_FUNDS) or balance (SCHEDULED) |
| `TOP_UP` | amountInCents | POD, DEPOSITORY_ACCOUNT | Top up destination to target balance |
| `ROUND_DOWN` | amountInCents | All | Round source balance down to nearest amount |
| `NEXT_PAYMENT_MINIMUM` | N/A | LIABILITY_ACCOUNT | Pay minimum amount due |
| `TOTAL_AMOUNT_DUE` | N/A | LIABILITY_ACCOUNT | Pay full current balance |
| `PERCENTAGE_LIABILITY_BALANCE` | amountInPercentage | LIABILITY_ACCOUNT | Pay % of liability balance |
| `SNOWBALL` | amountInCents OR amountInPercentage | LIABILITY_ACCOUNT | Debt snowball (smallest balance first) |
| `AVALANCHE` | amountInCents OR amountInPercentage | LIABILITY_ACCOUNT | Debt avalanche (highest interest first) |

### Action Type Details

#### FIXED
Transfer a fixed dollar amount.
```json
{
  "type": "FIXED",
  "amountInCents": 50000,
  "amountInPercentage": null
}
```

#### PERCENTAGE
Transfer a percentage of funds.
- For INCOMING_FUNDS: percentage of the incoming transfer amount
- For SCHEDULED: percentage of the source account balance

```json
{
  "type": "PERCENTAGE",
  "amountInCents": null,
  "amountInPercentage": 50.0
}
```

#### TOP_UP
Bring destination balance up to target amount.
```json
{
  "type": "TOP_UP",
  "amountInCents": 100000,
  "amountInPercentage": null
}
```

#### ROUND_DOWN
Round source balance down to nearest specified amount.
```json
{
  "type": "ROUND_DOWN",
  "amountInCents": 10000,
  "amountInPercentage": null
}
```

#### NEXT_PAYMENT_MINIMUM
Pay the minimum payment due on a liability account.
```json
{
  "type": "NEXT_PAYMENT_MINIMUM",
  "amountInCents": null,
  "amountInPercentage": null
}
```

#### TOTAL_AMOUNT_DUE
Pay the entire current balance on a liability account.
```json
{
  "type": "TOTAL_AMOUNT_DUE",
  "amountInCents": null,
  "amountInPercentage": null
}
```

#### PERCENTAGE_LIABILITY_BALANCE
Pay a percentage of the liability balance.
```json
{
  "type": "PERCENTAGE_LIABILITY_BALANCE",
  "amountInCents": null,
  "amountInPercentage": 25.0
}
```

#### SNOWBALL / AVALANCHE
Debt payoff strategies. These can use either fixed amount or percentage.

**Snowball**: Pays debts from smallest to largest balance.
**Avalanche**: Pays debts from highest to lowest interest rate.

```json
{
  "type": "SNOWBALL",
  "amountInCents": 50000,
  "amountInPercentage": null
}
```

or

```json
{
  "type": "AVALANCHE",
  "amountInCents": null,
  "amountInPercentage": 100.0
}
```

### Action Groups

Actions within a step are organized into groups via `groupIndex`:
- Actions with the same `groupIndex` are in the same group
- Multiple destinations can be in one group (for PERCENTAGE type actions)
- Percentages within a group must sum to ≤ 100%

**Example: Split 100% between two accounts**
```json
{
  "actions": [
    {
      "type": "PERCENTAGE",
      "amountInPercentage": 60,
      "destinationId": "savings-id",
      "groupIndex": 0
    },
    {
      "type": "PERCENTAGE",
      "amountInPercentage": 40,
      "destinationId": "investment-id",
      "groupIndex": 0
    }
  ]
}
```

---

## Limits

Limits constrain how much can be transferred.

### RuleActionLimit Structure

```json
{
  "type": "PER_MONTH",
  "amountInCents": 100000
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | RuleActionLimitType | Yes | Time period for the limit |
| `amountInCents` | integer | Yes | Maximum amount in cents |

### RuleActionLimitType (Enum)

| Value | Description |
|-------|-------------|
| `PER_TRANSFER` | Maximum per single transfer |
| `PER_WEEK` | Maximum per week (cumulative) |
| `PER_MONTH` | Maximum per month (cumulative) |
| `PER_YEAR` | Maximum per year (cumulative) |

---

## Viewport

Controls the visual display of the map.

### ViewportMapState Structure

```json
{
  "x": 0,
  "y": 0,
  "zoom": 1.0
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `x` | float | Yes | Horizontal pan position |
| `y` | float | Yes | Vertical pan position |
| `zoom` | float | Yes | Zoom level (typical range: 0.1 - 10) |

---

## Validation Rules & Business Logic

### 1. Cyclic Rule Prevention

**Rule**: For INCOMING_FUNDS triggers, you cannot create rules that form cycles.

If Rule A transfers from Node X → Node Y, you **cannot** have another INCOMING_FUNDS rule that transfers from Node Y → Node X.

This prevents infinite loops where incoming funds trigger transfers back and forth.

### 2. Percentage Constraints

- Within an action group, all PERCENTAGE actions must sum to ≤ 100%
- If the first action in a group is PERCENTAGE, additional actions can be added to the same group
- Non-percentage actions start a new group

### 3. Trigger-Specific Restrictions

**INCOMING_FUNDS**:
- Can use TRANSFER_AMOUNT fact in conditions
- PERCENTAGE actions calculate from incoming transfer amount

**SCHEDULED**:
- Cannot use TRANSFER_AMOUNT fact in conditions
- PERCENTAGE actions calculate from source account balance
- Requires valid cron expression

### 4. Destination-Specific Actions

| Action Type | Valid Destinations |
|-------------|-------------------|
| FIXED, PERCENTAGE, ROUND_DOWN | All node types |
| TOP_UP | POD, DEPOSITORY_ACCOUNT only |
| NEXT_PAYMENT_MINIMUM | LIABILITY_ACCOUNT only |
| TOTAL_AMOUNT_DUE | LIABILITY_ACCOUNT only |
| PERCENTAGE_LIABILITY_BALANCE | LIABILITY_ACCOUNT only |
| SNOWBALL, AVALANCHE | LIABILITY_ACCOUNT only |

### 5. Amount Field Requirements

| Action Type | amountInCents | amountInPercentage |
|-------------|---------------|-------------------|
| FIXED | Required (> 0) | null |
| PERCENTAGE | null | Required (0-100) |
| TOP_UP | Required (≥ 0) | null |
| ROUND_DOWN | Required (≥ 0) | null |
| NEXT_PAYMENT_MINIMUM | null | null |
| TOTAL_AMOUNT_DUE | null | null |
| PERCENTAGE_LIABILITY_BALANCE | null | Required (0-100) |
| SNOWBALL | One or the other | One or the other |
| AVALANCHE | One or the other | One or the other |

### 6. Condition Limits

- Maximum 2 conditions per step (UI enforced)
- Conditions are chained with AND or OR (not mixed in same level)

### 7. Value Constraints

| Field | Constraint |
|-------|------------|
| amountInCents | Integer, typically > 0 |
| amountInPercentage | Float, 0-100 |
| balance | Integer in cents |
| DATE condition value | 1-31 (day of month) |
| zoom | Float, typically 0.1-10 |

---

## Complete JSON Examples

### Example 1: Simple Incoming Funds Rule

Transfer 50% of any incoming funds to savings.

```json
{
  "name": "My Budget",
  "ownerFingerprint": "user-fingerprint-123",
  "nodes": [
    {
      "id": "port-1",
      "name": "Paycheck",
      "icon": "briefcase",
      "balance": 0,
      "type": "PORT",
      "position": { "x": 100, "y": 100 }
    },
    {
      "id": "checking-1",
      "name": "Chase Checking",
      "icon": "bank",
      "balance": 500000,
      "type": "DEPOSITORY_ACCOUNT",
      "subtype": "CHECKING",
      "position": { "x": 300, "y": 100 }
    },
    {
      "id": "savings-1",
      "name": "Emergency Fund",
      "icon": "piggy-bank",
      "balance": 1000000,
      "type": "POD",
      "subtype": "SAVINGS",
      "position": { "x": 500, "y": 100 }
    }
  ],
  "rules": [
    {
      "id": "rule-1",
      "sourceId": "port-1",
      "trigger": {
        "type": "INCOMING_FUNDS",
        "sourceId": "port-1"
      },
      "steps": [
        {
          "conditions": { "all": [] },
          "actions": [
            {
              "type": "PERCENTAGE",
              "amountInCents": null,
              "amountInPercentage": 50,
              "sourceId": "port-1",
              "destinationId": "savings-1",
              "groupIndex": 0,
              "upToEnabled": false,
              "limit": null
            }
          ]
        }
      ]
    }
  ],
  "viewport": {
    "x": 0,
    "y": 0,
    "zoom": 1
  }
}
```

### Example 2: Scheduled Rule with Conditions

Pay credit card on the 15th, but only if balance is greater than $1000.

```json
{
  "name": "Credit Card Payments",
  "ownerFingerprint": "user-fingerprint-456",
  "nodes": [
    {
      "id": "checking-1",
      "name": "Main Checking",
      "icon": "bank",
      "balance": 300000,
      "type": "DEPOSITORY_ACCOUNT",
      "subtype": "CHECKING",
      "position": { "x": 100, "y": 200 }
    },
    {
      "id": "credit-1",
      "name": "Chase Sapphire",
      "icon": "credit-card",
      "balance": -250000,
      "type": "LIABILITY_ACCOUNT",
      "subtype": "CREDIT_CARD",
      "position": { "x": 400, "y": 200 }
    }
  ],
  "rules": [
    {
      "id": "rule-cc-payment",
      "sourceId": "checking-1",
      "trigger": {
        "type": "SCHEDULED",
        "sourceId": "checking-1",
        "cron": "0 0 15 * *"
      },
      "steps": [
        {
          "conditions": {
            "all": [
              {
                "condition": {
                  "fact": "BALANCE",
                  "operator": "GREATER_THAN",
                  "value": "100000",
                  "params": {
                    "accountId": "checking-1"
                  }
                }
              }
            ]
          },
          "actions": [
            {
              "type": "TOTAL_AMOUNT_DUE",
              "amountInCents": null,
              "amountInPercentage": null,
              "sourceId": "checking-1",
              "destinationId": "credit-1",
              "groupIndex": 0,
              "upToEnabled": false,
              "limit": {
                "type": "PER_MONTH",
                "amountInCents": 500000
              }
            }
          ]
        }
      ]
    }
  ],
  "viewport": {
    "x": 0,
    "y": 0,
    "zoom": 1
  }
}
```

### Example 3: Multi-Action Split

Split incoming funds: 60% to checking, 30% to savings, 10% to investment.

```json
{
  "rules": [
    {
      "id": "rule-split",
      "sourceId": "port-1",
      "trigger": {
        "type": "INCOMING_FUNDS",
        "sourceId": "port-1"
      },
      "steps": [
        {
          "conditions": { "all": [] },
          "actions": [
            {
              "type": "PERCENTAGE",
              "amountInPercentage": 60,
              "sourceId": "port-1",
              "destinationId": "checking-1",
              "groupIndex": 0
            },
            {
              "type": "PERCENTAGE",
              "amountInPercentage": 30,
              "sourceId": "port-1",
              "destinationId": "savings-1",
              "groupIndex": 0
            },
            {
              "type": "PERCENTAGE",
              "amountInPercentage": 10,
              "sourceId": "port-1",
              "destinationId": "investment-1",
              "groupIndex": 0
            }
          ]
        }
      ]
    }
  ]
}
```

### Example 4: Debt Snowball Strategy

Pay off multiple credit cards using the snowball method.

```json
{
  "rules": [
    {
      "id": "rule-snowball",
      "sourceId": "checking-1",
      "trigger": {
        "type": "SCHEDULED",
        "sourceId": "checking-1",
        "cron": "0 0 1 * *"
      },
      "steps": [
        {
          "conditions": { "all": [] },
          "actions": [
            {
              "type": "SNOWBALL",
              "amountInCents": 50000,
              "amountInPercentage": null,
              "sourceId": "checking-1",
              "destinationId": "credit-card-1",
              "groupIndex": 0,
              "limit": {
                "type": "PER_MONTH",
                "amountInCents": 100000
              }
            },
            {
              "type": "SNOWBALL",
              "amountInCents": 50000,
              "amountInPercentage": null,
              "sourceId": "checking-1",
              "destinationId": "credit-card-2",
              "groupIndex": 0
            }
          ]
        }
      ]
    }
  ]
}
```

### Example 5: Conditional with OR Logic

Transfer if balance > $5000 OR it's the last day of the month.

```json
{
  "steps": [
    {
      "conditions": {
        "any": [
          {
            "condition": {
              "fact": "BALANCE",
              "operator": "GREATER_THAN",
              "value": "500000",
              "params": { "accountId": "checking-1" }
            }
          },
          {
            "condition": {
              "fact": "DATE",
              "operator": "EQUALS",
              "value": "31"
            }
          }
        ]
      },
      "actions": [
        {
          "type": "FIXED",
          "amountInCents": 100000,
          "sourceId": "checking-1",
          "destinationId": "savings-1",
          "groupIndex": 0
        }
      ]
    }
  ]
}
```

### Example 6: Dynamic Value Comparison

Transfer when checking balance exceeds savings balance.

```json
{
  "steps": [
    {
      "conditions": {
        "all": [
          {
            "condition": {
              "fact": "BALANCE",
              "operator": "GREATER_THAN",
              "valueFact": "BALANCE",
              "params": { "accountId": "checking-1" },
              "valueFactParams": { "accountId": "savings-1" }
            }
          }
        ]
      },
      "actions": [
        {
          "type": "FIXED",
          "amountInCents": 50000,
          "sourceId": "checking-1",
          "destinationId": "savings-1",
          "groupIndex": 0
        }
      ]
    }
  ]
}
```

---

## Quick Reference: Money Conversions

All monetary values in the JSON are stored in **cents** (integers).

| Display Value | JSON Value (cents) |
|---------------|-------------------|
| $1.00 | 100 |
| $10.00 | 1000 |
| $100.00 | 10000 |
| $1,000.00 | 100000 |
| $10,000.00 | 1000000 |

**Conversion formulas:**
- Dollars to cents: `dollars * 100`
- Cents to dollars: `cents / 100`

---

## Source File References

Key implementation files for reference:

| File | Purpose |
|------|---------|
| `packages/api/src/graphql/modules/playground/playground-types.ts` | TypeScript interfaces |
| `packages/api/src/graphql/modules/playground/typeDefs/playground.graphql` | GraphQL schema |
| `packages/api/src/graphql/modules/rules/typeDefs/rules.graphql` | Rule types and conditions |
| `packages/playground-webapp/src/components/RuleForm/RuleFormProvider/utils.ts` | Form transformation logic |
| `packages/playground-webapp/src/components/RuleForm/constants.ts` | Action type mappings |
| `packages/playground-webapp/src/components/RuleForm/RuleFormProvider/schemas.ts` | Validation schemas |
