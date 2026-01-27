# Node Types Reference

## Overview

Nodes are the building blocks of a Sequence map. Each node represents a place where money can be held or flow through.

## Node Types

### PORT
**Purpose:** Entry point for external income

**Use cases:**
- Direct deposit from employer
- Payment processor deposits (Stripe, Square, PayPal)
- Rental income
- Side gig income (DoorDash, Uber, freelance)

**Behavior:**
- Money flows IN from external sources
- Triggers INCOMING_FUNDS rules when deposits arrive
- Cannot hold a balance (pass-through)

**Naming conventions observed:**
- `[Source] Deposit` - "Payroll Deposit", "Stripe Deposit"
- `[Source] Income` - "Rental Income", "Side Gig Income"
- `Income - [Source]` - "Income - DoorDash"

### POD
**Purpose:** Virtual envelope for organizing funds

**Use cases:**
- Budget categories (Groceries, Entertainment, Gas)
- Bill holding (Rent Pod, Utilities Pod)
- Savings goals (Emergency Fund, Vacation)
- Routing/sweep accounts (Router Pod, Sweep Pod)

**Behavior:**
- Holds a virtual balance within a connected account
- Can have target amounts for TOP_UP actions
- Triggers rules when funds arrive or thresholds met

**Naming conventions observed:**
- Category names: "Groceries", "Entertainment"
- Bill names: "Rent", "Electric", "Phone"
- With prefixes: "P - " (Personal), "B - " (Business), "LH - " (entity initials)
- Functional: "Router", "Sweep - OPEX", "Buffer"

### DEPOSITORY_ACCOUNT
**Purpose:** Connected bank account

**Subtypes:**
- `CHECKING` - Primary transaction account
- `SAVINGS` - Savings account

**Use cases:**
- Main checking account
- High-yield savings
- Separate business account
- Payroll account

**Behavior:**
- Real bank account connected via Plaid
- Actual money movements happen here
- Can trigger rules based on balance thresholds

**Naming conventions observed:**
- Bank + type: "Chase Checking", "Ally Savings"
- Purpose: "Operating Account", "Tax Account", "Payroll"
- With last 4: "Chase - 1234"

### LIABILITY_ACCOUNT
**Purpose:** Debt account to pay off

**Subtypes:**
- `CREDIT_CARD` - Credit card
- `LOAN` - Personal loan, auto loan, mortgage
- `LINE_OF_CREDIT` - HELOC, business line

**Use cases:**
- Credit card payments
- Loan payments
- Debt payoff strategies

**Behavior:**
- Tracks balance owed
- Supports special actions: AVALANCHE, SNOWBALL, NEXT_PAYMENT_MINIMUM, TOTAL_AMOUNT_DUE
- Can be grouped for debt payoff strategies

**Naming conventions observed:**
- Card name: "Chase Sapphire", "Amex Gold"
- With last 4: "CC - BofA - 0731"
- Descriptive: "Car Loan", "Student Loan", "Mortgage"

## Node Properties

### Common Properties
```json
{
  "id": "unique-uuid",
  "type": "POD|PORT|DEPOSITORY_ACCOUNT|LIABILITY_ACCOUNT",
  "name": "Human readable name",
  "balance": 0,
  "position": {"x": 100, "y": 200}
}
```

### Type-Specific Properties

**POD:**
- `target_amount` - For TOP_UP actions, the goal balance

**DEPOSITORY_ACCOUNT:**
- `subtype` - "CHECKING" or "SAVINGS"

**LIABILITY_ACCOUNT:**
- `subtype` - "CREDIT_CARD", "LOAN", "LINE_OF_CREDIT"

## Node Naming Best Practices

### Personal Finance
- Use clear category names: "Groceries", "Gas", "Entertainment"
- Include due dates in bill pods: "15th - Rent", "1st - Car Payment"
- Prefix debt by type: "CC - ", "Loan - "

### Business Finance
- Prefix by entity: "BIZ - ", "[Company Initials] - "
- Separate P&L categories: "COGS", "OPEX", "Payroll"
- Clear revenue streams: "Stripe Deposits", "Invoice Payments"

### Hybrid (Personal + Business)
- Clear separation: "P - " for personal, "B - " for business
- Example: "P - Groceries", "B - Software Subscriptions"

## Common Node Structures

### Basic Personal Budget
```
Nodes:
- Paycheck (PORT)
- Checking (DEPOSITORY_ACCOUNT)
- Needs (POD)
- Wants (POD)
- Savings (POD)
- Credit Card (LIABILITY_ACCOUNT)
```

### Debt Payoff Focus
```
Nodes:
- Income (PORT)
- Router (POD)
- Bills (POD)
- Debt Payment (POD)
- CC 1, CC 2, CC 3 (LIABILITY_ACCOUNT)
- Emergency Fund (POD)
```

### Small Business
```
Nodes:
- Revenue Deposit (PORT)
- Operating (DEPOSITORY_ACCOUNT)
- Profit (POD)
- Owner Pay (POD)
- Tax Reserve (POD)
- Payroll (DEPOSITORY_ACCOUNT)
- OpEx (POD)
```
