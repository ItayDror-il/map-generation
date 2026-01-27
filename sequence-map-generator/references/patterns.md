# Automation Patterns

## Table of Contents
- [Debt Payoff Patterns](#debt-payoff-patterns)
- [Budget Allocation Patterns](#budget-allocation-patterns)
- [Business Patterns](#business-patterns)
- [Savings Patterns](#savings-patterns)
- [Bill Automation Patterns](#bill-automation-patterns)

## Debt Payoff Patterns

### Avalanche Method
Prioritize paying off highest-interest debt first. Mathematically optimal.

**Structure:**
```
Income Port → Debt Payment Pod → [Multiple Liabilities with AVALANCHE action]
```

**Rules:**
1. When funds received at Income: PERCENTAGE (X%) to Debt Payment Pod
2. When funds received at Debt Payment Pod: AVALANCHE to all liabilities (same priority group)

**Best for:** Users focused on minimizing total interest paid

### Snowball Method
Prioritize paying off smallest balance first. Psychologically motivating.

**Structure:**
```
Income Port → Debt Payment Pod → [Multiple Liabilities with SNOWBALL action]
```

**Rules:**
1. When funds received at Income: PERCENTAGE (X%) to Debt Payment Pod
2. When funds received at Debt Payment Pod: SNOWBALL to all liabilities

**Best for:** Users who need motivation from quick wins

### Hybrid: Minimums + Extra to Target
Pay minimums on all debts, then extra toward one target.

**Structure:**
```
Income → Bill Pay Pod → [Liabilities: NEXT_PAYMENT_MINIMUM]
                     → Target Debt Pod → [One Liability: REMAINDER]
```

**Rules:**
1. Income → FIXED amounts to Bill Pay Pod (sum of minimums + buffer)
2. Bill Pay Pod → NEXT_PAYMENT_MINIMUM to each liability
3. Extra income → Target Debt Pod → REMAINDER to target liability

## Budget Allocation Patterns

### Percentage-Based Budget (50/30/20 style)
Split income by percentage into categories.

**Structure:**
```
Income Port → Router Pod → Needs Pod (50%)
                        → Wants Pod (30%)
                        → Savings Pod (20%)
```

**Rules:**
1. When Income receives funds: 100% to Router Pod
2. When Router receives funds:
   - PERCENTAGE 50% to Needs
   - PERCENTAGE 30% to Wants
   - PERCENTAGE 20% to Savings

### Envelope System
Allocate fixed amounts to spending categories.

**Structure:**
```
Income → [Multiple Category Pods with FIXED amounts]
      → Overflow/Savings Pod (REMAINDER)
```

**Rules:**
1. When Income receives funds:
   - FIXED $X to Groceries Pod
   - FIXED $Y to Gas Pod
   - FIXED $Z to Entertainment Pod
   - REMAINDER to Savings Pod

### Zero-Based Budget
Every dollar has a job. Nothing left unallocated.

**Structure:**
```
Income → Router → [All category pods]
              → Final Sweep Pod (REMAINDER catches anything left)
```

**Key:** Use REMAINDER as final action to catch rounding/overflow

## Business Patterns

### Profit First
Based on Mike Michalowicz's methodology.

**Structure:**
```
Income Deposit → Operating Account → Profit Pod (5-20%)
                                  → Owner Pay Pod (50%)
                                  → Tax Pod (15%)
                                  → OpEx Pod (REMAINDER)
```

**Rules:**
1. When Income Deposit receives: 100% to Operating Account
2. When Operating Account receives:
   - PERCENTAGE 5-20% to Profit (based on revenue)
   - PERCENTAGE ~50% to Owner Pay
   - PERCENTAGE ~15% to Tax
   - REMAINDER to OpEx

**Percentages scale with business maturity**

### Sweep to Operating + Reserves
Consolidate income, maintain reserves.

**Structure:**
```
[Multiple Income Sources] → Central Router → Operating Account
                                          → Tax Reserve (%)
                                          → Emergency Fund (TOP_UP)
```

**Rules:**
1. All income sources: 100% to Central Router
2. Router:
   - PERCENTAGE X% to Tax Reserve
   - TOP_UP Emergency Fund to target (e.g., $10,000)
   - REMAINDER to Operating Account

### Bill Payment Segregation
Separate bill money from spending money.

**Structure:**
```
Income → Bills Pod → [Individual bill pods with FIXED amounts]
      → Spending Pod (REMAINDER)
```

**Each bill pod triggers payment when funded**

## Savings Patterns

### Pay Yourself First
Savings taken before anything else.

**Structure:**
```
Income → Savings Pod (PERCENTAGE, first priority)
      → Everything Else (REMAINDER)
```

**Rules:**
1. When Income receives: PERCENTAGE 20% to Savings (priority 0)
2. REMAINDER to Spending/Bills (priority 1)

### Goal-Based Savings
Multiple savings goals with targets.

**Structure:**
```
Income → Router → Emergency Fund (TOP_UP to $10,000)
              → Vacation Fund (FIXED $200)
              → House Fund (PERCENTAGE 10%)
              → General Savings (REMAINDER)
```

**Rules use TOP_UP for goals with targets, FIXED for consistent contributions**

### Overflow Savings
Save anything above a threshold.

**Structure:**
```
Checking Account ← When balance > $5,000: Move excess to Savings
```

**Rules:**
1. Trigger: BALANCE_THRESHOLD on Checking (greaterThan $5,000)
2. Action: Move funds to Savings (excess above threshold)

## Bill Automation Patterns

### Due-Date Based Payments
Schedule payments around due dates.

**Structure:**
```
Bill Holding Pod → [Liabilities with SCHEDULED triggers]
```

**Rules:**
1. Income → FIXED to Bill Holding Pod (total of all bills)
2. SCHEDULED (cron for each due date): Pay from Bill Holding to each liability

**Example crons:**
- `0 0 1 * *` - 1st of month
- `0 0 15 * *` - 15th of month
- `0 0 * * FRI` - Every Friday

### Statement Balance Automation
Pay full balance when statement closes.

**Structure:**
```
Bill Pod → Credit Card (TOTAL_AMOUNT_DUE)
```

**Rules:**
1. When Bill Pod receives funds: TOTAL_AMOUNT_DUE to Credit Card

### Minimum + Extra Pattern
Ensure minimums, apply extra strategically.

**Structure:**
```
Bill Pod → Card 1 (NEXT_PAYMENT_MINIMUM, priority 0)
        → Card 2 (NEXT_PAYMENT_MINIMUM, priority 0)
        → Target Card (REMAINDER, priority 1)
```

## Pattern Selection by User Profile

### By Income Level
- **< $50K**: Focus on bill automation, debt snowball
- **$50K-$100K**: Percentage budgets, savings goals
- **$100K-$250K**: Tax reserves, investment automation
- **> $250K**: Profit first (if business), multi-goal savings

### By Primary Goal
- **DEBT_PAYMENT / PAY_OFF_DEBT**: Avalanche or Snowball patterns
- **AUTOMATE_MY_BUDGETING**: Percentage or envelope patterns
- **PROFIT_FIRST**: Business profit first pattern
- **MAXIMIZE_SAVINGS**: Pay yourself first + overflow
- **SAVE_FOR_TAXES**: Tax reserve pattern
- **OPTIMIZE_CASH_FLOW**: Sweep patterns, threshold-based moves

### By User Type
- **INDIVIDUAL**: Personal budget, debt payoff, savings goals
- **BUSINESS**: Profit first, payroll reserves, tax segregation
