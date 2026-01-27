# Sequence Map JSON Schema

## Complete Schema Reference

### Map Object
```typescript
interface SequenceMap {
  nodes: Node[];
  rules: Rule[];
  viewport: Viewport;
}
```

### Node Object
```typescript
interface Node {
  id: string;                    // UUID
  type: NodeType;
  subtype?: NodeSubtype | null;
  name: string;
  balance: number;               // In cents
  icon: string;                  // Emoji
  position: Position;
  target_amount?: number | null; // For PODs with TOP_UP goals
  metadata?: {
    original_name?: string;
  };
}

type NodeType =
  | 'POD'
  | 'PORT'
  | 'DEPOSITORY_ACCOUNT'
  | 'LIABILITY_ACCOUNT';

type NodeSubtype =
  | 'CHECKING'
  | 'SAVINGS'
  | 'CREDIT_CARD'
  | 'LOAN'
  | 'LINE_OF_CREDIT';

interface Position {
  x: number;
  y: number;
}
```

### Rule Object
```typescript
interface Rule {
  id: string;                    // UUID
  name?: string;                 // Optional descriptive name
  sourceId: string;              // Node UUID that triggers this rule
  trigger: Trigger;
  steps: Step[];
  enabled?: boolean;             // Default true
}
```

### Trigger Object
```typescript
interface Trigger {
  type: TriggerType;
  sourceId: string;              // Same as rule sourceId
  cron?: string | null;          // Only for SCHEDULED
  condition?: BalanceCondition;  // Only for BALANCE_THRESHOLD
}

type TriggerType =
  | 'INCOMING_FUNDS'
  | 'SCHEDULED'
  | 'BALANCE_THRESHOLD';

interface BalanceCondition {
  operator: 'greaterThan' | 'lessThan' | 'equals';
  amountInCents: number;
}
```

### Step Object
```typescript
interface Step {
  actions: Action[];
  condition?: StepCondition;     // Optional conditional logic
}

interface StepCondition {
  type: 'transferAmount';
  operator: 'greaterThan' | 'lessThan';
  amountInCents: number;
}
```

### Action Object
```typescript
interface Action {
  type: ActionType;
  sourceId: string;              // Node funds come from
  destinationId: string;         // Node funds go to
  amountInCents: number;         // For FIXED actions
  amountInPercentage: number;    // For PERCENTAGE actions (0-100)
  groupIndex: number;            // Execution order (0 = first)
  limit?: number | null;         // Max amount in cents
  upToEnabled?: boolean | null;  // For TOP_UP and similar
}

type ActionType =
  | 'PERCENTAGE'
  | 'FIXED'
  | 'REMAINDER'
  | 'TOP_UP'
  | 'AVALANCHE'
  | 'SNOWBALL'
  | 'NEXT_PAYMENT_MINIMUM'
  | 'TOTAL_AMOUNT_DUE'
  | 'ROUND_DOWN';
```

### Viewport Object
```typescript
interface Viewport {
  x: number;
  y: number;
  zoom: number;                  // Typically 0.5 - 2.0
}
```

## Action Type Details

### PERCENTAGE
Moves a percentage of incoming funds.

```json
{
  "type": "PERCENTAGE",
  "amountInPercentage": 30,
  "amountInCents": 0
}
```

### FIXED
Moves an exact dollar amount.

```json
{
  "type": "FIXED",
  "amountInCents": 50000,
  "amountInPercentage": 0
}
```
Note: $500.00 = 50000 cents

### REMAINDER
Moves all remaining funds after other actions.

```json
{
  "type": "REMAINDER",
  "amountInCents": 0,
  "amountInPercentage": 0
}
```
Always use as last priority (highest groupIndex).

### TOP_UP
Fills destination up to a target amount.

```json
{
  "type": "TOP_UP",
  "amountInCents": 0,
  "amountInPercentage": 0,
  "limit": 1000000,
  "upToEnabled": true
}
```
This fills the destination until it reaches $10,000.

### AVALANCHE
Debt payoff prioritizing highest interest rate.

```json
{
  "type": "AVALANCHE",
  "amountInPercentage": 100,
  "groupIndex": 0
}
```
All liabilities in the same groupIndex compete via avalanche logic.

### SNOWBALL
Debt payoff prioritizing smallest balance.

```json
{
  "type": "SNOWBALL",
  "amountInPercentage": 100,
  "groupIndex": 0
}
```
All liabilities in the same groupIndex compete via snowball logic.

### NEXT_PAYMENT_MINIMUM
Pays the minimum due on a liability.

```json
{
  "type": "NEXT_PAYMENT_MINIMUM",
  "upToEnabled": true
}
```

### TOTAL_AMOUNT_DUE
Pays the full statement balance on a liability.

```json
{
  "type": "TOTAL_AMOUNT_DUE",
  "amountInCents": 0,
  "amountInPercentage": 0
}
```

### ROUND_DOWN
Rounds balance down to nearest threshold, moves excess.

```json
{
  "type": "ROUND_DOWN",
  "amountInCents": 0,
  "amountInPercentage": 0
}
```

## Group Index Behavior

Actions execute in groupIndex order:
- `groupIndex: 0` executes first
- `groupIndex: 1` executes second
- etc.

Actions with the same groupIndex execute "together":
- For PERCENTAGE: All run on the original amount
- For AVALANCHE/SNOWBALL: All compete in the same debt payoff calculation

## Conditional Rules

Rules can have conditions on the transfer amount:

```json
{
  "trigger": {
    "type": "INCOMING_FUNDS",
    "sourceId": "node-income"
  },
  "steps": [{
    "condition": {
      "type": "transferAmount",
      "operator": "greaterThan",
      "amountInCents": 12000000
    },
    "actions": [...]
  }]
}
```
This rule only triggers when incoming amount > $120,000.

## Balance Threshold Triggers

```json
{
  "trigger": {
    "type": "BALANCE_THRESHOLD",
    "sourceId": "node-checking",
    "condition": {
      "operator": "greaterThan",
      "amountInCents": 500000
    }
  }
}
```
Triggers when checking balance exceeds $5,000.

## Validation Rules

1. **Node IDs must be unique**
2. **All sourceId/destinationId must reference existing nodes**
3. **PERCENTAGE actions should sum to <= 100% per groupIndex**
4. **REMAINDER should be last (highest groupIndex)**
5. **LIABILITY_ACCOUNT nodes should have subtype**
6. **SCHEDULED triggers must have valid cron**
7. **AVALANCHE/SNOWBALL destinations must be LIABILITY_ACCOUNT**

## Default Values

When creating new nodes/rules, use these defaults:

```json
{
  "balance": 0,
  "icon": "💰",
  "position": {"x": 500, "y": 300},
  "viewport": {"x": 300, "y": 100, "zoom": 0.9},
  "enabled": true,
  "amountInCents": 0,
  "amountInPercentage": 0,
  "groupIndex": 0,
  "limit": null,
  "upToEnabled": null
}
```
