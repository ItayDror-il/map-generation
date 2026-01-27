# Test Prompts for Sequence Map Generator

## 1. Basic Budgeter (Individual, Entry-Level)
> "I just got my first job making $45,000. I want to save 15% of each paycheck and set aside money for rent ($1,200/month) and groceries."

**Profile:** Individual, 18-24, BETWEEN_25K_AND_50K

---

## 2. Aggressive Debt Payoff
> "I have a Chase Sapphire card with $8,000 balance at 22% APR and a Discover card with $3,500 at 18% APR. I want to crush this debt using the avalanche method while keeping $100/month for savings."

**Profile:** Individual, 25-34, BETWEEN_50K_AND_100K

---

## 3. Small Business Owner (Profit First)
> "I'm a freelance designer making about $120,000 annually. I need to set aside taxes, pay myself a consistent salary, and build a profit buffer."

**Profile:** Business, SELF_EMPLOYED, BETWEEN_100K_AND_250K

---

## 4. Emergency Fund Builder
> "I want to build a 6-month emergency fund. My monthly expenses are about $4,000. Send 25% of every paycheck to emergency savings until I hit my goal."

**Profile:** Individual, 35-44, BETWEEN_100K_AND_250K

---

## 5. Multiple Income Sources
> "I have a W-2 job and do Uber on weekends. My salary should go toward bills and savings. My Uber income should be 30% taxes and the rest goes to paying off my car loan."

**Profile:** Individual, 25-34, GIG_WORKER, BETWEEN_50K_AND_100K

---

## 6. Family with Kids (Complex)
> "We're a family of 4. I want to split our income: 20% to retirement, 10% to kids' college funds (529), pay the mortgage ($2,500), set aside grocery money ($800), and put the rest in checking for daily expenses."

**Profile:** Individual, 35-44, BETWEEN_100K_AND_250K

---

## 7. High Earner with Investment Focus
> "I make $300k and want to max out retirement accounts first, then funnel excess into a taxable brokerage account. Keep 2 months expenses liquid."

**Profile:** Individual, 45-54, EXECUTIVE_OR_MANAGER, OVER_250K

---

## 8. Variable Income Freelancer
> "My income is unpredictable - some months I make $15k, others $3k. I need to smooth out my spending by keeping 40% in a reserve account and only pulling a fixed $4,000 monthly for personal expenses."

**Profile:** Business, FREELANCER, BETWEEN_50K_AND_100K

---

## 9. Snowball Debt Strategy (Multiple Small Debts)
> "I have 4 credit cards with balances of $500, $1,200, $2,000, and $4,500. I want to use the snowball method to knock them out one by one while saving a tiny bit for emergencies."

**Profile:** Individual, 18-24, BETWEEN_25K_AND_50K

---

## 10. Near-Retirement Conservative
> "I'm 58 and want to shift to preservation mode. Take 15% of income for retirement catch-up contributions, keep 6 months expenses in savings, and minimize any aggressive debt payments."

**Profile:** Individual, 55-64, BETWEEN_100K_AND_250K

---

## Testing Tips

- **Edge cases**: Test prompts with unusual phrasing, typos, or vague goals like "help me with money"
- **Profile mismatches**: Try a prompt about business taxes with an Individual profile to see how it handles conflicts
- **Missing info**: Test with minimal profiles (just USER_TYPE) to check fallback behavior
- **Named accounts**: Include specific bank/card names (Chase, Amex, Wells Fargo) to test account recognition
