# Product Requirements Document: Sequence Map Generator Widget

## Overview

A top-of-funnel lead generation tool that allows prospective users to experience Sequence's financial automation capabilities without signing up. Users input their financial profile and goals, receive a personalized automation "map" with explanation, and can explore it in the Sequence playground.

---

## 1. Scope Definition

### 1.1 Boundaries

**In Scope:**
- Standalone full-page React application (opens in new tab from marketing site)
- Profile input form (structured fields)
- Free-text financial goal input
- Map generation via LLM backend
- Human-readable explanation of generated map
- Integration with Sequence playground via API (opens in new tab from results)
- Anonymous usage (no authentication)
- Client-side rate limiting (10 req/min per session)

**Out of Scope:**
- User accounts or authentication
- Saving/retrieving previous maps
- Editing generated maps in the widget
- Mobile native apps
- Backend API development (exists separately)
- Sequence playground modifications
- Analytics integration (deferred to V2)

### 1.2 MVP Definition

**Must Have (P0):**
- Profile input form with 5 fields
- Free-text goal input area
- Generate button with loading state
- Display generated map explanation
- "Open in Playground" button that redirects to playground with map ID

**Should Have (P1):**
- Form validation with helpful error messages
- Example prompts/suggestions
- Responsive design for mobile browsers

**Won't Have (V1):**
- Map visualization within widget
- Social sharing
- Multiple map comparison
- Chat-style refinement of maps

### 1.3 Assumptions

- Backend API endpoint exists and returns map ID after accepting JSON
- Sequence playground accepts `?id=<map_id>` query parameter
- LLM generator (llm_generator.py) is deployed as API service
- Target users arrive from paid ads with basic financial automation awareness

---

## 2. Requirements

### 2.1 Functional Requirements

#### FR-1: Profile Input Form
- **FR-1.1**: Display dropdown for USER_TYPE with options: INDIVIDUAL, BUSINESS
- **FR-1.2**: Display dropdown for ANNUALINCOME with ranges: UNDER_25K, BETWEEN_25K_AND_50K, BETWEEN_50K_AND_100K, BETWEEN_100K_AND_250K, OVER_250K
- **FR-1.3**: Display dropdown for AGE_GROUP with options: 18-24, 25-34, 35-44, 45-54, 55-64, 65+
- **FR-1.4**: Display dropdown for OCCUPATION with options: EMPLOYED, SELF_EMPLOYED, FREELANCER, GIG_WORKER, BUSINESS_OWNER, RETIRED, STUDENT
- **FR-1.5**: Display multi-select for PRODUCTGOAL with options: DEBT_PAYOFF, SAVINGS, BUDGETING, TAX_PLANNING, INVESTMENT, EMERGENCY_FUND
- **FR-1.6**: All fields required except PRODUCTGOAL (defaults to BUDGETING if empty)

#### FR-2: Financial Goal Input
- **FR-2.1**: Display textarea for free-text financial goal description
- **FR-2.2**: Minimum 20 characters, maximum 1000 characters
- **FR-2.3**: Placeholder text shows example: "I want to pay off my credit cards while saving for emergencies..."

#### FR-3: Map Generation
- **FR-3.1**: "Generate My Map" button triggers generation
- **FR-3.2**: Button disabled while form invalid or generation in progress
- **FR-3.3**: Display loading spinner with "Creating your personalized plan..." message
- **FR-3.4**: Handle API errors with user-friendly message and retry option

#### FR-4: Results Display
- **FR-4.1**: Display human-readable explanation of the generated map
- **FR-4.2**: Explanation includes: what accounts are created, what rules/automations are set up, why this approach matches their goals
- **FR-4.3**: Display "Open in Playground" button prominently
- **FR-4.4**: Button opens playground.getsequence.io/?id=<generated_id> in new tab

#### FR-5: Reset Flow
- **FR-5.1**: "Start Over" button returns to form with fields cleared
- **FR-5.2**: Back browser navigation returns to form with previous values preserved

### 2.2 Non-Functional Requirements

#### NFR-1: Performance
- Page load under 2 seconds on 3G connection
- Map generation feedback (loading state) within 200ms of click
- Total generation time under 30 seconds (API timeout)

#### NFR-2: Accessibility
- WCAG 2.1 AA compliance
- All form fields have proper labels
- Keyboard navigation support
- Screen reader compatible
- Color contrast ratio minimum 4.5:1

#### NFR-3: Browser Support
- Chrome 90+, Firefox 88+, Safari 14+, Edge 90+
- Mobile browsers: iOS Safari 14+, Chrome Android 90+

#### NFR-4: Security
- No PII collection or storage
- HTTPS only
- CSP headers configured
- Input sanitization on all text fields

#### NFR-5: Rate Limiting
- 10 requests per minute per session (client-side enforcement)
- Display friendly message when limit reached: "Please wait a moment before generating another map"
- Countdown timer showing when next request available

---

## 3. Prioritization (MoSCoW)

| Feature | Priority | Rationale |
|---------|----------|-----------|
| Profile form (5 fields) | Must | Core input mechanism |
| Free-text goal input | Must | Enables personalization |
| Generate button + loading | Must | Core action |
| Explanation display | Must | Value demonstration |
| Playground redirect (new tab) | Must | Conversion path |
| Form validation | Should | UX quality |
| Example prompts | Should | Reduces friction |
| Mobile responsive | Should | ~40% traffic expected |
| Error handling | Should | Production readiness |
| Rate limiting (10/min) | Should | Abuse prevention |
| Analytics events | Won't | Deferred to V2 |
| A/B test variants | Won't | Post-launch |

---

## 4. Acceptance Criteria

### AC-1: Profile Form Submission
```
GIVEN a user on the widget page
WHEN they select valid values for all required fields
AND enter a goal description of 20+ characters
AND click "Generate My Map"
THEN the form submits and shows loading state
```

### AC-2: Successful Map Generation
```
GIVEN a submitted form with valid inputs
WHEN the API returns successfully with a map ID
THEN display the map explanation text
AND display "Open in Playground" button
AND the button links to playground.getsequence.io/?id=<returned_id>
```

### AC-3: API Error Handling
```
GIVEN a submitted form
WHEN the API returns an error or times out
THEN display friendly error message: "We couldn't generate your map. Please try again."
AND display "Try Again" button that re-submits with same values
AND log error details to console for debugging
```

### AC-4: Form Validation
```
GIVEN a user with empty required fields
WHEN they click "Generate My Map"
THEN highlight invalid fields with red border
AND display field-specific error messages
AND prevent form submission
```

### AC-5: Playground Redirect
```
GIVEN a successfully generated map
WHEN user clicks "Open in Playground"
THEN open new browser tab
AND navigate to playground.getsequence.io/?id=<map_id>
AND original widget tab remains on results view
```

### AC-6: Mobile Responsive
```
GIVEN a user on a mobile device (viewport < 768px)
WHEN they view the widget
THEN all form fields are full-width stacked
AND buttons are touch-friendly (min 44px height)
AND text is readable without zooming
```

### AC-7: Rate Limiting
```
GIVEN a user who has made 10 requests in the last minute
WHEN they click "Generate My Map"
THEN display message: "Please wait a moment before generating another map"
AND show countdown timer to next available request
AND disable the generate button until cooldown expires
```

---

## 5. Technical Specification

### 5.1 System Context

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Marketing Site │     │   Map Generator │     │    Sequence     │
│    (Widget)     │────▶│      API        │────▶│   Playground    │
│   React/TS      │     │  (LLM Backend)  │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       ▲
        │                       │                       │
        └───────────────────────┴───────────────────────┘
                    (redirect with map ID)
```

### 5.2 API Contract

**Request: POST /api/generate-map**
```typescript
interface GenerateMapRequest {
  profile: {
    USER_TYPE: 'INDIVIDUAL' | 'BUSINESS';
    ANNUALINCOME: string;
    AGE_GROUP: string;
    OCCUPATION: string;
    PRODUCTGOAL: string[];
  };
  prompt: string; // Free-text goal description
}
```

**Response: 200 OK**
```typescript
interface GenerateMapResponse {
  id: string;           // UUID for playground URL
  explanation: string;  // Human-readable map description
  map: object;          // Full map JSON (not displayed, for debugging)
}
```

**Response: 4xx/5xx Error**
```typescript
interface ErrorResponse {
  error: string;
  message: string;
}
```

### 5.3 Component Architecture

```
src/
├── components/
│   ├── MapGeneratorWidget/
│   │   ├── index.tsx           # Main widget container
│   │   ├── ProfileForm.tsx     # Form with 5 profile fields
│   │   ├── GoalInput.tsx       # Free-text textarea
│   │   ├── GenerateButton.tsx  # Submit button with loading
│   │   ├── ResultsDisplay.tsx  # Explanation + playground link
│   │   └── ErrorMessage.tsx    # Error state component
│   └── ui/
│       ├── Select.tsx          # Styled dropdown
│       ├── MultiSelect.tsx     # Multi-select for goals
│       ├── Textarea.tsx        # Styled textarea
│       └── Button.tsx          # Styled button
├── hooks/
│   └── useMapGenerator.ts      # API call + state management
├── types/
│   └── index.ts                # TypeScript interfaces
└── styles/
    └── theme.ts                # Purple/white color tokens
```

### 5.4 Data Model

```typescript
interface ProfileFormData {
  userType: 'INDIVIDUAL' | 'BUSINESS';
  annualIncome: AnnualIncomeRange;
  ageGroup: AgeGroup;
  occupation: Occupation;
  productGoals: ProductGoal[];
  goalDescription: string;
}

type AnnualIncomeRange =
  | 'UNDER_25K'
  | 'BETWEEN_25K_AND_50K'
  | 'BETWEEN_50K_AND_100K'
  | 'BETWEEN_100K_AND_250K'
  | 'OVER_250K';

type AgeGroup = '18-24' | '25-34' | '35-44' | '45-54' | '55-64' | '65+';

type Occupation =
  | 'EMPLOYED'
  | 'SELF_EMPLOYED'
  | 'FREELANCER'
  | 'GIG_WORKER'
  | 'BUSINESS_OWNER'
  | 'RETIRED'
  | 'STUDENT';

type ProductGoal =
  | 'DEBT_PAYOFF'
  | 'SAVINGS'
  | 'BUDGETING'
  | 'TAX_PLANNING'
  | 'INVESTMENT'
  | 'EMERGENCY_FUND';
```

### 5.5 Hosting

**Recommended: Vercel (Free Tier)**
- Zero-config React deployment
- Automatic HTTPS
- Global CDN
- Easy custom domain setup
- GitHub integration for CI/CD

**Alternative: GitHub Pages**
- Free static hosting
- Requires build step configuration
- Custom domain via CNAME

---

## 6. Design Specification

### 6.1 Color Palette

| Token | Value | Usage |
|-------|-------|-------|
| primary | #7C3AED (Purple 600) | Buttons, links, accents |
| primary-hover | #6D28D9 (Purple 700) | Button hover state |
| background | #FFFFFF | Page background |
| surface | #F9FAFB (Gray 50) | Form container |
| text-primary | #111827 (Gray 900) | Headings, body |
| text-secondary | #6B7280 (Gray 500) | Labels, placeholders |
| error | #DC2626 (Red 600) | Validation errors |
| success | #059669 (Green 600) | Success states |

### 6.2 Typography

- **Headings**: Inter, 600 weight
- **Body**: Inter, 400 weight
- **Scale**: 14px base, 1.5 line-height

### 6.3 Layout

- Max-width: 480px for form container
- Padding: 24px on desktop, 16px on mobile
- Field spacing: 16px vertical gap
- Border radius: 8px for cards, 6px for inputs

### 6.4 States

**Button States:**
- Default: Purple background, white text
- Hover: Darker purple, subtle shadow
- Loading: Spinner icon, "Generating..." text, disabled
- Disabled: Gray background, reduced opacity

**Input States:**
- Default: Gray border, white background
- Focus: Purple border, subtle purple shadow
- Error: Red border, red error text below
- Filled: Slightly darker background

---

## 7. Success Metrics

| Metric | Target | Measurement (V2 with analytics) |
|--------|--------|-------------|
| Generation success rate | >90% | % of submissions that return valid map |
| Playground click-through | >40% | % of generated maps opened in playground |
| Avg generation time | <15s | API response time P95 |

*Note: Detailed funnel metrics (load rate, form completion) deferred to V2 analytics integration.*

---

## 8. Risks and Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| LLM generates invalid JSON | High | Medium | Validation layer + retry logic |
| API rate limits hit | Medium | Low | Queue system + rate limiting UI |
| Users confused by output | Medium | Medium | Clear explanation formatting |
| Mobile form difficult | Medium | Medium | Extensive mobile testing |

---

## 9. Decisions Log

| Question | Decision | Rationale |
|----------|----------|-----------|
| API Authentication | None required | Public top-of-funnel tool, no sensitive data |
| Rate Limiting | 10 requests/minute per IP | Prevents abuse while allowing genuine exploration |
| Analytics | Out of scope (V1) | Focus on core functionality first |
| Widget Format | Full standalone page | Opens in new tab from marketing site; playground also opens in new tab from results |

---

## 10. Glossary

| Term | Definition |
|------|------------|
| Map | A Sequence automation configuration defining accounts, rules, and money flows |
| Node | An account or container in a map (PORT, POD, DEPOSITORY_ACCOUNT, etc.) |
| Rule | An automation trigger and action set that moves money between nodes |
| Playground | Sequence's interactive map visualization tool |
| Avalanche | Debt payoff strategy prioritizing highest interest rate first |
| Snowball | Debt payoff strategy prioritizing smallest balance first |
| Profit First | Business allocation strategy: Profit → Owner Pay → Tax → Operating |

---

*Document Version: 1.0*
*Last Updated: 2026-01-28*
*Author: Claude (AI Assistant)*
