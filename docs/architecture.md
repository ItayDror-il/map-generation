# Sequence Map Generator - Architecture

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND                                 │
│  React + TypeScript (localhost:3000)                            │
│                                                                  │
│  ┌─────────────────┐                                            │
│  │ useMapGenerator │ ─── Single hook, single API call           │
│  └────────┬────────┘                                            │
│           │                                                      │
│           │  POST /api/generate-map                             │
│           │  { profile, prompt }                                │
│           ▼                                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP (CORS)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         BACKEND                                  │
│  FastAPI (localhost:8000)                                       │
│                                                                  │
│  ┌─────────────────┐     ┌──────────────────┐                  │
│  │   server.py     │────▶│  llm_generator   │                  │
│  │   (API layer)   │     │  (ML + Claude)   │                  │
│  └────────┬────────┘     └──────────────────┘                  │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────┐     ┌──────────────────┐                  │
│  │ generate_       │     │ Sequence GraphQL │                  │
│  │ explanation()   │     │ Admin API        │                  │
│  └─────────────────┘     └────────┬─────────┘                  │
│                                   │                              │
│  Returns: { id, explanation, map }│                              │
└───────────────────────────────────┼──────────────────────────────┘
                                    │
                                    │ GraphQL mutation
                                    ▼
                    ┌───────────────────────────────┐
                    │   Sequence Playground         │
                    │   playground.getsequence.io   │
                    │   /?id=<generated_id>         │
                    └───────────────────────────────┘
```

## Data Flow

1. **User fills form** → Frontend collects profile + prompt
2. **Frontend calls API** → `POST /api/generate-map`
3. **Backend generates map** → LLM creates JSON structure
4. **Backend saves to Sequence** → GraphQL mutation creates playground map
5. **Backend returns ID** → Real Sequence playground ID
6. **User clicks "Open in Playground"** → Opens `playground.getsequence.io/?id=<id>`

## Key Files

| Layer | File | Purpose |
|-------|------|---------|
| **Frontend** | `frontend/src/hooks/useMapGenerator.ts` | Single API hook |
| **Frontend** | `frontend/.env.development` | API URL config |
| **Backend** | `api/server.py` | FastAPI + Sequence GraphQL integration |
| **Backend** | `llm_generator.py` | LLM + ML generation logic |

## Separation Guarantees

1. **Frontend only knows one URL**: `VITE_API_URL` - backend can change completely
2. **Backend returns stable contract**: `{ id, explanation, map }` - tweak algorithms freely
3. **No shared code**: Frontend is React/TS, backend is Python
4. **Real playground IDs**: Maps are saved to Sequence, not local storage

## API Contract

### POST /api/generate-map

**Request:**
```json
{
  "profile": {
    "USER_TYPE": "INDIVIDUAL" | "BUSINESS",
    "ANNUALINCOME": "UNDER_25K" | "BETWEEN_25K_AND_50K" | ...,
    "AGE_GROUP": "18-24" | "25-34" | ...,
    "OCCUPATION": "EMPLOYED" | "SELF_EMPLOYED" | ...,
    "PRODUCTGOAL": ["SAVINGS", "DEBT_PAYOFF", ...]
  },
  "prompt": "User's free-text financial goal (20-1000 chars)"
}
```

**Response:**
```json
{
  "id": "cc5f75b7-2659-4b16-a0b9-583a4d4e7313",
  "explanation": "Human-readable summary of the map",
  "map": { /* Full Sequence map JSON */ }
}
```

The `id` is a real Sequence playground ID - open `playground.getsequence.io/?id=<id>` to view.

## Sequence Integration

The backend calls Sequence's Admin GraphQL API to persist maps:

```graphql
mutation {
  adminCreatePlaygroundMapFromJson(
    name: "Map Generator: user prompt..."
    mapJson: "{ stringified map JSON }"
    ownerFingerprint: "map-generator-api"
  )
}
```

**Endpoint:** `https://app.getsequence.io/api/admin-graphql`
**Auth Header:** `x-admin-key: <admin key>`

## Running

```bash
# Install dependencies
make install

# Run both (or separately)
make dev        # Both frontend + backend
make backend    # API only (localhost:8000)
make frontend   # React only (localhost:3000)

# Test end-to-end
python scripts/test-api.py
```

## Environment Variables

| Variable | Where | Purpose |
|----------|-------|---------|
| `ANTHROPIC_API_KEY` | Backend | Claude API access |
| `SEQUENCE_ADMIN_KEY` | Backend | Sequence Admin GraphQL (optional, has default) |
| `VITE_API_URL` | Frontend | Backend URL (default: http://localhost:8000) |

## Production Deployment

1. **Backend**: Deploy `api/server.py` to any Python hosting (Railway, Render, AWS)
2. **Frontend**: Build with `make build`, deploy `frontend/dist/` to Vercel/Netlify
3. **Configure**:
   - Set `ANTHROPIC_API_KEY` on backend
   - Set `VITE_API_URL` to production backend URL before frontend build
