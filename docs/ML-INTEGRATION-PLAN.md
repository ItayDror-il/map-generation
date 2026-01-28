# ML Integration Plan for Map Generator

## Current State

The backend currently uses a **simplified flow**:
1. Frontend sends `profile + prompt`
2. Backend calls Claude directly with basic profile context
3. Claude generates map JSON
4. Backend saves to Sequence GraphQL API

## Target State

The ML-enriched flow:
1. Frontend sends `profile + prompt`
2. **ML finds similar users** → extracts patterns from their rule setups
3. **ML predicts goal** → classifies user intent (debt, savings, business, etc.)
4. Backend enriches Claude prompt with ML context
5. Claude generates map JSON (informed by similar user patterns)
6. Backend saves to Sequence GraphQL API

## ML Components

### Models (in `models/` directory)
| File | Size | Purpose |
|------|------|---------|
| `similarity_matcher.pkl` | 905KB | Find similar users by profile+rules features |
| `goal_classifier.pkl` | 7.8MB | Predict user goal (LightGBM) |
| `user_clusters.pkl` | 13KB | User archetypes (KMeans) |
| `user_vectors.npy` | 587KB | Pre-computed user feature vectors |
| `user_id_mapping.json` | 91KB | User ID lookup |
| `feature_config.json` | 4KB | Feature encoding config |

### Data Files (in root)
| File | Purpose |
|------|---------|
| `Itaytestfinal.csv` | User rules data (for pattern extraction) |
| `enrichment_data.csv` | User profile data |

### Dependencies
```
numpy
scikit-learn
lightgbm
polars
joblib
```

## Integration Options

### Option 1: Include Models in Git (Recommended for MVP)

**Approach:** Remove `models/` from `.gitignore` and commit directly.

**Pros:**
- Simple, works immediately
- No external dependencies
- Fast startup (models already on disk)

**Cons:**
- Increases repo size (~9MB)
- Model updates require git commits

**Steps:**
1. Remove `models/` from `.gitignore`
2. Add CSV files to git (or keep them gitignored and bundle in Docker)
3. Commit models and push
4. Update `api/server.py` to use `llm_generator.py`

### Option 2: Cloud Storage + Download on Startup

**Approach:** Store models in S3/GCS, download on Railway startup.

**Pros:**
- Keeps repo small
- Easy to update models without code changes

**Cons:**
- Adds startup latency (download time)
- Requires cloud storage setup
- Cold starts are slower

**Steps:**
1. Upload models to S3/GCS bucket
2. Add startup script to download models
3. Use environment variable for bucket URL
4. Cache downloaded models in Railway volume

### Option 3: Containerize with Models (Docker)

**Approach:** Build Docker image with models baked in.

**Pros:**
- Fast startup (no downloads)
- Reproducible deployments
- Works well with Railway

**Cons:**
- Larger Docker image
- Need to rebuild image for model updates

**Steps:**
1. Create `Dockerfile` that copies models
2. Push to container registry
3. Deploy container to Railway

## Recommended Implementation (Option 1)

### Step 1: Update Dependencies

Update `requirements.txt`:
```
# Backend API
fastapi>=0.109.0
uvicorn>=0.27.0
pydantic>=2.5.0
httpx>=0.26.0
anthropic>=0.18.0

# ML Dependencies
numpy>=1.24.0
scikit-learn>=1.4.0
lightgbm>=4.0.0
polars>=0.20.0
joblib>=1.3.0
```

### Step 2: Update .gitignore

Remove `models/` line from `.gitignore`:
```diff
- # Models (large files)
- models/
```

### Step 3: Copy Original llm_generator.py Back

The full `llm_generator.py` includes:
- `ContextBuilder` class - builds ML context from similar users
- `get_financial_advice()` - profile-based financial planning rules
- `LLMGenerator` class - orchestrates ML context + Claude API

### Step 4: Update server.py to Use LLMGenerator

```python
from llm_generator import LLMGenerator, validate_sequence_json

# Initialize on startup
generator = LLMGenerator(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# In generate_map endpoint:
result = generator.generate(prompt=request.prompt, profile=profile)
map_data = result["json_map"]
```

### Step 5: Handle Data Files

Option A: Include CSVs in git (simplest)
Option B: Load from environment-configured path
Option C: Make ML enrichment optional (fallback to simple mode)

### Step 6: Deploy

1. Commit all changes
2. Push to GitHub
3. Railway auto-deploys
4. Test end-to-end

## Fallback Strategy

If ML models fail to load, fall back to simple mode:

```python
try:
    generator = LLMGenerator(api_key=api_key)
except Exception as e:
    print(f"ML models not available: {e}")
    generator = None  # Use simple Claude-only mode
```

## Performance Considerations

- **Model Loading**: ~2-3 seconds on startup
- **Inference Time**: ~50-100ms per request (similarity + goal prediction)
- **Memory**: ~100MB for loaded models

## Next Steps

1. [ ] Choose integration option (recommend Option 1)
2. [ ] Update `.gitignore` and commit models
3. [ ] Test locally with full ML flow
4. [ ] Deploy to Railway and verify
5. [ ] Monitor performance and iterate
