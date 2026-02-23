# PGA Markov Simulator (Local macOS App)

This project runs a local PGA Tour event simulator using:
- DataGolf API data for player fields and priors.
- A Markov transition process for player round-to-round state evolution.
- Stochastic Monte Carlo sampling for probabilistic outcome generation.

Outputs are player-by-player probabilities for:
- `win`
- `top_3`
- `top_5`
- `top_10`

Defaults now prioritize high-resolution automation:
- `simulations=1,000,000`
- `current_season_weight=0.85`

## Architecture

- `pga_sim/datagolf_client.py`: DataGolf API client.
- `pga_sim/service.py`: Data extraction, normalization, feature synthesis.
- `pga_sim/simulation.py`: Hybrid Markov + Monte Carlo engine (vectorized NumPy).
- `pga_sim/learning.py`: Local SQLite learning store, outcome ingestion, and calibration retraining.
- `pga_sim/api.py`: FastAPI server and local GUI routes.
- `pga_sim/web/index.html`: Browser GUI entry page.
- `pga_sim/web/assets/app.js`: GUI event loading + simulation actions.
- `pga_sim/web/assets/styles.css`: GUI styling and responsive layout.
- `pga_sim/cli.py`: Optional CLI for quick scripted runs.

## Performance characteristics

- Vectorized simulation in NumPy for high-throughput runs (thousands+ of full tournaments).
- Precomputed per-player, per-state transition CDFs to reduce repeated compute inside loops.
- Reproducible runs via `seed`.

## 1. Setup

```bash
cd /Users/paulzip84/Documents/New\ project
python3 -m venv .venv
source .venv/bin/activate
python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel
python -m pip install ".[dev]"
cp .env.example .env
```

Then set your key in `.env`:

```bash
DATAGOLF_API_KEY=your_key_here
```

Authentication defaults to local open access (`APP_AUTH_MODE=none`).
For shared web deployment:
- quickest on Render domain: `APP_AUTH_MODE=basic_auth`
- strongest with SSO/policies: `APP_AUTH_MODE=cloudflare_access` (requires your own domain)

## 2. Run local GUI (recommended)

Start the server:

```bash
python -m pga_sim.api
```

Then open:

```text
http://127.0.0.1:8000
```

The GUI lets you:
- Load upcoming events by tour.
- See which events are currently simulatable (others appear as unavailable in the dropdown).
- Set simulations, cut size, mean reversion, and optional seed.
- Configure a seasonal-form blend using last season as baseline and current season as form delta.
- Sync completed event outcomes and retrain local probability calibration over time.
- Run the model with one click.
- View top players and probabilities (win/top 3/top 5/top 10) in table + chart form.
- Inspect seasonal diagnostics in the UI status line (events/players loaded per season and matched active-field players).
- Track per-event simulation snapshot versions (`v1`, `v2`, ...) for before-round and in-play comparisons.

If you prefer the installed command, run:

```bash
python -m pip install ".[dev]"
pga-sim-api
```

## 3. Optional CLI usage

List events:

```bash
pga-sim events --tour pga --limit 10
```

Run 25,000 simulations:

```bash
pga-sim simulate --tour pga --simulations 25000 --top 25 --seed 42
```

Run for a specific event:

```bash
pga-sim simulate --tour pga --event-id your_event_id --simulations 50000 --seed 7
```

Run with explicit seasonal blend controls:

```bash
pga-sim simulate \
  --tour pga \
  --simulations 30000 \
  --baseline-season 2025 \
  --current-season 2026 \
  --seasonal-form-weight 0.4 \
  --current-season-weight 0.65 \
  --form-delta-weight 0.3
```

## 4. API endpoints

Server runs on `http://127.0.0.1:8000`.

Endpoints:
- `GET /health`
- `GET /` (GUI)
- `GET /ui` (GUI alias)
- `GET /auth/me`
- `GET /events/upcoming?tour=pga&limit=12`
- `POST /simulate`
- `GET /learning/status?tour=pga`
- `POST /learning/sync-train`
- `GET /learning/event-trends?tour=pga&event_id=14`
- `GET /lifecycle/status?tour=pga`
- `POST /lifecycle/run?tour=pga`

When auth is enabled:
- `POST /learning/sync-train` requires `admin` role.
- All routes except `/health` require authentication by default.

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/simulate" \
  -H "Content-Type: application/json" \
  -d '{
    "tour": "pga",
    "resolution_mode": "fixed_cap",
    "simulations": 20000,
    "enable_adaptive_simulation": false,
    "seed": 123,
    "cut_size": 70,
    "mean_reversion": 0.1,
    "enable_seasonal_form": true,
    "baseline_season": 2025,
    "current_season": 2026,
    "seasonal_form_weight": 0.35,
    "current_season_weight": 0.85,
    "form_delta_weight": 0.25
  }'
```

## Model notes

- The Markov state is cumulative strokes relative to field baseline.
- Transition probabilities depend on player skill and current state:
  - `skill` sets baseline expected round delta.
  - `mean_reversion` dampens extreme hot/cold states.
- Seasonal baseline/form layer:
  - Loads historical rounds for `baseline_season` (default last year) and `current_season` (default current year).
  - Computes per-player season metrics and blends them into simulation skill:
    - `seasonal_form_weight` controls impact of seasonal blend versus current DataGolf priors.
    - `current_season_weight` is now a base prior that is adapted per player using starts, recent finishes, and baseline-season seasonality/hot-streak signals.
    - `form_delta_weight` controls extra lift/penalty from shrunk current-vs-baseline seasonal delta.
  - Start-count aware shrinkage:
    - Players with few or zero current-season starts are shrunk toward their baseline seasonal anchor.
    - This avoids incorrectly treating no-start players as "out of form".
  - Baseline seasonality:
    - Baseline event history is phase-weighted (early/mid/late season) to build a phase-specific anchor for each player.
    - Baseline win clustering feeds a hot-streak signal used in player-specific seasonal blending.
- Monte Carlo draws trajectories through the Markov chain for all players, all rounds.
- PGA cut logic is applied after round 2 (top `cut_size`, including ties).
- `top_3`, `top_5`, and `top_10` are literal finish-place probabilities (`finish_rank <= 3/5/10`), not field-percentile buckets.
- Self-learning loop:
  - Every simulation run is logged locally to SQLite (`.pga_sim_learning.sqlite3` by default).
  - For completed events, outcomes are pulled from DataGolf `historical-event-data/events`.
  - If official historical outcomes are temporarily delayed, the sync step can fallback to a completed live leaderboard snapshot as a provisional outcome source, then retrain and replace with official outcomes once published.
  - The app retrains per-market probability calibration (`win`, `top_3`, `top_5`, `top_10`) and applies it to future runs.
- Live probability movement:
  - Enable `Live Auto-Refresh` in the GUI to rerun in-play conditioned simulations on a cadence.
  - Expanded player rows show win-probability trend snapshots and deltas over the tournament.
- High-resolution defaults:
  - `resolution_mode=fixed_cap`
  - `simulations=1,000,000` cap
  - `min_simulations=250,000` (used when you switch to `auto_target`)
  - `ci_confidence=0.975`
  - `ci_half_width_target=0.0015`
  - `ci_top_n=15`
- Simulation snapshot versioning:
  - Every logged run stores an event-scoped `simulation_version` that increments per event-year (`v1`, `v2`, ...).
  - Versions are shown in the GUI summary card and trend snapshots to support pre-round vs. final-outcome review.
- Lifecycle automation (new):
  - A background lifecycle worker can automatically:
    - run leakage-safe historical backfill simulation + training in chronological order,
    - capture one consistent pre-event snapshot for the active event (snapshot type `pre_event`),
    - capture live in-play snapshots for the active event while play is ongoing,
    - sync outcomes when events complete,
    - retrain calibration when new outcomes are available.
  - Lifecycle status is visible in the GUI and through `/lifecycle/status`.
  - Manual override is available with `POST /lifecycle/run`.
  - The GUI now defaults to automation-focused mode (historical timeline + automation status, fewer manual controls).
  - The GUI auto-runs scheduled simulations on a cadence and shows explicit automation run status.

## Secure web deployment (recommended)

Deploying for friends is easiest with:
- Render (host FastAPI app)
- Basic Auth (works immediately on Render-provided domain)
- optional upgrade: Cloudflare Access (authentication + policy control)

### Render

1. Create a Python web service from this repo.
2. Build command:
   - `pip install --upgrade pip && pip install .`
3. Start command:
   - `uvicorn pga_sim.api:app --host 0.0.0.0 --port $PORT`
4. Set env vars:
   - `DATAGOLF_API_KEY=...`
   - `LEARNING_DATABASE_PATH=/var/data/pga_sim_learning.sqlite3` (if using persistent disk)
   - `APP_AUTH_MODE=basic_auth`
   - `APP_AUTH_BASIC_USERNAME=<your-username>`
   - `APP_AUTH_BASIC_PASSWORD=<long-random-password>`
   - `APP_AUTH_BASIC_ROLE=admin`
   - optional lifecycle automation controls:
     - `LIFECYCLE_AUTOMATION_ENABLED=true`
     - `LIFECYCLE_AUTOMATION_INTERVAL_SECONDS=600`
     - `LIFECYCLE_TOUR=pga`
     - `LIFECYCLE_PRE_EVENT_SIMULATIONS=1000000`
     - `SIMULATION_MAX_SYNC_SIMULATIONS=250000` (recommended for Render to avoid gateway timeouts)
     - `SIMULATION_MAX_BATCH_SIZE=2000` (recommended for Render Starter 512MB)
     - `LIFECYCLE_PRE_EVENT_SEED=20260223`
     - `LIFECYCLE_SYNC_MAX_EVENTS=40`
     - `LIFECYCLE_BACKFILL_ENABLED=true`
     - `LIFECYCLE_BACKFILL_BATCH_SIZE=25`

### Basic Auth quick start (no custom domain required)

Use your Render URL directly (example `https://golfsimulator.onrender.com`) with:
- `APP_AUTH_MODE=basic_auth`
- `APP_AUTH_BASIC_USERNAME=<username>`
- `APP_AUTH_BASIC_PASSWORD=<long-random-password>`
- `APP_AUTH_BASIC_ROLE=admin` (or `user`)

Browser requests will prompt for username/password and the GUI/API will stay protected.

### Optional upgrade: Cloudflare Access (requires custom domain)

If you want IdP-backed auth, per-user policies, and easier user revocation:
- Add a custom domain in Render.
- Put it behind Cloudflare DNS + Access.
- Then set:
   - `APP_AUTH_MODE=cloudflare_access`
   - `CLOUDFLARE_ACCESS_TEAM_DOMAIN=your-team.cloudflareaccess.com`
   - `CLOUDFLARE_ACCESS_AUDIENCE=<cloudflare-access-aud-tag>`
   - `AUTH_ALLOWED_EMAILS=friend1@email.com,friend2@email.com` (optional allowlist)
   - `AUTH_ALLOWED_EMAIL_DOMAINS=yourdomain.com` (optional domain allowlist)
   - `AUTH_ADMIN_EMAILS=you@email.com` (admin actions like retrain)

### Quick fallback: shared bearer token

If you want a simpler temporary setup:
- `APP_AUTH_MODE=shared_token`
- `APP_AUTH_SHARED_TOKEN=<long-random-token>`
- `APP_AUTH_SHARED_TOKEN_ROLE=admin` or `user`

Then clients send:

```bash
Authorization: Bearer <your-token>
```

## Testing

```bash
pytest -q
```

## Troubleshooting

- Page does not load at `http://127.0.0.1:8000`:
  - Confirm the server is running in another terminal:
    - `source .venv/bin/activate`
    - `pga-sim-api`
  - If port 8000 is busy, run:
    - `uvicorn pga_sim.api:app --host 127.0.0.1 --port 8001`
  - Then open `http://127.0.0.1:8001`.

- GUI opens but event loading/simulation fails with API key error:
  - Create `/Users/paulzip84/Documents/New project/.env`
  - Add:
    - `DATAGOLF_API_KEY=your_key_here`
  - Restart `pga-sim-api`.
