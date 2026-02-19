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

## Architecture

- `pga_sim/datagolf_client.py`: DataGolf API client.
- `pga_sim/service.py`: Data extraction, normalization, feature synthesis.
- `pga_sim/simulation.py`: Hybrid Markov + Monte Carlo engine (vectorized NumPy).
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
- Run the model with one click.
- View top players and probabilities (win/top 3/top 5/top 10) in table + chart form.
- Inspect seasonal diagnostics in the UI status line (events/players loaded per season and matched active-field players).

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
- `GET /events/upcoming?tour=pga&limit=12`
- `POST /simulate`

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/simulate" \
  -H "Content-Type: application/json" \
  -d '{
    "tour": "pga",
    "simulations": 20000,
    "seed": 123,
    "cut_size": 70,
    "mean_reversion": 0.1,
    "enable_seasonal_form": true,
    "baseline_season": 2025,
    "current_season": 2026,
    "seasonal_form_weight": 0.35,
    "current_season_weight": 0.6,
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
    - `current_season_weight` controls baseline-vs-current emphasis.
    - `form_delta_weight` controls the lift/penalty from current minus baseline season form.
- Monte Carlo draws trajectories through the Markov chain for all players, all rounds.
- PGA cut logic is applied after round 2 (top `cut_size`, including ties).

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
