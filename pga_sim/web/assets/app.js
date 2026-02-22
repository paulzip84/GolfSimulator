const state = {
  events: [],
  latestResult: null,
  expandedPlayerKey: null,
  latestLearningStatus: null,
  eventTrends: null,
  trendPlayerByKey: {},
  autoRefreshTimerId: null,
  simulationInFlight: false,
};

const TABLE_TOOLTIPS = {
  rank: {
    layman: "Where this player ranks overall in this simulation run.",
    source: "Calculated by this app from simulated tournament outcomes.",
    calculation: "Players are sorted from highest to lowest Win %.",
  },
  player: {
    layman: "The golfer's name.",
    source: "DataGolf current-week field and prediction feeds.",
    calculation: "Name is matched across field, prediction, and decomposition data.",
  },
  current_position: {
    layman: "Where the player currently sits on the live leaderboard.",
    source: "DataGolf `field-updates` endpoint (current-week live event feed).",
    calculation: "Direct leaderboard position value provided by DataGolf for the active event.",
  },
  current_score_to_par: {
    layman: "Player's live total score relative to par.",
    source: "DataGolf `field-updates` endpoint.",
    calculation: "Direct live to-par tournament total (E = even par, negative is better).",
  },
  today_score_to_par: {
    layman: "Player's score relative to par for the current round only.",
    source: "DataGolf `field-updates` endpoint.",
    calculation: "Direct round-to-date to-par value from live scoring data.",
  },
  current_thru: {
    layman: "How many holes the player has completed (or final status).",
    source: "DataGolf `field-updates` endpoint.",
    calculation: "Direct live progress indicator (for example 12, 18, or F).",
  },
  win_probability: {
    layman: "Chance this player wins the tournament.",
    source: "Hybrid Markov + Monte Carlo simulation built from DataGolf player inputs.",
    calculation: "Share of simulations where the player finishes with the best score (ties split proportionally).",
  },
  win_delta_prev: {
    layman: "How this player's win chance changed versus the previous snapshot.",
    source: "Calculated from locally stored simulation snapshots for this event.",
    calculation: "Current Win % minus previous snapshot Win %.",
  },
  win_delta_start: {
    layman: "How this player's win chance changed versus the first snapshot in this event.",
    source: "Calculated from locally stored simulation snapshots for this event.",
    calculation: "Current Win % minus first snapshot Win %.",
  },
  top_3_probability: {
    layman: "Chance this player finishes in the top 3.",
    source: "Hybrid Markov + Monte Carlo simulation.",
    calculation: "Share of simulations where the player's simulated finish rank is 1-3.",
  },
  top_5_probability: {
    layman: "Chance this player finishes in the top 5.",
    source: "Hybrid Markov + Monte Carlo simulation.",
    calculation: "Share of simulations where the player's simulated finish rank is 1-5.",
  },
  top_10_probability: {
    layman: "Chance this player finishes in the top 10.",
    source: "Hybrid Markov + Monte Carlo simulation.",
    calculation: "Share of simulations where the player's simulated finish rank is 1-10.",
  },
  mean_finish: {
    layman: "Average finishing position across all simulations.",
    source: "Hybrid Markov + Monte Carlo simulation.",
    calculation: "Mean of simulated finish ranks over all tournament runs.",
  },
  baseline_win_probability: {
    layman: "DataGolf's pre-tournament win probability baseline.",
    source: "DataGolf `preds/pre-tournament` endpoint.",
    calculation: "Direct baseline value from DataGolf; shown as a percent.",
  },
  baseline_season_metric: {
    layman: "Player's baseline performance level from the selected baseline season.",
    source: "DataGolf `historical-event-data/event-list` and `historical-event-data/events`.",
    calculation: "Average event-level form metric across baseline-season events (DG Points first, then fallback metrics).",
  },
  current_season_metric: {
    layman: "Player's performance level from the selected current season.",
    source: "DataGolf `historical-event-data/event-list` and `historical-event-data/events`.",
    calculation: "Average event-level form metric across current-season events (same metric logic as baseline season).",
  },
  form_delta_metric: {
    layman: "How much current-season form differs from baseline season form.",
    source: "Derived from baseline and current season metrics in this app.",
    calculation: "Current Season metric minus Baseline Season metric.",
  },
};

const CONTROL_TOOLTIPS = {
  tourSelect:
    "Choose which tour feed to simulate. Events and player pools come from this selected tour.",
  eventSelect:
    "Select a specific active event. Auto uses the latest event currently available in DataGolf feeds.",
  simulationsInput:
    "Maximum simulations to run (hard cap). In Auto Target mode this is a safety cap.",
  liveAutoRefreshSelect:
    "When enabled, reruns the simulation automatically on a fixed interval to track live probability movement.",
  liveRefreshSecondsInput:
    "Seconds between automatic simulation refreshes when Live Auto-Refresh is enabled.",
  resolutionModeSelect:
    "Auto Target: stop when CI precision target is met; Fixed Cap: always run exactly Simulations.",
  minSimulationsInput:
    "Minimum simulations to run before adaptive early-stop is allowed. If this is above Simulations, it is effectively capped at Simulations.",
  simulationBatchSizeInput:
    "Simulations are processed in batches for adaptive checks. Larger batches are faster but stop less precisely.",
  cutSizeInput:
    "Projected cut line size after round 2 (ties included).",
  meanReversionInput:
    "Base strength of pull back toward field-average state each round.",
  sharedRoundShockInput:
    "Common round shock applied to all players each round (weather/course day effect correlation).",
  useAdaptiveSimulationSelect:
    "If enabled, simulation can stop early once top-N win probability confidence intervals are narrow enough.",
  ciConfidenceInput:
    "Confidence level for adaptive stopping (for example 0.95). Higher confidence usually requires more simulations.",
  ciHalfWidthTargetInput:
    "Target half-width for top-N win probability confidence intervals. Smaller target means more simulations.",
  ciTopNInput:
    "Adaptive stopping monitors the worst CI half-width among the top-N win probability players.",
  useInPlayConditioningSelect:
    "If enabled, simulations condition on current live scores/thru and only simulate remaining tournament steps.",
  useSeasonalFormSelect:
    "Enable blending baseline vs current-season form metrics into player skill inputs.",
  baselineSeasonInput:
    "Baseline comparison season for form modeling.",
  currentSeasonInput:
    "Current season for form modeling and delta calculations.",
  seasonalWeightInput:
    "Overall weight of seasonal form signal versus baseline DataGolf priors.",
  currentSeasonWeightInput:
    "Within seasonal form, weight assigned to current season vs baseline season.",
  formDeltaWeightInput:
    "Extra emphasis on the change from baseline season to current season form.",
  seedInput:
    "Optional random seed for reproducible simulation outputs. Leave blank for random run-to-run variation.",
  rowsInput:
    "Number of ranked players to display in the results table.",
  syncLearningButton:
    "Fetch outcomes for previously predicted events from DataGolf historical endpoints, then retrain calibration.",
  refreshLearningButton:
    "Reload learning stats without retraining.",
};

const ui = {
  tourSelect: document.getElementById("tourSelect"),
  eventSelect: document.getElementById("eventSelect"),
  simulationsInput: document.getElementById("simulationsInput"),
  liveAutoRefreshSelect: document.getElementById("liveAutoRefreshSelect"),
  liveRefreshSecondsInput: document.getElementById("liveRefreshSecondsInput"),
  resolutionModeSelect: document.getElementById("resolutionModeSelect"),
  minSimulationsInput: document.getElementById("minSimulationsInput"),
  simulationBatchSizeInput: document.getElementById("simulationBatchSizeInput"),
  cutSizeInput: document.getElementById("cutSizeInput"),
  meanReversionInput: document.getElementById("meanReversionInput"),
  meanReversionValue: document.getElementById("meanReversionValue"),
  sharedRoundShockInput: document.getElementById("sharedRoundShockInput"),
  sharedRoundShockValue: document.getElementById("sharedRoundShockValue"),
  useAdaptiveSimulationSelect: document.getElementById("useAdaptiveSimulationSelect"),
  ciConfidenceInput: document.getElementById("ciConfidenceInput"),
  ciConfidenceValue: document.getElementById("ciConfidenceValue"),
  ciHalfWidthTargetInput: document.getElementById("ciHalfWidthTargetInput"),
  ciTopNInput: document.getElementById("ciTopNInput"),
  useInPlayConditioningSelect: document.getElementById("useInPlayConditioningSelect"),
  useSeasonalFormSelect: document.getElementById("useSeasonalFormSelect"),
  baselineSeasonInput: document.getElementById("baselineSeasonInput"),
  currentSeasonInput: document.getElementById("currentSeasonInput"),
  seasonalWeightInput: document.getElementById("seasonalWeightInput"),
  seasonalWeightValue: document.getElementById("seasonalWeightValue"),
  currentSeasonWeightInput: document.getElementById("currentSeasonWeightInput"),
  currentSeasonWeightValue: document.getElementById("currentSeasonWeightValue"),
  formDeltaWeightInput: document.getElementById("formDeltaWeightInput"),
  formDeltaWeightValue: document.getElementById("formDeltaWeightValue"),
  seedInput: document.getElementById("seedInput"),
  rowsInput: document.getElementById("rowsInput"),
  loadEventsButton: document.getElementById("loadEventsButton"),
  simulateButton: document.getElementById("simulateButton"),
  syncLearningButton: document.getElementById("syncLearningButton"),
  refreshLearningButton: document.getElementById("refreshLearningButton"),
  applyRecommendationButton: document.getElementById("applyRecommendationButton"),
  status: document.getElementById("status"),
  formStatus: document.getElementById("formStatus"),
  learningStatus: document.getElementById("learningStatus"),
  error: document.getElementById("error"),
  resultsSection: document.getElementById("resultsSection"),
  eventLabel: document.getElementById("eventLabel"),
  simLabel: document.getElementById("simLabel"),
  generatedLabel: document.getElementById("generatedLabel"),
  winnerLabel: document.getElementById("winnerLabel"),
  seasonWindowLabel: document.getElementById("seasonWindowLabel"),
  calibrationLabel: document.getElementById("calibrationLabel"),
  winChart: document.getElementById("winChart"),
  resultsBody: document.getElementById("resultsBody"),
};

function setStatus(text, running = false) {
  ui.status.textContent = text;
  ui.status.classList.toggle("running", running);
  ui.status.classList.toggle("idle", !running);
}

function setError(message = "") {
  if (!message) {
    ui.error.hidden = true;
    ui.error.textContent = "";
    return;
  }
  ui.error.textContent = message;
  ui.error.hidden = false;
}

function setFormStatus(message = "", running = false) {
  ui.formStatus.textContent = message;
  ui.formStatus.classList.toggle("running", running);
  ui.formStatus.classList.toggle("idle", !running);
}

function setLearningStatus(message = "", running = false) {
  ui.learningStatus.textContent = message;
  ui.learningStatus.classList.toggle("running", running);
  ui.learningStatus.classList.toggle("idle", !running);
}

function setBusy(isBusy) {
  ui.simulateButton.disabled = isBusy;
  ui.loadEventsButton.disabled = isBusy;
  if (ui.liveAutoRefreshSelect) {
    ui.liveAutoRefreshSelect.disabled = isBusy;
  }
  if (ui.liveRefreshSecondsInput) {
    ui.liveRefreshSecondsInput.disabled = isBusy;
  }
  if (ui.syncLearningButton) {
    ui.syncLearningButton.disabled = isBusy;
  }
  if (ui.refreshLearningButton) {
    ui.refreshLearningButton.disabled = isBusy;
  }
  if (ui.applyRecommendationButton) {
    ui.applyRecommendationButton.disabled = isBusy;
  }
}

function formatPct(probability) {
  if (probability == null || Number.isNaN(probability)) {
    return "-";
  }
  return `${(probability * 100).toFixed(2)}%`;
}

function formatDate(dateValue) {
  if (!dateValue) {
    return "-";
  }
  const parsed = new Date(dateValue);
  if (Number.isNaN(parsed.getTime())) {
    return String(dateValue);
  }
  return parsed.toLocaleString();
}

function formatMetric(value) {
  if (value == null || Number.isNaN(value)) {
    return "-";
  }
  return Number(value).toFixed(3);
}

function formatScoreToPar(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return "-";
  }
  if (Math.abs(numeric) < 1e-9) {
    return "E";
  }
  const absoluteText = Number.isInteger(numeric)
    ? Math.abs(numeric).toFixed(0)
    : Math.abs(numeric).toFixed(1).replace(/\.0$/, "");
  return `${numeric > 0 ? "+" : "-"}${absoluteText}`;
}

function formatRoundScore(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return "-";
  }
  return Number.isInteger(numeric) ? String(Math.trunc(numeric)) : numeric.toFixed(1);
}

function formatThru(value) {
  if (value == null || value === "") {
    return "-";
  }
  return String(value);
}

function formatSignedPct(value) {
  if (value == null || Number.isNaN(Number(value))) {
    return "-";
  }
  const numeric = Number(value);
  const abs = Math.abs(numeric * 100).toFixed(2);
  if (numeric > 0) {
    return `+${abs}%`;
  }
  if (numeric < 0) {
    return `-${abs}%`;
  }
  return "0.00%";
}

function formatScore(value, digits = 4) {
  if (value == null || Number.isNaN(Number(value))) {
    return "-";
  }
  return Number(value).toFixed(digits);
}

function normalizeNameKey(name) {
  if (!name) {
    return "";
  }
  let out = String(name).trim().toLowerCase().replace(/\./g, "");
  if (out.includes(",")) {
    const pieces = out.split(",").map((piece) => piece.trim()).filter(Boolean);
    if (pieces.length >= 2) {
      out = `${pieces.slice(1).join(" ")} ${pieces[0]}`.trim();
    }
  }
  return out.replace(/\s+/g, " ");
}

function canonicalPlayerKey(playerId, playerName) {
  const id = String(playerId || "").trim().toLowerCase().replace(/\s+/g, "");
  if (id) {
    return id;
  }
  return normalizeNameKey(playerName);
}

function trendForPlayer(player) {
  const idKey = canonicalPlayerKey(player.player_id, "");
  if (idKey && state.trendPlayerByKey[idKey]) {
    return state.trendPlayerByKey[idKey];
  }
  const nameKey = canonicalPlayerKey("", player.player_name);
  if (nameKey && state.trendPlayerByKey[nameKey]) {
    return state.trendPlayerByKey[nameKey];
  }
  return null;
}

function createWinTrendSparkline(points) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 360 88");
  svg.setAttribute("class", "trend-sparkline");
  if (!Array.isArray(points) || points.length < 2) {
    return svg;
  }

  const values = points.map((point) => Number(point.win_probability || 0));
  const minV = Math.min(...values);
  const maxV = Math.max(...values);
  const span = Math.max(maxV - minV, 1e-6);
  const left = 8;
  const right = 352;
  const top = 8;
  const bottom = 80;
  const usableW = right - left;
  const usableH = bottom - top;
  const xStep = usableW / Math.max(1, points.length - 1);

  const toX = (idx) => left + (idx * xStep);
  const toY = (value) => bottom - (((value - minV) / span) * usableH);
  const pathData = points
    .map((point, idx) => `${idx === 0 ? "M" : "L"}${toX(idx).toFixed(2)} ${toY(Number(point.win_probability || 0)).toFixed(2)}`)
    .join(" ");

  const areaPath = document.createElementNS("http://www.w3.org/2000/svg", "path");
  const areaData = `${pathData} L ${toX(points.length - 1).toFixed(2)} ${bottom} L ${toX(0).toFixed(2)} ${bottom} Z`;
  areaPath.setAttribute("d", areaData);
  areaPath.setAttribute("class", "trend-area");

  const linePath = document.createElementNS("http://www.w3.org/2000/svg", "path");
  linePath.setAttribute("d", pathData);
  linePath.setAttribute("class", "trend-line");

  const firstDot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
  firstDot.setAttribute("cx", toX(0).toFixed(2));
  firstDot.setAttribute("cy", toY(values[0]).toFixed(2));
  firstDot.setAttribute("r", "3");
  firstDot.setAttribute("class", "trend-dot trend-dot-first");

  const lastDot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
  lastDot.setAttribute("cx", toX(values.length - 1).toFixed(2));
  lastDot.setAttribute("cy", toY(values[values.length - 1]).toFixed(2));
  lastDot.setAttribute("r", "3.5");
  lastDot.setAttribute("class", "trend-dot trend-dot-last");

  svg.appendChild(areaPath);
  svg.appendChild(linePath);
  svg.appendChild(firstDot);
  svg.appendChild(lastDot);
  return svg;
}

function findMarketStatus(payload, marketName) {
  if (!payload || !Array.isArray(payload.markets)) {
    return null;
  }
  return payload.markets.find((market) => market.market === marketName) || null;
}

function renderLearningStatus(payload) {
  state.latestLearningStatus = payload;
  const winMarket = findMarketStatus(payload, "win");
  const brierSummary =
    winMarket && winMarket.samples > 0
      ? `win Brier ${formatScore(winMarket.brier_before)} -> ${formatScore(winMarket.brier_after)}`
      : "win Brier unavailable (need resolved outcomes)";
  setLearningStatus(
    `Learning v${payload.calibration_version} | resolved events=${payload.resolved_events} | pending=${payload.pending_events} | ${brierSummary}`
  );
}

function playerRowKey(player, index) {
  const id = String(player.player_id || "").trim();
  if (id) {
    return id;
  }
  return `${player.player_name || "player"}-${index}`;
}

function tooltipForColumn(columnKey) {
  const entry = TABLE_TOOLTIPS[columnKey];
  if (!entry) {
    return "";
  }
  return `Layman's Terms: ${entry.layman}\nData Source: ${entry.source}\nCalculation: ${entry.calculation}`;
}

function applyTableHeaderTooltips() {
  const headers = document.querySelectorAll("thead th[data-col]");
  headers.forEach((header) => {
    const key = header.dataset.col;
    const tooltip = tooltipForColumn(key);
    if (!tooltip) {
      return;
    }
    header.title = tooltip;
  });
}

function applyControlTooltips() {
  Object.entries(CONTROL_TOOLTIPS).forEach(([elementId, tooltip]) => {
    const input = document.getElementById(elementId);
    if (!input || !tooltip) {
      return;
    }
    input.title = tooltip;
    const label = input.closest("label");
    if (label) {
      label.title = tooltip;
      const caption = label.querySelector("span");
      if (caption) {
        caption.title = tooltip;
      }
    }
  });
}

function appendResultCell(tr, columnKey, text, numeric = true, extraClassName = "") {
  const td = document.createElement("td");
  const classNames = [];
  if (numeric) {
    classNames.push("num");
  }
  if (extraClassName) {
    classNames.push(extraClassName);
  }
  if (classNames.length > 0) {
    td.className = classNames.join(" ");
  }
  td.textContent = text;
  const tooltip = tooltipForColumn(columnKey);
  if (tooltip) {
    td.title = tooltip;
  }
  tr.appendChild(td);
}

function deltaClassName(value) {
  if (value == null || Number.isNaN(Number(value))) {
    return "delta-flat";
  }
  const numeric = Number(value);
  if (numeric > 1e-9) {
    return "delta-up";
  }
  if (numeric < -1e-9) {
    return "delta-down";
  }
  return "delta-flat";
}

function buildPlayerCell(player, rowKey, expanded) {
  const td = document.createElement("td");
  const button = document.createElement("button");
  button.type = "button";
  button.className = "player-toggle";
  button.setAttribute("aria-expanded", String(expanded));
  button.title = tooltipForColumn("player");

  const caret = document.createElement("span");
  caret.className = "player-toggle-caret";
  caret.textContent = expanded ? "▼" : "▶";

  const label = document.createElement("span");
  label.className = "player-toggle-label";
  label.textContent = player.player_name;

  button.appendChild(caret);
  button.appendChild(label);
  button.addEventListener("click", () => {
    if (!state.latestResult) {
      return;
    }
    state.expandedPlayerKey = state.expandedPlayerKey === rowKey ? null : rowKey;
    renderTable(state.latestResult.players);
  });

  td.appendChild(button);
  return td;
}

function resultColumnCount() {
  return document.querySelectorAll("thead th").length;
}

function buildScorecardDetails(player) {
  const wrapper = document.createElement("div");
  wrapper.className = "scorecard-detail";

  const summary = document.createElement("div");
  summary.className = "scorecard-meta";
  const summaryItems = [
    { label: "Position", value: player.current_position || "-" },
    { label: "Total", value: formatScoreToPar(player.current_score_to_par) },
    { label: "Today", value: formatScoreToPar(player.today_score_to_par) },
    { label: "Thru", value: formatThru(player.current_thru) },
  ];

  summaryItems.forEach((item) => {
    const pill = document.createElement("span");
    pill.className = "scorecard-pill";
    pill.textContent = `${item.label}: ${item.value}`;
    summary.appendChild(pill);
  });
  wrapper.appendChild(summary);

  const roundScores = Array.isArray(player.round_scores) ? player.round_scores : [];
  if (roundScores.length > 0) {
    const rounds = document.createElement("div");
    rounds.className = "scorecard-rounds";
    rounds.textContent = roundScores
      .map((score, index) => `R${index + 1}: ${formatRoundScore(score)}`)
      .join(" | ");
    wrapper.appendChild(rounds);
  }

  const holeScores = Array.isArray(player.hole_scores) ? player.hole_scores : [];
  if (holeScores.length > 0) {
    const holes = document.createElement("div");
    holes.className = "scorecard-holes";
    holeScores.forEach((score, index) => {
      const chip = document.createElement("span");
      chip.className = "hole-chip";
      chip.textContent = `${index + 1}:${formatRoundScore(score)}`;
      holes.appendChild(chip);
    });
    wrapper.appendChild(holes);
  }

  if (roundScores.length === 0 && holeScores.length === 0) {
    const empty = document.createElement("div");
    empty.className = "scorecard-empty";
    empty.textContent =
      "DataGolf live feeds currently expose leaderboard status (position/score/thru), but not per-hole scorecards for individual players.";
    wrapper.appendChild(empty);
  }

  const trend = trendForPlayer(player);
  if (trend && Array.isArray(trend.points) && trend.points.length > 0) {
    const trendBlock = document.createElement("div");
    trendBlock.className = "trend-detail";

    const title = document.createElement("div");
    title.className = "trend-title";
    title.textContent = `Win% Trend (${trend.points.length} snapshots)`;
    trendBlock.appendChild(title);

    const statRow = document.createElement("div");
    statRow.className = "trend-stats";
    const chips = [
      `Latest: ${formatPct(trend.latest_win_probability)}`,
      `Since First: ${formatPct(trend.delta_win_since_first)}`,
      `Since Previous: ${formatPct(trend.delta_win_since_previous)}`,
    ];
    chips.forEach((chipText) => {
      const chip = document.createElement("span");
      chip.className = "scorecard-pill";
      chip.textContent = chipText;
      statRow.appendChild(chip);
    });
    trendBlock.appendChild(statRow);
    trendBlock.appendChild(createWinTrendSparkline(trend.points));

    const lastPoint = trend.points[trend.points.length - 1];
    if (lastPoint && lastPoint.created_at) {
      const meta = document.createElement("div");
      meta.className = "trend-meta";
      meta.textContent = `Latest snapshot: ${formatDate(lastPoint.created_at)}`;
      trendBlock.appendChild(meta);
    }
    wrapper.appendChild(trendBlock);
  }

  return wrapper;
}

function updateEventSelect(events) {
  const priorValue = ui.eventSelect.value;
  ui.eventSelect.innerHTML = "";

  const autoOption = document.createElement("option");
  autoOption.value = "";
  autoOption.textContent = "Auto (use latest available event)";
  ui.eventSelect.appendChild(autoOption);

  let firstSimulatableValue = "";
  for (const event of events) {
    const option = document.createElement("option");
    option.value = event.event_id;
    const datePart = event.start_date ? ` (${event.start_date})` : "";
    if (event.simulatable) {
      option.textContent = `${event.event_name}${datePart}`;
      if (!firstSimulatableValue) {
        firstSimulatableValue = event.event_id;
      }
    } else {
      option.textContent = `${event.event_name}${datePart} - unavailable now`;
      option.disabled = true;
      if (event.unavailable_reason) {
        option.title = event.unavailable_reason;
      }
    }
    ui.eventSelect.appendChild(option);
  }

  if (events.some((event) => event.event_id === priorValue)) {
    ui.eventSelect.value = priorValue;
    const selected = events.find((event) => event.event_id === priorValue);
    if (selected && !selected.simulatable) {
      ui.eventSelect.value = "";
    }
    return;
  }

  if (firstSimulatableValue) {
    ui.eventSelect.value = firstSimulatableValue;
  }
}

async function loadEvents() {
  setError();
  setFormStatus("");
  setBusy(true);
  setStatus("Loading upcoming events...", true);
  try {
    const tour = encodeURIComponent(ui.tourSelect.value);
    const response = await fetch(`/events/upcoming?tour=${tour}&limit=40`);
    if (!response.ok) {
      let detail = `Unable to fetch events (${response.status})`;
      try {
        const errPayload = await response.json();
        if (errPayload?.detail) {
          detail = String(errPayload.detail);
        }
      } catch (_) {
        // Keep default message when response body is not JSON.
      }
      throw new Error(detail);
    }
    const events = await response.json();
    state.events = events;
    updateEventSelect(events);
    const simulatableCount = events.filter((event) => event.simulatable).length;
    setStatus(
      `Loaded ${events.length} events (${simulatableCount} currently simulatable) for ${ui.tourSelect.value.toUpperCase()}.`
    );
  } catch (error) {
    setStatus("Unable to load events.");
    setError(error.message || "Unexpected error while fetching events.");
  } finally {
    setBusy(false);
  }
}

async function loadLearningStatus(silent = false) {
  if (!silent) {
    setLearningStatus("Loading learning status...", true);
  }
  try {
    const tour = encodeURIComponent(ui.tourSelect.value);
    const response = await fetch(`/learning/status?tour=${tour}`);
    if (!response.ok) {
      let detail = `Unable to load learning status (${response.status})`;
      try {
        const errPayload = await response.json();
        if (errPayload?.detail) {
          detail = String(errPayload.detail);
        }
      } catch (_) {
        // Keep default detail message when body is not JSON.
      }
      throw new Error(detail);
    }
    const payload = await response.json();
    renderLearningStatus(payload);
  } catch (error) {
    if (!silent) {
      setLearningStatus("Unable to load learning status.");
      setError(error.message || "Unexpected error while loading learning status.");
    }
  }
}

async function syncLearningAndRetrain() {
  setError();
  setBusy(true);
  setLearningStatus("Syncing outcomes and retraining calibration...", true);
  try {
    const response = await fetch("/learning/sync-train", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        tour: ui.tourSelect.value,
        max_events: 40,
      }),
    });
    if (!response.ok) {
      let detail = `Learning sync failed (${response.status})`;
      try {
        const errPayload = await response.json();
        if (errPayload?.detail) {
          detail = String(errPayload.detail);
        }
      } catch (_) {
        // Keep default detail message when body is not JSON.
      }
      throw new Error(detail);
    }
    const payload = await response.json();
    renderLearningStatus(payload);
    const statusParts = [
      `events_processed=${payload.events_processed}`,
      `outcomes_fetched=${payload.outcomes_fetched}`,
      `awaiting_official=${payload.awaiting_outcomes_count || 0}`,
      `retrain=${payload.retrain_executed ? "yes" : "no"}`,
      `version=v${payload.calibration_version}`,
    ];
    if (payload.sync_note) {
      statusParts.push(payload.sync_note);
    }
    setStatus(`Learning sync complete. ${statusParts.join(" | ")}`);
  } catch (error) {
    setStatus("Learning sync/retrain failed.");
    setError(error.message || "Unexpected error during learning retrain.");
  } finally {
    setBusy(false);
  }
}

function resetEventTrends() {
  state.eventTrends = null;
  state.trendPlayerByKey = {};
}

function indexEventTrends(payload) {
  const byKey = {};
  const players = Array.isArray(payload?.players) ? payload.players : [];
  players.forEach((player) => {
    const idKey = canonicalPlayerKey(player.player_id, "");
    const nameKey = canonicalPlayerKey("", player.player_name);
    if (idKey) {
      byKey[idKey] = player;
    }
    if (nameKey) {
      byKey[nameKey] = player;
    }
  });
  state.eventTrends = payload;
  state.trendPlayerByKey = byKey;
}

async function loadEventTrendsForCurrentEvent({ silent = true } = {}) {
  if (!state.latestResult || !state.latestResult.event_id) {
    resetEventTrends();
    return;
  }
  try {
    const query = new URLSearchParams({
      tour: state.latestResult.tour || ui.tourSelect.value,
      event_id: state.latestResult.event_id,
      max_snapshots: "80",
      max_players: "80",
    });
    const response = await fetch(`/learning/event-trends?${query.toString()}`);
    if (!response.ok) {
      let detail = `Unable to fetch event trends (${response.status})`;
      try {
        const errPayload = await response.json();
        if (errPayload?.detail) {
          detail = String(errPayload.detail);
        }
      } catch (_) {
        // Keep default message when response body is not JSON.
      }
      throw new Error(detail);
    }
    const payload = await response.json();
    indexEventTrends(payload);
    if (state.latestResult) {
      renderTable(state.latestResult.players);
    }
  } catch (error) {
    if (!silent) {
      setError(error.message || "Unexpected error while loading event trends.");
    }
    resetEventTrends();
  }
}

function stopAutoRefresh() {
  if (state.autoRefreshTimerId != null) {
    window.clearInterval(state.autoRefreshTimerId);
    state.autoRefreshTimerId = null;
  }
}

function applyAutoRefreshSchedule() {
  stopAutoRefresh();
  if (!ui.liveAutoRefreshSelect || ui.liveAutoRefreshSelect.value !== "yes") {
    return;
  }
  const seconds = Math.max(
    15,
    Math.min(900, Number.parseInt(ui.liveRefreshSecondsInput.value, 10) || 60)
  );
  state.autoRefreshTimerId = window.setInterval(() => {
    if (state.simulationInFlight) {
      return;
    }
    void runSimulation(true);
  }, seconds * 1000);
  setStatus(`Live auto-refresh enabled every ${seconds}s.`);
}

function renderWinChart(players) {
  ui.winChart.innerHTML = "";
  const topTen = players.slice(0, 10);
  for (const player of topTen) {
    const row = document.createElement("div");
    row.className = "bar-row";

    const name = document.createElement("span");
    name.className = "bar-name";
    name.textContent = player.player_name;

    const track = document.createElement("div");
    track.className = "bar-track";

    const fill = document.createElement("div");
    fill.className = "bar-fill";
    const width = Math.max(1, Math.min(100, player.win_probability * 100));
    fill.style.setProperty("--w", `${width}%`);
    track.appendChild(fill);

    const value = document.createElement("span");
    value.className = "bar-value";
    value.textContent = formatPct(player.win_probability);

    row.appendChild(name);
    row.appendChild(track);
    row.appendChild(value);
    ui.winChart.appendChild(row);
  }
}

function renderTable(players) {
  ui.resultsBody.innerHTML = "";
  const rowLimit = Math.max(5, Math.min(100, Number.parseInt(ui.rowsInput.value, 10) || 25));
  const rows = players.slice(0, rowLimit);
  const visibleRowKeys = new Set(rows.map((player, index) => playerRowKey(player, index)));

  if (state.expandedPlayerKey && !visibleRowKeys.has(state.expandedPlayerKey)) {
    state.expandedPlayerKey = null;
  }

  rows.forEach((player, index) => {
    const rowKey = playerRowKey(player, index);
    const expanded = state.expandedPlayerKey === rowKey;
    const tr = document.createElement("tr");
    if (expanded) {
      tr.classList.add("result-row-expanded");
    }
    appendResultCell(tr, "rank", String(index + 1), true);
    tr.appendChild(buildPlayerCell(player, rowKey, expanded));
    appendResultCell(tr, "current_position", player.current_position || "-", false);
    appendResultCell(
      tr,
      "current_score_to_par",
      formatScoreToPar(player.current_score_to_par),
      true
    );
    appendResultCell(
      tr,
      "today_score_to_par",
      formatScoreToPar(player.today_score_to_par),
      true
    );
    appendResultCell(tr, "current_thru", formatThru(player.current_thru), false);
    appendResultCell(tr, "win_probability", formatPct(player.win_probability), true);
    const trend = trendForPlayer(player);
    appendResultCell(
      tr,
      "win_delta_prev",
      formatSignedPct(trend ? trend.delta_win_since_previous : null),
      true,
      deltaClassName(trend ? trend.delta_win_since_previous : null)
    );
    appendResultCell(
      tr,
      "win_delta_start",
      formatSignedPct(trend ? trend.delta_win_since_first : null),
      true,
      deltaClassName(trend ? trend.delta_win_since_first : null)
    );
    appendResultCell(tr, "top_3_probability", formatPct(player.top_3_probability), true);
    appendResultCell(tr, "top_5_probability", formatPct(player.top_5_probability), true);
    appendResultCell(tr, "top_10_probability", formatPct(player.top_10_probability), true);
    appendResultCell(tr, "mean_finish", Number(player.mean_finish).toFixed(2), true);
    appendResultCell(
      tr,
      "baseline_win_probability",
      formatPct(player.baseline_win_probability),
      true
    );
    appendResultCell(
      tr,
      "baseline_season_metric",
      formatMetric(player.baseline_season_metric),
      true
    );
    appendResultCell(
      tr,
      "current_season_metric",
      formatMetric(player.current_season_metric),
      true
    );
    appendResultCell(tr, "form_delta_metric", formatMetric(player.form_delta_metric), true);
    ui.resultsBody.appendChild(tr);

    if (expanded) {
      const detailTr = document.createElement("tr");
      detailTr.className = "scorecard-row";
      const detailTd = document.createElement("td");
      detailTd.colSpan = resultColumnCount();
      detailTd.appendChild(buildScorecardDetails(player));
      detailTr.appendChild(detailTd);
      ui.resultsBody.appendChild(detailTr);
    }
  });
}

function renderResult(payload) {
  state.latestResult = payload;
  state.expandedPlayerKey = null;
  const topPlayer = payload.players[0];

  ui.eventLabel.textContent = payload.event_name || payload.event_id || "Unknown event";
  if (
    payload.requested_simulations &&
    payload.requested_simulations !== payload.simulations
  ) {
    ui.simLabel.textContent = `${payload.simulations.toLocaleString()} / ${payload.requested_simulations.toLocaleString()} requested`;
  } else {
    ui.simLabel.textContent = payload.simulations.toLocaleString();
  }
  ui.generatedLabel.textContent = formatDate(payload.generated_at);
  ui.winnerLabel.textContent = topPlayer
    ? `${topPlayer.player_name} (${formatPct(topPlayer.win_probability)})`
    : "-";
  if (payload.baseline_season && payload.current_season) {
    ui.seasonWindowLabel.textContent = `${payload.baseline_season} vs ${payload.current_season}`;
  } else {
    ui.seasonWindowLabel.textContent = "-";
  }
  if (payload.calibration_applied) {
    ui.calibrationLabel.textContent = `Applied v${payload.calibration_version}`;
  } else if (payload.calibration_version > 0) {
    ui.calibrationLabel.textContent = `Available v${payload.calibration_version}`;
  } else {
    ui.calibrationLabel.textContent = "Not trained";
  }
  if (payload.calibration_note) {
    ui.calibrationLabel.title = payload.calibration_note;
  }
  setFormStatus(
    payload.form_adjustment_note
      ? payload.form_adjustment_note
      : payload.form_adjustment_applied
        ? "Seasonal form adjustment applied."
        : "Seasonal form adjustment not applied."
  );

  if (
    ui.applyRecommendationButton &&
    payload.recommended_simulations &&
    payload.stop_reason === "max_simulations_reached"
  ) {
    ui.applyRecommendationButton.hidden = false;
    ui.applyRecommendationButton.textContent = `Use Recommended Cap (${payload.recommended_simulations.toLocaleString()})`;
  } else if (ui.applyRecommendationButton) {
    ui.applyRecommendationButton.hidden = true;
  }

  renderWinChart(payload.players);
  renderTable(payload.players);
  ui.resultsSection.hidden = false;
}

async function runSimulation(fromAutoRefresh = false) {
  if (state.simulationInFlight || ui.simulateButton.disabled) {
    return;
  }
  state.simulationInFlight = true;
  setError();
  setFormStatus("Loading seasonal form data...", true);
  setBusy(true);
  setStatus(
    fromAutoRefresh ? "Running auto-refresh simulation..." : "Running simulation...",
    true
  );

  try {
    const simulations = Number.parseInt(ui.simulationsInput.value, 10) || 10000;
    const minSimulations = Number.parseInt(ui.minSimulationsInput.value, 10) || 5000;
    const simulationBatchSize = Number.parseInt(ui.simulationBatchSizeInput.value, 10) || 5000;
    const cutSize = Number.parseInt(ui.cutSizeInput.value, 10) || 70;
    const meanReversion = Number.parseFloat(ui.meanReversionInput.value) || 0.1;
    const sharedRoundShockSigma = Number.parseFloat(ui.sharedRoundShockInput.value) || 0.35;
    const useAdaptiveSimulation = ui.useAdaptiveSimulationSelect.value === "yes";
    const ciConfidence = Number.parseFloat(ui.ciConfidenceInput.value) || 0.95;
    const ciHalfWidthTarget =
      Number.parseFloat(ui.ciHalfWidthTargetInput.value) || 0.0025;
    const ciTopN = Number.parseInt(ui.ciTopNInput.value, 10) || 10;
    const useInPlayConditioning = ui.useInPlayConditioningSelect.value === "yes";
    const useSeasonalForm = ui.useSeasonalFormSelect.value === "yes";
    const baselineSeason = Number.parseInt(ui.baselineSeasonInput.value, 10);
    const currentSeason = Number.parseInt(ui.currentSeasonInput.value, 10);
    const seasonalWeight = Number.parseFloat(ui.seasonalWeightInput.value);
    const currentSeasonWeight = Number.parseFloat(ui.currentSeasonWeightInput.value);
    const formDeltaWeight = Number.parseFloat(ui.formDeltaWeightInput.value);
    const seedRaw = ui.seedInput.value.trim();

    const requestBody = {
      tour: ui.tourSelect.value,
      event_id: ui.eventSelect.value || null,
      resolution_mode: ui.resolutionModeSelect.value || "auto_target",
      simulations: simulations,
      min_simulations: minSimulations,
      simulation_batch_size: simulationBatchSize,
      cut_size: cutSize,
      mean_reversion: meanReversion,
      shared_round_shock_sigma: sharedRoundShockSigma,
      enable_adaptive_simulation: useAdaptiveSimulation,
      ci_confidence: ciConfidence,
      ci_half_width_target: ciHalfWidthTarget,
      ci_top_n: ciTopN,
      enable_in_play_conditioning: useInPlayConditioning,
      enable_seasonal_form: useSeasonalForm,
      seasonal_form_weight: seasonalWeight,
      current_season_weight: currentSeasonWeight,
      form_delta_weight: formDeltaWeight,
    };

    if (!Number.isNaN(baselineSeason)) {
      requestBody.baseline_season = baselineSeason;
    }
    if (!Number.isNaN(currentSeason)) {
      requestBody.current_season = currentSeason;
    }

    if (seedRaw.length > 0) {
      requestBody.seed = Number.parseInt(seedRaw, 10);
    }

    const response = await fetch("/simulate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      let detail = `Simulation failed (${response.status})`;
      try {
        const errPayload = await response.json();
        if (errPayload?.detail) {
          detail = String(errPayload.detail);
        }
      } catch (_) {
        // Keep default detail message when body is not JSON.
      }
      throw new Error(detail);
    }

    const payload = await response.json();
    renderResult(payload);
    await loadEventTrendsForCurrentEvent({ silent: true });
    const statusBits = [];
    if (payload.stop_reason) {
      statusBits.push(`stop=${payload.stop_reason}`);
    }
    if (payload.win_ci_half_width_top_n != null) {
      statusBits.push(`top-N CI half-width=${(payload.win_ci_half_width_top_n * 100).toFixed(3)}%`);
    }
    if (payload.in_play_conditioning_note) {
      statusBits.push(payload.in_play_conditioning_note);
    }
    if (payload.calibration_note) {
      statusBits.push(payload.calibration_note);
    }
    setStatus(statusBits.length > 0 ? `Simulation complete. ${statusBits.join(" | ")}` : "Simulation complete.");
    void loadLearningStatus(true);
  } catch (error) {
    setStatus("Simulation failed.");
    setError(error.message || "Unexpected error while running the simulation.");
  } finally {
    setBusy(false);
    state.simulationInFlight = false;
  }
}

function bindEvents() {
  ui.loadEventsButton.addEventListener("click", loadEvents);
  ui.simulateButton.addEventListener("click", () => {
    void runSimulation(false);
  });
  if (ui.syncLearningButton) {
    ui.syncLearningButton.addEventListener("click", syncLearningAndRetrain);
  }
  if (ui.refreshLearningButton) {
    ui.refreshLearningButton.addEventListener("click", () => loadLearningStatus());
  }
  ui.tourSelect.addEventListener("change", () => {
    resetEventTrends();
    void loadEvents();
    void loadLearningStatus(true);
  });
  if (ui.eventSelect) {
    ui.eventSelect.addEventListener("change", () => {
      resetEventTrends();
    });
  }
  if (ui.liveAutoRefreshSelect) {
    ui.liveAutoRefreshSelect.addEventListener("change", applyAutoRefreshSchedule);
  }
  if (ui.liveRefreshSecondsInput) {
    ui.liveRefreshSecondsInput.addEventListener("change", applyAutoRefreshSchedule);
  }
  ui.meanReversionInput.addEventListener("input", () => {
    ui.meanReversionValue.textContent = Number.parseFloat(ui.meanReversionInput.value).toFixed(2);
  });
  ui.sharedRoundShockInput.addEventListener("input", () => {
    ui.sharedRoundShockValue.textContent = Number.parseFloat(ui.sharedRoundShockInput.value).toFixed(2);
  });
  ui.ciConfidenceInput.addEventListener("input", () => {
    ui.ciConfidenceValue.textContent = Number.parseFloat(ui.ciConfidenceInput.value).toFixed(3);
  });
  ui.seasonalWeightInput.addEventListener("input", () => {
    ui.seasonalWeightValue.textContent = Number.parseFloat(ui.seasonalWeightInput.value).toFixed(2);
  });
  ui.currentSeasonWeightInput.addEventListener("input", () => {
    ui.currentSeasonWeightValue.textContent = Number.parseFloat(ui.currentSeasonWeightInput.value).toFixed(2);
  });
  ui.formDeltaWeightInput.addEventListener("input", () => {
    ui.formDeltaWeightValue.textContent = Number.parseFloat(ui.formDeltaWeightInput.value).toFixed(2);
  });
  ui.rowsInput.addEventListener("change", () => {
    if (state.latestResult) {
      renderTable(state.latestResult.players);
    }
  });
  if (ui.applyRecommendationButton) {
    ui.applyRecommendationButton.addEventListener("click", () => {
      if (!state.latestResult || !state.latestResult.recommended_simulations) {
        return;
      }
      ui.simulationsInput.value = String(state.latestResult.recommended_simulations);
      if (ui.minSimulationsInput) {
        const currentMin = Number.parseInt(ui.minSimulationsInput.value, 10) || 0;
        if (currentMin > state.latestResult.recommended_simulations) {
          ui.minSimulationsInput.value = String(state.latestResult.recommended_simulations);
        }
      }
      setStatus(
        `Updated Simulations cap to ${Number(state.latestResult.recommended_simulations).toLocaleString()}. Run again to target CI precision.`
      );
    });
  }
}

function init() {
  const currentYear = new Date().getFullYear();
  ui.currentSeasonInput.value = String(currentYear);
  ui.baselineSeasonInput.value = String(currentYear - 1);
  ui.resolutionModeSelect.value = "auto_target";
  applyTableHeaderTooltips();
  applyControlTooltips();
  bindEvents();
  ui.meanReversionValue.textContent = Number.parseFloat(ui.meanReversionInput.value).toFixed(2);
  ui.sharedRoundShockValue.textContent = Number.parseFloat(ui.sharedRoundShockInput.value).toFixed(2);
  ui.ciConfidenceValue.textContent = Number.parseFloat(ui.ciConfidenceInput.value).toFixed(3);
  ui.seasonalWeightValue.textContent = Number.parseFloat(ui.seasonalWeightInput.value).toFixed(2);
  ui.currentSeasonWeightValue.textContent = Number.parseFloat(ui.currentSeasonWeightInput.value).toFixed(2);
  ui.formDeltaWeightValue.textContent = Number.parseFloat(ui.formDeltaWeightInput.value).toFixed(2);
  loadEvents();
  loadLearningStatus(true);
  applyAutoRefreshSchedule();
}

init();
