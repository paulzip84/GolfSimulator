const SUPPORTED_TOURS = [
  { value: "pga", label: "PGA" },
  { value: "liv", label: "LIV" },
  { value: "euro", label: "DP" },
  { value: "kft", label: "Korn Ferry" },
];

const state = {
  events: [],
  latestResult: null,
  expandedPlayerKey: null,
  latestLearningStatus: null,
  latestLifecycleStatus: null,
  eventTrends: null,
  trendPlayerByKey: {},
  autoRefreshTimerId: null,
  autoSimulationTimerId: null,
  lifecyclePollTimerId: null,
  liveScorePollTimerId: null,
  simulationInFlight: false,
  currentSimulationTour: null,
  simulationQueue: [],
  tabSimulationStateByTour: {},
  eventsRequestToken: 0,
  powerRankingReport: null,
};
const AUTOMATION_SIMULATION_INTERVAL_SECONDS = 120;
const LIVE_SCORE_POLL_INTERVAL_SECONDS = 30;

function isFiniteNumber(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric);
}

function snapshotHydrationNeedsFreshSimulation(payload) {
  const players = Array.isArray(payload?.players) ? payload.players : [];
  if (players.length < 8) {
    return {
      needsRefresh: true,
      reason: "Stored snapshot has too few players.",
    };
  }

  const requiredFields = [
    "baseline_win_probability",
    "baseline_top_3_probability",
    "baseline_top_5_probability",
    "baseline_top_10_probability",
    "baseline_season_metric",
    "current_season_metric",
    "form_delta_metric",
  ];
  const missingByField = {};
  requiredFields.forEach((field) => {
    missingByField[field] = 0;
  });

  players.forEach((player) => {
    requiredFields.forEach((field) => {
      if (!isFiniteNumber(player[field])) {
        missingByField[field] += 1;
      }
    });
  });

  const totalPlayers = players.length;
  const incompleteFields = requiredFields.filter((field) => missingByField[field] > 0);
  if (incompleteFields.length === 0) {
    return { needsRefresh: false, reason: "" };
  }

  const preview = incompleteFields
    .slice(0, 3)
    .map((field) => `${field} (${missingByField[field]}/${totalPlayers})`)
    .join(", ");
  return {
    needsRefresh: true,
    reason: `Stored snapshot has missing fields: ${preview}.`,
  };
}

function lifecycleState() {
  return String(state.latestLifecycleStatus?.active_event_state || "").trim().toLowerCase();
}

function shouldAutoRunSimulationNow() {
  return lifecycleState() === "in_play";
}

function autoSimulationPausedMessage() {
  const currentState = lifecycleState();
  if (!currentState) {
    return "Simulation automation: waiting for lifecycle status before running.";
  }
  return `Simulation automation: paused until event is in-play (state=${currentState}).`;
}

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
    "When enabled, reruns the simulation automatically on a fixed interval to track live probability movement (in-play only).",
  liveRefreshSecondsInput:
    "Seconds between automatic simulation refreshes when Live Auto-Refresh is enabled (in-play only).",
  resolutionModeSelect:
    "Auto Target: stop when CI precision target is met; Fixed Cap: always run exactly Simulations (default for maximum resolution).",
  minSimulationsInput:
    "Minimum simulations to run before adaptive early-stop is allowed. If this is above Simulations, it is effectively capped at Simulations.",
  simulationBatchSizeInput:
    "Simulations are processed in batches for adaptive checks. Larger batches are faster but stop less precisely. The server applies a memory-safe cap automatically.",
  cutSizeInput:
    "Projected cut line size after round 2 (ties included).",
  meanReversionInput:
    "Base strength of pull back toward field-average state each round.",
  sharedRoundShockInput:
    "Common round shock applied to all players each round (weather/course day effect correlation).",
  useAdaptiveSimulationSelect:
    "If enabled, simulation can stop early once top-N win probability confidence intervals are narrow enough (ignored in Fixed Cap mode).",
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
  runLifecycleButton:
    "Run one lifecycle automation cycle now (pre-event snapshot + outcome sync/retrain checks).",
  powerLookbackInput:
    "Number of recent events to include in power ranking trend history.",
  powerTopNInput:
    "Number of ranked players to plot and show in the power ranking report.",
  refreshPowerReportButton:
    "Refresh the power ranking report for the selected tour.",
  warmStartPowerReportButton:
    "Seed missing snapshot history for all tours so power-ranking reports can render immediately.",
};

const ui = {
  tourTabs: document.getElementById("tourTabs"),
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
  runLifecycleButton: document.getElementById("runLifecycleButton"),
  applyRecommendationButton: document.getElementById("applyRecommendationButton"),
  powerLookbackInput: document.getElementById("powerLookbackInput"),
  powerTopNInput: document.getElementById("powerTopNInput"),
  refreshPowerReportButton: document.getElementById("refreshPowerReportButton"),
  warmStartPowerReportButton: document.getElementById("warmStartPowerReportButton"),
  powerRankingStatus: document.getElementById("powerRankingStatus"),
  powerRankingChart: document.getElementById("powerRankingChart"),
  powerRankingTableBody: document.getElementById("powerRankingTableBody"),
  status: document.getElementById("status"),
  formStatus: document.getElementById("formStatus"),
  learningStatus: document.getElementById("learningStatus"),
  simulationAutomationStatus: document.getElementById("simulationAutomationStatus"),
  lifecycleStatus: document.getElementById("lifecycleStatus"),
  error: document.getElementById("error"),
  resultsSection: document.getElementById("resultsSection"),
  eventLabel: document.getElementById("eventLabel"),
  simLabel: document.getElementById("simLabel"),
  snapshotLabel: document.getElementById("snapshotLabel"),
  versionSimulationTile: document.getElementById("versionSimulationTile"),
  versionSimulationValue: document.getElementById("versionSimulationValue"),
  versionSimulationMeta: document.getElementById("versionSimulationMeta"),
  versionRetrainTile: document.getElementById("versionRetrainTile"),
  versionRetrainValue: document.getElementById("versionRetrainValue"),
  versionRetrainMeta: document.getElementById("versionRetrainMeta"),
  generatedLabel: document.getElementById("generatedLabel"),
  winnerLabel: document.getElementById("winnerLabel"),
  seasonWindowLabel: document.getElementById("seasonWindowLabel"),
  calibrationLabel: document.getElementById("calibrationLabel"),
  winChart: document.getElementById("winChart"),
  resultsBody: document.getElementById("resultsBody"),
  lifecycleHistoryBody: document.getElementById("lifecycleHistoryBody"),
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

function setSimulationAutomationStatus(message = "", running = false) {
  if (!ui.simulationAutomationStatus) {
    return;
  }
  ui.simulationAutomationStatus.textContent = message;
  ui.simulationAutomationStatus.classList.toggle("running", running);
  ui.simulationAutomationStatus.classList.toggle("idle", !running);
}

function setLifecycleStatus(message = "", running = false) {
  if (!ui.lifecycleStatus) {
    return;
  }
  ui.lifecycleStatus.textContent = message;
  ui.lifecycleStatus.classList.toggle("running", running);
  ui.lifecycleStatus.classList.toggle("idle", !running);
}

function setPowerRankingStatus(message = "", running = false) {
  if (!ui.powerRankingStatus) {
    return;
  }
  ui.powerRankingStatus.textContent = message;
  ui.powerRankingStatus.classList.toggle("running", running);
  ui.powerRankingStatus.classList.toggle("idle", !running);
}

function normalizedTourValue(value) {
  const normalized = String(value || "pga").trim().toLowerCase();
  if (SUPPORTED_TOURS.some((tour) => tour.value === normalized)) {
    return normalized;
  }
  return "pga";
}

function getTabSimulationState(tour) {
  const key = normalizedTourValue(tour);
  if (!state.tabSimulationStateByTour[key]) {
    state.tabSimulationStateByTour[key] = "idle";
  }
  return state.tabSimulationStateByTour[key];
}

function setTourTabSimulationState(tour, tabState) {
  const key = normalizedTourValue(tour);
  const allowedStates = new Set(["idle", "queued", "running", "complete", "failed"]);
  state.tabSimulationStateByTour[key] = allowedStates.has(tabState) ? tabState : "idle";
  renderTourTabs();
}

function renderTourTabs() {
  if (!ui.tourTabs) {
    return;
  }
  const activeTour = normalizedTourValue(ui.tourSelect?.value || "pga");
  const buttons = ui.tourTabs.querySelectorAll(".tour-tab[data-tour]");
  buttons.forEach((button) => {
    const tour = normalizedTourValue(button.getAttribute("data-tour"));
    const active = tour === activeTour;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-selected", active ? "true" : "false");

    const badge = button.querySelector("[data-tour-badge]");
    if (!badge) {
      return;
    }
    const tabState = getTabSimulationState(tour);
    badge.textContent = tabState;
    badge.className = `tour-tab-badge state-${tabState}`;
  });
}

function queueSimulationRequest({ tour, eventId, fromAutoRefresh = false } = {}) {
  const normalizedTour = normalizedTourValue(tour || ui.tourSelect?.value || "pga");
  const normalizedEventId = String(eventId || "").trim();
  const duplicate = state.simulationQueue.some(
    (item) => item.tour === normalizedTour && item.eventId === normalizedEventId
  );
  if (!duplicate) {
    state.simulationQueue.push({
      tour: normalizedTour,
      eventId: normalizedEventId,
      fromAutoRefresh: Boolean(fromAutoRefresh),
    });
  }
  setTourTabSimulationState(normalizedTour, "queued");
}

function shouldAutoSimulateForTour(tour) {
  const tabState = getTabSimulationState(tour);
  return tabState === "idle" || tabState === "failed";
}

function hasSimulatableEvents() {
  return Array.isArray(state.events) && state.events.some((event) => event && event.simulatable);
}

function processNextQueuedSimulation() {
  if (state.simulationInFlight || state.simulationQueue.length === 0) {
    return;
  }
  const next = state.simulationQueue.shift();
  if (!next) {
    return;
  }
  const background = normalizedTourValue(next.tour) !== normalizedTourValue(ui.tourSelect?.value || "pga");
  void runSimulation({
    fromAutoRefresh: next.fromAutoRefresh,
    targetTour: next.tour,
    targetEventId: next.eventId,
    background,
  });
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
  if (ui.runLifecycleButton) {
    ui.runLifecycleButton.disabled = isBusy;
  }
  if (ui.applyRecommendationButton) {
    ui.applyRecommendationButton.disabled = isBusy;
  }
  if (ui.refreshPowerReportButton) {
    ui.refreshPowerReportButton.disabled = isBusy;
  }
  if (ui.warmStartPowerReportButton) {
    ui.warmStartPowerReportButton.disabled = isBusy;
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

function formatLifecycleState(state) {
  if (!state) {
    return "-";
  }
  return String(state).replace(/_/g, " ");
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

function formatSignedPct(value, decimals = 2) {
  if (value == null || Number.isNaN(Number(value))) {
    return "-";
  }
  const numeric = Number(value);
  const abs = Math.abs(numeric * 100).toFixed(decimals);
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

function numericOrNull(raw) {
  const numeric = Number(raw);
  if (!Number.isFinite(numeric)) {
    return null;
  }
  return numeric;
}

function winDeltaPrevForPlayer(player, trend) {
  const playerDelta = numericOrNull(player.win_delta_prev);
  if (playerDelta != null) {
    return playerDelta;
  }
  return numericOrNull(trend ? trend.delta_win_since_previous : null);
}

function winDeltaStartForPlayer(player, trend) {
  const playerDelta = numericOrNull(player.win_delta_start);
  if (playerDelta != null) {
    return playerDelta;
  }
  return numericOrNull(trend ? trend.delta_win_since_first : null);
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

function updateVersionCallouts() {
  if (ui.versionSimulationValue) {
    let simValue = "v-";
    let simMeta = "No snapshot yet.";
    if (
      state.latestResult &&
      Number.isFinite(Number(state.latestResult.simulation_version))
    ) {
      simValue = `v${Number(state.latestResult.simulation_version)}`;
      const eventLabel = state.latestResult.event_name || state.latestResult.event_id || "Latest event";
      simMeta = `${eventLabel} | ${formatDate(state.latestResult.generated_at)}`;
    } else if (
      state.latestLifecycleStatus &&
      state.latestLifecycleStatus.pre_event_snapshot_version != null
    ) {
      simValue = `v${Number(state.latestLifecycleStatus.pre_event_snapshot_version)}`;
      const eventLabel =
        state.latestLifecycleStatus.active_event_name || state.latestLifecycleStatus.active_event_id || "Active event";
      simMeta = `${eventLabel} pre-event snapshot`;
    }
    ui.versionSimulationValue.textContent = simValue;
    if (ui.versionSimulationMeta) {
      ui.versionSimulationMeta.textContent = simMeta;
    }
    if (ui.versionSimulationTile) {
      ui.versionSimulationTile.classList.add("healthy");
    }
  }

  if (ui.versionRetrainValue) {
    let retrainValue = "v0";
    let retrainMeta = "Not trained.";
    let degraded = false;
    if (state.latestLearningStatus) {
      const version = Number(state.latestLearningStatus.calibration_version || 0);
      retrainValue = version > 0 ? `v${version}` : "v0";
      const winMarket = findMarketStatus(state.latestLearningStatus, "win");
      const before = Number(winMarket?.brier_before);
      const after = Number(winMarket?.brier_after);
      if (Number.isFinite(before) && Number.isFinite(after)) {
        if (after > before + 1e-9) {
          degraded = true;
          retrainMeta = `Win Brier degraded ${before.toFixed(4)} -> ${after.toFixed(4)}.`;
        } else {
          retrainMeta = `Win Brier ${before.toFixed(4)} -> ${after.toFixed(4)}.`;
        }
      } else {
        retrainMeta = `Resolved events=${Number(state.latestLearningStatus.resolved_events || 0)} | pending=${Number(state.latestLearningStatus.pending_events || 0)}`;
      }
    }
    ui.versionRetrainValue.textContent = retrainValue;
    if (ui.versionRetrainMeta) {
      ui.versionRetrainMeta.textContent = retrainMeta;
    }
    if (ui.versionRetrainTile) {
      ui.versionRetrainTile.classList.toggle("degraded", degraded);
      ui.versionRetrainTile.classList.toggle("healthy", !degraded);
    }
  }
}

function renderLearningStatus(payload) {
  state.latestLearningStatus = payload;
  const winMarket = findMarketStatus(payload, "win");
  let brierSummary = "win Brier unavailable (need resolved outcomes)";
  if (winMarket && winMarket.samples > 0) {
    const before = Number(winMarket.brier_before);
    const after = Number(winMarket.brier_after);
    if (Number.isFinite(before) && Number.isFinite(after)) {
      if (after > before + 1e-9) {
        brierSummary = `win Brier degraded ${formatScore(before)} -> ${formatScore(after)} (calibration quarantined; run retrain)`;
      } else {
        brierSummary = `win Brier ${formatScore(before)} -> ${formatScore(after)}`;
      }
    }
  }
  setLearningStatus(
    `Learning v${payload.calibration_version} | resolved events=${payload.resolved_events} | pending=${payload.pending_events} | ${brierSummary}`
  );
  updateVersionCallouts();
}

function renderLifecycleStatus(payload) {
  state.latestLifecycleStatus = payload;
  const activeLabel =
    payload.active_event_name && payload.active_event_id
      ? `${payload.active_event_name} (${payload.active_event_id}:${payload.active_event_year || "-"})`
      : "No active event";
  const preEventLabel = payload.pre_event_snapshot_ready
    ? `yes (v${payload.pre_event_snapshot_version || "-"})`
    : "no";
  const runNote = payload.last_run_note ? ` | ${payload.last_run_note}` : "";
  setLifecycleStatus(
    `Lifecycle: active=${activeLabel} | state=${payload.active_event_state || "-"} | pre-event snapshot=${preEventLabel} | pending=${payload.pending_events}${runNote}`
  );
  if (state.autoSimulationTimerId != null && !state.simulationInFlight && !shouldAutoRunSimulationNow()) {
    setSimulationAutomationStatus(autoSimulationPausedMessage(), false);
  }
  if (!state.simulationInFlight && shouldAutoRunSimulationNow() && state.latestResult) {
    void refreshLiveScoresOnly({ silent: true });
  }
  renderLifecycleHistory(payload.recent_events || []);
  updateVersionCallouts();
}

function renderLifecycleHistory(events) {
  if (!ui.lifecycleHistoryBody) {
    return;
  }
  ui.lifecycleHistoryBody.innerHTML = "";
  if (!Array.isArray(events) || events.length === 0) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 7;
    td.textContent = "No lifecycle events yet.";
    tr.appendChild(td);
    ui.lifecycleHistoryBody.appendChild(tr);
    return;
  }
  events.forEach((event) => {
    const tr = document.createElement("tr");
    const cells = [
      event.event_name || event.event_id || "-",
      event.event_year != null ? String(event.event_year) : "-",
      formatLifecycleState(event.state),
      event.pre_event_snapshot_version != null ? `v${event.pre_event_snapshot_version}` : "-",
      event.retrain_version != null ? `v${event.retrain_version}` : "-",
      event.outcomes_source || "-",
      formatDate(event.updated_at),
    ];
    cells.forEach((value) => {
      const td = document.createElement("td");
      td.textContent = value;
      tr.appendChild(td);
    });
    ui.lifecycleHistoryBody.appendChild(tr);
  });
}

function formatPowerScore(value) {
  if (value == null || Number.isNaN(Number(value))) {
    return "-";
  }
  return Number(value).toFixed(2);
}

function compactEventLabel(eventName, eventDate) {
  const name = String(eventName || "").trim();
  if (!name) {
    return String(eventDate || "-");
  }
  if (name.length <= 18) {
    return name;
  }
  return `${name.slice(0, 17)}…`;
}

function clearPowerRankingReport(message = "No power ranking data available yet.") {
  state.powerRankingReport = null;
  if (ui.powerRankingChart) {
    ui.powerRankingChart.innerHTML = "";
    const empty = document.createElement("div");
    empty.className = "bump-chart-empty";
    empty.textContent = message;
    ui.powerRankingChart.appendChild(empty);
  }
  if (ui.powerRankingTableBody) {
    ui.powerRankingTableBody.innerHTML = "";
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 5;
    td.textContent = message;
    tr.appendChild(td);
    ui.powerRankingTableBody.appendChild(tr);
  }
}

function renderPowerRankingTable(payload) {
  if (!ui.powerRankingTableBody) {
    return;
  }
  ui.powerRankingTableBody.innerHTML = "";
  const players = Array.isArray(payload?.players) ? payload.players : [];
  if (players.length === 0) {
    clearPowerRankingReport("No ranking rows to display.");
    return;
  }
  players.forEach((player) => {
    const tr = document.createElement("tr");
    const latestPoint = Array.isArray(player.points) && player.points.length > 0
      ? player.points[player.points.length - 1]
      : null;
    const cells = [
      player.latest_rank != null ? String(player.latest_rank) : "-",
      player.player_name || "-",
      formatPowerScore(player.latest_score),
      latestPoint?.event_name || latestPoint?.event_id || "-",
      latestPoint?.rank != null ? String(latestPoint.rank) : "-",
    ];
    cells.forEach((value, idx) => {
      const td = document.createElement("td");
      td.textContent = value;
      if (idx === 0 || idx === 2 || idx === 4) {
        td.className = "num";
      }
      tr.appendChild(td);
    });
    ui.powerRankingTableBody.appendChild(tr);
  });
}

function renderPowerRankingChart(payload) {
  if (!ui.powerRankingChart) {
    return;
  }
  ui.powerRankingChart.innerHTML = "";
  if (typeof window.d3 === "undefined") {
    clearPowerRankingReport("D3 did not load; unable to render bump chart.");
    return;
  }
  const d3 = window.d3;
  const events = Array.isArray(payload?.events) ? payload.events : [];
  const players = Array.isArray(payload?.players) ? payload.players : [];
  if (events.length < 2 || players.length === 0) {
    clearPowerRankingReport("Need at least two events and one player to draw bump chart.");
    return;
  }

  const eventKey = (event) => `${event.event_id}:${event.event_year}`;
  const eventKeys = events.map((event) => eventKey(event));
  const eventByKey = {};
  events.forEach((event) => {
    eventByKey[eventKey(event)] = event;
  });

  const orderedSeries = players.map((player) => {
    const pointByKey = {};
    (Array.isArray(player.points) ? player.points : []).forEach((point) => {
      pointByKey[`${point.event_id}:${point.event_year}`] = point;
    });
    return {
      player_name: player.player_name || "-",
      latest_rank: player.latest_rank,
      latest_score: player.latest_score,
      points: eventKeys.map((key) => {
        const point = pointByKey[key] || null;
        return {
          event_key: key,
          rank: point && point.rank != null ? Number(point.rank) : null,
          score: point && point.score != null ? Number(point.score) : null,
          event_score: point && point.event_score != null ? Number(point.event_score) : null,
        };
      }),
    };
  });

  const chartWidth = Math.max(760, ui.powerRankingChart.clientWidth || 760);
  const margin = { top: 24, right: 30, bottom: 56, left: 52 };
  const maxPointRank = orderedSeries.reduce((acc, row) => {
    const rowMax = row.points.reduce((rowAcc, point) => {
      const rank = Number(point.rank);
      return Number.isFinite(rank) ? Math.max(rowAcc, rank) : rowAcc;
    }, 0);
    return Math.max(acc, rowMax);
  }, 0);
  const maxRank = Math.max(
    maxPointRank,
    orderedSeries.reduce(
      (acc, row) => Math.max(acc, Number.isFinite(Number(row.latest_rank)) ? Number(row.latest_rank) : 0),
      0
    ),
    orderedSeries.length,
    Number(payload?.top_n || 10)
  );
  const chartHeight = Math.max(420, 160 + (maxRank * 24));

  const xScale = d3.scalePoint(eventKeys, [margin.left, chartWidth - margin.right]).padding(0.45);
  const yScale = d3.scaleLinear([1, maxRank], [margin.top, chartHeight - margin.bottom]);
  const line = d3
    .line()
    .defined((point) => point.rank != null && Number.isFinite(point.rank))
    .x((point) => xScale(point.event_key))
    .y((point) => yScale(point.rank))
    .curve(d3.curveMonotoneX);

  const colorPalette = [
    ...(d3.schemeTableau10 || []),
    ...(d3.schemeSet3 || []),
    ...(d3.schemePaired || []),
  ];
  const colorScale = d3
    .scaleOrdinal()
    .domain(orderedSeries.map((row) => row.player_name))
    .range(colorPalette.length > 0 ? colorPalette : ["#0c7a70"]);

  const svg = d3
    .select(ui.powerRankingChart)
    .append("svg")
    .attr("viewBox", `0 0 ${chartWidth} ${chartHeight}`)
    .attr("role", "img")
    .attr("aria-label", "Power ranking bump chart");

  const yTickStep = maxRank <= 12 ? 1 : maxRank <= 25 ? 2 : 5;
  const yTicks = d3.range(1, maxRank + 1, yTickStep);
  svg
    .append("g")
    .attr("class", "bump-grid")
    .selectAll("line")
    .data(yTicks)
    .join("line")
    .attr("x1", margin.left)
    .attr("x2", chartWidth - margin.right)
    .attr("y1", (rank) => yScale(rank))
    .attr("y2", (rank) => yScale(rank));

  const seriesGroup = svg.append("g").attr("class", "bump-series");
  const tooltip = document.createElement("div");
  tooltip.className = "bump-tooltip";
  tooltip.hidden = true;
  ui.powerRankingChart.appendChild(tooltip);

  const hideTooltip = () => {
    tooltip.hidden = true;
  };
  const showTooltip = (event, payloadRow, point) => {
    const meta = eventByKey[point.event_key] || {};
    const scoreText = point.score != null ? point.score.toFixed(2) : "-";
    const eventScoreText = point.event_score != null ? point.event_score.toFixed(2) : "-";
    tooltip.innerHTML = [
      `<strong>${payloadRow.player_name}</strong>`,
      `${meta.event_name || meta.event_id || point.event_key}`,
      `Rank: ${point.rank != null ? point.rank : "-"}`,
      `Power: ${scoreText}`,
      `Event Score: ${eventScoreText}`,
    ].join("<br />");
    const bounds = ui.powerRankingChart.getBoundingClientRect();
    const left = event.clientX - bounds.left + 12;
    const top = event.clientY - bounds.top + 12;
    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${top}px`;
    tooltip.hidden = false;
  };

  const lines = seriesGroup
    .selectAll("path")
    .data(orderedSeries)
    .join("path")
    .attr("class", "bump-line")
    .attr("d", (row) => line(row.points))
    .attr("stroke", (row) => colorScale(row.player_name))
    .on("mouseenter", function onEnter(_, row) {
      lines.classed("is-highlighted", false);
      d3.select(this).classed("is-highlighted", true);
      const lastPoint = row.points.find((point) => point.rank != null) || row.points[0];
      if (lastPoint) {
        showTooltip({ clientX: margin.left, clientY: margin.top }, row, lastPoint);
      }
    })
    .on("mouseleave", function onLeave() {
      d3.select(this).classed("is-highlighted", false);
      hideTooltip();
    });

  const dotRows = orderedSeries.flatMap((row) =>
    row.points
      .filter((point) => point.rank != null && Number.isFinite(point.rank))
      .map((point) => ({
        player_name: row.player_name,
        point,
      }))
  );
  seriesGroup
    .selectAll("circle")
    .data(dotRows)
    .join("circle")
    .attr("class", "bump-dot")
    .attr("cx", (row) => xScale(row.point.event_key))
    .attr("cy", (row) => yScale(row.point.rank))
    .attr("r", 3.4)
    .attr("fill", (row) => colorScale(row.player_name))
    .on("mousemove", (event, row) => {
      showTooltip(event, { player_name: row.player_name }, row.point);
    })
    .on("mouseleave", hideTooltip);

  const xAxis = d3
    .axisBottom(xScale)
    .tickFormat((key) => {
      const event = eventByKey[key] || {};
      return compactEventLabel(event.event_name, event.event_date);
    });
  svg
    .append("g")
    .attr("class", "bump-axis")
    .attr("transform", `translate(0, ${chartHeight - margin.bottom})`)
    .call(xAxis)
    .selectAll("text")
    .attr("transform", "translate(0,8) rotate(-22)")
    .style("text-anchor", "end");

  svg
    .append("g")
    .attr("class", "bump-axis")
    .attr("transform", `translate(${margin.left},0)`)
    .call(d3.axisLeft(yScale).tickValues(yTicks).tickFormat((value) => value));

  const legend = document.createElement("div");
  legend.className = "bump-legend";
  orderedSeries.forEach((row) => {
    const item = document.createElement("span");
    item.className = "bump-legend-item";
    item.style.borderColor = colorScale(row.player_name);
    item.textContent = `#${row.latest_rank ?? "-"} ${row.player_name}`;
    legend.appendChild(item);
  });
  ui.powerRankingChart.appendChild(legend);
}

function renderPowerRankingReport(payload) {
  renderPowerRankingChart(payload);
  renderPowerRankingTable(payload);
}

async function loadPowerRankingReport(silent = false) {
  if (!ui.powerRankingStatus) {
    return;
  }
  const lookback = Math.max(3, Math.min(60, Number.parseInt(ui.powerLookbackInput?.value || "12", 10) || 12));
  const topN = Math.max(5, Math.min(40, Number.parseInt(ui.powerTopNInput?.value || "12", 10) || 12));
  if (!silent) {
    setPowerRankingStatus("Loading power rankings...", true);
  }
  try {
    const tour = encodeURIComponent(ui.tourSelect.value);
    const response = await fetch(
      `/reports/power-rankings?tour=${tour}&lookback_events=${lookback}&top_n=${topN}`
    );
    if (response.status === 404) {
      let detail = "No power ranking history available yet for this tour.";
      try {
        const errPayload = await response.json();
        if (errPayload?.detail) {
          detail = String(errPayload.detail);
        }
      } catch (_) {
        // Keep default detail when body is not JSON.
      }
      clearPowerRankingReport(detail);
      setPowerRankingStatus(detail, false);
      return;
    }
    if (!response.ok) {
      let detail = `Unable to load power rankings (${response.status})`;
      try {
        const errPayload = await response.json();
        if (errPayload?.detail) {
          detail = String(errPayload.detail);
        }
      } catch (_) {
        // Keep default detail when body is not JSON.
      }
      throw new Error(detail);
    }

    const payload = await response.json();
    state.powerRankingReport = payload;
    renderPowerRankingReport(payload);
    setPowerRankingStatus(
      `Power rankings loaded: ${payload.players.length} players across ${payload.events.length} events (${String(payload.tour || ui.tourSelect.value).toUpperCase()}).`
    );
  } catch (error) {
    clearPowerRankingReport("Unable to load power ranking report.");
    setPowerRankingStatus("Unable to load power rankings.", false);
    if (!silent) {
      setError(error.message || "Unexpected error while loading power rankings.");
    }
  }
}

async function warmStartPowerRankingReports() {
  setError();
  setBusy(true);
  setPowerRankingStatus("Running warm-start snapshots for all tours...", true);

  try {
    const simulationsRaw = Number.parseInt(ui.simulationsInput?.value || "100000", 10) || 100000;
    const eventYear = Number.parseInt(ui.currentSeasonInput?.value || "", 10);
    const requestBody = {
      tours: SUPPORTED_TOURS.map((entry) => entry.value),
      simulations: Math.max(500, Math.min(250000, simulationsRaw)),
      force: false,
    };
    if (!Number.isNaN(eventYear)) {
      requestBody.event_year = eventYear;
    }

    const response = await fetch("/reports/power-rankings/warm-start", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      let detail = `Warm-start failed (${response.status})`;
      try {
        const errPayload = await response.json();
        if (errPayload?.detail) {
          detail = String(errPayload.detail);
        }
      } catch (_) {
        // Keep default detail when body is not JSON.
      }
      throw new Error(detail);
    }

    const payload = await response.json();
    const rows = Array.isArray(payload?.results) ? payload.results : [];
    const simulated = rows.filter((row) => String(row?.status || "") === "simulated");
    const skipped = rows.filter((row) => String(row?.status || "") === "skipped_existing");
    const failed = rows.filter((row) => String(row?.status || "") === "failed");

    const failedTours = failed
      .map((row) => String(row?.tour || "").toUpperCase())
      .filter((value) => value.length > 0);

    const summary = `Warm-start complete: simulated=${simulated.length}, skipped=${skipped.length}, failed=${failed.length}.`;
    setStatus(summary);
    if (failedTours.length > 0) {
      setError(`Warm-start failures: ${failedTours.join(", ")}. Check status text for details.`);
    }

    await loadPowerRankingReport(false);
  } catch (error) {
    setPowerRankingStatus("Warm-start failed.", false);
    setError(error.message || "Unexpected error while warm-starting reports.");
  } finally {
    setBusy(false);
  }
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
      const versionText = Number.isFinite(Number(lastPoint.simulation_version))
        ? `v${Number(lastPoint.simulation_version)}`
        : "v?";
      meta.textContent = `Latest snapshot ${versionText}: ${formatDate(lastPoint.created_at)}`;
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
  const requestTour = normalizedTourValue(ui.tourSelect.value);
  const requestToken = ++state.eventsRequestToken;
  const isLatestRequest = () => requestToken === state.eventsRequestToken;

  setError();
  setFormStatus("");
  setBusy(true);
  setStatus("Loading upcoming events...", true);
  try {
    const tour = encodeURIComponent(requestTour);
    const response = await fetch(`/events/upcoming?tour=${tour}&limit=40`);
    if (!isLatestRequest()) {
      return;
    }
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
    if (!isLatestRequest()) {
      return;
    }
    state.events = events;
    updateEventSelect(events);
    const simulatableCount = events.filter((event) => event.simulatable).length;
    setStatus(
      `Loaded ${events.length} events (${simulatableCount} currently simulatable) for ${requestTour.toUpperCase()}.`
    );
  } catch (error) {
    if (!isLatestRequest()) {
      return;
    }
    state.events = [];
    updateEventSelect([]);
    setStatus("Unable to load events.");
    setError(error.message || "Unexpected error while fetching events.");
  } finally {
    if (isLatestRequest()) {
      setBusy(false);
    }
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

async function loadLifecycleStatus(silent = false) {
  if (!silent) {
    setLifecycleStatus("Loading lifecycle status...", true);
  }
  try {
    const tour = encodeURIComponent(ui.tourSelect.value);
    const response = await fetch(`/lifecycle/status?tour=${tour}`);
    if (!response.ok) {
      let detail = `Unable to load lifecycle status (${response.status})`;
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
    renderLifecycleStatus(payload);
  } catch (error) {
    if (!silent) {
      setLifecycleStatus("Unable to load lifecycle status.");
      setError(error.message || "Unexpected error while loading lifecycle status.");
    }
  }
}

async function runLifecycleNow() {
  setError();
  setBusy(true);
  setLifecycleStatus("Running lifecycle automation cycle...", true);
  try {
    const tour = encodeURIComponent(ui.tourSelect.value);
    const response = await fetch(`/lifecycle/run?tour=${tour}`, {
      method: "POST",
    });
    if (!response.ok) {
      let detail = `Lifecycle run failed (${response.status})`;
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
    renderLifecycleStatus(payload);
    setStatus("Lifecycle automation cycle completed.");
    void loadLearningStatus(true);
  } catch (error) {
    setStatus("Lifecycle automation cycle failed.");
    setError(error.message || "Unexpected error while running lifecycle automation.");
  } finally {
    setBusy(false);
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
      `provisional_outcomes=${payload.provisional_outcomes_fetched || 0}`,
      `awaiting_official=${payload.awaiting_outcomes_count || 0}`,
      `retrain=${payload.retrain_executed ? "yes" : "no"}`,
      `version=v${payload.calibration_version}`,
    ];
    if (payload.sync_note) {
      statusParts.push(payload.sync_note);
    }
    setStatus(`Learning sync complete. ${statusParts.join(" | ")}`);
    void loadLifecycleStatus(true);
    void loadPowerRankingReport(true);
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
      max_players: "200",
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

function mergeLiveScoresIntoLatestResult(payload) {
  if (!state.latestResult || !Array.isArray(state.latestResult.players)) {
    return 0;
  }
  const livePlayers = Array.isArray(payload?.players) ? payload.players : [];
  if (livePlayers.length === 0) {
    return 0;
  }

  const liveByKey = {};
  livePlayers.forEach((row) => {
    const idKey = canonicalPlayerKey(row.player_id, "");
    const nameKey = canonicalPlayerKey("", row.player_name);
    if (idKey) {
      liveByKey[idKey] = row;
    }
    if (nameKey) {
      liveByKey[nameKey] = row;
    }
  });

  let updated = 0;
  state.latestResult.players = state.latestResult.players.map((player) => {
    const idKey = canonicalPlayerKey(player.player_id, "");
    const nameKey = canonicalPlayerKey("", player.player_name);
    const liveRow = (idKey && liveByKey[idKey]) || (nameKey && liveByKey[nameKey]);
    if (!liveRow) {
      return player;
    }
    updated += 1;
    return {
      ...player,
      current_position:
        liveRow.current_position != null ? liveRow.current_position : player.current_position,
      current_score_to_par:
        liveRow.current_score_to_par != null
          ? liveRow.current_score_to_par
          : player.current_score_to_par,
      current_thru: liveRow.current_thru != null ? liveRow.current_thru : player.current_thru,
      today_score_to_par:
        liveRow.today_score_to_par != null ? liveRow.today_score_to_par : player.today_score_to_par,
      round_scores:
        Array.isArray(liveRow.round_scores) && liveRow.round_scores.length > 0
          ? liveRow.round_scores
          : player.round_scores,
      hole_scores:
        Array.isArray(liveRow.hole_scores) && liveRow.hole_scores.length > 0
          ? liveRow.hole_scores
          : player.hole_scores,
    };
  });

  if (updated > 0) {
    renderTable(state.latestResult.players);
  }
  return updated;
}

async function refreshLiveScoresOnly({ silent = true } = {}) {
  if (!state.latestResult) {
    return;
  }
  const tourValue = state.latestResult.tour || ui.tourSelect.value;
  const selectedEventId = (state.latestResult.event_id || ui.eventSelect?.value || "").trim();
  const query = new URLSearchParams({ tour: tourValue });
  if (selectedEventId) {
    query.set("event_id", selectedEventId);
  }
  try {
    const response = await fetch(`/live/scores?${query.toString()}`);
    if (!response.ok) {
      let detail = `Unable to refresh live scores (${response.status})`;
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
    const changedRows = mergeLiveScoresIntoLatestResult(payload);
    if (!silent && changedRows > 0) {
      setStatus(`Live scores refreshed (${changedRows} players).`);
    }
  } catch (error) {
    if (!silent) {
      setError(error.message || "Unexpected error while refreshing live scores.");
    }
  }
}

async function loadLatestSnapshotFromDb({ silent = true, runIfMissing = true } = {}) {
  const tourValue = normalizedTourValue(ui.tourSelect.value);
  const selectedEventId = (ui.eventSelect?.value || "").trim();
  if (!selectedEventId && !hasSimulatableEvents()) {
    setTourTabSimulationState(tourValue, "idle");
    setStatus(
      `No simulatable event currently available for ${tourValue.toUpperCase()}. Waiting for DataGolf current-week feed update.`
    );
    return;
  }
  const query = new URLSearchParams({ tour: tourValue });
  if (selectedEventId) {
    query.set("event_id", selectedEventId);
  }
  try {
    const response = await fetch(`/learning/latest-snapshot?${query.toString()}`);
    if (response.status === 404) {
      let detail = "No stored snapshot found for selected event.";
      try {
        const errPayload = await response.json();
        if (errPayload?.detail) {
          detail = String(errPayload.detail);
        }
      } catch (_) {
        // Keep default detail message when body is not JSON.
      }
      if (runIfMissing && !state.simulationInFlight) {
        setStatus(
          `${detail} Running initial simulation...`,
          true
        );
        await runSimulation({
          fromAutoRefresh: false,
          targetTour: tourValue,
          targetEventId: selectedEventId,
          background: false,
        });
        return;
      }
      if (runIfMissing && state.simulationInFlight) {
        queueSimulationRequest({
          tour: tourValue,
          eventId: selectedEventId,
          fromAutoRefresh: false,
        });
      }
      if (!silent) {
        setStatus(detail);
      }
      setTourTabSimulationState(tourValue, "idle");
      return;
    }
    if (!response.ok) {
      let detail = `Unable to load latest snapshot (${response.status})`;
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
    const hydrationCheck = snapshotHydrationNeedsFreshSimulation(payload);
    if (hydrationCheck.needsRefresh && runIfMissing && !state.simulationInFlight) {
      setStatus(`${hydrationCheck.reason} Running refresh simulation...`, true);
      await runSimulation({
        fromAutoRefresh: false,
        targetTour: tourValue,
        targetEventId: selectedEventId,
        background: false,
      });
      return;
    }
    if (hydrationCheck.needsRefresh && runIfMissing && state.simulationInFlight) {
      queueSimulationRequest({
        tour: tourValue,
        eventId: selectedEventId,
        fromAutoRefresh: false,
      });
    }
    renderResult(payload);
    await loadEventTrendsForCurrentEvent({ silent: true });
    setTourTabSimulationState(tourValue, "complete");
    setStatus(
      `Loaded latest snapshot from DB: v${Number(payload.simulation_version || 1)} (${Number(payload.simulations || 0).toLocaleString()} sims).`
    );
    void loadPowerRankingReport(true);
  } catch (error) {
    setTourTabSimulationState(tourValue, "failed");
    if (!silent) {
      setError(error.message || "Unexpected error while loading latest snapshot.");
    }
  }
}

function stopAutoRefresh() {
  if (state.autoRefreshTimerId != null) {
    window.clearInterval(state.autoRefreshTimerId);
    state.autoRefreshTimerId = null;
  }
}

function stopAutoSimulation() {
  if (state.autoSimulationTimerId != null) {
    window.clearInterval(state.autoSimulationTimerId);
    state.autoSimulationTimerId = null;
  }
}

function stopLifecyclePoll() {
  if (state.lifecyclePollTimerId != null) {
    window.clearInterval(state.lifecyclePollTimerId);
    state.lifecyclePollTimerId = null;
  }
}

function stopLiveScorePoll() {
  if (state.liveScorePollTimerId != null) {
    window.clearInterval(state.liveScorePollTimerId);
    state.liveScorePollTimerId = null;
  }
}

function startLifecyclePoll() {
  stopLifecyclePoll();
  state.lifecyclePollTimerId = window.setInterval(() => {
    void loadLifecycleStatus(true);
  }, 45000);
}

function startLiveScorePoll() {
  stopLiveScorePoll();
  state.liveScorePollTimerId = window.setInterval(() => {
    if (state.simulationInFlight) {
      return;
    }
    if (!shouldAutoRunSimulationNow()) {
      return;
    }
    void refreshLiveScoresOnly({ silent: true });
  }, LIVE_SCORE_POLL_INTERVAL_SECONDS * 1000);
}

function startAutoSimulation() {
  stopAutoSimulation();
  setSimulationAutomationStatus(
    `Simulation automation: enabled every ${AUTOMATION_SIMULATION_INTERVAL_SECONDS}s (in-play only).`,
    false
  );
  const runOnce = () => {
    if (state.simulationInFlight) {
      return;
    }
    if (!shouldAutoRunSimulationNow()) {
      setSimulationAutomationStatus(autoSimulationPausedMessage(), false);
      return;
    }
    void runSimulation({ fromAutoRefresh: true });
  };
  window.setTimeout(runOnce, 3000);
  state.autoSimulationTimerId = window.setInterval(
    runOnce,
    AUTOMATION_SIMULATION_INTERVAL_SECONDS * 1000
  );
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
    if (!shouldAutoRunSimulationNow()) {
      return;
    }
    void runSimulation({ fromAutoRefresh: true });
  }, seconds * 1000);
  setStatus(`Live auto-refresh enabled every ${seconds}s (runs in-play only).`);
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
  if (ui.snapshotLabel) {
    ui.snapshotLabel.textContent = `v${Number(payload.simulation_version || 1)}`;
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
  updateVersionCallouts();
}

async function runSimulation(options = {}) {
  const fromAutoRefresh = Boolean(options.fromAutoRefresh);
  const targetTour = normalizedTourValue(options.targetTour || ui.tourSelect.value);
  const activeTour = normalizedTourValue(ui.tourSelect.value);
  const targetEventId = String(
    options.targetEventId != null ? options.targetEventId : (targetTour === activeTour ? ui.eventSelect.value : "")
  ).trim();
  const background = Boolean(options.background || targetTour !== activeTour);
  if (targetTour === activeTour && !targetEventId && !hasSimulatableEvents()) {
    setTourTabSimulationState(targetTour, "idle");
    setStatus(
      `No simulatable event currently available for ${targetTour.toUpperCase()}. Waiting for DataGolf current-week feed update.`
    );
    return;
  }

  if (state.simulationInFlight || ui.simulateButton.disabled) {
    queueSimulationRequest({
      tour: targetTour,
      eventId: targetEventId,
      fromAutoRefresh,
    });
    if (!fromAutoRefresh && targetTour === activeTour) {
      setStatus(`Simulation queued for ${targetTour.toUpperCase()}.`, true);
    }
    return;
  }

  state.simulationInFlight = true;
  state.currentSimulationTour = targetTour;
  setTourTabSimulationState(targetTour, "running");
  if (!background) {
    setError();
    setFormStatus("Loading seasonal form data...", true);
    if (fromAutoRefresh) {
      setSimulationAutomationStatus("Simulation automation: running scheduled simulation...", true);
    }
    setStatus(
      fromAutoRefresh ? "Running auto-refresh simulation..." : "Running simulation...",
      true
    );
  } else {
    setSimulationAutomationStatus(
      `Simulation queue: running ${targetTour.toUpperCase()}...`,
      true
    );
  }
  setBusy(true);

  try {
    const simulations = Number.parseInt(ui.simulationsInput.value, 10) || 10000;
    const minSimulations = Number.parseInt(ui.minSimulationsInput.value, 10) || 250000;
    const simulationBatchSize = Number.parseInt(ui.simulationBatchSizeInput.value, 10) || 1000;
    const cutSize = Number.parseInt(ui.cutSizeInput.value, 10) || 70;
    const meanReversion = Number.parseFloat(ui.meanReversionInput.value) || 0.1;
    const sharedRoundShockSigma = Number.parseFloat(ui.sharedRoundShockInput.value) || 0.35;
    const useAdaptiveSimulation = ui.useAdaptiveSimulationSelect.value === "yes";
    const ciConfidence = Number.parseFloat(ui.ciConfidenceInput.value) || 0.975;
    const ciHalfWidthTarget =
      Number.parseFloat(ui.ciHalfWidthTargetInput.value) || 0.0015;
    const ciTopN = Number.parseInt(ui.ciTopNInput.value, 10) || 15;
    const useInPlayConditioning = ui.useInPlayConditioningSelect.value === "yes";
    const useSeasonalForm = ui.useSeasonalFormSelect.value === "yes";
    const baselineSeason = Number.parseInt(ui.baselineSeasonInput.value, 10);
    const currentSeason = Number.parseInt(ui.currentSeasonInput.value, 10);
    const seasonalWeight = Number.parseFloat(ui.seasonalWeightInput.value);
    const currentSeasonWeight = Number.parseFloat(ui.currentSeasonWeightInput.value);
    const formDeltaWeight = Number.parseFloat(ui.formDeltaWeightInput.value);
    const seedRaw = ui.seedInput.value.trim();

    const requestBody = {
      tour: targetTour,
      event_id: targetEventId || null,
      resolution_mode: ui.resolutionModeSelect.value || "fixed_cap",
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
        try {
          const bodyText = (await response.text()).trim();
          if (bodyText.length > 0) {
            detail = `${detail}: ${bodyText.slice(0, 220)}`;
          }
        } catch (__unused) {
          // Keep default detail message when body is not text-readable.
        }
      }
      throw new Error(detail);
    }

    const payload = await response.json();
    const shouldRenderNow = targetTour === normalizedTourValue(ui.tourSelect.value);
    if (shouldRenderNow) {
      renderResult(payload);
      await loadEventTrendsForCurrentEvent({ silent: true });
    }
    setTourTabSimulationState(targetTour, "complete");
    const statusBits = [];
    if (payload.stop_reason) {
      statusBits.push(`stop=${payload.stop_reason}`);
    }
    if (payload.simulations != null) {
      if (payload.requested_simulations && payload.requested_simulations !== payload.simulations) {
        statusBits.push(
          `sims=${Number(payload.simulations).toLocaleString()}/${Number(payload.requested_simulations).toLocaleString()}`
        );
      } else {
        statusBits.push(`sims=${Number(payload.simulations).toLocaleString()}`);
      }
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
    if (payload.simulation_version != null) {
      statusBits.push(`snapshot=v${Number(payload.simulation_version)}`);
    }
    if (shouldRenderNow) {
      setStatus(statusBits.length > 0 ? `Simulation complete. ${statusBits.join(" | ")}` : "Simulation complete.");
      if (fromAutoRefresh) {
        setSimulationAutomationStatus(
          `Simulation automation: last run ${new Date().toLocaleTimeString()} | next run in ${AUTOMATION_SIMULATION_INTERVAL_SECONDS}s.`
        );
      }
      void loadLearningStatus(true);
      void loadLifecycleStatus(true);
      void loadPowerRankingReport(true);
    }
  } catch (error) {
    setTourTabSimulationState(targetTour, "failed");
    if (targetTour === normalizedTourValue(ui.tourSelect.value)) {
      setStatus("Simulation failed.");
      if (fromAutoRefresh) {
        setSimulationAutomationStatus(
          "Simulation automation: latest scheduled run failed (see error).",
          false
        );
      }
      setError(error.message || "Unexpected error while running the simulation.");
    }
  } finally {
    setBusy(false);
    state.simulationInFlight = false;
    state.currentSimulationTour = null;
    processNextQueuedSimulation();
  }
}

function bindEvents() {
  ui.loadEventsButton.addEventListener("click", loadEvents);
  ui.simulateButton.addEventListener("click", () => {
    void runSimulation({ fromAutoRefresh: false });
  });
  if (ui.syncLearningButton) {
    ui.syncLearningButton.addEventListener("click", syncLearningAndRetrain);
  }
  if (ui.refreshLearningButton) {
    ui.refreshLearningButton.addEventListener("click", () => loadLearningStatus());
  }
  if (ui.runLifecycleButton) {
    ui.runLifecycleButton.addEventListener("click", runLifecycleNow);
  }
  if (ui.refreshPowerReportButton) {
    ui.refreshPowerReportButton.addEventListener("click", () => {
      void loadPowerRankingReport(false);
    });
  }
  if (ui.warmStartPowerReportButton) {
    ui.warmStartPowerReportButton.addEventListener("click", () => {
      void warmStartPowerRankingReports();
    });
  }
  if (ui.powerLookbackInput) {
    ui.powerLookbackInput.addEventListener("change", () => {
      void loadPowerRankingReport(false);
    });
  }
  if (ui.powerTopNInput) {
    ui.powerTopNInput.addEventListener("change", () => {
      void loadPowerRankingReport(false);
    });
  }
  if (ui.tourTabs) {
    ui.tourTabs.addEventListener("click", (event) => {
      const tab = event.target instanceof Element ? event.target.closest(".tour-tab[data-tour]") : null;
      if (!tab) {
        return;
      }
      const selectedTour = normalizedTourValue(tab.getAttribute("data-tour"));
      if (selectedTour === normalizedTourValue(ui.tourSelect.value)) {
        return;
      }
      ui.tourSelect.value = selectedTour;
      ui.tourSelect.dispatchEvent(new Event("change"));
    });
  }
  ui.tourSelect.addEventListener("change", () => {
    renderTourTabs();
    resetEventTrends();
    const selectedTour = normalizedTourValue(ui.tourSelect.value);
    const allowAutoSimulate = shouldAutoSimulateForTour(selectedTour);
    void loadEvents().then(() => {
      startAutoSimulation();
      void loadLatestSnapshotFromDb({
        silent: true,
        runIfMissing: allowAutoSimulate,
      });
    });
    void loadLearningStatus(true);
    void loadLifecycleStatus(true);
    void loadPowerRankingReport(true);
  });
  if (ui.eventSelect) {
    ui.eventSelect.addEventListener("change", () => {
      resetEventTrends();
      void loadLatestSnapshotFromDb({
        silent: true,
        runIfMissing: shouldAutoSimulateForTour(ui.tourSelect.value),
      });
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
  ui.resolutionModeSelect.value = "fixed_cap";
  if (ui.useAdaptiveSimulationSelect) {
    ui.useAdaptiveSimulationSelect.value = "no";
  }
  applyTableHeaderTooltips();
  applyControlTooltips();
  bindEvents();
  ui.meanReversionValue.textContent = Number.parseFloat(ui.meanReversionInput.value).toFixed(2);
  ui.sharedRoundShockValue.textContent = Number.parseFloat(ui.sharedRoundShockInput.value).toFixed(2);
  ui.ciConfidenceValue.textContent = Number.parseFloat(ui.ciConfidenceInput.value).toFixed(3);
  ui.seasonalWeightValue.textContent = Number.parseFloat(ui.seasonalWeightInput.value).toFixed(2);
  ui.currentSeasonWeightValue.textContent = Number.parseFloat(ui.currentSeasonWeightInput.value).toFixed(2);
  ui.formDeltaWeightValue.textContent = Number.parseFloat(ui.formDeltaWeightInput.value).toFixed(2);
  SUPPORTED_TOURS.forEach((tour) => {
    state.tabSimulationStateByTour[tour.value] = "idle";
  });
  renderTourTabs();
  updateVersionCallouts();
  const defaultTour = normalizedTourValue(ui.tourSelect.value);
  void loadEvents().then(() => {
    startAutoSimulation();
    void loadLatestSnapshotFromDb({
      silent: true,
      runIfMissing: shouldAutoSimulateForTour(defaultTour),
    });
  });
  loadLearningStatus(true);
  loadLifecycleStatus(true);
  loadPowerRankingReport(true);
  applyAutoRefreshSchedule();
  startLifecyclePoll();
  startLiveScorePoll();
}

init();
