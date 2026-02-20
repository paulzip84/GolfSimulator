const state = {
  events: [],
  latestResult: null,
  expandedPlayerKey: null,
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
};

const ui = {
  tourSelect: document.getElementById("tourSelect"),
  eventSelect: document.getElementById("eventSelect"),
  simulationsInput: document.getElementById("simulationsInput"),
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
  applyRecommendationButton: document.getElementById("applyRecommendationButton"),
  status: document.getElementById("status"),
  formStatus: document.getElementById("formStatus"),
  error: document.getElementById("error"),
  resultsSection: document.getElementById("resultsSection"),
  eventLabel: document.getElementById("eventLabel"),
  simLabel: document.getElementById("simLabel"),
  generatedLabel: document.getElementById("generatedLabel"),
  winnerLabel: document.getElementById("winnerLabel"),
  seasonWindowLabel: document.getElementById("seasonWindowLabel"),
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

function setBusy(isBusy) {
  ui.simulateButton.disabled = isBusy;
  ui.loadEventsButton.disabled = isBusy;
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
  ui.generatedLabel.textContent = formatDate(payload.generated_at);
  ui.winnerLabel.textContent = topPlayer
    ? `${topPlayer.player_name} (${formatPct(topPlayer.win_probability)})`
    : "-";
  if (payload.baseline_season && payload.current_season) {
    ui.seasonWindowLabel.textContent = `${payload.baseline_season} vs ${payload.current_season}`;
  } else {
    ui.seasonWindowLabel.textContent = "-";
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

async function runSimulation() {
  setError();
  setFormStatus("Loading seasonal form data...", true);
  setBusy(true);
  setStatus("Running simulation...", true);

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
    setStatus(statusBits.length > 0 ? `Simulation complete. ${statusBits.join(" | ")}` : "Simulation complete.");
  } catch (error) {
    setStatus("Simulation failed.");
    setError(error.message || "Unexpected error while running the simulation.");
  } finally {
    setBusy(false);
  }
}

function bindEvents() {
  ui.loadEventsButton.addEventListener("click", loadEvents);
  ui.simulateButton.addEventListener("click", runSimulation);
  ui.tourSelect.addEventListener("change", loadEvents);
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
}

init();
