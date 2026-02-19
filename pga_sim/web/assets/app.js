const state = {
  events: [],
  latestResult: null,
};

const ui = {
  tourSelect: document.getElementById("tourSelect"),
  eventSelect: document.getElementById("eventSelect"),
  simulationsInput: document.getElementById("simulationsInput"),
  cutSizeInput: document.getElementById("cutSizeInput"),
  meanReversionInput: document.getElementById("meanReversionInput"),
  meanReversionValue: document.getElementById("meanReversionValue"),
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

  rows.forEach((player, index) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="num">${index + 1}</td>
      <td>${player.player_name}</td>
      <td class="num">${formatPct(player.win_probability)}</td>
      <td class="num">${formatPct(player.top_3_probability)}</td>
      <td class="num">${formatPct(player.top_5_probability)}</td>
      <td class="num">${formatPct(player.top_10_probability)}</td>
      <td class="num">${Number(player.mean_finish).toFixed(2)}</td>
      <td class="num">${formatPct(player.baseline_win_probability)}</td>
      <td class="num">${formatMetric(player.baseline_season_metric)}</td>
      <td class="num">${formatMetric(player.current_season_metric)}</td>
      <td class="num">${formatMetric(player.form_delta_metric)}</td>
    `;
    ui.resultsBody.appendChild(tr);
  });
}

function renderResult(payload) {
  state.latestResult = payload;
  const topPlayer = payload.players[0];

  ui.eventLabel.textContent = payload.event_name || payload.event_id || "Unknown event";
  ui.simLabel.textContent = payload.simulations.toLocaleString();
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
    const cutSize = Number.parseInt(ui.cutSizeInput.value, 10) || 70;
    const meanReversion = Number.parseFloat(ui.meanReversionInput.value) || 0.1;
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
      simulations: simulations,
      cut_size: cutSize,
      mean_reversion: meanReversion,
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
    setStatus("Simulation complete.");
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
}

function init() {
  const currentYear = new Date().getFullYear();
  ui.currentSeasonInput.value = String(currentYear);
  ui.baselineSeasonInput.value = String(currentYear - 1);
  bindEvents();
  ui.meanReversionValue.textContent = Number.parseFloat(ui.meanReversionInput.value).toFixed(2);
  ui.seasonalWeightValue.textContent = Number.parseFloat(ui.seasonalWeightInput.value).toFixed(2);
  ui.currentSeasonWeightValue.textContent = Number.parseFloat(ui.currentSeasonWeightInput.value).toFixed(2);
  ui.formDeltaWeightValue.textContent = Number.parseFloat(ui.formDeltaWeightInput.value).toFixed(2);
  loadEvents();
}

init();
