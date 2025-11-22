const optionGroupsEl = document.getElementById("option-groups");
const toggleAdvancedEl = document.getElementById("toggle-advanced");
const fileInputEl = document.getElementById("audio-file");
const runBtn = document.getElementById("run-btn");
const resetBtn = document.getElementById("reset-btn");
const statusLog = document.getElementById("status-log");
const resultPanel = document.getElementById("result-panel");
const manifestPre = document.getElementById("manifest-preview");
const downloadLink = document.getElementById("download-link");
const jobLabel = document.getElementById("result-job");
const searchInputEl = document.getElementById("option-search");
const optionEmptyEl = document.getElementById("option-empty");
const cliPreviewEl = document.getElementById("cli-preview");
const configPreviewEl = document.getElementById("config-preview");
const copyCliBtn = document.getElementById("copy-cli");

const state = {
  metadata: null,
  optionElements: new Map(),
  optionWrappers: new Map(),
  groups: new Map(),
  searchQuery: "",
};

async function bootstrap() {
  try {
    await loadOptions();
    registerEvents();
    setStatus("Configuration loaded. Select audio to begin.");
  } catch (error) {
    console.error(error);
    setStatus(`Failed to load option metadata: ${error}`, true);
    runBtn.disabled = true;
  }
}

async function loadOptions() {
  const response = await fetch("/app/options");
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  const metadata = await response.json();
  state.metadata = metadata;
  renderOptions(metadata);
}

function renderOptions(metadata) {
  if (!metadata) {
    return;
  }
  optionGroupsEl.innerHTML = "";
  state.optionElements.clear();
  state.optionWrappers = new Map();
  state.groups = new Map();
  const groups = metadata.groups || [];
  const options = metadata.options || [];
  groups.forEach((groupName) => {
    const groupOptions = options.filter((opt) => opt.group === groupName);
    if (!groupOptions.length) {
      return;
    }
    const details = document.createElement("details");
    details.className = "option-group";
    details.open = true;
    const summary = document.createElement("summary");
    summary.textContent = groupName;
    details.appendChild(summary);
    const grid = document.createElement("div");
    grid.className = "option-grid";
    const record = { element: details, fields: [] };
    state.groups.set(groupName, record);
    groupOptions.forEach((opt) => {
      const field = buildField(opt);
      grid.appendChild(field);
      record.fields.push(field);
    });
    details.appendChild(grid);
    optionGroupsEl.appendChild(details);
  });
  filterOptions(state.searchQuery || "");
  updatePreview();
}

function buildField(option) {
  const wrapper = document.createElement("div");
  wrapper.className = "option-field";
  if (option.advanced) {
    wrapper.classList.add("advanced");
  }
  wrapper.dataset.optionKey = option.key;
  const searchText = `${option.label} ${option.description || ""} ${option.group || ""}`.toLowerCase();
  wrapper.dataset.searchText = searchText;
  const label = document.createElement("label");
  const inputId = `opt-${option.key}`;
  label.setAttribute("for", inputId);
  label.textContent = option.label;
  const helper = document.createElement("small");
  helper.textContent = option.description || "";

  const input = createInput(option, inputId);
  wrapper.appendChild(label);
  wrapper.appendChild(input);
  wrapper.appendChild(helper);
  state.optionElements.set(option.key, input);
  state.optionWrappers.set(option.key, wrapper);
  return wrapper;
}

function createInput(option, inputId) {
  const type = option.type;
  let input;
  if (type === "boolean") {
    input = document.createElement("input");
    input.type = "checkbox";
    input.checked = Boolean(option.default);
  } else if (type === "select") {
    input = document.createElement("select");
    (option.choices || []).forEach((choice) => {
      const opt = document.createElement("option");
      opt.value = choice.value ?? "";
      opt.textContent = choice.label ?? choice.value;
      if ((choice.value ?? "") === option.default) {
        opt.selected = true;
      }
      input.appendChild(opt);
    });
  } else if (type === "textarea") {
    input = document.createElement("textarea");
    input.value = option.default ?? "";
  } else {
    input = document.createElement("input");
    input.type = type === "integer" || type === "float" ? "number" : "text";
    if (option.step) {
      input.step = option.step;
    }
    if (option.default !== undefined && option.default !== null) {
      input.value = option.default;
    }
  }
  input.id = inputId;
  input.dataset.optionKey = option.key;
  input.dataset.fieldType = option.type;
  if (option.placeholder && "placeholder" in input) {
    input.placeholder = option.placeholder;
  }
  return input;
}

function registerEvents() {
  toggleAdvancedEl.addEventListener("change", () => {
    document.body.classList.toggle("show-advanced", toggleAdvancedEl.checked);
    filterOptions(state.searchQuery || "");
  });

  fileInputEl.addEventListener("change", () => {
    const hasFile = fileInputEl.files && fileInputEl.files.length > 0;
    runBtn.disabled = !hasFile;
    resetBtn.disabled = !hasFile;
  });

  if (searchInputEl) {
    searchInputEl.addEventListener("input", () => {
      state.searchQuery = searchInputEl.value;
      filterOptions(state.searchQuery);
    });
  }

  optionGroupsEl.addEventListener("input", () => {
    updatePreview();
  });
  optionGroupsEl.addEventListener("change", () => {
    updatePreview();
  });

  runBtn.addEventListener("click", async () => {
    if (!fileInputEl.files || !fileInputEl.files.length) {
      setStatus("Please select an audio file first.", true);
      return;
    }
    await runPipeline();
  });

  resetBtn.addEventListener("click", () => {
    if (!state.metadata) {
      return;
    }
    fileInputEl.value = "";
    runBtn.disabled = true;
    resetBtn.disabled = true;
    toggleAdvancedEl.checked = false;
    document.body.classList.remove("show-advanced");
    if (searchInputEl) {
      searchInputEl.value = "";
      state.searchQuery = "";
    }
    renderOptions(state.metadata);
    resultPanel.classList.add("hidden");
    setStatus("Configuration reset.");
  });

  if (copyCliBtn) {
    const defaultLabel = copyCliBtn.textContent;
    copyCliBtn.addEventListener("click", async () => {
      const command = (cliPreviewEl?.value || "").trim();
      if (!command) {
        return;
      }
      try {
        await navigator.clipboard.writeText(command);
        copyCliBtn.textContent = "Copied!";
        setTimeout(() => {
          copyCliBtn.textContent = defaultLabel;
        }, 1500);
      } catch (error) {
        console.error(error);
        setStatus(`Failed to copy CLI command: ${error}`, true);
      }
    });
  }
}

function collectOptions() {
  const payload = {};
  state.optionElements.forEach((input, key) => {
    const fieldType = input.dataset.fieldType;
    let value;
    if (fieldType === "boolean") {
      value = input.checked;
    } else if (fieldType === "integer") {
      value = input.value !== "" ? Number.parseInt(input.value, 10) : null;
    } else if (fieldType === "float") {
      value = input.value !== "" ? Number.parseFloat(input.value) : null;
    } else if (fieldType === "select") {
      value = input.value;
    } else {
      value = input.value.trim();
    }

    if (fieldType === "textarea" && input.value.trim() === "") {
      return;
    }
    if ((fieldType === "text" || fieldType === "select") && value === "") {
      return;
    }
    if ((fieldType === "integer" || fieldType === "float") && value === null) {
      return;
    }
    payload[key] = value;
  });
  return payload;
}

function updatePreview() {
  if (!state.metadata) {
    return;
  }
  const payload = collectOptions();
  if (configPreviewEl) {
    const preview = Object.keys(payload).length ? JSON.stringify(payload, null, 2) : "{}";
    configPreviewEl.textContent = preview;
  }
  if (cliPreviewEl) {
    cliPreviewEl.value = buildCliCommand(payload);
  }
}

function buildCliCommand(payload) {
  const commandParts = ["diaremot run <audio-path>"];
  const options = state.metadata?.options || [];
  options.forEach((meta) => {
    if (!meta.cliFlag) {
      return;
    }
    if (!(meta.key in payload)) {
      return;
    }
    const value = payload[meta.key];
    if (meta.type === "boolean") {
      const boolVal = Boolean(value);
      if (meta.cliInvert) {
        if (!boolVal) {
          commandParts.push(formatFlag(meta.cliFlag));
        }
      } else if (boolVal) {
        commandParts.push(formatFlag(meta.cliFlag));
      }
      return;
    }
    if (!hasMeaningfulValue(value)) {
      return;
    }
    if (meta.default !== undefined && meta.default !== null && valuesEqual(value, meta.default)) {
      return;
    }
    commandParts.push(`${formatFlag(meta.cliFlag)} ${formatCliValue(value)}`);
  });
  return commandParts.join(" \\\n  ");
}

function formatFlag(flag) {
  return flag.startsWith("--") ? flag : `--${flag}`;
}

function formatCliValue(value) {
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  const text = String(value ?? "");
  if (!text.length) {
    return "''";
  }
  if (/^[A-Za-z0-9._:@/\\-]+$/.test(text)) {
    return text;
  }
  return `'${text.replace(/'/g, "'\\''")}'`;
}

function hasMeaningfulValue(value) {
  return value !== null && value !== undefined && value !== "";
}

function valuesEqual(a, b) {
  if (typeof a === "number" && typeof b === "number") {
    return Object.is(a, b);
  }
  return a === b;
}

function filterOptions(rawQuery) {
  const query = (rawQuery || "").trim().toLowerCase();
  let visibleCount = 0;
  state.groups.forEach((record) => {
    let groupVisible = false;
    record.fields.forEach((field) => {
      const matches = !query || (field.dataset.searchText || "").includes(query);
      field.classList.toggle("search-hidden", !matches);
      const advancedHidden = field.classList.contains("advanced") && !toggleAdvancedEl.checked;
      if (matches && !advancedHidden) {
        groupVisible = true;
        visibleCount += 1;
      }
    });
    record.element.classList.toggle("search-hidden", !groupVisible);
  });
  if (optionEmptyEl) {
    optionEmptyEl.classList.toggle("hidden", visibleCount > 0);
  }
}

async function runPipeline() {
  const payload = collectOptions();
  const file = fileInputEl.files[0];
  const formData = new FormData();
  formData.append("audio", file);
  formData.append("options", JSON.stringify(payload));
  runBtn.disabled = true;
  setStatus("Uploading audio & preparing run...");
  try {
    const response = await fetch("/app/run", {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(error.detail || response.statusText);
    }
    const data = await response.json();
    handleSuccess(data);
    setStatus("Pipeline completed successfully.");
  } catch (error) {
    console.error(error);
    setStatus(`Pipeline failed: ${error}`, true);
  } finally {
    runBtn.disabled = false;
  }
}

function handleSuccess(data) {
  jobLabel.textContent = data.job_id;
  downloadLink.href = data.download_url;
  downloadLink.setAttribute("download", `diaremot_${data.job_id}.zip`);
  manifestPre.textContent = JSON.stringify(data.result, null, 2);
  resultPanel.classList.remove("hidden");
  resetBtn.disabled = false;
}

function setStatus(message, isError = false) {
  statusLog.innerHTML = "";
  const paragraph = document.createElement("p");
  paragraph.textContent = message;
  if (isError) {
    paragraph.style.color = "#f87171";
  }
  statusLog.appendChild(paragraph);
}

bootstrap();
