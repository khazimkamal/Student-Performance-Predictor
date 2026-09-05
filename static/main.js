document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("prediction-form");
  if (!form) return;

  const fields = ["study_time", "absences", "failures", "previous_g1", "previous_g2"];
  let debounceTimer = null;

  // Sync sliders and number inputs
  fields.forEach(field => {
    const range = document.getElementById(`${field}_range`);
    const input = document.getElementById(`${field}_input`);

    if (range && input) {
      range.addEventListener("input", (e) => {
        input.value = e.target.value;
        schedulePreview();
      });

      input.addEventListener("input", (e) => {
        const val = parseFloat(e.target.value);
        if (!isNaN(val)) {
          range.value = val;
        }
        schedulePreview();
      });
    }
  });

  function getPayload() {
    const payload = {};
    for (const f of fields) {
      const el = document.getElementById(`${f}_input`);
      if (!el || el.value.trim() === "") return null;
      const num = parseFloat(el.value);
      if (isNaN(num)) return null;
      payload[f] = num;
    }
    return payload;
  }

  function schedulePreview() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(fetchPreview, 250);
  }

  async function fetchPreview() {
    const payload = getPayload();
    const liveStatus = document.getElementById("live-status");
    if (!payload) {
      if (liveStatus) liveStatus.textContent = "Waiting";
      return;
    }

    if (liveStatus) liveStatus.textContent = "Updating...";

    try {
      const resp = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      const data = await resp.json();
      if (data.ok && data.result) {
        updateUI(data.result);
        if (liveStatus) liveStatus.textContent = "Live";
      } else {
        if (liveStatus) liveStatus.textContent = "Invalid";
      }
    } catch (err) {
      console.error("Live preview error:", err);
      if (liveStatus) liveStatus.textContent = "Offline";
    }
  }

  function updateUI(res) {
    const scoreEl = document.getElementById("preview-score");
    const gradeEl = document.getElementById("preview-grade");
    const failEl = document.getElementById("preview-fail-prob");
    const riskEl = document.getElementById("preview-risk");
    const dialEl = document.getElementById("gauge-dial");
    const modelEl = document.getElementById("preview-model");
    const tipBox = document.getElementById("preview-tips-box");
    const tipText = document.getElementById("preview-top-tip");

    const score = res.predicted_score;
    if (scoreEl) scoreEl.textContent = score.toFixed(1);
    if (gradeEl) {
      gradeEl.textContent = res.grade;
      gradeEl.className = `grade-badge grade-${res.grade.toLowerCase().replace('+', '')}`;
    }
    if (failEl) failEl.textContent = `${res.fail_probability.toFixed(1)}%`;
    if (modelEl) modelEl.textContent = res.model_used;

    if (riskEl) {
      riskEl.textContent = `${res.risk_level} Risk`;
      riskEl.className = `risk-badge risk-${res.risk_level.toLowerCase()}`;
    }

    if (dialEl) {
      const pct = Math.max(0, Math.min(100, score));
      let color = "#4f46e5";
      if (res.risk_level === "High") color = "#ef4444";
      else if (res.risk_level === "Medium") color = "#f59e0b";
      else color = "#10b981";

      dialEl.style.background = `conic-gradient(${color} ${pct}%, #e2e8f0 ${pct}% 100%)`;
    }

    if (tipBox && tipText) {
      if (res.tips && res.tips.length > 0) {
        tipText.textContent = res.tips[0];
        tipBox.style.display = "block";
      } else {
        tipBox.style.display = "none";
      }
    }
  }

  // Initial trigger on load
  fetchPreview();
});
