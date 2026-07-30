const results = {
  pqc: {
    distilbert: {name:"DistilBERT-base", rounds:16, accuracy:84.32, macro:84.22, weighted:84.68, train:"0.1 h", latency:"1 ms/ex.", throughput:"774.9 ex./s", gpu:"1.0 GB", privacy:"0.26 GB", summary:"Best overall accuracy with the lowest training, inference, memory, and communication cost."},
    smollm: {name:"SmolLM3-3B", rounds:16, accuracy:84.32, macro:84.03, weighted:84.49, train:"17.8 h", latency:"596 ms/ex.", throughput:"1.7 ex./s", gpu:"6.7 GB", privacy:"4.84 GB", summary:"Matches DistilBERT accuracy, but requires substantially more time and communication."},
    qwen: {name:"Qwen2.5-3B", rounds:30, accuracy:82.84, macro:82.73, weighted:82.77, train:"19.9 h", latency:"540 ms/ex.", throughput:"1.9 ex./s", gpu:"7.0 GB", privacy:"4.79 GB", summary:"Strong classification utility, but high training and ciphertext-transfer cost."},
    phi: {name:"Phi-3.5-mini", rounds:30, accuracy:82.04, macro:81.89, weighted:82.01, train:"18.0 h", latency:"295 ms/ex.", throughput:"3.4 ex./s", gpu:"6.7 GB", privacy:"4.03 GB", summary:"Fastest decoder at inference, though still far slower than the encoder."},
    llama: {name:"Llama-3.1-8B", rounds:30, accuracy:71.45, macro:70.26, weighted:71.38, train:"10.8 h", latency:"533 ms/ex.", throughput:"1.9 ex./s", gpu:"16.8 GB", privacy:"6.71 GB", summary:"Largest communication and memory cost with the weakest PQC classification result."}
  },
  dp: {
    distilbert: {name:"DistilBERT-base", rounds:23, accuracy:84.32, macro:84.20, weighted:84.72, train:"0.2 h", latency:"1 ms/ex.", throughput:"764.5 ex./s", gpu:"1.0 GB", privacy:"ε 1380.1", summary:"Maintains its PQC accuracy; the main cost is seven additional training rounds."},
    smollm: {name:"SmolLM3-3B", rounds:19, accuracy:77.57, macro:77.71, weighted:78.85, train:"3.1 h", latency:"489 ms/ex.", throughput:"2.0 ex./s", gpu:"26.3 GB", privacy:"ε 1156.8", summary:"Accuracy falls and peak GPU memory rises sharply under the DP implementation."},
    qwen: {name:"Qwen2.5-3B", rounds:26, accuracy:75.87, macro:74.93, weighted:75.15, train:"16.7 h", latency:"506 ms/ex.", throughput:"2.0 ex./s", gpu:"8.2 GB", privacy:"ε 1545.1", summary:"Experiences a substantial utility drop under noisy client-level aggregation."},
    phi: {name:"Phi-3.5-mini", rounds:30, accuracy:84.05, macro:83.96, weighted:84.41, train:"11.9 h", latency:"311 ms/ex.", throughput:"3.2 ex./s", gpu:"7.0 GB", privacy:"ε 1761.8", summary:"The strongest decoder under DP, but with a large accumulated privacy budget."},
    llama: {name:"Llama-3.1-8B", rounds:22, accuracy:75.41, macro:76.07, weighted:76.99, train:"5.5 h", latency:"439 ms/ex.", throughput:"2.3 ex./s", gpu:"34.0 GB", privacy:"ε 1321.8", summary:"Improves over its PQC score, but remains costly in GPU memory and inference."}
  }
};

let activeSetting = "pqc";
const modelSelect = document.getElementById("modelSelect");

function renderResult() {
  const key = modelSelect.value;
  const item = results[activeSetting][key];

  document.getElementById("accuracy").textContent = `${item.accuracy.toFixed(2)}%`;
  document.getElementById("macroF1").textContent = `${item.macro.toFixed(2)}%`;
  document.getElementById("weightedF1").textContent = `${item.weighted.toFixed(2)}%`;
  document.getElementById("macroBar").style.width = `${item.macro}%`;
  document.getElementById("weightedBar").style.width = `${item.weighted}%`;
  document.getElementById("rounds").textContent = item.rounds;
  document.getElementById("trainTime").textContent = item.train;
  document.getElementById("latency").textContent = item.latency;
  document.getElementById("throughput").textContent = item.throughput;
  document.getElementById("gpu").textContent = item.gpu;
  document.getElementById("privacyMetric").textContent = item.privacy;
  document.getElementById("privacyMetricLabel").textContent =
    activeSetting === "pqc" ? "Ciphertext/round" : "Privacy budget";
  document.getElementById("resultTitle").textContent =
    `${item.name} · ${activeSetting === "pqc" ? "PQC" : "Differential Privacy"}`;
  document.getElementById("resultSummary").textContent = item.summary;
  document.getElementById("scoreRing").style.background =
    `conic-gradient(var(--rose) 0 ${item.accuracy}%, #eee1e8 ${item.accuracy}% 100%)`;
}

function renderTable() {
  const body = document.getElementById("resultsTable");
  body.innerHTML = "";
  ["pqc", "dp"].forEach(setting => {
    Object.values(results[setting]).forEach(item => {
      const row = document.createElement("tr");
      row.innerHTML = `
        <td>${item.name}</td>
        <td>${setting === "pqc" ? "PQC" : "DP"}</td>
        <td>${item.rounds}</td>
        <td>${item.accuracy.toFixed(2)}%</td>
        <td>${item.macro.toFixed(2)}%</td>
        <td>${item.weighted.toFixed(2)}%</td>
        <td>${item.train}</td>
        <td>${item.latency}</td>
        <td>${item.gpu}</td>`;
      body.appendChild(row);
    });
  });
}

document.querySelectorAll("#privacyToggle button").forEach(button => {
  button.addEventListener("click", () => {
    activeSetting = button.dataset.setting;
    document.querySelectorAll("#privacyToggle button").forEach(b =>
      b.classList.toggle("active", b === button)
    );
    renderResult();
  });
});

modelSelect.addEventListener("change", renderResult);

const menuButton = document.getElementById("menuButton");
const navLinks = document.getElementById("navLinks");
menuButton.addEventListener("click", () => {
  const open = navLinks.classList.toggle("open");
  menuButton.setAttribute("aria-expanded", String(open));
});
navLinks.querySelectorAll("a").forEach(link => link.addEventListener("click", () => {
  navLinks.classList.remove("open");
  menuButton.setAttribute("aria-expanded", "false");
}));

renderTable();
renderResult();
