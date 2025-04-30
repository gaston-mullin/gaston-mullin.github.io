---
layout: archive
title: "Interactive: Sample Means of Normal Distribution"
permalink: /sample-means/
author_profile: true
---

<h2>Distribution of Sample Mean (1000 Replications)</h2>

<label for="nSlider">Sample size (n): <span id="nLabel">5</span></label>
<input type="range" id="nSlider" min="5" max="50" step="5" value="5">
<div id="plot"></div>

<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<script>
let dataStore;

fetch('/assets/js/means_data.json')
  .then(response => response.json())
  .then(data => {
    dataStore = data;
    updatePlot(5);
  });

const slider = document.getElementById('nSlider');
const label = document.getElementById('nLabel');

slider.addEventListener('input', () => {
  const n = slider.value;
  label.textContent = n;
  updatePlot(n);
});

function updatePlot(n) {
  if (!dataStore) return;
  const means = dataStore[n];
  const trace = {
    x: means,
    type: 'histogram',
    nbinsx: 30,
    marker: { line: { width: 1, color: '#333' } }
  };
  const layout = {
    title: `Sample Size n = ${n}`,
    xaxis: { title: 'Sample Mean' },
    yaxis: { title: 'Frequency' }
  };
  Plotly.newPlot('plot', [trace], layout);
}
</script>
