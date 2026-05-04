const models = {
  qwen: { accuracy: 83.94, macro: 81.79, weighted: 83.49, gpu: 6.01, gpuPct: 75 },
  phi: { accuracy: 82.51, macro: 80.32, weighted: 82.10, gpu: 7.39, gpuPct: 92 }
};
function updateModel(name){
  const m = models[name];
  document.querySelectorAll('.model-toggle button').forEach(b=>b.classList.toggle('active', b.dataset.model===name));
  document.getElementById('accuracy').textContent = `${m.accuracy.toFixed(2)}%`;
  document.getElementById('macro').textContent = `${m.macro.toFixed(2)}%`;
  document.getElementById('weighted').textContent = `${m.weighted.toFixed(2)}%`;
  document.getElementById('gpu').textContent = `${m.gpu.toFixed(2)} GB`;
  document.getElementById('macroBar').style.width = `${m.macro}%`;
  document.getElementById('weightedBar').style.width = `${m.weighted}%`;
  document.getElementById('gpuBar').style.width = `${m.gpuPct}%`;
  document.querySelector('.score-ring').style.background = `conic-gradient(var(--rose) 0 ${m.accuracy}%, #f0e1ed ${m.accuracy}% 100%)`;
}
document.querySelectorAll('.model-toggle button').forEach(btn=>btn.addEventListener('click',()=>updateModel(btn.dataset.model)));
updateModel('qwen');

document.getElementById('runDemo').addEventListener('click', () => {
  const lh = Number(document.getElementById('lh').value);
  const est = Number(document.getElementById('estrogen').value);
  const symptom = document.getElementById('symptom').value;
  let phase = 'Follicular';
  if (lh > 65 && est > 60) phase = 'Fertility';
  else if (symptom === 'cramps') phase = 'Menstrual';
  else if (['bloating','sore breasts','fatigue'].includes(symptom) && est < 55) phase = 'Luteal';
  const feedback = phase === 'Fertility'
    ? `Elevated LH and estrogen values may indicate a fertility-related pattern. ${symptom[0].toUpperCase()+symptom.slice(1)} is considered when generating personalized feedback.`
    : phase === 'Menstrual'
    ? `The symptom pattern is consistent with the menstrual phase. ${symptom[0].toUpperCase()+symptom.slice(1)} may be contributing to discomfort.`
    : phase === 'Luteal'
    ? `The reported pattern is consistent with the luteal phase. ${symptom[0].toUpperCase()+symptom.slice(1)} may be relevant to this phase.`
    : `The pattern suggests the follicular phase. The system creates concise feedback grounded in the reported features.`;
  document.getElementById('demoOutput').innerHTML = `<strong>Predicted phase: ${phase}</strong><p>${feedback}</p>`;
});
