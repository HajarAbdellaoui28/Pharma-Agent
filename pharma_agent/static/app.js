'use strict';
const $ = id => document.getElementById(id);
const questions = [
  'What are the main contraindications for warfarin?',
  'Compare the contraindications of warfarin and apixaban. What conditions are shared and which are unique to each drug?',
  'A patient currently on warfarin is prescribed sertraline. What is the risk of bleeding and what monitoring is recommended?',
  'A patient with an eGFR of 25 mL/min is being considered for metformin. Is metformin contraindicated at this level of renal function, and what is the threshold?'
];
function node(tag, text, cls) { const el = document.createElement(tag); if (text !== undefined) el.textContent = text; if (cls) el.className = cls; return el; }
function count() { $('char-count').textContent = `${$('question').value.length.toLocaleString()} / 2,000`; }
$('question').addEventListener('input', count);
document.querySelectorAll('[data-example]').forEach(button => button.addEventListener('click', () => { $('question').value = questions[Number(button.dataset.example)]; count(); $('question').focus(); }));
$('question').addEventListener('keydown', event => { if ((event.ctrlKey || event.metaKey) && event.key === 'Enter' && !$('ask-button').disabled) $('ask-form').requestSubmit(); });
async function checkHealth() {
  try { const res = await fetch('/health'); const health = await res.json(); $('health').className = `status-pill ${health.ready ? 'ready' : 'unavailable'}`; $('health-text').textContent = health.ready ? `${health.documents} source files · Corpus ready` : 'Research unavailable · Check configuration'; }
  catch { $('health').className = 'status-pill unavailable'; $('health-text').textContent = 'Connection unavailable'; }
}
function renderAnswer(data) {
  $('answer').replaceChildren(); $('citations').replaceChildren(); $('workflow').replaceChildren();
  const content = data.answer.replace(data.disclaimer, '').trim();
  for (const paragraph of content.split(/\n\s*\n/)) {
    const p = node('p');
    for (const part of paragraph.split(/(\[\d+\])/g)) {
      const match = part.match(/^\[(\d+)\]$/);
      if (match && data.citations.some(c => c.id === match[1])) { const link = node('a', part); link.href = `#citation-${match[1]}`; link.setAttribute('aria-label', `Read citation ${match[1]}`); p.append(link); }
      else p.append(document.createTextNode(part.replace(/^- /, '')));
    }
    $('answer').append(p);
  }
  $('disclaimer').textContent = data.disclaimer;
  $('answer-status').textContent = data.status === 'verified' ? 'Source review complete' : 'Evidence limited';
  $('metrics').replaceChildren(...[`${(data.metrics.latency_ms / 1000).toFixed(1)}s elapsed`, `${data.metrics.llm_calls} model calls`, `${data.metrics.tool_calls} tool calls`, `${data.metrics.research_rounds} research rounds`].map(value => node('span', value)));
  $('workflow').append(node('h3', 'RESEARCH PLAN'));
  const plan = node('ol'); data.plan.forEach(step => plan.append(node('li', step))); $('workflow').append(plan, node('h3', 'EXECUTION SUMMARY'));
  const events = node('ol'); data.workflow.forEach(event => events.append(node('li', `${event.stage.replaceAll('_', ' ')} — ${event.summary}`))); $('workflow').append(events);
  $('evidence-count').textContent = String(data.citations.length).padStart(2, '0');
  $('evidence-empty').hidden = data.citations.length > 0;
  data.citations.forEach(citation => {
    const card = node('article', undefined, 'citation-card'); card.id = `citation-${citation.id}`;
    const label = node('div', undefined, 'citation-label'); label.append(node('span', `EVIDENCE ${citation.id.padStart(2, '0')}`));
    if (citation.source_url) { try { const url = new URL(citation.source_url); if (url.protocol === 'https:' || url.protocol === 'http:') { const a = node('a', 'Source ↗'); a.href = url.href; a.target = '_blank'; a.rel = 'noopener noreferrer'; label.append(a); } } catch { /* Ignore malformed corpus URLs. */ } }
    card.append(label, node('h3', citation.source), node('div', `${citation.section} · Lines ${citation.line_start}–${citation.line_end}`, 'location'), node('blockquote', citation.quote));
    const detail = node('details'); detail.append(node('summary', 'Inspect full source passage')); let loaded = false;
    detail.addEventListener('toggle', async () => { if (!detail.open || loaded) return; loaded = true; const pre = node('pre', 'Loading passage…'); detail.append(pre); try { const res = await fetch(`/sources/${encodeURIComponent(citation.chunk_id)}`); if (!res.ok) throw new Error(); const chunk = await res.json(); pre.textContent = chunk.text; } catch { pre.textContent = 'Source passage is unavailable. Please retry the research request.'; } });
    card.append(detail); $('citations').append(card);
  });
  $('result').hidden = false;
}
$('ask-form').addEventListener('submit', async event => {
  event.preventDefault(); if ($('ask-button').disabled) return;
  const question = $('question').value.trim(); if (question.length < 3) { $('question').focus(); return; }
  $('error').hidden = true; $('result').hidden = true; $('citations').replaceChildren(); $('evidence-empty').hidden = false; $('evidence-count').textContent = '—'; $('loading').hidden = false; $('ask-button').disabled = true; $('question').readOnly = true;
  document.querySelectorAll('[data-example]').forEach(button => { button.disabled = true; });
  const started = Date.now(); $('elapsed').textContent = '0s'; const timer = setInterval(() => { $('elapsed').textContent = `${Math.floor((Date.now() - started) / 1000)}s`; }, 1000);
  try {
    const res = await fetch('/ask', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question }) });
    const data = await res.json();
    if (!res.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Please enter a valid question (3–2,000 characters).');
    renderAnswer(data);
  } catch (error) { $('error').textContent = error.message === 'Failed to fetch' ? 'The server could not be reached. Check the connection and try again.' : error.message; $('error').hidden = false; }
  finally { clearInterval(timer); $('loading').hidden = true; $('ask-button').disabled = false; $('question').readOnly = false; document.querySelectorAll('[data-example]').forEach(button => { button.disabled = false; }); checkHealth(); }
});
checkHealth();
