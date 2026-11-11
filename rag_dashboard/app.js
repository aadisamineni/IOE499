'use strict';
const $ = id => document.getElementById(id);
const state = {data: null, day: '', symbol: '', provider: '', query: '', page: 1, articles: [], request: 0, generation: 0};
const assets = [
  {symbol:'NVDA', name:'NVIDIA', color:'#142b43'},
  {symbol:'SPY', name:'S&P 500 ETF', color:'#e9b840'},
  {symbol:'META', name:'Meta Platforms', color:'#77a6a2'},
  {symbol:'PFE', name:'Pfizer', color:'#b0a6c6'}
];
const number = n => Number(n).toLocaleString('en-US');
const escape = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const niceDate = d => new Date(d+'T12:00:00Z').toLocaleDateString('en-US', {month:'short',day:'numeric',year:'numeric',timeZone:'UTC'});
const safeURL = value => {try {const u = new URL(value); return ['http:','https:'].includes(u.protocol) ? u.href : null;} catch {return null;}};
async function api(path, options) {const response = await fetch(path,options); const data = await response.json(); if (!response.ok) throw new Error(data.error || 'The request could not be completed.'); return data;}
function error(message) {$('global-error').textContent=message; $('global-error').hidden=!message;}
function dayRow() {return state.data.days.find(d=>d.date_utc===state.day);}
function resetBrief() {state.generation++; $('answer').innerHTML='<div class="answer-empty"><span>“</span><strong>Awaiting pipeline connection.</strong><p>Briefs and model results will appear after the actual pipeline is connected.</p></div>';}

function activity() {
  const index=state.data.days.findIndex(d=>d.date_utc===state.day), count=Number($('chart-range').value);
  const rows=state.data.days.slice(Math.max(0,index-count+1),index+1);
  const width=540, height=200, left=34, right=8, top=12, bottom=27;
  const plotW=width-left-right, plotH=height-top-bottom;
  const max=Math.max(1,...rows.map(r=>Number(r.article_count))), step=plotW/rows.length;
  let svg=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Provider article counts for ${rows.length} days ending ${state.day}">`;
  for(let i=0;i<4;i++) {const val=max*i/3, y=top+plotH*(1-i/3); svg+=`<line x1="${left}" x2="${width-right}" y1="${y}" y2="${y}" class="chart-grid"/><text x="${left-8}" y="${y+3}" text-anchor="end" class="chart-axis">${Math.round(val)}</text>`;}
  rows.forEach((r,i)=>{
    const x=left+i*step+step*.18, w=Math.max(1,step*.62), a=Number(r.alpaca_article_count)/max*plotH, f=Number(r.finnhub_article_count)/max*plotH;
    svg+=`<g tabindex="0" role="button" class="activity-bar" data-date="${r.date_utc}" aria-label="${r.date_utc}: ${r.article_count} articles"><title>${r.date_utc} · ${r.article_count} records</title><rect x="${x}" y="${top}" width="${w}" height="${plotH}" fill="transparent"/><rect x="${x}" y="${top+plotH-f}" width="${w}" height="${f}" fill="#142b43" rx="1"/><rect x="${x}" y="${top+plotH-f-a}" width="${w}" height="${a}" fill="#e9b840" rx="1"/></g>`;
  });
  [0,Math.floor((rows.length-1)/2),rows.length-1].filter((v,i,a)=>a.indexOf(v)===i).forEach(i=>{svg+=`<text x="${left+(i+.5)*step}" y="${height-6}" text-anchor="${i===0?'start':i===rows.length-1?'end':'middle'}" class="chart-axis">${new Date(rows[i].date_utc+'T12:00:00Z').toLocaleDateString('en-US',{month:'short',day:'numeric',timeZone:'UTC'})}</text>`;});
  $('activity-chart').innerHTML=svg+'</svg>';
  $('chart-total').textContent=number(rows.reduce((n,r)=>n+Number(r.article_count),0))+' records';
  $('activity-chart').querySelectorAll('[data-date]').forEach(el=>{
    const choose=()=>{$('date').value=el.dataset.date;selectDay();};
    el.addEventListener('click',choose);el.addEventListener('keydown',event=>{if(['Enter',' '].includes(event.key)){event.preventDefault();choose();}});
  });
}
function overview() {
  const {summary,days,audit}=state.data, row=dayRow();
  $('metric-day').textContent=number(row.article_count);
  $('metric-day-sub').textContent='All assets · both providers · UTC day';
  $('coverage-note').hidden=!audit.missing.length;
  $('coverage-note').textContent=`Searchable evidence: ${number(audit.indexed_articles)} records · ${audit.missing.length} source export unavailable locally · See Pipeline & provenance for coverage details.`;
  $('previous-day').disabled=state.day===days[0].date_utc;
  $('next-day').disabled=state.day===days[days.length-1].date_utc;
  activity();
  const sourceInfo=[['Dataset source',audit.source],['Source commit',audit.commit||'Local working directory'],['News indexed for search',number(audit.indexed_articles)],['Index created (UTC)',audit.indexed_at_utc],['Pipeline results','Awaiting connection'],['Deduplication',summary.deduplication]];
  let notice='';
  if(audit.missing.length)notice=`<div class="audit-notice"><strong>Partial search coverage</strong><p>The news collection totals come from a saved source-data summary. Search covers ${number(audit.indexed_articles)} available records. ${audit.missing.length} source export(s) could not be indexed. These are input-data statistics, not pipeline results.</p>${audit.missing.map(m=>`<p><code>${escape(m.file)}</code> — ${escape(m.reason)}</p>`).join('')}</div>`;
  $('audit-details').innerHTML=notice+`<table class="audit-table"><tbody>${sourceInfo.map(([k,v])=>`<tr><td>${escape(k)}</td><td>${escape(v)}</td></tr>`).join('')}</tbody></table>`;
}
async function loadArticles() {
  const request=++state.request;
  const params=new URLSearchParams({date:state.day,symbol:state.symbol,provider:state.provider,q:state.query,page:state.page});
  $('article-list').innerHTML='<div class="empty">Finding source records…</div>';
  try {
    const result=await api('/api/articles?'+params);
    if(request!==state.request)return;
    state.articles=result.articles;
    $('evidence-count').textContent=number(result.total);
    const partial=state.data.audit.missing.length?' · Partial source coverage':'';
    $('feed-context').textContent=`${niceDate(state.day)} · ${state.symbol||'All assets'} · ${state.query?'Ranked by keyword relevance':'Newest first'}${partial}`;
    $('article-list').innerHTML=result.articles.length?result.articles.map((r,index)=>{
      const url=safeURL(r.url), title=escape(r.headline||'Untitled record');
      return `<article class="article"><div class="article-meta"><span class="article-provider">${escape(r.provider)}</span><span>${escape(r.publisher||'Publisher unavailable')}</span><time>${escape(r.published.slice(11,16))} UTC</time></div><h3>${url?`<a href="${escape(url)}" target="_blank" rel="noopener noreferrer">${title} ↗</a>`:title}</h3><p id="summary-${index}">${escape(r.summary.slice(0,210))}${r.summary.length>210?'…':''}</p><div class="article-bottom"><span class="article-id">${escape(r.key)} · ${escape(r.symbols)}</span>${r.summary.length>210?`<button data-expand="${index}" aria-expanded="false">Read summary +</button>`:''}</div></article>`;
    }).join(''):'<div class="empty">No indexed articles match this day and these filters. Try another date, asset or keyword. Missing provider exports are listed under Pipeline & provenance.</div>';
    $('article-list').querySelectorAll('[data-expand]').forEach(button=>button.addEventListener('click',()=>{const i=Number(button.dataset.expand), expanded=button.getAttribute('aria-expanded')==='true';button.setAttribute('aria-expanded',String(!expanded));button.textContent=expanded?'Read summary +':'Collapse −';$('summary-'+i).textContent=expanded?state.articles[i].summary.slice(0,210)+'…':state.articles[i].summary;}));
    $('page-info').textContent=result.total?`${number((state.page-1)*8+1)}–${number(Math.min(state.page*8,result.total))} of ${number(result.total)} records`:'0 matching records';
    $('page-prev').disabled=state.page<=1;$('page-next').disabled=state.page*8>=result.total;
  } catch(e) {if(request===state.request){error(e.message);$('article-list').innerHTML='<div class="empty">Source records could not be loaded.</div>';}}
}
function selectDay() {
  const chosen=$('date').value;
  if(!state.data.days.some(d=>d.date_utc===chosen)){error('Choose a date inside the saved research period.');$('date').value=state.day;return;}
  state.day=chosen;state.page=1;error('');resetBrief();overview();loadArticles();
}
function navigate() {
  const view=location.hash.slice(1)||'overview', current=['overview','pipeline'].includes(view)?view:'overview';
  document.querySelectorAll('.nav-link').forEach(a=>{a.classList.toggle('active',a.dataset.view===current);if(a.dataset.view===current)a.setAttribute('aria-current','page');else a.removeAttribute('aria-current');});
  $('overview-view').hidden=current!=='overview';$('evidence-section').hidden=current==='pipeline';$('pipeline-view').hidden=current!=='pipeline';
  const copy={overview:['The signal, in context.','Explore the news behind your portfolio. Trace every insight back to its source.','Workspace overview'],pipeline:['Every step, accounted for.','Inspect the data sources, retrieval method and limits of this research snapshot.','Pipeline & provenance']};
  $('page-title').textContent=copy[current][0];$('page-subtitle').textContent=copy[current][1];$('breadcrumb').textContent=copy[current][2];
}
async function generate() {
  const generation=++state.generation, day=state.day, symbol=state.symbol, query=state.query;
  $('generate').disabled=true;$('generate').textContent='Building brief with local Qwen…';
  $('answer').textContent='Retrieving matching evidence and waiting for the local model. This may take a few minutes.';
  try {
    const result=await api('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({date:day,symbol,provider:state.provider,query,question:$('question').value})});
    if(generation!==state.generation)return;
    $('answer').replaceChildren();
    const label=document.createElement('span');label.className='answer-label';label.textContent='RESEARCH BRIEF · '+day;$('answer').append(label);
    const text=document.createElement('div');text.textContent=result.answer;$('answer').append(text);
    const detail=document.createElement('p');detail.className='briefing-note';detail.textContent=`${result.evidence.length} supplied records · ${result.retrieval} · ${result.model}`;$('answer').append(detail);
    if(result.unknown_citations.length||!result.has_citations||result.truncated){const warning=document.createElement('p');warning.className='alert';warning.textContent=result.unknown_citations.length?'The answer contains citation IDs outside the supplied evidence. Verify these claims.':!result.has_citations?'No valid source IDs were cited. Treat this answer as unverified.':'The model reached its response limit; the brief may be incomplete.';$('answer').append(warning);}
    result.evidence.forEach(r=>{const item=document.createElement('a');item.className='citation';item.textContent=`[${r.key}] ${r.headline}`;const url=safeURL(r.url);if(url){item.href=url;item.target='_blank';item.rel='noopener noreferrer';}$('answer').append(item);});
  } catch(e) {if(generation===state.generation)$('answer').textContent=e.message;}
  finally {$('generate').disabled=false;$('generate').innerHTML='Generate evidence brief <span>↗</span>';}
}
function exportSnapshot() {
  const snapshot={date_utc:state.day,pipeline_status:'awaiting_connection',source:state.data.audit,
    filters:{symbol:state.symbol,provider:state.provider,query:state.query},visible_evidence:state.articles,
    note:'Input-news preview only. No portfolio allocations or pipeline results are available. Visible evidence is a paginated subset.'};
  const link=document.createElement('a'),url=URL.createObjectURL(new Blob([JSON.stringify(snapshot,null,2)],{type:'application/json'}));
  link.href=url;link.download=`ioe499-snapshot-${state.day}.json`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
async function init() {
  try {
    state.data=await api('/api/overview');
    if(!state.data.days.length)throw new Error('The daily snapshot has no rows.');
    // Prefer the latest date with source news, while retaining zero-news days in the selector.
    state.day=[...state.data.days].reverse().find(r=>Number(r.article_count)>0)?.date_utc||state.data.days.at(-1).date_utc;
    $('date').min=state.data.days[0].date_utc;$('date').max=state.data.days.at(-1).date_utc;$('date').value=state.day;
    overview();navigate();resetBrief();await loadArticles();$('loading').hidden=true;
    $('generate').disabled=true;$('generate').textContent='Awaiting pipeline connection';
    $('model-status').textContent='Pipeline not connected · No model results';
  } catch(e) {error(e.message);$('loading').hidden=true;}
}
$('date').addEventListener('change',selectDay);
for(const [id,delta] of [['previous-day',-1],['next-day',1]])$(id).addEventListener('click',()=>{const i=state.data.days.findIndex(r=>r.date_utc===state.day);if(state.data.days[i+delta]){$('date').value=state.data.days[i+delta].date_utc;selectDay();}});
$('assets').querySelectorAll('button').forEach(button=>button.addEventListener('click',()=>{state.symbol=button.dataset.symbol;state.page=1;$('assets').querySelectorAll('button').forEach(b=>{b.classList.toggle('selected',b===button);b.setAttribute('aria-pressed',String(b===button));});resetBrief();loadArticles();}));
$('provider').addEventListener('change',()=>{state.provider=$('provider').value;state.page=1;resetBrief();loadArticles();});
$('search-form').addEventListener('submit',event=>{event.preventDefault();state.query=$('query').value.trim();state.page=1;resetBrief();loadArticles();});
$('chart-range').addEventListener('change',activity);
$('page-prev').addEventListener('click',()=>{state.page--;loadArticles();});$('page-next').addEventListener('click',()=>{state.page++;loadArticles();});
$('export').addEventListener('click',exportSnapshot);window.addEventListener('hashchange',()=>{if(state.data)navigate();});
init();
