const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const colors={healthy:'#16A784',attention:'#EAA33B',critical:'#E45C5C',unknown:'#A0AEC0',blue:'#3478F6',orange:'#FF8A24'};
const number=(v,d=0)=>Number.isFinite(Number(v))?Number(v).toLocaleString('en-US',{maximumFractionDigits:d}):'—';
export function histogram(assets,{metric='rul',label='Remaining useful life distribution',bins=15,max=150}={}){
  const values=assets.filter(a=>Number.isFinite(a[metric]));
  if(!values.length)return '<div class="chart-empty">No measured engine results in this view.</div>';
  const mobile=innerWidth<650,w=mobile?350:680,h=230,l=42,r=16,t=15,b=37;
  const groups=Array.from({length:bins},()=>({healthy:0,attention:0,critical:0,unknown:0}));
  for(const a of values){const index=Math.min(bins-1,Math.max(0,Math.floor(a[metric]/max*bins)));groups[index][a.health in colors?a.health:'unknown']++;}
  const ymax=Math.max(1,...groups.map(g=>Object.values(g).reduce((x,y)=>x+y,0))),step=(w-l-r)/bins;
  const y=n=>t+(1-n/ymax)*(h-t-b);
  return `<svg class="data-chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(label)}"><title>${esc(label)}. Hover on a bar for count and range.</title>${Array.from({length:5},(_,i)=>{const n=ymax*i/4;return `<line x1="${l}" x2="${w-r}" y1="${y(n)}" y2="${y(n)}" stroke="#E8EFF7"/><text x="${l-9}" y="${y(n)+3}" text-anchor="end" class="axis">${number(n)}</text>`}).join('')}${groups.map((g,i)=>{let bottom=0;return Object.entries(g).map(([health,count])=>{const low=bottom;bottom+=count;return count?`<rect tabindex="0" x="${l+i*step+3}" y="${y(bottom)}" width="${Math.max(3,step-6)}" height="${y(low)-y(bottom)}" rx="2" fill="${colors[health]}" aria-label="${health}, ${count} engines, ${number(i*max/bins,1)} to ${number((i+1)*max/bins,1)}"><title>${health}: ${count} engines · ${number(i*max/bins,1)}–${number((i+1)*max/bins,1)}${metric==='rul'?' cycles':' percentile'}</title></rect>`:''}).join('')}).join('')}${Array.from({length:6},(_,i)=>`<text x="${l+(w-l-r)*i/5}" y="${h-18}" text-anchor="middle" class="axis">${number(max*i/5,metric==='rul'?0:1)}</text>`).join('')}<text x="${w/2}" y="${h-2}" text-anchor="middle" class="axis-label">${metric==='rul'?'Predicted remaining useful life (cycles)':'Empirical anomaly percentile'}</text></svg>`;
}
export function donut(summary){
  const groups=[['healthy','Healthy'],['attention','Attention'],['critical','Critical'],['unknown','Awaiting review']];
  const total=groups.reduce((s,[k])=>s+Number(summary[k]||0),0),radius=68,circ=2*Math.PI*radius;
  let offset=0;
  const slices=groups.map(([key,label])=>{const n=Number(summary[key]||0),length=total?n/total*circ:0;const svg=`<circle cx="100" cy="100" r="${radius}" fill="none" stroke="${colors[key]}" stroke-width="24" stroke-dasharray="${length} ${circ-length}" stroke-dashoffset="${-offset}" transform="rotate(-90 100 100)"><title>${label}: ${n} engines</title></circle>`;offset+=length;return svg}).join('');
  return `<div class="donut-layout"><svg viewBox="0 0 200 200" class="donut" role="img" aria-label="Fleet health distribution"><circle cx="100" cy="100" r="${radius}" fill="none" stroke="#EFF4FA" stroke-width="24"/>${slices}<text x="100" y="100" text-anchor="middle" class="donut-value">${total?number(Number(summary.healthy||0)/total*100):'—'}${total?'%':''}</text><text x="100" y="121" text-anchor="middle" class="donut-caption">Healthy</text></svg><div class="donut-legend">${groups.map(([k,l])=>`<div><span><i style="background:${colors[k]}"></i>${l}</span><strong>${number(summary[k]||0)}</strong></div>`).join('')}</div></div>`;
}
export function lineChart(points,{series=[{key:'rul',label:'Predicted RUL',color:colors.orange}],label='Engine history',xKey='cycle',xLabel='Observed cycle',yLabel='RUL (cycles)',band=false,zero=true}={}){
  const valid=points.filter(p=>Number.isFinite(Number(p[xKey]))&&series.some(s=>p[s.key]!=null&&Number.isFinite(Number(p[s.key]))));
  if(!valid.length)return '<div class="chart-empty">History is unavailable for this selection.</div>';
  const mobile=innerWidth<650,w=mobile?350:720,h=235,l=45,r=17,t=16,b=38;
  const lo=Math.min(...valid.map(p=>Number(p[xKey]))),hi=Math.max(...valid.map(p=>Number(p[xKey])));
  const vals=valid.flatMap(p=>[...series.map(s=>p[s.key]),...(band?[p.lower,p.upper]:[])].filter(v=>v!=null&&Number.isFinite(Number(v))).map(Number));
  let min=zero?Math.min(0,...vals):Math.min(...vals),max=Math.max(...vals);
  if(min===max){min-=1;max+=1;}else if(!zero){const pad=(max-min)*.1;min-=pad;max+=pad;}else max*=1.08;
  const x=n=>l+(n-lo)/Math.max(1,hi-lo)*(w-l-r),y=n=>t+(max-n)/(max-min)*(h-t-b);
  const bpoints=valid.filter(p=>p.lower!=null&&p.upper!=null);
  const polygon=bpoints.map(p=>`${x(p[xKey])},${y(p.upper)}`).concat([...bpoints].reverse().map(p=>`${x(p[xKey])},${y(p.lower)}`)).join(' ');
  return `<div class="chart-legend">${series.map(s=>`<span><i style="background:${s.color}"></i>${esc(s.label)}</span>`).join('')}${band&&bpoints.length?'<span><i style="background:#FBE2CA"></i>Capped prediction interval</span>':''}</div><svg class="data-chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(label)}"><title>${esc(label)}</title>${Array.from({length:5},(_,i)=>{const v=min+(max-min)*i/4;return `<line x1="${l}" x2="${w-r}" y1="${y(v)}" y2="${y(v)}" stroke="#E8EFF7"/><text x="${l-9}" y="${y(v)+3}" text-anchor="end" class="axis">${number(v,Math.abs(max)<5?2:0)}</text>`}).join('')}${band&&polygon?`<polygon points="${polygon}" fill="#FFF0E1"/>`:''}${series.map(s=>{const filtered=valid.filter(p=>p[s.key]!=null&&Number.isFinite(Number(p[s.key])));return `<path d="${filtered.map((p,i)=>`${i?'L':'M'}${x(p[xKey]).toFixed(2)},${y(p[s.key]).toFixed(2)}`).join(' ')}" fill="none" stroke="${s.color}" stroke-width="2.1" stroke-linecap="round"/>${filtered.filter((p,i)=>i%Math.max(1,Math.floor(filtered.length/20))===0||i===filtered.length-1).map(p=>`<circle tabindex="0" cx="${x(p[xKey])}" cy="${y(p[s.key])}" r="3" fill="${s.color}" opacity=".85"><title>${esc(s.label)} · ${esc(xLabel)} ${number(p[xKey])}: ${number(p[s.key],2)}</title></circle>`).join('')}`}).join('')}${Array.from({length:5},(_,i)=>{const v=lo+(hi-lo)*i/4;return `<text x="${x(v)}" y="${h-18}" text-anchor="middle" class="axis">${number(v)}</text>`}).join('')}<text x="${w/2}" y="${h-1}" text-anchor="middle" class="axis-label">${esc(xLabel)} · ${esc(yLabel)}</text></svg>`;
}
export function sparkline(points,key='value',color=colors.blue){
  const valid=points.filter(p=>p[key]!=null&&Number.isFinite(Number(p[key])));if(valid.length<2)return '<span class="muted">No trend</span>';
  const values=valid.map(p=>Number(p[key])),lo=Math.min(...values),hi=Math.max(...values),span=hi-lo||1;
  return `<svg class="sparkline" viewBox="0 0 120 30" role="img" aria-label="Recent sensor trend"><path d="${values.map((v,i)=>`${i?'L':'M'}${3+i/(values.length-1)*114},${26-(v-lo)/span*22}`).join(' ')}" fill="none" stroke="${color}" stroke-width="1.8" stroke-linecap="round"/></svg>`;
}
export {colors};
