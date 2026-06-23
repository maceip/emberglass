window.run = async () => {
  const out = [];
  out.push('navigator.ml = ' + (typeof navigator.ml));
  if (navigator.ml) {
    for (const dt of ['gpu','npu','cpu']) {
      try { const ctx = await navigator.ml.createContext({ deviceType: dt }); out.push(dt + ': context OK');
        if (ctx.opSupportLimits) { const l = ctx.opSupportLimits(); out.push('  ' + dt + ' dataTypes(input)=' + JSON.stringify(l.input?.dataTypes||'?')); }
      } catch(e) { out.push(dt + ': ' + e.message.slice(0,80)); }
    }
    out.push('MLGraphBuilder = ' + (typeof MLGraphBuilder));
  }
  for (const l of out) console.log('VWG ' + l);
  console.log('VWG DONE');
};
window.addEventListener('DOMContentLoaded', () => window.run().catch(e=>console.log('VWG ERROR '+e.message)));
