window.run = async () => {
  const a = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  console.log('VWG features: ' + [...a.features].filter(f=>/subgroup|f16|timestamp/.test(f)).join(', '));
  console.log('VWG subgroupMinSize=' + a.limits.subgroupMinSize + ' subgroupMaxSize=' + a.limits.subgroupMaxSize);
  console.log('VWG maxComputeInvocationsPerWorkgroup=' + a.limits.maxComputeInvocationsPerWorkgroup);
  console.log('VWG DONE');
};
window.addEventListener('DOMContentLoaded', () => window.run().catch(e=>console.log('VWG ERROR '+e.message)));
