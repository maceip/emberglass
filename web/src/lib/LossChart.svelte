<script>
  let { points = [], color = '#ff8a5c', height = 64 } = $props();

  const path = $derived.by(() => {
    if (points.length < 2) return '';
    const ys = points.map((p) => p.loss);
    const min = Math.min(...ys), max = Math.max(...ys);
    const span = max - min || 1;
    const w = 100, h = height;
    return points
      .map((p, i) => {
        const x = (i / (points.length - 1)) * w;
        const y = h - ((p.loss - min) / span) * (h - 8) - 4;
        return `${i ? 'L' : 'M'}${x.toFixed(2)},${y.toFixed(2)}`;
      })
      .join(' ');
  });
  const last = $derived(points.length ? points[points.length - 1].loss : null);
</script>

<div class="chart">
  <svg viewBox={`0 0 100 ${height}`} preserveAspectRatio="none">
    <path d={path} fill="none" stroke={color} stroke-width="1.4" vector-effect="non-scaling-stroke" />
  </svg>
  {#if last != null}<span class="val" style:color>{last.toFixed(4)}</span>{/if}
</div>

<style>
  .chart { position: relative; width: 100%; height: 64px; }
  svg { width: 100%; height: 100%; display: block; }
  .val { position: absolute; right: 4px; top: 2px; font-family: var(--mono); font-size: 12px; font-weight: 600; }
</style>
