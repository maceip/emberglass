// Operation pipelines for the showcase. These drive the UI + 3D loading animations
// with realistic staging and metrics. Compute here is SIMULATED (clearly labelled in
// the UI) so the page deploys to static GitHub Pages without multi-GB model weights;
// the real numbers come from the WebGPU engine in ../../../src. Each runner streams
// events: { phase, progress (0..1), label, log, metric }.

const sleep = (ms, signal) =>
  new Promise((res, rej) => {
    const t = setTimeout(res, ms);
    signal?.addEventListener('abort', () => { clearTimeout(t); rej(new DOMException('aborted', 'AbortError')); }, { once: true });
  });

const rnd = (a, b) => a + Math.random() * (b - a);

// A plausible decaying loss with noise.
function lossAt(step, total, start = 4.2, floor = 0.5) {
  const p = step / total;
  const decay = floor + (start - floor) * Math.exp(-3.2 * p);
  return Math.max(0.02, decay + rnd(-0.12, 0.12) * (1 - p));
}

export const OPS = {
  demo: { label: 'Demo', color: '#6ad7ff', blurb: 'Spin up the runtime and watch the model come alive.' },
  train: { label: 'Train', color: '#ff8a5c', blurb: 'Full backward pass + AdamW over the frozen int4 base.' },
  tune: { label: 'Tune', color: '#c08bff', blurb: 'Sweep LoRA rank / lr / scale and lock in a schedule.' },
  inference: { label: 'Inference', color: '#5cffb0', blurb: 'Hot-swap an adapter and stream tokens.' },
};

export async function runDemo({ signal, emit }) {
  const phases = [
    ['booting WebGPU device', 700],
    ['streaming + quantizing weights (int4)', 1500],
    ['warming attention kernels', 900],
    ['ready — adapters hot-swap live', 500],
  ];
  let done = 0;
  for (const [label, ms] of phases) {
    emit({ phase: label, label, log: `▶ ${label}` });
    const steps = Math.max(4, Math.round(ms / 90));
    for (let i = 0; i < steps; i++) {
      await sleep(ms / steps, signal);
      done += 1;
      emit({ progress: done / (phases.reduce((s, p) => s + Math.max(4, Math.round(p[1] / 90)), 0)) });
    }
  }
  emit({ progress: 1, log: '✓ runtime live', label: 'live' });
}

export async function runTrain({ signal, emit, params }) {
  const steps = params?.steps ?? 50;
  emit({ phase: 'preparing batches', label: 'preparing batches', log: `▶ ${steps} steps · rank ${params?.rank ?? 16} · lr ${params?.lr ?? '2e-4'}` });
  await sleep(500, signal);
  for (let s = 1; s <= steps; s++) {
    for (const sub of ['forward', 'backward', 'optimizer']) {
      emit({ phase: sub });
      await sleep(rnd(35, 70), signal);
    }
    const loss = lossAt(s, steps);
    emit({ progress: s / steps, metric: { step: s, loss }, log: s % 5 === 0 || s === 1 ? `step ${s}/${steps}  loss=${loss.toFixed(4)}` : undefined });
  }
  emit({ progress: 1, label: 'converged', log: '✓ training complete' });
}

export async function runTune({ signal, emit }) {
  const grid = [
    { rank: 8, lr: '1e-4' },
    { rank: 16, lr: '2e-4' },
    { rank: 16, lr: '5e-4' },
    { rank: 32, lr: '2e-4' },
  ];
  emit({ phase: 'loading base adapter', label: 'loading base adapter', log: '▶ sweeping LoRA configs' });
  await sleep(500, signal);
  let best = { loss: Infinity };
  for (let i = 0; i < grid.length; i++) {
    const g = grid[i];
    emit({ phase: `trial ${i + 1}: r${g.rank} lr${g.lr}`, label: `trial ${i + 1}/${grid.length}` });
    let loss = 0;
    const inner = 14;
    for (let s = 1; s <= inner; s++) {
      await sleep(rnd(40, 80), signal);
      loss = lossAt(s, inner, 3.8, 0.6 + (g.rank === 16 && g.lr === '2e-4' ? 0 : 0.25));
      emit({ progress: (i + s / inner) / grid.length, metric: { step: i * inner + s, loss } });
    }
    emit({ log: `trial ${i + 1}  r=${g.rank} lr=${g.lr}  → loss=${loss.toFixed(4)}` });
    if (loss < best.loss) best = { ...g, loss };
  }
  emit({ progress: 1, label: 'locked', log: `✓ best: r=${best.rank} lr=${best.lr} (loss ${best.loss.toFixed(4)})` });
}

const SAMPLE = `Sure — here's a tight plan. First we hot-swap the adapter onto the frozen int4 base, then stream tokens straight from the WebGPU runtime. The decode loop stays resident on the GPU, so latency stays low even for long reasoning traces.`;

export async function runInference({ signal, emit, params }) {
  emit({ phase: 'tokenizing prompt', label: 'tokenizing', log: `▶ prompt: "${(params?.prompt || 'explain the plan').slice(0, 40)}"` });
  await sleep(450, signal);
  emit({ phase: 'prefill', label: 'prefill' });
  await sleep(700, signal);
  const words = SAMPLE.split(' ');
  let text = '';
  for (let i = 0; i < words.length; i++) {
    await sleep(rnd(45, 110), signal);
    text += (i ? ' ' : '') + words[i];
    emit({ phase: 'decoding', progress: (i + 1) / words.length, token: words[i], text });
  }
  emit({ progress: 1, label: 'done', log: `✓ ${words.length} tokens` });
}

export function runner(kind) {
  return { demo: runDemo, train: runTrain, tune: runTune, inference: runInference }[kind];
}
