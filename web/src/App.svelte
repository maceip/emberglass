<script>
  import { Canvas } from '@threlte/core';
  import SceneRoot from './lib/scenes/SceneRoot.svelte';
  import LossChart from './lib/LossChart.svelte';
  import { OPS, runner } from './lib/ops.js';
  import { probeGPU, fmtBytes } from './lib/gpu.js';

  let mode = $state('demo');
  let running = $state(false);
  let progress = $state(0);
  let phase = $state('idle');
  let logs = $state([]);
  let metrics = $state([]);
  let outText = $state('');
  let pointer = $state({ x: 0, y: 0 });
  let gpu = $state(null);

  let steps = $state(50);
  let rank = $state(16);
  let lr = $state('2e-4');
  let prompt = $state('Explain the plan in one paragraph.');

  let controller = null;
  const color = $derived(OPS[mode].color);

  $effect(() => {
    probeGPU().then((g) => (gpu = g));
  });

  function emit(ev) {
    if (ev.progress != null) progress = ev.progress;
    if (ev.phase) phase = ev.phase;
    if (ev.label) phase = ev.label;
    if (ev.log) logs = [...logs, ev.log].slice(-140);
    if (ev.metric) metrics = [...metrics, ev.metric].slice(-400);
    if (ev.text != null) outText = ev.text;
  }

  async function run() {
    if (running) return stop();
    running = true;
    progress = 0;
    metrics = [];
    outText = '';
    logs = [`— ${OPS[mode].label} started —`];
    controller = new AbortController();
    const params = { steps, rank, lr, prompt };
    try {
      await runner(mode)({ signal: controller.signal, emit, params });
    } catch (e) {
      if (e?.name !== 'AbortError') logs = [...logs, `✗ ${e?.message || e}`];
      else logs = [...logs, '— stopped —'];
    } finally {
      running = false;
    }
  }

  function stop() {
    controller?.abort();
    running = false;
  }

  function pick(m) {
    if (running) stop();
    mode = m;
    progress = 0;
    phase = 'idle';
    metrics = [];
    outText = '';
  }

  function onMove(e) {
    pointer = { x: (e.clientX / window.innerWidth) * 2 - 1, y: (e.clientY / window.innerHeight) * 2 - 1 };
  }
</script>

<svelte:window on:pointermove={onMove} />

<div class="stage" style:--accent={color}>
  <div class="bg">
    <Canvas>
      <SceneRoot {mode} {progress} {color} {pointer} />
    </Canvas>
  </div>

  <header class="top">
    <div class="brand">
      <span class="dot" style:background={color}></span>
      <b>Emberglass</b>
      <span class="sub">WebGPU&nbsp;LoRA</span>
    </div>
    <div class="gpu glass">
      {#if gpu == null}
        probing GPU…
      {:else if gpu.ok}
        <span class="ok">●</span> WebGPU · {gpu.adapter.vendor || 'gpu'}
        {gpu.adapter.architecture}
        · {fmtBytes(gpu.adapter.maxBufferSize)}
        {#if gpu.adapter.f16}· f16{/if}
      {:else}
        <span class="bad">●</span> {gpu.reason}
      {/if}
    </div>
  </header>

  <nav class="nav glass">
    {#each Object.entries(OPS) as [key, op]}
      <button class="tab" class:active={mode === key} style:--c={op.color} onclick={() => pick(key)}>
        {op.label}
      </button>
    {/each}
  </nav>

  <section class="dock glass">
    <div class="head">
      <h2 style:color>{OPS[mode].label}</h2>
      <p>{OPS[mode].blurb}</p>
    </div>

    {#if mode === 'train'}
      <div class="params">
        <label>steps<input type="range" min="10" max="200" bind:value={steps} /><b>{steps}</b></label>
        <label>rank<select bind:value={rank}><option>8</option><option>16</option><option>32</option><option>64</option></select></label>
        <label>lr<select bind:value={lr}><option>1e-4</option><option>2e-4</option><option>5e-4</option></select></label>
      </div>
    {:else if mode === 'inference'}
      <div class="params">
        <label class="full">prompt<input type="text" bind:value={prompt} /></label>
      </div>
    {/if}

    <button class="run" class:running onclick={run} style:--c={color}>
      {running ? 'Stop' : `Run ${OPS[mode].label}`}
    </button>

    <div class="progress">
      <div class="bar" style:width={`${progress * 100}%`} style:background={color}></div>
    </div>
    <div class="phase"><span class="spinner" class:on={running} style:border-top-color={color}></span>{phase} · {(progress * 100).toFixed(0)}%</div>

    {#if (mode === 'train' || mode === 'tune') && metrics.length}
      <LossChart points={metrics} {color} />
    {/if}

    {#if mode === 'inference' && outText}
      <div class="stream">{outText}<span class="caret" class:on={running}>▍</span></div>
    {/if}

    <div class="console">
      {#each logs as line}<div class="ln">{line}</div>{/each}
    </div>

    <p class="note">
      Visualization is live in your browser. Heavy compute is <b>simulated</b> for this static page — the
      real backward pass + AdamW runs on the WebGPU engine in the repo (BYO model).
    </p>
  </section>
</div>

<style>
  .stage { position: fixed; inset: 0; }
  .bg { position: absolute; inset: 0; z-index: 0; }

  header.top {
    position: absolute; top: 0; left: 0; right: 0; z-index: 2;
    display: flex; align-items: center; justify-content: space-between;
    padding: 16px 20px; pointer-events: none;
  }
  .brand { display: flex; align-items: baseline; gap: 8px; }
  .brand b { font-size: 18px; letter-spacing: 0.2px; }
  .brand .sub { color: var(--muted); font-size: 12px; }
  .brand .dot { width: 9px; height: 9px; border-radius: 50%; align-self: center; box-shadow: 0 0 12px currentColor; }
  .gpu { pointer-events: auto; padding: 8px 12px; font-size: 12px; color: var(--muted); font-family: var(--mono); }
  .gpu .ok { color: #5cffb0; } .gpu .bad { color: #ff6b6b; }

  nav.nav {
    position: absolute; z-index: 2; top: 64px; left: 50%; transform: translateX(-50%);
    display: flex; gap: 4px; padding: 5px;
  }
  .tab {
    background: transparent; border: 0; color: var(--muted); cursor: pointer;
    padding: 9px 18px; border-radius: 11px; font-size: 14px; font-weight: 600; transition: all 0.2s;
  }
  .tab:hover { color: var(--text); }
  .tab.active { color: #0a0a0f; background: var(--c); box-shadow: 0 6px 20px -6px var(--c); }

  section.dock {
    position: absolute; z-index: 2; right: 20px; bottom: 20px; top: 120px;
    width: 360px; max-width: calc(100vw - 40px);
    padding: 18px; display: flex; flex-direction: column; gap: 12px; overflow: hidden;
  }
  .head h2 { margin: 0 0 2px; font-size: 22px; }
  .head p { margin: 0; color: var(--muted); font-size: 13px; line-height: 1.4; }

  .params { display: flex; flex-wrap: wrap; gap: 10px; }
  .params label { display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--muted); }
  .params label.full { flex: 1 1 100%; }
  .params input[type='range'] { accent-color: var(--accent); }
  .params input[type='text'], .params select {
    flex: 1; background: rgba(0,0,0,0.35); border: 1px solid var(--panel-line); color: var(--text);
    border-radius: 8px; padding: 7px 9px; font-family: var(--mono); font-size: 12px;
  }
  .params b { color: var(--text); font-family: var(--mono); }

  .run {
    border: 0; border-radius: 12px; padding: 12px; cursor: pointer; font-size: 15px; font-weight: 700;
    color: #0a0a0f; background: var(--c); box-shadow: 0 8px 24px -8px var(--c); transition: transform 0.12s, filter 0.2s;
  }
  .run:hover { filter: brightness(1.08); }
  .run:active { transform: scale(0.98); }
  .run.running { background: #2a2a3a; color: #ff7a7a; box-shadow: none; }

  .progress { height: 6px; border-radius: 6px; background: rgba(255,255,255,0.08); overflow: hidden; }
  .progress .bar { height: 100%; transition: width 0.15s linear; box-shadow: 0 0 12px currentColor; }
  .phase { display: flex; align-items: center; gap: 8px; font-family: var(--mono); font-size: 12px; color: var(--muted); }

  .spinner { width: 12px; height: 12px; border-radius: 50%; border: 2px solid rgba(255,255,255,0.15); border-top-color: #fff; }
  .spinner.on { animation: spin 0.7s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }

  .stream { font-family: var(--mono); font-size: 13px; line-height: 1.5; color: var(--text); background: rgba(0,0,0,0.3); border-radius: 10px; padding: 10px; max-height: 120px; overflow: auto; }
  .caret { opacity: 0; } .caret.on { animation: blink 1s steps(2) infinite; }
  @keyframes blink { 50% { opacity: 1; } }

  .console { flex: 1; min-height: 60px; overflow: auto; background: rgba(0,0,0,0.28); border-radius: 10px; padding: 8px 10px; font-family: var(--mono); font-size: 11.5px; color: var(--muted); }
  .console .ln { white-space: pre-wrap; line-height: 1.5; }

  .note { margin: 0; font-size: 11px; line-height: 1.4; color: #6f6f86; }
  .note b { color: #c0c0d0; }

  @media (max-width: 720px) {
    section.dock { left: 12px; right: 12px; width: auto; top: auto; height: 52vh; bottom: 12px; }
    nav.nav { top: 56px; }
  }
</style>
