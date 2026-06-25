<script>
  import { T, useTask } from '@threlte/core';
  import * as THREE from 'three';
  import Stars from './Stars.svelte';
  import DemoScene from './DemoScene.svelte';
  import TrainScene from './TrainScene.svelte';
  import TuneScene from './TuneScene.svelte';
  import InferScene from './InferScene.svelte';

  let { mode = 'demo', progress = 0, color = '#6ad7ff', pointer = { x: 0, y: 0 } } = $props();

  let rig = $state();
  useTask((d) => {
    if (!rig) return;
    // gentle parallax toward the pointer
    rig.rotation.y += (pointer.x * 0.35 - rig.rotation.y) * Math.min(1, d * 2.5);
    rig.rotation.x += (-pointer.y * 0.25 - rig.rotation.x) * Math.min(1, d * 2.5);
  });
</script>

<T.PerspectiveCamera makeDefault position={[0, 1.4, 6.2]} fov={50} oncreate={(c) => c.lookAt(0, 0, 0)} />
<T.AmbientLight intensity={0.35} />
<T.DirectionalLight position={[4, 6, 3]} intensity={0.6} />

<T.Group bind:ref={rig}>
  <Stars />
  {#if mode === 'demo'}
    <DemoScene {progress} {color} />
  {:else if mode === 'train'}
    <TrainScene {progress} {color} />
  {:else if mode === 'tune'}
    <TuneScene {progress} {color} />
  {:else if mode === 'inference'}
    <InferScene {progress} {color} />
  {/if}
</T.Group>
