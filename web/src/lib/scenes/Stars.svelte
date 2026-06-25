<script>
  import { T, useTask } from '@threlte/core';
  import * as THREE from 'three';

  let { count = 1200, radius = 14, color = '#9bb0ff' } = $props();

  const pos = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    const r = radius * (0.5 + Math.random() * 0.5);
    const a = Math.random() * Math.PI * 2;
    const b = Math.acos(2 * Math.random() - 1);
    pos[i * 3] = r * Math.sin(b) * Math.cos(a);
    pos[i * 3 + 1] = r * Math.cos(b);
    pos[i * 3 + 2] = r * Math.sin(b) * Math.sin(a);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const mat = new THREE.PointsMaterial({ color, size: 0.045, transparent: true, opacity: 0.55, sizeAttenuation: true });
  const points = new THREE.Points(geo, mat);

  useTask((d) => {
    points.rotation.y += d * 0.01;
    points.rotation.x += d * 0.004;
  });
</script>

<T is={points} />
