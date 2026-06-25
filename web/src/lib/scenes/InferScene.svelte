<script>
  import { T, useTask } from '@threlte/core';
  import * as THREE from 'three';

  let { progress = 0, color = '#5cffb0' } = $props();
  const c = new THREE.Color(color);
  const dim = new THREE.Color('#1d3a30');

  const group = new THREE.Group();
  const N = 28;
  const cubes = [];
  const geo = new THREE.BoxGeometry(0.26, 0.26, 0.26);
  for (let i = 0; i < N; i++) {
    const m = new THREE.Mesh(
      geo,
      new THREE.MeshStandardMaterial({ color: dim, emissive: dim, emissiveIntensity: 0.4, metalness: 0.5, roughness: 0.3 }),
    );
    m.position.x = (i - (N - 1) / 2) * 0.34;
    cubes.push(m);
    group.add(m);
  }

  let t = 0;
  useTask((d) => {
    t += d;
    group.rotation.y = Math.sin(t * 0.15) * 0.25;
    const revealed = progress * N;
    for (let i = 0; i < N; i++) {
      const m = cubes[i];
      const on = i < revealed;
      const justIn = revealed - i; // >0 and small = freshly emitted
      m.position.y = Math.sin(t * 2 + i * 0.4) * (on ? 0.12 : 0.02);
      m.position.z = on ? 0 : -0.4;
      const pulse = on ? 1 + Math.max(0, 1 - Math.abs(justIn - 0.5)) * 1.6 : 0.3;
      m.material.emissiveIntensity = pulse;
      m.material.color.copy(on ? c : dim);
      m.material.emissive.copy(on ? c : dim);
      const s = on ? 1 + Math.max(0, 1.2 - justIn) * 0.4 : 0.7;
      m.scale.setScalar(THREE.MathUtils.lerp(m.scale.x, s, 0.2));
    }
  });
</script>

<T is={group} />
<T.PointLight position={[0, 1, 4]} intensity={4} color={c} />
