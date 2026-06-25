<script>
  import { T, useTask } from '@threlte/core';
  import * as THREE from 'three';

  let { progress = 0, color = '#6ad7ff' } = $props();
  const c = new THREE.Color(color);
  const ember = new THREE.Color('#ff7a3c');

  const core = new THREE.Mesh(
    new THREE.IcosahedronGeometry(1.15, 1),
    new THREE.MeshStandardMaterial({ color: c, wireframe: true, emissive: c, emissiveIntensity: 0.3, metalness: 0.3, roughness: 0.4 }),
  );
  const inner = new THREE.Mesh(
    new THREE.IcosahedronGeometry(0.7, 0),
    new THREE.MeshStandardMaterial({ color: c, emissive: c, emissiveIntensity: 0.6, transparent: true, opacity: 0.35 }),
  );

  const N = 1500;
  const pos = new Float32Array(N * 3);
  const base = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    const r = 1.8 + Math.random() * 1.8;
    const a = Math.random() * Math.PI * 2;
    const b = Math.acos(2 * Math.random() - 1);
    base[i * 3] = r * Math.sin(b) * Math.cos(a);
    base[i * 3 + 1] = r * Math.cos(b);
    base[i * 3 + 2] = r * Math.sin(b) * Math.sin(a);
  }
  pos.set(base);
  const pgeo = new THREE.BufferGeometry();
  pgeo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const pmat = new THREE.PointsMaterial({ color: c, size: 0.028, transparent: true, opacity: 0.7 });
  const halo = new THREE.Points(pgeo, pmat);

  let t = 0;
  useTask((d) => {
    t += d;
    core.rotation.y += d * 0.45;
    core.rotation.x += d * 0.15;
    inner.rotation.y -= d * 0.6;
    halo.rotation.y -= d * 0.1;
    const ign = THREE.MathUtils.lerp(0.25, 1.8, progress);
    core.material.emissiveIntensity = ign;
    core.material.color.copy(c).lerp(ember, progress * 0.6);
    core.material.emissive.copy(c).lerp(ember, progress * 0.6);
    core.scale.setScalar(1 + progress * 0.12);
    // particles breathe inward as it ignites
    const arr = pgeo.attributes.position.array;
    const k = 1 - progress * 0.28 + Math.sin(t * 0.8) * 0.01;
    for (let i = 0; i < N * 3; i++) arr[i] = base[i] * k;
    pgeo.attributes.position.needsUpdate = true;
    pmat.opacity = 0.4 + progress * 0.4;
  });
</script>

<T is={core} />
<T is={inner} />
<T is={halo} />
<T.PointLight position={[0, 0, 0]} intensity={2 + progress * 6} color={core.material.emissive} distance={10} />
