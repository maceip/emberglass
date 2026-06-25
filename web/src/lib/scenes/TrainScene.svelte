<script>
  import { T, useTask } from '@threlte/core';
  import * as THREE from 'three';

  let { progress = 0, color = '#ff8a5c' } = $props();
  const c = new THREE.Color(color);
  const K = 0.42; // bowl curvature

  // Loss surface (paraboloid) as a wireframe terrain.
  const seg = 48;
  const sgeo = new THREE.PlaneGeometry(4.2, 4.2, seg, seg);
  const p = sgeo.attributes.position;
  for (let i = 0; i < p.count; i++) {
    const x = p.getX(i), y = p.getY(i);
    p.setZ(i, K * (x * x + y * y) * 0.25);
  }
  sgeo.computeVertexNormals();
  const surf = new THREE.Mesh(
    sgeo,
    new THREE.MeshStandardMaterial({ color: c, wireframe: true, emissive: c, emissiveIntensity: 0.25, transparent: true, opacity: 0.55 }),
  );

  // Descending optimizer marble.
  const marble = new THREE.Mesh(
    new THREE.SphereGeometry(0.13, 24, 24),
    new THREE.MeshStandardMaterial({ color: '#fff', emissive: c, emissiveIntensity: 2.2 }),
  );

  // Trail of past positions.
  const TN = 60;
  const tpos = new Float32Array(TN * 3);
  const tgeo = new THREE.BufferGeometry();
  tgeo.setAttribute('position', new THREE.BufferAttribute(tpos, 3));
  const trail = new THREE.Points(tgeo, new THREE.PointsMaterial({ color: c, size: 0.06, transparent: true, opacity: 0.8 }));

  const group = new THREE.Group();
  group.rotation.x = -Math.PI / 2.35;
  group.add(surf, marble, trail);

  const localOf = (t) => {
    const r = (1 - t) * 1.9;
    const a = t * Math.PI * 5;
    const x = r * Math.cos(a), y = r * Math.sin(a);
    return [x, y, K * (x * x + y * y) * 0.25 + 0.12];
  };

  let frame = 0;
  useTask((d) => {
    group.rotation.z += d * 0.08;
    const [x, y, z] = localOf(progress);
    marble.position.set(x, y, z);
    marble.material.emissiveIntensity = 1.6 + Math.sin(frame * 0.2) * 0.5;
    // push trail every few frames
    if (frame % 2 === 0) {
      for (let i = TN - 1; i > 0; i--) {
        tpos[i * 3] = tpos[(i - 1) * 3];
        tpos[i * 3 + 1] = tpos[(i - 1) * 3 + 1];
        tpos[i * 3 + 2] = tpos[(i - 1) * 3 + 2];
      }
      tpos[0] = x; tpos[1] = y; tpos[2] = z;
      tgeo.attributes.position.needsUpdate = true;
    }
    frame++;
  });
</script>

<T is={group} />
<T.PointLight position={[2, 4, 2]} intensity={5} color={c} />
