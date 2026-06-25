<script>
  import { T, useTask } from '@threlte/core';
  import * as THREE from 'three';

  let { progress = 0, color = '#c08bff' } = $props();
  const c = new THREE.Color(color);

  const group = new THREE.Group();
  const rings = [];
  const RN = 5;
  for (let i = 0; i < RN; i++) {
    const radius = 0.7 + i * 0.42;
    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(radius, 0.022 + (RN - i) * 0.004, 16, 120),
      new THREE.MeshStandardMaterial({ color: c, emissive: c, emissiveIntensity: 0.8, metalness: 0.4, roughness: 0.3 }),
    );
    ring.userData.tilt = (Math.random() - 0.5) * 1.6;
    ring.userData.tilt2 = (Math.random() - 0.5) * 1.6;
    ring.userData.spin = (0.3 + Math.random() * 0.6) * (i % 2 ? 1 : -1);
    rings.push(ring);
    group.add(ring);
  }
  // marker bead per ring that snaps to top when "tuned"
  const beads = [];
  for (let i = 0; i < RN; i++) {
    const bead = new THREE.Mesh(
      new THREE.SphereGeometry(0.06, 16, 16),
      new THREE.MeshStandardMaterial({ color: '#fff', emissive: '#fff', emissiveIntensity: 2 }),
    );
    beads.push(bead);
    rings[i].add(bead);
  }

  let t = 0;
  useTask((d) => {
    t += d;
    group.rotation.y += d * 0.25;
    const ease = progress * progress * (3 - 2 * progress); // smoothstep
    for (let i = 0; i < RN; i++) {
      const r = rings[i];
      // tilts converge to aligned (0) as we lock in
      r.rotation.x = r.userData.tilt * (1 - ease);
      r.rotation.y = r.userData.tilt2 * (1 - ease) + t * r.userData.spin * (1 - ease * 0.7);
      const ei = 0.5 + ease * 2.2;
      r.material.emissiveIntensity = ei;
      const radius = 0.7 + i * 0.42;
      const ang = t * r.userData.spin + (1 - ease) * i;
      beads[i].position.set(Math.cos(ang) * radius, Math.sin(ang) * radius, 0);
      beads[i].material.emissiveIntensity = 1 + ease * 3;
    }
  });
</script>

<T is={group} />
<T.PointLight position={[0, 0, 4]} intensity={4} color={c} />
