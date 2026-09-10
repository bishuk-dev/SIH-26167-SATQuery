import { useEffect, useRef } from "react";
import * as THREE from "three";

interface OrbitDefinition {
  radius: number;
  rotation: [number, number, number];
  speed: number;
  phase: number;
  color: number;
}

const ORBITS: OrbitDefinition[] = [
  { radius: 0.49, rotation: [0.35, 0.68, 0.12], speed: 0.34, phase: 0.2, color: 0x35d9ff },
  { radius: 0.52, rotation: [1.02, -0.28, 0.62], speed: 0.27, phase: 2.1, color: 0x65efff },
  { radius: 0.55, rotation: [0.72, 0.32, -0.72], speed: 0.3, phase: 3.8, color: 0x397cff },
  { radius: 0.58, rotation: [1.3, 0.18, 0.18], speed: 0.23, phase: 5.2, color: 0x36c7ff },
];

function createOrbitPoints(radius: number, start = 0, arc = Math.PI * 2, segments = 160) {
  return Array.from({ length: segments + 1 }, (_, index) => {
    const angle = start + (index / segments) * arc;
    return new THREE.Vector3(Math.cos(angle) * radius, Math.sin(angle) * radius, 0);
  });
}

function createSatellite(materials: {
  body: THREE.Material;
  panel: THREE.Material;
  antenna: THREE.Material;
}) {
  const satellite = new THREE.Group();

  const body = new THREE.Mesh(new THREE.BoxGeometry(0.027, 0.02, 0.02), materials.body);
  const leftPanel = new THREE.Mesh(new THREE.BoxGeometry(0.045, 0.014, 0.003), materials.panel);
  const rightPanel = leftPanel.clone();
  const antenna = new THREE.Mesh(new THREE.SphereGeometry(0.009, 10, 10), materials.antenna);

  leftPanel.position.x = -0.038;
  rightPanel.position.x = 0.038;
  antenna.position.y = 0.018;
  satellite.add(body, leftPanel, rightPanel, antenna);
  satellite.scale.setScalar(0.3);

  return satellite;
}

/**
 * EarthCanvas
 * Renders a rotating 3-D Earth and continuously orbiting satellites using the
 * textures from /public/earth/.
 * The canvas is absolutely positioned, filling its container, so you control
 * size and placement purely via the parent element's CSS.
 *
 * The scene shows just the Earth sphere + cloud layer (no starfield bg) so the
 * parent page background shows through on the transparent renderer.
 */
export default function EarthCanvas() {
  const mountRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    // ─── Scene ────────────────────────────────────────────────────────────
    const scene = new THREE.Scene();

    // ─── Camera ───────────────────────────────────────────────────────────
    const camera = new THREE.PerspectiveCamera(
      45,
      mount.clientWidth / mount.clientHeight,
      0.01,
      1000
    );
    camera.position.z = 1.6;

    // ─── Renderer (transparent bg so page bg shows through) ───────────────
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,        // transparent canvas background
    });
    // A capped pixel ratio keeps the animation smooth on high-density displays.
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    renderer.setClearColor(0x000000, 0);   // fully transparent
    mount.appendChild(renderer.domElement);

    // ─── Lighting ─────────────────────────────────────────────────────────
    scene.add(new THREE.AmbientLight(0x333333));
    const sunLight = new THREE.DirectionalLight(0xffffff, 1.2);
    sunLight.position.set(5, 3, 5);
    scene.add(sunLight);

    // ─── Texture loader ───────────────────────────────────────────────────
    const loader = new THREE.TextureLoader();

    // ─── Earth sphere ─────────────────────────────────────────────────────
    // 0.4 is exactly 80% of the original 0.5-radius globe.
    const radius = 0.4;
    const segments = 48;

    const earthMat = new THREE.MeshPhongMaterial({
      map:         loader.load("/earth/2_no_clouds_4k.jpg"),
      bumpMap:     loader.load("/earth/elev_bump_4k.jpg"),
      bumpScale:   0.005,
      specularMap: loader.load("/earth/water_4k.png"),
      specular:    new THREE.Color("grey"),
    });
    const sphere = new THREE.Mesh(
      new THREE.SphereGeometry(radius, segments, segments),
      earthMat
    );
    sphere.rotation.y = 6;
    scene.add(sphere);

    // ─── Cloud layer ──────────────────────────────────────────────────────
    const cloudMat = new THREE.MeshPhongMaterial({
      map:         loader.load("/earth/fair_clouds_4k.png"),
      transparent: true,
      opacity:     0.7,
    });
    const clouds = new THREE.Mesh(
      new THREE.SphereGeometry(radius + 0.003, segments, segments),
      cloudMat
    );
    clouds.rotation.y = 6;
    scene.add(clouds);

    // ─── Atmosphere glow (additive blending rim) ──────────────────────────
    const atmMat = new THREE.MeshPhongMaterial({
      color:       new THREE.Color(0x4488ff),
      transparent: true,
      opacity:     0.08,
      side:        THREE.FrontSide,
    });
    const atmosphere = new THREE.Mesh(
      new THREE.SphereGeometry(radius + 0.02, segments, segments),
      atmMat
    );
    scene.add(atmosphere);

    // ─── Orbital rings and satellites ────────────────────────────────────
    const satelliteMaterials = {
      body: new THREE.MeshBasicMaterial({ color: 0xf5fbff }),
      panel: new THREE.MeshBasicMaterial({ color: 0x4bc9ff }),
      antenna: new THREE.MeshBasicMaterial({ color: 0xffffff }),
    };
    const satellites: Array<{
      satellite: THREE.Group;
      trail: THREE.Line;
      speed: number;
      angle: number;
      radius: number;
    }> = [];

    ORBITS.forEach((orbit) => {
      const orbitPlane = new THREE.Group();
      orbitPlane.rotation.set(...orbit.rotation);

      const ringGeometry = new THREE.BufferGeometry().setFromPoints(
        createOrbitPoints(orbit.radius)
      );
      const ringMaterial = new THREE.LineBasicMaterial({
        color: orbit.color,
        transparent: true,
        opacity: 0.24,
        blending: THREE.AdditiveBlending,
      });
      orbitPlane.add(new THREE.LineLoop(ringGeometry, ringMaterial));

      const trailGeometry = new THREE.BufferGeometry().setFromPoints(
        createOrbitPoints(orbit.radius, -0.72, 0.72, 36)
      );
      const trailMaterial = new THREE.LineBasicMaterial({
        color: orbit.color,
        transparent: true,
        opacity: 0.9,
        blending: THREE.AdditiveBlending,
      });
      const trail = new THREE.Line(trailGeometry, trailMaterial);
      orbitPlane.add(trail);

      const satellite = createSatellite(satelliteMaterials);
      orbitPlane.add(satellite);
      scene.add(orbitPlane);
      satellites.push({
        satellite,
        trail,
        speed: orbit.speed,
        angle: orbit.phase,
        radius: orbit.radius,
      });
    });

    // ─── Animation loop ───────────────────────────────────────────────────
    let animId: number;
    let lastFrame = performance.now();
    function animate(now: number) {
      animId = requestAnimationFrame(animate);
      const deltaSeconds = Math.min((now - lastFrame) / 1000, 0.05);
      lastFrame = now;

      sphere.rotation.y += 0.048 * deltaSeconds;
      clouds.rotation.y += 0.06 * deltaSeconds;
      atmosphere.rotation.y += 0.048 * deltaSeconds;

      satellites.forEach((orbitState) => {
        const { satellite, trail, speed, radius: orbitRadius } = orbitState;
        orbitState.angle += speed * deltaSeconds;
        const angle = orbitState.angle;
        satellite.position.set(
          Math.cos(angle) * orbitRadius,
          Math.sin(angle) * orbitRadius,
          0
        );
        satellite.rotation.z = angle + Math.PI / 2;
        trail.rotation.z = angle;
      });

      renderer.render(scene, camera);
    }
    animId = requestAnimationFrame(animate);

    // ─── Resize handler ───────────────────────────────────────────────────
    function onResize() {
      if (!mount) return;
      const w = mount.clientWidth;
      const h = mount.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.setSize(w, h);
    }
    window.addEventListener("resize", onResize);

    // ─── Cleanup ──────────────────────────────────────────────────────────
    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener("resize", onResize);
      scene.traverse((object) => {
        if (object instanceof THREE.Mesh || object instanceof THREE.Line) {
          object.geometry.dispose();
          const materials = Array.isArray(object.material) ? object.material : [object.material];
          materials.forEach((material) => material.dispose());
        }
      });
      Object.values(satelliteMaterials).forEach((material) => material.dispose());
      [earthMat.map, earthMat.bumpMap, earthMat.specularMap, cloudMat.map]
        .filter((texture): texture is THREE.Texture => texture !== null)
        .forEach((texture) => texture.dispose());
      renderer.dispose();
      if (mount.contains(renderer.domElement)) {
        mount.removeChild(renderer.domElement);
      }
    };
  }, []);

  return (
    <div
      ref={mountRef}
      className="w-full h-full"
      style={{ display: "block" }}
    />
  );
}
