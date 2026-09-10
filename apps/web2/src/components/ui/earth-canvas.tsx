import { useEffect, useRef } from "react";
import * as THREE from "three";

/**
 * EarthCanvas
 * Renders a rotating 3-D Earth using the same textures from /public/earth/.
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
    renderer.setPixelRatio(window.devicePixelRatio);
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
    const radius = 0.5;
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

    // ─── Animation loop ───────────────────────────────────────────────────
    let animId: number;
    function animate() {
      animId = requestAnimationFrame(animate);
      sphere.rotation.y  += 0.0008;
      clouds.rotation.y  += 0.0010;
      atmosphere.rotation.y += 0.0008;
      renderer.render(scene, camera);
    }
    animate();

    // ─── Resize handler ───────────────────────────────────────────────────
    function onResize() {
      if (!mount) return;
      const w = mount.clientWidth;
      const h = mount.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    }
    window.addEventListener("resize", onResize);

    // ─── Cleanup ──────────────────────────────────────────────────────────
    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener("resize", onResize);
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
