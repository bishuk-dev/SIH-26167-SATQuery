import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';

export const ThreeBackground = () => {
  const mountRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!mountRef.current) return;

    const container = mountRef.current;
    const width = window.innerWidth;
    const height = window.innerHeight;

    // 1. Scene Setup
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.set(0, 0, 18);

    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    // 2. Lighting configuration
    const ambientLight = new THREE.AmbientLight(0x0a1530, 2.0);
    scene.add(ambientLight);

    const sunLight = new THREE.DirectionalLight(0x38bdf8, 3.8);
    sunLight.position.set(15, 8, 12);
    scene.add(sunLight);

    const secondaryLight = new THREE.DirectionalLight(0x818cf8, 1.8);
    secondaryLight.position.set(-15, -5, -8);
    scene.add(secondaryLight);

    // 3. Earth Group positioned on the right/bottom
    const earthGroup = new THREE.Group();
    earthGroup.position.set(8.5, -2.5, 0);
    scene.add(earthGroup);

    const earthRadius = 7.0;
    const earthGeo = new THREE.SphereGeometry(earthRadius, 64, 64);

    // Canvas generated Earth procedural texture with glowing night clusters
    const canvas = document.createElement('canvas');
    canvas.width = 2048;
    canvas.height = 1024;
    const ctx = canvas.getContext('2d');

    if (ctx) {
      // Deep ocean gradient
      const grad = ctx.createLinearGradient(0, 0, 0, 1024);
      grad.addColorStop(0, '#030712');
      grad.addColorStop(0.5, '#02182b');
      grad.addColorStop(1, '#050b14');
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, 2048, 1024);

      // Procedural landmasses
      for (let i = 0; i < 70; i++) {
        const cx = Math.random() * 2048;
        const cy = 200 + Math.random() * 624;
        const radX = 50 + Math.random() * 160;
        const radY = 40 + Math.random() * 120;
        ctx.beginPath();
        ctx.ellipse(cx, cy, radX, radY, Math.random() * Math.PI, 0, Math.PI * 2);
        ctx.fillStyle = '#0b2e3b';
        ctx.fill();

        // City cyan clusters
        ctx.fillStyle = '#38bdf8';
        for (let j = 0; j < 30; j++) {
          const lx = cx + (Math.random() - 0.5) * radX * 1.4;
          const ly = cy + (Math.random() - 0.5) * radY * 1.4;
          ctx.fillRect(lx, ly, 2, 2);
        }
        // Amber cluster lights
        ctx.fillStyle = '#fbbf24';
        for (let j = 0; j < 20; j++) {
          const lx = cx + (Math.random() - 0.5) * radX * 1.2;
          const ly = cy + (Math.random() - 0.5) * radY * 1.2;
          ctx.fillRect(lx, ly, 1.5, 1.5);
        }
      }
    }

    const earthTexture = new THREE.CanvasTexture(canvas);
    const earthMat = new THREE.MeshPhongMaterial({
      map: earthTexture,
      bumpScale: 0.05,
      specular: new THREE.Color(0x22d3ee),
      shininess: 25,
      emissive: new THREE.Color(0x031828),
      emissiveIntensity: 0.6
    });
    const earthMesh = new THREE.Mesh(earthGeo, earthMat);
    earthGroup.add(earthMesh);

    // Earth Atmosphere Glow Shell
    const atmoGeo = new THREE.SphereGeometry(earthRadius * 1.035, 48, 48);
    const atmoMat = new THREE.MeshPhongMaterial({
      color: 0x38bdf8,
      transparent: true,
      opacity: 0.22,
      blending: THREE.AdditiveBlending,
      side: THREE.BackSide
    });
    const atmoMesh = new THREE.Mesh(atmoGeo, atmoMat);
    earthGroup.add(atmoMesh);

    // Outer Cyan Rim Glow
    const rimGeo = new THREE.SphereGeometry(earthRadius * 1.08, 48, 48);
    const rimMat = new THREE.MeshBasicMaterial({
      color: 0x06b6d4,
      transparent: true,
      opacity: 0.12,
      blending: THREE.AdditiveBlending,
      side: THREE.BackSide
    });
    const rimMesh = new THREE.Mesh(rimGeo, rimMat);
    earthGroup.add(rimMesh);

    // 4. Secondary Distant Saturn-like Planet (left side)
    const planetLeftGroup = new THREE.Group();
    planetLeftGroup.position.set(-11, -3.5, -4);
    scene.add(planetLeftGroup);

    const planetLeftGeo = new THREE.SphereGeometry(4.5, 32, 32);
    const planetLeftMat = new THREE.MeshLambertMaterial({
      color: 0x1e1b4b,
      emissive: 0x0f172a,
      emissiveIntensity: 0.8
    });
    const planetLeftMesh = new THREE.Mesh(planetLeftGeo, planetLeftMat);
    planetLeftGroup.add(planetLeftMesh);

    // Rings around left planet
    const ringGeo = new THREE.RingGeometry(5.2, 7.0, 48);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0x4338ca,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.25
    });
    const ringMesh = new THREE.Mesh(ringGeo, ringMat);
    ringMesh.rotation.x = Math.PI * 0.4;
    ringMesh.rotation.y = Math.PI * 0.15;
    planetLeftGroup.add(ringMesh);

    // Moon in upper left
    const moonGeo = new THREE.SphereGeometry(0.9, 24, 24);
    const moonMat = new THREE.MeshLambertMaterial({ color: 0x64748b, emissive: 0x1e293b });
    const moonMesh = new THREE.Mesh(moonGeo, moonMat);
    moonMesh.position.set(-8, 5, -2);
    scene.add(moonMesh);

    // 5. Starfield Particles
    const starGeo = new THREE.BufferGeometry();
    const starCount = 800;
    const starPos = new Float32Array(starCount * 3);
    for (let i = 0; i < starCount * 3; i += 3) {
      starPos[i] = (Math.random() - 0.5) * 80;
      starPos[i + 1] = (Math.random() - 0.5) * 60;
      starPos[i + 2] = (Math.random() - 0.5) * 40 - 10;
    }
    starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
    const starMat = new THREE.PointsMaterial({
      color: 0x93c5fd,
      size: 0.15,
      transparent: true,
      opacity: 0.85
    });
    const starField = new THREE.Points(starGeo, starMat);
    scene.add(starField);

    // 6. Detailed Satellite Model
    const satellite = new THREE.Group();

    // Golden foil body
    const bodyGeo = new THREE.BoxGeometry(0.7, 0.7, 1.2);
    const bodyMat = new THREE.MeshStandardMaterial({
      color: 0xd97706,
      metalness: 0.6,
      roughness: 0.3
    });
    const satBody = new THREE.Mesh(bodyGeo, bodyMat);
    satellite.add(satBody);

    // High-gain dish antenna
    const dishGeo = new THREE.ConeGeometry(0.5, 0.3, 24, 1, true);
    const dishMat = new THREE.MeshStandardMaterial({ color: 0xe2e8f0, side: THREE.DoubleSide });
    const satDish = new THREE.Mesh(dishGeo, dishMat);
    satDish.rotation.x = Math.PI * 0.7;
    satDish.position.set(0, 0.6, 0.2);
    satellite.add(satDish);

    // Optical sensor payload cylinder
    const sensorGeo = new THREE.CylinderGeometry(0.2, 0.25, 0.5, 16);
    const sensorMat = new THREE.MeshStandardMaterial({ color: 0x0284c7 });
    const satSensor = new THREE.Mesh(sensorGeo, sensorMat);
    satSensor.position.set(0, -0.5, 0);
    satellite.add(satSensor);

    // Solar Panels
    const panelGeo = new THREE.BoxGeometry(2.4, 0.04, 0.7);
    const panelMat = new THREE.MeshStandardMaterial({
      color: 0x1d4ed8,
      roughness: 0.2,
      metalness: 0.8
    });

    const leftWing = new THREE.Mesh(panelGeo, panelMat);
    leftWing.position.set(-1.6, 0, 0);
    satellite.add(leftWing);

    const rightWing = new THREE.Mesh(panelGeo, panelMat);
    rightWing.position.set(1.6, 0, 0);
    satellite.add(rightWing);

    // Panel connectors
    const trussGeo = new THREE.CylinderGeometry(0.04, 0.04, 0.6, 8);
    const trussMat = new THREE.MeshBasicMaterial({ color: 0x94a3b8 });
    const leftTruss = new THREE.Mesh(trussGeo, trussMat);
    leftTruss.rotation.z = Math.PI / 2;
    leftTruss.position.set(-0.55, 0, 0);
    satellite.add(leftTruss);

    const rightTruss = new THREE.Mesh(trussGeo, trussMat);
    rightTruss.rotation.z = Math.PI / 2;
    rightTruss.position.set(0.55, 0, 0);
    satellite.add(rightTruss);

    // Beacon light
    const beaconLight = new THREE.PointLight(0x38bdf8, 1.2, 4);
    beaconLight.position.set(0, 0.6, 0.6);
    satellite.add(beaconLight);

    scene.add(satellite);

    // 7. Orbit parameters
    let orbitAngle = 0.8;
    const orbitRadiusX = 10.2;
    const orbitRadiusY = 5.2;
    const orbitRadiusZ = 6.0;

    // Parallax interaction
    let mouseX = 0;
    let mouseY = 0;
    const handleMouseMove = (e: MouseEvent) => {
      mouseX = (e.clientX / window.innerWidth - 0.5) * 0.8;
      mouseY = (e.clientY / window.innerHeight - 0.5) * 0.8;
    };
    window.addEventListener('mousemove', handleMouseMove);

    // Responsive resize
    const handleResize = () => {
      const w = window.innerWidth;
      const h = window.innerHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    // 8. Render loop
    const clock = new THREE.Clock();
    let animationFrameId: number;

    function animate() {
      animationFrameId = requestAnimationFrame(animate);
      const delta = clock.getDelta();
      const time = clock.getElapsedTime();

      // Earth rotation
      earthMesh.rotation.y += delta * 0.12;
      earthMesh.rotation.x = 0.15 * Math.sin(time * 0.05);
      rimMesh.rotation.y += delta * 0.06;
      atmoMesh.rotation.y += delta * 0.09;

      // Satellite orbital movement
      orbitAngle += delta * 0.35;
      const satX = earthGroup.position.x - Math.cos(orbitAngle) * orbitRadiusX;
      const satY = earthGroup.position.y + Math.sin(orbitAngle) * orbitRadiusY + Math.cos(orbitAngle * 0.5) * 1.5;
      const satZ = earthGroup.position.z + Math.sin(orbitAngle) * orbitRadiusZ;
      
      satellite.position.set(satX, satY, satZ);
      satellite.lookAt(earthGroup.position);
      satellite.rotation.z += Math.sin(time * 1.5) * 0.02;

      // Background planet gentle drift
      planetLeftGroup.rotation.y += delta * 0.03;
      moonMesh.rotation.y += delta * 0.05;

      // Subtle camera response
      camera.position.x += (mouseX - camera.position.x) * 0.04;
      camera.position.y += (-mouseY - camera.position.y) * 0.04;
      camera.lookAt(0, 0, 0);

      renderer.render(scene, camera);
    }

    animate();

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(animationFrameId);
      if (mountRef.current) {
        mountRef.current.removeChild(renderer.domElement);
      }
      
      // Cleanup
      earthTexture.dispose();
      earthGeo.dispose();
      earthMat.dispose();
      atmoGeo.dispose();
      atmoMat.dispose();
      rimGeo.dispose();
      rimMat.dispose();
      planetLeftGeo.dispose();
      planetLeftMat.dispose();
      ringGeo.dispose();
      ringMat.dispose();
      moonGeo.dispose();
      moonMat.dispose();
      starGeo.dispose();
      starMat.dispose();
      bodyGeo.dispose();
      bodyMat.dispose();
      dishGeo.dispose();
      dishMat.dispose();
      sensorGeo.dispose();
      sensorMat.dispose();
      panelGeo.dispose();
      panelMat.dispose();
      trussGeo.dispose();
      trussMat.dispose();
    };
  }, []);

  return (
    <>
      <div 
        ref={mountRef} 
        className="fixed inset-0 w-full h-full pointer-events-none z-0 overflow-hidden" 
        data-purpose="3d-scene-background"
      />
      {/* Ambient Space Vignette Overlay */}
      <div className="fixed inset-0 space-vignette pointer-events-none z-[1]"></div>
    </>
  );
};
