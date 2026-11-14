/**
 * CityVerse Web Application Frontend Logic
 * Realistic 3D Urban Simulation Twin with Moving Traffic & Hydrodynamic Flooding
 * 
 * Powered by Three.js WebGL Engine & Vectorized 2D Physics.
 */

// Application State Container
const state = {
    isPlaying: false,
    playTimer: null,
    currentStepIdx: 0,
    totalSteps: 36,
    stepDtSeconds: 300.0, // 5 minutes per step
    engineMode: "physics", // Default to exact numerical physics
    currentBranch: "baseline",
    forkedBranch: null,

    // 3D Scene Core
    scene: null,
    camera: null,
    renderer: null,
    controls: null,
    clock: new THREE.Clock(),

    // 3D Meshes & Groups
    terrainMesh: null,
    waterMesh: null,
    roadLinesGroup: null,
    barriersGroup: null,
    buildingsGroup: null,
    facilitiesGroup: null,
    trafficGroup: null,
    rainSystem: null,
    dirLight: null,
    hemiLight: null,
    flashLight: null,

    // Dynamic Atmospheric SkyDome, Sun Disc & Aviation Beacons
    skyDome: null,
    skyCanvas: null,
    skyCanvasTex: null,
    sunMesh: null,
    aviationBeacons: [],
    rainData: [],

    // Environmental Atmospheric Weather & Lighting System
    weatherMode: "auto", // "auto", "day", "night"
    currentWeatherFactor: 0.5, // 0.0 (Clear Sun) to 1.0 (Severe Tempest)
    targetWeatherFactor: 0.5,

    // Traffic Fleet
    vehicles: [],
    roadGraph: {}, // edge_id -> { p1, p2, length, class, connections }

    // Camera Modes
    isDroneOrbit: false,
    orbitAngle: 0.0,

    // GIS Data
    terrain: null,
    roads: [],
    facilities: [],
    footprints: [],
    blocks: [],

    // Simulation Trajectory Cache
    frames: [],
    currentFrame: null,

    // Sparklines Telemetry
    historyFloodedArea: [],
    historyEmsTime: [],
};

// World Dimensions: 6000m x 6000m -> 600 x 600 units in 3D (-300 to 300)
const SCALE = 0.1;
const HALF_EXTENT = 300.0;

// DOM References
const elSimTime = document.getElementById("simTimeDisplay");
const elEngineBadge = document.getElementById("engineBadge");
const elRunScenario = document.getElementById("btnRunFullScenario");
const elPlayPause = document.getElementById("btnPlayPause");
const elPlayPauseLabel = document.getElementById("playPauseLabel");
const elPlayIcon = document.getElementById("playIcon");
const elReset = document.getElementById("btnReset");

// Timeline Controls
const elTimelineScrubber = document.getElementById("timelineScrubber");
const elTimelineTimeLabel = document.getElementById("timelineTimeLabel");
const elTimelineTotalLabel = document.getElementById("timelineTotalLabel");
const elBtnStepBack = document.getElementById("btnStepBack");
const elBtnStepPlay = document.getElementById("btnStepPlay");
const elBtnStepForward = document.getElementById("btnStepForward");
const elTimelinePlayIcon = document.getElementById("timelinePlayIcon");

// Sliders
const sliderRain = document.getElementById("sliderRainfall");
const sliderDuration = document.getElementById("sliderDuration");
const sliderDrain = document.getElementById("sliderDrainage");
const sliderPump1 = document.getElementById("sliderPump1");
const sliderPump2 = document.getElementById("sliderPump2");

const valRain = document.getElementById("valRainfall");
const valDuration = document.getElementById("valDuration");
const valDrain = document.getElementById("valDrainage");
const valPump1 = document.getElementById("valPump1");
const valPump2 = document.getElementById("valPump2");

// NLP
const btnNlp = document.getElementById("btnNlpExecute");
const nlpInput = document.getElementById("nlpInput");
const nlpTags = document.getElementById("nlpTags");

// Engine Mode Buttons
const modeNeuralBtn = document.getElementById("modeNeural");
const modePhysicsBtn = document.getElementById("modePhysics");

// Branching
const btnFork = document.getElementById("btnForkBranch");
const btnCompare = document.getElementById("btnCompareBranches");
const diffSection = document.getElementById("diffSection");

// Inspector
const inspectorCard = document.getElementById("inspectorCard");
const inspectorContent = document.getElementById("inspectorContent");

// KPIs
const kpiFlooded = document.getElementById("kpiFloodedArea");
const kpiFlooded30 = document.getElementById("kpiFloodedArea30cm");
const kpiMaxDepth = document.getElementById("kpiMaxDepth");
const kpiTotalVolume = document.getElementById("kpiTotalVolume");
const kpiClosedRoads = document.getElementById("kpiClosedRoads");
const kpiClosedEmergency = document.getElementById("kpiClosedEmergency");
const kpiEms = document.getElementById("kpiEmsResponse");
const kpiEmsCov = document.getElementById("kpiEmsCoverage");
const kpiSubs = document.getElementById("kpiSubstations");
const kpiSubLoad = document.getElementById("kpiSubstationLoad");
const kpiPumps = document.getElementById("kpiPumps");
const kpiPumpStatus = document.getElementById("kpiPumpStatus");


// =========================================================================
// APPLICATION INITIALIZATION
// =========================================================================

async function init() {
    try {
        console.log("Initializing CityVerse 3D Realistic Simulation Environment...");

        // 1. Initialize Three.js WebGL Scene
        initThreeScene();

        // 2. Fetch GIS City Models
        await loadCityData();

        // 3. Build 3D City Environment (Terrain, Buildings, Roads, Facilities, Rain)
        build3DCityScene();

        // 4. Setup UI & Camera Listeners
        setupEventListeners();

        // 5. Fast Load Pre-Computed Simulation Trajectory FIRST (calculates road flooding & closures)
        const loaded = await loadCachedFrames();
        if (!loaded) {
            await runScenario(false);
        }

        // 6. Spawn Moving Traffic Fleet strictly on Open Dry Roads
        spawnTrafficFleet(180);

        // 7. Start Animation & Physics Render Loop
        animate();

        console.log("CityVerse 3D initialized successfully.");
    } catch (err) {
        console.error("Initialization error:", err);
    }
}


// =========================================================================
// PROCEDURAL ARCHITECTURAL FACADE & ATMOSPHERIC TEXTURES
// =========================================================================

function interpolateHexColor(hex1, hex2, t) {
    const r1 = parseInt(hex1.slice(1, 3), 16), g1 = parseInt(hex1.slice(3, 5), 16), b1 = parseInt(hex1.slice(5, 7), 16);
    const r2 = parseInt(hex2.slice(1, 3), 16), g2 = parseInt(hex2.slice(3, 5), 16), b2 = parseInt(hex2.slice(5, 7), 16);
    const r = Math.round(r1 + (r2 - r1) * t);
    const g = Math.round(g1 + (g2 - g1) * t);
    const b = Math.round(b1 + (b2 - b1) * t);
    return `rgb(${r},${g},${b})`;
}

function createCommercialGlassTexture() {
    const canvas = document.createElement("canvas");
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext("2d");

    ctx.fillStyle = "#0a0f1d";
    ctx.fillRect(0, 0, 512, 512);

    const floors = 16;
    const bays = 16;
    const bayW = 512 / bays;
    const floorH = 512 / floors;

    for (let f = 0; f < floors; f++) {
        const y = f * floorH;
        ctx.fillStyle = "#1e293b";
        ctx.fillRect(0, y, 512, 5);

        for (let b = 0; b < bays; b++) {
            const x = b * bayW;
            ctx.fillStyle = "#0f172a";
            ctx.fillRect(x, y + 5, 3, floorH - 5);

            const winX = x + 3;
            const winY = y + 5;
            const winW = bayW - 4;
            const winH = floorH - 6;

            const hash = Math.sin(f * 29.3 + b * 71.9) * 10000;
            const rand = hash - Math.floor(hash);

            if (rand > 0.58) {
                const grad = ctx.createLinearGradient(winX, winY, winX, winY + winH);
                if (rand > 0.88) {
                    grad.addColorStop(0, "#fef08a");
                    grad.addColorStop(1, "#f59e0b");
                } else if (rand > 0.74) {
                    grad.addColorStop(0, "#e0f2fe");
                    grad.addColorStop(1, "#93c5fd");
                } else {
                    grad.addColorStop(0, "#fed7aa");
                    grad.addColorStop(1, "#fb923c");
                }
                ctx.fillStyle = grad;
            } else {
                const grad = ctx.createLinearGradient(winX, winY, winX, winY + winH);
                grad.addColorStop(0, "#1e3a5f");
                grad.addColorStop(1, "#0a192f");
                ctx.fillStyle = grad;
            }
            ctx.fillRect(winX, winY, winW, winH);
        }
    }

    const tex = new THREE.CanvasTexture(canvas);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    return tex;
}

function createCommercialEmissiveTexture() {
    const canvas = document.createElement("canvas");
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext("2d");

    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, 512, 512);

    const floors = 16;
    const bays = 16;
    const bayW = 512 / bays;
    const floorH = 512 / floors;

    for (let f = 0; f < floors; f++) {
        const y = f * floorH;
        for (let b = 0; b < bays; b++) {
            const x = b * bayW;
            const winX = x + 3;
            const winY = y + 5;
            const winW = bayW - 4;
            const winH = floorH - 6;

            const hash = Math.sin(f * 29.3 + b * 71.9) * 10000;
            const rand = hash - Math.floor(hash);

            if (rand > 0.58) {
                if (rand > 0.88) ctx.fillStyle = "#fef08a";
                else if (rand > 0.74) ctx.fillStyle = "#93c5fd";
                else ctx.fillStyle = "#fed7aa";
                ctx.fillRect(winX, winY, winW, winH);
            }
        }
    }

    const tex = new THREE.CanvasTexture(canvas);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    return tex;
}

function createResidentialTexture() {
    const canvas = document.createElement("canvas");
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext("2d");

    ctx.fillStyle = "#334155";
    ctx.fillRect(0, 0, 512, 512);

    const floors = 12;
    const bays = 10;
    const bayW = 512 / bays;
    const floorH = 512 / floors;

    for (let f = 0; f < floors; f++) {
        const y = f * floorH;
        ctx.fillStyle = "#475569";
        ctx.fillRect(0, y + floorH - 8, 512, 7);

        for (let b = 0; b < bays; b++) {
            const x = b * bayW;
            const winX = x + 10;
            const winY = y + 8;
            const winW = bayW - 20;
            const winH = floorH - 22;

            ctx.fillStyle = "#1e293b";
            ctx.fillRect(winX - 3, winY - 3, winW + 6, winH + 6);

            const hash = Math.sin(f * 37.1 + b * 61.3) * 10000;
            const rand = hash - Math.floor(hash);

            if (rand > 0.45) {
                ctx.fillStyle = rand > 0.75 ? "#fef3c7" : "#fde68a";
            } else {
                ctx.fillStyle = "#0f172a";
            }
            ctx.fillRect(winX, winY, winW, winH);
        }
    }

    const tex = new THREE.CanvasTexture(canvas);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    return tex;
}

function createResidentialEmissiveTexture() {
    const canvas = document.createElement("canvas");
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext("2d");

    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, 512, 512);

    const floors = 12;
    const bays = 10;
    const bayW = 512 / bays;
    const floorH = 512 / floors;

    for (let f = 0; f < floors; f++) {
        const y = f * floorH;
        for (let b = 0; b < bays; b++) {
            const x = b * bayW;
            const winX = x + 10;
            const winY = y + 8;
            const winW = bayW - 20;
            const winH = floorH - 22;

            const hash = Math.sin(f * 37.1 + b * 61.3) * 10000;
            const rand = hash - Math.floor(hash);

            if (rand > 0.45) {
                ctx.fillStyle = rand > 0.75 ? "#fef3c7" : "#fde68a";
                ctx.fillRect(winX, winY, winW, winH);
            }
        }
    }

    const tex = new THREE.CanvasTexture(canvas);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    return tex;
}

function createIndustrialTexture() {
    const canvas = document.createElement("canvas");
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext("2d");

    ctx.fillStyle = "#1e293b";
    ctx.fillRect(0, 0, 512, 512);

    for (let x = 0; x < 512; x += 12) {
        ctx.fillStyle = (x % 24 === 0) ? "#334155" : "#0f172a";
        ctx.fillRect(x, 0, 6, 512);
    }

    for (let b = 24; b < 480; b += 110) {
        ctx.fillStyle = "#475569";
        ctx.fillRect(b, 320, 80, 180);
        for (let sy = 328; sy < 490; sy += 10) {
            ctx.fillStyle = "#1e293b";
            ctx.fillRect(b + 4, sy, 72, 3);
        }
    }

    ctx.fillStyle = "#0284c7";
    ctx.fillRect(16, 60, 480, 36);

    const tex = new THREE.CanvasTexture(canvas);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    return tex;
}

function createRoofTexture() {
    const canvas = document.createElement("canvas");
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext("2d");

    ctx.fillStyle = "#1e293b";
    ctx.fillRect(0, 0, 256, 256);

    for (let i = 0; i < 900; i++) {
        const rx = Math.random() * 256;
        const ry = Math.random() * 256;
        ctx.fillStyle = Math.random() > 0.5 ? "#334155" : "#0f172a";
        ctx.fillRect(rx, ry, 2, 2);
    }

    ctx.strokeStyle = "#0f172a";
    ctx.lineWidth = 2;
    ctx.strokeRect(32, 32, 192, 192);

    const tex = new THREE.CanvasTexture(canvas);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    return tex;
}

function createHelipadTexture() {
    const canvas = document.createElement("canvas");
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext("2d");

    ctx.fillStyle = "#1e293b";
    ctx.beginPath();
    ctx.arc(128, 128, 120, 0, Math.PI * 2);
    ctx.fill();

    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 8;
    ctx.beginPath();
    ctx.arc(128, 128, 110, 0, Math.PI * 2);
    ctx.stroke();

    ctx.strokeStyle = "#facc15";
    ctx.lineWidth = 10;
    ctx.beginPath();
    ctx.arc(128, 128, 76, 0, Math.PI * 2);
    ctx.stroke();

    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 90px 'Arial Black', sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("H", 128, 128);

    for (let a = 0; a < Math.PI * 2; a += Math.PI / 6) {
        const cx = 128 + Math.cos(a) * 110;
        const cy = 128 + Math.sin(a) * 110;
        ctx.fillStyle = "#10b981";
        ctx.beginPath();
        ctx.arc(cx, cy, 4, 0, Math.PI * 2);
        ctx.fill();
    }

    return new THREE.CanvasTexture(canvas);
}

function createSunCoronaTexture() {
    const canvas = document.createElement("canvas");
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext("2d");

    const grad = ctx.createRadialGradient(128, 128, 12, 128, 128, 126);
    grad.addColorStop(0.0, "rgba(255, 255, 255, 1.0)");
    grad.addColorStop(0.2, "rgba(254, 240, 138, 0.95)");
    grad.addColorStop(0.5, "rgba(245, 158, 11, 0.45)");
    grad.addColorStop(0.8, "rgba(234, 88, 12, 0.15)");
    grad.addColorStop(1.0, "rgba(234, 88, 12, 0.0)");

    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 256, 256);

    return new THREE.CanvasTexture(canvas);
}

function createRainDropTexture() {
    const canvas = document.createElement("canvas");
    canvas.width = 16;
    canvas.height = 64;
    const ctx = canvas.getContext("2d");
    const grad = ctx.createLinearGradient(8, 0, 8, 64);
    grad.addColorStop(0.0, "rgba(224, 242, 254, 0.0)");
    grad.addColorStop(0.3, "rgba(186, 230, 253, 0.35)");
    grad.addColorStop(0.8, "rgba(147, 197, 253, 0.75)");
    grad.addColorStop(1.0, "rgba(255, 255, 255, 0.90)");
    ctx.fillStyle = grad;
    ctx.fillRect(4, 0, 8, 64);
    return new THREE.CanvasTexture(canvas);
}

function applyBoxFaceUVs(geom, width, height, depth, tileW = 5.0, tileH = 3.6) {
    const uvAttr = geom.attributes.uv;
    if (!uvAttr) return;

    for (let i = 0; i < uvAttr.count; i++) {
        let u = uvAttr.getX(i);
        let v = uvAttr.getY(i);

        if (i < 8) {
            u *= Math.max(1.0, depth / tileW);
            v *= Math.max(1.0, height / tileH);
        } else if (i < 16) {
            u *= Math.max(1.0, width / tileW);
            v *= Math.max(1.0, depth / tileW);
        } else {
            u *= Math.max(1.0, width / tileW);
            v *= Math.max(1.0, height / tileH);
        }
        uvAttr.setXY(i, u, v);
    }
    uvAttr.needsUpdate = true;
}

function initSkyDome() {
    const canvas = document.createElement("canvas");
    canvas.width = 512;
    canvas.height = 512;
    state.skyCanvas = canvas;
    state.skyCanvasTex = new THREE.CanvasTexture(canvas);

    const skyGeom = new THREE.SphereGeometry(1400, 32, 24);
    const skyMat = new THREE.MeshBasicMaterial({
        map: state.skyCanvasTex,
        side: THREE.BackSide,
        depthWrite: false,
        fog: false
    });
    state.skyDome = new THREE.Mesh(skyGeom, skyMat);
    state.scene.add(state.skyDome);

    // Glowing Sun Disc with Atmospheric Flare
    const sunGeom = new THREE.PlaneGeometry(80, 80);
    const sunMat = new THREE.MeshBasicMaterial({
        map: createSunCoronaTexture(),
        transparent: true,
        opacity: 1.0,
        side: THREE.DoubleSide,
        depthWrite: false,
        fog: false,
        blending: THREE.AdditiveBlending
    });
    state.sunMesh = new THREE.Mesh(sunGeom, sunMat);
    const sunDir = new THREE.Vector3(200, 350, 150).normalize();
    state.sunMesh.position.copy(sunDir.multiplyScalar(1250));
    state.scene.add(state.sunMesh);

    updateSkyDomeGradient(0.0);
}

function updateSkyDomeGradient(wf) {
    if (!state.skyCanvas) return;
    const ctx = state.skyCanvas.getContext("2d");
    const grad = ctx.createLinearGradient(0, 0, 0, 512);

    if (state.weatherMode === "night" || wf < -0.5) {
        grad.addColorStop(0.0, "#020617");
        grad.addColorStop(0.4, "#060b19");
        grad.addColorStop(0.75, "#0d1b2a");
        grad.addColorStop(1.0, "#1b2a47");
    } else {
        const t = npClamp(wf, 0.0, 1.0);
        const zenith = interpolateHexColor("#0284c7", "#050a14", t);
        const midSky = interpolateHexColor("#38bdf8", "#0f172a", t);
        const lowerSky = interpolateHexColor("#bae6fd", "#1e293b", t);
        const horizon = interpolateHexColor("#fef08a", "#182234", t);

        grad.addColorStop(0.0, zenith);
        grad.addColorStop(0.4, midSky);
        grad.addColorStop(0.75, lowerSky);
        grad.addColorStop(1.0, horizon);
    }

    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 512, 512);
    if (state.skyCanvasTex) state.skyCanvasTex.needsUpdate = true;
}


// =========================================================================
// THREE.JS 3D SCENE & LIGHTING SETUP
// =========================================================================

function initThreeScene() {
    const container = document.getElementById("city3dViewport");
    const width = container.clientWidth || window.innerWidth;
    const height = container.clientHeight || (window.innerHeight - 150);

    // Scene with dynamic atmospheric fog and sky
    state.scene = new THREE.Scene();
    state.scene.fog = new THREE.FogExp2(0xcbe8fc, 0.00035);

    // Perspective Camera
    state.camera = new THREE.PerspectiveCamera(45, width / height, 1.0, 3000);
    state.camera.position.set(0, 260, 320);

    // WebGL Renderer with Shadows
    state.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    state.renderer.setSize(width, height);
    state.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2.0));
    state.renderer.shadowMap.enabled = true;
    state.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(state.renderer.domElement);

    // Orbit Controls
    state.controls = new THREE.OrbitControls(state.camera, state.renderer.domElement);
    state.controls.enableDamping = true;
    state.controls.dampingFactor = 0.05;
    state.controls.maxPolarAngle = Math.PI / 2.08; // Prevent going under ground
    state.controls.minDistance = 30;
    state.controls.maxDistance = 900;
    state.controls.target.set(0, 15, 0);

    // Dynamic Realistic SkyDome & Sun Disc
    initSkyDome();

    // Ambient Hemisphere Lighting
    state.hemiLight = new THREE.HemisphereLight(0xe0f2fe, 0x475569, 0.75);
    state.scene.add(state.hemiLight);

    // Directional Sunlight with Shadows
    state.dirLight = new THREE.DirectionalLight(0xfffaed, 1.85);
    state.dirLight.position.set(200, 350, 150);
    state.dirLight.castShadow = true;
    state.dirLight.shadow.mapSize.width = 2048;
    state.dirLight.shadow.mapSize.height = 2048;
    state.dirLight.shadow.camera.near = 50;
    state.dirLight.shadow.camera.far = 800;
    state.dirLight.shadow.camera.left = -320;
    state.dirLight.shadow.camera.right = 320;
    state.dirLight.shadow.camera.top = 320;
    state.dirLight.shadow.camera.bottom = -320;
    state.scene.add(state.dirLight);

    // Lightning Flash Light for heavy storms
    state.flashLight = new THREE.PointLight(0xdbeafe, 0, 1200);
    state.flashLight.position.set(0, 300, 0);
    state.scene.add(state.flashLight);

    // Initialize atmospheric weather from current slider value
    const initRain = parseFloat(document.getElementById("sliderRainfall")?.value || 0.0);
    state.targetWeatherFactor = npClamp(initRain / 130.0, 0.0, 1.0);
    state.currentWeatherFactor = state.targetWeatherFactor;
    applyAtmosphereUpdate(1.0);

    // Window Resize Handler
    window.addEventListener("resize", onWindowResize);
}

function onWindowResize() {
    const container = document.getElementById("city3dViewport");
    if (!container || !state.camera || !state.renderer) return;
    const width = container.clientWidth;
    const height = container.clientHeight;
    state.camera.aspect = width / height;
    state.camera.updateProjectionMatrix();
    state.renderer.setSize(width, height);
}


// =========================================================================
// GIS DATA LOADING
// =========================================================================

async function loadCityData() {
    // 1. Terrain DEM
    const terrainRes = await fetch("/api/city/terrain");
    state.terrain = await terrainRes.json();
    removeCanalFromTerrain();

    // 2. Roads
    const roadsRes = await fetch("/api/city/roads");
    const roadsGeo = await roadsRes.json();
    state.roads = roadsGeo.features || [];

    // 3. Critical Facilities
    const facRes = await fetch("/api/city/facilities");
    const facGeo = await facRes.json();
    state.facilities = facGeo.features || [];

    // 4. Building Footprints
    try {
        const fpRes = await fetch("/api/city/footprints");
        if (fpRes.ok) {
            const fpGeo = await fpRes.json();
            state.footprints = fpGeo.features || [];
        }
    } catch (e) {
        console.warn("Footprints not loaded:", e);
    }
}

// -------------------------------------------------------------------------
// CANAL REMOVAL: Inpaint & Smooth Terrain Across The Former River Channel
// -------------------------------------------------------------------------
function removeCanalFromTerrain() {
    if (!state.terrain || !state.terrain.river_mask) return;

    state.originalRiverMask = state.terrain.river_mask.map(row => [...row]);
    const rows = state.terrain.rows;
    const cols = state.terrain.cols;
    const grid = state.terrain.elevation_grid;

    // Create a 4-cell dilation corridor around the river cut to capture the channel and steep banks
    const inChannelOrBank = Array.from({ length: rows }, () => new Uint8Array(cols));
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            if (state.originalRiverMask[r][c]) {
                for (let dr = -4; dr <= 4; dr++) {
                    for (let dc = -4; dc <= 4; dc++) {
                        const nr = r + dr;
                        const nc = c + dc;
                        if (nr >= 1 && nr < rows - 1 && nc >= 1 && nc < cols - 1) {
                            if (dr * dr + dc * dc <= 18) {
                                inChannelOrBank[nr][nc] = 1;
                            }
                        }
                    }
                }
            }
        }
    }

    // Laplace relaxation: seamlessly inpaint elevation across the canal to match surrounding city streets
    for (let iter = 0; iter < 75; iter++) {
        for (let r = 1; r < rows - 1; r++) {
            for (let c = 1; c < cols - 1; c++) {
                if (inChannelOrBank[r][c]) {
                    grid[r][c] = (grid[r - 1][c] + grid[r + 1][c] + grid[r][c - 1] + grid[r][c + 1]) * 0.25;
                }
            }
        }
    }

    // Recompute global min/max elevation bounds & clear river mask
    let minE = Infinity, maxE = -Infinity;
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const v = grid[r][c];
            if (v < minE) minE = v;
            if (v > maxE) maxE = v;
            state.terrain.river_mask[r][c] = false; // Canal completely removed
        }
    }
    state.terrain.min_elevation_m = minE;
    state.terrain.max_elevation_m = maxE;
}


// =========================================================================
// 3D CITY ENVIRONMENT BUILDER
// =========================================================================

function build3DCityScene() {
    // 1. Realistic 3D Terrain Plane
    buildTerrainMesh();

    // 2. Dynamic 3D Water Surface
    buildWaterMesh();

    // 3. 3D Road Network with Lane Markings
    buildRoadNetwork3D();

    // 4. 3D Architectural Buildings (Extrusions with Window Glow)
    buildBuildings3D();

    // 5. 3D Critical Facility Monuments
    buildFacilities3D();

    // 6. Dynamic Storm Rain Particle FX
    buildRainWeatherFX();
}

function buildTerrainMesh() {
    const rows = state.terrain.rows;
    const cols = state.terrain.cols;

    const geom = new THREE.PlaneGeometry(600, 600, cols - 1, rows - 1);
    geom.rotateX(-Math.PI / 2); // Lay flat on X-Z plane

    const pos = geom.attributes.position;
    const colors = [];

    const minElev = state.terrain.min_elevation_m;
    const maxElev = state.terrain.max_elevation_m;

    for (let i = 0; i < pos.count; i++) {
        const c = i % cols;
        const r = Math.floor(i / cols);

        const elev = state.terrain.elevation_grid[r][c];

        // Vertical elevation scaling - smooth continuous natural terrain
        let y = (elev - minElev) * 0.55;
        pos.setY(i, y);

        // Hypsometric natural terrain vertex colors (seamless urban ground)
        const norm = (elev - minElev) / (maxElev - minElev);
        let cr, cg, cb;
        if (norm < 0.25) {
            // Lowland valley parkland
            cr = 0.13 + norm * 0.08; cg = 0.20 + norm * 0.10; cb = 0.16 + norm * 0.08;
        } else if (norm < 0.65) {
            // Mid-elevation urban asphalt & loam
            cr = 0.18 + norm * 0.10; cg = 0.22 + norm * 0.10; cb = 0.24 + norm * 0.10;
        } else {
            // Highland rock
            cr = 0.28; cg = 0.32; cb = 0.35;
        }
        colors.push(cr, cg, cb);
    }

    geom.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    geom.computeVertexNormals();

    const mat = new THREE.MeshStandardMaterial({
        vertexColors: true,
        roughness: 0.85,
        metalness: 0.15,
        flatShading: false
    });

    state.terrainMesh = new THREE.Mesh(geom, mat);
    state.terrainMesh.receiveShadow = true;
    state.scene.add(state.terrainMesh);
}

function buildWaterMesh() {
    const rows = state.terrain.rows;
    const cols = state.terrain.cols;

    const geom = new THREE.PlaneGeometry(600, 600, cols - 1, rows - 1);
    geom.rotateX(-Math.PI / 2);

    const pos = geom.attributes.position;
    const minElev = state.terrain.min_elevation_m;
    for (let i = 0; i < pos.count; i++) {
        const c = i % cols;
        const r = Math.floor(i / cols);
        const elev = state.terrain.elevation_grid[r][c];
        const baseElevY = (elev - minElev) * 0.55;
        pos.setY(i, baseElevY - 0.15); // Sits safely 0.15m beneath ground initially (no cliffs/chasms)
    }

    const mat = new THREE.MeshStandardMaterial({
        color: 0x0284c7,
        roughness: 0.12,
        metalness: 0.2,
        transparent: true,
        opacity: 0.82
    });

    state.waterMesh = new THREE.Mesh(geom, mat);
    state.waterMesh.receiveShadow = true;
    state.scene.add(state.waterMesh);
}

function buildRoadNetwork3D() {
    state.roadLinesGroup = new THREE.Group();
    state.barriersGroup = new THREE.Group();
    state.roadGraph = {};
    state.reciprocalEdges = {};

    // 1. Map directed (u, v) pairs to identify reciprocal opposing edges for U-turns & dual-lane traffic
    const edgeByUV = {};
    state.roads.forEach(feat => {
        const p = feat.properties;
        edgeByUV[`${p.u}_${p.v}`] = p.edge_id;
    });

    state.roads.forEach(feat => {
        const p = feat.properties;
        const oppId = edgeByUV[`${p.v}_${p.u}`];
        if (oppId) {
            state.reciprocalEdges[p.edge_id] = oppId;
        }
    });

    // 2. Track unique physical streets to render EXACTLY ONE 3D visual mesh per street (100% eliminating z-fighting & missing road sections)
    const builtStreets = new Set();

    state.roads.forEach(feat => {
        const coords = feat.geometry.coordinates;
        const p = feat.properties;
        const isHighway = (p.road_class === "highway");
        const isBridge = !!p.is_bridge;

        const p1 = worldTo3D(coords[0][0], coords[0][1]);
        const p2 = worldTo3D(coords[1][0], coords[1][1]);

        let y1 = getTerrainElevation(coords[0][0], coords[0][1]) + 0.38;
        let y2 = getTerrainElevation(coords[1][0], coords[1][1]) + 0.38;

        const startVec = new THREE.Vector3(p1.x, y1, p1.z);
        const endVec = new THREE.Vector3(p2.x, y2, p2.z);

        const length = startVec.distanceTo(endVec);
        const width = isHighway ? 4.6 : (p.road_class === "arterial" ? 3.4 : 2.6);
        const streetKey = p.u < p.v ? `${p.u}_${p.v}` : `${p.v}_${p.u}`;

        // Store directed edge in traffic graph
        state.roadGraph[p.edge_id] = {
            id: p.edge_id,
            start: startVec,
            end: endVec,
            startWorld: { x: coords[0][0], y: coords[0][1] },
            endWorld: { x: coords[1][0], y: coords[1][1] },
            u: p.u,
            v: p.v,
            streetKey: streetKey,
            length: length,
            speedLimit: p.free_flow_speed_kmh || 40.0,
            roadClass: p.road_class,
            width: width,
            isBridge: false,
            isClosed: false,
            waterDepth: 0.0
        };

        // Render EXACTLY ONE 3D mesh and barricade pair per physical street
        if (!builtStreets.has(streetKey)) {
            builtStreets.add(streetKey);

            const roadGroup = createRoadRibbon(coords[0], coords[1], p1, p2, width, isHighway ? 0x94a3b8 : 0x334155, false);
            roadGroup.userData = { 
                streetKey: streetKey,
                edgeIds: [p.edge_id], 
                baseColor: isHighway ? 0x94a3b8 : 0x334155, 
                width: width,
                asphalt: roadGroup.userData.asphalt,
                centerLine: roadGroup.userData.centerLine
            };
            state.roadLinesGroup.add(roadGroup);

            const barrierPair = createRoadBarricadePair(coords[0], coords[1], p1, p2, width, false);
            barrierPair.userData = { streetKey: streetKey, edgeIds: [p.edge_id] };
            barrierPair.visible = false;
            state.barriersGroup.add(barrierPair);
        } else {
            // Append reciprocal edge id to existing visual meshes
            const existingRoad = state.roadLinesGroup.children.find(m => m.userData && m.userData.streetKey === streetKey);
            if (existingRoad && existingRoad.userData.edgeIds) {
                existingRoad.userData.edgeIds.push(p.edge_id);
            }
            const existingBarrier = state.barriersGroup.children.find(b => b.userData && b.userData.streetKey === streetKey);
            if (existingBarrier && existingBarrier.userData.edgeIds) {
                existingBarrier.userData.edgeIds.push(p.edge_id);
            }
        }
    });

    state.scene.add(state.roadLinesGroup);
    state.scene.add(state.barriersGroup);
}

function createBridgeSupportPillars(p1, p2) {
    const group = new THREE.Group();
    const mid = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);
    const height = Math.max(2.0, mid.y + 1.0);

    const pillarGeom = new THREE.CylinderGeometry(0.8, 1.1, height, 8);
    const pillarMat = new THREE.MeshStandardMaterial({ color: 0x64748b, roughness: 0.8 });
    const pier = new THREE.Mesh(pillarGeom, pillarMat);
    pier.position.set(mid.x, mid.y / 2.0, mid.z);
    pier.castShadow = true;
    pier.receiveShadow = true;
    group.add(pier);
    return group;
}

function createRoadRibbon(coord1, coord2, p1, p2, width, colorHex, isBridge) {
    const group = new THREE.Group();
    const dx = p2.x - p1.x;
    const dz = p2.z - p1.z;
    const horizLen = Math.hypot(dx, dz);
    if (horizLen < 0.001) return group;

    const dir = new THREE.Vector3(dx, 0, dz).normalize();
    const up = new THREE.Vector3(0, 1, 0);
    const side = new THREE.Vector3().crossVectors(dir, up).normalize().multiplyScalar(width / 2.0);
    const cSide = side.clone().normalize().multiplyScalar(0.09); // yellow divider line half-width (0.18m total width)

    // Subdivide road into 2.5m elevation-sampled slices so the deck perfectly conforms to the terrain
    const numSegments = Math.max(3, Math.ceil(horizLen / 2.5));

    // Each quad has 2 triangles = 6 vertices. 3 floats per vertex (x, y, z)
    const asphaltVerts = new Float32Array(numSegments * 18);
    const centerVerts = new Float32Array(numSegments * 18);

    const pts = [];
    const cPts = [];
    const bridgeY = isBridge ? (Math.max(
        getTerrainElevation(coord1[0], coord1[1]),
        getTerrainElevation(coord2[0], coord2[1])
    ) + 2.5) : 0;

    for (let k = 0; k <= numSegments; k++) {
        const s = k / numSegments;
        const wx = coord1[0] * (1.0 - s) + coord2[0] * s;
        const wy = coord1[1] * (1.0 - s) + coord2[1] * s;
        const px = p1.x * (1.0 - s) + p2.x * s;
        const pz = p1.z * (1.0 - s) + p2.z * s;

        // Road sits strictly +0.38m above terrain elevation contours so it NEVER clips or sinks under ground
        const py = isBridge ? bridgeY : (getTerrainElevation(wx, wy) + 0.38);

        pts.push({
            lx: px - side.x, ly: py, lz: pz - side.z,
            rx: px + side.x, ry: py, rz: pz + side.z
        });

        cPts.push({
            lx: px - cSide.x, ly: py + 0.03, lz: pz - cSide.z,
            rx: px + cSide.x, ry: py + 0.03, rz: pz + cSide.z
        });
    }

    let aIdx = 0;
    let cIdx = 0;
    for (let k = 0; k < numSegments; k++) {
        const pA = pts[k];
        const pB = pts[k + 1];

        // Triangle 1: left[k], right[k], right[k+1]
        asphaltVerts[aIdx++] = pA.lx; asphaltVerts[aIdx++] = pA.ly; asphaltVerts[aIdx++] = pA.lz;
        asphaltVerts[aIdx++] = pA.rx; asphaltVerts[aIdx++] = pA.ry; asphaltVerts[aIdx++] = pA.rz;
        asphaltVerts[aIdx++] = pB.rx; asphaltVerts[aIdx++] = pB.ry; asphaltVerts[aIdx++] = pB.rz;

        // Triangle 2: left[k], right[k+1], left[k+1]
        asphaltVerts[aIdx++] = pA.lx; asphaltVerts[aIdx++] = pA.ly; asphaltVerts[aIdx++] = pA.lz;
        asphaltVerts[aIdx++] = pB.rx; asphaltVerts[aIdx++] = pB.ry; asphaltVerts[aIdx++] = pB.rz;
        asphaltVerts[aIdx++] = pB.lx; asphaltVerts[aIdx++] = pB.ly; asphaltVerts[aIdx++] = pB.lz;

        // Center divider line
        const cpA = cPts[k];
        const cpB = cPts[k + 1];

        centerVerts[cIdx++] = cpA.lx; centerVerts[cIdx++] = cpA.ly; centerVerts[cIdx++] = cpA.lz;
        centerVerts[cIdx++] = cpA.rx; centerVerts[cIdx++] = cpA.ry; centerVerts[cIdx++] = cpA.rz;
        centerVerts[cIdx++] = cpB.rx; centerVerts[cIdx++] = cpB.ry; centerVerts[cIdx++] = cpB.rz;

        centerVerts[cIdx++] = cpA.lx; centerVerts[cIdx++] = cpA.ly; centerVerts[cIdx++] = cpA.lz;
        centerVerts[cIdx++] = cpB.rx; centerVerts[cIdx++] = cpB.ry; centerVerts[cIdx++] = cpB.rz;
        centerVerts[cIdx++] = cpB.lx; centerVerts[cIdx++] = cpB.ly; centerVerts[cIdx++] = cpB.lz;
    }

    const asphaltGeom = new THREE.BufferGeometry();
    asphaltGeom.setAttribute("position", new THREE.BufferAttribute(asphaltVerts, 3));
    asphaltGeom.computeVertexNormals();

    const asphaltMat = new THREE.MeshStandardMaterial({
        color: colorHex,
        roughness: 0.75,
        metalness: 0.1,
        side: THREE.DoubleSide,
        polygonOffset: true,
        polygonOffsetFactor: -1.0,
        polygonOffsetUnits: -2.0
    });
    const asphaltMesh = new THREE.Mesh(asphaltGeom, asphaltMat);
    asphaltMesh.receiveShadow = true;
    group.add(asphaltMesh);

    const centerGeom = new THREE.BufferGeometry();
    centerGeom.setAttribute("position", new THREE.BufferAttribute(centerVerts, 3));
    centerGeom.computeVertexNormals();

    const centerMat = new THREE.MeshBasicMaterial({
        color: 0xf59e0b, // Yellow center divider line
        transparent: true,
        opacity: 0.95,
        side: THREE.DoubleSide,
        polygonOffset: true,
        polygonOffsetFactor: -2.0,
        polygonOffsetUnits: -4.0
    });
    const centerLine = new THREE.Mesh(centerGeom, centerMat);
    group.add(centerLine);

    group.userData = { asphalt: asphaltMesh, centerLine: centerLine };
    return group;
}

function createRoadBarricadePair(coord1, coord2, p1, p2, width, isBridge) {
    const group = new THREE.Group();
    const dx = p2.x - p1.x;
    const dz = p2.z - p1.z;
    const len = Math.hypot(dx, dz);
    if (len < 0.001) return group;

    const dir = new THREE.Vector3(dx, 0, dz).normalize();
    const bridgeY = isBridge ? (Math.max(
        getTerrainElevation(coord1[0], coord1[1]),
        getTerrainElevation(coord2[0], coord2[1])
    ) + 2.5) : 0;

    if (len < 12.0) {
        // Single barricade at midpoint for very short edges
        const s = 0.5;
        const wx = coord1[0] * (1.0 - s) + coord2[0] * s;
        const wy = coord1[1] * (1.0 - s) + coord2[1] * s;
        const px = p1.x * (1.0 - s) + p2.x * s;
        const pz = p1.z * (1.0 - s) + p2.z * s;
        const py = isBridge ? bridgeY : (getTerrainElevation(wx, wy) + 0.38);

        const b = createSingleBarricade(width, dir);
        b.position.set(px, py, pz);
        group.add(b);
    } else {
        // Dual barricades positioned at both entrances of the road segment
        const offset = Math.min(3.5, len * 0.12);
        const s1 = offset / len;
        const s2 = 1.0 - s1;

        // Entrance 1
        const wx1 = coord1[0] * (1.0 - s1) + coord2[0] * s1;
        const wy1 = coord1[1] * (1.0 - s1) + coord2[1] * s1;
        const px1 = p1.x * (1.0 - s1) + p2.x * s1;
        const pz1 = p1.z * (1.0 - s1) + p2.z * s1;
        const py1 = isBridge ? bridgeY : (getTerrainElevation(wx1, wy1) + 0.38);

        const b1 = createSingleBarricade(width, dir);
        b1.position.set(px1, py1, pz1);
        group.add(b1);

        // Entrance 2
        const wx2 = coord1[0] * (1.0 - s2) + coord2[0] * s2;
        const wy2 = coord1[1] * (1.0 - s2) + coord2[1] * s2;
        const px2 = p1.x * (1.0 - s2) + p2.x * s2;
        const pz2 = p1.z * (1.0 - s2) + p2.z * s2;
        const py2 = isBridge ? bridgeY : (getTerrainElevation(wx2, wy2) + 0.38);

        const b2 = createSingleBarricade(width, dir);
        b2.position.set(px2, py2, pz2);
        group.add(b2);
    }

    return group;
}

function createSingleBarricade(roadWidth, dir) {
    const bGroup = new THREE.Group();
    const barWidth = Math.max(2.8, roadWidth * 1.15);

    // Upper reflective orange hazard board
    const boardGeom = new THREE.BoxGeometry(barWidth, 0.4, 0.2);
    const boardMat1 = new THREE.MeshStandardMaterial({
        color: 0xf97316,
        roughness: 0.3,
        emissive: 0xc2410c,
        emissiveIntensity: 0.5
    });
    const board1 = new THREE.Mesh(boardGeom, boardMat1);
    board1.position.y = 1.35;
    board1.castShadow = true;
    bGroup.add(board1);

    // Lower reflective white hazard board
    const boardMat2 = new THREE.MeshStandardMaterial({
        color: 0xffffff,
        roughness: 0.3,
        emissive: 0x334155,
        emissiveIntensity: 0.2
    });
    const board2 = new THREE.Mesh(boardGeom, boardMat2);
    board2.position.y = 0.75;
    board2.castShadow = true;
    bGroup.add(board2);

    // Vertical black metal support posts
    const postGeom = new THREE.CylinderGeometry(0.08, 0.08, 1.6, 8);
    const postMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.5 });
    
    const postLeft = new THREE.Mesh(postGeom, postMat);
    postLeft.position.set(-barWidth * 0.42, 0.8, 0);
    bGroup.add(postLeft);

    const postRight = new THREE.Mesh(postGeom, postMat);
    postRight.position.set(barWidth * 0.42, 0.8, 0);
    bGroup.add(postRight);

    // Dual flashing emergency strobe beacon lights on top of each barricade
    const beaconGeom = new THREE.SphereGeometry(0.18, 8, 8);
    const beaconMat1 = new THREE.MeshBasicMaterial({ color: 0xef4444 });
    const beaconMat2 = new THREE.MeshBasicMaterial({ color: 0xf59e0b });
    
    const beacon1 = new THREE.Mesh(beaconGeom, beaconMat1);
    beacon1.position.set(-barWidth * 0.38, 1.7, 0);
    bGroup.add(beacon1);

    const beacon2 = new THREE.Mesh(beaconGeom, beaconMat2);
    beacon2.position.set(barWidth * 0.38, 1.7, 0);
    bGroup.add(beacon2);

    bGroup.userData.beacons = [beacon1, beacon2];

    // Align barricade perpendicular to road travel direction
    bGroup.rotation.y = Math.atan2(dir.x, dir.z);

    return bGroup;
}

function buildBuildings3D() {
    state.buildingsGroup = new THREE.Group();
    state.aviationBeacons = [];

    // Initialize Procedural Facade Textures & Materials
    const commTex = createCommercialGlassTexture();
    const commEmissiveTex = createCommercialEmissiveTexture();
    const resTex = createResidentialTexture();
    const resEmissiveTex = createResidentialEmissiveTexture();
    const indTex = createIndustrialTexture();
    const roofTex = createRoofTexture();
    const helipadTex = createHelipadTexture();

    // Wall & Roof Materials
    const commWallMat = new THREE.MeshStandardMaterial({
        map: commTex,
        emissiveMap: commEmissiveTex,
        emissive: new THREE.Color(0xffffff),
        emissiveIntensity: 0.25,
        roughness: 0.25,
        metalness: 0.65
    });
    commWallMat.userData = { baseEmissive: 0.25 };

    const resWallMat = new THREE.MeshStandardMaterial({
        map: resTex,
        emissiveMap: resEmissiveTex,
        emissive: new THREE.Color(0xffffff),
        emissiveIntensity: 0.18,
        roughness: 0.55,
        metalness: 0.20
    });
    resWallMat.userData = { baseEmissive: 0.18 };

    const indWallMat = new THREE.MeshStandardMaterial({
        map: indTex,
        roughness: 0.65,
        metalness: 0.40
    });

    const roofMat = new THREE.MeshStandardMaterial({
        map: roofTex,
        roughness: 0.85,
        metalness: 0.10
    });

    const helipadMat = new THREE.MeshStandardMaterial({
        map: helipadTex,
        roughness: 0.45,
        metalness: 0.20
    });

    const commMaterials = [commWallMat, commWallMat, roofMat, roofMat, commWallMat, commWallMat];
    const resMaterials = [resWallMat, resWallMat, roofMat, roofMat, resWallMat, resWallMat];
    const indMaterials = [indWallMat, indWallMat, roofMat, roofMat, indWallMat, indWallMat];

    state.footprints.forEach(bldg => {
        const p = bldg.properties;
        const center = p.center;

        const p3d = worldTo3D(center[0], center[1]);
        const bw = Math.max(3.2, (p.width_m || 25.0) * SCALE);
        const bd = Math.max(3.2, (p.depth_m || 25.0) * SCALE);
        const bh = Math.max(4.5, (p.height_m || 30.0) * 0.45);
        const groundY = (p.ground_elev_m - state.terrain.min_elevation_m) * 0.55;

        const bldgGroup = new THREE.Group();
        bldgGroup.position.set(p3d.x, 0, p3d.z);

        const isCommercial = (p.zoning === "commercial_high" || p.zoning === "commercial_medium");
        const isResidential = (p.zoning && p.zoning.startsWith("residential"));
        const bldgMats = isCommercial ? commMaterials : (isResidential ? resMaterials : indMaterials);

        const isSkyscraper = (p.height_m >= 45.0 || (p.zoning === "commercial_high" && p.height_m >= 35.0));

        if (isSkyscraper) {
            // MULTI-TIER STEPPED SETBACK SKYSCRAPER ARCHITECTURE
            const h1 = bh * 0.22; // Base podium tier
            const h2 = bh * 0.58; // Main tower shaft
            const h3 = bh * 0.20; // Crown / penthouse setback

            const w1 = bw, d1 = bd;
            const w2 = bw * 0.88, d2 = bd * 0.88;
            const w3 = bw * 0.72, d3 = bd * 0.72;

            // Tier 1: Base Podium
            const g1 = new THREE.BoxGeometry(w1, h1, d1);
            applyBoxFaceUVs(g1, w1, h1, d1, 5.0, 3.6);
            const m1 = new THREE.Mesh(g1, bldgMats);
            m1.position.y = groundY + h1 / 2.0;
            m1.castShadow = true;
            m1.receiveShadow = true;
            bldgGroup.add(m1);

            // Tier 2: Main Shaft
            const g2 = new THREE.BoxGeometry(w2, h2, d2);
            applyBoxFaceUVs(g2, w2, h2, d2, 5.0, 3.6);
            const m2 = new THREE.Mesh(g2, bldgMats);
            m2.position.y = groundY + h1 + h2 / 2.0;
            m2.castShadow = true;
            m2.receiveShadow = true;
            bldgGroup.add(m2);

            // Tier 3: Crown Setback
            const g3 = new THREE.BoxGeometry(w3, h3, d3);
            applyBoxFaceUVs(g3, w3, h3, d3, 5.0, 3.6);
            const m3 = new THREE.Mesh(g3, bldgMats);
            m3.position.y = groundY + h1 + h2 + h3 / 2.0;
            m3.castShadow = true;
            m3.receiveShadow = true;
            bldgGroup.add(m3);

            const roofTopY = groundY + bh;

            // Rooftop Parapet Border Wall
            const parapetH = 0.55;
            const parapetGeom = new THREE.BoxGeometry(w3 * 0.98, parapetH, d3 * 0.98);
            const parapetMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.6 });
            const parapet = new THREE.Mesh(parapetGeom, parapetMat);
            parapet.position.y = roofTopY + parapetH / 2.0;
            bldgGroup.add(parapet);

            // Rooftop HVAC Chiller Unit
            const hvacW = Math.max(1.8, w3 * 0.28);
            const hvacD = Math.max(1.8, d3 * 0.28);
            const hvacGeom = new THREE.BoxGeometry(hvacW, 1.3, hvacD);
            const hvacMat = new THREE.MeshStandardMaterial({ color: 0x475569, metalness: 0.7, roughness: 0.3 });
            const hvac = new THREE.Mesh(hvacGeom, hvacMat);
            hvac.position.set(-w3 * 0.2, roofTopY + 0.65, -d3 * 0.2);
            bldgGroup.add(hvac);

            // Helipad on high-rises (height >= 75m)
            if (p.height_m >= 75.0 && w3 >= 6.0) {
                const padRadius = Math.min(w3 * 0.42, 6.5);
                const padGeom = new THREE.CylinderGeometry(padRadius, padRadius, 0.25, 24);
                const padSideMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.7 });
                const padMats = [padSideMat, helipadMat, padSideMat];
                const pad = new THREE.Mesh(padGeom, padMats);
                pad.position.set(0, roofTopY + 0.15, 0);
                bldgGroup.add(pad);
            }

            // Communications Spire & FAA Warning Beacon on top tallest towers (height >= 80m)
            if (p.height_m >= 80.0) {
                const spireH = 8.0 + (p.height_m - 80.0) * 0.25;
                const spireGeom = new THREE.CylinderGeometry(0.12, 0.45, spireH, 8);
                const spireMat = new THREE.MeshStandardMaterial({ color: 0xd1d5db, metalness: 0.85, roughness: 0.2 });
                const spire = new THREE.Mesh(spireGeom, spireMat);
                spire.position.set(w3 * 0.18, roofTopY + spireH / 2.0, d3 * 0.18);
                bldgGroup.add(spire);

                // FAA Obstruction Red Warning Beacon Light
                const beaconGeom = new THREE.SphereGeometry(0.38, 8, 8);
                const beaconMat = new THREE.MeshBasicMaterial({ color: 0xff0000 });
                const beacon = new THREE.Mesh(beaconGeom, beaconMat);
                beacon.position.set(w3 * 0.18, roofTopY + spireH + 0.2, d3 * 0.18);
                bldgGroup.add(beacon);
                state.aviationBeacons.push(beacon);
            }

        } else if (isResidential) {
            // RESIDENTIAL MULTI-FAMILY / APARTMENT HOUSING
            const g = new THREE.BoxGeometry(bw, bh, bd);
            applyBoxFaceUVs(g, bw, bh, bd, 4.5, 3.2);
            const m = new THREE.Mesh(g, bldgMats);
            m.position.y = groundY + bh / 2.0;
            m.castShadow = true;
            m.receiveShadow = true;
            bldgGroup.add(m);

            const roofTopY = groundY + bh;

            // Rooftop Stairwell / Penthouse Hatch
            const phGeom = new THREE.BoxGeometry(bw * 0.35, 1.8, bd * 0.35);
            const phMat = new THREE.MeshStandardMaterial({ color: 0x334155, roughness: 0.7 });
            const ph = new THREE.Mesh(phGeom, phMat);
            ph.position.set(bw * 0.2, roofTopY + 0.9, -bd * 0.2);
            bldgGroup.add(ph);

            // Classic Rooftop Cylindrical Water Tank on Stilts
            const tankRadius = Math.max(0.8, Math.min(1.4, bw * 0.18));
            const tankH = 2.2;
            const tankGeom = new THREE.CylinderGeometry(tankRadius, tankRadius, tankH, 12);
            const tankMat = new THREE.MeshStandardMaterial({ color: 0x78350f, roughness: 0.8 });
            const tank = new THREE.Mesh(tankGeom, tankMat);
            tank.position.set(-bw * 0.25, roofTopY + 1.2 + tankH / 2.0, bd * 0.2);
            bldgGroup.add(tank);

            // Conical Roof Cap on Tank
            const capGeom = new THREE.ConeGeometry(tankRadius * 1.1, 0.9, 12);
            const capMat = new THREE.MeshStandardMaterial({ color: 0x451a03, roughness: 0.6 });
            const cap = new THREE.Mesh(capGeom, capMat);
            cap.position.set(-bw * 0.25, roofTopY + 1.2 + tankH + 0.45, bd * 0.2);
            bldgGroup.add(cap);

        } else {
            // INDUSTRIAL WAREHOUSES & FACILITIES
            const g = new THREE.BoxGeometry(bw, bh, bd);
            applyBoxFaceUVs(g, bw, bh, bd, 6.0, 4.0);
            const m = new THREE.Mesh(g, bldgMats);
            m.position.y = groundY + bh / 2.0;
            m.castShadow = true;
            m.receiveShadow = true;
            bldgGroup.add(m);

            const roofTopY = groundY + bh;

            // Rooftop Industrial Ventilation Monitor / Exhaust Ridge
            const ventGeom = new THREE.BoxGeometry(bw * 0.7, 1.2, bd * 0.25);
            const ventMat = new THREE.MeshStandardMaterial({ color: 0x334155, metalness: 0.6, roughness: 0.4 });
            const vent = new THREE.Mesh(ventGeom, ventMat);
            vent.position.set(0, roofTopY + 0.6, 0);
            bldgGroup.add(vent);
        }

        bldgGroup.userData = { id: p.id, district: p.district, zoning: p.zoning, height: p.height_m, baseEmissive: 0.25 };
        state.buildingsGroup.add(bldgGroup);
    });

    state.scene.add(state.buildingsGroup);
}

function buildFacilities3D() {
    state.facilitiesGroup = new THREE.Group();

    state.facilities.forEach(fac => {
        const p = fac.properties;
        const [fx, fy] = fac.geometry.coordinates;
        const p3d = worldTo3D(fx, fy);
        const gy = getTerrainElevation(fx, fy);

        // Facility Base
        let color = 0x00f0ff;
        if (p.category === "hospitals") color = 0xef4444;
        else if (p.category === "pumping_stations") color = 0x10b981;
        else if (p.category === "ems_stations") color = 0xf59e0b;

        const geom = new THREE.CylinderGeometry(3.5, 4.0, 4.0, 16);
        const mat = new THREE.MeshStandardMaterial({ color: color, roughness: 0.3, emissive: color, emissiveIntensity: 0.4 });
        const base = new THREE.Mesh(geom, mat);
        base.position.set(p3d.x, gy + 2.0, p3d.z);

        // Floating holographic marker badge above
        const badgeGeom = new THREE.SphereGeometry(1.5, 12, 12);
        const badgeMat = new THREE.MeshBasicMaterial({ color: color });
        const badge = new THREE.Mesh(badgeGeom, badgeMat);
        badge.position.set(p3d.x, gy + 9.0, p3d.z);

        base.userData = { id: p.name, category: p.category, badge: badge };
        state.facilitiesGroup.add(base);
        state.facilitiesGroup.add(badge);
    });

    state.scene.add(state.facilitiesGroup);
}

function buildRainWeatherFX() {
    const rainCount = 2400;
    const geom = new THREE.BufferGeometry();
    const positions = new Float32Array(rainCount * 3);

    state.rainData = [];
    for (let i = 0; i < rainCount; i++) {
        const x = (Math.random() - 0.5) * 620;
        const y = Math.random() * 260 + 5;
        const z = (Math.random() - 0.5) * 620;
        const speed = 160 + Math.random() * 80;
        state.rainData.push({ x, y, z, speed });

        positions[i * 3] = x;
        positions[i * 3 + 1] = y;
        positions[i * 3 + 2] = z;
    }

    geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));

    const mat = new THREE.PointsMaterial({
        color: 0xbae6fd,
        size: 1.8,
        map: createRainDropTexture(),
        transparent: true,
        opacity: 0.0,
        depthWrite: false
    });

    state.rainSystem = new THREE.Points(geom, mat);
    state.scene.add(state.rainSystem);
}


// =========================================================================
// MOVING TRAFFIC SIMULATION (Cars Running, Braking, Diverting, Resuming)
// =========================================================================

function spawnTrafficFleet(count = 180) {
    state.trafficGroup = new THREE.Group();
    state.vehicles = [];

    // Filter available directed edges to only open and dry roads (no flooded roads or water channels)
    const openDryEdges = Object.values(state.roadGraph).filter(e => !e.isClosed && (e.waterDepth || 0.0) < 0.05);
    const candidateEdges = openDryEdges.length > 0 ? openDryEdges : Object.values(state.roadGraph);
    if (candidateEdges.length === 0) return;

    const carColors = [0xf59e0b, 0xffffff, 0x38bdf8, 0xe2e8f0, 0x94a3b8, 0xef4444, 0x10b981];

    // Spatial occupancy map per directed edge to prevent initial spawn overlap
    const edgeOccupancy = new Map();

    let spawned = 0;
    let attempts = 0;
    while (spawned < count && attempts < count * 8) {
        attempts++;
        const edge = candidateEdges[Math.floor(Math.random() * candidateEdges.length)];

        if (!edgeOccupancy.has(edge.id)) {
            edgeOccupancy.set(edge.id, []);
        }
        const existingTs = edgeOccupancy.get(edge.id);

        // Pick a candidate position along directed edge with safe buffer spacing
        const candidateT = 0.12 + Math.random() * 0.76;
        const minGapT = 4.8 / Math.max(10.0, edge.length);
        const hasOverlap = existingTs.some(t => Math.abs(t - candidateT) < minGapT);

        // At most 2 cars per directed lane to ensure free-flowing traffic
        if (hasOverlap || existingTs.length >= 2) {
            continue;
        }

        existingTs.push(candidateT);

        const isEmergency = (Math.random() < 0.08); // 8% Ambulances / Fire
        const color = isEmergency ? 0xef4444 : carColors[Math.floor(Math.random() * carColors.length)];

        // 3D Car Body Mesh (compact scale to fit lane cleanly)
        const carMesh = createCarMesh(color, isEmergency);
        state.trafficGroup.add(carMesh);

        state.vehicles.push({
            id: spawned,
            mesh: carMesh,
            edgeId: edge.id,
            t: candidateT,
            baseSpeed: (13.0 + Math.random() * 8.0) * (isEmergency ? 1.4 : 1.0),
            currentSpeed: 13.0,
            isEmergency: isEmergency,
            isStalled: false,
            waitingTimer: 0
        });

        spawned++;
    }

    state.scene.add(state.trafficGroup);
}

function createCarMesh(colorHex, isEmergency) {
    const group = new THREE.Group();

    // Body (width: 1.25, height: 0.55, length: 2.6 - fits neatly into designated lane)
    const bodyGeom = new THREE.BoxGeometry(1.25, 0.55, 2.6);
    const bodyMat = new THREE.MeshStandardMaterial({ color: colorHex, roughness: 0.4, metalness: 0.3 });
    const body = new THREE.Mesh(bodyGeom, bodyMat);
    body.position.y = 0.45;
    body.castShadow = true;
    group.add(body);

    // Windshield Cabin
    const cabinGeom = new THREE.BoxGeometry(1.05, 0.45, 1.35);
    const cabinMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.1 });
    const cabin = new THREE.Mesh(cabinGeom, cabinMat);
    cabin.position.set(0, 0.82, -0.15);
    group.add(cabin);

    // Headlights
    const lightGeom = new THREE.BoxGeometry(0.24, 0.12, 0.08);
    const lightMat = new THREE.MeshBasicMaterial({ color: 0xffedd5 });
    const hl1 = new THREE.Mesh(lightGeom, lightMat);
    hl1.position.set(-0.42, 0.45, 1.3);
    const hl2 = hl1.clone();
    hl2.position.set(0.42, 0.45, 1.3);
    group.add(hl1);
    group.add(hl2);

    // Brake / Tail lights
    const brakeMat = new THREE.MeshBasicMaterial({ color: 0xef4444 });
    const bl1 = new THREE.Mesh(lightGeom, brakeMat);
    bl1.position.set(-0.42, 0.45, -1.3);
    const bl2 = bl1.clone();
    bl2.position.set(0.42, 0.45, -1.3);
    group.add(bl1);
    group.add(bl2);
    group.userData.tailLights = bl1;

    // Emergency Siren
    if (isEmergency) {
        const sirenGeom = new THREE.CylinderGeometry(0.18, 0.18, 0.25, 8);
        const sirenMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8 });
        const siren = new THREE.Mesh(sirenGeom, sirenMat);
        siren.position.set(0, 1.15, -0.15);
        group.add(siren);
        group.userData.siren = siren;
    }

    return group;
}

function updateTraffic(dt) {
    if (!state.trafficGroup || state.vehicles.length === 0) return;

    // 1. Group vehicles by directed edge (edgeId) for car-following & anti-overlap physics
    const laneGroups = new Map();
    state.vehicles.forEach(veh => {
        if (!laneGroups.has(veh.edgeId)) laneGroups.set(veh.edgeId, []);
        laneGroups.get(veh.edgeId).push(veh);
    });

    // 2. Intra-Lane Car-Following Model (Intelligent Driver Model / Anti-Tailgating)
    laneGroups.forEach((cars, edgeId) => {
        if (cars.length <= 1) return;
        const edge = state.roadGraph[edgeId];
        if (!edge) return;

        // Sort cars along travel direction: largest t is in front (0 -> 1)
        cars.sort((a, b) => a.t - b.t);

        // Apply car-following headway constraint from rear to front
        for (let i = 0; i < cars.length - 1; i++) {
            const follower = cars[i];
            const leader = cars[i + 1];

            const headwayDist = (leader.t - follower.t) * edge.length;
            const safeGap = 5.0;       // Buffer to start matching speed (meters in 3D)
            const stopGap = 3.4;       // Bumper-to-bumper stopping distance (strictly no overlap)

            if (headwayDist < safeGap) {
                // Decelerate trailing vehicle to match leader
                const speedFactor = Math.max(0.0, (headwayDist - stopGap) / (safeGap - stopGap));
                follower.currentSpeed = Math.min(follower.currentSpeed, leader.currentSpeed * speedFactor);

                // Hard anti-overlap distance enforcement
                if (headwayDist < stopGap) {
                    follower.currentSpeed = 0.0;
                    const minGapT = stopGap / Math.max(10.0, edge.length);
                    follower.t = Math.min(follower.t, leader.t - minGapT);
                }
            }
        }
    });

    // 3. Update Individual Vehicle Kinematics, Flooding Reactions, and 3D Transformations
    state.vehicles.forEach(veh => {
        let edge = state.roadGraph[veh.edgeId];
        if (!edge) return;

        const water = edge.waterDepth || 0.0;
        const isRoadClosed = edge.isClosed || (water >= 0.15);

        // Hydrodynamic Passability & Hazard Reaction
        if (isRoadClosed) {
            if (!veh.isEmergency) {
                // Approaching the barricade (t >= 0.78) or facing flood water:
                const approachingBarricade = (veh.t >= 0.78);
                const facingWater = (water >= 0.15);

                if (approachingBarricade || facingWater) {
                    // Decelerate smoothly to a stop before the barricade
                    veh.currentSpeed = Math.max(0.0, veh.currentSpeed - 14.0 * dt);
                    veh.waitingTimer += dt;

                    // Execute smooth U-turn into reciprocal opposing lane to retreat back towards dry intersection
                    if (veh.currentSpeed <= 0.3 || veh.waitingTimer > 0.4) {
                        const returnEdgeId = state.reciprocalEdges[veh.edgeId];
                        const returnEdge = returnEdgeId ? state.roadGraph[returnEdgeId] : null;

                        if (returnEdge) {
                            // Switch to opposing lane: physical position is exactly (1.0 - t)!
                            veh.edgeId = returnEdge.id;
                            veh.t = npClamp(1.0 - veh.t, 0.05, 0.95);
                            veh.currentSpeed = veh.baseSpeed * 0.55;
                            veh.isStalled = false;
                            veh.waitingTimer = 0;
                            edge = returnEdge; // update reference for step 4
                        } else {
                            veh.isStalled = true;
                        }
                    } else {
                        veh.isStalled = true;
                    }
                } else {
                    // Traveling back along return lane away from closure towards open intersection
                    veh.currentSpeed = Math.min(veh.baseSpeed * 0.70, veh.currentSpeed + 7.0 * dt);
                    veh.isStalled = false;
                    veh.waitingTimer = 0;
                }

                // If completely stranded in deep flood (>0.30m), hazard lights flash
                if (water >= 0.30) {
                    veh.currentSpeed = 0.0;
                    veh.isStalled = true;
                }

                // Flash hazard lights while stalled
                if (veh.isStalled && veh.mesh.userData.tailLights) {
                    const blink = Math.floor(Date.now() / 320) % 2 === 0;
                    veh.mesh.userData.tailLights.material.color.setHex(blink ? 0xf59e0b : 0xef4444);
                }
            } else {
                // Emergency Ambulance can navigate moderate flooding (<0.30m)
                if (water >= 0.30) {
                    veh.currentSpeed = 0.0;
                    veh.isStalled = true;
                } else {
                    veh.currentSpeed = veh.baseSpeed * 0.40;
                    veh.isStalled = false;
                }
            }
        } else if (water >= 0.05) {
            // Shallow Ponding: Speed degraded (Pregnolato curve)
            veh.currentSpeed = Math.min(veh.currentSpeed, veh.baseSpeed * 0.50);
            veh.isStalled = false;
            veh.waitingTimer = 0;
        } else {
            // Open Dry Road: Smoothly regain free flow speed
            if (veh.currentSpeed < veh.baseSpeed * 0.95) {
                veh.currentSpeed = Math.min(veh.baseSpeed, veh.currentSpeed + 8.0 * dt);
            }
            veh.isStalled = false;
            veh.waitingTimer = 0;
        }

        // Kinematic Position Step forward along edge (0 -> 1)
        const stepDist = (veh.currentSpeed * dt) / Math.max(12.0, edge.length);
        veh.t += stepDist;

        // Approaching road intersection (edge endpoint: t >= 1.0)
        if (veh.t >= 1.0) {
            const hitNode = edge.v;
            const oppEdgeId = state.reciprocalEdges[edge.id];

            // Find outgoing edges from this intersection that are open and dry
            const availableOutgoing = Object.values(state.roadGraph).filter(e => 
                e.u === hitNode && 
                e.id !== oppEdgeId && 
                !e.isClosed && 
                (e.waterDepth || 0.0) < 0.05
            );

            if (availableOutgoing.length > 0) {
                const nextEdge = availableOutgoing[Math.floor(Math.random() * availableOutgoing.length)];
                veh.edgeId = nextEdge.id;
                veh.t = 0.02; // Seamless continuous handoff at intersection
                veh.currentSpeed = veh.baseSpeed * 0.75;
                veh.isStalled = false;
                veh.waitingTimer = 0;
                edge = nextEdge; // update reference for step 4
            } else {
                // Dead end or surrounded by barricades: U-turn back onto return edge
                const returnEdgeId = state.reciprocalEdges[edge.id];
                if (returnEdgeId) {
                    veh.edgeId = returnEdgeId;
                    veh.t = 0.05;
                    veh.currentSpeed = veh.baseSpeed * 0.5;
                    veh.isStalled = false;
                    edge = state.roadGraph[returnEdgeId];
                } else {
                    veh.currentSpeed = 0.0;
                    veh.isStalled = true;
                    veh.t = 0.98;
                }
            }
        }

        // Clamp position along edge
        veh.t = npClamp(veh.t, 0.0, 1.0);

        // 4. Compute 3D Position with Right-Hand Traffic Lane Offset & Exact Terrain Elevation
        const dx = edge.end.x - edge.start.x;
        const dz = edge.end.z - edge.start.z;
        const forward = new THREE.Vector3(dx, 0, dz).normalize();
        const up = new THREE.Vector3(0, 1, 0);
        const right = new THREE.Vector3().crossVectors(forward, up).normalize();

        // Driving on the right-hand lane (offset along +right):
        const laneOffsetDist = edge.width * 0.26;
        const lateralShift = right.clone().multiplyScalar(laneOffsetDist);

        const curWx = edge.startWorld.x * (1.0 - veh.t) + edge.endWorld.x * veh.t;
        const curWy = edge.startWorld.y * (1.0 - veh.t) + edge.endWorld.y * veh.t;

        const posX = (edge.start.x * (1.0 - veh.t) + edge.end.x * veh.t) + lateralShift.x;
        const posZ = (edge.start.z * (1.0 - veh.t) + edge.end.z * veh.t) + lateralShift.z;

        let roadElevY;
        if (edge.isBridge) {
            roadElevY = edge.start.y;
        } else {
            roadElevY = getTerrainElevation(curWx, curWy) + 0.38;
        }

        const finalPos = new THREE.Vector3(posX, roadElevY + 0.35, posZ);
        veh.mesh.position.copy(finalPos);

        // Orient vehicle along lane direction, pitching with road slope
        const lookT = npClamp(veh.t + 0.04, 0.0, 1.0);
        const lookWx = edge.startWorld.x * (1.0 - lookT) + edge.endWorld.x * lookT;
        const lookWy = edge.startWorld.y * (1.0 - lookT) + edge.endWorld.y * lookT;
        const lookElevY = edge.isBridge ? edge.start.y : (getTerrainElevation(lookWx, lookWy) + 0.38);

        const lookPosX = (edge.start.x * (1.0 - lookT) + edge.end.x * lookT) + lateralShift.x;
        const lookPosZ = (edge.start.z * (1.0 - lookT) + edge.end.z * lookT) + lateralShift.z;

        const lookTarget = new THREE.Vector3(lookPosX, lookElevY + 0.35, lookPosZ);
        if (lookTarget.distanceTo(finalPos) < 0.001) {
            lookTarget.add(forward.clone().multiplyScalar(0.5));
        }
        veh.mesh.lookAt(lookTarget);

        // Flashing emergency siren
        if (veh.isEmergency && veh.mesh.userData.siren) {
            const isRed = Math.floor(Date.now() / 150) % 2 === 0;
            veh.mesh.userData.siren.material.color.setHex(isRed ? 0xef4444 : 0x00f0ff);
        }
    });
}


// =========================================================================
// FAST TRAJECTORY LOADING & TIMELINE SCRUBBER
// =========================================================================

async function loadCachedFrames() {
    try {
        const res = await fetch(`/api/simulation/frames_all?branch_id=${state.currentBranch}`);
        if (res.ok) {
            const data = await res.json();
            if (data.frames && data.frames.length > 0) {
                state.frames = data.frames;
                state.totalSteps = state.frames.length - 1;
                elTimelineScrubber.max = state.totalSteps;

                if (data.summary) {
                    state.historyFloodedArea = data.summary.flooded_area_km2_series;
                    state.historyEmsTime = data.summary.ems_time_min_series;
                    updateSparklines();
                }

                // Immediately scrub to storm peak (T+02:00) so user sees active 3D flooding & traffic
                const previewStep = Math.min(24, state.totalSteps);
                scrubToStep(previewStep);
                return true;
            }
        }
    } catch (e) {
        console.warn("Fast trajectory loading notice:", e);
    }
    return false;
}

async function runScenario(autoPlay = true) {
    const rainfall = parseFloat(sliderRain.value);
    const duration = parseFloat(sliderDuration.value);
    const drainage = parseFloat(sliderDrain.value) / 100.0;
    const p1 = parseFloat(sliderPump1.value) / 100.0;
    const p2 = parseFloat(sliderPump2.value) / 100.0;

    try {
        elRunScenario.disabled = true;
        document.getElementById("runScenarioLabel").innerText = "SIMULATING (2D Physics)...";

        const payload = {
            rainfall_mm_hr: rainfall,
            duration_hours: duration,
            drainage_capacity_factor: drainage,
            pump_overrides: { "PUMP-1": p1, "PUMP-2": p2 },
            engine_mode: state.engineMode
        };

        const res = await fetch("/api/simulation/run_scenario", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error("Scenario rollout failed");
        const summary = await res.json();

        state.totalSteps = summary.total_frames - 1;
        elTimelineScrubber.max = state.totalSteps;
        elTimelineTotalLabel.innerText = `T+${formatHoursMinutes(duration * 3600)}`;

        // Batch fetch all frames in 1 single call
        const fRes = await fetch(`/api/simulation/frames_all?branch_id=${state.currentBranch}`);
        if (fRes.ok) {
            const data = await fRes.json();
            state.frames = data.frames;
        }

        // Update sparkline series
        state.historyFloodedArea = summary.flooded_area_km2_series;
        state.historyEmsTime = summary.ems_time_min_series;
        updateSparklines();

        // Start playback from T+00:00 to T+03:00
        scrubToStep(0);
        if (autoPlay) {
            startPlayback();
        }
    } catch (err) {
        console.error("Run scenario error:", err);
    } finally {
        elRunScenario.disabled = false;
        document.getElementById("runScenarioLabel").innerText = "RUN SCENARIO (3 HR)";
    }
}

function scrubToStep(stepIdx) {
    stepIdx = Math.max(0, Math.min(stepIdx, state.totalSteps));
    state.currentStepIdx = stepIdx;
    elTimelineScrubber.value = stepIdx;

    const frame = state.frames[stepIdx];
    if (!frame) return;

    state.currentFrame = frame;

    const timeSec = frame.time_s;
    elSimTime.innerText = `T+${formatTime(timeSec)}`;
    elTimelineTimeLabel.innerText = `T+${formatHoursMinutes(timeSec)}`;

    // 1. Physically Raise/Lower 3D Water Surface
    update3DWaterSurface(frame.surface_depth_grid);

    // 2. Update 3D Road Colors & Roadblock Barricades
    update3DRoadConditions(frame.surface_depth_grid);

    // 3. Update Facility Operational Status & Blackout FX
    update3DFacilitiesStatus(frame.surface_depth_grid, frame.snapshot);

    // 4. Update Dynamic Atmospheric Weather & Sky based on current rainfall
    if (state.weatherMode === "auto" && frame.snapshot) {
        const rain = frame.snapshot.rainfall_rate_mm_hr !== undefined 
            ? frame.snapshot.rainfall_rate_mm_hr 
            : parseFloat(sliderRain.value);
        state.targetWeatherFactor = npClamp(rain / 130.0, 0.0, 1.0);
    }

    // 5. Update KPI Telemetry Cards
    updateTelemetry(frame.snapshot);
}


// =========================================================================
// 3D DYNAMIC WATER SURFACE & ROADS ELEVATION
// =========================================================================

function update3DWaterSurface(depthGrid) {
    if (!state.waterMesh || !depthGrid) return;

    const pos = state.waterMesh.geometry.attributes.position;
    const rows = state.terrain.rows;
    const cols = state.terrain.cols;
    const minElev = state.terrain.min_elevation_m;

    for (let i = 0; i < pos.count; i++) {
        const c = i % cols;
        const r = Math.floor(i / cols);

        let h = depthGrid[r][c] || 0.0;
        // Canal completely removed: zero out artificial channel water depth
        if (state.originalRiverMask && state.originalRiverMask[r][c]) {
            h = 0.0;
        }

        const elev = state.terrain.elevation_grid[r][c];
        const baseElevY = (elev - minElev) * 0.55;

        if (h > 0.04) {
            // Water physically rises above terrain surface only where true surface flooding ponds
            const waterHeight = Math.max(0.06, h * 0.75);
            pos.setY(i, baseElevY + waterHeight);
        } else {
            // Dry ground: water surface sits safely 0.15m beneath terrain (strictly no canal or cliffs)
            pos.setY(i, baseElevY - 0.15);
        }
    }

    pos.needsUpdate = true;
    state.waterMesh.geometry.computeVertexNormals();
}

function update3DRoadConditions(depthGrid) {
    if (!state.roadLinesGroup || !depthGrid) return;

    state.roadLinesGroup.children.forEach(mesh => {
        const edgeIds = mesh.userData.edgeIds || (mesh.userData.edgeId ? [mesh.userData.edgeId] : null);
        if (!edgeIds || edgeIds.length === 0) return; // Skip non-edge children like bridge piers

        const primaryEdge = state.roadGraph[edgeIds[0]];
        if (!primaryEdge) return;

        // Sample water depth across 5 points along road segment using exact world coords
        let maxWater = 0.0;
        const steps = 4;
        for (let s = 0; s <= steps; s++) {
            const frac = s / steps;
            const wx = primaryEdge.startWorld.x + (primaryEdge.endWorld.x - primaryEdge.startWorld.x) * frac;
            const wy = primaryEdge.startWorld.y + (primaryEdge.endWorld.y - primaryEdge.startWorld.y) * frac;
            const c = Math.floor(npClamp(wx / 50.0, 0, 119));
            const r = Math.floor(npClamp((6000.0 - wy) / 50.0, 0, 119));
            let d = depthGrid[r][c] || 0.0;
            // Canal removed: zero out artificial river depth so crossing streets are normal dry streets
            if (state.originalRiverMask && state.originalRiverMask[r][c]) {
                d = 0.0;
            }
            if (d > maxWater) maxWater = d;
        }

        // Update BOTH directed edges in state.roadGraph
        edgeIds.forEach(eid => {
            const edge = state.roadGraph[eid];
            if (edge) {
                edge.waterDepth = maxWater;
                edge.isClosed = (maxWater >= 0.15);
            }
        });

        // Find associated barrier mesh group by streetKey or edgeId
        const streetKey = mesh.userData.streetKey;
        const barrier = state.barriersGroup.children.find(b => 
            (streetKey && b.userData.streetKey === streetKey) || 
            (b.userData.edgeId && edgeIds.includes(b.userData.edgeId))
        );
        const asphalt = mesh.userData.asphalt || mesh;

        if (maxWater >= 0.30) {
            // Severe Flooding / Emergency Cutoff: Hazard Red ribbon + Active Barricades
            asphalt.material.color.setHex(0xef4444);
            asphalt.material.emissive.setHex(0xb91c1c);
            asphalt.material.emissiveIntensity = 0.8;
            if (barrier) barrier.visible = true;
        } else if (maxWater >= 0.15) {
            // Passenger Stall / Closed Road: Glowing Hazard Orange-Red + Active Barricades
            asphalt.material.color.setHex(0xf97316);
            asphalt.material.emissive.setHex(0xc2410c);
            asphalt.material.emissiveIntensity = 0.6;
            if (barrier) barrier.visible = true;
        } else if (maxWater >= 0.05) {
            // Shallow Ponding: Speed Degraded (Amber)
            asphalt.material.color.setHex(0xeab308);
            asphalt.material.emissive.setHex(0x78350f);
            asphalt.material.emissiveIntensity = 0.2;
            if (barrier) barrier.visible = false;
        } else {
            // Open Road: Asphalt Dark Slate
            asphalt.material.color.setHex(mesh.userData.baseColor);
            asphalt.material.emissive.setHex(0x000000);
            asphalt.material.emissiveIntensity = 0.0;
            if (barrier) barrier.visible = false;
        }
    });
}

function update3DFacilitiesStatus(depthGrid, snap) {
    if (!state.facilitiesGroup) return;

    const opSubs = snap ? snap.operational_substations : 3;

    state.facilitiesGroup.children.forEach(obj => {
        if (!obj.userData || !obj.userData.category) return;

        if (obj.userData.category === "electrical_substations") {
            const isTripped = (opSubs < 3);
            obj.material.color.setHex(isTripped ? 0xef4444 : 0x00f0ff);
            if (obj.userData.badge) {
                obj.userData.badge.material.color.setHex(isTripped ? 0xef4444 : 0x00f0ff);
            }
        }
    });

    // Cascading blackout visual effect: dim building window lights if grid trips
    if (state.buildingsGroup) {
        const isBlackout = (opSubs < 3);
        state.buildingsGroup.children.forEach(bldg => {
            if (bldg.position.z > 50) { // South alluvial district
                bldg.traverse(child => {
                    if (child.isMesh && child.material) {
                        const mats = Array.isArray(child.material) ? child.material : [child.material];
                        mats.forEach(m => {
                            if (m.userData && m.userData.baseEmissive !== undefined) {
                                m.emissiveIntensity = isBlackout ? 0.02 : m.userData.baseEmissive;
                            }
                        });
                    }
                });
            }
        });
    }
}


// =========================================================================
// CAMERA PRESETS & DRONE ORBIT
// =========================================================================

function setCameraPreset(mode) {
    state.isDroneOrbit = false;
    document.querySelectorAll(".cam-btn").forEach(btn => btn.classList.remove("active"));

    if (mode === "overview") {
        document.getElementById("camOverview").classList.add("active");
        tweenCamera(new THREE.Vector3(0, 270, 320), new THREE.Vector3(0, 15, 0));
    } else if (mode === "downtown") {
        document.getElementById("camDowntown").classList.add("active");
        tweenCamera(new THREE.Vector3(-40, 75, -20), new THREE.Vector3(20, 25, 10));
    } else if (mode === "hospital") {
        document.getElementById("camHospital").classList.add("active");
        tweenCamera(new THREE.Vector3(-150, 65, -190), new THREE.Vector3(-120, 20, -180));
    } else if (mode === "river") {
        document.getElementById("camRiver").classList.add("active");
        tweenCamera(new THREE.Vector3(140, 55, 170), new THREE.Vector3(120, 10, 140));
    } else if (mode === "drone") {
        state.isDroneOrbit = true;
        document.getElementById("camDrone").classList.add("active");
    }
}

function tweenCamera(targetPos, targetLookAt) {
    state.camera.position.copy(targetPos);
    state.controls.target.copy(targetLookAt);
    state.controls.update();
}

function setWeatherMode(mode) {
    state.weatherMode = mode;
    document.querySelectorAll(".weather-hud .cam-btn").forEach(btn => btn.classList.remove("active"));
    if (mode === "auto") {
        document.getElementById("envAuto")?.classList.add("active");
        const r = parseFloat(sliderRain.value);
        state.targetWeatherFactor = npClamp(r / 130.0, 0.0, 1.0);
    } else if (mode === "day") {
        document.getElementById("envDay")?.classList.add("active");
        state.targetWeatherFactor = 0.0;
    } else if (mode === "night") {
        document.getElementById("envNight")?.classList.add("active");
        state.targetWeatherFactor = -1.0;
    }
}

function applyAtmosphereUpdate(dt) {
    if (!state.scene || !state.dirLight || !state.hemiLight) return;

    // Smooth weather interpolation
    const lerpSpeed = Math.min(1.0, dt * 3.5);
    state.currentWeatherFactor = THREE.MathUtils.lerp(state.currentWeatherFactor, state.targetWeatherFactor, lerpSpeed);

    const wf = state.currentWeatherFactor;

    // 1. Update Dynamic Hemispherical SkyDome Gradient & Sun Disc
    updateSkyDomeGradient(wf);

    let fogCol, fogDens, sunCol, sunInt, hemiSky, hemiGnd, hemiInt, rainOpac, emissiveScale;

    if (state.weatherMode === "night" || wf < -0.5) {
        // Midnight City Twin Atmosphere
        fogCol = new THREE.Color(0x050a18);
        fogDens = 0.0012;
        sunCol = new THREE.Color(0x38bdf8); // Cool moonlight
        sunInt = 0.35;
        hemiSky = new THREE.Color(0x1e1b4b);
        hemiGnd = new THREE.Color(0x020617);
        hemiInt = 0.25;
        rainOpac = (parseFloat(sliderRain.value) > 5) ? 0.55 : 0.0;
        emissiveScale = 1.9;

        if (state.sunMesh) state.sunMesh.visible = false;
    } else {
        // Daytime to Tempest Storm continuum: wf from 0.0 (Sunny Day) to 1.0 (Severe Deluge)
        const clampedWf = npClamp(wf, 0.0, 1.0);

        // Clear Sunny Day Colors
        const dayFog = new THREE.Color(0xbfe3fc);      // Soft daylight horizon haze
        const daySun = new THREE.Color(0xfffaed);      // Golden natural sunlight
        const dayHemiSky = new THREE.Color(0xe0f2fe);
        const dayHemiGnd = new THREE.Color(0x475569);

        // Severe Storm Deluge Colors
        const stormFog = new THREE.Color(0x0a101d);    // Dark rain fog
        const stormSun = new THREE.Color(0x7dd3fc);    // Overcast diffused light
        const stormHemiSky = new THREE.Color(0x1e293b);
        const stormHemiGnd = new THREE.Color(0x020617);

        fogCol = dayFog.clone().lerp(stormFog, clampedWf);
        fogDens = THREE.MathUtils.lerp(0.00035, 0.0016, clampedWf);

        sunCol = daySun.clone().lerp(stormSun, clampedWf);
        sunInt = THREE.MathUtils.lerp(1.95, 0.90, clampedWf);

        hemiSky = dayHemiSky.clone().lerp(stormHemiSky, clampedWf);
        hemiGnd = dayHemiGnd.clone().lerp(stormHemiGnd, clampedWf);
        hemiInt = THREE.MathUtils.lerp(0.85, 0.55, clampedWf);

        rainOpac = (clampedWf < 0.04) ? 0.0 : THREE.MathUtils.lerp(0.12, 0.50, (clampedWf - 0.04) / 0.96);
        emissiveScale = THREE.MathUtils.lerp(0.20, 1.5, clampedWf);

        if (state.sunMesh) {
            state.sunMesh.visible = (clampedWf < 0.75);
            state.sunMesh.material.opacity = Math.max(0.0, 1.0 - clampedWf * 2.2);
            state.sunMesh.lookAt(state.camera.position);
        }
    }

    // Three.js scene fog
    if (state.scene.fog) {
        state.scene.fog.color.copy(fogCol);
        state.scene.fog.density = fogDens;
    }

    // Directional Sunlight & Shadows
    state.dirLight.color.copy(sunCol);
    state.dirLight.intensity = sunInt;

    // Ambient Hemisphere Lighting
    state.hemiLight.color.copy(hemiSky);
    state.hemiLight.groundColor.copy(hemiGnd);
    state.hemiLight.intensity = hemiInt;

    // Rain Particle Streaks (opacity & visibility)
    if (state.rainSystem) {
        const chk = document.getElementById("chkRain");
        const isChecked = chk ? chk.checked : true;
        if (rainOpac <= 0.01 || !isChecked) {
            state.rainSystem.visible = false;
        } else {
            state.rainSystem.visible = true;
            state.rainSystem.material.opacity = rainOpac;
        }
    }

    // Dynamic PBR Road Surface Wetness (Glossy water sheen and reflections)
    const wetness = (state.weatherMode === "night") ? ((parseFloat(sliderRain.value) > 5) ? 0.85 : 0.1) : npClamp(wf, 0.0, 1.0);
    const roadRoughness = THREE.MathUtils.lerp(0.82, 0.15, wetness);
    const roadMetalness = THREE.MathUtils.lerp(0.08, 0.45, wetness);

    if (state.roadLinesGroup) {
        state.roadLinesGroup.children.forEach(grp => {
            if (grp.userData && grp.userData.asphalt) {
                grp.userData.asphalt.material.roughness = roadRoughness;
                grp.userData.asphalt.material.metalness = roadMetalness;
            }
        });
    }

    // Dynamic Ground Wetness
    if (state.terrainMesh) {
        state.terrainMesh.material.roughness = THREE.MathUtils.lerp(0.85, 0.38, wetness);
        state.terrainMesh.material.metalness = THREE.MathUtils.lerp(0.05, 0.28, wetness);
    }

    // Building Window Emissive Glow (Daylight vs Storm vs Blackout)
    if (state.buildingsGroup) {
        const isGridBlackout = (state.currentFrame && state.currentFrame.snapshot && state.currentFrame.snapshot.operational_substations < 3);
        state.buildingsGroup.children.forEach(bldgGroup => {
            const isBlackedOut = isGridBlackout && bldgGroup.position.z > 50;
            bldgGroup.traverse(child => {
                if (child.isMesh && child.material) {
                    const mats = Array.isArray(child.material) ? child.material : [child.material];
                    mats.forEach(m => {
                        if (m && m.userData && m.userData.baseEmissive !== undefined) {
                            if (isBlackedOut) {
                                m.emissiveIntensity = 0.02;
                            } else {
                                m.emissiveIntensity = m.userData.baseEmissive * emissiveScale;
                            }
                        }
                    });
                }
            });
        });
    }
}


// =========================================================================
// ANIMATION & RENDER LOOP
// =========================================================================

function animate() {
    requestAnimationFrame(animate);

    const dt = state.clock.getDelta();
    const elapsedTime = state.clock.getElapsedTime();

    // 0. Update Dynamic Atmospheric Weather & Lighting (Smooth Day/Storm Transition)
    applyAtmosphereUpdate(dt);

    // 1. Update Traffic Simulation (Two-way right-hand lanes & anti-overlap car following)
    updateTraffic(dt);

    // 2. Animate Water Surface Ripples
    if (state.waterMesh && state.waterMesh.visible) {
        state.waterMesh.material.opacity = 0.80 + Math.sin(elapsedTime * 2.5) * 0.05;
    }

    // 3. Animate Natural Falling Rain Particles with Wind Drift
    if (state.rainSystem && state.rainSystem.visible && state.rainData && state.rainData.length > 0) {
        const wf = npClamp(state.currentWeatherFactor, 0.0, 1.0);
        const speedMult = 0.85 + wf * 0.75;
        const windSpeedX = -20.0 * wf;
        const windSpeedZ = 8.0 * wf;
        const pos = state.rainSystem.geometry.attributes.position;

        for (let i = 0; i < state.rainData.length; i++) {
            const r = state.rainData[i];
            r.y -= (r.speed * speedMult) * dt;
            r.x += windSpeedX * dt;
            r.z += windSpeedZ * dt;

            if (r.y < 0) {
                r.y = 250 + Math.random() * 25;
                r.x = (Math.random() - 0.5) * 620;
                r.z = (Math.random() - 0.5) * 620;
            }

            pos.setXYZ(i, r.x, r.y, r.z);
        }
        pos.needsUpdate = true;
    }

    // 4. Pulse FAA Aviation Warning Beacons on Skyscraper Broadcast Spires
    if (state.aviationBeacons && state.aviationBeacons.length > 0) {
        const isBeaconFlash = (elapsedTime % 1.0) < 0.28;
        const beaconColor = isBeaconFlash ? 0xff2222 : 0x330505;
        for (let i = 0; i < state.aviationBeacons.length; i++) {
            state.aviationBeacons[i].material.color.setHex(beaconColor);
        }
    }

    // 5. Drone Orbit Mode
    if (state.isDroneOrbit) {
        state.orbitAngle += dt * 0.15;
        const radius = 340;
        state.camera.position.x = Math.sin(state.orbitAngle) * radius;
        state.camera.position.z = Math.cos(state.orbitAngle) * radius;
        state.camera.position.y = 190 + Math.sin(state.orbitAngle * 0.5) * 30;
        state.controls.target.set(0, 15, 0);
    }

    // 6. Random Storm Lightning Flashes (only in severe storms)
    if (state.currentWeatherFactor > 0.65 && Math.random() < 0.009) {
        state.flashLight.intensity = 4.5;
        setTimeout(() => { if (state.flashLight) state.flashLight.intensity = 0.0; }, 75);
    }

    // 7. Flash Warning Strobe Beacons on Road Barricades
    if (state.barriersGroup && state.barriersGroup.visible) {
        const flashTick = Math.floor(Date.now() / 250) % 2 === 0;
        state.barriersGroup.children.forEach(pair => {
            if (pair.visible) {
                pair.children.forEach(single => {
                    if (single.userData && single.userData.beacons) {
                        single.userData.beacons[0].material.color.setHex(flashTick ? 0xef4444 : 0x450a0a);
                        single.userData.beacons[1].material.color.setHex(!flashTick ? 0xf59e0b : 0x451a03);
                    }
                });
            }
        });
    }

    state.controls.update();
    state.renderer.render(state.scene, state.camera);
}


// =========================================================================
// PLAYBACK ANIMATION LOOP
// =========================================================================

function togglePlayback() {
    if (state.isPlaying) stopPlayback();
    else startPlayback();
}

function startPlayback() {
    state.isPlaying = true;
    elPlayPauseLabel.innerText = "PAUSE";
    elPlayIcon.innerText = "⏸";
    elTimelinePlayIcon.innerText = "⏸";

    if (state.playTimer) clearInterval(state.playTimer);
    state.playTimer = setInterval(() => {
        if (state.currentStepIdx >= state.totalSteps) {
            stopPlayback();
            return;
        }
        scrubToStep(state.currentStepIdx + 1);
    }, 400);
}

function stopPlayback() {
    state.isPlaying = false;
    elPlayPauseLabel.innerText = "PLAY";
    elPlayIcon.innerText = "▶";
    elTimelinePlayIcon.innerText = "▶";
    if (state.playTimer) {
        clearInterval(state.playTimer);
        state.playTimer = null;
    }
}


// =========================================================================
// UI EVENT LISTENERS
// =========================================================================

function setupEventListeners() {
    // Header Actions
    elRunScenario.addEventListener("click", () => runScenario(true));
    elPlayPause.addEventListener("click", togglePlayback);
    elReset.addEventListener("click", () => scrubToStep(0));

    // Timeline Scrubber
    elTimelineScrubber.addEventListener("input", (e) => {
        stopPlayback();
        scrubToStep(parseInt(e.target.value));
    });
    elBtnStepBack.addEventListener("click", () => {
        stopPlayback();
        scrubToStep(state.currentStepIdx - 1);
    });
    elBtnStepPlay.addEventListener("click", togglePlayback);
    elBtnStepForward.addEventListener("click", () => {
        stopPlayback();
        scrubToStep(state.currentStepIdx + 1);
    });

    // Camera Presets
    document.getElementById("camOverview").addEventListener("click", () => setCameraPreset("overview"));
    document.getElementById("camDowntown").addEventListener("click", () => setCameraPreset("downtown"));
    document.getElementById("camHospital").addEventListener("click", () => setCameraPreset("hospital"));
    document.getElementById("camRiver").addEventListener("click", () => setCameraPreset("river"));
    document.getElementById("camDrone").addEventListener("click", () => setCameraPreset("drone"));

    // Atmosphere & Environment Presets
    const btnEnvAuto = document.getElementById("envAuto");
    const btnEnvDay = document.getElementById("envDay");
    const btnEnvNight = document.getElementById("envNight");
    if (btnEnvAuto) btnEnvAuto.addEventListener("click", () => setWeatherMode("auto"));
    if (btnEnvDay) btnEnvDay.addEventListener("click", () => setWeatherMode("day"));
    if (btnEnvNight) btnEnvNight.addEventListener("click", () => setWeatherMode("night"));

    // Sliders
    sliderRain.addEventListener("input", (e) => { 
        valRain.innerText = `${e.target.value} mm/hr`; 
        if (state.weatherMode === "auto") {
            const r = parseFloat(e.target.value);
            state.targetWeatherFactor = npClamp(r / 130.0, 0.0, 1.0);
        }
    });
    sliderDuration.addEventListener("input", (e) => { valDuration.innerText = `${parseFloat(e.target.value).toFixed(1)} hrs`; });
    sliderDrain.addEventListener("input", (e) => { valDrain.innerText = `${e.target.value}%`; });
    sliderPump1.addEventListener("input", (e) => { valPump1.innerText = `${e.target.value}%`; });
    sliderPump2.addEventListener("input", (e) => { valPump2.innerText = `${e.target.value}%`; });

    // Dynamics Engine Selector
    modePhysicsBtn.addEventListener("click", () => setEngineMode("physics"));
    modeNeuralBtn.addEventListener("click", () => setEngineMode("world_model"));

    // NLP Execution
    btnNlp.addEventListener("click", handleNlpPrompt);

    // Branching
    btnFork.addEventListener("click", handleForkScenario);
    btnCompare.addEventListener("click", handleCompareBranches);

    // 3D Layer Checkboxes
    setupLayerToggle("chkWater", () => state.waterMesh);
    setupLayerToggle("chkTraffic", () => state.trafficGroup);
    setupLayerToggle("chkRoads", () => state.roadLinesGroup);
    setupLayerToggle("chkBuildings", () => state.buildingsGroup);
    setupLayerToggle("chkFacilities", () => state.facilitiesGroup);
    setupLayerToggle("chkRain", () => state.rainSystem);

    // Raycast Inspector
    const container = document.getElementById("city3dViewport");
    container.addEventListener("mousemove", handle3DMouseMove);
    container.addEventListener("mouseleave", () => inspectorCard.classList.add("hidden"));
}

function setupLayerToggle(id, layerGetter) {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener("change", (e) => {
        const layer = layerGetter();
        if (layer) layer.visible = e.target.checked;
    });
}

function setEngineMode(mode) {
    state.engineMode = mode;
    if (mode === "physics") {
        modePhysicsBtn.classList.add("active");
        modeNeuralBtn.classList.remove("active");
        elEngineBadge.innerText = "COUPLED 2D PHYSICS (Exact)";
        elEngineBadge.className = "val val-success";
    } else {
        modeNeuralBtn.classList.add("active");
        modePhysicsBtn.classList.remove("active");
        elEngineBadge.innerText = "NEURAL WORLD MODEL (2ms)";
        elEngineBadge.className = "val val-accent";
    }
}


// =========================================================================
// NLP SCENARIO PARSER HANDLER
// =========================================================================

async function handleNlpPrompt() {
    const prompt = nlpInput.value.trim();
    if (!prompt) return;

    try {
        const res = await fetch("/api/scenario/parse", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prompt })
        });
        const parsed = await res.json();

        // Update UI controls
        sliderRain.value = parsed.rainfall_mm_hr;
        valRain.innerText = `${parsed.rainfall_mm_hr} mm/hr`;

        if (parsed.duration_hours) {
            sliderDuration.value = parsed.duration_hours;
            valDuration.innerText = `${parsed.duration_hours.toFixed(1)} hrs`;
        }

        sliderDrain.value = Math.round(parsed.drainage_capacity_factor * 100);
        valDrain.innerText = `${sliderDrain.value}%`;

        if (parsed.pump_overrides && "PUMP-1" in parsed.pump_overrides) {
            sliderPump1.value = Math.round(parsed.pump_overrides["PUMP-1"] * 100);
            valPump1.innerText = `${sliderPump1.value}%`;
        }
        if (parsed.pump_overrides && "PUMP-2" in parsed.pump_overrides) {
            sliderPump2.value = Math.round(parsed.pump_overrides["PUMP-2"] * 100);
            valPump2.innerText = `${sliderPump2.value}%`;
        }

        nlpTags.innerHTML = "";
        if (parsed.recognized_entities) {
            parsed.recognized_entities.forEach(ent => {
                const tag = document.createElement("span");
                tag.className = "tag";
                tag.innerText = ent;
                nlpTags.appendChild(tag);
            });
        }

        // Run full scenario automatically
        await runScenario(true);
    } catch (err) {
        console.error("NLP execute error:", err);
    }
}


// =========================================================================
// COUNTERFACTUAL BRANCHING & COMPARISON
// =========================================================================

async function handleForkScenario() {
    const branchName = `intervention_branch_${Date.now().toString().slice(-4)}`;
    try {
        const res = await fetch("/api/branch/fork", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                source_branch_id: state.currentBranch,
                new_branch_id: branchName,
                intervention: {
                    pump_overrides: { "PUMP-1": 1.0, "PUMP-2": 1.0 },
                    drainage_capacity_factor: 1.0
                }
            })
        });

        if (res.ok) {
            state.forkedBranch = branchName;
            state.currentBranch = branchName;
            document.getElementById("currentBranchDisplay").innerText = branchName;
            btnCompare.disabled = false;
            btnFork.innerText = "FORK ANOTHER BRANCH";
            alert(`Forked timeline: '${branchName}' with optimal stormwater countermeasures.`);
        }
    } catch (err) {
        console.error("Fork error:", err);
    }
}

async function handleCompareBranches() {
    if (!state.forkedBranch) return;

    try {
        const res = await fetch("/api/branch/compare", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                branch_a_id: "baseline",
                branch_b_id: state.forkedBranch
            })
        });

        if (!res.ok) throw new Error("Compare failed");
        const diff = await res.json();

        diffSection.classList.remove("hidden");
        document.getElementById("diffFloodArea").innerText = `${diff.delta_flooded_area_km2 > 0 ? '+' : ''}${diff.delta_flooded_area_km2.toFixed(3)} km²`;
        document.getElementById("diffFloodArea").className = diff.delta_flooded_area_km2 < 0 ? "val-success" : "val-danger";

        document.getElementById("diffEmsTime").innerText = `${diff.delta_avg_ems_time_min > 0 ? '+' : ''}${diff.delta_avg_ems_time_min.toFixed(2)} min`;
        document.getElementById("diffEmsTime").className = diff.delta_avg_ems_time_min < 0 ? "val-success" : "val-danger";

        document.getElementById("diffClosedRoads").innerText = `${diff.delta_closed_roads_passenger > 0 ? '+' : ''}${diff.delta_closed_roads_passenger}`;

        const recBadge = document.getElementById("diffRecommendation");
        if (diff.is_branch_b_better) {
            recBadge.innerText = "COUNTERMEASURE RECOMMENDED: Reduces flooded area and preserves critical trauma routing.";
            recBadge.className = "diff-badge badge-success";
        } else {
            recBadge.innerText = "NEGLIGIBLE IMPACT: Intervention insufficient to arrest basin inundation.";
            recBadge.className = "diff-badge badge-warning";
        }
    } catch (err) {
        console.error("Compare error:", err);
    }
}


// =========================================================================
// RAYCAST LOCATION & TRAFFIC INSPECTOR
// =========================================================================

const raycaster = new THREE.Raycaster();
const mouseVec = new THREE.Vector2();

function handle3DMouseMove(e) {
    if (!state.terrainMesh) return;
    const rect = state.renderer.domElement.getBoundingClientRect();
    mouseVec.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    mouseVec.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouseVec, state.camera);
    const intersects = raycaster.intersectObject(state.terrainMesh);

    if (intersects.length > 0) {
        const pt = intersects[0].point;
        const wx = Math.round(pt.x * 10.0 + 3000.0);
        const wy = Math.round(3000.0 - pt.z * 10.0);

        const col = Math.floor(npClamp(wx / 50.0, 0, 119));
        const row = Math.floor(npClamp((6000.0 - wy) / 50.0, 0, 119));

        const elev = state.terrain.elevation_grid[row][col];
        const depthGrid = state.currentFrame ? state.currentFrame.surface_depth_grid : null;
        const water = depthGrid ? depthGrid[row][col] : 0.0;

        inspectorCard.classList.remove("hidden");
        inspectorContent.innerHTML = `
            <div class="inspector-row"><span>3D Coordinates:</span><strong>(${wx}m, ${wy}m)</strong></div>
            <div class="inspector-row"><span>Ground Elevation:</span><strong>${elev.toFixed(1)} m</strong></div>
            <div class="inspector-row"><span>Water Inundation:</span><strong style="color: ${water > 0.15 ? 'var(--danger-red)' : (water > 0.04 ? 'var(--accent-cyan)' : 'var(--text-primary)')}">${(water * 100).toFixed(1)} cm</strong></div>
            <div class="inspector-row"><span>Road Traffic:</span><strong style="color: ${water >= 0.15 ? 'var(--danger-red)' : 'var(--success-green)'}">${water >= 0.15 ? 'STALLED (Flooded)' : 'FLOWING (Open)'}</strong></div>
            <div class="inspector-row"><span>Emergency Route:</span><strong style="color: ${water >= 0.30 ? 'var(--danger-red)' : 'var(--success-green)'}">${water >= 0.30 ? 'IMPASSABLE' : 'PASSABLE'}</strong></div>
        `;
    } else {
        inspectorCard.classList.add("hidden");
    }
}


// =========================================================================
// TELEMETRY & SPARKLINES
// =========================================================================

function updateTelemetry(snap) {
    if (!snap) return;

    const area15 = snap.flooded_area_km2_gt15cm || 0.0;
    const area30 = snap.flooded_area_km2_gt30cm || 0.0;
    kpiFlooded.innerText = `${area15.toFixed(2)} km²`;
    kpiFlooded30.innerText = `${area30.toFixed(2)} km² >30cm`;

    const maxD = snap.max_water_depth_m || 0.0;
    kpiMaxDepth.innerText = `${maxD.toFixed(2)} m`;

    const closedRoads = snap.closed_roads_passenger || 0;
    const closedEmerg = snap.closed_roads_emergency || 0;
    kpiClosedRoads.innerText = `${closedRoads} Closed`;
    kpiClosedEmergency.innerText = `${closedEmerg} emergency severed`;

    const emsTime = snap.avg_ems_response_min || 5.2;
    const emsCov = snap.ems_coverage_pct || 100.0;
    kpiEms.innerText = `${emsTime.toFixed(1)} min`;
    kpiEmsCov.innerText = `${emsCov.toFixed(1)}% Reachable`;

    const subs = snap.operational_substations !== undefined ? snap.operational_substations : 3;
    kpiSubs.innerText = `${subs}/3 Active`;
    kpiSubs.className = subs < 3 ? "kpi-value val-danger" : "kpi-value val-success";
    kpiSubLoad.innerText = subs < 3 ? "Local Grid Blackout" : "All Grids Energized";

    const disPumps = snap.disabled_pumps || 0;
    const activePumpCap = Math.max(0, 25.0 - disPumps * 10.0);
    kpiPumps.innerText = `${activePumpCap.toFixed(0)} m³/s`;
    kpiPumpStatus.innerText = disPumps > 0 ? `${disPumps} Pump(s) Offline` : "Pumps 1 & 2 Operational";
}

function updateSparklines() {
    drawSparkline("chartFlood", state.historyFloodedArea, "#00f0ff");
    drawSparkline("chartEms", state.historyEmsTime, "#f59e0b");
}

function drawSparkline(canvasId, series, color) {
    const c = document.getElementById(canvasId);
    if (!c || !series || series.length < 2) return;
    const cx = c.getContext("2d");
    cx.clearRect(0, 0, c.width, c.height);

    const min = 0;
    const max = Math.max(...series, 1.0);

    cx.strokeStyle = color;
    cx.lineWidth = 1.8;
    cx.beginPath();

    series.forEach((val, i) => {
        const x = (i / (series.length - 1)) * c.width;
        const y = c.height - ((val - min) / (max - min)) * (c.height - 12) - 6;
        if (i === 0) cx.moveTo(x, y);
        else cx.lineTo(x, y);
    });
    cx.stroke();
}

// Helpers
function worldTo3D(x, y) {
    return {
        x: (x - 3000.0) * SCALE,
        z: (3000.0 - y) * SCALE
    };
}

function getTerrainElevation(x, y) {
    if (!state.terrain) return 0.0;
    const fx = npClamp(x / 50.0, 0, 118.999);
    const fy = npClamp((6000.0 - y) / 50.0, 0, 118.999);

    const c0 = Math.floor(fx);
    const r0 = Math.floor(fy);
    const c1 = Math.min(119, c0 + 1);
    const r1 = Math.min(119, r0 + 1);

    const tx = fx - c0;
    const ty = fy - r0;

    const e00 = state.terrain.elevation_grid[r0][c0];
    const e10 = state.terrain.elevation_grid[r0][c1];
    const e01 = state.terrain.elevation_grid[r1][c0];
    const e11 = state.terrain.elevation_grid[r1][c1];

    const elev = (e00 * (1.0 - tx) + e10 * tx) * (1.0 - ty) + (e01 * (1.0 - tx) + e11 * tx) * ty;
    return (elev - state.terrain.min_elevation_m) * 0.55;
}

function npClamp(val, min, max) {
    return Math.max(min, Math.min(val, max));
}

function formatTime(totalSeconds) {
    const hrs = Math.floor(totalSeconds / 3600);
    const mins = Math.floor((totalSeconds % 3600) / 60);
    const secs = Math.floor(totalSeconds % 60);
    return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

function formatHoursMinutes(totalSeconds) {
    const hrs = Math.floor(totalSeconds / 3600);
    const mins = Math.floor((totalSeconds % 3600) / 60);
    return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}`;
}

// Start application on DOM ready
window.addEventListener("DOMContentLoaded", init);
