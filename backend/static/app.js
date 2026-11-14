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

    // Interactive 3D Interventions (Sandbag Berms, Pump Overrides & Countermeasures)
    sandbagsGroup: null,
    sandbagBerms: new Map(), // streetKey -> { mesh, edge, u, v }
    activeInterveneTool: null, // "sandbag" | "boost_pumps" | null
    isDragging: false,

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

let sliderDebounceTimer = null;

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

// =========================================================================
// ARCHITECTURAL PALETTES & PROCEDURAL TEXTURES (COLORS & CORES)
// =========================================================================

const ARCH_PALETTES = {
    azure_glass: {
        name: "Azure Glass Curtain Wall",
        baseBg: "#0284c7",
        spandrel: "#cbd5e1",
        spandrelShadow: "#475569",
        mullion: "#e2e8f0",
        mullionShadow: "#334155",
        winDark: "#0369a1",
        winReflect: "#38bdf8",
        lights: ["#fef08a", "#bae6fd", "#e0f2fe", "#fde047"],
        roughness: 0.16,
        metalness: 0.82,
        crownColor: 0x38bdf8,
        baseEmissive: 0.28
    },
    bronze_glass: {
        name: "Warm Bronze & Champagne High-Rise",
        baseBg: "#78350f",
        spandrel: "#d97706",
        spandrelShadow: "#451a03",
        mullion: "#b45309",
        mullionShadow: "#291002",
        winDark: "#451a03",
        winReflect: "#f59e0b",
        lights: ["#fef3c7", "#fde68a", "#fed7aa", "#f59e0b"],
        roughness: 0.20,
        metalness: 0.78,
        crownColor: 0xf59e0b,
        baseEmissive: 0.26
    },
    silver_tower: {
        name: "Silver & Platinum Modern Tower",
        baseBg: "#f1f5f9",
        spandrel: "#e2e8f0",
        spandrelShadow: "#1e293b",
        mullion: "#cbd5e1",
        mullionShadow: "#0f172a",
        winDark: "#1e3a5f",
        winReflect: "#60a5fa",
        lights: ["#fef9c3", "#93c5fd", "#ffffff", "#bae6fd"],
        roughness: 0.35,
        metalness: 0.38,
        crownColor: 0xffffff,
        baseEmissive: 0.25
    },
    emerald_glass: {
        name: "Emerald Low-E Eco-Tower",
        baseBg: "#0f766e",
        spandrel: "#0d9488",
        spandrelShadow: "#042f2e",
        mullion: "#14b8a6",
        mullionShadow: "#021a19",
        winDark: "#042f2e",
        winReflect: "#5eead4",
        lights: ["#fef08a", "#a7f3d0", "#fde047", "#6ee7b7"],
        roughness: 0.18,
        metalness: 0.75,
        crownColor: 0x10b981,
        baseEmissive: 0.25
    },
    cobalt_tower: {
        name: "Cobalt Prestige Navy Tower",
        baseBg: "#1e3a8a",
        spandrel: "#2563eb",
        spandrelShadow: "#0c1b33",
        mullion: "#f8fafc",
        mullionShadow: "#172554",
        winDark: "#0c1b33",
        winReflect: "#60a5fa",
        lights: ["#dbeafe", "#bfdbfe", "#fef08a", "#93c5fd"],
        roughness: 0.15,
        metalness: 0.85,
        crownColor: 0x60a5fa,
        baseEmissive: 0.27
    }
};

function createGlassCurtainTexture(p) {
    const canvas = document.createElement("canvas");
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext("2d");

    ctx.fillStyle = p.baseBg;
    ctx.fillRect(0, 0, 512, 512);

    const floors = 16;
    const bays = 16;
    const bayW = 512 / bays;
    const floorH = 512 / floors;

    for (let f = 0; f < floors; f++) {
        const y = f * floorH;

        // Horizontal architectural spandrel band
        ctx.fillStyle = p.spandrel;
        ctx.fillRect(0, y, 512, 6);
        ctx.fillStyle = p.spandrelShadow;
        ctx.fillRect(0, y + 5, 512, 1);

        for (let b = 0; b < bays; b++) {
            const x = b * bayW;

            // Vertical structural mullion rib
            ctx.fillStyle = p.mullion;
            ctx.fillRect(x, y + 6, 3, floorH - 6);
            ctx.fillStyle = p.mullionShadow;
            ctx.fillRect(x + 2, y + 6, 1, floorH - 6);

            const winX = x + 3;
            const winY = y + 6;
            const winW = bayW - 4;
            const winH = floorH - 7;

            const hash = Math.sin(f * 29.3 + b * 71.9) * 10000;
            const rand = hash - Math.floor(hash);

            if (rand > 0.52) {
                // Illuminated office workspace
                const grad = ctx.createLinearGradient(winX, winY, winX, winY + winH);
                const lightCol = p.lights[Math.floor(rand * p.lights.length) % p.lights.length];
                grad.addColorStop(0, lightCol);
                grad.addColorStop(1, p.winDark);
                ctx.fillStyle = grad;
            } else {
                // Daytime reflective glass pane reflecting the sky
                const grad = ctx.createLinearGradient(winX, winY, winX, winY + winH);
                grad.addColorStop(0, p.winReflect);
                grad.addColorStop(1, p.winDark);
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

function createGlassCurtainEmissiveTexture(p) {
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
            const winY = y + 6;
            const winW = bayW - 4;
            const winH = floorH - 7;

            const hash = Math.sin(f * 29.3 + b * 71.9) * 10000;
            const rand = hash - Math.floor(hash);

            if (rand > 0.52) {
                const lightCol = p.lights[Math.floor(rand * p.lights.length) % p.lights.length];
                ctx.fillStyle = lightCol;
                ctx.fillRect(winX, winY, winW, winH);
            }
        }
    }

    const tex = new THREE.CanvasTexture(canvas);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    return tex;
}

function createTerracottaBrickTexture() {
    const canvas = document.createElement("canvas");
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext("2d");

    // Mortar joint background
    ctx.fillStyle = "#d6d3d1";
    ctx.fillRect(0, 0, 512, 512);

    const floors = 12;
    const bays = 10;
    const bayW = 512 / bays;
    const floorH = 512 / floors;

    const brickColors = ["#c2410c", "#9a3412", "#ea580c", "#b91c1c", "#7c2d12"];

    for (let f = 0; f < floors; f++) {
        const y = f * floorH;

        // Decorative stone cornice / belt course
        ctx.fillStyle = "#f5f5f4";
        ctx.fillRect(0, y + floorH - 8, 512, 6);
        ctx.fillStyle = "#78716c";
        ctx.fillRect(0, y + floorH - 2, 512, 2);

        // Brick courses
        const rowsPerFloor = 6;
        const rowH = (floorH - 8) / rowsPerFloor;
        for (let r = 0; r < rowsPerFloor; r++) {
            const ry = y + r * rowH;
            const brickW = 18;
            const offset = (r % 2 === 0) ? 0 : 9;
            for (let bx = -offset; bx < 512; bx += brickW) {
                const hash = Math.sin(f * 43.1 + r * 19.7 + bx * 7.3) * 1000;
                const cIdx = Math.floor(Math.abs(hash)) % brickColors.length;
                ctx.fillStyle = brickColors[cIdx];
                ctx.fillRect(bx + 1, ry + 1, brickW - 2, rowH - 2);
            }
        }

        // Punched residential windows with stone lintels and sills
        for (let b = 0; b < bays; b++) {
            const x = b * bayW;
            const winX = x + 10;
            const winY = y + 8;
            const winW = bayW - 20;
            const winH = floorH - 24;

            // Stone lintel (top)
            ctx.fillStyle = "#f8fafc";
            ctx.fillRect(winX - 2, winY - 3, winW + 4, 3);
            // Stone sill (bottom)
            ctx.fillStyle = "#e2e8f0";
            ctx.fillRect(winX - 3, winY + winH, winW + 6, 3);

            // Dark window frame
            ctx.fillStyle = "#292524";
            ctx.fillRect(winX, winY, winW, winH);

            const hash = Math.sin(f * 37.1 + b * 61.3) * 10000;
            const rand = hash - Math.floor(hash);

            if (rand > 0.40) {
                const grad = ctx.createLinearGradient(winX, winY, winX, winY + winH);
                if (rand > 0.75) {
                    grad.addColorStop(0, "#fef3c7");
                    grad.addColorStop(1, "#fde68a");
                } else {
                    grad.addColorStop(0, "#fed7aa");
                    grad.addColorStop(1, "#fb923c");
                }
                ctx.fillStyle = grad;
            } else {
                ctx.fillStyle = "#1c1917";
            }
            ctx.fillRect(winX + 2, winY + 2, winW - 4, winH - 4);

            // Window muntins (crossbars)
            ctx.fillStyle = "#44403c";
            ctx.fillRect(winX + winW / 2 - 1, winY + 2, 2, winH - 4);
            ctx.fillRect(winX + 2, winY + winH / 2 - 1, winW - 4, 2);
        }
    }

    const tex = new THREE.CanvasTexture(canvas);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    return tex;
}

function createTerracottaBrickEmissiveTexture() {
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
            const winH = floorH - 24;

            const hash = Math.sin(f * 37.1 + b * 61.3) * 10000;
            const rand = hash - Math.floor(hash);

            if (rand > 0.40) {
                ctx.fillStyle = (rand > 0.75) ? "#fef3c7" : "#fed7aa";
                ctx.fillRect(winX + 2, winY + 2, winW - 4, winH - 4);
            }
        }
    }

    const tex = new THREE.CanvasTexture(canvas);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    return tex;
}

function createLimestoneTexture() {
    const canvas = document.createElement("canvas");
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext("2d");

    // Natural cream French limestone base
    ctx.fillStyle = "#f5f5f4";
    ctx.fillRect(0, 0, 512, 512);

    const floors = 14;
    const bays = 12;
    const bayW = 512 / bays;
    const floorH = 512 / floors;

    for (let f = 0; f < floors; f++) {
        const y = f * floorH;

        // Classical stone cornice / entablature
        ctx.fillStyle = "#e7e5e4";
        ctx.fillRect(0, y, 512, 5);
        ctx.fillStyle = "#a8a29e";
        ctx.fillRect(0, y + 4, 512, 1);

        for (let b = 0; b < bays; b++) {
            const x = b * bayW;

            // Fluted stone pilasters between window bays
            ctx.fillStyle = "#d6d3d1";
            ctx.fillRect(x, y + 5, 4, floorH - 5);
            ctx.fillStyle = "#f5f5f4";
            ctx.fillRect(x + 2, y + 5, 2, floorH - 5);

            // Punch windows with bronze trim
            const winX = x + 7;
            const winY = y + 8;
            const winW = bayW - 14;
            const winH = floorH - 16;

            ctx.fillStyle = "#44403c";
            ctx.fillRect(winX, winY, winW, winH);

            const hash = Math.sin(f * 31.7 + b * 53.9) * 10000;
            const rand = hash - Math.floor(hash);

            if (rand > 0.42) {
                const grad = ctx.createLinearGradient(winX, winY, winX, winY + winH);
                if (rand > 0.78) {
                    grad.addColorStop(0, "#fef3c7");
                    grad.addColorStop(1, "#fde68a");
                } else {
                    grad.addColorStop(0, "#e0f2fe");
                    grad.addColorStop(1, "#93c5fd");
                }
                ctx.fillStyle = grad;
            } else {
                ctx.fillStyle = "#1c1917";
            }
            ctx.fillRect(winX + 2, winY + 2, winW - 4, winH - 4);
        }
    }

    const tex = new THREE.CanvasTexture(canvas);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    return tex;
}

function createLimestoneEmissiveTexture() {
    const canvas = document.createElement("canvas");
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext("2d");

    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, 512, 512);

    const floors = 14;
    const bays = 12;
    const bayW = 512 / bays;
    const floorH = 512 / floors;

    for (let f = 0; f < floors; f++) {
        const y = f * floorH;
        for (let b = 0; b < bays; b++) {
            const x = b * bayW;
            const winX = x + 7;
            const winY = y + 8;
            const winW = bayW - 14;
            const winH = floorH - 16;

            const hash = Math.sin(f * 31.7 + b * 53.9) * 10000;
            const rand = hash - Math.floor(hash);

            if (rand > 0.42) {
                ctx.fillStyle = (rand > 0.78) ? "#fef3c7" : "#93c5fd";
                ctx.fillRect(winX + 2, winY + 2, winW - 4, winH - 4);
            }
        }
    }

    const tex = new THREE.CanvasTexture(canvas);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    return tex;
}

function createElevatorCoreTexture() {
    const canvas = document.createElement("canvas");
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext("2d");

    // Dark graphite metallic core wall
    ctx.fillStyle = "#1e293b";
    ctx.fillRect(0, 0, 256, 256);

    // Architectural horizontal acoustic louvers
    for (let y = 16; y < 180; y += 8) {
        ctx.fillStyle = "#0f172a";
        ctx.fillRect(16, y, 224, 4);
        ctx.fillStyle = "#475569";
        ctx.fillRect(16, y + 4, 224, 2);
    }

    // Heavy steel mechanical access service door
    ctx.fillStyle = "#0f172a";
    ctx.fillRect(96, 185, 64, 71);
    ctx.fillStyle = "#334155";
    ctx.fillRect(99, 188, 58, 68);

    // Door handle & caution signage
    ctx.fillStyle = "#facc15";
    ctx.fillRect(116, 205, 24, 10);
    ctx.fillStyle = "#94a3b8";
    ctx.fillRect(150, 225, 4, 8);

    const tex = new THREE.CanvasTexture(canvas);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    return tex;
}

function createLobbyGlassTexture() {
    const canvas = document.createElement("canvas");
    canvas.width = 256;
    canvas.height = 128;
    const ctx = canvas.getContext("2d");

    // Warm golden illuminated interior atrium
    const grad = ctx.createLinearGradient(0, 0, 0, 128);
    grad.addColorStop(0, "#fef3c7");
    grad.addColorStop(0.6, "#fde68a");
    grad.addColorStop(1, "#38bdf8");
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 256, 128);

    // Structural steel framing
    ctx.fillStyle = "#1e293b";
    ctx.fillRect(0, 0, 256, 8);
    ctx.fillRect(0, 120, 256, 8);

    // Revolving doors & vestibule mullions
    for (let x = 0; x <= 256; x += 32) {
        ctx.fillStyle = "#0f172a";
        ctx.fillRect(x, 0, 4, 128);
        ctx.fillStyle = "#cbd5e1";
        ctx.fillRect(x + 1, 0, 2, 128);
    }

    const tex = new THREE.CanvasTexture(canvas);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    return tex;
}

// Backward-compatible aliases
function createCommercialGlassTexture() {
    return createGlassCurtainTexture(ARCH_PALETTES.azure_glass);
}
function createCommercialEmissiveTexture() {
    return createGlassCurtainEmissiveTexture(ARCH_PALETTES.azure_glass);
}
function createResidentialTexture() {
    return createTerracottaBrickTexture();
}
function createResidentialEmissiveTexture() {
    return createTerracottaBrickEmissiveTexture();
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

    // 7. Interactive Sandbag Berms Group
    state.sandbagsGroup = new THREE.Group();
    state.scene.add(state.sandbagsGroup);
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
    state.nodePositions = {};

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
        state.nodePositions[p.u] = startVec;
        state.nodePositions[p.v] = endVec;

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

    // Shared Base Roof & Helipad Materials
    const roofTex = createRoofTexture();
    const helipadTex = createHelipadTexture();
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

    // Central Elevator & Mechanical Penthouse Core Materials ("cors")
    const coreTex = createElevatorCoreTexture();
    const coreWallMat = new THREE.MeshStandardMaterial({
        map: coreTex,
        roughness: 0.50,
        metalness: 0.65
    });
    const coreRoofMat = new THREE.MeshStandardMaterial({
        color: 0x0f172a,
        roughness: 0.90,
        metalness: 0.10
    });
    const coreMaterials = [coreWallMat, coreWallMat, coreRoofMat, coreRoofMat, coreWallMat, coreWallMat];

    // Ground Floor Entrance Lobby Atrium Glass
    const lobbyTex = createLobbyGlassTexture();
    const lobbyGlassMat = new THREE.MeshStandardMaterial({
        map: lobbyTex,
        emissiveMap: lobbyTex,
        emissive: new THREE.Color(0xffffff),
        emissiveIntensity: 0.35,
        roughness: 0.20,
        metalness: 0.40
    });
    lobbyGlassMat.userData = { baseEmissive: 0.35 };

    // Initialize 5 Distinct Curtain-Glass Architectural Themes
    const glassThemes = {};
    for (const [key, conf] of Object.entries(ARCH_PALETTES)) {
        const tex = createGlassCurtainTexture(conf);
        const emTex = createGlassCurtainEmissiveTexture(conf);
        const wallMat = new THREE.MeshStandardMaterial({
            map: tex,
            emissiveMap: emTex,
            emissive: new THREE.Color(0xffffff),
            emissiveIntensity: conf.baseEmissive,
            roughness: conf.roughness,
            metalness: conf.metalness
        });
        wallMat.userData = { baseEmissive: conf.baseEmissive };
        glassThemes[key] = {
            materials: [wallMat, wallMat, roofMat, roofMat, wallMat, wallMat],
            crownColor: conf.crownColor,
            wallMat: wallMat,
            baseEmissive: conf.baseEmissive
        };
    }

    // Terracotta & Warm Red Brick Residential Theme
    const terraTex = createTerracottaBrickTexture();
    const terraEmTex = createTerracottaBrickEmissiveTexture();
    const terraWallMat = new THREE.MeshStandardMaterial({
        map: terraTex,
        emissiveMap: terraEmTex,
        emissive: new THREE.Color(0xffffff),
        emissiveIntensity: 0.20,
        roughness: 0.78,
        metalness: 0.08
    });
    terraWallMat.userData = { baseEmissive: 0.20 };
    const terracottaTheme = {
        materials: [terraWallMat, terraWallMat, roofMat, roofMat, terraWallMat, terraWallMat],
        crownColor: 0xfdba74,
        wallMat: terraWallMat,
        baseEmissive: 0.20
    };

    // French Limestone & Travertine Classical Stone Theme
    const limeTex = createLimestoneTexture();
    const limeEmTex = createLimestoneEmissiveTexture();
    const limeWallMat = new THREE.MeshStandardMaterial({
        map: limeTex,
        emissiveMap: limeEmTex,
        emissive: new THREE.Color(0xffffff),
        emissiveIntensity: 0.18,
        roughness: 0.65,
        metalness: 0.15
    });
    limeWallMat.userData = { baseEmissive: 0.18 };
    const limestoneTheme = {
        materials: [limeWallMat, limeWallMat, roofMat, roofMat, limeWallMat, limeWallMat],
        crownColor: 0xfef3c7,
        wallMat: limeWallMat,
        baseEmissive: 0.18
    };

    // Industrial Logistics & Warehouse Theme
    const indTex = createIndustrialTexture();
    const indWallMat = new THREE.MeshStandardMaterial({
        map: indTex,
        roughness: 0.65,
        metalness: 0.40
    });
    const industrialMaterials = [indWallMat, indWallMat, roofMat, roofMat, indWallMat, indWallMat];
    const industrialTheme = {
        materials: industrialMaterials,
        crownColor: 0x64748b,
        wallMat: indWallMat,
        baseEmissive: 0.10
    };

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

        const isCommercialHigh = (p.zoning === "commercial_high");
        const isCommercialMed = (p.zoning === "commercial_medium");
        const isResidential = (p.zoning && p.zoning.startsWith("residential"));

        // Deterministic architectural style hashing based on building coordinates and height
        const seed = Math.abs(Math.sin(center[0] * 12.9898 + center[1] * 78.233 + (p.height_m || 20) * 3.14159) * 43758.5453);
        const rand = seed - Math.floor(seed);

        let selectedTheme = null;
        let isSkyscraper = false;

        if (isCommercialHigh) {
            isSkyscraper = true;
            if (rand < 0.32) selectedTheme = glassThemes.azure_glass;
            else if (rand < 0.54) selectedTheme = glassThemes.bronze_glass;
            else if (rand < 0.74) selectedTheme = glassThemes.silver_tower;
            else if (rand < 0.88) selectedTheme = glassThemes.emerald_glass;
            else selectedTheme = glassThemes.cobalt_tower;
        } else if (isCommercialMed) {
            isSkyscraper = (p.height_m >= 45.0);
            if (rand < 0.36) selectedTheme = limestoneTheme;
            else if (rand < 0.65) selectedTheme = glassThemes.silver_tower;
            else if (rand < 0.84) selectedTheme = glassThemes.azure_glass;
            else selectedTheme = glassThemes.bronze_glass;
        } else if (isResidential) {
            isSkyscraper = (p.height_m >= 50.0);
            if (rand < 0.45) selectedTheme = terracottaTheme;
            else if (rand < 0.75) selectedTheme = limestoneTheme;
            else if (rand < 0.90) selectedTheme = glassThemes.silver_tower;
            else selectedTheme = glassThemes.bronze_glass;
        } else {
            selectedTheme = industrialTheme;
        }

        const bldgMats = selectedTheme.materials;

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

            // Ground Floor Entrance Lobby Core & Canopy
            const canopyW = Math.min(w1 * 0.45, 12.0);
            const canopyD = 2.4;
            const canopyH = 0.32;
            const canopyGeom = new THREE.BoxGeometry(canopyW, canopyH, canopyD);
            const canopyMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.4, metalness: 0.8 });
            const canopy = new THREE.Mesh(canopyGeom, canopyMat);
            canopy.position.set(0, groundY + 2.7, d1 / 2.0 + canopyD / 2.0 - 0.2);
            bldgGroup.add(canopy);

            const lobbyW = Math.min(w1 * 0.42, 11.0);
            const lobbyH = 2.6;
            const lobbyGeom = new THREE.BoxGeometry(lobbyW, lobbyH, 0.4);
            const lobby = new THREE.Mesh(lobbyGeom, lobbyGlassMat);
            lobby.position.set(0, groundY + lobbyH / 2.0, d1 / 2.0 + 0.1);
            bldgGroup.add(lobby);

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
            const parapetH = 0.60;
            const parapetGeom = new THREE.BoxGeometry(w3 * 0.98, parapetH, d3 * 0.98);
            const parapetMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.6 });
            const parapet = new THREE.Mesh(parapetGeom, parapetMat);
            parapet.position.y = roofTopY + parapetH / 2.0;
            bldgGroup.add(parapet);

            // Architectural Crown Accent Trim (Glows with the building's theme color)
            const crownTrimGeom = new THREE.BoxGeometry(w3 * 0.96, 0.15, d3 * 0.96);
            const crownTrimMat = new THREE.MeshStandardMaterial({
                color: selectedTheme.crownColor,
                emissive: selectedTheme.crownColor,
                emissiveIntensity: 0.55,
                roughness: 0.25
            });
            crownTrimMat.userData = { baseEmissive: 0.55 };
            const crownTrim = new THREE.Mesh(crownTrimGeom, crownTrimMat);
            crownTrim.position.y = roofTopY + parapetH + 0.08;
            bldgGroup.add(crownTrim);

            // Central Elevator & Mechanical Penthouse Core Tower ("cors")
            const coreW = Math.max(3.5, w3 * 0.44);
            const coreD = Math.max(3.5, d3 * 0.44);
            const coreH = Math.max(2.8, bh * 0.095);
            const coreOffX = (rand > 0.5 ? 1 : -1) * w3 * 0.12;
            const coreOffZ = (rand > 0.5 ? -1 : 1) * d3 * 0.12;

            const coreGeom = new THREE.BoxGeometry(coreW, coreH, coreD);
            applyBoxFaceUVs(coreGeom, coreW, coreH, coreD, 4.0, 3.0);
            const coreMesh = new THREE.Mesh(coreGeom, coreMaterials);
            coreMesh.position.set(coreOffX, roofTopY + coreH / 2.0, coreOffZ);
            coreMesh.castShadow = true;
            coreMesh.receiveShadow = true;
            bldgGroup.add(coreMesh);

            // Rooftop Machinery Overrun Cap atop Elevator Core
            const capW = coreW * 0.55, capD = coreD * 0.55, capH = 1.2;
            const capGeom = new THREE.BoxGeometry(capW, capH, capD);
            const capMesh = new THREE.Mesh(capGeom, coreMaterials);
            capMesh.position.set(coreOffX, roofTopY + coreH + capH / 2.0, coreOffZ);
            bldgGroup.add(capMesh);

            // Rooftop HVAC Chiller Bank & Dual Cooling Fans (opposite quadrant)
            const chillerW = Math.max(2.2, w3 * 0.30);
            const chillerD = Math.max(2.2, d3 * 0.28);
            const chillerH = 1.4;
            const chillerX = -coreOffX * 0.85;
            const chillerZ = -coreOffZ * 0.85;

            const chillerGeom = new THREE.BoxGeometry(chillerW, chillerH, chillerD);
            const chillerMat = new THREE.MeshStandardMaterial({ color: 0x475569, metalness: 0.7, roughness: 0.35 });
            const chiller = new THREE.Mesh(chillerGeom, chillerMat);
            chiller.position.set(chillerX, roofTopY + chillerH / 2.0, chillerZ);
            bldgGroup.add(chiller);

            // Dual Cylindrical Cooling Fan Cowlings atop Chiller
            const fanGeom = new THREE.CylinderGeometry(0.55, 0.55, 0.35, 12);
            const fanMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, metalness: 0.8, roughness: 0.25 });
            const fan1 = new THREE.Mesh(fanGeom, fanMat);
            fan1.position.set(chillerX - 0.7, roofTopY + chillerH + 0.18, chillerZ);
            bldgGroup.add(fan1);

            const fan2 = new THREE.Mesh(fanGeom, fanMat);
            fan2.position.set(chillerX + 0.7, roofTopY + chillerH + 0.18, chillerZ);
            bldgGroup.add(fan2);

            // Helipad on high-rises (height >= 75m and roof space w3 >= 6.0)
            if (p.height_m >= 75.0 && w3 >= 6.0) {
                const padRadius = Math.min(w3 * 0.40, 6.5);
                const padGeom = new THREE.CylinderGeometry(padRadius, padRadius, 0.25, 24);
                const padSideMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.7 });
                const padMats = [padSideMat, helipadMat, padSideMat];
                const pad = new THREE.Mesh(padGeom, padMats);
                pad.position.set(0, roofTopY + 0.15, 0);
                bldgGroup.add(pad);
            }

            // Communications Spire & FAA Warning Beacon on tall towers (height >= 80m)
            if (p.height_m >= 80.0) {
                const spireH = 8.0 + (p.height_m - 80.0) * 0.25;
                const spireGeom = new THREE.CylinderGeometry(0.12, 0.45, spireH, 8);
                const spireMat = new THREE.MeshStandardMaterial({ color: 0xd1d5db, metalness: 0.85, roughness: 0.2 });
                const spire = new THREE.Mesh(spireGeom, spireMat);
                spire.position.set(coreOffX, roofTopY + coreH + capH + spireH / 2.0, coreOffZ);
                bldgGroup.add(spire);

                // FAA Obstruction Red Warning Beacon Light
                const beaconGeom = new THREE.SphereGeometry(0.40, 8, 8);
                const beaconMat = new THREE.MeshBasicMaterial({ color: 0xff0000 });
                const beacon = new THREE.Mesh(beaconGeom, beaconMat);
                beacon.position.set(coreOffX, roofTopY + coreH + capH + spireH + 0.2, coreOffZ);
                bldgGroup.add(beacon);
                state.aviationBeacons.push(beacon);
            }

        } else if (isResidential) {
            // RESIDENTIAL MULTI-FAMILY & MID-RISE HOUSING
            const g = new THREE.BoxGeometry(bw, bh, bd);
            applyBoxFaceUVs(g, bw, bh, bd, 4.5, 3.2);
            const m = new THREE.Mesh(g, bldgMats);
            m.position.y = groundY + bh / 2.0;
            m.castShadow = true;
            m.receiveShadow = true;
            bldgGroup.add(m);

            // Ground-Floor Entrance Stoop Awning
            const entryW = Math.min(bw * 0.35, 6.0);
            const entryD = 1.8;
            const entryGeom = new THREE.BoxGeometry(entryW, 0.25, entryD);
            const entryMat = new THREE.MeshStandardMaterial({ color: 0x334155, roughness: 0.6 });
            const entry = new THREE.Mesh(entryGeom, entryMat);
            entry.position.set(0, groundY + 2.4, bd / 2.0 + entryD / 2.0 - 0.2);
            bldgGroup.add(entry);

            const roofTopY = groundY + bh;

            // Rooftop Stairwell / Elevator Core Bulkhead ("cors")
            const phW = bw * 0.35;
            const phD = bd * 0.35;
            const phH = 2.4;
            const phGeom = new THREE.BoxGeometry(phW, phH, phD);
            applyBoxFaceUVs(phGeom, phW, phH, phD, 4.0, 3.0);
            const ph = new THREE.Mesh(phGeom, bldgMats);
            ph.position.set(bw * 0.2, roofTopY + phH / 2.0, -bd * 0.2);
            ph.castShadow = true;
            bldgGroup.add(ph);

            // Classic Rooftop Cylindrical Cedar Water Tank on Stilts
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

            // Rooftop AC Compressor Units
            const acGeom = new THREE.BoxGeometry(1.2, 0.9, 1.1);
            const acMat = new THREE.MeshStandardMaterial({ color: 0x475569, metalness: 0.6, roughness: 0.4 });
            const ac1 = new THREE.Mesh(acGeom, acMat);
            ac1.position.set(-bw * 0.2, roofTopY + 0.45, -bd * 0.2);
            bldgGroup.add(ac1);

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
            const ventGeom = new THREE.BoxGeometry(bw * 0.7, 1.3, bd * 0.25);
            const ventMat = new THREE.MeshStandardMaterial({ color: 0x334155, metalness: 0.6, roughness: 0.4 });
            const vent = new THREE.Mesh(ventGeom, ventMat);
            vent.position.set(0, roofTopY + 0.65, 0);
            bldgGroup.add(vent);

            // Loading Dock Roll-Up Doors along front base
            const bayW = Math.min(bw * 0.35, 6.0);
            const bayH = 3.2;
            const bayGeom = new THREE.BoxGeometry(bayW, bayH, 0.3);
            const bayMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, metalness: 0.7, roughness: 0.3 });
            const bay = new THREE.Mesh(bayGeom, bayMat);
            bay.position.set(0, groundY + bayH / 2.0, bd / 2.0 + 0.1);
            bldgGroup.add(bay);
        }

        bldgGroup.userData = { id: p.id, district: p.district, zoning: p.zoning, height: p.height_m, baseEmissive: selectedTheme.baseEmissive || 0.25 };
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

    // 1. Identify Priority Emergency Dispatch Hubs (Level-1 Trauma Centers & EMS Bases)
    const hospitalNodes = [];
    const emsNodes = [];
    if (state.facilities && state.facilities.length > 0) {
        state.facilities.forEach(fac => {
            const p = fac.properties;
            if (p.category === "hospitals" && p.nearest_road_node) {
                hospitalNodes.push(p.nearest_road_node);
            } else if (p.category === "ems_stations" && p.nearest_road_node) {
                emsNodes.push(p.nearest_road_node);
            }
        });
    }
    if (hospitalNodes.length === 0) hospitalNodes.push("INT-0025", "INT-0101");
    if (emsNodes.length === 0) emsNodes.push("INT-0016", "INT-0062", "INT-0118", "INT-0036");

    const allEmergencyNodes = [...hospitalNodes, ...emsNodes];

    // Find outgoing and connected edges directly touching or adjacent to these trauma/EMS nodes
    const emergencyDispatchEdges = [];
    allEmergencyNodes.forEach(nodeId => {
        Object.values(state.roadGraph).forEach(edge => {
            if ((edge.u === nodeId || edge.v === nodeId) && (edge.waterDepth || 0.0) < 0.30) {
                emergencyDispatchEdges.push({ edge: edge, homeNode: nodeId });
            }
        });
    });

    // Supplement with edges within 350m of hospital nodes if sparse
    if (emergencyDispatchEdges.length < 12) {
        hospitalNodes.forEach(hNode => {
            const hPos = state.nodePositions ? state.nodePositions[hNode] : null;
            if (hPos) {
                Object.values(state.roadGraph).forEach(edge => {
                    if (edge.start.distanceTo(hPos) < 350.0 && (edge.waterDepth || 0.0) < 0.30) {
                        emergencyDispatchEdges.push({ edge: edge, homeNode: hNode });
                    }
                });
            }
        });
    }

    const carColors = [0xf59e0b, 0xffffff, 0x38bdf8, 0xe2e8f0, 0x94a3b8, 0x3b82f6, 0x10b981];
    const edgeOccupancy = new Map();
    let spawned = 0;

    // 2. Spawn Dedicated Emergency Fleet (Ambulances & Fire Rescue Trucks from Trauma Hubs)
    const emergencyFleetCount = 12;
    for (let i = 0; i < emergencyFleetCount; i++) {
        let dispatchTarget;
        if (emergencyDispatchEdges.length > 0) {
            dispatchTarget = emergencyDispatchEdges[i % emergencyDispatchEdges.length];
        } else {
            const fallbackEdge = candidateEdges[Math.floor(Math.random() * candidateEdges.length)];
            dispatchTarget = { edge: fallbackEdge, homeNode: hospitalNodes[0] };
        }

        const edge = dispatchTarget.edge;
        const isAmbulance = (i % 2 === 0);
        const emergencyType = isAmbulance ? "ambulance" : "fire_truck";
        const mesh = isAmbulance ? createAmbulanceMesh() : createFireTruckMesh();

        const candidateT = 0.08 + (i * 0.12) % 0.80;
        if (!edgeOccupancy.has(edge.id)) edgeOccupancy.set(edge.id, []);
        edgeOccupancy.get(edge.id).push(candidateT);

        state.trafficGroup.add(mesh);
        state.vehicles.push({
            id: spawned,
            mesh: mesh,
            edgeId: edge.id,
            t: candidateT,
            baseSpeed: 21.0 + Math.random() * 4.0, // High-speed emergency response (80-95 km/h)
            currentSpeed: 21.0,
            isEmergency: true,
            emergencyType: emergencyType,
            waterClearance: isAmbulance ? 0.32 : 0.45,
            homeHospitalNode: dispatchTarget.homeNode,
            missionState: "responding",
            missionTimer: 25.0 + Math.random() * 25.0,
            isStalled: false,
            waitingTimer: 0
        });

        spawned++;
    }

    // 3. Spawn Standard Passenger Cars across Open Dry Network
    let attempts = 0;
    while (spawned < count && attempts < count * 8) {
        attempts++;
        const edge = candidateEdges[Math.floor(Math.random() * candidateEdges.length)];

        if (!edgeOccupancy.has(edge.id)) {
            edgeOccupancy.set(edge.id, []);
        }
        const existingTs = edgeOccupancy.get(edge.id);

        const candidateT = 0.12 + Math.random() * 0.76;
        const minGapT = 4.8 / Math.max(10.0, edge.length);
        const hasOverlap = existingTs.some(t => Math.abs(t - candidateT) < minGapT);

        if (hasOverlap || existingTs.length >= 2) {
            continue;
        }

        existingTs.push(candidateT);
        const color = carColors[Math.floor(Math.random() * carColors.length)];
        const carMesh = createCarMesh(color);
        state.trafficGroup.add(carMesh);

        state.vehicles.push({
            id: spawned,
            mesh: carMesh,
            edgeId: edge.id,
            t: candidateT,
            baseSpeed: 12.0 + Math.random() * 6.0,
            currentSpeed: 12.0,
            isEmergency: false,
            emergencyType: null,
            waterClearance: 0.15,
            isStalled: false,
            waitingTimer: 0
        });

        spawned++;
    }

    state.scene.add(state.trafficGroup);
}

function createAmbulanceMesh() {
    const group = new THREE.Group();

    // 1. Lower Chassis & Wheels frame
    const chassisGeom = new THREE.BoxGeometry(1.36, 0.22, 2.76);
    const chassisMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.8 });
    const chassis = new THREE.Mesh(chassisGeom, chassisMat);
    chassis.position.y = 0.24;
    group.add(chassis);

    // 2. Cab Section (White)
    const cabGeom = new THREE.BoxGeometry(1.30, 0.62, 1.05);
    const whiteMat = new THREE.MeshStandardMaterial({ color: 0xf8fafc, roughness: 0.25, metalness: 0.1 });
    const cab = new THREE.Mesh(cabGeom, whiteMat);
    cab.position.set(0, 0.58, 0.82);
    cab.castShadow = true;
    group.add(cab);

    // 3. Dark Cab Glass / Windshield
    const glassGeom = new THREE.BoxGeometry(1.18, 0.38, 0.58);
    const glassMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.1 });
    const glass = new THREE.Mesh(glassGeom, glassMat);
    glass.position.set(0, 0.90, 0.85);
    group.add(glass);

    // 4. Main Medical Box Module (High-top Patient Compartment)
    const boxGeom = new THREE.BoxGeometry(1.38, 0.98, 1.82);
    const boxMesh = new THREE.Mesh(boxGeom, whiteMat);
    boxMesh.position.set(0, 0.85, -0.45);
    boxMesh.castShadow = true;
    group.add(boxMesh);

    // 5. Bold Emergency Red Reflective Stripes (Left & Right Flanks)
    const redStripeMat = new THREE.MeshStandardMaterial({ color: 0xef4444, roughness: 0.3, emissive: 0x7f1d1d, emissiveIntensity: 0.3 });
    const stripeGeom = new THREE.BoxGeometry(0.04, 0.16, 1.78);
    const leftStripe = new THREE.Mesh(stripeGeom, redStripeMat);
    leftStripe.position.set(-0.70, 0.78, -0.45);
    const rightStripe = new THREE.Mesh(stripeGeom, redStripeMat);
    rightStripe.position.set(0.70, 0.78, -0.45);
    group.add(leftStripe);
    group.add(rightStripe);

    // 6. Medical Red Cross Emblems on Flanks & Rear Door
    const crossVertGeom = new THREE.BoxGeometry(0.03, 0.30, 0.08);
    const crossHorizGeom = new THREE.BoxGeometry(0.03, 0.08, 0.30);
    const crossMat = new THREE.MeshBasicMaterial({ color: 0xdc2626 });

    // Left Cross
    const lCrossV = new THREE.Mesh(crossVertGeom, crossMat);
    lCrossV.position.set(-0.71, 0.92, -0.45);
    const lCrossH = new THREE.Mesh(crossHorizGeom, crossMat);
    lCrossH.position.set(-0.71, 0.92, -0.45);
    group.add(lCrossV);
    group.add(lCrossH);

    // Right Cross
    const rCrossV = new THREE.Mesh(crossVertGeom, crossMat);
    rCrossV.position.set(0.71, 0.92, -0.45);
    const rCrossH = new THREE.Mesh(crossHorizGeom, crossMat);
    rCrossH.position.set(0.71, 0.92, -0.45);
    group.add(rCrossV);
    group.add(rCrossH);

    // Rear Cross
    const rearCrossV = new THREE.BoxGeometry(0.08, 0.28, 0.03);
    const rearCrossH = new THREE.BoxGeometry(0.28, 0.08, 0.03);
    const rcV = new THREE.Mesh(rearCrossV, crossMat);
    rcV.position.set(0.0, 0.88, -1.37);
    const rcH = new THREE.Mesh(rearCrossH, crossMat);
    rcH.position.set(0.0, 0.88, -1.37);
    group.add(rcV);
    group.add(rcH);

    // 7. Headlights & Taillights
    const lightGeom = new THREE.BoxGeometry(0.22, 0.12, 0.08);
    const hlMat = new THREE.MeshBasicMaterial({ color: 0xffedd5 });
    const hl1 = new THREE.Mesh(lightGeom, hlMat);
    hl1.position.set(-0.46, 0.45, 1.36);
    const hl2 = hl1.clone();
    hl2.position.set(0.46, 0.45, 1.36);
    group.add(hl1);
    group.add(hl2);

    const brakeMat = new THREE.MeshBasicMaterial({ color: 0xef4444 });
    const bl1 = new THREE.Mesh(lightGeom, brakeMat);
    bl1.position.set(-0.48, 0.45, -1.37);
    const bl2 = bl1.clone();
    bl2.position.set(0.48, 0.45, -1.37);
    group.add(bl1);
    group.add(bl2);

    // 8. Emergency Siren Lightbar & Beacons (Dual Red/Blue High-Intensity LEDs)
    const barBaseGeom = new THREE.BoxGeometry(0.96, 0.06, 0.22);
    const barBaseMat = new THREE.MeshStandardMaterial({ color: 0x334155, roughness: 0.4 });
    const barBase = new THREE.Mesh(barBaseGeom, barBaseMat);
    barBase.position.set(0, 1.36, 0.65);
    group.add(barBase);

    const beaconGeom = new THREE.BoxGeometry(0.32, 0.14, 0.18);
    const redBeaconMat = new THREE.MeshStandardMaterial({ color: 0xef4444, emissive: 0xff1111, emissiveIntensity: 2.0 });
    const blueBeaconMat = new THREE.MeshStandardMaterial({ color: 0x0284c7, emissive: 0x00f0ff, emissiveIntensity: 2.0 });
    const centerWhiteMat = new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xffffff, emissiveIntensity: 1.5 });

    const leftBeacon = new THREE.Mesh(beaconGeom, redBeaconMat);
    leftBeacon.position.set(-0.34, 1.44, 0.65);
    const rightBeacon = new THREE.Mesh(beaconGeom, blueBeaconMat);
    rightBeacon.position.set(0.34, 1.44, 0.65);
    const centerBeacon = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.12, 0.16), centerWhiteMat);
    centerBeacon.position.set(0, 1.44, 0.65);
    group.add(leftBeacon);
    group.add(rightBeacon);
    group.add(centerBeacon);

    // Rear Roof Corner Flashers
    const cornerGeom = new THREE.BoxGeometry(0.12, 0.12, 0.12);
    const rearLeftBeacon = new THREE.Mesh(cornerGeom, redBeaconMat.clone());
    rearLeftBeacon.position.set(-0.62, 1.36, -1.34);
    const rearRightBeacon = new THREE.Mesh(cornerGeom, blueBeaconMat.clone());
    rearRightBeacon.position.set(0.62, 1.36, -1.34);
    group.add(rearLeftBeacon);
    group.add(rearRightBeacon);

    // 9. Dynamic Siren Road Strobe Light (Ground and Façade Illumination)
    const sirenLight = new THREE.PointLight(0xff1111, 1.5, 14, 2.0);
    sirenLight.position.set(0, 1.8, 0.4);
    group.add(sirenLight);

    group.userData = {
        vehicleType: "ambulance",
        leftBeacons: [leftBeacon, rearLeftBeacon],
        rightBeacons: [rightBeacon, rearRightBeacon],
        centerBeacon: centerBeacon,
        sirenLight: sirenLight,
        tailLights: bl1
    };

    return group;
}

function createFireTruckMesh() {
    const group = new THREE.Group();

    // 1. Heavy Chassis & Bumper Frame
    const chassisGeom = new THREE.BoxGeometry(1.50, 0.28, 3.76);
    const chassisMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.8 });
    const chassis = new THREE.Mesh(chassisGeom, chassisMat);
    chassis.position.y = 0.26;
    group.add(chassis);

    // 2. Heavy Chrome Front Bumper & Grill
    const bumperGeom = new THREE.BoxGeometry(1.52, 0.34, 0.24);
    const chromeMat = new THREE.MeshStandardMaterial({ color: 0xcfd8dc, roughness: 0.15, metalness: 0.8 });
    const bumper = new THREE.Mesh(bumperGeom, chromeMat);
    bumper.position.set(0, 0.36, 1.86);
    group.add(bumper);

    // 3. Front Crew Cab (Fire Engine Red)
    const fireRedMat = new THREE.MeshStandardMaterial({ color: 0xdc2626, roughness: 0.3, metalness: 0.2 });
    const cabGeom = new THREE.BoxGeometry(1.46, 0.84, 1.36);
    const cab = new THREE.Mesh(cabGeom, fireRedMat);
    cab.position.set(0, 0.74, 1.15);
    cab.castShadow = true;
    group.add(cab);

    // 4. Cab Windshield & Side Windows (Dark Glass)
    const glassGeom = new THREE.BoxGeometry(1.34, 0.44, 0.65);
    const glassMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.1 });
    const glass = new THREE.Mesh(glassGeom, glassMat);
    glass.position.set(0, 1.05, 1.25);
    group.add(glass);

    // 5. Heavy Equipment Bay Module (Rear Body)
    const bayGeom = new THREE.BoxGeometry(1.46, 1.12, 2.30);
    const bayMesh = new THREE.Mesh(bayGeom, fireRedMat);
    bayMesh.position.set(0, 0.86, -0.62);
    bayMesh.castShadow = true;
    group.add(bayMesh);

    // 6. Silver Roll-Up Equipment Tool Bays (Left & Right Flanks)
    const bayDoorMat = new THREE.MeshStandardMaterial({ color: 0x94a3b8, roughness: 0.4, metalness: 0.5 });
    const bayDoorGeom = new THREE.BoxGeometry(0.04, 0.68, 1.45);
    const leftDoor = new THREE.Mesh(bayDoorGeom, bayDoorMat);
    leftDoor.position.set(-0.74, 0.85, -0.62);
    const rightDoor = new THREE.Mesh(bayDoorGeom, bayDoorMat);
    rightDoor.position.set(0.74, 0.85, -0.62);
    group.add(leftDoor);
    group.add(rightDoor);

    // 7. Top-Mounted Chrome Heavy Rescue Ladder
    const ladderGroup = new THREE.Group();
    const railGeom = new THREE.BoxGeometry(0.04, 0.06, 2.50);
    const lRail = new THREE.Mesh(railGeom, chromeMat);
    lRail.position.set(-0.35, 1.50, -0.55);
    const rRail = new THREE.Mesh(railGeom, chromeMat);
    rRail.position.set(0.35, 1.50, -0.55);
    ladderGroup.add(lRail);
    ladderGroup.add(rRail);

    const rungGeom = new THREE.BoxGeometry(0.66, 0.03, 0.03);
    for (let rz = -1.6; rz <= 0.5; rz += 0.35) {
        const rung = new THREE.Mesh(rungGeom, chromeMat);
        rung.position.set(0, 1.50, rz);
        ladderGroup.add(rung);
    }
    group.add(ladderGroup);

    // 8. Heavy Dual Headlights & Tail Lights
    const lightGeom = new THREE.BoxGeometry(0.24, 0.14, 0.08);
    const hlMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    const hl1 = new THREE.Mesh(lightGeom, hlMat);
    hl1.position.set(-0.52, 0.48, 1.88);
    const hl2 = hl1.clone();
    hl2.position.set(0.52, 0.48, 1.88);
    group.add(hl1);
    group.add(hl2);

    const brakeMat = new THREE.MeshBasicMaterial({ color: 0xef4444 });
    const bl1 = new THREE.Mesh(lightGeom, brakeMat);
    bl1.position.set(-0.52, 0.48, -1.78);
    const bl2 = bl1.clone();
    bl2.position.set(0.52, 0.48, -1.78);
    group.add(bl1);
    group.add(bl2);

    // 9. Quadruple High-Intensity Siren Beacons (Cab Roof & Rear Upper Corners)
    const beaconGeom = new THREE.BoxGeometry(0.34, 0.16, 0.20);
    const redBeaconMat = new THREE.MeshStandardMaterial({ color: 0xef4444, emissive: 0xff1111, emissiveIntensity: 2.2 });
    const blueBeaconMat = new THREE.MeshStandardMaterial({ color: 0x0284c7, emissive: 0x00f0ff, emissiveIntensity: 2.2 });
    const centerWhiteMat = new THREE.MeshStandardMaterial({ color: 0xffedd5, emissive: 0xffffff, emissiveIntensity: 1.8 });

    const cabLeftBeacon = new THREE.Mesh(beaconGeom, redBeaconMat);
    cabLeftBeacon.position.set(-0.48, 1.58, 1.05);
    const cabRightBeacon = new THREE.Mesh(beaconGeom, blueBeaconMat);
    cabRightBeacon.position.set(0.48, 1.58, 1.05);
    const cabCenterBeacon = new THREE.Mesh(new THREE.BoxGeometry(0.20, 0.14, 0.18), centerWhiteMat);
    cabCenterBeacon.position.set(0, 1.58, 1.05);
    group.add(cabLeftBeacon);
    group.add(cabRightBeacon);
    group.add(cabCenterBeacon);

    const cornerGeom = new THREE.BoxGeometry(0.14, 0.14, 0.14);
    const rearLeftBeacon = new THREE.Mesh(cornerGeom, redBeaconMat.clone());
    rearLeftBeacon.position.set(-0.68, 1.46, -1.74);
    const rearRightBeacon = new THREE.Mesh(cornerGeom, blueBeaconMat.clone());
    rearRightBeacon.position.set(0.68, 1.46, -1.74);
    group.add(rearLeftBeacon);
    group.add(rearRightBeacon);

    // 10. Dynamic Ground Strobe Light
    const sirenLight = new THREE.PointLight(0xff1111, 1.8, 16, 2.0);
    sirenLight.position.set(0, 2.0, 0.8);
    group.add(sirenLight);

    group.userData = {
        vehicleType: "fire_truck",
        leftBeacons: [cabLeftBeacon, rearLeftBeacon],
        rightBeacons: [cabRightBeacon, rearRightBeacon],
        centerBeacon: cabCenterBeacon,
        sirenLight: sirenLight,
        tailLights: bl1
    };

    return group;
}

function createCarMesh(colorHex) {
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

    group.userData = {
        vehicleType: "car",
        tailLights: bl1
    };

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

    // 2. Intra-Lane Car-Following Model (Intelligent Driver Model & Emergency Siren Priority)
    laneGroups.forEach((cars, edgeId) => {
        if (cars.length <= 1) return;
        const edge = state.roadGraph[edgeId];
        if (!edge) return;

        // Sort cars along travel direction: largest t is in front (0 -> 1)
        cars.sort((a, b) => a.t - b.t);

        for (let i = 0; i < cars.length - 1; i++) {
            const follower = cars[i];
            const leader = cars[i + 1];

            const headwayDist = (leader.t - follower.t) * edge.length;
            const safeGap = follower.isEmergency ? 4.2 : 5.0;
            const stopGap = follower.isEmergency ? 3.0 : 3.4;

            // If an emergency vehicle is approaching from behind, civilian car yields / pulls forward
            if (follower.isEmergency && !leader.isEmergency && headwayDist < 12.0) {
                leader.currentSpeed = Math.max(leader.currentSpeed, Math.min(leader.baseSpeed * 1.1, follower.currentSpeed));
            }

            if (headwayDist < safeGap) {
                const speedFactor = Math.max(0.0, (headwayDist - stopGap) / (safeGap - stopGap));
                follower.currentSpeed = Math.min(follower.currentSpeed, leader.currentSpeed * speedFactor);

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
        const clearance = veh.waterClearance || 0.15;

        // Hydrodynamic Passability & Hazard Reaction
        if (!veh.isEmergency) {
            // Passenger Cars: Barricaded at 0.15m water depth
            const isRoadClosed = edge.isClosed || (water >= 0.15);

            if (isRoadClosed) {
                const approachingBarricade = (veh.t >= 0.78);
                const facingWater = (water >= 0.15);

                if (approachingBarricade || facingWater) {
                    veh.currentSpeed = Math.max(0.0, veh.currentSpeed - 14.0 * dt);
                    veh.waitingTimer += dt;

                    // Smooth U-turn into reciprocal opposing lane to retreat back towards dry intersection
                    if (veh.currentSpeed <= 0.3 || veh.waitingTimer > 0.4) {
                        const returnEdgeId = state.reciprocalEdges[veh.edgeId];
                        const returnEdge = returnEdgeId ? state.roadGraph[returnEdgeId] : null;

                        if (returnEdge) {
                            veh.edgeId = returnEdge.id;
                            veh.t = npClamp(1.0 - veh.t, 0.05, 0.95);
                            veh.currentSpeed = veh.baseSpeed * 0.55;
                            veh.isStalled = false;
                            veh.waitingTimer = 0;
                            edge = returnEdge;
                        } else {
                            veh.isStalled = true;
                        }
                    } else {
                        veh.isStalled = true;
                    }
                } else {
                    veh.currentSpeed = Math.min(veh.baseSpeed * 0.70, veh.currentSpeed + 7.0 * dt);
                    veh.isStalled = false;
                    veh.waitingTimer = 0;
                }

                if (water >= 0.30) {
                    veh.currentSpeed = 0.0;
                    veh.isStalled = true;
                }

                // Flash amber hazard lights while stalled
                if (veh.isStalled && veh.mesh.userData.tailLights) {
                    const blink = Math.floor(Date.now() / 320) % 2 === 0;
                    veh.mesh.userData.tailLights.material.color.setHex(blink ? 0xf59e0b : 0xef4444);
                }
            } else if (water >= 0.05) {
                // Shallow Ponding
                veh.currentSpeed = Math.min(veh.currentSpeed, veh.baseSpeed * 0.50);
                veh.isStalled = false;
                veh.waitingTimer = 0;
            } else {
                // Free flow
                if (veh.currentSpeed < veh.baseSpeed * 0.95) {
                    veh.currentSpeed = Math.min(veh.baseSpeed, veh.currentSpeed + 8.0 * dt);
                }
                veh.isStalled = false;
                veh.waitingTimer = 0;
            }
        } else {
            // Specialized Emergency Vehicles (Ambulances & Heavy Fire Engines)
            if (water >= clearance) {
                // Deep water exceeds vehicle clearance: execute tactical U-turn detour
                veh.currentSpeed = Math.max(0.0, veh.currentSpeed - 16.0 * dt);
                veh.waitingTimer += dt;

                if (veh.currentSpeed <= 0.3 || veh.waitingTimer > 0.4) {
                    const returnEdgeId = state.reciprocalEdges[veh.edgeId];
                    const returnEdge = returnEdgeId ? state.roadGraph[returnEdgeId] : null;

                    if (returnEdge) {
                        veh.edgeId = returnEdge.id;
                        veh.t = npClamp(1.0 - veh.t, 0.05, 0.95);
                        veh.currentSpeed = veh.baseSpeed * 0.60;
                        veh.isStalled = false;
                        veh.waitingTimer = 0;
                        edge = returnEdge;
                    } else {
                        veh.isStalled = true;
                    }
                } else {
                    veh.isStalled = true;
                }
            } else if (water >= 0.15) {
                // Road barricaded to civilian cars, but emergency vehicle pushes through flood water!
                veh.currentSpeed = Math.min(veh.baseSpeed * 0.65, veh.currentSpeed + 8.0 * dt);
                veh.isStalled = false;
                veh.waitingTimer = 0;
            } else if (water >= 0.05) {
                // Shallow ponding: fast priority passage
                veh.currentSpeed = Math.min(veh.baseSpeed * 0.82, veh.currentSpeed + 10.0 * dt);
                veh.isStalled = false;
                veh.waitingTimer = 0;
            } else {
                // Dry road: full response speed
                if (veh.currentSpeed < veh.baseSpeed) {
                    veh.currentSpeed = Math.min(veh.baseSpeed, veh.currentSpeed + 12.0 * dt);
                }
                veh.isStalled = false;
                veh.waitingTimer = 0;
            }
        }

        // Kinematic Position Step forward along edge (0 -> 1)
        const stepDist = (veh.currentSpeed * dt) / Math.max(12.0, edge.length);
        veh.t += stepDist;

        // Approaching road intersection (edge endpoint: t >= 1.0)
        if (veh.t >= 1.0) {
            const hitNode = edge.v;
            const oppEdgeId = state.reciprocalEdges[edge.id];

            if (!veh.isEmergency) {
                // Civilian Passenger Cars: pick random open dry outgoing edge
                const availableOutgoing = Object.values(state.roadGraph).filter(e => 
                    e.u === hitNode && 
                    e.id !== oppEdgeId && 
                    !e.isClosed && 
                    (e.waterDepth || 0.0) < 0.05
                );

                if (availableOutgoing.length > 0) {
                    const nextEdge = availableOutgoing[Math.floor(Math.random() * availableOutgoing.length)];
                    veh.edgeId = nextEdge.id;
                    veh.t = 0.02;
                    veh.currentSpeed = veh.baseSpeed * 0.75;
                    veh.isStalled = false;
                    veh.waitingTimer = 0;
                    edge = nextEdge;
                } else {
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
            } else {
                // Emergency Vehicles: Tactical Dynamic Dispatch Routing
                const passableOutgoing = Object.values(state.roadGraph).filter(e => 
                    e.u === hitNode && 
                    e.id !== oppEdgeId && 
                    (e.waterDepth || 0.0) < clearance
                );

                // Update mission lifecycle
                veh.missionTimer -= 4.0;
                if (veh.missionTimer <= 0) {
                    if (veh.missionState === "responding") {
                        veh.missionState = "returning";
                    } else {
                        veh.missionState = "responding";
                        veh.missionTimer = 25.0 + Math.random() * 25.0;
                    }
                }

                let selectedEdge = null;
                if (passableOutgoing.length > 0) {
                    if (veh.missionState === "responding") {
                        // Priority 1: Seek active incident edges (standing water < clearance)
                        const incidentEdges = passableOutgoing.filter(e => (e.waterDepth || 0.0) > 0.02);
                        if (incidentEdges.length > 0) {
                            selectedEdge = incidentEdges[Math.floor(Math.random() * incidentEdges.length)];
                        } else {
                            // Priority 2: Follow fastest arterial open edge
                            selectedEdge = passableOutgoing[Math.floor(Math.random() * passableOutgoing.length)];
                        }
                    } else {
                        // Returning to Level-1 Trauma Center / EMS Base
                        const homePos = (state.nodePositions && veh.homeHospitalNode) ? state.nodePositions[veh.homeHospitalNode] : null;
                        if (homePos) {
                            let bestDist = Infinity;
                            passableOutgoing.forEach(e => {
                                const dist = e.end.distanceTo(homePos);
                                if (dist < bestDist) {
                                    bestDist = dist;
                                    selectedEdge = e;
                                }
                            });
                            // If reached hospital zone (<80m), reload mission to responding
                            if (bestDist < 80.0) {
                                veh.missionState = "responding";
                                veh.missionTimer = 30.0 + Math.random() * 20.0;
                            }
                        } else {
                            selectedEdge = passableOutgoing[Math.floor(Math.random() * passableOutgoing.length)];
                        }
                    }
                }

                if (selectedEdge) {
                    veh.edgeId = selectedEdge.id;
                    veh.t = 0.02;
                    veh.currentSpeed = veh.baseSpeed * 0.85;
                    veh.isStalled = false;
                    veh.waitingTimer = 0;
                    edge = selectedEdge;
                } else {
                    const returnEdgeId = state.reciprocalEdges[edge.id];
                    if (returnEdgeId) {
                        veh.edgeId = returnEdgeId;
                        veh.t = 0.05;
                        veh.currentSpeed = veh.baseSpeed * 0.65;
                        veh.isStalled = false;
                        edge = state.roadGraph[returnEdgeId];
                    } else {
                        veh.currentSpeed = 0.0;
                        veh.isStalled = true;
                        veh.t = 0.98;
                    }
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

        // 5. Dual-Frequency Alternating Siren Strobes & Dynamic Ground Illumination
        if (veh.isEmergency && veh.mesh.userData) {
            const ud = veh.mesh.userData;
            // 4-Phase rapid high-intensity emergency strobe cadence (Left-Left-Right-Right)
            const strobePhase = Math.floor(Date.now() / 90) % 4;
            const isLeftActive = (strobePhase === 0 || strobePhase === 1);
            const isRightActive = (strobePhase === 2 || strobePhase === 3);

            // Left Red Strobes
            if (ud.leftBeacons && ud.leftBeacons.length > 0) {
                const redColor = isLeftActive ? 0xff2222 : 0x3b0707;
                const redEmissive = isLeftActive ? 0xff0000 : 0x000000;
                const redInt = isLeftActive ? 3.5 : 0.0;
                ud.leftBeacons.forEach(b => {
                    b.material.color.setHex(redColor);
                    b.material.emissive.setHex(redEmissive);
                    b.material.emissiveIntensity = redInt;
                });
            }

            // Right Blue Strobes
            if (ud.rightBeacons && ud.rightBeacons.length > 0) {
                const blueColor = isRightActive ? 0x00d9ff : 0x03223f;
                const blueEmissive = isRightActive ? 0x0088ff : 0x000000;
                const blueInt = isRightActive ? 3.5 : 0.0;
                ud.rightBeacons.forEach(b => {
                    b.material.color.setHex(blueColor);
                    b.material.emissive.setHex(blueEmissive);
                    b.material.emissiveIntensity = blueInt;
                });
            }

            // Center Strobe
            if (ud.centerBeacon) {
                const centerActive = (strobePhase === 0 || strobePhase === 2);
                ud.centerBeacon.material.emissiveIntensity = centerActive ? 2.5 : 0.1;
            }

            // Dynamic PointLight Ground Strobe (Casts red/blue pulses on road and buildings)
            if (ud.sirenLight) {
                if (isLeftActive) {
                    ud.sirenLight.color.setHex(0xff1111);
                    ud.sirenLight.intensity = 1.6;
                } else {
                    ud.sirenLight.color.setHex(0x0088ff);
                    ud.sirenLight.intensity = 1.6;
                }
            }
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

async function runScenario(autoPlay = true, scrubToPeak = false) {
    const rainfall = parseFloat(sliderRain.value);
    const duration = parseFloat(sliderDuration.value);
    const drainage = parseFloat(sliderDrain.value) / 100.0;
    const p1 = parseFloat(sliderPump1.value) / 100.0;
    const p2 = parseFloat(sliderPump2.value) / 100.0;

    try {
        elRunScenario.disabled = true;
        const modeLabel = (state.engineMode === "world_model") ? "Neural World Model (2ms)" : "Coupled 2D Physics";
        document.getElementById("runScenarioLabel").innerText = `SIMULATING (${modeLabel})...`;

        const sandbagPairs = [];
        if (state.sandbagBerms) {
            state.sandbagBerms.forEach(item => {
                sandbagPairs.push([item.u, item.v]);
            });
        }

        const payload = {
            rainfall_mm_hr: rainfall,
            duration_hours: duration,
            drainage_capacity_factor: drainage,
            pump_overrides: { "PUMP-1": p1, "PUMP-2": p2 },
            engine_mode: state.engineMode,
            branch_id: state.currentBranch,
            sandbag_edges: sandbagPairs
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

        if (scrubToPeak) {
            // Jump directly to peak storm impact so the user immediately sees the visual consequence of their controls!
            const peakStep = Math.min(24, state.totalSteps);
            scrubToStep(peakStep);
        } else {
            // Start playback from T+00:00 to T+03:00
            scrubToStep(0);
            if (autoPlay) {
                startPlayback();
            }
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
            // Water physically rises above terrain surface only where true surface flooding ponds (clamped to realistic urban basin depth)
            const waterHeight = Math.min(5.5, Math.max(0.06, h * 0.75));
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

    // 8. Flash Warning Beacons on Deployed Sandbag Barriers
    if (state.sandbagsGroup && state.sandbagsGroup.children.length > 0) {
        const amberBlink = Math.floor(Date.now() / 180) % 2 === 0;
        state.sandbagsGroup.children.forEach(berm => {
            if (berm.userData && berm.userData.beacons) {
                const intVal = amberBlink ? 3.0 : 0.3;
                berm.userData.beacons.forEach(b => {
                    b.material.emissiveIntensity = intVal;
                });
                if (berm.userData.bermLight) {
                    berm.userData.bermLight.intensity = amberBlink ? 1.4 : 0.2;
                }
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

    // Sliders with live atmospheric feedback & debounced auto-rollout
    function onSliderLiveChange() {
        if (state.weatherMode === "auto") {
            const r = parseFloat(sliderRain.value);
            state.targetWeatherFactor = npClamp(r / 130.0, 0.0, 1.0);
        }
        if (sliderDebounceTimer) clearTimeout(sliderDebounceTimer);
        sliderDebounceTimer = setTimeout(() => {
            runScenario(false, true); // Auto-rollout and jump directly to peak storm impact!
        }, 450);
    }

    sliderRain.addEventListener("input", (e) => { 
        valRain.innerText = `${e.target.value} mm/hr`; 
        onSliderLiveChange();
    });
    sliderDuration.addEventListener("input", (e) => { 
        valDuration.innerText = `${parseFloat(e.target.value).toFixed(1)} hrs`; 
        onSliderLiveChange();
    });
    sliderDrain.addEventListener("input", (e) => { 
        const cap = parseInt(e.target.value, 10);
        valDrain.innerText = `${cap}% (${100 - cap}% Blocked)`; 
        onSliderLiveChange();
    });
    sliderPump1.addEventListener("input", (e) => { 
        valPump1.innerText = `${e.target.value}%`; 
        onSliderLiveChange();
    });
    sliderPump2.addEventListener("input", (e) => { 
        valPump2.innerText = `${e.target.value}%`; 
        onSliderLiveChange();
    });

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

    // Raycast Inspector & 3D Click Interaction
    const container = document.getElementById("city3dViewport");
    let clickStartX = 0;
    let clickStartY = 0;
    container.addEventListener("mousedown", (e) => {
        clickStartX = e.clientX;
        clickStartY = e.clientY;
        state.isDragging = false;
    });
    container.addEventListener("mousemove", (e) => {
        if (Math.hypot(e.clientX - clickStartX, e.clientY - clickStartY) > 5) {
            state.isDragging = true;
        }
        handle3DMouseMove(e);
    });
    container.addEventListener("mouseup", (e) => {
        if (!state.isDragging && Math.hypot(e.clientX - clickStartX, e.clientY - clickStartY) <= 5) {
            handle3DClick(e);
        }
    });
    container.addEventListener("mouseleave", () => inspectorCard.classList.add("hidden"));

    // Intervention Tool Buttons
    const btnSandbag = document.getElementById("btnToolSandbag");
    const btnPumps = document.getElementById("btnToolBoostPumps");
    const btnClear = document.getElementById("btnClearInterventions");

    if (btnSandbag) {
        btnSandbag.addEventListener("click", () => {
            if (state.activeInterveneTool === "sandbag") {
                state.activeInterveneTool = null;
                btnSandbag.classList.remove("active-sandbag");
                container.classList.remove("intervene-cursor");
                showInterventionToast("ℹ️", "Sandbag placement tool deactivated.");
            } else {
                state.activeInterveneTool = "sandbag";
                btnSandbag.classList.add("active-sandbag");
                container.classList.add("intervene-cursor");
                showInterventionToast("🚧", "Sandbag tool active! Click any road in 3D to deploy barriers.");
            }
        });
    }

    if (btnPumps) {
        btnPumps.addEventListener("click", () => {
            // Toggle global Turbo Pump Boost
            const isCurrentlyBoosted = (parseFloat(sliderPump1.value) > 100.0);
            const targetVal = isCurrentlyBoosted ? 100.0 : 150.0;
            sliderPump1.value = targetVal;
            valPump1.innerText = `${Math.round(targetVal)}%`;
            sliderPump2.value = targetVal;
            valPump2.innerText = `${Math.round(targetVal)}%`;

            if (!isCurrentlyBoosted) {
                btnPumps.classList.add("active-pump");
                showInterventionToast("⚡", "ALL PUMPS TURBO-BOOSTED (15.0 m³/s): Rapid flood drawdown active!");
                if (state.facilitiesGroup) {
                    state.facilitiesGroup.children.forEach(f => {
                        if (f.userData && f.userData.category === "pumping_stations") {
                            spawnInterventionShockwave(f.position, 0x10b981);
                        }
                    });
                }
            } else {
                btnPumps.classList.remove("active-pump");
                showInterventionToast("🔄", "Pumps reset to rated capacity (100%).");
            }
            updateClearButtonVisibility();
            debounceAutoRollout(150);
        });
    }

    if (btnClear) {
        btnClear.addEventListener("click", clearAllInterventions);
    }
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

        // Run full scenario automatically and jump directly to peak storm impact
        await runScenario(false, true);
    } catch (err) {
        console.error("NLP execute error:", err);
    }
}


// =========================================================================
// INTERACTIVE 3D CLICK-TO-INTERVENE & COUNTERFACTUAL BRANCHING
// =========================================================================

let toastTimeout = null;
function showInterventionToast(icon, message, durationMs = 3800) {
    const toast = document.getElementById("interventionToast");
    const iconEl = document.getElementById("toastIcon");
    const msgEl = document.getElementById("toastMsg");
    if (!toast || !iconEl || !msgEl) return;

    iconEl.innerText = icon;
    msgEl.innerText = message;
    toast.classList.remove("hidden");

    if (toastTimeout) clearTimeout(toastTimeout);
    toastTimeout = setTimeout(() => {
        toast.classList.add("hidden");
    }, durationMs);
}

function spawnInterventionShockwave(position, colorHex = 0x10b981) {
    if (!state.scene) return;
    const ringGeom = new THREE.RingGeometry(0.6, 1.8, 32);
    const ringMat = new THREE.MeshBasicMaterial({
        color: colorHex,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.95
    });
    const ring = new THREE.Mesh(ringGeom, ringMat);
    ring.rotation.x = -Math.PI / 2;
    ring.position.copy(position);
    ring.position.y += 0.55;
    state.scene.add(ring);

    let progress = 0;
    const animInterval = setInterval(() => {
        progress += 0.045;
        const scale = 1.0 + progress * 24.0;
        ring.scale.set(scale, scale, scale);
        ring.material.opacity = Math.max(0, 0.95 - progress * 1.05);
        if (progress >= 1.0) {
            clearInterval(animInterval);
            state.scene.remove(ring);
            ring.geometry.dispose();
            ring.material.dispose();
        }
    }, 25);
}

function updateClearButtonVisibility() {
    const btnClear = document.getElementById("btnClearInterventions");
    if (!btnClear) return;
    const count = (state.sandbagBerms ? state.sandbagBerms.size : 0);
    const p1Boost = parseFloat(sliderPump1.value) > 100.0;
    const p2Boost = parseFloat(sliderPump2.value) > 100.0;

    if (count > 0 || p1Boost || p2Boost) {
        btnClear.style.display = "inline-block";
        btnClear.innerText = `🧹 Clear (${count + (p1Boost ? 1 : 0) + (p2Boost ? 1 : 0)})`;
    } else {
        btnClear.style.display = "none";
    }
}

function createSandbagBermMesh(edge) {
    const group = new THREE.Group();

    const start = edge.start;
    const end = edge.end;
    const mid = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
    const forward = new THREE.Vector3().subVectors(end, start).normalize();

    const roadWidth = edge.width + 1.4;

    // 1. Lower Layer of Canvas Burlap Sandbags
    const bagMat = new THREE.MeshStandardMaterial({
        color: 0x92400e, // Warm earthen burlap tone
        roughness: 0.92,
        metalness: 0.08
    });

    const lowerGeom = new THREE.BoxGeometry(roadWidth, 0.44, 0.72);
    const lowerBerm = new THREE.Mesh(lowerGeom, bagMat);
    lowerBerm.position.y = 0.22;
    lowerBerm.castShadow = true;
    group.add(lowerBerm);

    // 2. Upper Layer of Sandbags (nested on top for pyramid interlocking berm profile)
    const upperGeom = new THREE.BoxGeometry(roadWidth * 0.92, 0.40, 0.54);
    const upperBerm = new THREE.Mesh(upperGeom, bagMat);
    upperBerm.position.y = 0.58;
    upperBerm.castShadow = true;
    group.add(upperBerm);

    // 3. High-Visibility Caution Hazard Crest Tape (Black and Amber Chevron)
    const tapeGeom = new THREE.BoxGeometry(roadWidth * 0.94, 0.08, 0.56);
    const tapeMat = new THREE.MeshStandardMaterial({
        color: 0xf59e0b,
        emissive: 0x78350f,
        emissiveIntensity: 0.4,
        roughness: 0.3
    });
    const tape = new THREE.Mesh(tapeGeom, tapeMat);
    tape.position.y = 0.80;
    group.add(tape);

    // 4. Strobe Warning Hazard Beacons on Left & Right Flanks
    const beaconGeom = new THREE.CylinderGeometry(0.12, 0.14, 0.22, 8);
    const amberBeaconMat = new THREE.MeshStandardMaterial({
        color: 0xf59e0b,
        emissive: 0xffb703,
        emissiveIntensity: 2.5
    });

    const leftBeacon = new THREE.Mesh(beaconGeom, amberBeaconMat);
    leftBeacon.position.set(-roadWidth * 0.44, 0.90, 0);
    const rightBeacon = new THREE.Mesh(beaconGeom, amberBeaconMat.clone());
    rightBeacon.position.set(roadWidth * 0.44, 0.90, 0);
    group.add(leftBeacon);
    group.add(rightBeacon);

    // 5. Dynamic Local Amber Strobe PointLight
    const bermLight = new THREE.PointLight(0xf59e0b, 1.2, 9, 2.0);
    bermLight.position.set(0, 1.2, 0);
    group.add(bermLight);

    // Position and Orient group across the road
    group.position.copy(mid);
    const lookTarget = mid.clone().add(forward);
    group.lookAt(lookTarget);

    group.userData = {
        isSandbagBerm: true,
        streetKey: edge.streetKey,
        edge: edge,
        beacons: [leftBeacon, rightBeacon],
        bermLight: bermLight
    };

    return group;
}

function handleRoadIntervention(streetKey) {
    if (state.sandbagBerms.has(streetKey)) {
        removeSandbagBerm(streetKey);
    } else {
        deploySandbagBerm(streetKey);
    }
}

function deploySandbagBerm(streetKey) {
    const candidateEdge = Object.values(state.roadGraph).find(e => e.streetKey === streetKey);
    if (!candidateEdge) return;

    const bermMesh = createSandbagBermMesh(candidateEdge);
    state.sandbagsGroup.add(bermMesh);

    state.sandbagBerms.set(streetKey, {
        mesh: bermMesh,
        edge: candidateEdge,
        u: candidateEdge.u,
        v: candidateEdge.v
    });

    candidateEdge.hasSandbag = true;
    const oppId = state.reciprocalEdges[candidateEdge.id];
    if (oppId && state.roadGraph[oppId]) {
        state.roadGraph[oppId].hasSandbag = true;
    }

    updateClearButtonVisibility();
    showInterventionToast("🚧", `Sandbag Barrier Deployed on ${streetKey} (+0.85m Defense)! Rerolling hydrodynamics...`);
    spawnInterventionShockwave(candidateEdge.start.clone().add(candidateEdge.end).multiplyScalar(0.5), 0xf59e0b);

    if (sliderDebounceTimer) clearTimeout(sliderDebounceTimer);
    sliderDebounceTimer = setTimeout(() => {
        runScenario(false, true);
    }, 180);
}

function removeSandbagBerm(streetKey) {
    if (!state.sandbagBerms.has(streetKey)) return;
    const item = state.sandbagBerms.get(streetKey);
    state.sandbagsGroup.remove(item.mesh);
    state.sandbagBerms.delete(streetKey);

    item.edge.hasSandbag = false;
    const oppId = state.reciprocalEdges[item.edge.id];
    if (oppId && state.roadGraph[oppId]) {
        state.roadGraph[oppId].hasSandbag = false;
    }

    updateClearButtonVisibility();
    showInterventionToast("🧹", `Sandbag Barrier Removed from ${streetKey}. Recalculating runoff...`);

    if (sliderDebounceTimer) clearTimeout(sliderDebounceTimer);
    sliderDebounceTimer = setTimeout(() => {
        runScenario(false, true);
    }, 180);
}

function handleFacilityIntervention(hitFac) {
    const cat = hitFac.userData.category;
    const facId = hitFac.userData.id;
    const pos = hitFac.position;

    if (cat === "pumping_stations") {
        // Toggle Pump Turbo Throttle (100% -> 150% Turbo Boost)
        const isPump1 = facId.includes("1") || facId.toLowerCase().includes("south");
        const slider = isPump1 ? sliderPump1 : sliderPump2;
        const valEl = isPump1 ? valPump1 : valPump2;

        const currentVal = parseFloat(slider.value);
        let newVal;
        if (currentVal <= 100.0) {
            newVal = 150.0;
            showInterventionToast("⚡", `${facId}: TURBO-BOOST (15.0 m³/s) Activated! Drainage enhanced.`);
            spawnInterventionShockwave(pos, 0x10b981);
        } else {
            newVal = 100.0;
            showInterventionToast("🔄", `${facId}: Reset to Standard Operational Throttle (10.0 m³/s).`);
            spawnInterventionShockwave(pos, 0x00f0ff);
        }

        slider.value = newVal;
        valEl.innerText = `${Math.round(newVal)}%`;
        updateClearButtonVisibility();

        if (sliderDebounceTimer) clearTimeout(sliderDebounceTimer);
        sliderDebounceTimer = setTimeout(() => {
            runScenario(false, true);
        }, 180);

    } else if (cat === "electrical_substations") {
        showInterventionToast("🔋", `${facId}: Auxiliary Generator Hardening Deployed (Immune to Flood Trip)!`);
        spawnInterventionShockwave(pos, 0x00f0ff);

    } else if (cat === "hospitals") {
        showInterventionToast("🚑", `${facId}: Emergency Surge Activated — 4 Extra Ambulances Dispatched!`);
        spawnInterventionShockwave(pos, 0xef4444);
        for (let k = 0; k < 4; k++) {
            const amb = createAmbulanceMesh();
            state.trafficGroup.add(amb);
            const edge = Object.values(state.roadGraph).find(e => e.u === "INT-0025" || e.v === "INT-0025") || Object.values(state.roadGraph)[0];
            state.vehicles.push({
                id: state.vehicles.length,
                mesh: amb,
                edgeId: edge.id,
                t: 0.1 + k * 0.2,
                baseSpeed: 23.0,
                currentSpeed: 23.0,
                isEmergency: true,
                emergencyType: "ambulance",
                waterClearance: 0.32,
                homeHospitalNode: "INT-0025",
                missionState: "responding",
                missionTimer: 35.0,
                isStalled: false,
                waitingTimer: 0
            });
        }
    }
}

function clearAllInterventions() {
    state.sandbagBerms.forEach(item => {
        state.sandbagsGroup.remove(item.mesh);
        item.edge.hasSandbag = false;
        const oppId = state.reciprocalEdges[item.edge.id];
        if (oppId && state.roadGraph[oppId]) {
            state.roadGraph[oppId].hasSandbag = false;
        }
    });
    state.sandbagBerms.clear();

    sliderPump1.value = 100;
    valPump1.innerText = "100%";
    sliderPump2.value = 100;
    valPump2.innerText = "100%";

    const btnPumps = document.getElementById("btnToolBoostPumps");
    if (btnPumps) btnPumps.classList.remove("active-pump");

    updateClearButtonVisibility();
    showInterventionToast("🧹", "All sandbag barriers and pump overrides reset to baseline.");

    if (sliderDebounceTimer) clearTimeout(sliderDebounceTimer);
    sliderDebounceTimer = setTimeout(() => {
        runScenario(false, true);
    }, 180);
}

function handle3DClick(e) {
    if (!state.renderer || !state.camera) return;
    const rect = state.renderer.domElement.getBoundingClientRect();
    mouseVec.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    mouseVec.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouseVec, state.camera);

    // 1. Check if user clicked an existing Sandbag Berm
    if (state.sandbagsGroup && state.sandbagsGroup.children.length > 0) {
        const bermIntersects = raycaster.intersectObjects(state.sandbagsGroup.children, true);
        if (bermIntersects.length > 0) {
            let hitBerm = bermIntersects[0].object;
            while (hitBerm && (!hitBerm.userData || !hitBerm.userData.isSandbagBerm) && hitBerm.parent) {
                hitBerm = hitBerm.parent;
            }
            if (hitBerm && hitBerm.userData && hitBerm.userData.isSandbagBerm) {
                const sKey = hitBerm.userData.streetKey;
                removeSandbagBerm(sKey);
                return;
            }
        }
    }

    // 2. Check if user clicked a Critical Facility (Pump Station, Substation, Hospital)
    if (state.facilitiesGroup && state.facilitiesGroup.children.length > 0) {
        const facIntersects = raycaster.intersectObjects(state.facilitiesGroup.children, true);
        if (facIntersects.length > 0) {
            let hitFac = facIntersects[0].object;
            while (hitFac && (!hitFac.userData || !hitFac.userData.category) && hitFac.parent) {
                hitFac = hitFac.parent;
            }
            if (hitFac && hitFac.userData && hitFac.userData.category) {
                handleFacilityIntervention(hitFac);
                return;
            }
        }
    }

    // 3. Check if user clicked a Road to deploy sandbags
    if (state.roadLinesGroup && state.roadLinesGroup.children.length > 0) {
        const roadIntersects = raycaster.intersectObjects(state.roadLinesGroup.children, true);
        if (roadIntersects.length > 0) {
            let hitRoad = roadIntersects[0].object;
            while (hitRoad && (!hitRoad.userData || !hitRoad.userData.streetKey) && hitRoad.parent) {
                hitRoad = hitRoad.parent;
            }
            if (hitRoad && hitRoad.userData && hitRoad.userData.streetKey) {
                handleRoadIntervention(hitRoad.userData.streetKey);
                return;
            }
        }
    }
}

async function handleForkScenario() {
    const branchName = `intervention_branch_B`;
    try {
        const sandbagPairs = [];
        state.sandbagBerms.forEach(item => {
            sandbagPairs.push([item.u, item.v]);
        });
        const p1 = parseFloat(sliderPump1.value) / 100.0;
        const p2 = parseFloat(sliderPump2.value) / 100.0;

        const res = await fetch("/api/branch/fork", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                source_branch_id: "baseline",
                new_branch_id: branchName,
                intervention: {
                    pump_overrides: { "PUMP-1": p1, "PUMP-2": p2 },
                    drainage_capacity_factor: parseFloat(sliderDrain.value) / 100.0
                }
            })
        });

        if (res.ok) {
            state.forkedBranch = branchName;
            state.currentBranch = branchName;
            document.getElementById("currentBranchDisplay").innerText = branchName;
            btnCompare.disabled = false;
            btnFork.innerText = "RE-FORK BRANCH B";

            showInterventionToast("🔀", `Forked timeline '${branchName}' with ${sandbagPairs.length} sandbag barriers & pump boosts.`);

            // Re-roll scenario for Branch B with all active interventions!
            await runScenario(false, true);
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
            recBadge.innerText = "COUNTERMEASURE RECOMMENDED: Sandbag barriers & turbo pumping reduced flood inundation and preserved emergency trauma routes!";
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

        // 1. Proximity detection for active emergency vehicles (within 6 meters)
        let nearbyEmergency = null;
        if (state.vehicles && state.vehicles.length > 0) {
            nearbyEmergency = state.vehicles.find(v => v.isEmergency && v.mesh && v.mesh.position.distanceTo(pt) < 6.0);
        }

        let emergencyFleetRow = "";
        if (nearbyEmergency) {
            const badgeType = nearbyEmergency.emergencyType === "fire_truck" ? "🚒 Heavy Fire Rescue Squad" : "🚑 Trauma Life-Support Ambulance";
            const missionTxt = nearbyEmergency.missionState === "responding" ? "PRIORITY DISPATCH (Incident Response)" : "PATIENT TRANSPORT (Returning to Hospital)";
            emergencyFleetRow = `<div class="inspector-row" style="border-top:1px dashed rgba(239,68,68,0.4); margin-top:5px; padding-top:4px;"><span style="color:#ef4444;font-weight:600">${badgeType}</span><strong style="color:#00f0ff">${missionTxt}</strong></div>`;
        }

        // 2. Proximity detection for Sandbag Berms (within 10 meters)
        let sandbagRow = "";
        if (state.sandbagsGroup && state.sandbagsGroup.children.length > 0) {
            const nearBerm = state.sandbagsGroup.children.find(b => b.position.distanceTo(pt) < 10.0);
            if (nearBerm) {
                sandbagRow = `<div class="inspector-row" style="border-top:1px dashed rgba(245,158,11,0.5); margin-top:5px; padding-top:4px;"><span style="color:#f59e0b;font-weight:700">🚧 ACTIVE SANDBAG BERM</span><strong style="color:#f59e0b">Click to Remove</strong></div>`;
            }
        }

        // 3. Proximity detection for Critical Facilities (within 18 meters)
        let facilityInterveneRow = "";
        if (state.facilitiesGroup && state.facilitiesGroup.children.length > 0) {
            const nearFac = state.facilitiesGroup.children.find(f => f.userData && f.userData.category && f.position.distanceTo(pt) < 18.0);
            if (nearFac) {
                const cat = nearFac.userData.category;
                const name = nearFac.userData.id;
                if (cat === "pumping_stations") {
                    facilityInterveneRow = `<div class="inspector-row" style="border-top:1px dashed rgba(16,185,129,0.5); margin-top:5px; padding-top:4px;"><span style="color:#10b981;font-weight:700">⚡ ${name}</span><strong style="color:#10b981">Click to Turbo-Boost (15 m³/s)</strong></div>`;
                } else if (cat === "electrical_substations") {
                    facilityInterveneRow = `<div class="inspector-row" style="border-top:1px dashed rgba(0,240,255,0.5); margin-top:5px; padding-top:4px;"><span style="color:#00f0ff;font-weight:700">🔋 ${name}</span><strong style="color:#00f0ff">Click to Deploy Backup Gen</strong></div>`;
                } else if (cat === "hospitals") {
                    facilityInterveneRow = `<div class="inspector-row" style="border-top:1px dashed rgba(239,68,68,0.5); margin-top:5px; padding-top:4px;"><span style="color:#ef4444;font-weight:700">🏥 ${name}</span><strong style="color:#ef4444">Click for Ambulance Surge</strong></div>`;
                }
            }
        }

        let roadInterveneRow = "";
        if (!sandbagRow && !facilityInterveneRow) {
            roadInterveneRow = `<div class="inspector-row" style="color:#94a3b8;font-size:10px;margin-top:4px;border-top:1px dotted rgba(255,255,255,0.15);padding-top:3px;"><span>💡 Action:</span><em style="color:#f59e0b">Click road to deploy sandbags</em></div>`;
        }

        inspectorCard.classList.remove("hidden");
        inspectorContent.innerHTML = `
            <div class="inspector-row"><span>3D Coordinates:</span><strong>(${wx}m, ${wy}m)</strong></div>
            <div class="inspector-row"><span>Ground Elevation:</span><strong>${elev.toFixed(1)} m</strong></div>
            <div class="inspector-row"><span>Water Inundation:</span><strong style="color: ${water > 0.15 ? 'var(--danger-red)' : (water > 0.04 ? 'var(--accent-cyan)' : 'var(--text-primary)')}">${(water * 100).toFixed(1)} cm</strong></div>
            <div class="inspector-row"><span>Road Traffic:</span><strong style="color: ${water >= 0.15 ? 'var(--danger-red)' : 'var(--success-green)'}">${water >= 0.15 ? 'STALLED (Flooded)' : 'FLOWING (Open)'}</strong></div>
            <div class="inspector-row"><span>Emergency Route:</span><strong style="color: ${water >= 0.30 ? 'var(--danger-red)' : 'var(--success-green)'}">${water >= 0.30 ? 'IMPASSABLE' : 'PASSABLE'}</strong></div>
            ${emergencyFleetRow}
            ${sandbagRow}
            ${facilityInterveneRow}
            ${roadInterveneRow}
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
    kpiClosedEmergency.innerText = `${closedEmerg} emergency severed | 12 EMS Fleet Active`;

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
