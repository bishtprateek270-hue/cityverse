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
    flashLight: null,

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

        // 4. Spawn Moving Traffic Fleet
        spawnTrafficFleet(180);

        // 5. Setup UI & Camera Listeners
        setupEventListeners();

        // 6. Fast Load Pre-Computed Simulation Trajectory
        const loaded = await loadCachedFrames();
        if (!loaded) {
            await runScenario(false);
        }

        // 7. Start Animation & Physics Render Loop
        animate();

        console.log("CityVerse 3D initialized successfully.");
    } catch (err) {
        console.error("Initialization error:", err);
    }
}


// =========================================================================
// THREE.JS 3D SCENE & LIGHTING SETUP
// =========================================================================

function initThreeScene() {
    const container = document.getElementById("city3dViewport");
    const width = container.clientWidth || window.innerWidth;
    const height = container.clientHeight || (window.innerHeight - 150);

    // Scene with atmospheric fog
    state.scene = new THREE.Scene();
    state.scene.background = new THREE.Color(0x060a14);
    state.scene.fog = new THREE.FogExp2(0x060a14, 0.0014);

    // Perspective Camera
    state.camera = new THREE.PerspectiveCamera(45, width / height, 1.0, 2500);
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
    state.controls.maxDistance = 850;
    state.controls.target.set(0, 15, 0);

    // Ambient Lighting
    const hemiLight = new THREE.HemisphereLight(0x7dd3fc, 0x0f172a, 0.6);
    state.scene.add(hemiLight);

    // Directional Sunlight with Shadows
    state.dirLight = new THREE.DirectionalLight(0xffedd5, 1.3);
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
        const isRiver = state.terrain.river_mask[r][c];

        // Vertical elevation scaling
        let y = (elev - minElev) * 0.55;
        if (isRiver) y -= 1.6; // Carve natural river channel below ground

        pos.setY(i, y);

        // Hypsometric terrain vertex colors
        const norm = (elev - minElev) / (maxElev - minElev);
        let cr, cg, cb;
        if (isRiver) {
            // Dark riverbed silt
            cr = 0.08; cg = 0.12; cb = 0.16;
        } else if (norm < 0.25) {
            // Lowland valley slate-green
            cr = 0.12 + norm * 0.1; cg = 0.18 + norm * 0.15; cb = 0.16 + norm * 0.1;
        } else if (norm < 0.65) {
            // Mid-elevation urban asphalt & loam
            cr = 0.18 + norm * 0.1; cg = 0.22 + norm * 0.1; cb = 0.24 + norm * 0.1;
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
    for (let i = 0; i < pos.count; i++) {
        pos.setY(i, -100); // Submerged initially until water is added
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

    state.roads.forEach(feat => {
        const coords = feat.geometry.coordinates;
        const p = feat.properties;
        const isHighway = (p.road_class === "highway");
        const isBridge = !!p.is_bridge;

        const p1 = worldTo3D(coords[0][0], coords[0][1]);
        const p2 = worldTo3D(coords[1][0], coords[1][1]);

        // Sample elevation at road ends
        let y1 = getTerrainElevation(coords[0][0], coords[0][1]) + 0.35;
        let y2 = getTerrainElevation(coords[1][0], coords[1][1]) + 0.35;

        if (isBridge) {
            // Elevate bridge deck cleanly above riverbed and channel water
            const bridgeDeckY = Math.max(y1, y2) + 2.5;
            y1 = bridgeDeckY;
            y2 = bridgeDeckY;
        }

        const startVec = new THREE.Vector3(p1.x, y1, p1.z);
        const endVec = new THREE.Vector3(p2.x, y2, p2.z);

        // Store edge in traffic routing graph
        const length = startVec.distanceTo(endVec);
        // Realistic Two-Way Road Widths
        const width = isHighway ? 4.6 : (p.road_class === "arterial" ? 3.4 : 2.6);

        state.roadGraph[p.edge_id] = {
            id: p.edge_id,
            start: startVec,
            end: endVec,
            startWorld: { x: coords[0][0], y: coords[0][1] },
            endWorld: { x: coords[1][0], y: coords[1][1] },
            u: p.u,
            v: p.v,
            length: length,
            speedLimit: p.free_flow_speed_kmh || 40.0,
            roadClass: p.road_class,
            width: width,
            isBridge: isBridge,
            isClosed: false,
            waterDepth: 0.0
        };

        // 3D Road Ribbon with Dual-Lane Markings & Center Divider Line
        const roadGroup = createRoadRibbon(startVec, endVec, width, isHighway ? 0x94a3b8 : 0x334155);
        roadGroup.userData = { 
            edgeId: p.edge_id, 
            baseColor: isHighway ? 0x94a3b8 : 0x334155, 
            width: width,
            asphalt: roadGroup.userData.asphalt,
            centerLine: roadGroup.userData.centerLine
        };
        state.roadLinesGroup.add(roadGroup);

        // If it's a bridge, add bridge support piers underneath
        if (isBridge) {
            const bridgePillars = createBridgeSupportPillars(startVec, endVec);
            state.roadLinesGroup.add(bridgePillars);
        }

        // Roadblock Barrier Pair (placed at both entrances of road segment)
        const barrierPair = createRoadBarricadePair(startVec, endVec, width);
        barrierPair.userData = { edgeId: p.edge_id };
        barrierPair.visible = false;
        state.barriersGroup.add(barrierPair);
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

function createRoadRibbon(p1, p2, width, colorHex) {
    const group = new THREE.Group();
    const dir = new THREE.Vector3().subVectors(p2, p1).normalize();
    const up = new THREE.Vector3(0, 1, 0);
    const side = new THREE.Vector3().crossVectors(dir, up).normalize().multiplyScalar(width / 2.0);

    // 1. Asphalt Road Surface Quad
    const v1 = new THREE.Vector3().subVectors(p1, side);
    const v2 = new THREE.Vector3().addVectors(p1, side);
    const v3 = new THREE.Vector3().addVectors(p2, side);
    const v4 = new THREE.Vector3().subVectors(p2, side);

    const geom = new THREE.BufferGeometry();
    const vertices = new Float32Array([
        v1.x, v1.y, v1.z,  v2.x, v2.y, v2.z,  v3.x, v3.y, v3.z,
        v1.x, v1.y, v1.z,  v3.x, v3.y, v3.z,  v4.x, v4.y, v4.z
    ]);
    geom.setAttribute("position", new THREE.BufferAttribute(vertices, 3));
    geom.computeVertexNormals();

    const mat = new THREE.MeshStandardMaterial({
        color: colorHex,
        roughness: 0.75,
        metalness: 0.1
    });

    const asphaltMesh = new THREE.Mesh(geom, mat);
    asphaltMesh.receiveShadow = true;
    group.add(asphaltMesh);

    // 2. High-Visibility Yellow Centerline Divider (Two-Way Street Separation)
    // Offset slightly (+0.02) to prevent z-fighting with asphalt deck
    const cWidth = 0.14;
    const cSide = side.clone().normalize().multiplyScalar(cWidth / 2.0);
    const cv1 = new THREE.Vector3().subVectors(p1, cSide); cv1.y += 0.02;
    const cv2 = new THREE.Vector3().addVectors(p1, cSide); cv2.y += 0.02;
    const cv3 = new THREE.Vector3().addVectors(p2, cSide); cv3.y += 0.02;
    const cv4 = new THREE.Vector3().subVectors(p2, cSide); cv4.y += 0.02;

    const cGeom = new THREE.BufferGeometry();
    const cVertices = new Float32Array([
        cv1.x, cv1.y, cv1.z,  cv2.x, cv2.y, cv2.z,  cv3.x, cv3.y, cv3.z,
        cv1.x, cv1.y, cv1.z,  cv3.x, cv3.y, cv3.z,  cv4.x, cv4.y, cv4.z
    ]);
    cGeom.setAttribute("position", new THREE.BufferAttribute(cVertices, 3));
    cGeom.computeVertexNormals();

    const cMat = new THREE.MeshBasicMaterial({
        color: 0xf59e0b, // Yellow center divider line
        transparent: true,
        opacity: 0.85
    });
    const centerLine = new THREE.Mesh(cGeom, cMat);
    group.add(centerLine);

    group.userData = { asphalt: asphaltMesh, centerLine: centerLine };
    return group;
}

function createRoadBarricadePair(p1, p2, width) {
    const group = new THREE.Group();
    const len = p1.distanceTo(p2);
    const dir = new THREE.Vector3().subVectors(p2, p1).normalize();

    if (len < 12.0) {
        // Single barricade at midpoint for very short edges
        const mid = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);
        const b = createSingleBarricade(width, dir);
        b.position.copy(mid);
        group.add(b);
    } else {
        // Dual barricades positioned at both entrances of the road segment
        const offset = Math.min(3.5, len * 0.12);

        const pos1 = new THREE.Vector3().addVectors(p1, dir.clone().multiplyScalar(offset));
        const b1 = createSingleBarricade(width, dir);
        b1.position.copy(pos1);
        group.add(b1);

        const pos2 = new THREE.Vector3().subVectors(p2, dir.clone().multiplyScalar(offset));
        const b2 = createSingleBarricade(width, dir);
        b2.position.copy(pos2);
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

    state.footprints.forEach(bldg => {
        const p = bldg.properties;
        const center = p.center;

        const p3d = worldTo3D(center[0], center[1]);
        const bw = Math.max(2.5, (p.width_m || 25.0) * SCALE);
        const bd = Math.max(2.5, (p.depth_m || 25.0) * SCALE);
        const bh = Math.max(4.0, (p.height_m || 30.0) * 0.45);

        const groundY = (p.ground_elev_m - state.terrain.min_elevation_m) * 0.55;

        const geom = new THREE.BoxGeometry(bw, bh, bd);

        // Architectural zoning styling
        let color = 0x1e293b;
        let emissive = 0x000000;
        let emissiveIntensity = 0.0;

        if (p.zoning === "commercial_high") {
            // Downtown glass skyscraper
            color = 0x0f172a;
            emissive = 0x38bdf8;
            emissiveIntensity = 0.25;
        } else if (p.zoning === "residential_high" || p.zoning === "residential_medium") {
            color = 0x334155;
            emissive = 0xfef08a;
            emissiveIntensity = 0.15;
        } else if (p.zoning === "industrial_heavy") {
            color = 0x262626;
        }

        const mat = new THREE.MeshStandardMaterial({
            color: color,
            emissive: emissive,
            emissiveIntensity: emissiveIntensity,
            roughness: 0.4,
            metalness: 0.6
        });

        const mesh = new THREE.Mesh(geom, mat);
        mesh.position.set(p3d.x, groundY + bh / 2.0, p3d.z);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        mesh.userData = { id: p.id, district: p.district, zoning: p.zoning, height: p.height_m };

        state.buildingsGroup.add(mesh);
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
    const rainCount = 2000;
    const geom = new THREE.BufferGeometry();
    const positions = new Float32Array(rainCount * 3);

    for (let i = 0; i < rainCount; i++) {
        positions[i * 3] = (Math.random() - 0.5) * 600;
        positions[i * 3 + 1] = Math.random() * 250 + 10;
        positions[i * 3 + 2] = (Math.random() - 0.5) * 600;
    }

    geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));

    const mat = new THREE.PointsMaterial({
        color: 0x93c5fd,
        size: 1.2,
        transparent: true,
        opacity: 0.5
    });

    state.rainSystem = new THREE.Points(geom, mat);
    state.scene.add(state.rainSystem);
}


// =========================================================================
// MOVING TRAFFIC SIMULATION (Cars Running, Braking, Diverting, Resuming)
// =========================================================================

function spawnTrafficFleet(count = 160) {
    state.trafficGroup = new THREE.Group();
    state.vehicles = [];

    // Filter available edges to only open and dry roads (no flooded roads or water channels)
    const openDryEdges = Object.values(state.roadGraph).filter(e => !e.isClosed && (e.waterDepth || 0.0) < 0.05);
    const candidateEdges = openDryEdges.length > 0 ? openDryEdges : Object.values(state.roadGraph);
    if (candidateEdges.length === 0) return;

    const carColors = [0xf59e0b, 0xffffff, 0x38bdf8, 0xe2e8f0, 0x94a3b8, 0xef4444, 0x10b981];

    // Spatial occupancy map to prevent initial spawn overlap
    const laneOccupancy = new Map();

    let spawned = 0;
    let attempts = 0;
    while (spawned < count && attempts < count * 8) {
        attempts++;
        const edge = candidateEdges[Math.floor(Math.random() * candidateEdges.length)];
        const dir = Math.random() > 0.5 ? 1 : -1;
        const laneKey = `${edge.id}_${dir}`;

        if (!laneOccupancy.has(laneKey)) {
            laneOccupancy.set(laneKey, []);
        }
        const existingTs = laneOccupancy.get(laneKey);

        // Pick a candidate position along edge with safe buffer spacing
        const candidateT = 0.12 + Math.random() * 0.76;
        const minGapT = 4.8 / Math.max(10.0, edge.length);
        const hasOverlap = existingTs.some(t => Math.abs(t - candidateT) < minGapT);

        // At most 2 cars per lane on standard segments to ensure free-flowing traffic
        if (hasOverlap || existingTs.length >= 2) {
            continue;
        }

        existingTs.push(candidateT);

        const isEmergency = (Math.random() < 0.08); // 8% Ambulances / Fire
        const color = isEmergency ? 0xef4444 : carColors[Math.floor(Math.random() * carColors.length)];

        // 3D Car Body Mesh (compact scale to fit lanes cleanly)
        const carMesh = createCarMesh(color, isEmergency);
        state.trafficGroup.add(carMesh);

        state.vehicles.push({
            id: spawned,
            mesh: carMesh,
            edgeId: edge.id,
            t: candidateT,
            baseSpeed: (13.0 + Math.random() * 8.0) * (isEmergency ? 1.4 : 1.0),
            currentSpeed: 13.0,
            direction: dir,
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

    // 1. Group vehicles by lane (edgeId + direction) for car-following & anti-overlap physics
    const laneGroups = new Map();
    state.vehicles.forEach(veh => {
        const key = `${veh.edgeId}_${veh.direction}`;
        if (!laneGroups.has(key)) laneGroups.set(key, []);
        laneGroups.get(key).push(veh);
    });

    // 2. Intra-Lane Car-Following Model (Intelligent Driver Model / Anti-Tailgating)
    laneGroups.forEach((cars, key) => {
        if (cars.length <= 1) return;
        const dir = cars[0].direction;
        const edge = state.roadGraph[cars[0].edgeId];
        if (!edge) return;

        // Sort cars along travel direction:
        // When dir === 1: larger t is ahead (0 -> 1)
        // When dir === -1: smaller t is ahead (1 -> 0)
        if (dir === 1) {
            cars.sort((a, b) => a.t - b.t);
        } else {
            cars.sort((a, b) => b.t - a.t);
        }

        // Apply car-following headway constraint from rear to front
        for (let i = 0; i < cars.length - 1; i++) {
            const follower = cars[i];
            const leader = cars[i + 1];

            const headwayDist = (dir === 1)
                ? (leader.t - follower.t) * edge.length
                : (follower.t - leader.t) * edge.length;

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
                    if (dir === 1) {
                        follower.t = Math.min(follower.t, leader.t - minGapT);
                    } else {
                        follower.t = Math.max(follower.t, leader.t + minGapT);
                    }
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

        // Hydrodynamic Passability Logic
        if (isRoadClosed) {
            if (!veh.isEmergency) {
                // Passenger car stops before flood / barricade
                veh.currentSpeed = 0.0;
                veh.isStalled = true;

                // Evacuation check: attempt retreat onto dry connected road
                veh.waitingTimer += dt;
                if (veh.waitingTimer > 1.2) {
                    const nearNode = (veh.t < 0.5) ? edge.u : edge.v;
                    const dryOptions = Object.values(state.roadGraph).filter(e => 
                        (e.u === nearNode || e.v === nearNode) && 
                        e.id !== edge.id && 
                        !e.isClosed && 
                        (e.waterDepth || 0.0) < 0.05
                    );

                    if (dryOptions.length > 0) {
                        const nextEdge = dryOptions[Math.floor(Math.random() * dryOptions.length)];
                        veh.edgeId = nextEdge.id;
                        veh.t = (nextEdge.u === nearNode) ? 0.02 : 0.98;
                        veh.direction = (nextEdge.u === nearNode) ? 1 : -1;
                        veh.isStalled = false;
                        veh.currentSpeed = veh.baseSpeed * 0.7;
                        veh.waitingTimer = 0;
                        return;
                    }
                }

                // Flash hazard lights while stopped
                if (veh.mesh.userData.tailLights) {
                    const blink = Math.floor(Date.now() / 350) % 2 === 0;
                    veh.mesh.userData.tailLights.material.color.setHex(blink ? 0xf59e0b : 0xef4444);
                }
                return; // Do not advance into flood
            } else {
                // Emergency Ambulance
                if (water >= 0.30) {
                    veh.currentSpeed = 0.0;
                    veh.isStalled = true;
                    return;
                } else {
                    veh.currentSpeed = veh.baseSpeed * 0.35;
                    veh.isStalled = false;
                }
            }
        } else if (water >= 0.05) {
            // Shallow Ponding: Speed degraded (Pregnolato curve)
            veh.currentSpeed = Math.min(veh.currentSpeed, veh.baseSpeed * 0.50);
            veh.isStalled = false;
            veh.waitingTimer = 0;
        } else {
            // Open Dry Road: Smoothly regain free flow speed if clear ahead
            if (veh.currentSpeed < veh.baseSpeed * 0.9) {
                veh.currentSpeed = Math.min(veh.baseSpeed, veh.currentSpeed + 8.0 * dt);
            }
            veh.isStalled = false;
            veh.waitingTimer = 0;
        }

        // Kinematic Position Step
        const stepDist = (veh.currentSpeed * dt) / Math.max(12.0, edge.length);
        veh.t += stepDist * veh.direction;

        // Approaching road intersection (edge endpoint)
        if (veh.t >= 1.0 || veh.t <= 0.0) {
            const hitNode = (veh.t >= 1.0) ? edge.v : edge.u;

            // Find connected edges that are open and dry
            const availableConnections = Object.values(state.roadGraph).filter(e => 
                (e.u === hitNode || e.v === hitNode) && 
                e.id !== edge.id && 
                !e.isClosed && 
                (e.waterDepth || 0.0) < 0.05
            );

            if (availableConnections.length > 0) {
                const nextEdge = availableConnections[Math.floor(Math.random() * availableConnections.length)];
                veh.edgeId = nextEdge.id;
                veh.t = (nextEdge.u === hitNode) ? 0.02 : 0.98;
                veh.direction = (nextEdge.u === hitNode) ? 1 : -1;
            } else {
                // Dead end or surrounded by barricades: U-turn back onto open road
                veh.direction *= -1;
                veh.t = npClamp(veh.t, 0.03, 0.97);
            }
        }

        // Clamp position along edge
        veh.t = npClamp(veh.t, 0.0, 1.0);

        // 4. Compute 3D Position with Two-Way Lane Offset (Right-Hand Traffic)
        const forward = new THREE.Vector3().subVectors(edge.end, edge.start).normalize();
        const up = new THREE.Vector3(0, 1, 0);
        const right = new THREE.Vector3().crossVectors(forward, up).normalize();

        // Driving on the right-hand side of the road:
        // direction = 1 -> offset along +right (Right Lane)
        // direction = -1 -> offset along -right (Opposing Lane's Right)
        const laneOffsetDist = (edge.width * 0.26) * veh.direction;
        const lateralShift = right.clone().multiplyScalar(laneOffsetDist);

        const centerPos = new THREE.Vector3().lerpVectors(edge.start, edge.end, veh.t);
        const finalPos = centerPos.add(lateralShift);
        finalPos.y += 0.35; // Sit on road surface

        veh.mesh.position.copy(finalPos);

        // Orient vehicle along lane direction
        const lookTarget = veh.direction > 0
            ? edge.end.clone().add(lateralShift)
            : edge.start.clone().add(lateralShift);
        veh.mesh.lookAt(lookTarget.x, finalPos.y, lookTarget.z);

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

    // 4. Update KPI Telemetry Cards
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

        const h = depthGrid[r][c];
        const elev = state.terrain.elevation_grid[r][c];
        const isRiver = state.terrain.river_mask[r][c];

        let baseElevY = (elev - minElev) * 0.55;
        if (isRiver) baseElevY -= 1.6;

        if (h > 0.04 || isRiver) {
            // Physically rise water surface above terrain
            const waterHeight = Math.max(0.2, h * 1.6);
            pos.setY(i, baseElevY + waterHeight + 0.15);
        } else {
            // Submerged dry ground
            pos.setY(i, -100);
        }
    }

    pos.needsUpdate = true;
    state.waterMesh.geometry.computeVertexNormals();
}

function update3DRoadConditions(depthGrid) {
    if (!state.roadLinesGroup || !depthGrid) return;

    state.roadLinesGroup.children.forEach(mesh => {
        const edgeId = mesh.userData.edgeId;
        if (!edgeId) return; // Skip non-edge children like bridge piers
        const edge = state.roadGraph[edgeId];
        if (!edge) return;

        // Sample water depth across 5 points along road segment using exact world coords
        let maxWater = 0.0;
        const steps = 4;
        for (let s = 0; s <= steps; s++) {
            const frac = s / steps;
            const wx = edge.startWorld.x + (edge.endWorld.x - edge.startWorld.x) * frac;
            const wy = edge.startWorld.y + (edge.endWorld.y - edge.startWorld.y) * frac;
            const c = Math.floor(npClamp(wx / 50.0, 0, 119));
            const r = Math.floor(npClamp((6000.0 - wy) / 50.0, 0, 119));
            const d = depthGrid[r][c] || 0.0;
            if (d > maxWater) maxWater = d;
        }

        // Bridge handling: bridge is elevated above river
        if (edge.isBridge) {
            // Bridges only submerge if water reaches bridge deck height (overtopping >= 2.5m)
            if (maxWater < 2.5) {
                edge.waterDepth = 0.0;
            } else {
                edge.waterDepth = maxWater - 2.5;
            }
        } else {
            edge.waterDepth = maxWater;
        }

        // Update road passability status
        edge.isClosed = (edge.waterDepth >= 0.15);

        // Find associated barrier mesh group
        const barrier = state.barriersGroup.children.find(b => b.userData.edgeId === edgeId);
        const asphalt = mesh.userData.asphalt || mesh;

        if (edge.waterDepth >= 0.30) {
            // Severe Flooding / Emergency Cutoff: Hazard Red ribbon + Active Barricades
            asphalt.material.color.setHex(0xef4444);
            asphalt.material.emissive.setHex(0xb91c1c);
            asphalt.material.emissiveIntensity = 0.8;
            if (barrier) barrier.visible = true;
        } else if (edge.waterDepth >= 0.15) {
            // Passenger Stall / Closed Road: Glowing Hazard Orange-Red + Active Barricades
            asphalt.material.color.setHex(0xf97316);
            asphalt.material.emissive.setHex(0xc2410c);
            asphalt.material.emissiveIntensity = 0.6;
            if (barrier) barrier.visible = true;
        } else if (edge.waterDepth >= 0.05) {
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
                bldg.material.emissiveIntensity = isBlackout ? 0.02 : 0.25;
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


// =========================================================================
// ANIMATION & RENDER LOOP
// =========================================================================

function animate() {
    requestAnimationFrame(animate);

    const dt = state.clock.getDelta();
    const elapsedTime = state.clock.getElapsedTime();

    // 1. Update Traffic Simulation (Cars moving along roads)
    updateTraffic(dt);

    // 2. Animate Water Surface Ripples
    if (state.waterMesh && state.waterMesh.visible) {
        state.waterMesh.material.opacity = 0.80 + Math.sin(elapsedTime * 2.5) * 0.05;
    }

    // 3. Animate Rain Particles
    if (state.rainSystem && state.rainSystem.visible) {
        const pos = state.rainSystem.geometry.attributes.position;
        for (let i = 0; i < pos.count; i++) {
            let py = pos.getY(i) - 180 * dt;
            if (py < 0) py = 250;
            pos.setY(i, py);
        }
        pos.needsUpdate = true;
    }

    // 4. Drone Orbit Mode
    if (state.isDroneOrbit) {
        state.orbitAngle += dt * 0.15;
        const radius = 340;
        state.camera.position.x = Math.sin(state.orbitAngle) * radius;
        state.camera.position.z = Math.cos(state.orbitAngle) * radius;
        state.camera.position.y = 190 + Math.sin(state.orbitAngle * 0.5) * 30;
        state.controls.target.set(0, 15, 0);
    }

    // 5. Random Storm Lightning Flashes during heavy rain
    const rainIntensity = parseFloat(sliderRain.value);
    if (rainIntensity > 100 && Math.random() < 0.008) {
        state.flashLight.intensity = 4.0;
        setTimeout(() => { if (state.flashLight) state.flashLight.intensity = 0.0; }, 80);
    }

    // 6. Flash Warning Strobe Beacons on Road Barricades
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

    // Sliders
    sliderRain.addEventListener("input", (e) => { valRain.innerText = `${e.target.value} mm/hr`; });
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
    const col = Math.floor(npClamp(x / 50.0, 0, 119));
    const row = Math.floor(npClamp((6000.0 - y) / 50.0, 0, 119));
    const elev = state.terrain.elevation_grid[row][col];
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
