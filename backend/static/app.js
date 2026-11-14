/**
 * CityVerse Web Application Frontend Logic
 * Realistic Geography-Based Urban Digital Twin
 * 
 * Features:
 * - Top-down analytical GIS map with DEM hillshading and hypsometric terrain relief.
 * - Road hierarchy (Highways, Arterials, Local streets, River bridges) with dynamic passability.
 * - Non-abstract urban block parcels and architectural building footprints with zoning.
 * - Continuous hydrodynamic flood inundation raster layer with physical depth thresholds.
 * - Interactive bottom timeline scrubber for exploring 3-hour storm evolution.
 * - NLP scenario parser and counterfactual branch comparator.
 */

// Global State Container
const state = {
    isPlaying: false,
    playTimer: null,
    currentStepIdx: 0,
    totalSteps: 36,
    stepDtSeconds: 300.0, // 5 minutes per step
    engineMode: "world_model", // "world_model" or "physics"
    currentBranch: "baseline",
    forkedBranch: null,

    // GIS Layers
    terrain: null,
    hillshade: null,
    blocks: [],
    footprints: [],
    roads: [],
    facilities: [],

    // Active Simulation Trajectory Cache
    frames: [],
    currentFrame: null,

    // Telemetry History for Sparklines
    historyFloodedArea: [],
    historyEmsTime: [],
};

// Canvas references
const canvas = document.getElementById("cityCanvas");
const ctx = canvas.getContext("2d");

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


// --- Application Initialization ---

async function init() {
    try {
        console.log("Initializing CityVerse Geographical Digital Twin...");

        // 1. Fetch Terrain Elevation & River
        const terrainRes = await fetch("/api/city/terrain");
        state.terrain = await terrainRes.json();

        // 2. Fetch Analytical Hillshade Layer
        try {
            const hsRes = await fetch("/api/city/hillshade");
            if (hsRes.ok) {
                const hsData = await hsRes.json();
                state.hillshade = hsData.hillshade_grid;
            }
        } catch (e) {
            console.warn("Could not load hillshade:", e);
        }

        // 3. Fetch Road Network
        const roadsRes = await fetch("/api/city/roads");
        const roadsGeo = await roadsRes.json();
        state.roads = roadsGeo.features || [];

        // 4. Fetch Critical Facilities
        const facRes = await fetch("/api/city/facilities");
        const facGeo = await facRes.json();
        state.facilities = facGeo.features || [];

        // 5. Fetch Urban Blocks & Footprints
        try {
            const blocksRes = await fetch("/api/city/blocks");
            if (blocksRes.ok) {
                const bdata = await blocksRes.json();
                state.blocks = bdata.features || [];
            }
        } catch (e) {
            console.warn("Could not load blocks:", e);
        }

        try {
            const fpRes = await fetch("/api/city/footprints");
            if (fpRes.ok) {
                const fpData = await fpRes.json();
                state.footprints = fpData.features || [];
            }
        } catch (e) {
            console.warn("Could not load footprints:", e);
        }

        // Setup UI event listeners
        setupEventListeners();

        // Run default scenario (120 mm/hr, 3 hr, 60% capacity) to populate timeline
        await runScenario(false);

        console.log("CityVerse initialized successfully.");
    } catch (err) {
        console.error("Initialization error:", err);
    }
}


// --- Scenario Execution & Timeline Rollouts ---

async function runScenario(autoPlay = true) {
    const rainfall = parseFloat(sliderRain.value);
    const duration = parseFloat(sliderDuration.value);
    const drainage = parseFloat(sliderDrain.value) / 100.0;
    const p1 = parseFloat(sliderPump1.value) / 100.0;
    const p2 = parseFloat(sliderPump2.value) / 100.0;

    try {
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

        // Fetch all frames into local cache
        state.frames = [];
        for (let s = 0; s <= state.totalSteps; s++) {
            const fRes = await fetch(`/api/simulation/frame?step_idx=${s}&branch_id=${state.currentBranch}`);
            if (fRes.ok) {
                const frame = await fRes.json();
                state.frames.push(frame);
            }
        }

        // Scrub to initial frame
        scrubToStep(0);

        // Update sparkline graphs from summary series
        state.historyFloodedArea = summary.flooded_area_km2_series;
        state.historyEmsTime = summary.ems_time_min_series;
        updateSparklines();

        if (autoPlay) {
            startPlayback();
        }
    } catch (err) {
        console.error("Run scenario error:", err);
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

    updateTelemetry(frame.snapshot);
    renderGeoMap();
}


// --- Playback Animation Loop ---

function togglePlayback() {
    if (state.isPlaying) {
        stopPlayback();
    } else {
        startPlayback();
    }
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
    }, 450);
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


// --- Geographical Map Rendering Engine ---

function renderGeoMap() {
    if (!state.terrain) return;

    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const showWater = document.getElementById("chkWater").checked;
    const showRoads = document.getElementById("chkRoads").checked;
    const showBlocks = document.getElementById("chkBlocks").checked;
    const showFootprints = document.getElementById("chkFootprints").checked;
    const showHillshade = document.getElementById("chkHillshade").checked;
    const showFacilities = document.getElementById("chkFacilities").checked;

    const rows = state.terrain.rows;
    const cols = state.terrain.cols;
    const cellW = W / cols;
    const cellH = H / rows;

    const depthGrid = state.currentFrame ? state.currentFrame.surface_depth_grid : null;

    // 1. Terrain Topography & Analytical Hillshade
    const imgData = ctx.createImageData(W, H);
    const data = imgData.data;

    const minElev = state.terrain.min_elevation_m;
    const maxElev = state.terrain.max_elevation_m;

    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const elev = state.terrain.elevation_grid[r][c];
            const isRiver = state.terrain.river_mask[r][c];
            const shade = (showHillshade && state.hillshade) ? state.hillshade[r][c] : 0.5;

            // Normalized elevation (0.0 to 1.0)
            const normElev = (elev - minElev) / (maxElev - minElev);

            // Hypsometric terrain palette: valley green/slate -> mid-plateau -> highland ridge
            let cr, cg, cb;
            if (isRiver) {
                // Natural river channel: deep cerulean
                cr = 3; cg = 105; cb = 161;
            } else {
                // Valley greens to upland rock
                const baseR = 18 + normElev * 25;
                const baseG = 28 + normElev * 35;
                const baseB = 24 + normElev * 40;

                // Modulate with analytical hillshade illumination
                const lightFactor = 0.5 + shade * 0.75;
                cr = Math.floor(npClamp(baseR * lightFactor, 0, 255));
                cg = Math.floor(npClamp(baseG * lightFactor, 0, 255));
                cb = Math.floor(npClamp(baseB * lightFactor, 0, 255));
            }

            const startX = Math.floor(c * cellW);
            const startY = Math.floor(r * cellH);
            const endX = Math.floor((c + 1) * cellW);
            const endY = Math.floor((r + 1) * cellH);

            for (let py = startY; py < endY; py++) {
                for (let px = startX; px < endX; px++) {
                    const idx = (py * W + px) * 4;
                    data[idx] = cr;
                    data[idx + 1] = cg;
                    data[idx + 2] = cb;
                    data[idx + 3] = 255;
                }
            }
        }
    }
    ctx.putImageData(imgData, 0, 0);

    // 2. Urban Blocks (Zoning Parcels)
    if (showBlocks && state.blocks.length > 0) {
        ctx.lineWidth = 1.0;
        state.blocks.forEach(blk => {
            const coords = blk.geometry.coordinates[0];
            const props = blk.properties;

            ctx.beginPath();
            coords.forEach((pt, idx) => {
                const px = (pt[0] / 6000.0) * W;
                const py = (1.0 - pt[1] / 6000.0) * H;
                if (idx === 0) ctx.moveTo(px, py);
                else ctx.lineTo(px, py);
            });
            ctx.closePath();

            // Subtle zoning fill
            ctx.fillStyle = `${props.color}0d`;
            ctx.fill();
            ctx.strokeStyle = `${props.color}33`;
            ctx.stroke();
        });
    }

    // 3. Architectural Building Footprints
    if (showFootprints && state.footprints.length > 0) {
        state.footprints.forEach(bldg => {
            const coords = bldg.geometry.coordinates[0];
            const props = bldg.properties;

            const p1 = projectTopDown(coords[0][0], coords[0][1], W, H);
            const p2 = projectTopDown(coords[2][0], coords[2][1], W, H);
            const bw = Math.abs(p2.x - p1.x);
            const bh = Math.abs(p2.y - p1.y);

            // Drop shadow
            ctx.fillStyle = "rgba(0, 0, 0, 0.4)";
            ctx.fillRect(Math.min(p1.x, p2.x) + 2, Math.min(p1.y, p2.y) + 2, bw, bh);

            // Roof footprint fill by zoning type
            let roofColor = "#1e293b";
            if (props.zoning === "commercial_high" || props.zoning === "commercial_medium") {
                roofColor = "#0f172a";
            } else if (props.zoning === "industrial_heavy" || props.zoning === "port_industrial") {
                roofColor = "#262626";
            }

            ctx.fillStyle = roofColor;
            ctx.fillRect(Math.min(p1.x, p2.x), Math.min(p1.y, p2.y), bw, bh);

            ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
            ctx.lineWidth = 0.8;
            ctx.strokeRect(Math.min(p1.x, p2.x), Math.min(p1.y, p2.y), bw, bh);
        });
    }

    // 4. Road Network with Hierarchy & Dynamic Passability
    if (showRoads && state.roads.length > 0) {
        state.roads.forEach(feat => {
            const coords = feat.geometry.coordinates;
            const props = feat.properties;
            const isHighway = props.road_class === "highway";
            const isBridge = props.is_bridge;

            const p1 = projectTopDown(coords[0][0], coords[0][1], W, H);
            const p2 = projectTopDown(coords[1][0], coords[1][1], W, H);

            // Sample midpoint water depth
            const midX = (coords[0][0] + coords[1][0]) / 2.0;
            const midY = (coords[0][1] + coords[1][1]) / 2.0;
            const mc = Math.floor(midX / 50.0);
            const mr = Math.floor((6000.0 - midY) / 50.0);

            let water = 0.0;
            if (depthGrid && mr >= 0 && mr < rows && mc >= 0 && mc < cols) {
                water = depthGrid[mr][mc];
            }

            // Determine road passability color & line width
            let roadColor = "#334155";
            let roadWidth = isHighway ? 5.5 : 2.8;

            if (water >= 0.30) {
                roadColor = "#ef4444"; // Closed emergency / severed
                roadWidth += 1.5;
            } else if (water >= 0.15) {
                roadColor = "#f59e0b"; // Closed passenger
                roadWidth += 1.0;
            } else if (water >= 0.05) {
                roadColor = "#eab308"; // Speed degraded
            } else if (isHighway) {
                roadColor = "#475569";
            }

            ctx.strokeStyle = roadColor;
            ctx.lineWidth = roadWidth;
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.stroke();

            // Highway centerline dashed lane stripes
            if (isHighway && water < 0.15) {
                ctx.strokeStyle = "#94a3b8";
                ctx.lineWidth = 1.0;
                ctx.setLineDash([4, 4]);
                ctx.beginPath();
                ctx.moveTo(p1.x, p1.y);
                ctx.lineTo(p2.x, p2.y);
                ctx.stroke();
                ctx.setLineDash([]);
            }

            // Bridge abutment markers
            if (isBridge) {
                ctx.strokeStyle = "#f8fafc";
                ctx.lineWidth = 1.0;
                ctx.stroke();
            }
        });
    }

    // 5. Continuous Hydrodynamic Flood Inundation Raster
    if (showWater && depthGrid) {
        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                const h = depthGrid[r][c];
                if (h <= 0.03) continue;

                const px = (c / cols) * W;
                const py = (r / rows) * H;
                const pw = cellW + 0.5;
                const ph = cellH + 0.5;

                // Scientific continuous flood depth coloring
                if (h < 0.15) {
                    // Shallow: 0.05m to 0.15m (Sky Blue, 45% opacity)
                    ctx.fillStyle = "rgba(56, 189, 248, 0.45)";
                } else if (h < 0.30) {
                    // Passenger stall: 0.15m to 0.30m (Azure Blue, 70% opacity)
                    ctx.fillStyle = "rgba(2, 132, 199, 0.70)";
                } else if (h < 0.60) {
                    // Emergency cutoff: 0.30m to 0.60m (Deep Cobalt, 85% opacity)
                    ctx.fillStyle = "rgba(29, 78, 216, 0.85)";
                } else {
                    // Severe / Dangerous: >0.60m (Indigo/Violet, 95% opacity)
                    ctx.fillStyle = "rgba(67, 56, 202, 0.95)";
                }

                ctx.fillRect(px, py, pw, ph);
            }
        }
    }

    // 6. Critical Facilities
    if (showFacilities && state.facilities.length > 0) {
        state.facilities.forEach(feat => {
            const props = feat.properties;
            const [fx, fy] = feat.geometry.coordinates;
            const pt = projectTopDown(fx, fy, W, H);

            // Sample water depth at facility
            const mc = Math.floor(fx / 50.0);
            const mr = Math.floor((6000.0 - fy) / 50.0);
            let water = (depthGrid && mr >= 0 && mr < rows && mc >= 0 && mc < cols) ? depthGrid[mr][mc] : 0.0;

            if (props.category === "hospitals") {
                // Hospital: Red circle with white medical cross
                ctx.fillStyle = "#ef4444";
                ctx.beginPath();
                ctx.arc(pt.x, pt.y, 7, 0, Math.PI * 2);
                ctx.fill();
                ctx.strokeStyle = "#ffffff";
                ctx.lineWidth = 1.5;
                ctx.stroke();

                ctx.fillStyle = "#ffffff";
                ctx.font = "bold 9px sans-serif";
                ctx.fillText("+", pt.x - 3, pt.y + 3);
            } else if (props.category === "electrical_substations") {
                // Substation: Square with lightning bolt
                const isTripped = water >= props.flood_vulnerability_threshold_m;
                ctx.fillStyle = isTripped ? "#ef4444" : "#00f0ff";
                ctx.fillRect(pt.x - 5, pt.y - 5, 10, 10);
                ctx.strokeStyle = "#ffffff";
                ctx.lineWidth = 1;
                ctx.strokeRect(pt.x - 5, pt.y - 5, 10, 10);

                ctx.fillStyle = "#090d16";
                ctx.font = "bold 8px monospace";
                ctx.fillText("⚡", pt.x - 4, pt.y + 3);
            } else if (props.category === "pumping_stations") {
                // Pump Station: Diamond with P
                ctx.fillStyle = "#10b981";
                ctx.beginPath();
                ctx.arc(pt.x, pt.y, 6, 0, Math.PI * 2);
                ctx.fill();
                ctx.strokeStyle = "#ffffff";
                ctx.stroke();

                ctx.fillStyle = "#ffffff";
                ctx.font = "bold 7px sans-serif";
                ctx.fillText("P", pt.x - 2.5, pt.y + 2.5);
            } else if (props.category === "ems_stations") {
                // EMS Station: Amber dot
                ctx.fillStyle = "#f59e0b";
                ctx.beginPath();
                ctx.arc(pt.x, pt.y, 5, 0, Math.PI * 2);
                ctx.fill();
                ctx.strokeStyle = "#ffffff";
                ctx.stroke();
            }
        });
    }
}

function projectTopDown(x, y, W, H) {
    return {
        x: (x / 6000.0) * W,
        y: (1.0 - y / 6000.0) * H
    };
}

function npClamp(val, min, max) {
    return Math.max(min, Math.min(val, max));
}


// --- Event Listeners & Sliders ---

function setupEventListeners() {
    // Primary Header Actions
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

    // Sliders
    sliderRain.addEventListener("input", (e) => {
        valRain.innerText = `${e.target.value} mm/hr`;
    });

    sliderDuration.addEventListener("input", (e) => {
        valDuration.innerText = `${parseFloat(e.target.value).toFixed(1)} hrs`;
    });

    sliderDrain.addEventListener("input", (e) => {
        valDrain.innerText = `${e.target.value}%`;
    });

    sliderPump1.addEventListener("input", (e) => {
        valPump1.innerText = `${e.target.value}%`;
    });

    sliderPump2.addEventListener("input", (e) => {
        valPump2.innerText = `${e.target.value}%`;
    });

    // Engine toggle
    modeNeuralBtn.addEventListener("click", () => setEngineMode("world_model"));
    modePhysicsBtn.addEventListener("click", () => setEngineMode("physics"));

    // NLP Execution
    btnNlp.addEventListener("click", handleNlpPrompt);

    // Branching
    btnFork.addEventListener("click", handleForkScenario);
    btnCompare.addEventListener("click", handleCompareBranches);

    // Canvas Hover Inspector
    canvas.addEventListener("mousemove", handleCanvasHover);
    canvas.addEventListener("mouseleave", () => inspectorCard.classList.add("hidden"));

    // Layer checkboxes
    ["chkWater", "chkRoads", "chkBlocks", "chkFootprints", "chkHillshade", "chkFacilities"].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.addEventListener("change", renderGeoMap);
    });
}

function setEngineMode(mode) {
    state.engineMode = mode;
    if (mode === "world_model") {
        modeNeuralBtn.classList.add("active");
        modePhysicsBtn.classList.remove("active");
        elEngineBadge.innerText = "NEURAL WORLD MODEL (2ms)";
        elEngineBadge.className = "val val-accent";
    } else {
        modePhysicsBtn.classList.add("active");
        modeNeuralBtn.classList.remove("active");
        elEngineBadge.innerText = "COUPLED 2D PHYSICS (Exact)";
        elEngineBadge.className = "val val-success";
    }
}


// --- NLP Scenario Parser Handler ---

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


// --- Counterfactual Timeline Branching ---

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


// --- Location Inspector Hover ---

function handleCanvasHover(e) {
    if (!state.terrain) return;

    const rect = canvas.getBoundingClientRect();
    const xPx = e.clientX - rect.left;
    const yPx = e.clientY - rect.top;

    const normX = xPx / canvas.width;
    const normY = yPx / canvas.height;

    const col = Math.floor(normX * state.terrain.cols);
    const row = Math.floor(normY * state.terrain.rows);

    if (row < 0 || row >= state.terrain.rows || col < 0 || col >= state.terrain.cols) {
        inspectorCard.classList.add("hidden");
        return;
    }

    const elev = state.terrain.elevation_grid[row][col];
    const depthGrid = state.currentFrame ? state.currentFrame.surface_depth_grid : null;
    const water = depthGrid ? depthGrid[row][col] : 0.0;
    const worldX = Math.round(col * 50.0);
    const worldY = Math.round(6000.0 - (row * 50.0));

    inspectorCard.classList.remove("hidden");
    inspectorContent.innerHTML = `
        <div class="inspector-row"><span>Coordinates:</span><strong>(${worldX}m, ${worldY}m)</strong></div>
        <div class="inspector-row"><span>Ground Elevation:</span><strong>${elev.toFixed(1)} m</strong></div>
        <div class="inspector-row"><span>Water Depth:</span><strong style="color: ${water > 0.15 ? 'var(--danger-red)' : (water > 0.03 ? 'var(--accent-cyan)' : 'var(--text-primary)')}">${(water * 100).toFixed(1)} cm</strong></div>
        <div class="inspector-row"><span>Passenger Traffic:</span><strong style="color: ${water >= 0.15 ? 'var(--danger-red)' : 'var(--success-green)'}">${water >= 0.15 ? 'CLOSED (Stalled)' : 'OPEN'}</strong></div>
        <div class="inspector-row"><span>Emergency Route:</span><strong style="color: ${water >= 0.30 ? 'var(--danger-red)' : 'var(--success-green)'}">${water >= 0.30 ? 'IMPASSABLE' : 'PASSABLE'}</strong></div>
    `;
}


// --- Telemetry Displays & Sparklines ---

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

// Start application
window.addEventListener("DOMContentLoaded", init);
