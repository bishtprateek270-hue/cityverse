/**
 * CityVerse Web Application Frontend Logic
 * Realistic Geography-Based Urban Digital Twin
 * 
 * Powered by Leaflet.js Cartographic Basemaps & Vectorized 2D Hydrodynamic Physics.
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

    // Leaflet Map & Layer References
    map: null,
    baseLayers: {},
    activeBaseLayer: null,
    floodOverlay: null,
    riverLayer: null,
    roadsLayer: null,
    facilitiesLayer: null,
    blocksLayer: null,
    footprintsLayer: null,
    contoursLayer: null,

    // GeoJSON Data Caches
    riverGeo: null,
    roadsGeo: null,
    facilitiesGeo: null,
    blocksGeo: null,
    footprintsGeo: null,
    contoursGeo: null,
    geoBounds: null,

    // Offscreen Canvas for Fast Flood Raster Interpolation
    floodCanvas: null,
    floodCtx: null,

    // Active Simulation Trajectory Cache
    frames: [],
    currentFrame: null,

    // Telemetry History for Sparklines
    historyFloodedArea: [],
    historyEmsTime: [],
};

// Geographical Coordinates Extent for MetroVerdia (Buffalo Bayou / Houston Domain)
const GEO_BOUNDS = [
    [29.7200, -95.4000],  // Southwest corner [lat, lon]
    [29.77412, -95.33792] // Northeast corner [lat, lon]
];
const GEO_CENTER = [29.74706, -95.36896];

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
        console.log("Initializing CityVerse Geographical Digital Twin with Leaflet.js...");

        // Setup offscreen canvas for continuous flood raster rendering (240x240 interpolated)
        state.floodCanvas = document.createElement("canvas");
        state.floodCanvas.width = 240;
        state.floodCanvas.height = 240;
        state.floodCtx = state.floodCanvas.getContext("2d");

        // 1. Initialize Leaflet Map with Clean Free Tile Basemaps
        initLeafletMap();

        // 2. Fetch and Load Geographic Vector Layers
        await loadGeographicLayers();

        // 3. Setup UI Event Listeners
        setupEventListeners();

        // 4. Load Pre-Computed Simulation Trajectory Instantly
        const loaded = await loadCachedFrames();
        if (!loaded) {
            await runScenario(false);
        }

        console.log("CityVerse initialized successfully.");
    } catch (err) {
        console.error("Initialization error:", err);
    }
}


// =========================================================================
// LEAFLET MAP & CLEAN BASEMAP CONFIGURATION
// =========================================================================

function initLeafletMap() {
    state.map = L.map("cityMap", {
        center: GEO_CENTER,
        zoom: 13,
        minZoom: 11,
        maxZoom: 18,
        zoomControl: true,
        attributionControl: false
    });

    // Clean, Free, Watermark-Free Basemap Providers
    state.baseLayers = {
        dark: L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}", {
            maxZoom: 16,
            attribution: "Esri, HERE, Garmin"
        }),
        satellite: L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
            maxZoom: 18,
            attribution: "Esri, Maxar, Earthstar Geographics"
        }),
        street: L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}", {
            maxZoom: 18,
            attribution: "Esri, DeLorme, NAVTEQ"
        }),
        osm: L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
            maxZoom: 19,
            attribution: "© OpenStreetMap contributors"
        })
    };

    // Default to Dark Gray GIS Basemap
    state.activeBaseLayer = state.baseLayers.dark;
    state.activeBaseLayer.addTo(state.map);

    // Initial Empty Flood Image Overlay
    const emptyDataUrl = state.floodCanvas.toDataURL();
    state.floodOverlay = L.imageOverlay(emptyDataUrl, GEO_BOUNDS, {
        opacity: 0.88,
        interactive: false,
        zIndex: 400
    }).addTo(state.map);

    // Auto-fit city bounds so MetroVerdia fills the entire screen
    state.map.fitBounds(GEO_BOUNDS, { padding: [25, 25] });

    // Add Scale Bar
    L.control.scale({ imperial: false, metric: true, position: "bottomright" }).addTo(state.map);

    // Map Mouse Move Inspector
    state.map.on("mousemove", handleMapMouseMove);
    state.map.on("mouseout", () => inspectorCard.classList.add("hidden"));
}

function setBasemap(type) {
    if (!state.baseLayers[type]) return;

    if (state.activeBaseLayer) {
        state.map.removeLayer(state.activeBaseLayer);
    }
    state.activeBaseLayer = state.baseLayers[type];
    state.activeBaseLayer.addTo(state.map);
    state.activeBaseLayer.bringToBack();

    // Update button states
    ["bmDark", "bmSatellite", "bmStreet", "bmOsm"].forEach(id => {
        const btn = document.getElementById(id);
        if (btn) btn.classList.remove("active");
    });
    if (type === "dark") document.getElementById("bmDark").classList.add("active");
    if (type === "satellite") document.getElementById("bmSatellite").classList.add("active");
    if (type === "street") document.getElementById("bmStreet").classList.add("active");
    if (type === "osm") document.getElementById("bmOsm").classList.add("active");
}


// =========================================================================
// GEOGRAPHIC VECTOR LAYERS (WGS84 GeoJSON)
// =========================================================================

async function loadGeographicLayers() {
    // 1. River / Bayou System Layer
    try {
        const res = await fetch("/api/city/geo/river");
        if (res.ok) {
            state.riverGeo = await res.json();
            state.riverLayer = L.geoJSON(state.riverGeo, {
                style: {
                    color: "#0284c7",
                    weight: 2.5,
                    fillColor: "#0369a1",
                    fillOpacity: 0.85
                },
                onEachFeature: (feature, layer) => {
                    layer.bindPopup(`<strong>${feature.properties.name || "Central River Channel"}</strong><br>Hydrologic Basin: Alluvial Floodplain`);
                }
            }).addTo(state.map);
        }
    } catch (e) {
        console.warn("Could not load river GeoJSON:", e);
    }

    // 2. Urban Blocks (Zoning Parcels)
    try {
        const res = await fetch("/api/city/geo/blocks");
        if (res.ok) {
            state.blocksGeo = await res.json();
            state.blocksLayer = L.geoJSON(state.blocksGeo, {
                style: (feature) => {
                    const color = feature.properties.color || "#38bdf8";
                    return {
                        color: color,
                        weight: 1.2,
                        opacity: 0.5,
                        fillColor: color,
                        fillOpacity: 0.15
                    };
                },
                onEachFeature: (feature, layer) => {
                    const p = feature.properties;
                    layer.bindTooltip(`<strong>${p.district}</strong><br>Zoning: ${p.zoning}<br>Area: ${(p.area_m2 / 10000).toFixed(1)} ha`, { sticky: true });
                }
            }).addTo(state.map);
        }
    } catch (e) {
        console.warn("Could not load blocks GeoJSON:", e);
    }

    // 3. 2.5D Building Footprints
    try {
        const res = await fetch("/api/city/geo/footprints");
        if (res.ok) {
            state.footprintsGeo = await res.json();
            state.footprintsLayer = L.geoJSON(state.footprintsGeo, {
                style: (feature) => {
                    const z = feature.properties.zoning;
                    let fill = "#1e293b";
                    if (z === "commercial_high") fill = "#0f172a";
                    else if (z === "industrial_heavy") fill = "#262626";
                    return {
                        color: "rgba(255, 255, 255, 0.25)",
                        weight: 0.9,
                        fillColor: fill,
                        fillOpacity: 0.70
                    };
                },
                onEachFeature: (feature, layer) => {
                    const p = feature.properties;
                    layer.bindPopup(`<strong>${p.use_type}</strong><br>Height: ${p.height_m.toFixed(1)}m (${p.stories} stories)<br>Elevation: ${p.ground_elev_m.toFixed(1)}m<br>District: ${p.district}`);
                }
            }).addTo(state.map);
        }
    } catch (e) {
        console.warn("Could not load footprints GeoJSON:", e);
    }

    // 4. Topographic Elevation Contours
    try {
        const res = await fetch("/api/city/geo/contours");
        if (res.ok) {
            state.contoursGeo = await res.json();
            state.contoursLayer = L.geoJSON(state.contoursGeo, {
                style: (feature) => {
                    const elev = feature.properties.elevation_m;
                    const isIndex = (elev % 20 === 0);
                    return {
                        color: isIndex ? "#94a3b8" : "#475569",
                        weight: isIndex ? 1.4 : 0.7,
                        opacity: isIndex ? 0.6 : 0.35,
                        dashArray: isIndex ? null : "3, 3"
                    };
                },
                onEachFeature: (feature, layer) => {
                    layer.bindTooltip(`Elevation: ${feature.properties.elevation_m}m MSL`, { sticky: true });
                }
            }).addTo(state.map);
        }
    } catch (e) {
        console.warn("Could not load contours GeoJSON:", e);
    }

    // 5. Road Network with Dynamic Passability
    try {
        const res = await fetch("/api/city/geo/roads");
        if (res.ok) {
            state.roadsGeo = await res.json();
            state.roadsLayer = L.geoJSON(state.roadsGeo, {
                style: (feature) => getRoadStyle(feature, 0.0),
                onEachFeature: (feature, layer) => {
                    layer.on("click", () => handleRoadClick(feature, layer));
                }
            }).addTo(state.map);
        }
    } catch (e) {
        console.warn("Could not load roads GeoJSON:", e);
    }

    // 6. Critical Facilities with Custom Markers
    try {
        const res = await fetch("/api/city/geo/facilities");
        if (res.ok) {
            state.facilitiesGeo = await res.json();
            state.facilitiesLayer = L.geoJSON(state.facilitiesGeo, {
                pointToLayer: (feature, latlng) => {
                    const p = feature.properties;
                    const icon = createFacilityIcon(p.category, false);
                    return L.marker(latlng, { icon: icon });
                },
                onEachFeature: (feature, layer) => {
                    const p = feature.properties;
                    layer.bindPopup(`
                        <div style="font-family: var(--font-sans); min-width: 180px;">
                            <strong style="color: var(--accent-cyan); font-size: 13px;">${p.name}</strong><br>
                            <span style="color: var(--text-muted); font-size: 10px; text-transform: uppercase;">${p.category.replace('_', ' ')}</span>
                            <hr style="border: 0; border-top: 1px solid var(--border-color); margin: 6px 0;">
                            <div>Elevation: <strong>${(p.elevation_m || 25.0).toFixed(1)} m</strong></div>
                            <div>Flood Threshold: <strong>${(p.flood_vulnerability_threshold_m || 0.3).toFixed(2)} m</strong></div>
                            <div>Capacity: <strong>${p.capacity || p.rated_capacity_m3s || 'Nominal'}</strong></div>
                        </div>
                    `);
                }
            }).addTo(state.map);
        }
    } catch (e) {
        console.warn("Could not load facilities GeoJSON:", e);
    }
}

function getRoadStyle(feature, waterDepth) {
    const p = feature.properties;
    const isHighway = (p.road_class === "highway");
    let baseWidth = isHighway ? 5.5 : (p.road_class === "arterial" ? 3.8 : 2.2);
    let color = isHighway ? "#cbd5e1" : (p.road_class === "arterial" ? "#94a3b8" : "#64748b");

    if (waterDepth >= 0.30) {
        // Closed emergency / impassable
        color = "#ef4444";
        baseWidth += 2.5;
    } else if (waterDepth >= 0.15) {
        // Stalled passenger vehicles
        color = "#f97316";
        baseWidth += 1.6;
    } else if (waterDepth >= 0.05) {
        // Speed degraded
        color = "#eab308";
        baseWidth += 0.8;
    }

    return {
        color: color,
        weight: baseWidth,
        opacity: waterDepth >= 0.15 ? 1.0 : 0.85
    };
}

function createFacilityIcon(category, isTripped) {
    let cls = "icon-substation";
    let symbol = "⚡";
    let size = [26, 26];

    if (category === "hospitals") {
        cls = "icon-hospital";
        symbol = "+";
        size = [28, 28];
    } else if (category === "electrical_substations") {
        cls = isTripped ? "icon-substation tripped" : "icon-substation";
        symbol = "⚡";
    } else if (category === "pumping_stations") {
        cls = isTripped ? "icon-pump tripped" : "icon-pump";
        symbol = "P";
    } else if (category === "ems_stations") {
        cls = "icon-ems";
        symbol = "🚒";
    }

    return L.divIcon({
        className: `facility-marker-icon ${cls}`,
        html: `<span>${symbol}</span>`,
        iconSize: size,
        iconAnchor: [size[0] / 2, size[1] / 2]
    });
}


// =========================================================================
// FAST TRAJECTORY LOADING & SCENARIO EXECUTION
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

                // Immediately scrub to storm peak (T+02:00 or T+03:00) so user sees realistic flooding!
                const previewStep = Math.min(24, state.totalSteps);
                scrubToStep(previewStep);
                return true;
            }
        }
    } catch (e) {
        console.warn("Fast frame loading notice:", e);
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

        // Fetch all frames in ONE fast batch request
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

    // 1. Update Inundation Raster Overlay
    updateFloodOverlay(frame.surface_depth_grid);

    // 2. Update Dynamic Road Network Colors
    updateRoadPassability(frame.surface_depth_grid);

    // 3. Update Facility Operational Status
    updateFacilityStatus(frame.surface_depth_grid);

    // 4. Update KPI Telemetry Cards
    updateTelemetry(frame.snapshot);
}


// =========================================================================
// HYDRODYNAMIC FLOOD RASTER OVERLAY GENERATION
// =========================================================================

function updateFloodOverlay(depthGrid) {
    if (!depthGrid || !state.floodOverlay) return;

    const ctx = state.floodCtx;
    const cw = state.floodCanvas.width;
    const ch = state.floodCanvas.height;
    ctx.clearRect(0, 0, cw, ch);

    const rows = depthGrid.length;
    const cols = depthGrid[0].length;
    const imgData = ctx.createImageData(cw, ch);
    const data = imgData.data;

    const scaleX = cols / cw;
    const scaleY = rows / ch;

    // Bilinear interpolation across 240x240 canvas
    for (let py = 0; py < ch; py++) {
        const rFloat = py * scaleY;
        const r0 = Math.floor(rFloat);
        const r1 = Math.min(r0 + 1, rows - 1);
        const dr = rFloat - r0;

        for (let px = 0; px < cw; px++) {
            const cFloat = px * scaleX;
            const c0 = Math.floor(cFloat);
            const c1 = Math.min(c0 + 1, cols - 1);
            const dc = cFloat - c0;

            // Bilinear depth sampling
            const h = (1 - dr) * (1 - dc) * depthGrid[r0][c0] +
                      (1 - dr) * dc * depthGrid[r0][c1] +
                      dr * (1 - dc) * depthGrid[r1][c0] +
                      dr * dc * depthGrid[r1][c1];

            if (h < 0.04) continue; // Dry or negligible ponding

            const idx = (py * cw + px) * 4;

            // Continuous scientific water depth colormap
            if (h < 0.15) {
                // Shallow: 0.05m to 0.15m (Sky Blue)
                data[idx] = 56;
                data[idx + 1] = 189;
                data[idx + 2] = 248;
                data[idx + 3] = Math.floor(130 * Math.min(1.0, (h - 0.04) / 0.11));
            } else if (h < 0.30) {
                // Passenger stall: 0.15m to 0.30m (Vibrant Cyan-Blue)
                data[idx] = 14;
                data[idx + 1] = 165;
                data[idx + 2] = 233;
                data[idx + 3] = 185;
            } else if (h < 0.60) {
                // Emergency cutoff: 0.30m to 0.60m (Cobalt Royal Blue)
                data[idx] = 37;
                data[idx + 1] = 99;
                data[idx + 2] = 235;
                data[idx + 3] = 220;
            } else {
                // Extreme deep: >0.60m (Deep Indigo / Violet)
                data[idx] = 49;
                data[idx + 1] = 46;
                data[idx + 2] = 129;
                data[idx + 3] = 240;
            }
        }
    }

    ctx.putImageData(imgData, 0, 0);
    state.floodOverlay.setUrl(state.floodCanvas.toDataURL());
}

function updateRoadPassability(depthGrid) {
    if (!state.roadsLayer || !depthGrid) return;

    state.roadsLayer.eachLayer(layer => {
        const feat = layer.feature;
        const coords = feat.geometry.coordinates;

        // Sample midpoint water depth
        const midLon = (coords[0][0] + coords[1][0]) / 2.0;
        const midLat = (coords[0][1] + coords[1][1]) / 2.0;

        const cell = latLngToGrid(midLat, midLon);
        let water = 0.0;
        if (cell.r >= 0 && cell.r < depthGrid.length && cell.c >= 0 && cell.c < depthGrid[0].length) {
            water = depthGrid[cell.r][cell.c];
        }

        layer.setStyle(getRoadStyle(feat, water));
    });
}

function updateFacilityStatus(depthGrid) {
    if (!state.facilitiesLayer || !depthGrid) return;

    state.facilitiesLayer.eachLayer(layer => {
        const feat = layer.feature;
        const [lon, lat] = feat.geometry.coordinates;
        const cell = latLngToGrid(lat, lon);

        let water = 0.0;
        if (cell.r >= 0 && cell.r < depthGrid.length && cell.c >= 0 && cell.c < depthGrid[0].length) {
            water = depthGrid[cell.r][cell.c];
        }

        const thresh = feat.properties.flood_vulnerability_threshold_m || 0.30;
        const isTripped = (water >= thresh);

        layer.setIcon(createFacilityIcon(feat.properties.category, isTripped));
    });
}

function latLngToGrid(lat, lon) {
    const latMin = GEO_BOUNDS[0][0];
    const latMax = GEO_BOUNDS[1][0];
    const lonMin = GEO_BOUNDS[0][1];
    const lonMax = GEO_BOUNDS[1][1];

    const normX = (lon - lonMin) / (lonMax - lonMin);
    const normY = (lat - latMin) / (latMax - latMin);

    const c = Math.floor(normX * 120.0);
    const r = Math.floor((1.0 - normY) * 120.0);
    return { r, c };
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

    // Basemap Switcher Buttons
    document.getElementById("bmDark").addEventListener("click", () => setBasemap("dark"));
    document.getElementById("bmSatellite").addEventListener("click", () => setBasemap("satellite"));
    document.getElementById("bmStreet").addEventListener("click", () => setBasemap("street"));
    document.getElementById("bmOsm").addEventListener("click", () => setBasemap("osm"));
    document.getElementById("btnFitCity").addEventListener("click", () => {
        state.map.fitBounds(GEO_BOUNDS, { padding: [25, 25] });
    });

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

    // Layer Checkboxes
    setupLayerToggle("chkWater", () => state.floodOverlay);
    setupLayerToggle("chkRoads", () => state.roadsLayer);
    setupLayerToggle("chkRiver", () => state.riverLayer);
    setupLayerToggle("chkFootprints", () => state.footprintsLayer);
    setupLayerToggle("chkBlocks", () => state.blocksLayer);
    setupLayerToggle("chkContours", () => state.contoursLayer);
    setupLayerToggle("chkFacilities", () => state.facilitiesLayer);
}

function setupLayerToggle(id, layerGetter) {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener("change", (e) => {
        const layer = layerGetter();
        if (!layer) return;
        if (e.target.checked) {
            state.map.addLayer(layer);
        } else {
            state.map.removeLayer(layer);
        }
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
// MAP HOVER INSPECTOR
// =========================================================================

function handleMapMouseMove(e) {
    const lat = e.latlng.lat;
    const lon = e.latlng.lng;

    const cell = latLngToGrid(lat, lon);
    const depthGrid = state.currentFrame ? state.currentFrame.surface_depth_grid : null;
    let water = 0.0;
    if (depthGrid && cell.r >= 0 && cell.r < depthGrid.length && cell.c >= 0 && cell.c < depthGrid[0].length) {
        water = depthGrid[cell.r][cell.c];
    }

    // Estimate elevation from row/col
    const elev = 15.0 + ((119 - cell.r) / 119.0) * 35.0;

    inspectorCard.classList.remove("hidden");
    inspectorContent.innerHTML = `
        <div class="inspector-row"><span>Geographic GPS:</span><strong>${lat.toFixed(4)}°N, ${Math.abs(lon).toFixed(4)}°W</strong></div>
        <div class="inspector-row"><span>Ground Elevation:</span><strong>${elev.toFixed(1)} m</strong></div>
        <div class="inspector-row"><span>Water Inundation:</span><strong style="color: ${water > 0.15 ? 'var(--danger-red)' : (water > 0.04 ? 'var(--accent-cyan)' : 'var(--text-primary)')}">${(water * 100).toFixed(1)} cm</strong></div>
        <div class="inspector-row"><span>Passenger Traffic:</span><strong style="color: ${water >= 0.15 ? 'var(--danger-red)' : 'var(--success-green)'}">${water >= 0.15 ? 'CLOSED (Stalled)' : 'OPEN'}</strong></div>
        <div class="inspector-row"><span>Emergency Route:</span><strong style="color: ${water >= 0.30 ? 'var(--danger-red)' : 'var(--success-green)'}">${water >= 0.30 ? 'IMPASSABLE' : 'PASSABLE'}</strong></div>
    `;
}

function handleRoadClick(feature, layer) {
    const p = feature.properties;
    const coords = feature.geometry.coordinates;
    const midLon = (coords[0][0] + coords[1][0]) / 2.0;
    const midLat = (coords[0][1] + coords[1][1]) / 2.0;

    const cell = latLngToGrid(midLat, midLon);
    const depthGrid = state.currentFrame ? state.currentFrame.surface_depth_grid : null;
    let water = 0.0;
    if (depthGrid && cell.r >= 0 && cell.r < depthGrid.length && cell.c >= 0 && cell.c < depthGrid[0].length) {
        water = depthGrid[cell.r][cell.c];
    }

    let status = "OPEN (Normal Free Flow)";
    if (water >= 0.30) status = "CLOSED (Emergency Impassable)";
    else if (water >= 0.15) status = "STALLED (Passenger Cutoff)";
    else if (water >= 0.05) status = "SPEED DEGRADED (Pregnolato Curve)";

    layer.bindPopup(`
        <div style="font-family: var(--font-sans); min-width: 170px;">
            <strong style="color: var(--accent-cyan); font-size: 13px;">${p.edge_id}</strong><br>
            <span style="color: var(--text-muted); font-size: 10px; text-transform: uppercase;">Class: ${p.road_class}</span>
            <hr style="border: 0; border-top: 1px solid var(--border-color); margin: 6px 0;">
            <div>Speed Limit: <strong>${p.free_flow_speed_kmh} km/h</strong></div>
            <div>Inundation Depth: <strong style="color: ${water >= 0.15 ? '#ef4444' : '#00f0ff'};">${(water * 100).toFixed(1)} cm</strong></div>
            <div>Status: <strong>${status}</strong></div>
        </div>
    `).openPopup();
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
