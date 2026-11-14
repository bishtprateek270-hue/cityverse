"""
backend.nlp_parser
==================
Natural Language Scenario Parser for CityVerse.
Extracts structured simulation parameters from unstructured user text requests
without fabricating physical outcomes.
"""

from typing import Dict, Any, Optional
import re


class NaturalLanguageScenarioParser:
    """
    Parses user natural language prompts into structured scenario configurations.
    Example input:
        "Simulate 110 mm/hr storm over 2.5 hours, with drainage capacity reduced by 40% and pump 1 failed."
    Output:
        {
            "rainfall_mm_hr": 110.0,
            "duration_hours": 2.5,
            "drainage_capacity_factor": 0.6,
            "pump_overrides": {"PUMP-1": 0.0},
            "profile_type": "chicago"
        }
    """

    @staticmethod
    def parse_prompt(prompt: str) -> Dict[str, Any]:
        text = prompt.lower().strip()
        
        # Default scenario parameters
        params = {
            "rainfall_mm_hr": 80.0,
            "duration_hours": 2.0,
            "drainage_capacity_factor": 1.0,
            "pump_overrides": {},
            "profile_type": "chicago",
            "recognized_entities": []
        }
        
        # 1. Rainfall Intensity: e.g. "120 mm", "120 mm/hr", "85mm", "rain of 90"
        rain_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:mm(?:/hr)?|millimeters)', text)
        if rain_match:
            val = float(rain_match.group(1))
            params["rainfall_mm_hr"] = min(max(val, 0.0), 250.0)
            params["recognized_entities"].append(f"Rainfall: {params['rainfall_mm_hr']} mm/hr")
        elif "heavy rain" in text or "severe storm" in text:
            params["rainfall_mm_hr"] = 120.0
            params["recognized_entities"].append("Rainfall: 120.0 mm/hr (Heavy Storm)")
        elif "light rain" in text or "drizzle" in text:
            params["rainfall_mm_hr"] = 25.0
            params["recognized_entities"].append("Rainfall: 25.0 mm/hr (Light Rain)")
            
        # 2. Duration: e.g. "3 hours", "2.5 hrs", "90 minutes", "4 hr"
        dur_hr_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)', text)
        dur_min_match = re.search(r'(\d+)\s*(?:minutes?|mins?)', text)
        
        if dur_hr_match:
            val = float(dur_hr_match.group(1))
            params["duration_hours"] = min(max(val, 0.5), 12.0)
            params["recognized_entities"].append(f"Duration: {params['duration_hours']} hours")
        elif dur_min_match:
            mins = float(dur_min_match.group(1))
            params["duration_hours"] = min(max(mins / 60.0, 0.5), 12.0)
            params["recognized_entities"].append(f"Duration: {params['duration_hours']:.1f} hours ({mins} min)")
            
        # 3. Drainage Capacity / Blockage: e.g. "drainage reduced by 40%", "40% drainage blockage", "capacity 60%", "drainage 50%", "choked by 30%"
        drain_block_front_match = re.search(r'(\d+)%\s+(?:drainage\s+)?(?:blockage|clogged|reduction)', text)
        drain_red_match = re.search(r'(?:drainage|capacity)\s+(?:reduced|decreased|blocked|choked)\s+by\s+(\d+)%', text)
        drain_cap_match = re.search(r'(?:drainage|capacity)\s+(?:at|of|is)?\s*(\d+)%', text)
        
        if drain_block_front_match:
            blockage = float(drain_block_front_match.group(1)) / 100.0
            cap = max(0.0, 1.0 - blockage)
            params["drainage_capacity_factor"] = round(cap, 2)
            params["recognized_entities"].append(f"Drainage: {params['drainage_capacity_factor']*100:.0f}% capacity ({int(blockage*100)}% blockage)")
        elif drain_red_match:
            reduction = float(drain_red_match.group(1)) / 100.0
            cap = max(0.0, 1.0 - reduction)
            params["drainage_capacity_factor"] = round(cap, 2)
            params["recognized_entities"].append(f"Drainage: {params['drainage_capacity_factor']*100:.0f}% capacity")
        elif drain_cap_match:
            cap = float(drain_cap_match.group(1)) / 100.0
            params["drainage_capacity_factor"] = round(min(max(cap, 0.0), 1.0), 2)
            params["recognized_entities"].append(f"Drainage: {params['drainage_capacity_factor']*100:.0f}% capacity")
        elif "blocked drainage" in text or "clogged pipes" in text:
            params["drainage_capacity_factor"] = 0.3
            params["recognized_entities"].append("Drainage: 30% capacity (Clogged)")
            
        # 4. Pump Failure / Overrides: e.g. "pump 1 failed", "pump 2 offline", "all pumps disabled"
        if "pump 1" in text and ("fail" in text or "offline" in text or "down" in text or "stop" in text):
            params["pump_overrides"]["PUMP-1"] = 0.0
            params["recognized_entities"].append("Pump 1: OFFLINE (0%)")
        if "pump 2" in text and ("fail" in text or "offline" in text or "down" in text or "stop" in text):
            params["pump_overrides"]["PUMP-2"] = 0.0
            params["recognized_entities"].append("Pump 2: OFFLINE (0%)")
        if "all pumps" in text and ("fail" in text or "offline" in text or "stop" in text):
            params["pump_overrides"]["PUMP-1"] = 0.0
            params["pump_overrides"]["PUMP-2"] = 0.0
            params["recognized_entities"].append("All Pumps: OFFLINE (0%)")
        if "boost pump" in text or "pump full power" in text or "pump 100%" in text:
            params["pump_overrides"]["PUMP-1"] = 1.0
            params["pump_overrides"]["PUMP-2"] = 1.0
            params["recognized_entities"].append("All Pumps: BOOSTED (100%)")
            
        # 5. Profile Type
        if "triangular" in text:
            params["profile_type"] = "triangular"
        elif "uniform" in text or "block" in text:
            params["profile_type"] = "block"
        else:
            params["profile_type"] = "chicago"
            
        return params

