"""
Tools for EcoHome Energy Advisor Agent
"""
import os
import json
import random
from datetime import datetime, timedelta
from typing import Dict, Any
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from models.energy import DatabaseManager

# Initialize database manager
db_manager = DatabaseManager()

# TODO: Implement get_weather_forecast tool
@tool
def get_weather_forecast(location: str, days: int = 3) -> Dict[str, Any]:
    """
    Get weather forecast for a specific location and number of days.
    
    Args:
        location (str): Location to get weather for (e.g., "San Francisco, CA")
        days (int): Number of days to forecast (1-7)
    
    Returns:
        Dict[str, Any]: Weather forecast data including temperature, conditions, and solar irradiance
        E.g:
        forecast = {
            "location": ...,
            "forecast_days": ...,
            "current": {
                "temperature_c": ...,
                "condition": random.choice(["sunny", "partly_cloudy", "cloudy"]),
                "humidity": ...,
                "wind_speed": ...
            },
            "hourly": [
                {
                    "hour": ..., # for hour in range(24)
                    "temperature_c": ...,
                    "condition": ...,
                    "solar_irradiance": ...,
                    "humidity": ...,
                    "wind_speed": ...
                },
            ]
        }
    """
    def _clamp_int(x: int, lo: int, hi: int)-> int:
        return max(lo, min(hi, x))
    
    def _mock_forecast(loc: str, d: int)-> Dict[str, Any]:
        from datetime import timezone
        import hashlib, math, random

        now= datetime.now(timezone.utc)
        seed_str = f"{loc.strip().lower()}|{now.strftime('%Y-%m-%d')}"
        seed= int(hashlib.sha256(seed_str.encode("utf-8")).hexdigest()[:8], 16)
        rng= random.Random(seed)

        month= now.month
        seasonal = {
            1: 6, 2: 7, 3: 10, 4: 13, 5: 17, 6: 21,
            7: 24, 8: 24, 9: 20, 10: 15, 11: 10, 12: 7
        }[month]

        loc_l= loc.lower()

        if any(k in loc_l for k in ["oslo", "helsinki", "stockholm", "reykjavik"]):
            seasonal -= 6
        
        if any(k in loc_l for k in ["madrid", "sevilla", "valencia", "barcelona", "lisbon"]):
            seasonal += 3
        
        if any(k in loc_l for k in ["cairo", "dubai", "riyadh", "miami"]):
            seasonal += 7

        conditions = ["sunny", "partly_cloudy", "cloudy", "rainy"]
        base_condition= rng.choices(conditions, weights= [35, 35, 20, 10], k=1)[0]

        def solar_irradiance_wm2(hour_local: int, cloud_factor: float)-> float:
            if hour_local<7 or hour_local>19:
                return 0.0
            
            x= (hour_local-13)/4.0
            clear_peak= 800.0
            return max(0.0, clear_peak*math.exp(-(x*x))*cloud_factor)
        
        hourly= []

        start= now.replace(minute= 0, second=0, microsecond=0)
        total_hours= 24*d

        for h in range(total_hours):
            t= start + timedelta(hours=h)
            hour= t.hour

            diurnal= 4.5 * math.sin((hour - 6)*math.pi/12)
            noise= rng.uniform(-0.6, 0.6)
            temp_c= seasonal + diurnal + noise


            if base_condition== "sunny":
                cloud= rng.uniform(0.0, 0.2)
            elif base_condition=="partly_cloudy":
                cloud= rng.uniform(0.2, 0.5)
            elif base_condition== "cloudy":
                cloud= rng.uniform(0.5, 0.85)
            else: #rainy
                cloud= rng.uniform(0.7, 1.0)
            
            cloud_factor= 1.0 - cloud
            irr= solar_irradiance_wm2(hour, cloud_factor=cloud_factor)

            cond= base_condition
            if rng.random()< 0.08:
                cond= rng.choice(conditions)
            
            hourly.append({
                "timestamp": t.isoformat(),
                "hour":hour,
                "temperature_c": round(temp_c, 1),
                "condition": cond,
                "humidity": int(_clamp_int(int(rng.gauss(55,12)), 20, 95)),
                "wind_speed": round(max(0.0, rng.gauss(55,12)), 1),
                "solar_irradiance": round(irr, 1)
            })
        
        current= hourly[0].copy()
        current.pop("timestamp", None)
        current.pop("hour", None)
    
        return {
            "location": loc,
            "forecast_days":d,
            "source":"mock",
            "current": current,
            "hourly": hourly

        }
    
    # ---------- validate inputs ----------
    try:
        if not isinstance(location, str) or not location.strip():
            return {"error": "location must be a non-empty string"}
        if not isinstance(days, int):
            return {"error": "days must be an integer"}
        days = _clamp_int(days, 1, 7)
    except Exception as e:
        return {"error": f"invalid inputs: {e}"}

    # ---------- try OpenWeatherMap ----------
    import os
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # No key → fallback mock
        return _mock_forecast(location, days)

    # Use requests if available, else urllib
    try:
        import requests  # type: ignore
        http_get = "requests"
    except Exception:
        requests = None
        http_get = "urllib"

    try:
        # 1) Geocoding: location -> lat/lon
        geo_url = "OPENAI_BASE_URL"
        geo_params = {"q": location, "limit": 1, "appid": api_key}

        if requests:
            geo_resp = requests.get(geo_url, params=geo_params, timeout=10)
            geo_resp.raise_for_status()
            geo = geo_resp.json()
        else:
            import urllib.parse, urllib.request, json
            url = geo_url + "?" + urllib.parse.urlencode(geo_params)
            with urllib.request.urlopen(url, timeout=10) as r:
                geo = json.loads(r.read().decode("utf-8"))

        if not geo:
            return {"error": f"location not found: {location}"}

        lat = geo[0]["lat"]
        lon = geo[0]["lon"]
        resolved_name = ", ".join([x for x in [geo[0].get("name"), geo[0].get("state"), geo[0].get("country")] if x])

        # 2) Forecast (One Call 3.0)
        # Nota: One Call 3.0 suele requerir plan/permiso; si falla, cae a mock.
        onecall_url = "https://api.openweathermap.org/data/3.0/onecall"
        onecall_params = {
            "lat": lat,
            "lon": lon,
            "exclude": "minutely,alerts",
            "units": "metric",
            "appid": api_key
        }

        if requests:
            oc_resp = requests.get(onecall_url, params=onecall_params, timeout=10)
            oc_resp.raise_for_status()
            oc = oc_resp.json()
        else:
            import urllib.parse, urllib.request, json
            url = onecall_url + "?" + urllib.parse.urlencode(onecall_params)
            with urllib.request.urlopen(url, timeout=10) as r:
                oc = json.loads(r.read().decode("utf-8"))

        # Build response: we want ~24*days hours
        hourly_raw = oc.get("hourly", [])
        take = min(len(hourly_raw), 24 * days)

        hourly = []
        for i in range(take):
            h = hourly_raw[i]
            # OWM gives "dt" (unix), "temp", "humidity", "wind_speed", "clouds", "uvi", "weather"
            ts = datetime.utcfromtimestamp(h["dt"]).isoformat() + "Z"
            clouds = float(h.get("clouds", 0)) / 100.0  # 0..1
            # crude irradiance estimate: scale with daylight proxy (uvi) and clouds
            uvi = float(h.get("uvi", 0.0))
            # Map uvi (0..~10) -> peak W/m2 rough (0..900), then reduce by clouds
            irr = max(0.0, min(900.0, uvi * 90.0)) * (1.0 - 0.75 * clouds)

            cond = "unknown"
            w = h.get("weather") or []
            if w and isinstance(w, list):
                main = (w[0].get("main") or "").lower()
                if "clear" in main:
                    cond = "sunny"
                elif "cloud" in main:
                    cond = "cloudy" if clouds > 0.5 else "partly_cloudy"
                elif "rain" in main or "drizzle" in main or "thunder" in main:
                    cond = "rainy"
                else:
                    cond = main or "unknown"

            temp_c = float(h.get("temp"))
            humidity = int(h.get("humidity", 0))
            wind = float(h.get("wind_speed", 0.0))

            hour_localish = datetime.utcfromtimestamp(h["dt"]).hour

            hourly.append({
                "timestamp": ts,
                "hour": hour_localish,
                "temperature_c": round(temp_c, 1),
                "condition": cond,
                "humidity": humidity,
                "wind_speed": round(wind, 1),
                "solar_irradiance": round(irr, 1),
            })

        # current from "current"
        cur = oc.get("current", {})
        cur_clouds = float(cur.get("clouds", 0)) / 100.0
        cur_uvi = float(cur.get("uvi", 0.0))
        cur_irr = max(0.0, min(900.0, cur_uvi * 90.0)) * (1.0 - 0.75 * cur_clouds)

        cur_cond = "unknown"
        cw = cur.get("weather") or []
        if cw and isinstance(cw, list):
            main = (cw[0].get("main") or "").lower()
            if "clear" in main:
                cur_cond = "sunny"
            elif "cloud" in main:
                cur_cond = "cloudy" if cur_clouds > 0.5 else "partly_cloudy"
            elif "rain" in main or "drizzle" in main or "thunder" in main:
                cur_cond = "rainy"
            else:
                cur_cond = main or "unknown"

        current = {
            "temperature_c": round(float(cur.get("temp", hourly[0]["temperature_c"] if hourly else 0.0)), 1),
            "condition": cur_cond,
            "humidity": int(cur.get("humidity", hourly[0]["humidity"] if hourly else 0)),
            "wind_speed": round(float(cur.get("wind_speed", hourly[0]["wind_speed"] if hourly else 0.0)), 1),
            "solar_irradiance": round(cur_irr, 1),
        }

        return {
            "location": resolved_name or location,
            "forecast_days": days,
            "source": "openweathermap",
            "http_client": http_get,
            "current": current,
            "hourly": hourly
        }

    except Exception as e:
        # If API fails for any reason, fallback so agent still works
        fallback = _mock_forecast(location, days)
        fallback["source"] = "mock_fallback"
        fallback["api_error"] = str(e)
        return fallback
 

# TODO: Implement get_electricity_prices tool
@tool
def get_electricity_prices(date: str = None) -> Dict[str, Any]:
    """
    Get electricity prices for a specific date or current day.
    
    Args:
        date (str): Date in YYYY-MM-DD format (defaults to today)
    
    Returns:
        Dict[str, Any]: Electricity pricing data with hourly rates 
        E.g: 
        prices = {
            "date": ...,
            "pricing_type": "time_of_use",
            "currency": "USD",
            "unit": "per_kWh",
            "hourly_rates": [
                {
                    "hour": .., # for hour in range(24)
                    "rate": ..,
                    "period": ..,
                    "demand_charge": ...
                }
            ]
        }
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    # Mock electricity pricing - in real implementation, this would call a pricing API
    # Use a base price per kWh    
    # Then generate hourly rates with peak/off-peak pricing
    # Peak normally between 6 and 22...
    # demand_charge should be 0 if off-peak

    return 

@tool
def query_energy_usage(start_date: str, end_date: str, device_type: str = None) -> Dict[str, Any]:
    """
    Query energy usage data from the database for a specific date range.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        device_type (str): Optional device type filter (e.g., "EV", "HVAC", "appliance")
    
    Returns:
        Dict[str, Any]: Energy usage data with consumption details
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
        
        records = db_manager.get_usage_by_date_range(start_dt, end_dt)
        
        if device_type:
            records = [r for r in records if r.device_type == device_type]
        
        usage_data = {
            "start_date": start_date,
            "end_date": end_date,
            "device_type": device_type,
            "total_records": len(records),
            "total_consumption_kwh": round(sum(r.consumption_kwh for r in records), 2),
            "total_cost_usd": round(sum(r.cost_usd or 0 for r in records), 2),
            "records": []
        }
        
        for record in records:
            usage_data["records"].append({
                "timestamp": record.timestamp.isoformat(),
                "consumption_kwh": record.consumption_kwh,
                "device_type": record.device_type,
                "device_name": record.device_name,
                "cost_usd": record.cost_usd
            })
        
        return usage_data
    except Exception as e:
        return {"error": f"Failed to query energy usage: {str(e)}"}

@tool
def query_solar_generation(start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Query solar generation data from the database for a specific date range.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
    
    Returns:
        Dict[str, Any]: Solar generation data with production details
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
        
        records = db_manager.get_generation_by_date_range(start_dt, end_dt)
        
        generation_data = {
            "start_date": start_date,
            "end_date": end_date,
            "total_records": len(records),
            "total_generation_kwh": round(sum(r.generation_kwh for r in records), 2),
            "average_daily_generation": round(sum(r.generation_kwh for r in records) / max(1, (end_dt - start_dt).days), 2),
            "records": []
        }
        
        for record in records:
            generation_data["records"].append({
                "timestamp": record.timestamp.isoformat(),
                "generation_kwh": record.generation_kwh,
                "weather_condition": record.weather_condition,
                "temperature_c": record.temperature_c,
                "solar_irradiance": record.solar_irradiance
            })
        
        return generation_data
    except Exception as e:
        return {"error": f"Failed to query solar generation: {str(e)}"}

@tool
def get_recent_energy_summary(hours: int = 24) -> Dict[str, Any]:
    """
    Get a summary of recent energy usage and solar generation.
    
    Args:
        hours (int): Number of hours to look back (default 24)
    
    Returns:
        Dict[str, Any]: Summary of recent energy data
    """
    try:
        usage_records = db_manager.get_recent_usage(hours)
        generation_records = db_manager.get_recent_generation(hours)
        
        summary = {
            "time_period_hours": hours,
            "usage": {
                "total_consumption_kwh": round(sum(r.consumption_kwh for r in usage_records), 2),
                "total_cost_usd": round(sum(r.cost_usd or 0 for r in usage_records), 2),
                "device_breakdown": {}
            },
            "generation": {
                "total_generation_kwh": round(sum(r.generation_kwh for r in generation_records), 2),
                "average_weather": "sunny" if generation_records else "unknown"
            }
        }
        
        # Calculate device breakdown
        for record in usage_records:
            device = record.device_type or "unknown"
            if device not in summary["usage"]["device_breakdown"]:
                summary["usage"]["device_breakdown"][device] = {
                    "consumption_kwh": 0,
                    "cost_usd": 0,
                    "records": 0
                }
            summary["usage"]["device_breakdown"][device]["consumption_kwh"] += record.consumption_kwh
            summary["usage"]["device_breakdown"][device]["cost_usd"] += record.cost_usd or 0
            summary["usage"]["device_breakdown"][device]["records"] += 1
        
        # Round the breakdown values
        for device_data in summary["usage"]["device_breakdown"].values():
            device_data["consumption_kwh"] = round(device_data["consumption_kwh"], 2)
            device_data["cost_usd"] = round(device_data["cost_usd"], 2)
        
        return summary
    except Exception as e:
        return {"error": f"Failed to get recent energy summary: {str(e)}"}

@tool
def search_energy_tips(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search for energy-saving tips and best practices using RAG.
    
    Args:
        query (str): Search query for energy tips
        max_results (int): Maximum number of results to return
    
    Returns:
        Dict[str, Any]: Relevant energy tips and best practices
    """
    try:
        # Initialize vector store if it doesn't exist
        persist_directory = "data/vectorstore"
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)
        
        # Load documents if vector store doesn't exist
        if not os.path.exists(os.path.join(persist_directory, "chroma.sqlite3")):
            # Load documents
            documents = []
            for doc_path in ["data/documents/tip_device_best_practices.txt", "data/documents/tip_energy_savings.txt"]:
                if os.path.exists(doc_path):
                    loader = TextLoader(doc_path)
                    docs = loader.load()
                    documents.extend(docs)
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)
            
            # Create vector store
            embeddings = OpenAIEmbeddings()
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=persist_directory
            )
        else:
            # Load existing vector store
            embeddings = OpenAIEmbeddings()
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )
        
        # Search for relevant documents
        docs = vectorstore.similarity_search(query, k=max_results)
        
        results = {
            "query": query,
            "total_results": len(docs),
            "tips": []
        }
        
        for i, doc in enumerate(docs):
            results["tips"].append({
                "rank": i + 1,
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "relevance_score": "high" if i < 2 else "medium" if i < 4 else "low"
            })
        
        return results
    except Exception as e:
        return {"error": f"Failed to search energy tips: {str(e)}"}

@tool
def calculate_energy_savings(device_type: str, current_usage_kwh: float, 
                           optimized_usage_kwh: float, price_per_kwh: float = 0.12) -> Dict[str, Any]:
    """
    Calculate potential energy savings from optimization.
    
    Args:
        device_type (str): Type of device being optimized
        current_usage_kwh (float): Current energy usage in kWh
        optimized_usage_kwh (float): Optimized energy usage in kWh
        price_per_kwh (float): Price per kWh (default 0.12)
    
    Returns:
        Dict[str, Any]: Savings calculation results
    """
    savings_kwh = current_usage_kwh - optimized_usage_kwh
    savings_usd = savings_kwh * price_per_kwh
    savings_percentage = (savings_kwh / current_usage_kwh) * 100 if current_usage_kwh > 0 else 0
    
    return {
        "device_type": device_type,
        "current_usage_kwh": current_usage_kwh,
        "optimized_usage_kwh": optimized_usage_kwh,
        "savings_kwh": round(savings_kwh, 2),
        "savings_usd": round(savings_usd, 2),
        "savings_percentage": round(savings_percentage, 1),
        "price_per_kwh": price_per_kwh,
        "annual_savings_usd": round(savings_usd * 365, 2)
    }


TOOL_KIT = [
    get_weather_forecast,
    get_electricity_prices,
    query_energy_usage,
    query_solar_generation,
    get_recent_energy_summary,
    search_energy_tips,
    calculate_energy_savings
]
