import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import os

np.random.seed(42)

CITIES = [
    {"city": "Jakarta",       "country": "Indonesia",   "lat": -6.21,  "lon": 106.85, "climate": "tropical"},
    {"city": "Bali",          "country": "Indonesia",   "lat": -8.34,  "lon": 115.09, "climate": "tropical"},
    {"city": "Tokyo",         "country": "Japan",       "lat": 35.68,  "lon": 139.69, "climate": "temperate"},
    {"city": "Sydney",        "country": "Australia",   "lat": -33.87, "lon": 151.21, "climate": "subtropical"},
    {"city": "London",        "country": "UK",          "lat": 51.51,  "lon": -0.13,  "climate": "oceanic"},
    {"city": "New York",      "country": "USA",         "lat": 40.71,  "lon": -74.01, "climate": "continental"},
    {"city": "Dubai",         "country": "UAE",         "lat": 25.20,  "lon": 55.27,  "climate": "desert"},
    {"city": "Paris",         "country": "France",      "lat": 48.86,  "lon": 2.35,   "climate": "oceanic"},
    {"city": "Mumbai",        "country": "India",       "lat": 19.08,  "lon": 72.88,  "climate": "tropical"},
    {"city": "São Paulo",     "country": "Brazil",      "lat": -23.55, "lon": -46.63, "climate": "subtropical"},
    {"city": "Cairo",         "country": "Egypt",       "lat": 30.04,  "lon": 31.24,  "climate": "desert"},
    {"city": "Toronto",       "country": "Canada",      "lat": 43.65,  "lon": -79.38, "climate": "continental"},
    {"city": "Singapore",     "country": "Singapore",   "lat": 1.35,   "lon": 103.82, "climate": "tropical"},
    {"city": "Moscow",        "country": "Russia",      "lat": 55.75,  "lon": 37.62,  "climate": "continental"},
    {"city": "Cape Town",     "country": "South Africa","lat": -33.93, "lon": 18.42,  "climate": "mediterranean"},
    {"city": "Seoul",         "country": "South Korea", "lat": 37.57,  "lon": 126.98, "climate": "continental"},
    {"city": "Mexico City",   "country": "Mexico",      "lat": 19.43,  "lon": -99.13, "climate": "subtropical"},
    {"city": "Berlin",        "country": "Germany",     "lat": 52.52,  "lon": 13.41,  "climate": "oceanic"},
    {"city": "Bangkok",       "country": "Thailand",    "lat": 13.76,  "lon": 100.50, "climate": "tropical"},
    {"city": "Lagos",         "country": "Nigeria",     "lat": 6.52,   "lon": 3.38,   "climate": "tropical"},
]

CLIMATE_PARAMS = {
    "tropical":      {"base_temp": 28, "temp_var": 4,  "base_rain": 180, "base_humidity": 80, "base_wind": 15},
    "subtropical":   {"base_temp": 22, "temp_var": 8,  "base_rain": 100, "base_humidity": 65, "base_wind": 18},
    "temperate":     {"base_temp": 14, "temp_var": 12, "base_rain": 90,  "base_humidity": 60, "base_wind": 20},
    "oceanic":       {"base_temp": 11, "temp_var": 8,  "base_rain": 70,  "base_humidity": 75, "base_wind": 22},
    "continental":   {"base_temp": 8,  "temp_var": 20, "base_rain": 60,  "base_humidity": 55, "base_wind": 17},
    "desert":        {"base_temp": 32, "temp_var": 14, "base_rain": 10,  "base_humidity": 25, "base_wind": 25},
    "mediterranean": {"base_temp": 18, "temp_var": 10, "base_rain": 50,  "base_humidity": 60, "base_wind": 19},
}

CONDITIONS = ["Sunny", "Partly Cloudy", "Cloudy", "Rainy", "Stormy", "Foggy", "Windy"]

rows = []
start = datetime(2025, 1, 1)

for city_info in CITIES:
    params = CLIMATE_PARAMS[city_info["climate"]]
    for day_offset in range(181):  # Jan–Jun 2025
        date = start + timedelta(days=day_offset)
        month = date.month
        seasonal = np.sin((month - 3) / 12 * 2 * np.pi)  # seasonal wave

        # Flip seasons for southern hemisphere
        if city_info["lat"] < 0:
            seasonal = -seasonal

        temp = (params["base_temp"] + seasonal * params["temp_var"] / 2
                + np.random.normal(0, params["temp_var"] * 0.15))
        humidity = np.clip(params["base_humidity"] + np.random.normal(0, 8), 10, 100)
        rain = max(0, params["base_rain"] / 30 * (1 + seasonal * 0.3) + np.random.normal(0, 3))
        wind = max(0, params["base_wind"] + np.random.normal(0, 4))
        pressure = np.random.normal(1013, 8)
        uv_index = max(0, round(8 - abs(city_info["lat"]) / 15 + seasonal * 2 + np.random.normal(0, 1), 1))

        # Condition probabilities
        if rain > 8:
            cond = np.random.choice(["Rainy", "Stormy"], p=[0.7, 0.3])
        elif humidity > 85:
            cond = np.random.choice(["Cloudy", "Foggy"], p=[0.7, 0.3])
        elif rain > 2:
            cond = "Partly Cloudy"
        else:
            cond = np.random.choice(["Sunny", "Partly Cloudy", "Windy"], p=[0.6, 0.3, 0.1])

        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "month": month,
            "day_of_year": day_offset + 1,
            "city": city_info["city"],
            "country": city_info["country"],
            "lat": city_info["lat"],
            "lon": city_info["lon"],
            "climate_zone": city_info["climate"],
            "temperature_c": round(temp, 1),
            "humidity_pct": round(humidity, 1),
            "rainfall_mm": round(rain, 1),
            "wind_speed_kmh": round(wind, 1),
            "pressure_hpa": round(pressure, 1),
            "uv_index": uv_index,
            "condition": cond,
        })

df = pd.DataFrame(rows)
os.makedirs("data", exist_ok=True)
df.to_csv("data/weather_2025.csv", index=False)
print(f"Generated {len(df)} rows for {len(CITIES)} cities")
print(df.head())
