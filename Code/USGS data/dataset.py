import requests
import pandas as pd
import matplotlib.pyplot as plt

# ---- SETTINGS ----
site = "02169506"
parameter = "00065"

start = "2026-08-26T08:00:00-05:00"
end   = "2026-09-17T08:00:00-15:00"

url = "https://waterservices.usgs.gov/nwis/iv/"

params = {
    "format": "json",
    "sites": site,
    "parameterCd": parameter,
    "startDT": start,
    "endDT": end,
    "siteStatus": "all"
}

# ---- GET DATA FROM USGS ----
response = requests.get(url, params=params)
response.raise_for_status()

data = response.json()

# ---- EXTRACT THE MEASUREMENTS ----
time_series = data["value"]["timeSeries"]

if len(time_series) == 0:
    print("No data found.")
else:
    values = time_series[0]["values"][0]["value"]

    # ---- CREATE DATASET ----
    df = pd.DataFrame(values)

    df["dateTime"] = pd.to_datetime(df["dateTime"])
    df["value"] = pd.to_numeric(df["value"])

    # ---- CLEAN DATASET ----
    df = df[["dateTime", "value"]]
    df = df.rename(columns={"value": "gage_height_ft"})

    # ---- SAVE DATASET ----
    df.to_csv("usgs_water_height_dataset.csv", index=False)

    print(df.head())
    print("Dataset saved as usgs_water_height_dataset.csv")

    plt.figure(figsize=(10, 5))
    plt.plot(df["dateTime"], df["gage_height_ft"], linewidth=2)

    plt.xlabel("Hours from start")
    plt.ylabel("Stage (ft)")
    plt.title("USGS 02169506 Rocky Branch (Whaley St)\nStage Height (2-hour intervals)")
    plt.grid(True)
    plt.show()
