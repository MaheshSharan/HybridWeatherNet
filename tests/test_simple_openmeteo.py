import os
import sys
from src.data.simple_openmeteo import SimpleOpenMeteoDownloader

# Setup paths
PROJECT_DIR = os.getcwd()
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
os.makedirs(os.path.join(DATA_DIR, 'raw', 'openmeteo'), exist_ok=True)

# Initialize downloader
downloader = SimpleOpenMeteoDownloader(
    output_dir=os.path.join(DATA_DIR, 'raw', 'openmeteo'),
    start_date='2018-01-01',
    end_date='2018-12-31'
)

# Test location
location = {
    "name": "Berlin",
    "lat": 52.52,
    "lon": 13.41
}

# Download data
print(f"Downloading data for {location['name']}...")
output_file = downloader.download_data(
    latitude=location["lat"],
    longitude=location["lon"],
    location_name=location["name"]
)

if output_file:
    print(f"Successfully downloaded data to {output_file}")
else:
    print("Failed to download data") 