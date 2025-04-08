import os
import requests
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_isd_lite():
    """Test ISD-Lite downloader for one station."""
    def download_isd_lite(year, station_id, save_dir="data/raw/isd_test"):
        url = f"https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/{year}/{station_id}-{year}.gz"
        save_path = Path(save_dir) / f"{station_id}-{year}.gz"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(save_path, "wb") as f:
                f.write(response.content)
            logger.info(f"Downloaded {station_id} for {year} to {save_path}")
            return str(save_path)
        except Exception as e:
            logger.error(f"Error downloading {station_id} for {year}: {e}")
            return None
    
    station_id = "037720-99999"  # Heathrow
    for year in range(2018, 2019):  # Just 2018 for speed
        download_isd_lite(year, station_id)

if __name__ == "__main__":
    test_isd_lite()