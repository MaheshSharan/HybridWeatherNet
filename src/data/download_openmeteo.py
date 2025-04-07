import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import openmeteo_requests
import requests_cache
import retry_requests
from typing import List, Dict, Tuple, Optional
from pathlib import Path

class OpenMeteoDownloader:
    """Class to handle downloading weather data from Open-Meteo API."""
    
    def __init__(self, output_dir: str, start_date: str, end_date: str):
        """Initialize the downloader with output directory and date range.
        
        Args:
            output_dir: Directory to save downloaded data
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        """
        self.output_dir = Path(output_dir)
        self.start_date = start_date
        self.end_date = end_date
        
        # Setup the Open-Meteo API client with cache and retry on error
        self.cache_session = requests_cache.CachingSession('.cache', expire_after=3600)
        retry_session = retry_requests.RetrySession(retries=5, backoff_factor=0.5)
        self.openmeteo = openmeteo_requests.Client(session=retry_session)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def download_data(self, latitude: float, longitude: float, location_name: str) -> str:
        """Download weather data for a specific location.
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            location_name: Name of the location for file naming
            
        Returns:
            Path to the saved data file
        """
        self.logger.info(f"Downloading data for {location_name} ({latitude}, {longitude})")
        
        # Setup the API parameters
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "wind_speed_10m",
                "wind_direction_10m",
                "cloud_cover",
                "cloud_cover_low",
                "cloud_cover_mid",
                "cloud_cover_high"
            ],
            "timezone": "UTC"
        }
        
        try:
            responses = self.openmeteo.weather_api(url, params=params)
            response = responses[0]
            
            # Process hourly data
            hourly = response.Hourly()
            hourly_data = {
                "time": pd.date_range(
                    start=pd.to_datetime(hourly.Time(), unit="s"),
                    end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
                    freq=pd.Timedelta(seconds=hourly.Interval()),
                    inclusive="left"
                ),
                "temperature": hourly.Variables(0).ValuesAsNumpy(),
                "relative_humidity": hourly.Variables(1).ValuesAsNumpy(),
                "wind_speed": hourly.Variables(2).ValuesAsNumpy(),
                "wind_direction": hourly.Variables(3).ValuesAsNumpy(),
                "cloud_cover": hourly.Variables(4).ValuesAsNumpy(),
                "cloud_cover_low": hourly.Variables(5).ValuesAsNumpy(),
                "cloud_cover_mid": hourly.Variables(6).ValuesAsNumpy(),
                "cloud_cover_high": hourly.Variables(7).ValuesAsNumpy()
            }
            
            # Create DataFrame
            df = pd.DataFrame(hourly_data)
            df.set_index('time', inplace=True)
            
            # Convert temperature from °C to Kelvin
            df['temperature'] += 273.15
            
            # Save to file
            output_file = self.output_dir / f"{location_name}_{self.start_date}_{self.end_date}.csv"
            df.to_csv(output_file)
            
            self.logger.info(f"Data saved to {output_file}")
            self._log_statistics(df)
            
            return str(output_file)
            
        except Exception as e:
            self.logger.error(f"Error downloading data for {location_name}: {str(e)}")
            raise
    
    def _log_statistics(self, df: pd.DataFrame) -> None:
        """Log basic statistics for the downloaded data."""
        for column in df.columns:
            stats = df[column].describe()
            self.logger.info(f"\nStatistics for {column}:")
            self.logger.info(f"  Min: {stats['min']}")
            self.logger.info(f"  Max: {stats['max']}")
            self.logger.info(f"  Mean: {stats['mean']}")
            self.logger.info(f"  Missing values: {df[column].isna().sum()}")

def main():
    """Main function to demonstrate usage."""
    downloader = OpenMeteoDownloader(
        output_dir="weather_data",
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    
    # Example locations
    locations = [
        {"name": "Berlin", "lat": 52.52, "lon": 13.41},
        {"name": "New_York", "lat": 40.71, "lon": -74.01},
        {"name": "Tokyo", "lat": 35.68, "lon": 139.77}
    ]
    
    for location in locations:
        try:
            downloader.download_data(
                latitude=location["lat"],
                longitude=location["lon"],
                location_name=location["name"]
            )
        except Exception as e:
            logging.error(f"Failed to download data for {location['name']}: {str(e)}")

if __name__ == "__main__":
    main() 