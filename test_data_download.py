import os
import sys
import requests
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import shutil
import openmeteo_requests
import requests_cache
from retry_requests import retry

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WeatherDataTester:
    def __init__(self, test_dir='test_data'):
        self.test_dir = test_dir
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.openmeteo = openmeteo_requests.Client(session=retry_session)
        
        # Base URL for historical weather data
        self.url = "https://archive-api.open-meteo.com/v1/archive"

    def test_data_download(self, latitude, longitude, start_date, end_date):
        """Test downloading weather data from Open-Meteo."""
        try:
            # Parameters for the API request
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_date,
                "end_date": end_date,
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
                "timezone": "GMT"
            }
            
            logger.info(f"Testing data download for coordinates ({latitude}, {longitude})")
            logger.info(f"Time period: {start_date} to {end_date}")
            
            # Make the API request
            responses = self.openmeteo.weather_api(self.url, params=params)
            response = responses[0]
            
            # Log basic information
            logger.info(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
            logger.info(f"Elevation {response.Elevation()} m asl")
            logger.info(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
            
            # Process hourly data
            hourly = response.Hourly()
            hourly_data = {
                "date": pd.date_range(
                    start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=hourly.Interval()),
                    inclusive="left"
                )
            }
            
            # Get all variables
            for idx, variable in enumerate(params["hourly"]):
                values = hourly.Variables(idx).ValuesAsNumpy()
                hourly_data[variable] = values
                
                # Log statistics for each variable
                logger.info(f"\nStatistics for {variable}:")
                logger.info(f"  Min: {np.min(values)}")
                logger.info(f"  Max: {np.max(values)}")
                logger.info(f"  Mean: {np.mean(values)}")
                logger.info(f"  Missing values: {np.sum(np.isnan(values))}")
            
            # Create DataFrame
            df = pd.DataFrame(data=hourly_data)
            
            # Save to CSV for inspection
            output_file = os.path.join(self.test_dir, f"weather_data_{start_date}_{end_date}.csv")
            df.to_csv(output_file, index=False)
            logger.info(f"\nData saved to {output_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error downloading data: {str(e)}")
            return False

    def run_tests(self):
        """Run all tests."""
        logger.info("Starting weather data download tests...")
        
        # Test a single location and time period first
        test_location = (52.52, 13.41, "Berlin")  # Berlin
        test_period = ("2018-01-22", "2018-12-31")  # Test period
        
        logger.info(f"\nTesting location: {test_location[2]}")
        self.test_data_download(test_location[0], test_location[1], test_period[0], test_period[1])
        
        # If successful, test more locations and periods
        test_locations = [
            (40.71, -74.01, "New York"),  # New York
            (35.68, 139.77, "Tokyo")  # Tokyo
        ]
        
        test_periods = [
            ("2017-01-01", "2017-12-31"),
            ("2019-01-01", "2019-12-31")
        ]
        
        for lat, lon, name in test_locations:
            logger.info(f"\nTesting location: {name}")
            for start_date, end_date in test_periods:
                self.test_data_download(lat, lon, start_date, end_date)

    def cleanup(self):
        """Clean up test files."""
        try:
            shutil.rmtree(self.test_dir)
            logger.info("Test directory cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up test directory: {str(e)}")

if __name__ == "__main__":
    # Create and run tests
    tester = WeatherDataTester()
    try:
        tester.run_tests()
    finally:
        tester.cleanup() 