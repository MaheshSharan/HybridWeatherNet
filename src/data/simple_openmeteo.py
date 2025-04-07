import os
import logging
import requests
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any

class SimpleOpenMeteoDownloader:
    """Simple downloader for Open-Meteo weather data."""
    
    def __init__(self, output_dir: str, start_date: str, end_date: str):
        """Initialize the downloader.
        
        Args:
            output_dir: Directory to save downloaded data
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        """
        self.output_dir = output_dir
        self.start_date = start_date
        self.end_date = end_date
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # API parameters
        self.base_url = "https://archive-api.open-meteo.com/v1/archive"
        self.params = {
            "latitude": None,
            "longitude": None,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "wind_speed_10m",
                "wind_direction_10m",
                "cloud_cover_low",
                "cloud_cover_mid",
                "cloud_cover_high"
            ],
            "timezone": "auto"
        }
    
    def download_data(self, latitude: float, longitude: float, location_name: str) -> Optional[str]:
        """Download weather data for a specific location.
        
        Args:
            latitude: Latitude of the location
            longitude: Longitude of the location
            location_name: Name of the location (used for output file)
            
        Returns:
            Path to the downloaded file if successful, None otherwise
        """
        try:
            self.logger.info(f"Downloading Open-Meteo data for {location_name}...")
            
            # Update parameters
            self.params["latitude"] = latitude
            self.params["longitude"] = longitude
            
            # Make API request
            response = requests.get(self.base_url, params=self.params)
            response.raise_for_status()
            data = response.json()
            
            # Process data
            df = self._process_data(data)
            
            # Save to file
            output_file = os.path.join(
                self.output_dir,
                f"{location_name}_{self.start_date}_{self.end_date}.csv"
            )
            df.to_csv(output_file, index=False)
            
            self.logger.info(f"Successfully downloaded data for {location_name}")
            return output_file
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error downloading data: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            return None
    
    def _process_data(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Process the API response data into a DataFrame.
        
        Args:
            data: API response data
            
        Returns:
            DataFrame with processed data
        """
        # Extract time and variables
        times = pd.to_datetime(data["hourly"]["time"])
        variables = {
            "temp": data["hourly"]["temperature_2m"],
            "rh": data["hourly"]["relative_humidity_2m"],
            "wind_speed": data["hourly"]["wind_speed_10m"],
            "wind_dir": data["hourly"]["wind_direction_10m"],
            "cloud_cover_low": data["hourly"]["cloud_cover_low"],
            "cloud_cover_mid": data["hourly"]["cloud_cover_mid"],
            "cloud_cover_high": data["hourly"]["cloud_cover_high"]
        }
        
        # Create DataFrame
        df = pd.DataFrame(variables, index=times)
        df.index.name = "date"
        df.reset_index(inplace=True)
        
        # Log statistics
        self._log_statistics(df)
        
        return df
    
    def _log_statistics(self, df: pd.DataFrame):
        """Log statistics about the downloaded data.
        
        Args:
            df: DataFrame with the downloaded data
        """
        for column in df.columns:
            if column != "date":
                stats = df[column].describe()
                self.logger.info(f"\nStatistics for {column}:")
                self.logger.info(f"  Count: {stats['count']}")
                self.logger.info(f"  Mean: {stats['mean']:.2f}")
                self.logger.info(f"  Std: {stats['std']:.2f}")
                self.logger.info(f"  Min: {stats['min']:.2f}")
                self.logger.info(f"  Max: {stats['max']:.2f}")

if __name__ == "__main__":
    # Example usage
    downloader = SimpleOpenMeteoDownloader(
        output_dir="data/raw/openmeteo",
        start_date="2018-01-01",
        end_date="2018-12-31"
    )
    
    # Test location
    location = {
        "name": "Berlin",
        "lat": 52.52,
        "lon": 13.41
    }
    
    # Download data
    output_file = downloader.download_data(
        latitude=location["lat"],
        longitude=location["lon"],
        location_name=location["name"]
    )
    
    if output_file:
        print(f"Successfully downloaded data to {output_file}")
    else:
        print("Failed to download data") 