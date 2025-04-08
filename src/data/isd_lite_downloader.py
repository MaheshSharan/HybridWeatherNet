import os
import gzip
import requests
import pandas as pd
import logging
from pathlib import Path
from typing import Optional, List
from datetime import datetime

class ISDLiteDownloader:
    """Class to download weather data from NOAA ISD-Lite."""
    
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
        
        # ISD-Lite base URL
        self.base_url = "https://www.ncei.noaa.gov/pub/data/noaa/isd-lite"
        
        # Convert dates to datetime objects
        self.start = datetime.strptime(start_date, '%Y-%m-%d')
        self.end = datetime.strptime(end_date, '%Y-%m-%d')
    
    def download_station_data(self, station_id: str) -> Optional[List[str]]:
        """Download data for a specific station.
        
        Args:
            station_id: ISD-Lite station ID (e.g., '037720-99999')
            
        Returns:
            List of paths to downloaded files if successful, None otherwise
        """
        try:
            self.logger.info(f"Downloading ISD-Lite data for station {station_id}...")
            
            downloaded_files = []
            
            # Download data for each year in the range
            for year in range(self.start.year, self.end.year + 1):
                # Construct URL and save path
                url = f"{self.base_url}/{year}/{station_id}-{year}.gz"
                save_path = Path(self.output_dir) / f"{station_id}-{year}.gz"
                
                # Download file
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                # Save file
                with open(save_path, "wb") as f:
                    f.write(response.content)
                
                self.logger.info(f"Downloaded {station_id} for {year} to {save_path}")
                downloaded_files.append(str(save_path))
            
            return downloaded_files
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error downloading data: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            return None
    
    def process_station_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """Process a downloaded ISD-Lite file into a DataFrame.
        
        Args:
            file_path: Path to the downloaded .gz file
            
        Returns:
            Processed DataFrame if successful, None otherwise
        """
        try:
            # Read the gzipped file
            with gzip.open(file_path, 'rt') as f:
                df = pd.read_csv(f, delim_whitespace=True, header=None,
                               names=['year', 'month', 'day', 'hour', 'temp',
                                     'dewpoint', 'pressure', 'wind_dir',
                                     'wind_speed', 'sky', 'precip1', 'precip6'])
            
            # Convert -9999 to NaN
            df = df.replace(-9999, pd.NA)
            
            # Create datetime index
            df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
            df.set_index('date', inplace=True)
            
            # Convert units: tenths of Celsius to Celsius, tenths of kph to kph
            df['temp'] = df['temp'] / 10.0
            df['dewpoint'] = df['dewpoint'] / 10.0
            df['wind_speed'] = df['wind_speed'] / 10.0
            
            # Rename columns
            df = df.rename(columns={
                'temp': 'temp_avg',
                'dewpoint': 'dewpoint',
                'pressure': 'pressure',
                'wind_dir': 'wind_direction',
                'wind_speed': 'wind_speed',
                'precip1': 'precipitation_1h',
                'precip6': 'precipitation_6h'
            })
            
            # Resample to daily frequency
            daily_df = df.resample('D').agg({
                'temp_avg': 'mean',
                'dewpoint': 'mean',
                'pressure': 'mean',
                'wind_direction': lambda x: x.mean() if not x.isna().all() else pd.NA,
                'wind_speed': 'mean',
                'precipitation_1h': 'sum',
                'precipitation_6h': 'sum'
            })
            
            return daily_df
            
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {str(e)}")
            return None

def main():
    """Main function to download ISD-Lite data for European cities."""
    # Setup output directory
    output_dir = os.path.join("data", "raw", "isd")
    downloader = ISDLiteDownloader(
        output_dir=output_dir,
        start_date='2018-01-01',
        end_date='2023-12-31'
    )
    
    # Selected stations based on coverage and proximity to city centers
    stations = [
        {"id": "037720-99999", "name": "London - Heathrow"},
        {"id": "062400-99999", "name": "Amsterdam - Schiphol"},  # Fixed
        {"id": "071500-99999", "name": "Paris - Le Bourget"}
    ]
    
    # Download data for each station
    for station in stations:
        files = downloader.download_station_data(station["id"])
        if files:
            print(f"Successfully downloaded data for {station['name']}")
            for file in files:
                print(f"  - {file}")
        else:
            print(f"Failed to download data for {station['name']}")

if __name__ == "__main__":
    main()