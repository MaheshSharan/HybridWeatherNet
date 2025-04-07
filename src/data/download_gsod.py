import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import zipfile
import io
from tqdm import tqdm
import logging
from typing import List, Dict, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GSODDownloader:
    """
    Class to download and process NOAA GSOD (Global Surface Summary of the Day) data.
    """
    
    def __init__(self, output_dir: str, start_date: str, end_date: str):
        """
        Initialize the GSOD downloader.
        
        Args:
            output_dir (str): Directory to save the downloaded data
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
        """
        self.base_url = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/archive"
        self.output_dir = output_dir
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def download_station_data(self, station_id: str) -> Optional[str]:
        """
        Download GSOD data for a specific station.
        
        Args:
            station_id (str): GSOD station ID
            
        Returns:
            str: Path to the processed station data file
        """
        try:
            logger.info(f"Downloading GSOD data for station {station_id}...")
            
            # Calculate years to download
            start_year = self.start_date.year
            end_year = self.end_date.year
            
            # Download and process data for each year
            all_data = []
            for year in range(start_year, end_year + 1):
                # Download year data
                url = f"{self.base_url}/{year}.tar.gz"
                output_file = os.path.join(self.output_dir, f"{year}.tar.gz")
                
                try:
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    
                    with open(output_file, 'wb') as f:
                        with tqdm(total=int(response.headers.get('content-length', 0)), 
                                unit='B', unit_scale=True, 
                                desc=f"Downloading {year}") as pbar:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    pbar.update(len(chunk))
                    
                    # Extract the tar.gz file
                    import tarfile
                    with tarfile.open(output_file, 'r:gz') as tar:
                        tar.extractall(path=os.path.join(self.output_dir, str(year)))
                    
                    # Find and read the station's CSV file
                    year_dir = os.path.join(self.output_dir, str(year))
                    station_file = f"{station_id}.csv"
                    station_path = os.path.join(year_dir, station_file)
                    
                    if os.path.exists(station_path):
                        df = pd.read_csv(station_path)
                        all_data.append(df)
                    
                    # Clean up
                    os.remove(output_file)
                    
                except Exception as e:
                    logger.warning(f"Error processing year {year}: {str(e)}")
                    continue
            
            if not all_data:
                logger.error(f"No data found for station {station_id}")
                return None
            
            # Combine all years' data
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Convert date column
            combined_df['DATE'] = pd.to_datetime(combined_df['DATE'])
            
            # Filter by date range
            mask = (combined_df['DATE'] >= self.start_date) & (combined_df['DATE'] <= self.end_date)
            combined_df = combined_df[mask]
            
            # Select and rename columns
            columns = {
                'DATE': 'date',
                'STATION': 'station',
                'LATITUDE': 'latitude',
                'LONGITUDE': 'longitude',
                'TEMP': 'temp',
                'RH': 'rh',
                'WIND_SPEED': 'wind_speed',
                'WIND_DIR': 'wind_dir'
            }
            combined_df = combined_df[columns.keys()].rename(columns=columns)
            
            # Save processed data
            output_file = os.path.join(self.output_dir, f"{station_id}.csv")
            combined_df.to_csv(output_file, index=False)
            
            logger.info(f"Successfully processed data for station {station_id}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error downloading data for station {station_id}: {str(e)}")
            return None

def main():
    """Main function to demonstrate usage."""
    # Example usage
    output_dir = os.path.join("data", "raw", "gsod")
    downloader = GSODDownloader(
        output_dir=output_dir,
        start_date='2018-01-01',
        end_date='2023-12-31'
    )
    
    # Example stations
    stations = [
        {"id": "10384", "name": "Berlin"},
        {"id": "72506", "name": "New York"},
        {"id": "47662", "name": "Tokyo"}
    ]
    
    # Download data for each station
    for station in stations:
        output_file = downloader.download_station_data(station["id"])
        if output_file:
            print(f"Successfully downloaded data for {station['name']} to {output_file}")
        else:
            print(f"Failed to download data for {station['name']}")

if __name__ == "__main__":
    main() 