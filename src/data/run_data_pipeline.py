import os
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

from .download_openmeteo import OpenMeteoDownloader
from .download_gsod import GSODDownloader
from .data_alignment import DataAligner

class DataPipeline:
    """Class to run the complete data processing pipeline."""
    
    def __init__(self, config: Dict):
        """Initialize the data pipeline.
        
        Args:
            config: Dictionary containing configuration parameters
                - base_dir: Base directory for all data
                - start_date: Start date in YYYY-MM-DD format
                - end_date: End date in YYYY-MM-DD format
                - locations: List of dictionaries containing location information
                    - name: Location name
                    - lat: Latitude
                    - lon: Longitude
                    - gsod_station: GSOD station ID
        """
        self.config = config
        self.base_dir = Path(config['base_dir'])
        
        # Create directory structure
        self.dirs = {
            'openmeteo': self.base_dir / 'openmeteo',
            'gsod': self.base_dir / 'gsod',
            'aligned': self.base_dir / 'aligned'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def run(self):
        """Run the complete data pipeline."""
        self.logger.info("Starting data pipeline...")
        
        try:
            # Initialize downloaders
            openmeteo_downloader = OpenMeteoDownloader(
                output_dir=str(self.dirs['openmeteo']),
                start_date=self.config['start_date'],
                end_date=self.config['end_date']
            )
            
            gsod_downloader = GSODDownloader(
                output_dir=str(self.dirs['gsod']),
                start_date=self.config['start_date'],
                end_date=self.config['end_date']
            )
            
            # Initialize data aligner
            aligner = DataAligner(
                openmeteo_dir=str(self.dirs['openmeteo']),
                gsod_dir=str(self.dirs['gsod']),
                output_dir=str(self.dirs['aligned'])
            )
            
            # Process each location
            for location in self.config['locations']:
                self.logger.info(f"\nProcessing location: {location['name']}")
                
                try:
                    # Download Open-Meteo data
                    openmeteo_file = openmeteo_downloader.download_data(
                        latitude=location['lat'],
                        longitude=location['lon'],
                        location_name=location['name']
                    )
                    
                    # Download GSOD data
                    gsod_file = gsod_downloader.download_data(
                        station_id=location['gsod_station']
                    )
                    
                    # Align data
                    output_file = f"{location['name']}_{self.config['start_date']}_{self.config['end_date']}_aligned.csv"
                    aligner.align_data(
                        openmeteo_file=openmeteo_file,
                        gsod_file=gsod_file,
                        output_file=output_file
                    )
                    
                except Exception as e:
                    self.logger.error(f"Error processing location {location['name']}: {str(e)}")
                    continue
            
            self.logger.info("Data pipeline completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Error in data pipeline: {str(e)}")
            raise

def main():
    """Main function to run the data pipeline."""
    # Example configuration
    config = {
        'base_dir': 'weather_data',
        'start_date': '2023-01-01',
        'end_date': '2023-12-31',
        'locations': [
            {
                'name': 'Berlin',
                'lat': 52.52,
                'lon': 13.41,
                'gsod_station': '10384'
            },
            {
                'name': 'New_York',
                'lat': 40.71,
                'lon': -74.01,
                'gsod_station': '72506'
            },
            {
                'name': 'Tokyo',
                'lat': 35.68,
                'lon': 139.77,
                'gsod_station': '47662'
            }
        ]
    }
    
    pipeline = DataPipeline(config)
    pipeline.run()

if __name__ == "__main__":
    main() 