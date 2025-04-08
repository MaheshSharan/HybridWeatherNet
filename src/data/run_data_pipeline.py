from src.data.simple_openmeteo import SimpleOpenMeteoDownloader
from src.data.isd_lite_downloader import ISDLiteDownloader
from src.data.data_alignment import DataAligner
from typing import Dict
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPipeline:
    def __init__(self, config: Dict):
        self.config = config
        self.base_dir = Path(config['base_dir'])
        self.dirs = {
            'openmeteo': self.base_dir / 'raw' / 'openmeteo',
            'isd': self.base_dir / 'raw' / 'isd',
            'processed': self.base_dir / 'processed'
        }
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def run(self):
        logger.info("Starting data pipeline...")
        try:
            openmeteo_downloader = SimpleOpenMeteoDownloader(
                output_dir=str(self.dirs['openmeteo']),
                start_date=self.config['start_date'],
                end_date=self.config['end_date']
            )
            isd_downloader = ISDLiteDownloader(
                output_dir=str(self.dirs['isd']),
                start_date=self.config['start_date'],
                end_date=self.config['end_date']
            )
            aligner = DataAligner(
                openmeteo_dir=str(self.dirs['openmeteo']),
                isd_dir=str(self.dirs['isd']),
                output_dir=str(self.dirs['processed'])
            )
            
            for location in self.config['locations']:
                logger.info(f"\nProcessing location: {location['name']}")
                openmeteo_file = openmeteo_downloader.download_data(
                    latitude=location['lat'],
                    longitude=location['lon'],
                    location_name=location['name']
                )
                isd_files = isd_downloader.download_station_data(location['isd_station'])
                if isd_files:
                    # Combine all ISD years
                    combined_isd = None
                    for isd_file in isd_files:
                        df = isd_downloader.process_station_data(isd_file)
                        if combined_isd is None:
                            combined_isd = df
                        else:
                            combined_isd = pd.concat([combined_isd, df])
                    combined_isd_file = self.dirs['isd'] / f"{location['isd_station']}_combined.csv"
                    combined_isd.to_csv(combined_isd_file)
                    
                    # Align combined data
                    aligner.align_data(
                        openmeteo_file=openmeteo_file,
                        isd_file=str(combined_isd_file),
                        output_file=f"{location['name']}_{self.config['start_date']}_{self.config['end_date']}_aligned.csv"
                    )
            logger.info("Data pipeline completed!")
        except Exception as e:
            logger.error(f"Error in pipeline: {str(e)}")
            raise

def main():
    config = {
        'base_dir': 'data',
        'start_date': '2018-01-01',
        'end_date': '2023-12-31',
        'locations': [
            {'name': 'London', 'lat': 51.48, 'lon': -0.45, 'isd_station': '037720-99999'},
            {'name': 'Amsterdam', 'lat': 52.31, 'lon': 4.76, 'isd_station': '062400-99999'},
            {'name': 'Paris', 'lat': 48.97, 'lon': 2.40, 'isd_station': '071500-99999'}
        ]
    }
    pipeline = DataPipeline(config)
    pipeline.run()

if __name__ == "__main__":
    main()