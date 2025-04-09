from src.data.simple_openmeteo import SimpleOpenMeteoDownloader
from src.data.isd_lite_downloader import ISDLiteDownloader
from src.data.data_alignment import DataAligner
import logging
import pandas as pd
from pathlib import Path
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_march_2025_data(city_name="Amsterdam", lat=52.31, lon=4.76, isd_station="062400-99999"):
    """
    Download and align March 2025 data for testing the model on unseen data.
    
    Args:
        city_name: Name of the city to download data for
        lat: Latitude of the city
        lon: Longitude of the city
        isd_station: ISD station ID for the city
    """
    logger.info(f"Downloading March 2025 data for {city_name}...")
    
    # Setup directories
    base_dir = Path("data")
    dirs = {
        'openmeteo': base_dir / 'raw' / 'openmeteo',
        'isd': base_dir / 'raw' / 'isd',
        'processed': base_dir / 'processed'
    }
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Date range for March 2025
    start_date = '2025-03-01'
    end_date = '2025-03-31'
    
    try:
        # Download OpenMeteo data
        openmeteo_downloader = SimpleOpenMeteoDownloader(
            output_dir=str(dirs['openmeteo']),
            start_date=start_date,
            end_date=end_date
        )
        
        openmeteo_file = openmeteo_downloader.download_data(
            latitude=lat,
            longitude=lon,
            location_name=f"{city_name}_March2025"
        )
        
        # Download ISD data
        isd_downloader = ISDLiteDownloader(
            output_dir=str(dirs['isd']),
            start_date=start_date,
            end_date=end_date
        )
        
        isd_files = isd_downloader.download_station_data(isd_station)
        
        if not isd_files:
            logger.error(f"Failed to download ISD data for {isd_station}")
            return None
        
        # Process and combine ISD files
        combined_isd = None
        for isd_file in isd_files:
            df = isd_downloader.process_station_data(isd_file)
            if df is not None:
                if combined_isd is None:
                    combined_isd = df
                else:
                    combined_isd = pd.concat([combined_isd, df])
        
        if combined_isd is None or combined_isd.empty:
            logger.error("No valid ISD data processed")
            return None
        
        # Save combined ISD data
        combined_isd_file = dirs['isd'] / f"{isd_station}_March2025_combined.csv"
        combined_isd.to_csv(combined_isd_file)
        
        # Align data
        aligner = DataAligner(
            openmeteo_dir=str(dirs['openmeteo']),
            isd_dir=str(dirs['isd']),
            output_dir=str(dirs['processed'])
        )
        
        output_file = f"{city_name}_March2025_aligned.csv"
        aligner.align_data(
            openmeteo_file=openmeteo_file,
            isd_file=str(combined_isd_file),
            output_file=output_file
        )
        
        logger.info(f"Successfully created aligned test data: {output_file}")
        return str(dirs['processed'] / output_file)
    
    except Exception as e:
        logger.error(f"Error downloading March 2025 data: {str(e)}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download March 2025 weather data for model testing")
    parser.add_argument("--city", type=str, default="Amsterdam", help="City name")
    parser.add_argument("--lat", type=float, default=52.31, help="Latitude")
    parser.add_argument("--lon", type=float, default=4.76, help="Longitude")
    parser.add_argument("--station", type=str, default="062400-99999", help="ISD station ID")
    
    args = parser.parse_args()
    
    aligned_file = download_march_2025_data(
        city_name=args.city,
        lat=args.lat,
        lon=args.lon,
        isd_station=args.station
    )
    
    if aligned_file:
        print(f"\nTo test the model with this data, run:")
        print(f"streamlit run src/app/app.py")
        print(f"Then upload the file: {aligned_file}")
        print(f"And set the model path to: logs\\pc_training_corrected_v5\\checkpoints\\bias_correction-epoch=19-val_loss=0.00.ckpt")
