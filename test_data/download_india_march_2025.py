from pathlib import Path
import pandas as pd
import logging
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the necessary modules from the project
from src.data.simple_openmeteo import SimpleOpenMeteoDownloader
from src.data.isd_lite_downloader import ISDLiteDownloader
from src.data.data_alignment import DataAligner

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_india_stations(history_file):
    """
    Find ISD stations in India from the history file.
    
    Args:
        history_file: Path to the ISD history CSV file
    """
    try:
        # Read the CSV file
        df = pd.read_csv(history_file, low_memory=False)
        
        # Filter for Indian stations (CTRY code for India is 'IN')
        india_stations = df[df['CTRY'] == 'IN'].copy()
        
        # Add a column for the station ID in the format used by the downloader
        india_stations['STATION_ID'] = india_stations['USAF'] + '-' + india_stations['WBAN'].astype(str).str.zfill(5)
        
        # Sort by data coverage (BEGIN and END dates)
        india_stations['BEGIN'] = pd.to_datetime(india_stations['BEGIN'], format='%Y%m%d')
        india_stations['END'] = pd.to_datetime(india_stations['END'], format='%Y%m%d')
        
        # Calculate coverage period in years
        india_stations['COVERAGE_YEARS'] = (india_stations['END'] - india_stations['BEGIN']).dt.days / 365.25
        
        # Sort by coverage years (descending)
        india_stations = india_stations.sort_values('COVERAGE_YEARS', ascending=False)
        
        # Select relevant columns
        result = india_stations[['STATION_ID', 'STATION NAME', 'LAT', 'LON', 'ELEV(M)', 'BEGIN', 'END', 'COVERAGE_YEARS']]
        
        # Display the top stations
        logger.info(f"Found {len(result)} stations in India")
        logger.info("\nTop 10 stations by data coverage:")
        for i, row in result.head(10).iterrows():
            logger.info(f"ID: {row['STATION_ID']}, Name: {row['STATION NAME']}, "
                       f"Location: ({row['LAT']}, {row['LON']}), "
                       f"Coverage: {row['BEGIN'].year}-{row['END'].year} ({row['COVERAGE_YEARS']:.1f} years)")
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing station data: {str(e)}")
        return None

def download_india_march_2025(city_name, lat, lon, isd_station):
    """
    Download and align March 2025 data for an Indian city.
    
    Args:
        city_name: Name of the city
        lat: Latitude of the city
        lon: Longitude of the city
        isd_station: ISD station ID for the city
    """
    logger.info(f"Downloading March 2025 data for {city_name}, India...")
    
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
            location_name=f"{city_name}_India_March2025"
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
        combined_isd_file = dirs['isd'] / f"{isd_station}_India_March2025_combined.csv"
        combined_isd.to_csv(combined_isd_file)
        
        # Align data
        aligner = DataAligner(
            openmeteo_dir=str(dirs['openmeteo']),
            isd_dir=str(dirs['isd']),
            output_dir=str(dirs['processed'])
        )
        
        output_file = f"{city_name}_India_March2025_aligned.csv"
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
    # Path to the ISD history file
    history_file = Path("test_data/isd-history.csv")
    
    # Find Indian stations
    india_stations = find_india_stations(history_file)
    
    if india_stations is not None:
        # Save the full list to a CSV file
        output_file = Path("test_data/india_isd_stations.csv")
        india_stations.to_csv(output_file, index=False)
        logger.info(f"\nFull station list saved to {output_file}")
        
        # Use the first station in the list (best coverage)
        if not india_stations.empty:
            top_station = india_stations.iloc[0]
            station_id = top_station['STATION_ID']
            station_name = top_station['STATION NAME']
            lat = float(top_station['LAT'])
            lon = float(top_station['LON'])
            
            logger.info(f"\nUsing top station: {station_name} ({station_id}) at ({lat}, {lon})")
            
            # Download data for this station
            aligned_file = download_india_march_2025(
                city_name=station_name.replace(" ", "_"),
                lat=lat,
                lon=lon,
                isd_station=station_id
            )
            
            if aligned_file:
                print(f"\nTo test the model with this data, run:")
                print(f"streamlit run src/app/app.py")
                print(f"Then upload the file: {aligned_file}")
                print(f"And set the model path to: logs\\pc_training_corrected_v5\\checkpoints\\bias_correction-epoch=19-val_loss=0.00.ckpt")
