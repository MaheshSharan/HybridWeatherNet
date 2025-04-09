import requests
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_india_stations():
    """
    Fetch and display ISD stations in India.
    """
    try:
        # URL for the ISD station list
        url = r"C:\Users\SeoYea-Ji\weather_bias_correction\test_data\isd-history.csv"
        
        # Download the station list
        logger.info("Downloading ISD station list...")
        response = requests.get(url)
        response.raise_for_status()
        
        # Save to a temporary file
        temp_file = Path("data/isd-history.csv")
        temp_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(temp_file, "wb") as f:
            f.write(response.content)
        
        # Read the CSV file
        df = pd.read_csv(temp_file, low_memory=False)
        
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
        logger.error(f"Error fetching station data: {str(e)}")
        return None

if __name__ == "__main__":
    india_stations = get_india_stations()
    if india_stations is not None:
        # Save the full list to a CSV file
        output_file = Path("data/india_isd_stations.csv")
        india_stations.to_csv(output_file, index=False)
        logger.info(f"\nFull station list saved to {output_file}")
