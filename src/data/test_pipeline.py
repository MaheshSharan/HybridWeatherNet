import os
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime

from src.data.simple_openmeteo import SimpleOpenMeteoDownloader
from src.data.isd_lite_downloader import ISDLiteDownloader
from src.data.data_alignment import DataAligner

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_pipeline():
    """Test the complete data pipeline with ISD-Lite and OpenMeteo."""
    logger.info("Starting pipeline test...")
    
    # Setup directories
    base_dir = Path("data")
    openmeteo_dir = base_dir / "raw" / "openmeteo_test"
    isd_dir = base_dir / "raw" / "isd_test"
    processed_dir = base_dir / "processed_test"
    
    for dir_path in [openmeteo_dir, isd_dir, processed_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Test parameters
    start_date = "2018-01-01"
    end_date = "2018-12-31"  # Just test one year for speed
    
    # Test location (London Heathrow)
    location = {
        "name": "London",
        "lat": 51.47,
        "lon": -0.45,
        "isd_station": "037720-99999"  # Heathrow
    }
    
    # Step 1: Download OpenMeteo data
    logger.info(f"Downloading OpenMeteo data for {location['name']}...")
    openmeteo_downloader = SimpleOpenMeteoDownloader(
        output_dir=str(openmeteo_dir),
        start_date=start_date,
        end_date=end_date
    )
    
    openmeteo_file = openmeteo_downloader.download_data(
        latitude=location["lat"],
        longitude=location["lon"],
        location_name=location["name"]
    )
    
    if not openmeteo_file:
        logger.error("Failed to download OpenMeteo data")
        return False
    
    logger.info(f"OpenMeteo data downloaded to {openmeteo_file}")
    
    # Step 2: Download ISD-Lite data
    logger.info(f"Downloading ISD-Lite data for {location['name']}...")
    isd_downloader = ISDLiteDownloader(
        output_dir=str(isd_dir),
        start_date=start_date,
        end_date=end_date
    )
    
    isd_files = isd_downloader.download_station_data(
        station_id=location["isd_station"]
    )
    
    if not isd_files:
        logger.error("Failed to download ISD-Lite data")
        return False
    
    logger.info(f"ISD-Lite data downloaded to {isd_files}")
    
    # Step 3: Process ISD-Lite data
    logger.info("Processing ISD-Lite data...")
    isd_dfs = []
    for isd_file in isd_files:
        df = isd_downloader.process_station_data(isd_file)
        if df is not None:
            isd_dfs.append(df)
    
    if not isd_dfs:
        logger.error("Failed to process ISD-Lite data")
        return False
    
    # Combine all years if needed
    if len(isd_dfs) > 1:
        isd_df = pd.concat(isd_dfs)
    else:
        isd_df = isd_dfs[0]
    
    # Save processed ISD data
    isd_processed_file = isd_dir / f"{location['name']}_processed.csv"
    isd_df.to_csv(isd_processed_file)
    logger.info(f"Processed ISD-Lite data saved to {isd_processed_file}")
    
    # Step 4: Align data
    logger.info("Aligning data...")
    aligner = DataAligner(
        openmeteo_dir=str(openmeteo_dir),
        isd_dir=str(isd_dir),
        output_dir=str(processed_dir)
    )
    
    output_file = f"{location['name']}_{start_date}_{end_date}_aligned.csv"
    aligner.align_data(
        openmeteo_file=openmeteo_file,
        isd_file=str(isd_processed_file),
        output_file=output_file
    )
    
    # Step 5: Verify aligned data
    aligned_file = processed_dir / output_file
    if not aligned_file.exists():
        logger.error("Failed to create aligned data file")
        return False
    
    aligned_df = pd.read_csv(aligned_file)
    logger.info(f"Aligned data saved to {aligned_file}")
    logger.info(f"Aligned data shape: {aligned_df.shape}")
    
    # Check for key columns
    required_columns = ['date', 'temperature_model', 'temperature_obs']
    missing_columns = [col for col in required_columns if col not in aligned_df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    
    # Check data quality
    logger.info("\nData Quality Check:")
    logger.info(f"Number of records: {len(aligned_df)}")
    logger.info(f"Date range: {aligned_df['date'].min()} to {aligned_df['date'].max()}")
    
    # Check for missing values
    missing_values = aligned_df[['temperature_model', 'temperature_obs']].isna().sum()
    logger.info(f"Missing values in temperature_model: {missing_values['temperature_model']}")
    logger.info(f"Missing values in temperature_obs: {missing_values['temperature_obs']}")
    
    # Calculate basic statistics
    logger.info("\nTemperature Statistics:")
    logger.info(f"Model temperature - Mean: {aligned_df['temperature_model'].mean():.2f}°C")
    logger.info(f"Observed temperature - Mean: {aligned_df['temperature_obs'].mean():.2f}°C")
    
    # Calculate bias
    bias = aligned_df['temperature_model'] - aligned_df['temperature_obs']
    logger.info(f"Mean temperature bias: {bias.mean():.2f}°C")
    logger.info(f"Temperature bias std: {bias.std():.2f}°C")
    
    # Calculate correlation
    correlation = aligned_df['temperature_model'].corr(aligned_df['temperature_obs'])
    logger.info(f"Temperature correlation: {correlation:.2f}")
    
    logger.info("Pipeline test completed successfully!")
    return True

if __name__ == "__main__":
    test_pipeline() 