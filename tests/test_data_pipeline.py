import os
import sys
import logging
from src.data import GSODDownloader, NCEPDownloader, DataAligner

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_gsod_download():
    """Test GSOD data download with a small sample."""
    logger.info("Testing GSOD data download...")
    
    # Set up directories
    output_dir = os.path.join("data", "raw", "gsod_test")
    os.makedirs(output_dir, exist_ok=True)
    
    # Download data for a single year
    downloader = GSODDownloader(output_dir, start_year=2022, end_year=2022)
    
    # Download data
    successful_downloads = downloader.download_all()
    logger.info(f"Successfully downloaded GSOD data for years: {successful_downloads}")
    
    # Extract data
    successful_extractions = downloader.extract_all()
    logger.info(f"Successfully extracted GSOD data for years: {successful_extractions}")
    
    # Process data
    processed_data = downloader.process_all()
    logger.info(f"Successfully processed GSOD data for years: {list(processed_data.keys())}")
    
    return processed_data

def test_ncep_download():
    """Test NCEP data download with a small sample."""
    logger.info("Testing NCEP data download...")
    
    # Set up directories
    output_dir = os.path.join("data", "raw", "ncep_test")
    os.makedirs(output_dir, exist_ok=True)
    
    # Download data for a single year
    downloader = NCEPDownloader(output_dir, start_year=2022, end_year=2022)
    
    # Download data
    download_results = downloader.download_all()
    logger.info(f"Download results: {download_results}")
    
    # Process data
    processed_data = downloader.process_all()
    logger.info(f"Processed data: {list(processed_data.keys())}")
    
    # Combine variables
    combined_data = downloader.combine_all()
    logger.info(f"Combined data for years: {list(combined_data.keys())}")
    
    return combined_data

def test_data_alignment():
    """Test data alignment with a small sample."""
    logger.info("Testing data alignment...")
    
    # Set up directories
    gsod_dir = os.path.join("data", "raw", "gsod_test")
    ncep_dir = os.path.join("data", "raw", "ncep_test")
    output_dir = os.path.join("data", "processed", "test")
    os.makedirs(output_dir, exist_ok=True)
    
    # Align data
    aligner = DataAligner(gsod_dir, ncep_dir, output_dir, start_year=2022, end_year=2022)
    
    # Align data
    aligned_data = aligner.align_all_data()
    
    if aligned_data is not None:
        # Analyze bias
        bias_analysis = aligner.analyze_bias(aligned_data)
        logger.info(f"Overall mean bias: {bias_analysis['overall']['mean_bias']:.2f}°C")
    
    return aligned_data

def main():
    """Run the test pipeline."""
    logger.info("Starting test pipeline...")
    
    # Test GSOD download
    gsod_data = test_gsod_download()
    
    # Test NCEP download
    ncep_data = test_ncep_download()
    
    # Test data alignment
    aligned_data = test_data_alignment()
    
    logger.info("Test pipeline completed successfully!")

if __name__ == "__main__":
    main() 