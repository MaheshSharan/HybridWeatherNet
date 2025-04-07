import os
import argparse
import logging
from download_gsod import GSODDownloader
from download_ncep import NCEPDownloader
from data_alignment import DataAligner

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the weather bias correction data pipeline')
    
    parser.add_argument('--start_year', type=int, default=2018, help='Start year for data collection')
    parser.add_argument('--end_year', type=int, default=2023, help='End year for data collection')
    parser.add_argument('--skip_download', action='store_true', help='Skip data download step')
    parser.add_argument('--skip_processing', action='store_true', help='Skip data processing step')
    parser.add_argument('--skip_alignment', action='store_true', help='Skip data alignment step')
    
    return parser.parse_args()

def main():
    """Run the data pipeline."""
    args = parse_args()
    
    # Set up directories
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "data")
    raw_dir = os.path.join(data_dir, "raw")
    processed_dir = os.path.join(data_dir, "processed")
    
    gsod_dir = os.path.join(raw_dir, "gsod")
    ncep_dir = os.path.join(raw_dir, "ncep")
    
    # Create directories if they don't exist
    os.makedirs(gsod_dir, exist_ok=True)
    os.makedirs(ncep_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Step 1: Download GSOD data
    if not args.skip_download:
        logger.info("Step 1: Downloading GSOD data...")
        gsod_downloader = GSODDownloader(gsod_dir, args.start_year, args.end_year)
        successful_downloads = gsod_downloader.download_all()
        logger.info(f"Successfully downloaded GSOD data for years: {successful_downloads}")
        
        # Extract GSOD data
        successful_extractions = gsod_downloader.extract_all()
        logger.info(f"Successfully extracted GSOD data for years: {successful_extractions}")
        
        # Process GSOD data
        processed_data = gsod_downloader.process_all()
        logger.info(f"Successfully processed GSOD data for years: {list(processed_data.keys())}")
    
    # Step 2: Download NCEP data
    if not args.skip_download:
        logger.info("Step 2: Downloading NCEP data...")
        ncep_downloader = NCEPDownloader(ncep_dir, args.start_year, args.end_year)
        download_results = ncep_downloader.download_all()
        logger.info(f"Download results: {download_results}")
        
        # Process NCEP data
        processed_data = ncep_downloader.process_all()
        logger.info(f"Processed data: {list(processed_data.keys())}")
        
        # Combine variables
        combined_data = ncep_downloader.combine_all()
        logger.info(f"Combined data for years: {list(combined_data.keys())}")
    
    # Step 3: Align GSOD and NCEP data
    if not args.skip_alignment:
        logger.info("Step 3: Aligning GSOD and NCEP data...")
        aligner = DataAligner(gsod_dir, ncep_dir, processed_dir, args.start_year, args.end_year)
        
        # Align data
        aligned_data = aligner.align_all_data()
        
        if aligned_data is not None:
            # Analyze bias
            bias_analysis = aligner.analyze_bias(aligned_data)
            logger.info(f"Overall mean bias: {bias_analysis['overall']['mean_bias']:.2f}°C")
    
    logger.info("Data pipeline completed successfully!")

if __name__ == "__main__":
    main() 