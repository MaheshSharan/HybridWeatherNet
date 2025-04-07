import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import zipfile
import io
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GSODDownloader:
    """
    Class to download and process NOAA GSOD (Global Surface Summary of the Day) data.
    """
    
    def __init__(self, output_dir, start_year=2018, end_year=2023):
        """
        Initialize the GSOD downloader.
        
        Args:
            output_dir (str): Directory to save the downloaded data
            start_year (int): Start year for data download
            end_year (int): End year for data download
        """
        self.base_url = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/archive"
        self.output_dir = output_dir
        self.start_year = start_year
        self.end_year = end_year
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def download_year(self, year):
        """
        Download GSOD data for a specific year.
        
        Args:
            year (int): Year to download data for
            
        Returns:
            bool: True if download was successful, False otherwise
        """
        url = f"{self.base_url}/{year}.tar.gz"
        output_file = os.path.join(self.output_dir, f"{year}.tar.gz")
        
        try:
            logger.info(f"Downloading GSOD data for {year}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Get file size for progress bar
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_file, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {year}") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            logger.info(f"Successfully downloaded GSOD data for {year}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading GSOD data for {year}: {str(e)}")
            return False
    
    def download_all(self):
        """
        Download GSOD data for all years in the specified range.
        
        Returns:
            list: List of years that were successfully downloaded
        """
        successful_years = []
        
        for year in range(self.start_year, self.end_year + 1):
            if self.download_year(year):
                successful_years.append(year)
        
        return successful_years
    
    def extract_data(self, year):
        """
        Extract the downloaded tar.gz file for a specific year.
        
        Args:
            year (int): Year to extract data for
            
        Returns:
            bool: True if extraction was successful, False otherwise
        """
        tar_file = os.path.join(self.output_dir, f"{year}.tar.gz")
        extract_dir = os.path.join(self.output_dir, str(year))
        
        if not os.path.exists(tar_file):
            logger.error(f"Tar file for {year} not found")
            return False
        
        try:
            logger.info(f"Extracting GSOD data for {year}...")
            import tarfile
            
            with tarfile.open(tar_file, 'r:gz') as tar:
                tar.extractall(path=extract_dir)
            
            logger.info(f"Successfully extracted GSOD data for {year}")
            return True
            
        except Exception as e:
            logger.error(f"Error extracting GSOD data for {year}: {str(e)}")
            return False
    
    def extract_all(self):
        """
        Extract all downloaded tar.gz files.
        
        Returns:
            list: List of years that were successfully extracted
        """
        successful_years = []
        
        for year in range(self.start_year, self.end_year + 1):
            if self.extract_data(year):
                successful_years.append(year)
        
        return successful_years
    
    def process_data(self, year):
        """
        Process the extracted GSOD data for a specific year.
        
        Args:
            year (int): Year to process data for
            
        Returns:
            pd.DataFrame: Processed data for the year
        """
        year_dir = os.path.join(self.output_dir, str(year))
        
        if not os.path.exists(year_dir):
            logger.error(f"Directory for {year} not found")
            return None
        
        try:
            logger.info(f"Processing GSOD data for {year}...")
            
            # Find all CSV files in the year directory
            csv_files = [f for f in os.listdir(year_dir) if f.endswith('.csv')]
            
            if not csv_files:
                logger.warning(f"No CSV files found for {year}")
                return None
            
            # Read and concatenate all CSV files
            dfs = []
            for csv_file in csv_files:
                file_path = os.path.join(year_dir, csv_file)
                try:
                    df = pd.read_csv(file_path, low_memory=False)
                    dfs.append(df)
                except Exception as e:
                    logger.warning(f"Error reading {csv_file}: {str(e)}")
            
            if not dfs:
                logger.warning(f"No valid data found for {year}")
                return None
            
            # Concatenate all dataframes
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # Save processed data
            processed_file = os.path.join(self.output_dir, f"processed_{year}.csv")
            combined_df.to_csv(processed_file, index=False)
            
            logger.info(f"Successfully processed GSOD data for {year}")
            return combined_df
            
        except Exception as e:
            logger.error(f"Error processing GSOD data for {year}: {str(e)}")
            return None
    
    def process_all(self):
        """
        Process all extracted GSOD data.
        
        Returns:
            dict: Dictionary mapping years to processed dataframes
        """
        processed_data = {}
        
        for year in range(self.start_year, self.end_year + 1):
            df = self.process_data(year)
            if df is not None:
                processed_data[year] = df
        
        return processed_data

if __name__ == "__main__":
    # Example usage
    output_dir = os.path.join("data", "raw", "gsod")
    downloader = GSODDownloader(output_dir, start_year=2018, end_year=2023)
    
    # Download data
    successful_downloads = downloader.download_all()
    print(f"Successfully downloaded data for years: {successful_downloads}")
    
    # Extract data
    successful_extractions = downloader.extract_all()
    print(f"Successfully extracted data for years: {successful_extractions}")
    
    # Process data
    processed_data = downloader.process_all()
    print(f"Successfully processed data for years: {list(processed_data.keys())}") 