import os
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import gzip

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataAligner:
    """
    Class to align OpenMeteo and ISD-Lite data for bias correction.
    """
    
    def __init__(self, openmeteo_dir: str, isd_dir: str, output_dir: str):
        """
        Initialize the data aligner.
        
        Args:
            openmeteo_dir: Directory containing Open-Meteo data files
            isd_dir: Directory containing ISD-Lite data files
            output_dir: Directory to save aligned data
        """
        self.openmeteo_dir = Path(openmeteo_dir)
        self.isd_dir = Path(isd_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def load_openmeteo_data(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess Open-Meteo data.
        
        Args:
            file_path: Path to the Open-Meteo CSV file
            
        Returns:
            Preprocessed DataFrame
        """
        # Read the CSV file with the date column
        df = pd.read_csv(file_path)
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Set date as index
        df.set_index('date', inplace=True)
        
        # Rename columns to match our standard format
        df = df.rename(columns={
            'temp': 'temperature',
            'rh': 'humidity',
            'wind_speed': 'wind_speed',
            'wind_dir': 'wind_direction',
            'cloud_cover_low': 'cloud_cover_low',
            'cloud_cover_mid': 'cloud_cover_mid',
            'cloud_cover_high': 'cloud_cover_high'
        })
        
        # Resample to daily frequency (mean for most variables)
        daily_df = df.resample('D').agg({
            'temperature': 'mean',
            'humidity': 'mean',
            'wind_speed': 'mean',
            'wind_direction': lambda x: np.mean(np.radians(x)).round(4),  # Convert to radians for proper averaging
            'cloud_cover_low': 'mean',
            'cloud_cover_mid': 'mean',
            'cloud_cover_high': 'mean'
        })
        
        # Convert wind direction back to degrees
        daily_df['wind_direction'] = np.degrees(daily_df['wind_direction'])
        
        return daily_df
    
    def load_isd_data(self, file_path: str) -> pd.DataFrame:
        """Load preprocessed ISD-Lite data from CSV.
        
        Args:
            file_path: Path to the processed ISD-Lite CSV file
            
        Returns:
            Preprocessed DataFrame
        """
        try:
            # Read the processed CSV file from run_data_pipeline
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {str(e)}")
            return None
    
    def align_data(self, openmeteo_file: str, isd_file: str, output_file: str) -> None:
        """Align Open-Meteo and ISD-Lite data.
        
        Args:
            openmeteo_file: Path to Open-Meteo data file
            isd_file: Path to ISD-Lite data file
            output_file: Path to save aligned data
        """
        self.logger.info(f"Aligning data from {openmeteo_file} and {isd_file}")
        
        # Load data
        openmeteo_df = self.load_openmeteo_data(openmeteo_file)
        isd_df = self.load_isd_data(isd_file)
        
        if isd_df is None:
            self.logger.error("Failed to load ISD-Lite data")
            return
        
        # Merge the dataframes
        merged_df = pd.merge(
            openmeteo_df,
            isd_df,
            left_index=True,
            right_index=True,
            how='inner',
            suffixes=('_model', '_obs')
        )
        
        # Save aligned data
        merged_df.to_csv(self.output_dir / output_file)
        self.logger.info(f"Aligned data saved to {output_file}")
        self._log_statistics(merged_df)
    
    def _log_statistics(self, df: pd.DataFrame) -> None:
        """Log statistics about the aligned data.
        
        Args:
            df: DataFrame with aligned data
        """
        self.logger.info("\nData Statistics:")
        self.logger.info(f"Total records: {len(df)}")
        self.logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        
        for column in df.columns:
            if df[column].dtype in ['float64', 'int64']: 
                stats = df[column].describe()
                self.logger.info(f"\nStatistics for {column}:")
                self.logger.info(f"  Count: {stats['count']:.0f}")
                self.logger.info(f"  Mean: {stats['mean']:.2f}")
                self.logger.info(f"  Std: {stats['std']:.2f}")
                self.logger.info(f"  Min: {stats['min']:.2f}")
                self.logger.info(f"  Max: {stats['max']:.2f}")
            else:
                self.logger.info(f"\nStatistics for {column}: Non-numeric, skipped")

def main():
    """Main function to demonstrate usage."""
    # Example usage
    aligner = DataAligner(
        openmeteo_dir="data/raw/openmeteo",
        isd_dir="data/raw/isd",
        output_dir="data/processed"
    )
    
    # Example alignment
    aligner.align_data(
        openmeteo_file="data/raw/openmeteo/Berlin_2018-01-01_2018-12-31.csv",
        isd_file="data/raw/isd/UKM00003772_combined.csv",
        output_file="berlin_2018_aligned.csv"
    )

if __name__ == "__main__":
    main()