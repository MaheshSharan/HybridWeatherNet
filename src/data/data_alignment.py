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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataAligner:
    """
    Class to align GSOD and NCEP data for bias correction.
    """
    
    def __init__(self, openmeteo_dir: str, gsod_dir: str, output_dir: str):
        """
        Initialize the data aligner.
        
        Args:
            openmeteo_dir: Directory containing Open-Meteo data files
            gsod_dir: Directory containing GSOD data files
            output_dir: Directory to save aligned data
        """
        self.openmeteo_dir = Path(openmeteo_dir)
        self.gsod_dir = Path(gsod_dir)
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
        df = pd.read_csv(file_path, parse_dates=['time'], index_col='time')
        
        # Rename columns to match our standard format
        df = df.rename(columns={
            'temperature': 'temp',
            'relative_humidity': 'rh',
            'wind_speed': 'wind_speed',
            'wind_direction': 'wind_dir',
            'cloud_cover': 'cloud_cover_total',
            'cloud_cover_low': 'cloud_cover_low',
            'cloud_cover_mid': 'cloud_cover_mid',
            'cloud_cover_high': 'cloud_cover_high'
        })
        
        return df
    
    def load_gsod_data(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess GSOD data.
        
        Args:
            file_path: Path to the GSOD CSV file
            
        Returns:
            Preprocessed DataFrame
        """
        df = pd.read_csv(file_path, parse_dates=['DATE'])
        df.set_index('DATE', inplace=True)
        
        # Convert units and rename columns
        df['temp'] = (df['TEMP'] - 32) * 5/9 + 273.15  # Convert °F to K
        df['rh'] = df['RH']  # Relative humidity is already in %
        df['wind_speed'] = df['WIND_SPEED'] * 0.514444  # Convert knots to m/s
        df['wind_dir'] = df['WIND_DIR']  # Wind direction is already in degrees
        
        # Select only the columns we need
        df = df[['temp', 'rh', 'wind_speed', 'wind_dir']]
        
        return df
    
    def align_data(self, openmeteo_file: str, gsod_file: str, output_file: str) -> None:
        """Align Open-Meteo and GSOD data.
        
        Args:
            openmeteo_file: Path to Open-Meteo data file
            gsod_file: Path to GSOD data file
            output_file: Path to save aligned data
        """
        self.logger.info(f"Aligning data from {openmeteo_file} and {gsod_file}")
        
        # Load data
        openmeteo_df = self.load_openmeteo_data(openmeteo_file)
        gsod_df = self.load_gsod_data(gsod_file)
        
        # Resample Open-Meteo data to daily frequency (mean for most variables)
        openmeteo_daily = openmeteo_df.resample('D').agg({
            'temp': 'mean',
            'rh': 'mean',
            'wind_speed': 'mean',
            'wind_dir': lambda x: np.mean(np.radians(x)).round(4),  # Convert to radians for proper averaging
            'cloud_cover_total': 'mean',
            'cloud_cover_low': 'mean',
            'cloud_cover_mid': 'mean',
            'cloud_cover_high': 'mean'
        })
        
        # Convert wind direction back to degrees
        openmeteo_daily['wind_dir'] = np.degrees(openmeteo_daily['wind_dir'])
        
        # Merge the dataframes
        merged_df = pd.merge(
            openmeteo_daily,
            gsod_df,
            left_index=True,
            right_index=True,
            suffixes=('_model', '_obs')
        )
        
        # Save aligned data
        merged_df.to_csv(self.output_dir / output_file)
        self.logger.info(f"Aligned data saved to {output_file}")
        self._log_statistics(merged_df)
    
    def _log_statistics(self, df: pd.DataFrame) -> None:
        """Log basic statistics for the aligned data."""
        for column in df.columns:
            stats = df[column].describe()
            self.logger.info(f"\nStatistics for {column}:")
            self.logger.info(f"  Min: {stats['min']}")
            self.logger.info(f"  Max: {stats['max']}")
            self.logger.info(f"  Mean: {stats['mean']}")
            self.logger.info(f"  Missing values: {df[column].isna().sum()}")

def main():
    """Main function to demonstrate usage."""
    aligner = DataAligner(
        openmeteo_dir="weather_data/openmeteo",
        gsod_dir="weather_data/gsod",
        output_dir="weather_data/aligned"
    )
    
    # Example alignment
    aligner.align_data(
        openmeteo_file="weather_data/openmeteo/Berlin_2023-01-01_2023-12-31.csv",
        gsod_file="weather_data/gsod/10384.csv",
        output_file="berlin_2023_aligned.csv"
    )

if __name__ == "__main__":
    main() 