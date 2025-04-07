import os
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataAligner:
    """
    Class to align GSOD and NCEP data for bias correction.
    """
    
    def __init__(self, gsod_dir, ncep_dir, output_dir, start_year=2018, end_year=2023):
        """
        Initialize the data aligner.
        
        Args:
            gsod_dir (str): Directory containing processed GSOD data
            ncep_dir (str): Directory containing processed NCEP data
            output_dir (str): Directory to save aligned data
            start_year (int): Start year for data alignment
            end_year (int): End year for data alignment
        """
        self.gsod_dir = gsod_dir
        self.ncep_dir = ncep_dir
        self.output_dir = output_dir
        self.start_year = start_year
        self.end_year = end_year
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def load_gsod_data(self, year):
        """
        Load processed GSOD data for a specific year.
        
        Args:
            year (int): Year to load data for
            
        Returns:
            pd.DataFrame: GSOD data for the year
        """
        processed_file = os.path.join(self.gsod_dir, f"processed_{year}.csv")
        
        if not os.path.exists(processed_file):
            logger.error(f"Processed GSOD file for {year} not found")
            return None
        
        try:
            logger.info(f"Loading GSOD data for {year}...")
            df = pd.read_csv(processed_file)
            
            # Define column mappings
            column_mappings = {
                'date': ['DATE', 'date'],
                'station': ['STATION', 'station', 'station_id'],
                'temp': ['TEMP', 'temp', 'temperature'],
                'latitude': ['LATITUDE', 'latitude', 'lat'],
                'longitude': ['LONGITUDE', 'longitude', 'lon']
            }
            
            # Find and rename columns
            for target_col, possible_cols in column_mappings.items():
                for col in possible_cols:
                    if col in df.columns:
                        df[target_col] = df[col]
                        break
            
            # Convert date column to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Select only necessary columns
            required_columns = ['station', 'latitude', 'longitude', 'date', 'temp']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"Missing required columns in GSOD data: {missing_columns}")
                return None
            
            df = df[required_columns].copy()
            
            # Drop rows with missing values
            df = df.dropna()
            
            logger.info(f"Successfully loaded GSOD data for {year}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading GSOD data for {year}: {str(e)}")
            return None
    
    def load_ncep_data(self, year):
        """
        Load processed NCEP data for a specific year.
        
        Args:
            year (int): Year to load data for
            
        Returns:
            xarray.Dataset: NCEP data for the year
        """
        combined_file = os.path.join(self.ncep_dir, f"combined_{year}.nc")
        
        if not os.path.exists(combined_file):
            logger.error(f"Combined NCEP file for {year} not found")
            return None
        
        try:
            logger.info(f"Loading NCEP data for {year}...")
            ds = xr.open_dataset(combined_file)
            
            # Check if required variables are present
            required_vars = ['t2m', 'sp', 'rh2m', 'u10', 'v10', 'tcc']
            missing_vars = [var for var in required_vars if var not in ds.data_vars]
            
            if missing_vars:
                logger.error(f"Missing required variables in NCEP data: {missing_vars}")
                return None
            
            logger.info(f"Successfully loaded NCEP data for {year}")
            return ds
            
        except Exception as e:
            logger.error(f"Error loading NCEP data for {year}: {str(e)}")
            return None
    
    def extract_ncep_at_station(self, ds, latitude, longitude):
        """
        Extract NCEP data at a specific station location.
        
        Args:
            ds (xarray.Dataset): NCEP dataset
            latitude (float): Station latitude
            longitude (float): Station longitude
            
        Returns:
            pd.DataFrame: Extracted NCEP data at the station
        """
        try:
            # Find the closest grid point
            lat_idx = np.abs(ds.lat.values - latitude).argmin()
            lon_idx = np.abs(ds.lon.values - longitude).argmin()
            
            # Extract data at the closest grid point
            station_data = ds.isel(lat=lat_idx, lon=lon_idx).to_dataframe()
            
            # Reset index to get time as a column
            station_data = station_data.reset_index()
            
            # Rename time column to date
            station_data = station_data.rename(columns={'time': 'date'})
            
            return station_data
            
        except Exception as e:
            logger.error(f"Error extracting NCEP data at station: {str(e)}")
            return None
    
    def align_station_data(self, gsod_df, ncep_ds, station_id):
        """
        Align GSOD and NCEP data for a specific station.
        
        Args:
            gsod_df (pd.DataFrame): GSOD data
            ncep_ds (xarray.Dataset): NCEP data
            station_id (str): Station ID
            
        Returns:
            pd.DataFrame: Aligned data for the station
        """
        try:
            # Filter GSOD data for the station
            station_gsod = gsod_df[gsod_df['station_id'] == station_id].copy()
            
            if station_gsod.empty:
                logger.warning(f"No GSOD data found for station {station_id}")
                return None
            
            # Get station coordinates
            latitude = station_gsod['latitude'].iloc[0]
            longitude = station_gsod['longitude'].iloc[0]
            
            # Extract NCEP data at the station
            station_ncep = self.extract_ncep_at_station(ncep_ds, latitude, longitude)
            
            if station_ncep is None:
                logger.warning(f"No NCEP data found for station {station_id}")
                return None
            
            # Merge GSOD and NCEP data
            merged_data = pd.merge(
                station_gsod[['date', 'temp']],
                station_ncep,
                on='date',
                how='inner'
            )
            
            # Rename columns for clarity
            merged_data = merged_data.rename(columns={'temp': 'observed_temp', 't2m': 'forecast_temp'})
            
            # Calculate bias
            merged_data['bias'] = merged_data['observed_temp'] - merged_data['forecast_temp']
            
            # Add station information
            merged_data['station_id'] = station_id
            merged_data['latitude'] = latitude
            merged_data['longitude'] = longitude
            
            return merged_data
            
        except Exception as e:
            logger.error(f"Error aligning data for station {station_id}: {str(e)}")
            return None
    
    def align_all_data(self):
        """
        Align all GSOD and NCEP data.
        
        Returns:
            pd.DataFrame: Aligned data for all stations
        """
        all_aligned_data = []
        
        for year in range(self.start_year, self.end_year + 1):
            logger.info(f"Aligning data for {year}...")
            
            # Load GSOD data
            gsod_df = self.load_gsod_data(year)
            if gsod_df is None:
                continue
            
            # Load NCEP data
            ncep_ds = self.load_ncep_data(year)
            if ncep_ds is None:
                continue
            
            # Get unique stations
            stations = gsod_df['station_id'].unique()
            
            # Align data for each station
            for station_id in tqdm(stations, desc=f"Aligning stations for {year}"):
                aligned_data = self.align_station_data(gsod_df, ncep_ds, station_id)
                if aligned_data is not None:
                    all_aligned_data.append(aligned_data)
            
            logger.info(f"Successfully aligned data for {year}")
        
        if not all_aligned_data:
            logger.error("No aligned data found")
            return None
        
        # Concatenate all aligned data
        combined_data = pd.concat(all_aligned_data, ignore_index=True)
        
        # Save aligned data
        output_file = os.path.join(self.output_dir, "aligned_data.csv")
        combined_data.to_csv(output_file, index=False)
        
        logger.info(f"Successfully saved aligned data to {output_file}")
        return combined_data
    
    def analyze_bias(self, aligned_data):
        """
        Analyze bias in the aligned data.
        
        Args:
            aligned_data (pd.DataFrame): Aligned data
            
        Returns:
            dict: Bias analysis results
        """
        try:
            logger.info("Analyzing bias in aligned data...")
            
            # Calculate overall bias statistics
            overall_stats = {
                'mean_bias': aligned_data['bias'].mean(),
                'std_bias': aligned_data['bias'].std(),
                'min_bias': aligned_data['bias'].min(),
                'max_bias': aligned_data['bias'].max(),
                'median_bias': aligned_data['bias'].median()
            }
            
            # Calculate bias by station
            station_stats = aligned_data.groupby('station_id')['bias'].agg([
                'mean', 'std', 'count'
            ]).reset_index()
            
            # Calculate bias by month
            aligned_data['month'] = aligned_data['date'].dt.month
            month_stats = aligned_data.groupby('month')['bias'].agg([
                'mean', 'std', 'count'
            ]).reset_index()
            
            # Save analysis results
            station_stats.to_csv(os.path.join(self.output_dir, "station_bias_stats.csv"), index=False)
            month_stats.to_csv(os.path.join(self.output_dir, "month_bias_stats.csv"), index=False)
            
            # Create bias visualization
            self.visualize_bias(aligned_data)
            
            logger.info("Successfully analyzed bias in aligned data")
            return {
                'overall': overall_stats,
                'station': station_stats,
                'month': month_stats
            }
            
        except Exception as e:
            logger.error(f"Error analyzing bias: {str(e)}")
            return None
    
    def visualize_bias(self, aligned_data):
        """
        Create visualizations of bias in the aligned data.
        
        Args:
            aligned_data (pd.DataFrame): Aligned data
        """
        try:
            logger.info("Creating bias visualizations...")
            
            # Create output directory for visualizations
            viz_dir = os.path.join(self.output_dir, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            
            # 1. Histogram of bias
            plt.figure(figsize=(10, 6))
            sns.histplot(aligned_data['bias'], kde=True)
            plt.title('Distribution of Temperature Bias')
            plt.xlabel('Bias (°C)')
            plt.ylabel('Count')
            plt.savefig(os.path.join(viz_dir, "bias_histogram.png"))
            plt.close()
            
            # 2. Box plot of bias by month
            plt.figure(figsize=(12, 6))
            sns.boxplot(x='month', y='bias', data=aligned_data)
            plt.title('Temperature Bias by Month')
            plt.xlabel('Month')
            plt.ylabel('Bias (°C)')
            plt.savefig(os.path.join(viz_dir, "bias_by_month.png"))
            plt.close()
            
            # 3. Scatter plot of observed vs. forecast temperature
            plt.figure(figsize=(10, 10))
            sns.scatterplot(x='forecast_temp', y='observed_temp', data=aligned_data, alpha=0.1)
            plt.plot([aligned_data['forecast_temp'].min(), aligned_data['forecast_temp'].max()],
                    [aligned_data['forecast_temp'].min(), aligned_data['forecast_temp'].max()],
                    'r--')
            plt.title('Observed vs. Forecast Temperature')
            plt.xlabel('Forecast Temperature (°C)')
            plt.ylabel('Observed Temperature (°C)')
            plt.savefig(os.path.join(viz_dir, "observed_vs_forecast.png"))
            plt.close()
            
            # 4. Map of mean bias by station
            station_mean_bias = aligned_data.groupby('station_id').agg({
                'latitude': 'first',
                'longitude': 'first',
                'bias': 'mean'
            }).reset_index()
            
            plt.figure(figsize=(12, 8))
            plt.scatter(station_mean_bias['longitude'], station_mean_bias['latitude'],
                       c=station_mean_bias['bias'], cmap='RdBu_r', s=50)
            plt.colorbar(label='Mean Bias (°C)')
            plt.title('Mean Temperature Bias by Station')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.savefig(os.path.join(viz_dir, "bias_map.png"))
            plt.close()
            
            logger.info("Successfully created bias visualizations")
            
        except Exception as e:
            logger.error(f"Error creating bias visualizations: {str(e)}")

if __name__ == "__main__":
    # Example usage
    gsod_dir = os.path.join("data", "raw", "gsod")
    ncep_dir = os.path.join("data", "raw", "ncep")
    output_dir = os.path.join("data", "processed")
    
    aligner = DataAligner(gsod_dir, ncep_dir, output_dir, start_year=2018, end_year=2023)
    
    # Align data
    aligned_data = aligner.align_all_data()
    
    if aligned_data is not None:
        # Analyze bias
        bias_analysis = aligner.analyze_bias(aligned_data)
        print(f"Overall mean bias: {bias_analysis['overall']['mean_bias']:.2f}°C") 