import os
import requests
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from tqdm import tqdm
import logging
import netCDF4 as nc

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NCEPDownloader:
    """
    Class to download and process NCEP/NCAR Reanalysis 1 data.
    """
    
    def __init__(self, output_dir, start_year=2018, end_year=2023):
        """
        Initialize the NCEP downloader.
        
        Args:
            output_dir (str): Directory to save the downloaded data
            start_year (int): Start year for data download
            end_year (int): End year for data download
        """
        # Using NCEP/NCAR Reanalysis 1 data from NOAA
        self.base_url = "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis.dailyavgs/surface_gauss"
        self.output_dir = output_dir
        self.start_year = start_year
        self.end_year = end_year
        
        # Variables to download (updated with correct variable names)
        self.variables = {
            'air.2m.gauss': 't2m',      # 2m temperature
            'rhum.2m.gauss': 'rh2m',    # 2m relative humidity
            'uwnd.10m.gauss': 'u10',    # 10m U wind component
            'vwnd.10m.gauss': 'v10',    # 10m V wind component
            'tcdc.eatm.gauss': 'tcc'    # Total cloud cover
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Print initialization info
        logger.info(f"Initialized NCEP downloader:")
        logger.info(f"  Output directory: {output_dir}")
        logger.info(f"  Year range: {start_year}-{end_year}")
        logger.info(f"  Variables: {list(self.variables.keys())}")
    
    def download_variable(self, variable, year):
        """
        Download NCEP data for a specific variable and year.
        
        Args:
            variable (str): Variable to download
            year (int): Year to download data for
            
        Returns:
            bool: True if download was successful, False otherwise
        """
        url = f"{self.base_url}/{variable}.{year}.nc"
        output_file = os.path.join(self.output_dir, f"{variable}.{year}.nc")
        
        # Skip if file already exists
        if os.path.exists(output_file):
            logger.info(f"File {output_file} already exists, skipping download")
            return True
        
        try:
            logger.info(f"Downloading {variable} data for {year}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Get file size for progress bar
            total_size = int(response.headers.get('content-length', 0))
            
            # Create progress bar with more visible format
            pbar = tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc=f"Downloading {variable}.{year}",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            )
            
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            pbar.close()
            logger.info(f"Successfully downloaded {variable} data for {year}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading {variable} data for {year}: {str(e)}")
            return False
    
    def download_all(self):
        """
        Download NCEP data for all variables and years in the specified range.
        
        Returns:
            dict: Dictionary mapping (variable, year) tuples to download success (True/False)
        """
        download_results = {}
        
        for variable in self.variables:
            for year in range(self.start_year, self.end_year + 1):
                success = self.download_variable(variable, year)
                download_results[(variable, year)] = success
        
        logger.info(f"Download results: {download_results}")
        return download_results
    
    def process_variable(self, variable, year):
        """
        Process NCEP data for a specific variable and year.
        
        Args:
            variable (str): Variable to process
            year (int): Year to process data for
            
        Returns:
            bool: True if processing was successful, False otherwise
        """
        input_file = os.path.join(self.output_dir, f"{variable}.{year}.nc")
        output_file = os.path.join(self.output_dir, f"processed_{self.variables[variable]}_{year}.nc")
        
        if not os.path.exists(input_file):
            logger.error(f"File {input_file} not found")
            return False
        
        try:
            logger.info(f"Processing {variable} data for {year}...")
            
            # Read the NetCDF file
            ds = xr.open_dataset(input_file)
            
            # Extract the variable
            var_name = list(ds.data_vars)[0]
            data = ds[var_name]
            
            # Convert to DataFrame
            df = data.to_dataframe()
            
            # Reset index to get time as a column
            df = df.reset_index()
            
            # Rename columns
            df = df.rename(columns={'time': 'timestamp', var_name: self.variables[variable]})
            
            # Save to NetCDF
            ds_out = xr.Dataset()
            ds_out[self.variables[variable]] = xr.DataArray(
                df[self.variables[variable]].values.reshape(data.shape),
                dims=data.dims,
                attrs=data.attrs
            )
            
            # Add time dimension
            ds_out['time'] = xr.DataArray(
                df['timestamp'].values,
                dims=['time']
            )
            
            # Save to file
            ds_out.to_netcdf(output_file)
            
            logger.info(f"Successfully processed {variable} data for {year}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {variable} data for {year}: {str(e)}")
            return False
    
    def process_all(self):
        """
        Process all downloaded NCEP data.
        
        Returns:
            dict: Dictionary mapping (variable, year) tuples to processed data
        """
        processed_data = {}
        
        for variable in self.variables:
            for year in range(self.start_year, self.end_year + 1):
                if self.process_variable(variable, year):
                    processed_data[(self.variables[variable], year)] = True
        
        logger.info(f"Processed data: {list(processed_data.keys())}")
        return processed_data
    
    def combine_variables(self, year):
        """
        Combine all variables for a specific year.
        
        Args:
            year (int): Year to combine variables for
            
        Returns:
            bool: True if combination was successful, False otherwise
        """
        logger.info(f"Combining variables for {year}...")
        
        try:
            # Create a list to store all datasets
            datasets = []
            
            # Load each variable
            for variable in self.variables.values():
                processed_file = os.path.join(self.output_dir, f"processed_{variable}_{year}.nc")
                
                if not os.path.exists(processed_file):
                    logger.warning(f"Processed file for {variable} in {year} not found")
                    continue
                
                # Load the dataset
                ds = xr.open_dataset(processed_file)
                datasets.append(ds)
            
            if not datasets:
                logger.error(f"No datasets found for {year}")
                return False
            
            # Merge all datasets
            combined_ds = xr.merge(datasets)
            
            # Save to file
            output_file = os.path.join(self.output_dir, f"combined_{year}.nc")
            combined_ds.to_netcdf(output_file)
            
            logger.info(f"Successfully combined variables for {year}")
            return True
            
        except Exception as e:
            logger.error(f"Error combining variables for {year}: {str(e)}")
            return False
    
    def combine_all(self):
        """
        Combine all variables for all years.
        
        Returns:
            dict: Dictionary mapping years to combined data
        """
        combined_data = {}
        
        for year in range(self.start_year, self.end_year + 1):
            if self.combine_variables(year):
                combined_data[year] = True
        
        logger.info(f"Combined data for years: {list(combined_data.keys())}")
        return combined_data 