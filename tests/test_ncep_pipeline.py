import pytest
import os
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
import logging
from src.data.download_ncep import NCEPDownloader

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # Force the logging configuration
)
logger = logging.getLogger(__name__)

@pytest.fixture
def test_data_dir(tmp_path):
    """Create a temporary directory for test data."""
    logger.info(f"Created temporary directory at: {tmp_path}")
    return tmp_path

@pytest.fixture
def test_year():
    """Return the test year."""
    return 2022

def test_ncep_download(test_data_dir, test_year):
    """Test NCEP data download."""
    logger.info("Starting NCEP data download test...")
    
    # Initialize NCEP downloader
    ncep_dir = os.path.join(test_data_dir, "ncep")
    logger.info(f"Initializing NCEP downloader with directory: {ncep_dir}")
    downloader = NCEPDownloader(ncep_dir, test_year, test_year)
    
    # Download NCEP data
    logger.info("Attempting to download NCEP data...")
    download_results = downloader.download_all()
    logger.info(f"Download results: {download_results}")
    
    # Check if download was successful
    assert download_results, "NCEP download failed"
    
    # Process NCEP data
    logger.info("Attempting to process NCEP data...")
    processed_data = downloader.process_all()
    logger.info(f"Processing results: {processed_data}")
    
    # Check if processing was successful
    assert processed_data, "NCEP processing failed"
    
    # Combine variables
    logger.info("Attempting to combine variables...")
    combined_data = downloader.combine_all()
    logger.info(f"Combination results: {combined_data}")
    
    # Check if combination was successful
    assert combined_data, "NCEP combination failed"
    
    # Check if combined file exists
    combined_file = os.path.join(ncep_dir, f"combined_{test_year}.nc")
    logger.info(f"Checking for combined file at: {combined_file}")
    assert os.path.exists(combined_file), "Combined NCEP file not found"
    
    # Check if combined file contains required variables
    logger.info("Checking combined file contents...")
    ds = xr.open_dataset(combined_file)
    required_vars = ['t2m', 'rh2m', 'u10', 'v10', 'tcc']
    for var in required_vars:
        assert var in ds.data_vars, f"Required variable {var} not found in combined NCEP data"
    logger.info("All required variables found in combined file")

def test_ncep_variable_processing(test_data_dir, test_year):
    """Test processing of individual NCEP variables."""
    logger.info("Testing NCEP variable processing...")
    
    # Initialize NCEP downloader
    ncep_dir = os.path.join(test_data_dir, "ncep")
    downloader = NCEPDownloader(ncep_dir, test_year, test_year)
    
    # Test each variable
    for variable in downloader.variables:
        # Download variable
        success = downloader.download_variable(variable, test_year)
        assert success, f"Failed to download {variable}"
        
        # Process variable
        success = downloader.process_variable(variable, test_year)
        assert success, f"Failed to process {variable}"
        
        # Check if processed file exists
        processed_file = os.path.join(ncep_dir, f"processed_{downloader.variables[variable]}_{test_year}.nc")
        assert os.path.exists(processed_file), f"Processed file for {variable} not found"
        
        # Check if processed file contains the variable
        ds = xr.open_dataset(processed_file)
        assert downloader.variables[variable] in ds.data_vars, f"Variable {downloader.variables[variable]} not found in processed file"

def test_ncep_data_format(test_data_dir, test_year):
    """Test NCEP data format and structure."""
    logger.info("Testing NCEP data format...")
    
    # Initialize NCEP downloader
    ncep_dir = os.path.join(test_data_dir, "ncep")
    downloader = NCEPDownloader(ncep_dir, test_year, test_year)
    
    # Download and process one variable
    variable = list(downloader.variables.keys())[0]
    success = downloader.download_variable(variable, test_year)
    assert success, f"Failed to download {variable}"
    
    success = downloader.process_variable(variable, test_year)
    assert success, f"Failed to process {variable}"
    
    # Check data format
    processed_file = os.path.join(ncep_dir, f"processed_{downloader.variables[variable]}_{test_year}.nc")
    ds = xr.open_dataset(processed_file)
    
    # Check dimensions
    assert 'time' in ds.dims, "Time dimension not found"
    assert 'latitude' in ds.dims, "Latitude dimension not found"
    assert 'longitude' in ds.dims, "Longitude dimension not found"
    
    # Check coordinates
    assert 'time' in ds.coords, "Time coordinate not found"
    assert 'latitude' in ds.coords, "Latitude coordinate not found"
    assert 'longitude' in ds.coords, "Longitude coordinate not found"
    
    # Check data types
    assert ds[downloader.variables[variable]].dtype in [np.float32, np.float64], f"Variable {downloader.variables[variable]} should be float"
    assert ds.time.dtype == np.dtype('datetime64[ns]'), "Time should be datetime64"
    assert ds.latitude.dtype in [np.float32, np.float64], "Latitude should be float"
    assert ds.longitude.dtype in [np.float32, np.float64], "Longitude should be float"
    
    # Check value ranges
    assert ds.latitude.min() >= -90 and ds.latitude.max() <= 90, "Latitude should be between -90 and 90"
    assert ds.longitude.min() >= -180 and ds.longitude.max() <= 180, "Longitude should be between -180 and 180" 