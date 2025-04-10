import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from scipy import stats
import sys
import os
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.app.model_serving import ModelServer

def load_and_process_data(file_path):
    """Load and process a dataset, returning original features and actual bias."""
    data = pd.read_csv(file_path)
    
    # Calculate actual bias (difference between forecast and actual)
    data['actual_bias'] = data['temperature'] - data['temp_avg']
    
    # Get model prediction
    model = ModelServer(
        model_path=str(project_root / "logs/pc_training_corrected_v5/checkpoints/bias_correction-epoch=19-val_loss=0.00.ckpt"),
        device='cpu'
    )
    
    features = np.column_stack([
        data['temperature'],
        data['humidity'],
        data['wind_speed_model'],
        data['wind_direction_model'],
        data['cloud_cover_low'],
        data['cloud_cover_mid'],
        data['cloud_cover_high']
    ])
    
    predictions = model.predict(features)
    data['predicted_bias'] = predictions[0].squeeze()  # Get the first element (predictions) from the tuple
    
    return data

def analyze_feature_relationships(data, location):
    """Analyze relationships between features and bias."""
    features = ['temperature', 'humidity', 'wind_speed_model', 
               'wind_direction_model', 'cloud_cover_low', 
               'cloud_cover_mid', 'cloud_cover_high']
    
    # Create correlation matrix
    corr_matrix = data[features + ['actual_bias']].corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title(f'Feature Correlations with Actual Bias - {location}')
    plt.tight_layout()
    plt.savefig(f'analysis_correlation_{location}.png')
    plt.close()
    
    # Plot feature vs bias relationships
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.ravel()
    
    for i, feature in enumerate(features):
        sns.scatterplot(data=data, x=feature, y='actual_bias', ax=axes[i])
        axes[i].set_title(f'{feature} vs Actual Bias')
        
        # Add trend line
        z = np.polyfit(data[feature], data['actual_bias'], 1)
        p = np.poly1d(z)
        axes[i].plot(data[feature], p(data[feature]), "r--", alpha=0.8)
        
        # Calculate and display correlation
        corr = stats.pearsonr(data[feature], data['actual_bias'])[0]
        axes[i].text(0.05, 0.95, f'Correlation: {corr:.2f}', 
                    transform=axes[i].transAxes)
    
    plt.suptitle(f'Feature Relationships with Bias - {location}')
    plt.tight_layout()
    plt.savefig(f'analysis_feature_relationships_{location}.png')
    plt.close()

def analyze_error_patterns(data, location):
    """Analyze patterns in model errors."""
    # Calculate prediction error
    data['prediction_error'] = data['predicted_bias'] - data['actual_bias']
    
    # Time series of errors
    plt.figure(figsize=(15, 6))
    plt.plot(data.index, data['prediction_error'])
    plt.title(f'Prediction Error Over Time - {location}')
    plt.xlabel('Sample Index')
    plt.ylabel('Prediction Error (°C)')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.tight_layout()
    plt.savefig(f'analysis_error_time_{location}.png')
    plt.close()
    
    # Error distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data['prediction_error'], kde=True)
    plt.title(f'Distribution of Prediction Errors - {location}')
    plt.xlabel('Prediction Error (°C)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'analysis_error_distribution_{location}.png')
    plt.close()
    
    # Find worst performing cases
    largest = data.nlargest(5, 'prediction_error')
    smallest = data.nsmallest(5, 'prediction_error')
    worst_cases = pd.concat([largest, smallest], axis=0)
    
    return worst_cases[['date', 'temperature', 'temp_avg', 
                       'actual_bias', 'predicted_bias', 'prediction_error']]

def analyze_conditional_performance(data):
    """Analyze model performance under different conditions."""
    conditions = {
        'High Temperature': data['temperature'] > data['temperature'].quantile(0.75),
        'Low Temperature': data['temperature'] < data['temperature'].quantile(0.25),
        'High Humidity': data['humidity'] > data['humidity'].quantile(0.75),
        'Low Humidity': data['humidity'] < data['humidity'].quantile(0.25),
        'High Wind': data['wind_speed_model'] > data['wind_speed_model'].quantile(0.75),
        'Low Wind': data['wind_speed_model'] < data['wind_speed_model'].quantile(0.25),
    }
    
    results = []
    for condition_name, mask in conditions.items():
        subset = data[mask]
        rmse = np.sqrt(mean_squared_error(subset['actual_bias'], 
                                        subset['predicted_bias']))
        mean_error = (subset['predicted_bias'] - subset['actual_bias']).mean()
        results.append({
            'Condition': condition_name,
            'RMSE': rmse,
            'Mean Error': mean_error,
            'Sample Size': len(subset)
        })
    
    return pd.DataFrame(results)

def main():
    # Create output directory
    output_dir = project_root / "analysis_results"
    output_dir.mkdir(exist_ok=True)
    os.chdir(output_dir)
    
    # Analyze Amsterdam data
    print("Analyzing Amsterdam data...")
    amsterdam_data = load_and_process_data(
        str(project_root / "data/processed/Amsterdam_March2025_aligned.csv")
    )
    analyze_feature_relationships(amsterdam_data, "Amsterdam")
    worst_cases_ams = analyze_error_patterns(amsterdam_data, "Amsterdam")
    conditional_perf_ams = analyze_conditional_performance(amsterdam_data)
    
    # Analyze India data
    print("\nAnalyzing India data...")
    india_data = load_and_process_data(
        str(project_root / "data/processed/SAFDARJUNG_India_March2025_aligned.csv")
    )
    analyze_feature_relationships(india_data, "India")
    worst_cases_ind = analyze_error_patterns(india_data, "India")
    conditional_perf_ind = analyze_conditional_performance(india_data)
    
    # Save results to text file
    with open('analysis_results.txt', 'w') as f:
        f.write("Analysis Results\n")
        f.write("================\n\n")
        
        f.write("Amsterdam Analysis\n")
        f.write("-----------------\n")
        f.write("Worst performing cases:\n")
        f.write(worst_cases_ams.to_string())
        f.write("\n\nConditional performance:\n")
        f.write(conditional_perf_ams.to_string())
        
        f.write("\n\nIndia Analysis\n")
        f.write("-------------\n")
        f.write("Worst performing cases:\n")
        f.write(worst_cases_ind.to_string())
        f.write("\n\nConditional performance:\n")
        f.write(conditional_perf_ind.to_string())
    
    print("\nAnalysis complete! Results saved to analysis_results.txt")
    print("Plots saved as PNG files in the current directory")

if __name__ == "__main__":
    main()
