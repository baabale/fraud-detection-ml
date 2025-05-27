"""
Script for monitoring model performance and detecting data drift.
This helps ensure the fraud detection models remain effective over time.
"""
import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
import joblib
from scipy.stats import ks_2samp

# Configure TensorFlow to use GPU if available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth for all GPUs to prevent TensorFlow from allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found {len(gpus)} GPU(s). GPU acceleration enabled for model monitoring.")
    except RuntimeError as e:
        print(f"Error configuring GPUs: {e}")

# Add the project root to the path to ensure imports work from any directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the configuration manager
from src.utils.config_manager import config

def load_reference_data(reference_data_path):
    """
    Load reference data used for training the model.
    
    Args:
        reference_data_path (str): Path to reference data
        
    Returns:
        DataFrame: Reference data
    """
    print(f"Loading reference data from {reference_data_path}")
    
    if reference_data_path.endswith('.parquet'):
        df = pd.read_parquet(reference_data_path)
    elif reference_data_path.endswith('.csv'):
        df = pd.read_csv(reference_data_path)
    else:
        raise ValueError(f"Unsupported file format: {reference_data_path}")
    
    print(f"Loaded reference data with {len(df)} samples and {len(df.columns)} features")
    return df

def load_production_data(production_data_path):
    """
    Load production data for monitoring.
    
    Args:
        production_data_path (str): Path to production data
        
    Returns:
        DataFrame: Production data
    """
    print(f"Loading production data from {production_data_path}")
    
    if production_data_path.endswith('.parquet'):
        df = pd.read_parquet(production_data_path)
    elif production_data_path.endswith('.csv'):
        df = pd.read_csv(production_data_path)
    elif production_data_path.endswith('.json'):
        df = pd.read_json(production_data_path)
    else:
        raise ValueError(f"Unsupported file format: {production_data_path}")
    
    print(f"Loaded production data with {len(df)} samples and {len(df.columns)} features")
    return df

def load_model_artifacts(model_dir):
    """
    Load model and associated artifacts.
    
    Args:
        model_dir (str): Directory containing model artifacts
        
    Returns:
        tuple: Model, scaler, feature names
    """
    print(f"Loading model artifacts from {model_dir}")
    
    # Load model
    model_path = os.path.join(model_dir, 'classification_model.h5')
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print(f"Loaded model from {model_path}")
    else:
        model = None
        print(f"Warning: Model not found at {model_path}")
    
    # Load scaler
    scaler_path = os.path.join(model_dir, 'scaler.joblib')
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"Loaded scaler from {scaler_path}")
    else:
        scaler = None
        print(f"Warning: Scaler not found at {scaler_path}")
    
    # Load feature names
    feature_names_path = os.path.join(model_dir, 'feature_names.json')
    if os.path.exists(feature_names_path):
        with open(feature_names_path, 'r') as f:
            feature_names = json.load(f)
        print(f"Loaded {len(feature_names)} feature names")
    else:
        feature_names = None
        print(f"Warning: Feature names not found at {feature_names_path}")
    
    return model, scaler, feature_names

def calculate_performance_metrics(y_true, y_pred):
    """
    Calculate performance metrics for the model.
    
    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        
    Returns:
        dict: Performance metrics
    """
    # Check if there are any positive predictions
    if np.sum(y_pred) == 0:
        print("Warning: No positive predictions found. This may indicate a model issue.")
    
    # Calculate metrics with zero_division=0 to avoid warnings
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }
    
    # Add additional context if no positive predictions
    if np.sum(y_pred) == 0:
        metrics['note'] = "No positive predictions - model may need retraining"
    
    return metrics

def detect_data_drift(reference_df, production_df, feature_names, threshold=0.05):
    """
    Detect data drift between reference and production data.
    
    Args:
        reference_df (DataFrame): Reference data
        production_df (DataFrame): Production data
        feature_names (list): List of feature names to check
        threshold (float): P-value threshold for drift detection
        
    Returns:
        dict: Drift detection results
    """
    drift_results = {}
    drifted_features = []
    
    # Select only common features
    common_features = [f for f in feature_names if f in reference_df.columns and f in production_df.columns]
    
    # Check drift for each feature
    for feature in common_features:
        # Skip non-numeric features
        if not np.issubdtype(reference_df[feature].dtype, np.number):
            continue
        
        # Perform Kolmogorov-Smirnov test
        ks_statistic, p_value = ks_2samp(
            reference_df[feature].dropna(),
            production_df[feature].dropna()
        )
        
        drift_results[feature] = {
            'ks_statistic': ks_statistic,
            'p_value': p_value,
            'drift_detected': p_value < threshold
        }
        
        if p_value < threshold:
            drifted_features.append(feature)
    
    # Calculate overall drift percentage
    drift_percentage = len(drifted_features) / len(common_features) if common_features else 0
    
    return {
        'feature_drift': drift_results,
        'drifted_features': drifted_features,
        'drift_percentage': drift_percentage,
        'drift_threshold': threshold
    }

def visualize_feature_drift(reference_df, production_df, drifted_features, output_dir):
    """
    Visualize drift in feature distributions.
    
    Args:
        reference_df (DataFrame): Reference data
        production_df (DataFrame): Production data
        drifted_features (list): List of features with detected drift
        output_dir (str): Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Limit to top 10 drifted features if there are more
    features_to_plot = drifted_features[:10] if len(drifted_features) > 10 else drifted_features
    
    for feature in features_to_plot:
        plt.figure(figsize=(10, 6))
        
        # Plot distributions
        sns.histplot(reference_df[feature].dropna(), label='Reference', alpha=0.5, kde=True)
        sns.histplot(production_df[feature].dropna(), label='Production', alpha=0.5, kde=True)
        
        plt.title(f'Distribution Drift: {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.legend()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, f'drift_{feature}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def monitor_model_performance(model, scaler, reference_df, production_df, feature_names, output_dir):
    """
    Monitor model performance over time.
    
    Args:
        model: Trained model
        scaler: Feature scaler
        reference_df (DataFrame): Reference data
        production_df (DataFrame): Production data
        feature_names (list): List of feature names
        output_dir (str): Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Check which features are available in both datasets
    available_features = [f for f in feature_names if f in reference_df.columns and f in production_df.columns]
    missing_features = [f for f in feature_names if f not in available_features]
    
    if missing_features:
        print(f"Warning: The following features are missing in one or both datasets: {missing_features}")
        print(f"Using only the {len(available_features)} available features: {available_features}")
    
    if not available_features:
        print("Error: No common features found between reference and production data")
        return None
    
    # Get the expected input shape from the model
    expected_features = None
    if model is not None:
        try:
            # Try to get the expected input shape from the model
            input_shape = model.layers[0].input_shape
            if input_shape is not None:
                expected_feature_count = input_shape[1] if isinstance(input_shape, tuple) else input_shape[0][1]
                print(f"Model expects {expected_feature_count} features as input")
                
                # If we have more features than the model expects, select only the first N features
                if len(available_features) > expected_feature_count:
                    expected_features = available_features[:expected_feature_count]
                    print(f"Using only the first {expected_feature_count} features: {expected_features}")
        except Exception as e:
            print(f"Warning: Could not determine expected input shape from model directly: {e}")
            
            # Try to infer the expected shape by examining the model summary
            try:
                # For TensorFlow models, we can try to get the first layer's input shape
                first_layer = model.layers[0]
                if hasattr(first_layer, 'input_spec') and first_layer.input_spec is not None:
                    if hasattr(first_layer.input_spec[0], 'axes') and -1 in first_layer.input_spec[0].axes:
                        expected_feature_count = first_layer.input_spec[0].axes[-1]
                        print(f"Inferred that model expects {expected_feature_count} features as input")
                        
                        # If we have more features than the model expects, select only the first N features
                        if len(available_features) > expected_feature_count:
                            expected_features = available_features[:expected_feature_count]
                            print(f"Using only the first {expected_feature_count} features: {expected_features}")
            except Exception as nested_e:
                print(f"Warning: Could not infer expected input shape: {nested_e}")
                
            # Hardcoded fallback for this specific model which we know expects 6 features
            if expected_features is None and len(available_features) > 6:
                expected_features = available_features[:6]
                print(f"Using hardcoded knowledge that model expects 6 features: {expected_features}")
    
    # Use expected features if available, otherwise use all available features
    features_to_use = expected_features if expected_features is not None else available_features
    
    # Prepare reference data
    X_ref = reference_df[features_to_use].values
    y_ref = reference_df['is_fraud'].values if 'is_fraud' in reference_df.columns else None
    
    # Prepare production data
    X_prod = production_df[features_to_use].values
    y_prod = production_df['is_fraud'].values if 'is_fraud' in production_df.columns else None
    
    # Scale data if scaler is available
    if scaler is not None:
        try:
            X_ref = scaler.transform(X_ref)
            X_prod = scaler.transform(X_prod)
        except Exception as e:
            print(f"Warning: Could not scale data: {e}")
            print("Proceeding with unscaled data")
    
    # Make predictions if model is available
    if model is not None:
        try:
            y_ref_pred = (model.predict(X_ref) >= 0.5).astype(int).flatten() if y_ref is not None else None
            y_prod_pred = (model.predict(X_prod) >= 0.5).astype(int).flatten() if y_prod is not None else None
        except Exception as e:
            print(f"Warning: Could not make predictions: {e}")
            print("Skipping performance metrics calculation")
            return None
        
        # Calculate metrics
        results = {}
        
        if y_ref is not None and y_ref_pred is not None:
            results['reference_metrics'] = calculate_performance_metrics(y_ref, y_ref_pred)
        
        if y_prod is not None and y_prod_pred is not None:
            results['production_metrics'] = calculate_performance_metrics(y_prod, y_prod_pred)
            
            # Calculate the difference between reference and production metrics
            if 'reference_metrics' in results and 'production_metrics' in results:
                results['metrics_difference'] = {}
                for metric in results['reference_metrics']:
                    # Skip non-numeric metrics like 'note'
                    if metric != 'note' and isinstance(results['reference_metrics'][metric], (int, float)):
                        results['metrics_difference'][metric] = results['production_metrics'][metric] - results['reference_metrics'][metric]
        
        # Save results
        with open(os.path.join(output_dir, 'performance_metrics.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create performance comparison visualization
        if 'reference_metrics' in results and 'production_metrics' in results:
            # Filter out non-numeric metrics
            numeric_metrics = []
            ref_values = []
            prod_values = []
            
            for metric in results['reference_metrics']:
                if metric != 'note' and isinstance(results['reference_metrics'][metric], (int, float)):
                    numeric_metrics.append(metric)
                    ref_values.append(results['reference_metrics'][metric])
                    prod_values.append(results['production_metrics'][metric])
            
            if numeric_metrics:  # Only create visualization if there are numeric metrics
                x = np.arange(len(numeric_metrics))
                width = 0.35
                
                fig, ax = plt.subplots(figsize=(10, 6))
                plt.bar(x - width/2, ref_values, width, label='Reference')
                plt.bar(x + width/2, prod_values, width, label='Production')
                
                plt.xlabel('Metrics')
                plt.ylabel('Value')
                plt.title('Performance Comparison')
                plt.xticks(x, numeric_metrics)
                plt.legend()
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
                plt.close()
    
    return results if 'results' in locals() else None

def generate_recommendations(drift_percentage, performance_results=None):
    """
    Generate recommendations based on the drift percentage and model performance.
    
    Args:
        drift_percentage (float): Percentage of features with drift
        performance_results (dict, optional): Model performance metrics
        
    Returns:
        list: Recommendations
    """
    recommendations = []
    
    # Data drift recommendations
    if drift_percentage > 50:
        recommendations.append("**High data drift detected**: Consider retraining the model with more recent data.")
    elif drift_percentage > 20:
        recommendations.append("**Moderate data drift detected**: Monitor model performance closely.")
    else:
        recommendations.append("**Low data drift detected**: No immediate action required.")
    
    # Performance-based recommendations
    if performance_results:
        # Check for no positive predictions
        if ('production_metrics' in performance_results and 
            'note' in performance_results['production_metrics'] and 
            'no positive predictions' in performance_results['production_metrics']['note'].lower()):
            recommendations.append("**Model prediction issue**: The model is not predicting any positive cases. Consider adjusting the classification threshold or retraining with balanced data.")
        
        # Check for significant performance drop
        if ('metrics_difference' in performance_results and 
            'accuracy' in performance_results['metrics_difference'] and 
            performance_results['metrics_difference']['accuracy'] < -0.1):
            recommendations.append("**Performance degradation**: Model accuracy has decreased by more than 10%. Investigate feature importance and consider model retraining.")
        
        # Check for performance improvement (might indicate data leakage or other issues)
        if ('metrics_difference' in performance_results and 
            'accuracy' in performance_results['metrics_difference'] and 
            performance_results['metrics_difference']['accuracy'] > 0.1):
            recommendations.append("**Unexpected performance improvement**: Model accuracy has increased by more than 10%. Verify that there is no data leakage or sampling bias in the production data.")
    
    return recommendations

def generate_monitoring_report(performance_results, drift_results, output_dir):
    """
    Generate a comprehensive monitoring report.
    
    Args:
        performance_results (dict): Model performance results
        drift_results (dict): Data drift detection results
        output_dir (str): Directory to save the report
    """
    report_path = os.path.join(output_dir, 'monitoring_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Fraud Detection Model Monitoring Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Performance section
        f.write("## Model Performance\n\n")
        
        if performance_results:
            if 'reference_metrics' in performance_results:
                f.write("### Reference Data Performance\n\n")
                f.write("| Metric | Value |\n")
                f.write("| ------ | ----- |\n")
                for metric, value in performance_results['reference_metrics'].items():
                    if metric != 'note':  # Skip the note for the table
                        f.write(f"| {metric} | {value:.4f} |\n")
                # Add note if it exists
                if 'note' in performance_results['reference_metrics']:
                    f.write("\n")
                    f.write(f"**Note:** {performance_results['reference_metrics']['note']}\n")
                f.write("\n")
            
            if 'production_metrics' in performance_results:
                f.write("### Production Data Performance\n\n")
                f.write("| Metric | Value |\n")
                f.write("| ------ | ----- |\n")
                for metric, value in performance_results['production_metrics'].items():
                    if metric != 'note':  # Skip the note for the table
                        f.write(f"| {metric} | {value:.4f} |\n")
                # Add note if it exists
                if 'note' in performance_results['production_metrics']:
                    f.write("\n")
                    f.write(f"**Note:** {performance_results['production_metrics']['note']}\n")
                f.write("\n")
            
            if 'metrics_difference' in performance_results:
                f.write("### Performance Difference (Production - Reference)\n\n")
                f.write("| Metric | Difference |\n")
                f.write("| ------ | ---------- |\n")
                for metric, value in performance_results['metrics_difference'].items():
                    f.write(f"| {metric} | {value:.4f} |\n")
                f.write("\n")
        else:
            f.write("No performance metrics available.\n\n")
        
        # Data drift section
        f.write("## Data Drift Analysis\n\n")
        
        if drift_results:
            f.write(f"Drift threshold (p-value): {drift_results['drift_threshold']}\n\n")
            f.write(f"Overall drift percentage: {drift_results['drift_percentage']:.2%}\n\n")
            
            if drift_results['drifted_features']:
                f.write("### Drifted Features\n\n")
                f.write("| Feature | KS Statistic | P-value |\n")
                f.write("| ------- | ------------ | ------- |\n")
                
                for feature in drift_results['drifted_features']:
                    feature_result = drift_results['feature_drift'][feature]
                    f.write(f"| {feature} | {feature_result['ks_statistic']:.4f} | {feature_result['p_value']:.6f} |\n")
                
                f.write("\n")
            else:
                f.write("No significant drift detected in any feature.\n\n")
        else:
            f.write("No drift analysis results available.\n\n")
        
        # Recommendations section
        f.write("## Recommendations\n\n")
        
        recommendations = generate_recommendations(drift_results['drift_percentage'], performance_results)
        
        for recommendation in recommendations:
            f.write(f"- {recommendation}\n")
        
        f.write("\n")
        f.write("---\n")
        f.write("*This report was generated automatically by the model monitoring system.*\n")
    
    print(f"Monitoring report generated: {report_path}")

def main():
    """
    Main function to run model monitoring.
    """
    parser = argparse.ArgumentParser(description='Monitor fraud detection model performance')
    parser.add_argument('--reference-data', type=str,
                        help='Path to reference data used for training')
    parser.add_argument('--production-data', type=str,
                        help='Path to production data for monitoring')
    parser.add_argument('--model-dir', type=str,
                        help='Directory containing model artifacts')
    parser.add_argument('--output-dir', type=str,
                        help='Directory to save monitoring results')
    parser.add_argument('--drift-threshold', type=float, default=0.05,
                        help='P-value threshold for drift detection')
    args = parser.parse_args()
    
    # Use configuration values if arguments are not provided
    reference_data_path = args.reference_data or config.get_data_path('processed_path')
    production_data_path = args.production_data or config.get_data_path('test_path')
    model_dir = args.model_dir or config.get_model_path()
    output_dir = args.output_dir or config.get_absolute_path(config.get('evaluation.output_dir', 'results/monitoring'))
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    reference_df = load_reference_data(reference_data_path)
    production_df = load_production_data(production_data_path)
    
    # Load model artifacts
    model, scaler, feature_names = load_model_artifacts(model_dir)
    
    # Detect data drift
    drift_results = detect_data_drift(
        reference_df, 
        production_df, 
        feature_names, 
        args.drift_threshold
    )
    
    # Visualize drifted features
    if drift_results['drifted_features']:
        visualize_feature_drift(
            reference_df,
            production_df,
            drift_results['drifted_features'],
            os.path.join(output_dir, 'drift_visualizations')
        )
    
    # Monitor model performance
    performance_results = monitor_model_performance(
        model,
        scaler,
        reference_df,
        production_df,
        feature_names,
        output_dir
    )
    
    # Generate monitoring report
    generate_monitoring_report(
        performance_results,
        drift_results,
        output_dir
    )
    
    print("Model monitoring completed successfully!")

if __name__ == "__main__":
    main()
