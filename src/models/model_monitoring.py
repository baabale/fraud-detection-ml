"""
Script for monitoring model performance and detecting data drift.
This helps ensure the fraud detection models remain effective over time.
"""
import os
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
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }
    
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
    
    # Prepare reference data
    X_ref = reference_df[feature_names].values
    y_ref = reference_df['is_fraud'].values if 'is_fraud' in reference_df.columns else None
    
    # Prepare production data
    X_prod = production_df[feature_names].values
    y_prod = production_df['is_fraud'].values if 'is_fraud' in production_df.columns else None
    
    # Scale data
    if scaler is not None:
        X_ref = scaler.transform(X_ref)
        X_prod = scaler.transform(X_prod)
    
    # Make predictions
    if model is not None:
        y_ref_pred = (model.predict(X_ref) >= 0.5).astype(int).flatten() if y_ref is not None else None
        y_prod_pred = (model.predict(X_prod) >= 0.5).astype(int).flatten() if y_prod is not None else None
        
        # Calculate metrics
        results = {}
        
        if y_ref is not None and y_ref_pred is not None:
            results['reference_metrics'] = calculate_performance_metrics(y_ref, y_ref_pred)
        
        if y_prod is not None and y_prod_pred is not None:
            results['production_metrics'] = calculate_performance_metrics(y_prod, y_prod_pred)
            
            # Calculate performance difference
            if 'reference_metrics' in results:
                results['metrics_difference'] = {
                    metric: results['production_metrics'][metric] - results['reference_metrics'][metric]
                    for metric in results['reference_metrics']
                }
        
        # Save results
        with open(os.path.join(output_dir, 'performance_metrics.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Visualize metrics
        if 'reference_metrics' in results and 'production_metrics' in results:
            metrics = list(results['reference_metrics'].keys())
            ref_values = [results['reference_metrics'][m] for m in metrics]
            prod_values = [results['production_metrics'][m] for m in metrics]
            
            plt.figure(figsize=(10, 6))
            x = np.arange(len(metrics))
            width = 0.35
            
            plt.bar(x - width/2, ref_values, width, label='Reference')
            plt.bar(x + width/2, prod_values, width, label='Production')
            
            plt.xlabel('Metric')
            plt.ylabel('Value')
            plt.title('Model Performance Comparison')
            plt.xticks(x, metrics)
            plt.legend()
            
            plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    return results if 'results' in locals() else None

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
                    f.write(f"| {metric} | {value:.4f} |\n")
                f.write("\n")
            
            if 'production_metrics' in performance_results:
                f.write("### Production Data Performance\n\n")
                f.write("| Metric | Value |\n")
                f.write("| ------ | ----- |\n")
                for metric, value in performance_results['production_metrics'].items():
                    f.write(f"| {metric} | {value:.4f} |\n")
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
        
        if drift_results and drift_results['drift_percentage'] > 0.3:
            f.write("- **High data drift detected**: Consider retraining the model with more recent data.\n")
        elif drift_results and drift_results['drift_percentage'] > 0.1:
            f.write("- **Moderate data drift detected**: Monitor closely and consider updating feature distributions.\n")
        
        if performance_results and 'metrics_difference' in performance_results:
            if performance_results['metrics_difference']['f1_score'] < -0.05:
                f.write("- **Performance degradation detected**: Model performance has decreased significantly.\n")
                f.write("  - Consider retraining the model or investigating the root cause.\n")
            elif performance_results['metrics_difference']['f1_score'] < -0.02:
                f.write("- **Minor performance degradation**: Keep monitoring the model performance.\n")
        
        f.write("\n")
        f.write("---\n")
        f.write("*This report was generated automatically by the model monitoring system.*\n")
    
    print(f"Monitoring report generated: {report_path}")

def main():
    """
    Main function to run model monitoring.
    """
    parser = argparse.ArgumentParser(description='Monitor fraud detection model performance')
    parser.add_argument('--reference-data', type=str, required=True,
                        help='Path to reference data used for training')
    parser.add_argument('--production-data', type=str, required=True,
                        help='Path to production data for monitoring')
    parser.add_argument('--model-dir', type=str, default='../../results/deployment',
                        help='Directory containing model artifacts')
    parser.add_argument('--output-dir', type=str, default='../../results/monitoring',
                        help='Directory to save monitoring results')
    parser.add_argument('--drift-threshold', type=float, default=0.05,
                        help='P-value threshold for drift detection')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    reference_df = load_reference_data(args.reference_data)
    production_df = load_production_data(args.production_data)
    
    # Load model artifacts
    model, scaler, feature_names = load_model_artifacts(args.model_dir)
    
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
            os.path.join(args.output_dir, 'drift_visualizations')
        )
    
    # Monitor model performance
    performance_results = monitor_model_performance(
        model,
        scaler,
        reference_df,
        production_df,
        feature_names,
        args.output_dir
    )
    
    # Generate monitoring report
    generate_monitoring_report(
        performance_results,
        drift_results,
        args.output_dir
    )
    
    print("Model monitoring completed successfully!")

if __name__ == "__main__":
    main()
