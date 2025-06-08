#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced sampling techniques for handling class imbalance in fraud detection.
This module provides implementations of various sampling methods to improve
model performance on imbalanced datasets, particularly for fraud detection.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from collections import Counter

def smote(X, y, sampling_strategy=0.5, k_neighbors=5, random_state=None):
    """
    Synthetic Minority Over-sampling Technique (SMOTE) implementation.
    
    SMOTE works by creating synthetic samples from the minority class instead of
    creating copies. For each minority class sample, it finds its k nearest neighbors,
    randomly selects one of them, and creates a synthetic sample along the line
    connecting the two points.
    
    Args:
        X (numpy.ndarray): Feature matrix of shape (n_samples, n_features)
        y (numpy.ndarray): Target vector of shape (n_samples,)
        sampling_strategy (float): Desired ratio of minority to majority class samples
                                  after resampling (default: 0.5)
        k_neighbors (int): Number of nearest neighbors to use for synthetic sample generation
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_resampled, y_resampled) - Resampled feature matrix and target vector
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Identify minority and majority classes
    class_counts = Counter(y)
    minority_class = min(class_counts, key=class_counts.get)
    majority_class = max(class_counts, key=class_counts.get)
    
    # Get indices of minority and majority classes
    minority_indices = np.where(y == minority_class)[0]
    majority_indices = np.where(y == majority_class)[0]
    
    # Extract minority class samples
    X_minority = X[minority_indices]
    
    # Calculate number of synthetic samples to generate
    n_minority = len(minority_indices)
    n_majority = len(majority_indices)
    n_synthetic = int(sampling_strategy * n_majority) - n_minority
    
    if n_synthetic <= 0:
        print("No synthetic samples needed based on sampling strategy.")
        return X, y
    
    # Fit nearest neighbors model to minority class samples
    nn = NearestNeighbors(n_neighbors=k_neighbors+1)  # +1 because the sample itself is included
    nn.fit(X_minority)
    
    # Generate synthetic samples
    synthetic_samples = []
    for i in range(n_minority):
        # Find k nearest neighbors for each minority sample
        neighbors = nn.kneighbors(X_minority[i].reshape(1, -1), return_distance=False)[0]
        neighbors = neighbors[1:]  # Exclude the sample itself
        
        # Generate synthetic samples
        n_samples_per_point = int(np.ceil(n_synthetic / n_minority))
        for _ in range(min(n_samples_per_point, len(synthetic_samples) + n_samples_per_point <= n_synthetic)):
            # Randomly select one of the k neighbors
            neighbor_idx = np.random.choice(neighbors)
            
            # Create synthetic sample
            diff = X_minority[neighbor_idx] - X_minority[i]
            gap = np.random.random()
            synthetic_sample = X_minority[i] + gap * diff
            synthetic_samples.append(synthetic_sample)
            
            if len(synthetic_samples) >= n_synthetic:
                break
    
    # Combine original and synthetic samples
    X_resampled = np.vstack([X, np.array(synthetic_samples)])
    y_resampled = np.hstack([y, np.array([minority_class] * len(synthetic_samples))])
    
    print(f"Generated {len(synthetic_samples)} synthetic samples for the minority class.")
    print(f"Class distribution after SMOTE: {Counter(y_resampled)}")
    
    return X_resampled, y_resampled

def adasyn(X, y, sampling_strategy=0.5, k_neighbors=5, beta=0.5, random_state=None):
    """
    Adaptive Synthetic Sampling Approach for Imbalanced Learning (ADASYN).
    
    ADASYN is similar to SMOTE but generates more synthetic data for minority class
    samples that are harder to learn (those close to the decision boundary with the majority class).
    
    Args:
        X (numpy.ndarray): Feature matrix of shape (n_samples, n_features)
        y (numpy.ndarray): Target vector of shape (n_samples,)
        sampling_strategy (float): Desired ratio of minority to majority class samples
                                  after resampling (default: 0.5)
        k_neighbors (int): Number of nearest neighbors to use
        beta (float): Specifies the desired balance level after generation of synthetic samples
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_resampled, y_resampled) - Resampled feature matrix and target vector
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Identify minority and majority classes
    class_counts = Counter(y)
    minority_class = min(class_counts, key=class_counts.get)
    majority_class = max(class_counts, key=class_counts.get)
    
    # Get indices of minority and majority classes
    minority_indices = np.where(y == minority_class)[0]
    majority_indices = np.where(y == majority_class)[0]
    
    # Extract minority class samples
    X_minority = X[minority_indices]
    
    # Calculate number of synthetic samples to generate
    n_minority = len(minority_indices)
    n_majority = len(majority_indices)
    n_synthetic = int(beta * (n_majority - n_minority))
    
    if n_synthetic <= 0:
        print("No synthetic samples needed based on sampling strategy.")
        return X, y
    
    # Fit nearest neighbors model to all samples
    nn = NearestNeighbors(n_neighbors=k_neighbors+1)
    nn.fit(X)
    
    # Calculate density ratio for each minority sample
    density_ratios = []
    for i, idx in enumerate(minority_indices):
        neighbors = nn.kneighbors(X[idx].reshape(1, -1), return_distance=False)[0]
        neighbors = neighbors[1:]  # Exclude the sample itself
        
        # Count majority class neighbors
        majority_neighbors = sum(1 for n in neighbors if y[n] == majority_class)
        
        # Calculate density ratio
        density_ratio = majority_neighbors / k_neighbors
        density_ratios.append(density_ratio)
    
    # Normalize density ratios
    if sum(density_ratios) > 0:
        density_ratios = np.array(density_ratios) / sum(density_ratios)
    else:
        # If all density ratios are 0, use uniform distribution
        density_ratios = np.ones(n_minority) / n_minority
    
    # Calculate number of synthetic samples to generate for each minority sample
    n_samples_per_point = np.round(density_ratios * n_synthetic).astype(int)
    
    # Fit nearest neighbors model to minority class samples only
    nn_minority = NearestNeighbors(n_neighbors=k_neighbors+1)
    nn_minority.fit(X_minority)
    
    # Generate synthetic samples
    synthetic_samples = []
    for i, n_samples in enumerate(n_samples_per_point):
        if n_samples <= 0:
            continue
            
        # Find k nearest neighbors for each minority sample (within minority class)
        neighbors = nn_minority.kneighbors(X_minority[i].reshape(1, -1), return_distance=False)[0]
        neighbors = neighbors[1:]  # Exclude the sample itself
        
        # Generate synthetic samples
        for _ in range(n_samples):
            # Randomly select one of the k neighbors
            neighbor_idx = np.random.choice(neighbors)
            
            # Create synthetic sample
            diff = X_minority[neighbor_idx] - X_minority[i]
            gap = np.random.random()
            synthetic_sample = X_minority[i] + gap * diff
            synthetic_samples.append(synthetic_sample)
    
    # Combine original and synthetic samples
    if synthetic_samples:
        X_resampled = np.vstack([X, np.array(synthetic_samples)])
        y_resampled = np.hstack([y, np.array([minority_class] * len(synthetic_samples))])
        
        print(f"Generated {len(synthetic_samples)} synthetic samples for the minority class.")
        print(f"Class distribution after ADASYN: {Counter(y_resampled)}")
    else:
        X_resampled, y_resampled = X, y
        print("No synthetic samples were generated.")
    
    return X_resampled, y_resampled

def borderline_smote(X, y, sampling_strategy=0.5, k_neighbors=5, m_neighbors=10, kind='borderline1', random_state=None):
    """
    Borderline SMOTE: Over-sampling method that generates synthetic samples only for minority
    instances near the borderline between classes.
    
    Args:
        X (numpy.ndarray): Feature matrix of shape (n_samples, n_features)
        y (numpy.ndarray): Target vector of shape (n_samples,)
        sampling_strategy (float): Desired ratio of minority to majority class samples
                                  after resampling (default: 0.5)
        k_neighbors (int): Number of nearest neighbors to use for synthetic sample generation
        m_neighbors (int): Number of nearest neighbors to use for determining if a sample is in danger
        kind (str): Type of Borderline SMOTE to use:
                   'borderline1' - only samples in danger
                   'borderline2' - samples in danger and samples in noise
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_resampled, y_resampled) - Resampled feature matrix and target vector
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Identify minority and majority classes
    class_counts = Counter(y)
    minority_class = min(class_counts, key=class_counts.get)
    majority_class = max(class_counts, key=class_counts.get)
    
    # Get indices of minority and majority classes
    minority_indices = np.where(y == minority_class)[0]
    majority_indices = np.where(y == majority_class)[0]
    
    # Extract minority class samples
    X_minority = X[minority_indices]
    
    # Calculate number of synthetic samples to generate
    n_minority = len(minority_indices)
    n_majority = len(majority_indices)
    n_synthetic = int(sampling_strategy * n_majority) - n_minority
    
    if n_synthetic <= 0:
        print("No synthetic samples needed based on sampling strategy.")
        return X, y
    
    # Fit nearest neighbors model to all samples
    nn = NearestNeighbors(n_neighbors=m_neighbors+1)
    nn.fit(X)
    
    # Identify minority samples in danger (near the decision boundary)
    danger_indices = []
    noise_indices = []
    safe_indices = []
    
    for i, idx in enumerate(minority_indices):
        neighbors = nn.kneighbors(X[idx].reshape(1, -1), return_distance=False)[0]
        neighbors = neighbors[1:]  # Exclude the sample itself
        
        # Count majority class neighbors
        majority_neighbors = sum(1 for n in neighbors if y[n] == majority_class)
        
        # Classify the sample based on its neighborhood
        if majority_neighbors == m_neighbors:
            # All neighbors are from majority class - noise
            noise_indices.append(i)
        elif majority_neighbors > m_neighbors // 2:
            # More than half of neighbors are from majority class - danger
            danger_indices.append(i)
        else:
            # Less than half of neighbors are from majority class - safe
            safe_indices.append(i)
    
    # Select samples to oversample based on kind parameter
    if kind == 'borderline1':
        selected_indices = danger_indices
    elif kind == 'borderline2':
        selected_indices = danger_indices + noise_indices
    else:
        raise ValueError("kind must be either 'borderline1' or 'borderline2'")
    
    if not selected_indices:
        print("No samples in danger or noise found. Using all minority samples.")
        selected_indices = list(range(n_minority))
    
    # Fit nearest neighbors model to minority class samples
    nn_minority = NearestNeighbors(n_neighbors=k_neighbors+1)
    nn_minority.fit(X_minority)
    
    # Generate synthetic samples
    synthetic_samples = []
    n_samples_per_point = int(np.ceil(n_synthetic / len(selected_indices)))
    
    for i in selected_indices:
        # Find k nearest neighbors for each selected minority sample
        neighbors = nn_minority.kneighbors(X_minority[i].reshape(1, -1), return_distance=False)[0]
        neighbors = neighbors[1:]  # Exclude the sample itself
        
        # Generate synthetic samples
        for _ in range(min(n_samples_per_point, len(synthetic_samples) + n_samples_per_point <= n_synthetic)):
            # Randomly select one of the k neighbors
            neighbor_idx = np.random.choice(neighbors)
            
            # Create synthetic sample
            diff = X_minority[neighbor_idx] - X_minority[i]
            gap = np.random.random()
            synthetic_sample = X_minority[i] + gap * diff
            synthetic_samples.append(synthetic_sample)
            
            if len(synthetic_samples) >= n_synthetic:
                break
    
    # Combine original and synthetic samples
    if synthetic_samples:
        X_resampled = np.vstack([X, np.array(synthetic_samples)])
        y_resampled = np.hstack([y, np.array([minority_class] * len(synthetic_samples))])
        
        print(f"Generated {len(synthetic_samples)} synthetic samples for the minority class.")
        print(f"Class distribution after Borderline SMOTE: {Counter(y_resampled)}")
    else:
        X_resampled, y_resampled = X, y
        print("No synthetic samples were generated.")
    
    return X_resampled, y_resampled

def cluster_smote(X, y, n_clusters=3, sampling_strategy=0.5, k_neighbors=5, random_state=None):
    """
    Cluster-based SMOTE implementation that preserves minority class distribution.
    
    This technique first clusters the minority class samples before applying SMOTE,
    which helps preserve the distribution of minority class samples better than
    standard approaches. This is particularly valuable for fraud detection where
    different types of fraud may form distinct clusters.
    
    Args:
        X (numpy.ndarray): Feature matrix of shape (n_samples, n_features)
        y (numpy.ndarray): Target vector of shape (n_samples,)
        n_clusters (int): Number of clusters to create in the minority class
        sampling_strategy (float): Desired ratio of minority to majority class samples
                                  after resampling (default: 0.5)
        k_neighbors (int): Number of nearest neighbors to use for synthetic sample generation
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_resampled, y_resampled) - Resampled feature matrix and target vector
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        raise ImportError("sklearn.cluster is required for cluster_smote. Please install scikit-learn.")
    
    if random_state is not None:
        np.random.seed(random_state)
    
    # Identify minority and majority classes
    class_counts = Counter(y)
    minority_class = min(class_counts, key=class_counts.get)
    majority_class = max(class_counts, key=class_counts.get)
    
    # Get indices of minority and majority classes
    minority_indices = np.where(y == minority_class)[0]
    majority_indices = np.where(y == majority_class)[0]
    
    # Extract minority class samples
    X_minority = X[minority_indices]
    
    # Calculate number of synthetic samples to generate
    n_minority = len(minority_indices)
    n_majority = len(majority_indices)
    n_synthetic = int(sampling_strategy * n_majority) - n_minority
    
    if n_synthetic <= 0:
        print("No synthetic samples needed based on sampling strategy.")
        return X, y
    
    # Apply k-means clustering to minority samples
    actual_n_clusters = min(n_clusters, n_minority)
    if actual_n_clusters < n_clusters:
        print(f"Warning: Reduced number of clusters to {actual_n_clusters} because there are only {n_minority} minority samples.")
    
    kmeans = KMeans(n_clusters=actual_n_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(X_minority)
    
    # Generate synthetic samples for each cluster
    synthetic_samples = []
    samples_per_cluster = {i: n_synthetic // actual_n_clusters for i in range(actual_n_clusters)}
    
    # Distribute any remaining samples
    remainder = n_synthetic % actual_n_clusters
    for i in range(remainder):
        samples_per_cluster[i] += 1
    
    # Generate samples for each cluster
    for cluster_idx in range(actual_n_clusters):
        cluster_samples = X_minority[clusters == cluster_idx]
        
        if len(cluster_samples) <= 1:
            print(f"Skipping cluster {cluster_idx} with only {len(cluster_samples)} samples.")
            continue
        
        # Apply SMOTE within the cluster
        nn = NearestNeighbors(n_neighbors=min(k_neighbors+1, len(cluster_samples)))
        nn.fit(cluster_samples)
        
        for i in range(min(len(cluster_samples), samples_per_cluster[cluster_idx])):
            sample_idx = i % len(cluster_samples)
            neighbors = nn.kneighbors(cluster_samples[sample_idx].reshape(1, -1), 
                                     return_distance=False)[0]
            neighbors = neighbors[1:]  # Exclude the sample itself
            
            if len(neighbors) == 0:
                continue
            
            # Randomly select one of the k neighbors
            neighbor_idx = np.random.choice(neighbors)
            
            # Create synthetic sample
            diff = cluster_samples[neighbor_idx] - cluster_samples[sample_idx]
            gap = np.random.random()
            synthetic_sample = cluster_samples[sample_idx] + gap * diff
            synthetic_samples.append(synthetic_sample)
    
    # Combine original and synthetic samples
    if synthetic_samples:
        X_resampled = np.vstack([X, np.array(synthetic_samples)])
        y_resampled = np.hstack([y, np.array([minority_class] * len(synthetic_samples))])
        
        print(f"Generated {len(synthetic_samples)} synthetic samples for the minority class.")
        print(f"Class distribution after Cluster-SMOTE: {Counter(y_resampled)}")
    else:
        X_resampled, y_resampled = X, y
        print("No synthetic samples were generated.")
    
    return X_resampled, y_resampled

def get_sampling_technique(technique_name):
    """
    Get the sampling technique function by name.
    
    Args:
        technique_name (str): Name of the sampling technique
                             ('smote', 'adasyn', 'borderline_smote', 'cluster_smote')
        
    Returns:
        function: The corresponding sampling function
    """
    techniques = {
        'smote': smote,
        'adasyn': adasyn,
        'borderline_smote': borderline_smote,
        'cluster_smote': cluster_smote
    }
    
    if technique_name.lower() not in techniques:
        raise ValueError(f"Unknown sampling technique: {technique_name}. "
                         f"Available techniques: {list(techniques.keys())}")
    
    return techniques[technique_name.lower()]
