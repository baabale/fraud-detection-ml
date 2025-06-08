#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Custom loss functions for fraud detection models.
These loss functions are designed to address class imbalance and focus on
hard-to-classify examples, particularly for fraud detection tasks.
"""

import tensorflow as tf
import numpy as np

def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss implementation for binary classification.
    
    Focal loss addresses class imbalance by down-weighting easy examples and focusing
    on hard negative examples. It's particularly useful for fraud detection where
    the positive class (fraud) is the minority class.
    
    The focal loss is defined as:
    FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
    
    where p_t is the model's estimated probability for the target class.
    
    Args:
        gamma (float): Focusing parameter that adjusts the rate at which easy examples
                      are down-weighted. Higher gamma values focus more on hard examples.
        alpha (float): Weighting factor for the positive class. It's used to address class
                      imbalance by giving more weight to the minority class.
                      
    Returns:
        function: A loss function that takes y_true and y_pred as inputs
    """
    def loss_function(y_true, y_pred):
        # Clip predictions to avoid numerical instability
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate cross entropy
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        # Calculate focal term
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        focal_term = tf.pow(1 - p_t, gamma)
        
        # Calculate alpha term
        alpha_factor = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        
        # Combine all terms
        loss = alpha_factor * focal_term * cross_entropy
        
        # Return mean loss across samples
        return tf.reduce_mean(loss)
    
    return loss_function

def weighted_focal_loss(gamma=2.0, class_weights=None):
    """
    Weighted Focal Loss implementation for binary classification.
    
    This is a variant of focal loss that allows for more flexible class weighting.
    
    Args:
        gamma (float): Focusing parameter that adjusts the rate at which easy examples
                      are down-weighted. Higher gamma values focus more on hard examples.
        class_weights (dict): Dictionary mapping class indices to weights.
                             If None, all classes are weighted equally.
                      
    Returns:
        function: A loss function that takes y_true and y_pred as inputs
    """
    def loss_function(y_true, y_pred):
        # Clip predictions to avoid numerical instability
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate cross entropy
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        
        # Calculate focal term
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        focal_term = tf.pow(1 - p_t, gamma)
        
        # Apply class weights if provided
        if class_weights is not None:
            weight_map = tf.constant([class_weights[0], class_weights[1]], dtype=tf.float32)
            weights = tf.gather(weight_map, tf.cast(y_true, tf.int32))
            focal_term = focal_term * weights
        
        # Combine terms
        loss = focal_term * cross_entropy
        
        # Return mean loss across samples
        return tf.reduce_mean(loss)
    
    return loss_function

def asymmetric_focal_loss(gamma_pos=2.0, gamma_neg=4.0, clip=0.05):
    """
    Asymmetric Focal Loss for binary classification.
    
    This variant of focal loss uses different focusing parameters for positive and
    negative examples, which can be beneficial for highly imbalanced datasets.
    
    Args:
        gamma_pos (float): Focusing parameter for positive examples (fraud)
        gamma_neg (float): Focusing parameter for negative examples (non-fraud)
        clip (float): Clipping value to prevent numerical instability
                      
    Returns:
        function: A loss function that takes y_true and y_pred as inputs
    """
    def loss_function(y_true, y_pred):
        # Clip predictions to avoid numerical instability
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate cross entropy for positive and negative examples
        ce_pos = -y_true * tf.math.log(y_pred)
        ce_neg = -(1 - y_true) * tf.math.log(1 - y_pred)
        
        # Calculate focal terms with different gamma values
        focal_pos = tf.pow(1 - y_pred, gamma_pos)
        focal_neg = tf.pow(y_pred, gamma_neg)
        
        # Combine terms
        loss = focal_pos * ce_pos + focal_neg * ce_neg
        
        # Return mean loss across samples
        return tf.reduce_mean(loss)
    
    return loss_function

def adaptive_focal_loss(gamma=2.0, alpha=0.25, threshold=0.5):
    """
    Adaptive Focal Loss that adjusts focusing parameter based on training progress.
    
    This loss function starts with a lower gamma value and gradually increases it
    during training, allowing the model to focus more on hard examples as training progresses.
    
    Args:
        gamma (float): Initial focusing parameter
        alpha (float): Weighting factor for the positive class
        threshold (float): Classification threshold
                      
    Returns:
        function: A loss function that takes y_true, y_pred, and epoch as inputs
    """
    def loss_function(y_true, y_pred, epoch=0):
        # Adjust gamma based on epoch (increase focus on hard examples as training progresses)
        adaptive_gamma = gamma * (1 + 0.1 * tf.cast(epoch, tf.float32))
        
        # Clip predictions to avoid numerical instability
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate binary cross entropy
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        
        # Calculate focal term
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        focal_term = tf.pow(1 - p_t, adaptive_gamma)
        
        # Calculate alpha term
        alpha_factor = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        
        # Combine all terms
        loss = alpha_factor * focal_term * bce
        
        # Return mean loss across samples
        return tf.reduce_mean(loss)
    
    return loss_function

def get_loss_function(loss_name, **kwargs):
    """
    Get the loss function by name.
    
    Args:
        loss_name (str): Name of the loss function
                        ('focal', 'weighted_focal', 'asymmetric_focal', 'adaptive_focal')
        **kwargs: Additional arguments to pass to the loss function
        
    Returns:
        function: The corresponding loss function
    """
    loss_functions = {
        'focal': focal_loss,
        'weighted_focal': weighted_focal_loss,
        'asymmetric_focal': asymmetric_focal_loss,
        'adaptive_focal': adaptive_focal_loss
    }
    
    if loss_name.lower() not in loss_functions:
        raise ValueError(f"Unknown loss function: {loss_name}. "
                         f"Available loss functions: {list(loss_functions.keys())}")
    
    return loss_functions[loss_name.lower()](**kwargs)
