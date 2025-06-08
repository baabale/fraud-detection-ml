import tensorflow as tf
import time
import os
import logging
from contextlib import contextmanager

logger = logging.getLogger("fraud_detection")

def enable_gpu_debug_logging(verbose=False):
    """
    Enable detailed TensorFlow logging for GPU operations.

    Args:
        verbose: If True, sets TF CPP logging to level 1 (INFO)
    """
    if verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 0=all logs, 1=INFO, 2=WARNING, 3=ERROR
        tf.debugging.set_log_device_placement(True)
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only show WARNING and above
        tf.debugging.set_log_device_placement(False)

    # Check if GPU is available and configure for debugging
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"Found {len(gpus)} GPU(s)")
        # Enable memory growth to avoid allocating all GPU memory at once
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("Memory growth enabled on all GPUs")
        except RuntimeError as e:
            logger.error(f"Error configuring GPU: {e}")
    else:
        logger.warning("No GPUs found - running on CPU only")

    return gpus

def log_gpu_memory_usage():
    """
    Log current GPU memory usage.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        logger.info("No GPUs available to check memory usage")
        return

    try:
        for i, gpu in enumerate(gpus):
            memory_info = tf.config.experimental.get_memory_info(f'/device:GPU:{i}')
            logger.info(f"GPU:{i} Memory - Current: {memory_info['current']/1e6:.2f}MB, Peak: {memory_info['peak']/1e6:.2f}MB")
    except:
        logger.warning("Could not get detailed GPU memory info - this might not be supported on Metal backend")
        # For Metal plugin, we can only log that we tried to check memory
        logger.info("GPU memory usage check attempted (detailed stats not available for Metal)")

@contextmanager
def gpu_operation_logger(operation_name):
    """
    Context manager to time and log GPU operations.

    Args:
        operation_name: Name of the operation being performed
    """
    logger.info(f"Starting {operation_name}")

    # Create a simple test tensor to check if GPU is active
    with tf.device('/device:GPU:0'):
        try:
            test_tensor = tf.random.normal([1000, 1000])
            _ = test_tensor @ test_tensor  # Matrix multiplication
            device_status = "GPU active"
        except (RuntimeError, tf.errors.InvalidArgumentError):
            device_status = "GPU not active - using CPU"

    logger.info(f"Device status: {device_status}")
    start_time = time.time()

    try:
        yield  # Execute the operation
    finally:
        duration = time.time() - start_time
        logger.info(f"Completed {operation_name} in {duration:.2f}s")
        log_gpu_memory_usage()

def analyze_model_device_placement(model):
    """
    Analyze which layers of a Keras model are placed on which device.

    Args:
        model: A tf.keras.Model instance
    """
    logger.info("Analyzing model device placement:")
    cpu_ops = []
    gpu_ops = []
    unknown_ops = []

    # First check if eager execution is enabled
    if not tf.executing_eagerly():
        logger.warning("Not running in eager mode - device placement analysis may be limited")

    # Try to analyze layer by layer
    for i, layer in enumerate(model.layers):
        try:
            # Create a simple input for this layer
            if i == 0:  # First layer
                dummy_input = tf.zeros([1] + list(model.input_shape[1:]))
                with tf.device('/device:CPU:0'):
                    cpu_output = layer(dummy_input)
                with tf.device('/device:GPU:0'):
                    try:
                        gpu_output = layer(dummy_input)
                        if tf.reduce_all(cpu_output == gpu_output):
                            msg = f"Layer {i}: {layer.name} - Compatible with both CPU and GPU"
                            gpu_ops.append(layer.name)
                        else:
                            msg = f"Layer {i}: {layer.name} - WARNING: Different results on CPU vs GPU"
                            unknown_ops.append(layer.name)
                    except (tf.errors.InvalidArgumentError, RuntimeError) as e:
                        msg = f"Layer {i}: {layer.name} - NOT compatible with GPU: {str(e)}"
                        cpu_ops.append(layer.name)
            else:
                msg = f"Layer {i}: {layer.name} - (Nested analysis not performed)"
                unknown_ops.append(layer.name)

            logger.info(msg)
        except Exception as e:
            logger.error(f"Error analyzing layer {layer.name}: {str(e)}")
            unknown_ops.append(layer.name)

    # Summary
    logger.info(f"Device Placement Summary - "
                f"GPU Compatible: {len(gpu_ops)}, "
                f"CPU Only: {len(cpu_ops)}, "
                f"Unknown: {len(unknown_ops)}")

    return {
        "gpu_compatible": gpu_ops,
        "cpu_only": cpu_ops,
        "unknown": unknown_ops
    }

def profile_training_step(model, x, y, batch_size=32):
    """
    Profile a single training step to see which operations are slow.

    Args:
        model: A compiled tf.keras.Model
        x: Input data
        y: Target data
        batch_size: Batch size to use
    """
    # Handle TensorFlow dataset inputs (tuples)
    if isinstance(x, tuple) and len(x) == 2:
        # This is likely a dataset batch with (features, labels)
        x = x[0]  # Take only the features
    
    if isinstance(y, tuple) and len(y) == 2:
        y = y[0]  # Take only the features for autoencoder
        
    # Limit batch size if needed
    if hasattr(x, '__len__') and len(x) > batch_size:
        x = x[:batch_size]
        y = y[:batch_size]

    logger.info("Profiling training step...")
    
    # Time the forward pass
    with gpu_operation_logger("Forward pass"):
        _ = model(x, training=False)

    # Time the training step
    with gpu_operation_logger("Training step"):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            if hasattr(model, 'compiled_loss'):
                loss_value = model.compiled_loss(y, logits)
            else:
                if callable(model.loss):
                    loss_value = model.loss(y, logits)
                else:
                    loss_fn = getattr(tf.keras.losses, model.loss) if isinstance(model.loss, str) else model.loss
                    loss_value = loss_fn(y, logits)

        grads = tape.gradient(loss_value, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

    logger.info("Training step profile complete")
    return loss_value.numpy()

def print_gpu_operation_support_info():
    """
    Print information about which operations are supported on GPU.
    """
    logger.info("Checking GPU operation support:")

    # Test common operations
    operations = [
        ("Basic math (add/multiply)", lambda: tf.add(tf.ones([1000, 1000]), tf.ones([1000, 1000]))),
        ("Matrix multiplication", lambda: tf.matmul(tf.ones([1000, 1000]), tf.ones([1000, 1000]))),
        ("Convolution 2D", lambda: tf.nn.conv2d(tf.ones([1, 32, 32, 3]), tf.ones([3, 3, 3, 16]), strides=1, padding='SAME')),
        ("Pooling", lambda: tf.nn.max_pool2d(tf.ones([1, 32, 32, 3]), ksize=2, strides=2, padding='SAME')),
        ("ReLU activation", lambda: tf.nn.relu(tf.random.normal([1000, 1000]))),
        ("Softmax", lambda: tf.nn.softmax(tf.random.normal([1000, 10]))),
        ("Batch normalization", lambda: tf.keras.layers.BatchNormalization()(tf.ones([32, 10, 10, 64]), training=True))
    ]

    for name, op_fn in operations:
        try:
            with tf.device('/device:GPU:0'):
                start = time.time()
                result = op_fn()
                gpu_time = time.time() - start

            with tf.device('/device:CPU:0'):
                start = time.time()
                _ = op_fn()
                cpu_time = time.time() - start

            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            logger.info(f"{name}: SUCCESS on GPU (speedup: {speedup:.2f}x)")
        except Exception as e:
            logger.warning(f"{name}: FAILED on GPU - {str(e)}")