import tensorflow as tf
import time
import logging
import threading
import psutil
import os
import sys
import io
from tqdm import tqdm
from tqdm.utils import _term_move_up

logger = logging.getLogger("fraud_detection")


class TqdmToConsole(io.StringIO):
    """
    Custom output stream for tqdm which will output to console without
    being captured by loggers or other output processors.
    """
    def __init__(self):
        super().__init__()
        self.console = sys.stdout
        self.last_len = 0
        # Add a lock to prevent concurrent writes
        self._lock = threading.Lock()
        self.last_refresh_time = 0
        # Minimum time between refreshes (in seconds) to prevent flicker
        self.min_refresh_interval = 10

    def write(self, data):
        # Use a lock to prevent multiple threads from writing simultaneously
        with self._lock:
            # Throttle updates to avoid flickering
            current_time = time.time()
            if '\r' in data:
                # Only update if enough time has passed since last update
                if current_time - self.last_refresh_time > self.min_refresh_interval:
                    # Clear the previous line completely to avoid artifacts
                    clear_line = '\r' + ' ' * self.last_len + '\r'
                    self.console.write(clear_line)
                    self.last_len = len(data.rstrip())
                    self.console.write(data)
                    self.console.flush()
                    self.last_refresh_time = current_time
            else:
                # Always write non-progress bar updates
                self.console.write(data)
                self.console.flush()
            
    def flush(self):
        self.console.flush()


def create_progress_bar(total=100, desc="Progress", bar_format=None, postfix=None, console=None):
    """
    Create a tqdm progress bar with consistent settings that won't interfere with logging.
    
    Args:
        total: Total number of steps (default: 100)
        desc: Description text for the progress bar
        bar_format: Custom bar format string
        postfix: Dictionary of values to display at the end of the bar
        console: Custom output stream (if None, creates a new TqdmToConsole)
        
    Returns:
        A configured tqdm progress bar instance
    """
    if console is None:
        console = TqdmToConsole()
        
    if bar_format is None:
        bar_format = '{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}] {postfix}'
        
    if postfix is None:
        postfix = {}
        
    return tqdm(
        total=total,
        desc=desc,
        bar_format=bar_format,
        postfix=postfix,
        file=console,
        mininterval=0.5,  # Minimum interval between updates to reduce flickering
        leave=True,       # Keep the progress bar after completion
        dynamic_ncols=True  # Adjust to terminal width
    )


def safe_tqdm_write(message, file=None):
    """
    Safely write a message to console without disrupting tqdm progress bars.
    Uses tqdm's write function to properly position the cursor.
    
    Args:
        message: The message to write
        file: Output file (defaults to sys.stdout)
    """
    tqdm.write(message, file=file)

class GPUTrainingMonitor:
    """
    Class to monitor GPU utilization and memory during training.
    
    This class tracks GPU availability, utilization, and memory usage during model training.
    It also records batch processing times and loss values for performance monitoring.
    """
    
    def __init__(self, interval=10.0):
        """Initialize the GPU training monitor.
        
        Args:
            interval (float): Interval in seconds for checking resource usage
        """
        self.interval = interval
        self._stop_event = threading.Event()
        self._monitor_thread = None
        self.start_time = None
        self._batch_times = []
        self._loss_values = []
        self.progress_bar = None
        
        # Metrics for tracking
        self._gpu_available = False
        self._gpu_memory_mb = 0
        self._cpu_percent = 0
        self._memory_mb = 0

    def start(self):
        """Start the monitoring thread."""
        if self._monitor_thread is not None:
            logger.warning("Monitoring already in progress")
            return
            
        self.start_time = time.time()
        self._stop_event.clear()
        
        # Use custom console output to avoid logging interference
        self.tqdm_out = TqdmToConsole()
        self.progress_bar = create_progress_bar(
            total=100,
            desc="Training",
            postfix={"CPU": "0%", "Memory": "0MB"},
            console=self.tqdm_out
        )
        
        self._monitor_thread = threading.Thread(target=self._monitoring_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        
        # Starting message only to log once
        logger.info(f"Starting GPU/CPU monitoring (interval: {self.interval}s)")

    def stop(self):
        """Stop the monitoring thread and log summary statistics."""
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            logger.warning("Monitoring not started or already stopped")
            return
        
        self._stop_event.set()
        self._monitor_thread.join(timeout=5.0)
        
        # Close the progress bar if it exists
        if self.progress_bar:
            self.progress_bar.close()
            self.progress_bar = None
            
        duration = time.time() - self.start_time
        logger.info(f"Monitoring stopped after {duration:.2f}s")
        self._log_summary()

    def _monitoring_loop(self):
        """Main monitoring loop that checks resource usage periodically."""
        try:
            logger.info(f"Starting GPU/CPU monitoring (interval: {self.interval}s)")
            while not self._stop_event.is_set():
                # Only update progress bar, never use logger for frequent updates
                duration = time.time() - self.start_time
                
                # Update progress bar with resource information
                cpu_info = f"CPU: {self._cpu_percent:.1f}%" if hasattr(self, '_cpu_percent') else ''
                mem_info = f"Memory: {self._memory_mb:.1f}MB" if hasattr(self, '_memory_mb') else ''
                
                # Update progress bar description with timing and resource info ONLY
                # Do NOT update the progress bar position from here to avoid duplication
                if self.progress_bar:
                    self.progress_bar.set_description(f"Training [{duration:.1f}s] {cpu_info} {mem_info}")
                
                # Track resources internally without logging
                self._update_system_metrics()
                
                # Wait for the next interval or until stopped
                self._stop_event.wait(timeout=self.interval)
        except Exception as e:
            logger.error(f"Error in monitoring thread: {str(e)}")
    
    def _update_system_metrics(self):
        """Update system resource metrics internally without logging."""
        try:
            process = psutil.Process(os.getpid())
            self._cpu_percent = process.cpu_percent()
            memory_info = process.memory_info()
            self._memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
            # No logging here - we're just tracking metrics internally
        except:
            pass
    
    def _print_progress(self, message):
        """Print a progress message, overwriting the current line.
        
        This bypasses the logger completely to prevent duplicate entries."""
        # Use our custom progress bar for dynamic updates
        if hasattr(self, 'progress_bar') and self.progress_bar:
            self.progress_bar.set_description_str(message)
            self.progress_bar.refresh()
        else:
            # Fall back to direct console output with carriage return
            sys.stdout.write('\r' + message)
            sys.stdout.flush()
    
    def record_batch(self, batch_time, loss=None):
        """
        Record the processing time and loss for a batch.
        
        Args:
            batch_time (float): Processing time for the batch in seconds
            loss (float): Loss value for the batch, if available
        """
        self._batch_times.append(batch_time)
        
        if loss is not None:
            self._loss_values.append(loss)
    
    def _log_summary(self):
        """Log a summary of the collected metrics."""
        if not self._batch_times:
            logger.info("No batch metrics collected")
            return
            
        # Calculate batch statistics
        avg_batch_time = sum(self._batch_times) / len(self._batch_times)
        max_batch_time = max(self._batch_times)
        min_batch_time = min(self._batch_times)
        
        logger.info(f"Batch processing times - Avg: {avg_batch_time:.4f}s, "
                    f"Min: {min_batch_time:.4f}s, Max: {max_batch_time:.4f}s, "
                    f"Batches: {len(self._batch_times)}")
        
        # Loss information if available
        if self._loss_values:
            initial_loss = self._loss_values[0]
            final_loss = self._loss_values[-1]
            avg_loss = sum(self._loss_values) / len(self._loss_values)
            
            logger.info(f"Loss - Initial: {initial_loss:.6f}, Final: {final_loss:.6f}, "
                        f"Avg: {avg_loss:.6f}, Change: {final_loss-initial_loss:.6f}")


class BatchTimingCallback(tf.keras.callbacks.Callback):
    """
    Keras callback to monitor batch and epoch times with GPU utilization.
    """
    
    def __init__(self):
        super().__init__()
        self.monitor = GPUTrainingMonitor(interval=10.0)
        self.batch_start_time = None
        self.epoch_start_time = None
    
    def on_train_begin(self, logs=None):
        # This log entry is kept since it's not a progress update
        logger.info("Starting training with GPU monitoring")
        self.monitor.start()
    
    def on_train_end(self, logs=None):
        # Ensure we have a newline before stopping the monitor
        sys.stdout.write('\n')
        sys.stdout.flush()
        self.monitor.stop()
        logger.info("Training complete")
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        
        # Reset and create a new progress bar for this epoch
        if self.monitor.progress_bar:
            self.monitor.progress_bar.close()
        
        # Initialize progress bar for this epoch using custom output stream
        if not hasattr(self.monitor, 'tqdm_out'):
            self.monitor.tqdm_out = TqdmToConsole()
        
        self.monitor.progress_bar = create_progress_bar(
            total=100,  # Will be updated based on actual batch count
            desc=f"Epoch {epoch+1}",
            postfix={"loss": "?", "accuracy": "?"},
            console=self.monitor.tqdm_out
        )
    
    def on_epoch_end(self, epoch, logs=None):
        if self.epoch_start_time:
            duration = time.time() - self.epoch_start_time
            
            if logs and self.monitor.progress_bar:
                # Update the progress bar to 100% and set final metrics
                self.monitor.progress_bar.n = 100
                
                # Format metrics for display
                metrics = {}
                for k, v in logs.items():
                    if isinstance(v, float):
                        metrics[k] = f"{v:.4f}"
                    else:
                        metrics[k] = str(v)
                        
                # Update postfix with final metrics
                self.monitor.progress_bar.set_postfix(metrics)
                self.monitor.progress_bar.refresh()
                
                # Close the progress bar to finalize the epoch
                self.monitor.progress_bar.close()
                
                # Print a summary line for the epoch with most important metrics
                loss = logs.get('loss')
                val_loss = logs.get('val_loss')
                acc = logs.get('accuracy') or logs.get('acc')
                val_acc = logs.get('val_accuracy') or logs.get('val_acc')
                
                # Format and print a summary line for the epoch with most important metrics
                summary = (f"Epoch {epoch+1} summary - {duration:.2f}s - "
                          f"loss: {loss:.4f}" + 
                          (f" - val_loss: {val_loss:.4f}" if val_loss else "") +
                          (f" - acc: {acc:.4f}" if acc else "") +
                          (f" - val_acc: {val_acc:.4f}" if val_acc else ""))
                
                # Use safe_tqdm_write to avoid disrupting other progress bars
                safe_tqdm_write(summary)
        else:
            # If no logs or progress bar, just print a basic summary
            safe_tqdm_write(f"Epoch {epoch+1} completed in {duration:.2f}s")
            if self.monitor.progress_bar:
                self.monitor.progress_bar.close()
    
    def on_train_batch_begin(self, batch, logs=None):
        self.batch_start_time = time.time()
        # Store batch number for progress calculation
        self.current_batch = batch
    
    def on_train_batch_end(self, batch, logs=None):
        if self.batch_start_time and self.monitor.progress_bar:
            # Calculate batch time
            batch_time = time.time() - self.batch_start_time
            self.monitor.record_batch(batch_time, logs.get('loss', None))
            
            # Update progress bar with batch information
            if hasattr(self, 'params') and 'steps' in self.params and self.params['steps'] > 0:
                # Calculate progress percentage based on batch number and total steps
                progress = min(int((batch + 1) * 100 / self.params['steps']), 100)
                
                # Only update progress bar if it would actually change the position
                # to avoid unnecessary display updates
                if progress > self.monitor.progress_bar.n:
                    # Force update to a specific position rather than incremental
                    self.monitor.progress_bar.n = progress
                    
                    # Update metrics in progress bar postfix
                    if logs:
                        postfix = {}
                        for k, v in logs.items():
                            if isinstance(v, float):
                                postfix[k] = f"{v:.4f}"
                            else:
                                postfix[k] = str(v)
                        self.monitor.progress_bar.set_postfix(postfix)
                    
                    # Use display instead of refresh to force a complete redraw
                    # This prevents the progress bar from being duplicated
                    self.monitor.progress_bar.display()