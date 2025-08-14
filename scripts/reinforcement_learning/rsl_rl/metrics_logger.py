"""
Metrics logger for RSL-RL training.
"""

import os
import csv
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
import threading

class MetricsLogger:
    """
    A logger class for capturing and saving training metrics to CSV files.
    """
    
    def __init__(self, save_dir: str = "metrics_logs"):
        """
        Initialize the metrics logger.
        
        Args:
            save_dir: Directory to save metrics files
        """
        self.save_dir = save_dir
        self.metrics_data: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
        
        # Create directory if it doesn't exist
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
        
        print(f"[INFO] MetricsLogger initialized. Save directory: {self.save_dir}")
    
    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Log a set of metrics for the current iteration.
        
        Args:
            metrics: Dictionary containing metric names and values
        """
        with self.lock:
            # Add timestamp if not present
            if 'timestamp' not in metrics:
                metrics['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Store metrics
            self.metrics_data.append(metrics.copy())
    
    def save_to_csv(self, filename: Optional[str] = None) -> str:
        """
        Save all logged metrics to a CSV file.
        
        Args:
            filename: Optional filename. If None, auto-generates based on timestamp.
            
        Returns:
            str: Path to the saved CSV file
        """
        if not self.metrics_data:
            print("[WARNING] No metrics data to save.")
            return ""
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"training_metrics_{timestamp}.csv"
        
        filepath = os.path.join(self.save_dir, filename)
        
        try:
            with self.lock:
                # Get all unique keys from all metrics entries
                all_keys = set()
                for metrics in self.metrics_data:
                    all_keys.update(metrics.keys())
                
                # Sort keys for consistent column order
                fieldnames = sorted(all_keys)
                
                # Write to CSV
                with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(self.metrics_data)
                
                print(f"[INFO] Metrics saved to: {filepath}")
                print(f"[INFO] Total metrics entries: {len(self.metrics_data)}")
                
                return filepath
                
        except Exception as e:
            print(f"[ERROR] Failed to save metrics to CSV: {e}")
            return ""
    
    def save_to_json(self, filename: Optional[str] = None) -> str:
        """
        Save all logged metrics to a JSON file.
        
        Args:
            filename: Optional filename. If None, auto-generates based on timestamp.
            
        Returns:
            str: Path to the saved JSON file
        """
        if not self.metrics_data:
            print("[WARNING] No metrics data to save.")
            return ""
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"training_metrics_{timestamp}.json"
        
        filepath = os.path.join(self.save_dir, filename)
        
        try:
            with self.lock:
                with open(filepath, 'w', encoding='utf-8') as jsonfile:
                    json.dump(self.metrics_data, jsonfile, indent=2)
                
                print(f"[INFO] Metrics saved to: {filepath}")
                return filepath
                
        except Exception as e:
            print(f"[ERROR] Failed to save metrics to JSON: {e}")
            return ""
    
    def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent metrics entry.
        
        Returns:
            Dict containing the latest metrics, or None if no data
        """
        with self.lock:
            return self.metrics_data[-1].copy() if self.metrics_data else None
    
    def get_all_metrics(self) -> List[Dict[str, Any]]:
        """
        Get all logged metrics.
        
        Returns:
            List of all metrics dictionaries
        """
        with self.lock:
            return [m.copy() for m in self.metrics_data]
    
    def clear_metrics(self) -> None:
        """Clear all stored metrics data."""
        with self.lock:
            self.metrics_data.clear()
            print("[INFO] All metrics data cleared.")