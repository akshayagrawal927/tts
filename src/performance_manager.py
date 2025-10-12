import time
import logger
from collections import defaultdict
from typing import Dict, List
import streamlit as st


class PerformanceTimer:
    """Track and display performance metrics for Pulse AI"""
    
    def __init__(self):
        self.timings = defaultdict(list)
        self.current_timing = {}
        
    def start(self, metric_name: str):
        """Start timing a metric"""
        self.current_timing[metric_name] = time.time()
        logger.debug(f"Timer started: {metric_name}")
        
    def stop(self, metric_name: str) -> float:
        """Stop timing and return duration"""
        if metric_name not in self.current_timing:
            logger.warning(f"Timer '{metric_name}' was never started")
            return 0.0
            
        duration = time.time() - self.current_timing[metric_name]
        self.timings[metric_name].append(duration)
        del self.current_timing[metric_name]
        logger.debug(f"Timer stopped: {metric_name} - {duration:.3f}s")
        return duration
    
    def get_last_timing(self, metric_name: str) -> float:
        """Get the last recorded timing for a metric"""
        if metric_name in self.timings and self.timings[metric_name]:
            return self.timings[metric_name][-1]
        return 0.0
    
    def get_average(self, metric_name: str) -> float:
        """Get average timing for a metric"""
        if metric_name in self.timings and self.timings[metric_name]:
            return sum(self.timings[metric_name]) / len(self.timings[metric_name])
        return 0.0
    
    def reset(self):
        """Reset all timings"""
        self.timings.clear()
        self.current_timing.clear()