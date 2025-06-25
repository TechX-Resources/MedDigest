"""
Token Monitoring Utility

A comprehensive utility for tracking token usage across multiple LLM API calls.
Can be used independently or integrated into existing code.

Features:
- Token counting per call
- Rate limiting (tokens per minute)
- Cost calculation for different models
- Usage statistics and reporting
- Thread-safe operations
- Export capabilities
"""

import time
import logging
import json
import threading
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """Tracks a single API call's token and cost usage."""
    timestamp: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float


class TokenMonitor:
    """Tracks and limits LLM token usage per minute."""
    INPUT_COST_PER_MILLION = 0.05   
    OUTPUT_COST_PER_MILLION = 0.08  

    def __init__(self, max_tokens_per_minute: int = 16000):
        self.max_tokens_per_minute = max_tokens_per_minute
        self.usage_history: List[TokenUsage] = []
        self.lock = threading.Lock()
        self.tokens_this_minute = 0
        self.minute_start_time = time.time()
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0

    def record_usage(self, input_tokens: int, output_tokens: int) -> TokenUsage:
        #Record a call's token usage and enforce per-minute limit.
        with self.lock:
            total_tokens = input_tokens + output_tokens
            current_time = time.time()
            if current_time - self.minute_start_time >= 60:
                self.tokens_this_minute = 0
                self.minute_start_time = current_time
            if self.tokens_this_minute + total_tokens > self.max_tokens_per_minute:
                remaining_time = 60 - (current_time - self.minute_start_time)
                logger.info(f"Token limit reached. Sleeping for {remaining_time:.1f} seconds...")
                time.sleep(remaining_time)
                self.tokens_this_minute = 0
                self.minute_start_time = time.time()
            cost = self._calculate_cost(input_tokens, output_tokens)
            usage = TokenUsage(
                timestamp=time.time(),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cost_usd=cost
            )
            self.usage_history.append(usage)
            self.tokens_this_minute += total_tokens
            self.total_calls += 1
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cost += cost
            logger.info(
                f"Token usage: {input_tokens} input + {output_tokens} output = "
                f"{total_tokens} total tokens (${cost:.4f} USD)"
            )
            return usage

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        # Calculate cost for this API call
        input_cost = (input_tokens / 1_000_000) * self.INPUT_COST_PER_MILLION
        output_cost = (output_tokens / 1_000_000) * self.OUTPUT_COST_PER_MILLION
        return input_cost + output_cost

    def print_usage_summary(self):
        # Print a summary of usage statistics
        logger.info("=" * 60)
        logger.info("TOKEN USAGE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total calls: {self.total_calls}")
        logger.info(f"Total tokens: {self.total_input_tokens + self.total_output_tokens:,}")
        logger.info(f"  - Input: {self.total_input_tokens:,}")
        logger.info(f"  - Output: {self.total_output_tokens:,}")
        logger.info(f"Total cost: ${self.total_cost:.4f}")
        logger.info("=" * 60)


# Convenience function for quick usage tracking
def track_llm_call(monitor: TokenMonitor, input_tokens: int, output_tokens: int) -> TokenUsage:
    """
    Convenience function to track an LLM call.
    
    Args:
        monitor (TokenMonitor): The token monitor instance
        input_tokens (int): Number of input tokens
        output_tokens (int): Number of output tokens
        
    Returns:
        TokenUsage: Recorded usage information
    """
    return monitor.record_usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens
    ) 