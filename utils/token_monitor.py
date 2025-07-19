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
- Detailed prompt analysis
- Average tokens per paper tracking
- Token distribution analysis
"""

import time
import logging
import threading
from typing import Dict, List, Union
from dataclasses import dataclass
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """Tracks a single API call's token and cost usage."""
    timestamp: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    call_type: str = "unknown"  # e.g., "paper_analysis", "batch_analysis", "summary_generation"
    prompt_length: int = 0  # Length of the prompt in characters
    response_length: int = 0  # Length of the response in characters


@dataclass
class DetailedUsageStats:
    """Detailed usage statistics for comprehensive reporting."""
    total_calls: int
    total_tokens: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost: float
    average_tokens_per_call: float
    average_input_tokens: float
    average_output_tokens: float
    average_cost_per_call: float
    tokens_per_minute_rate: float
    calls_per_minute_rate: float
    call_type_breakdown: Dict[str, Dict[str, Union[int, float]]]
    prompt_length_stats: Dict[str, float]
    response_length_stats: Dict[str, float]
    processing_duration: float  # Total time from first to last call


class TokenMonitor:
    """Tracks and limits LLM token usage per minute with detailed analytics."""
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
        self.start_time = time.time()
        
        # Enhanced tracking for detailed analysis
        self.call_type_counts = defaultdict(int)
        self.call_type_tokens = defaultdict(list)
        self.prompt_lengths = []
        self.response_lengths = []
        self.token_ratios = []  # output/input ratio

    def record_usage(self, input_tokens: int, output_tokens: int, 
                    call_type: str = "unknown", prompt_length: int = 0, 
                    response_length: int = 0) -> TokenUsage:
        """
        Record a call's token usage and enforce per-minute limit.
        
        Args:
            input_tokens (int): Number of input tokens
            output_tokens (int): Number of output tokens
            call_type (str): Type of call (e.g., "paper_analysis", "batch_analysis")
            prompt_length (int): Length of prompt in characters
            response_length (int): Length of response in characters
            
        Returns:
            TokenUsage: Recorded usage information
        """
        with self.lock:
            total_tokens = input_tokens + output_tokens
            current_time = time.time()
            
            # Rate limiting logic
            if current_time - self.minute_start_time >= 60:
                self.tokens_this_minute = 0
                self.minute_start_time = current_time
                
            if self.tokens_this_minute + total_tokens > self.max_tokens_per_minute:
                remaining_time = 60 - (current_time - self.minute_start_time)
                logger.info(f"Token limit reached. Sleeping for {remaining_time:.1f} seconds...")
                time.sleep(remaining_time)
                self.tokens_this_minute = 0
                self.minute_start_time = time.time()
            
            # Calculate cost
            cost = self._calculate_cost(input_tokens, output_tokens)
            
            # Create usage record
            usage = TokenUsage(
                timestamp=current_time,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cost_usd=cost,
                call_type=call_type,
                prompt_length=prompt_length,
                response_length=response_length
            )
            
            # Update tracking data
            self.usage_history.append(usage)
            self.tokens_this_minute += total_tokens
            self.total_calls += 1
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cost += cost
            
            # Enhanced tracking
            self.call_type_counts[call_type] += 1
            self.call_type_tokens[call_type].append(total_tokens)
            self.prompt_lengths.append(prompt_length)
            self.response_lengths.append(response_length)
            if input_tokens > 0:
                self.token_ratios.append(output_tokens / input_tokens)
            
            logger.info(
                f"Token usage ({call_type}): {input_tokens} input + {output_tokens} output = "
                f"{total_tokens} total tokens (${cost:.4f} USD)"
            )
            return usage

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for this API call."""
        input_cost = (input_tokens / 1_000_000) * self.INPUT_COST_PER_MILLION
        output_cost = (output_tokens / 1_000_000) * self.OUTPUT_COST_PER_MILLION
        return input_cost + output_cost

    def count_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text string.
        
        Args:
            text (str): The text to count tokens for
            
        Returns:
            int: Estimated number of tokens (rough approximation: 1 token ≈ 4 characters)
        """
        if not text:
            return 0
        # Rough approximation: 1 token ≈ 4 characters
        return len(text) // 4

    def get_detailed_stats(self) -> DetailedUsageStats:
        """Calculate comprehensive usage statistics."""
        if not self.usage_history:
            return DetailedUsageStats(
                total_calls=0, total_tokens=0, total_input_tokens=0, total_output_tokens=0,
                total_cost=0.0, average_tokens_per_call=0.0, average_input_tokens=0.0,
                average_output_tokens=0.0, average_cost_per_call=0.0, tokens_per_minute_rate=0.0,
                call_type_breakdown={}, prompt_length_stats={}, response_length_stats={},
                processing_duration=0.0
            )
        
        # Basic stats
        total_tokens = self.total_input_tokens + self.total_output_tokens
        processing_duration = time.time() - self.start_time
                
        # Cost distribution
        all_costs = [usage.cost_usd for usage in self.usage_history]
        cost_distribution = {
            "min": min(all_costs),
            "max": max(all_costs),
            "median": statistics.median(all_costs),
            "mean": statistics.mean(all_costs),
            "std_dev": statistics.stdev(all_costs) if len(all_costs) > 1 else 0
        }
        
        # Call type breakdown
        call_type_breakdown = {}
        for call_type, tokens_list in self.call_type_tokens.items():
            call_type_breakdown[call_type] = {
                "count": self.call_type_counts[call_type],
                "total_tokens": sum(tokens_list),
                "average_tokens": statistics.mean(tokens_list),
                "total_cost": sum(usage.cost_usd for usage in self.usage_history 
                                if usage.call_type == call_type)
            }
        
        # Prompt and response length stats
        prompt_length_stats = {
            "min": min(self.prompt_lengths) if self.prompt_lengths else 0,
            "max": max(self.prompt_lengths) if self.prompt_lengths else 0,
            "median": statistics.median(self.prompt_lengths) if len(self.prompt_lengths) > 1 else 0,
            "mean": statistics.mean(self.prompt_lengths) if self.prompt_lengths else 0,
            "std_dev": statistics.stdev(self.prompt_lengths) if len(self.prompt_lengths) > 1 else 0
        }
        
        response_length_stats = {
            "min": min(self.response_lengths) if self.response_lengths else 0,
            "max": max(self.response_lengths) if self.response_lengths else 0,
            "median": statistics.median(self.response_lengths) if len(self.response_lengths) > 1 else 0,
            "mean": statistics.mean(self.response_lengths) if self.response_lengths else 0,
            "std_dev": statistics.stdev(self.response_lengths) if len(self.response_lengths) > 1 else 0
        }
        
        return DetailedUsageStats(
            total_calls=self.total_calls,
            total_tokens=total_tokens,
            total_input_tokens=self.total_input_tokens,
            total_output_tokens=self.total_output_tokens,
            total_cost=self.total_cost,
            average_tokens_per_call=total_tokens / self.total_calls if self.total_calls > 0 else 0,
            average_input_tokens=self.total_input_tokens / self.total_calls if self.total_calls > 0 else 0,
            average_output_tokens=self.total_output_tokens / self.total_calls if self.total_calls > 0 else 0,
            average_cost_per_call=self.total_cost / self.total_calls if self.total_calls > 0 else 0,
            tokens_per_minute_rate=(total_tokens / processing_duration * 60) if processing_duration > 0 else 0,
            calls_per_minute_rate=(self.total_calls / processing_duration * 60) if processing_duration > 0 else 0,
            call_type_breakdown=call_type_breakdown,
            prompt_length_stats=prompt_length_stats,
            response_length_stats=response_length_stats,
            processing_duration=processing_duration
        )

    def print_usage_summary(self):
        """Print a comprehensive summary of usage statistics."""
        stats = self.get_detailed_stats()
        
        logger.info("=" * 80)
        logger.info("DETAILED TOKEN USAGE SUMMARY")
        logger.info("=" * 80)
        
        # Basic overview
        logger.info("BASIC OVERVIEW")
        logger.info(f"   Total calls: {stats.total_calls}")
        logger.info(f"   Total tokens: {stats.total_tokens:,}")
        logger.info(f"     - Input: {stats.total_input_tokens:,}")
        logger.info(f"     - Output: {stats.total_output_tokens:,}")
        logger.info(f"   Total cost: ${stats.total_cost:.4f}")
        logger.info(f"   Processing duration: {stats.processing_duration:.1f} seconds")
        
        # Averages
        logger.info("\nAVERAGES")
        logger.info(f"   Average tokens per call: {stats.average_tokens_per_call:.1f}")
        logger.info(f"   Average input tokens: {stats.average_input_tokens:.1f}")
        logger.info(f"   Average output tokens: {stats.average_output_tokens:.1f}")
        logger.info(f"   Average cost per call: ${stats.average_cost_per_call:.4f}")
        
        # Rates
        logger.info("\nRATES")
        logger.info(f"   Tokens per minute: {stats.tokens_per_minute_rate:.1f}")
        logger.info(f"   Calls per minute: {stats.calls_per_minute_rate:.1f}")
        
        # Prompt and response analysis
        if stats.prompt_length_stats['mean'] > 0:
            logger.info("\nPROMPT ANALYSIS")
            logger.info(f"   Average prompt length: {stats.prompt_length_stats['mean']:.0f} characters")
            logger.info(f"   Min prompt length: {stats.prompt_length_stats['min']}")
            logger.info(f"   Max prompt length: {stats.prompt_length_stats['max']}")
            logger.info(f"   Prompt length std dev: {stats.prompt_length_stats['std_dev']:.1f}")
        
        if stats.response_length_stats['mean'] > 0:
            logger.info("\nRESPONSE ANALYSIS")
            logger.info(f"   Average response length: {stats.response_length_stats['mean']:.0f} characters")
            logger.info(f"   Min response length: {stats.response_length_stats['min']}")
            logger.info(f"   Max response length: {stats.response_length_stats['max']}")
            logger.info(f"   Response length std dev: {stats.response_length_stats['std_dev']:.1f}")
        
        logger.info("=" * 80)

# Convenience function for quick usage tracking
def track_llm_call(monitor: TokenMonitor, input_tokens: int, output_tokens: int, 
                  call_type: str = "unknown", prompt_length: int = 0, 
                  response_length: int = 0) -> TokenUsage:
    """
    Convenience function to track an LLM call.
    
    Args:
        monitor (TokenMonitor): The token monitor instance
        input_tokens (int): Number of input tokens
        output_tokens (int): Number of output tokens
        call_type (str): Type of call (e.g., "paper_analysis", "batch_analysis")
        prompt_length (int): Length of prompt in characters
        response_length (int): Length of response in characters
        
    Returns:
        TokenUsage: Recorded usage information
    """
    return monitor.record_usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        call_type=call_type,
        prompt_length=prompt_length,
        response_length=response_length
    ) 