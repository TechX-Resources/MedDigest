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
- Proactive rate limiting with early warnings
- Better time management and token reset logic
"""

import time
import logging
import threading
from typing import Dict, List, Union, Optional
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
    call_type: str = "unknown"
    prompt_length: int = 0
    response_length: int = 0


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

    def __init__(self, max_tokens_per_minute: int = 16000, warning_threshold: float = 0.9):
        """
        Initialize the token monitor.
        
        Args:
            max_tokens_per_minute (int): Maximum tokens allowed per minute
            warning_threshold (float): Threshold (0.0-1.0) for warning when approaching limit
        """
        self.max_tokens_per_minute = max_tokens_per_minute
        self.warning_threshold = warning_threshold
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
        
        # Rate limiting improvements
        self.last_reset_time = time.time()
        self.warning_issued = False
        
        # Performance tracking
        self.sleep_time_total = 0.0
        self.rate_limit_hits = 0

    def set_model_costs(self, input_cost_per_million: float, output_cost_per_million: float) -> None:
        """
        Set custom cost rates for different models.
        
        Args:
            input_cost_per_million (float): Cost per million input tokens
            output_cost_per_million (float): Cost per million output tokens
        """
        self.INPUT_COST_PER_MILLION = input_cost_per_million
        self.OUTPUT_COST_PER_MILLION = output_cost_per_million
        logger.info(f"Updated model costs: ${input_cost_per_million:.3f}/1M input, ${output_cost_per_million:.3f}/1M output")

    def estimate_batch_tokens(self, texts: List[str]) -> int:
        """
        Estimate total tokens for a batch of texts.
        
        Args:
            texts (List[str]): List of texts to estimate
            
        Returns:
            int: Estimated total tokens
        """
        return sum(self.count_tokens(text) for text in texts)

    def process_batch_with_rate_limiting(self, batch_data: List[Dict], 
                                       call_type: str = "batch_processing") -> List[TokenUsage]:
        """
        Process a batch of calls with automatic rate limiting.
        
        Args:
            batch_data (List[Dict]): List of dicts with 'input_tokens', 'output_tokens', etc.
            call_type (str): Type of batch processing
            
        Returns:
            List[TokenUsage]: List of usage records for each call
        """
        results = []
        
        for i, data in enumerate(batch_data):
            # Estimate tokens for this call
            estimated_tokens = data.get('estimated_tokens', 
                                     data.get('input_tokens', 0) + data.get('output_tokens', 0))
            
            # Wait if needed before making the call
            wait_time = self.wait_if_needed(estimated_tokens)
            if wait_time > 0:
                logger.info(f"Batch item {i+1}/{len(batch_data)}: Waited {wait_time:.1f}s for rate limit")
            
            # Record the actual usage after the call
            usage = self.record_usage(
                input_tokens=data.get('input_tokens', 0),
                output_tokens=data.get('output_tokens', 0),
                call_type=f"{call_type}_item_{i+1}",
                prompt_length=data.get('prompt_length', 0),
                response_length=data.get('response_length', 0)
            )
            results.append(usage)
        
        return results

    def get_rate_limiting_stats(self) -> Dict[str, Union[int, float]]:
        """
        Get statistics about rate limiting behavior.
        
        Returns:
            Dict containing rate limiting statistics
        """
        return {
            "total_sleep_time": self.sleep_time_total,
            "rate_limit_hits": self.rate_limit_hits,
            "average_sleep_per_hit": (self.sleep_time_total / self.rate_limit_hits 
                                    if self.rate_limit_hits > 0 else 0),
            "efficiency_ratio": (self.total_calls / (self.total_calls + self.rate_limit_hits) 
                               if self.total_calls > 0 else 1.0)
        }

    def reset_minute_window(self) -> None:
        """Manually reset the minute window (useful for testing or recovery)."""
        with self.lock:
            current_time = time.time()
            self.tokens_this_minute = 0
            self.minute_start_time = current_time
            self.last_reset_time = current_time
            self.warning_issued = False
            logger.info("Manually reset minute window")

    def _check_and_reset_minute_window(self, current_time: float) -> None:
        """
        Check if we need to reset the minute window and handle token reset logic.
        
        Args:
            current_time (float): Current timestamp
        """
        time_since_reset = current_time - self.last_reset_time
        
        if time_since_reset >= 60:
            # Reset the minute window
            self.tokens_this_minute = 0
            self.minute_start_time = current_time
            self.last_reset_time = current_time
            self.warning_issued = False
            logger.debug(f"Reset minute window. Time since last reset: {time_since_reset:.1f}s")

    def _should_sleep_for_rate_limit(self, total_tokens: int, current_time: float) -> Optional[float]:
        """
        Determine if we need to sleep due to rate limiting.
        
        Args:
            total_tokens (int): Tokens for this call
            current_time (float): Current timestamp
            
        Returns:
            Optional[float]: Sleep duration in seconds, or None if no sleep needed
        """
        # Check if this call would exceed the limit
        if self.tokens_this_minute + total_tokens > self.max_tokens_per_minute:
            # Calculate remaining time in current minute
            time_elapsed = current_time - self.minute_start_time
            remaining_time = 60 - time_elapsed
            
            if remaining_time > 0:
                logger.info(f"Token limit would be exceeded ({self.tokens_this_minute + total_tokens} > {self.max_tokens_per_minute}). "
                          f"Sleeping for {remaining_time:.1f} seconds...")
                self.rate_limit_hits += 1
                return remaining_time
            else:
                # Minute has already passed, reset immediately
                self._check_and_reset_minute_window(current_time)
                return None
        
        return None

    def _check_warning_threshold(self, total_tokens: int) -> None:
        """
        Check if we're approaching the token limit and issue a warning.
        
        Args:
            total_tokens (int): Tokens for this call
        """
        if not self.warning_issued:
            projected_tokens = self.tokens_this_minute + total_tokens
            usage_ratio = projected_tokens / self.max_tokens_per_minute
            
            if usage_ratio >= self.warning_threshold:
                logger.warning(f"Approaching token limit: {projected_tokens}/{self.max_tokens_per_minute} "
                             f"({usage_ratio:.1%}) - Consider reducing batch size or waiting")
                self.warning_issued = True

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
            
            # Check and reset minute window if needed
            self._check_and_reset_minute_window(current_time)
            
            # Check for rate limiting
            sleep_duration = self._should_sleep_for_rate_limit(total_tokens, current_time)
            if sleep_duration:
                time.sleep(sleep_duration)
                # Re-check after sleep
                current_time = time.time()
                self._check_and_reset_minute_window(current_time)
            
            # Check warning threshold
            self._check_warning_threshold(total_tokens)
            
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
                f"{total_tokens} total tokens (${cost:.4f} USD) "
                f"[{self.tokens_this_minute}/{self.max_tokens_per_minute} this minute]"
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

    def get_current_usage(self) -> Dict[str, Union[int, float]]:
        """
        Get current minute usage statistics.
        
        Returns:
            Dict containing current usage info
        """
        with self.lock:
            current_time = time.time()
            time_elapsed = current_time - self.minute_start_time
            time_remaining = max(0, 60 - time_elapsed)
            usage_ratio = self.tokens_this_minute / self.max_tokens_per_minute
            
            return {
                "tokens_used": self.tokens_this_minute,
                "tokens_remaining": self.max_tokens_per_minute - self.tokens_this_minute,
                "usage_ratio": usage_ratio,
                "time_elapsed": time_elapsed,
                "time_remaining": time_remaining,
                "calls_this_minute": len([u for u in self.usage_history 
                                        if u.timestamp >= self.minute_start_time])
            }

    def can_make_call(self, estimated_tokens: int) -> bool:
        """
        Check if a call with estimated tokens can be made without hitting rate limit.
        
        Args:
            estimated_tokens (int): Estimated tokens for the call
            
        Returns:
            bool: True if call can be made, False if it would exceed limit
        """
        with self.lock:
            current_time = time.time()
            self._check_and_reset_minute_window(current_time)
            return self.tokens_this_minute + estimated_tokens <= self.max_tokens_per_minute

    def wait_if_needed(self, estimated_tokens: int) -> float:
        """
        Wait if necessary to avoid hitting rate limit, return wait time.
        
        Args:
            estimated_tokens (int): Estimated tokens for the call
            
        Returns:
            float: Time waited in seconds
        """
        with self.lock:
            current_time = time.time()
            self._check_and_reset_minute_window(current_time)
            
            sleep_duration = self._should_sleep_for_rate_limit(estimated_tokens, current_time)
            if sleep_duration:
                time.sleep(sleep_duration)
                self.sleep_time_total += sleep_duration
                return sleep_duration
            return 0.0

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