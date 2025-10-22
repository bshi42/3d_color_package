"""
Reporting Module for DeCA/InterDeCA Analysis

This module provides tools for generating comprehensive analysis reports
from Dense Correspondence Analysis results.
"""

from .results_reporter import ResultsReporter, test_reporter

__all__ = ['ResultsReporter', 'test_reporter']