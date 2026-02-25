#!/usr/bin/env python3
"""
Refactored Activity Reporting Agent with Dependency Injection.

This module provides a testable implementation of the reporting agent
with clear separation of concerns and dependency injection.
"""

from .interfaces import (
    TimeProvider,
    FileSystemInterface,
    LLMInterface,
    VisualizationRenderer,
    ConfigProvider,
)

from .implementations import (
    RealTimeProvider,
    RealFileSystem,
    OpenAILLM,
    MatplotlibRenderer,
    DefaultConfig,
)

from .reporting_agent_refactored import (
    AgentState,
    Dependencies,
    VisualizationData,
    run_report,
    build_agent_graph,
    load_data,
    calculate_feed_metrics,
    calculate_desk_metrics,
    calculate_org_metrics,
    generate_feed_report_text,
    generate_desk_report_text,
    generate_org_report_text,
    prepare_data_summary,
    format_dict,
    format_activities,
)

__all__ = [
    # Interfaces
    "TimeProvider",
    "FileSystemInterface",
    "LLMInterface",
    "VisualizationRenderer",
    "ConfigProvider",
    # Implementations
    "RealTimeProvider",
    "RealFileSystem",
    "OpenAILLM",
    "MatplotlibRenderer",
    "DefaultConfig",
    # Core functionality
    "AgentState",
    "Dependencies",
    "VisualizationData",
    "run_report",
    "build_agent_graph",
    # Data functions
    "load_data",
    "calculate_feed_metrics",
    "calculate_desk_metrics",
    "calculate_org_metrics",
    # Report generation
    "generate_feed_report_text",
    "generate_desk_report_text",
    "generate_org_report_text",
    "prepare_data_summary",
    # Utilities
    "format_dict",
    "format_activities",
]

__version__ = "2.0.0"
