#!/usr/bin/env python3
"""
Abstract interfaces for dependency injection in reporting agent.

This module defines the contracts for all external dependencies,
enabling testability and flexibility through abstraction.
"""

from abc import ABC, abstractmethod
from typing import Any
from datetime import datetime
from pathlib import Path
import pandas as pd


class TimeProvider(ABC):
    """Abstract interface for time-related operations"""

    @abstractmethod
    def now(self) -> datetime:
        """Return the current datetime"""
        pass


class FileSystemInterface(ABC):
    """Abstract interface for file system operations"""

    @abstractmethod
    def mkdir(self, path: Path, parents: bool = True, exist_ok: bool = True) -> None:
        """Create a directory"""
        pass

    @abstractmethod
    def save_figure(self, figure: Any, path: Path) -> None:
        """Save a matplotlib figure to a file"""
        pass

    @abstractmethod
    def read_csv(self, path: str) -> pd.DataFrame:
        """Read a CSV file and return a DataFrame"""
        pass

    @abstractmethod
    def path_exists(self, path: Path) -> bool:
        """Check if a path exists"""
        pass


class LLMInterface(ABC):
    """Abstract interface for Language Model operations"""

    @abstractmethod
    def invoke(self, messages: list[Any]) -> Any:
        """
        Invoke the LLM with a list of messages.

        Args:
            messages: List of message objects (SystemMessage, HumanMessage, etc.)

        Returns:
            Response object with content attribute
        """
        pass


class VisualizationRenderer(ABC):
    """Abstract interface for creating visualizations"""

    @abstractmethod
    def create_bar_chart(
        self,
        data: pd.Series,
        title: str,
        xlabel: str,
        ylabel: str,
        colors: list[str],
        figsize: tuple[int, int] = (10, 6),
    ) -> Any:
        """Create a bar chart and return the figure object"""
        pass

    @abstractmethod
    def create_heatmap(
        self,
        data: pd.DataFrame,
        title: str,
        cmap: str = "YlOrRd",
        figsize: tuple[int, int] = (8, 6),
    ) -> Any:
        """Create a heatmap and return the figure object"""
        pass

    @abstractmethod
    def close_figure(self, figure: Any) -> None:
        """Close a figure to free resources"""
        pass


class ConfigProvider(ABC):
    """Abstract interface for configuration management"""

    @abstractmethod
    def get_llm_model(self) -> str:
        """Get LLM model name"""
        pass

    @abstractmethod
    def get_llm_temperature(self) -> float:
        """Get LLM temperature"""
        pass

    @abstractmethod
    def get_output_dir(self) -> Path:
        """Get output directory path"""
        pass

    @abstractmethod
    def get_visualization_dir(self) -> Path:
        """Get visualization directory path"""
        pass
