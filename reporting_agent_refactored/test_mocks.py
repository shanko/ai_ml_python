#!/usr/bin/env python3
"""
Mock implementations of interfaces for testing purposes.

This module provides test doubles (mocks, fakes) for all abstract interfaces
to enable fast, isolated unit tests without external dependencies.
"""

from datetime import datetime
from pathlib import Path
from typing import Any
import pandas as pd
from io import StringIO

from .interfaces import (
    TimeProvider,
    FileSystemInterface,
    LLMInterface,
    VisualizationRenderer,
    ConfigProvider,
)


class MockTimeProvider(TimeProvider):
    """Mock time provider that returns a fixed datetime"""

    def __init__(self, fixed_time: datetime | None = None):
        """
        Initialize with a fixed time.

        Args:
            fixed_time: Fixed datetime to return (defaults to 2024-01-01)
        """
        self._fixed_time = fixed_time or datetime(2024, 1, 1, 12, 0, 0)

    def now(self) -> datetime:
        """Return the fixed datetime"""
        return self._fixed_time

    def set_time(self, new_time: datetime) -> None:
        """Update the fixed time (useful for testing time-dependent behavior)"""
        self._fixed_time = new_time


class MockFileSystem(FileSystemInterface):
    """Mock file system that operates in memory"""

    def __init__(self):
        """Initialize mock file system with in-memory storage"""
        self.directories: set[Path] = set()
        self.files: dict[str, Any] = {}
        self.csv_data: dict[str, pd.DataFrame] = {}

    def mkdir(self, path: Path, parents: bool = True, exist_ok: bool = True) -> None:
        """Record directory creation"""
        self.directories.add(path)

    def save_figure(self, figure: Any, path: Path) -> None:
        """Record figure save (doesn't actually save)"""
        self.files[str(path)] = figure

    def read_csv(self, path: str) -> pd.DataFrame:
        """Return pre-loaded CSV data"""
        if path not in self.csv_data:
            raise FileNotFoundError(f"No mock data loaded for {path}")
        return self.csv_data[path].copy()

    def path_exists(self, path: Path) -> bool:
        """Check if path exists in mock file system"""
        return path in self.directories or str(path) in self.files

    def load_csv_data(self, path: str, df: pd.DataFrame) -> None:
        """Load mock CSV data for testing"""
        self.csv_data[path] = df

    def get_saved_files(self) -> list[str]:
        """Get list of saved file paths"""
        return list(self.files.keys())

    def clear(self) -> None:
        """Clear all mock data"""
        self.directories.clear()
        self.files.clear()
        self.csv_data.clear()


class MockLLM(LLMInterface):
    """Mock LLM that returns predefined responses"""

    def __init__(self, response_content: str = "Mock LLM response"):
        """
        Initialize with a fixed response.

        Args:
            response_content: Content to return in response
        """
        self.response_content = response_content
        self.call_history: list[list[Any]] = []

    def invoke(self, messages: list[Any]) -> Any:
        """
        Return a mock response and record the call.

        Args:
            messages: List of messages

        Returns:
            Mock response object with content attribute
        """
        self.call_history.append(messages)

        # Create a simple response object
        class MockResponse:
            def __init__(self, content: str):
                self.content = content

        return MockResponse(self.response_content)

    def set_response(self, content: str) -> None:
        """Update the response content"""
        self.response_content = content

    def get_last_call(self) -> list[Any] | None:
        """Get the last invocation's messages"""
        return self.call_history[-1] if self.call_history else None

    def get_call_count(self) -> int:
        """Get number of times invoke was called"""
        return len(self.call_history)


class MockVisualizationRenderer(VisualizationRenderer):
    """Mock visualization renderer that doesn't actually create plots"""

    def __init__(self):
        """Initialize mock renderer with call tracking"""
        self.created_charts: list[dict[str, Any]] = []
        self.closed_figures: list[Any] = []

    def create_bar_chart(
        self,
        data: pd.Series,
        title: str,
        xlabel: str,
        ylabel: str,
        colors: list[str],
        figsize: tuple[int, int] = (10, 6),
    ) -> Any:
        """Record bar chart creation and return mock figure"""
        chart_info = {
            "type": "bar",
            "data": data,
            "title": title,
            "xlabel": xlabel,
            "ylabel": ylabel,
            "colors": colors,
            "figsize": figsize,
        }
        self.created_charts.append(chart_info)

        # Return a simple mock figure object
        class MockFigure:
            def __init__(self, info):
                self.info = info

        return MockFigure(chart_info)

    def create_heatmap(
        self,
        data: pd.DataFrame,
        title: str,
        cmap: str = "YlOrRd",
        figsize: tuple[int, int] = (8, 6),
    ) -> Any:
        """Record heatmap creation and return mock figure"""
        chart_info = {
            "type": "heatmap",
            "data": data,
            "title": title,
            "cmap": cmap,
            "figsize": figsize,
        }
        self.created_charts.append(chart_info)

        class MockFigure:
            def __init__(self, info):
                self.info = info

        return MockFigure(chart_info)

    def close_figure(self, figure: Any) -> None:
        """Record figure closure"""
        self.closed_figures.append(figure)

    def get_chart_count(self) -> int:
        """Get number of charts created"""
        return len(self.created_charts)

    def get_charts_by_type(self, chart_type: str) -> list[dict]:
        """Get all charts of a specific type"""
        return [c for c in self.created_charts if c["type"] == chart_type]

    def clear(self) -> None:
        """Clear all recorded data"""
        self.created_charts.clear()
        self.closed_figures.clear()


class MockConfig(ConfigProvider):
    """Mock configuration provider with customizable values"""

    def __init__(
        self,
        llm_model: str = "gpt-4",
        llm_temperature: float = 0.2,
        output_dir: str = "./test_reports",
    ):
        """
        Initialize with test configuration.

        Args:
            llm_model: LLM model name
            llm_temperature: LLM temperature
            output_dir: Output directory path
        """
        self._llm_model = llm_model
        self._llm_temperature = llm_temperature
        self._output_dir = Path(output_dir)

    def get_llm_model(self) -> str:
        """Get LLM model name"""
        return self._llm_model

    def get_llm_temperature(self) -> float:
        """Get LLM temperature"""
        return self._llm_temperature

    def get_output_dir(self) -> Path:
        """Get output directory"""
        return self._output_dir

    def get_visualization_dir(self) -> Path:
        """Get visualization directory"""
        return self._output_dir / "visualizations"


# ==================== TEST DATA FACTORIES ====================


def create_test_dataframe(num_rows: int = 10) -> pd.DataFrame:
    """
    Create a test DataFrame with realistic activity data.

    Args:
        num_rows: Number of rows to generate

    Returns:
        DataFrame with test data
    """
    data = {
        "id": range(1, num_rows + 1),
        "feed_id": ["F1"] * (num_rows // 2) + ["F2"] * (num_rows - num_rows // 2),
        "desk_id": ["D1"] * num_rows,
        "org_id": ["O1"] * num_rows,
        "owner_id": [f"U{i % 3 + 1}" for i in range(num_rows)],
        "title": [f"Task {i}" for i in range(1, num_rows + 1)],
        "description": [f"Description for task {i}" for i in range(1, num_rows + 1)],
        "start_date": pd.date_range("2024-01-01", periods=num_rows, freq="D"),
        "end_date": pd.date_range("2024-02-01", periods=num_rows, freq="D"),
        "priority": ["low", "medium", "high"] * (num_rows // 3)
        + ["medium"] * (num_rows % 3),
        "percent_complete": [i * 10 for i in range(num_rows)],
        "estimated_hours": [8.0] * num_rows,
        "hours_spent": [float(i) for i in range(num_rows)],
        "complexity": ["simple", "moderate", "complex"] * (num_rows // 3)
        + ["simple"] * (num_rows % 3),
    }
    return pd.DataFrame(data)


def create_multi_entity_dataframe() -> pd.DataFrame:
    """
    Create a DataFrame with multiple feeds, desks, and orgs for testing.

    Returns:
        DataFrame with diverse test data
    """
    data = {
        "id": range(1, 31),
        "feed_id": ["F1"] * 10 + ["F2"] * 10 + ["F3"] * 10,
        "desk_id": ["D1"] * 10 + ["D1"] * 10 + ["D2"] * 10,
        "org_id": ["O1"] * 20 + ["O2"] * 10,
        "owner_id": [f"U{i % 5 + 1}" for i in range(30)],
        "title": [f"Activity {i}" for i in range(1, 31)],
        "description": [f"Description {i}" for i in range(1, 31)],
        "start_date": pd.date_range("2023-12-01", periods=30, freq="D"),
        "end_date": pd.date_range("2024-01-15", periods=30, freq="D"),
        "priority": (["low"] * 10 + ["medium"] * 10 + ["high"] * 10),
        "percent_complete": list(range(0, 100, 3))[:30]
        + [100] * (30 - len(list(range(0, 100, 3))[:30])),
        "estimated_hours": [10.0, 20.0, 5.0] * 10,
        "hours_spent": [8.0, 15.0, 7.0] * 10,
        "complexity": ["simple"] * 10 + ["moderate"] * 10 + ["complex"] * 10,
    }
    return pd.DataFrame(data)


def create_empty_dataframe() -> pd.DataFrame:
    """
    Create an empty DataFrame with correct schema.

    Returns:
        Empty DataFrame with all required columns
    """
    return pd.DataFrame(
        columns=[
            "id",
            "feed_id",
            "desk_id",
            "org_id",
            "owner_id",
            "title",
            "description",
            "start_date",
            "end_date",
            "priority",
            "percent_complete",
            "estimated_hours",
            "hours_spent",
            "complexity",
        ]
    )
