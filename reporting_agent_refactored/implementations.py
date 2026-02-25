#!/usr/bin/env python3
"""
Concrete implementations of interfaces for production use.

This module provides real implementations of all abstract interfaces
defined in interfaces.py.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_openai import ChatOpenAI

from .interfaces import (
    TimeProvider,
    FileSystemInterface,
    LLMInterface,
    VisualizationRenderer,
    ConfigProvider,
)


class RealTimeProvider(TimeProvider):
    """Production time provider using system time"""

    def now(self) -> datetime:
        """Return the current system datetime"""
        return datetime.now()


class RealFileSystem(FileSystemInterface):
    """Production file system using actual disk operations"""

    def mkdir(self, path: Path, parents: bool = True, exist_ok: bool = True) -> None:
        """Create a directory on disk"""
        path.mkdir(parents=parents, exist_ok=exist_ok)

    def save_figure(self, figure: Any, path: Path) -> None:
        """Save a matplotlib figure to disk"""
        figure.savefig(path)

    def read_csv(self, path: str) -> pd.DataFrame:
        """Read a CSV file from disk"""
        return pd.read_csv(path)

    def path_exists(self, path: Path) -> bool:
        """Check if a path exists on disk"""
        return path.exists()


class OpenAILLM(LLMInterface):
    """OpenAI implementation of LLM interface"""

    def __init__(
        self, model: str = "gpt-4", temperature: float = 0.2, api_key: str | None = None
    ):
        """
        Initialize OpenAI LLM client.

        Args:
            model: Model name (e.g., "gpt-4", "gpt-3.5-turbo")
            temperature: Sampling temperature
            api_key: OpenAI API key (uses env var if not provided)
        """
        self.model = model
        self.temperature = temperature
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = None

    @property
    def client(self) -> ChatOpenAI:
        """Lazy initialization of ChatOpenAI client"""
        if self._client is None:
            self._client = ChatOpenAI(
                model=self.model,
                temperature=self.temperature,
                api_key=self.api_key,
            )
        return self._client

    def invoke(self, messages: list[Any]) -> Any:
        """Invoke OpenAI LLM with messages"""
        return self.client.invoke(messages)


class MatplotlibRenderer(VisualizationRenderer):
    """Matplotlib/Seaborn implementation of visualization renderer"""

    def create_bar_chart(
        self,
        data: pd.Series,
        title: str,
        xlabel: str,
        ylabel: str,
        colors: list[str],
        figsize: tuple[int, int] = (10, 6),
    ) -> Any:
        """Create a bar chart using matplotlib"""
        fig, ax = plt.subplots(figsize=figsize)
        data.plot(kind="bar", ax=ax, color=colors)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        plt.xticks(rotation=0)
        plt.tight_layout()
        return fig

    def create_heatmap(
        self,
        data: pd.DataFrame,
        title: str,
        cmap: str = "YlOrRd",
        figsize: tuple[int, int] = (8, 6),
    ) -> Any:
        """Create a heatmap using seaborn"""
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(data, annot=True, fmt="d", cmap=cmap, ax=ax)
        ax.set_title(title)
        plt.tight_layout()
        return fig

    def close_figure(self, figure: Any) -> None:
        """Close a matplotlib figure"""
        plt.close(figure)


class DefaultConfig(ConfigProvider):
    """Default configuration implementation"""

    def __init__(
        self,
        llm_model: str = "gpt-4",
        llm_temperature: float = 0.2,
        output_dir: str = "./reports",
    ):
        """
        Initialize configuration.

        Args:
            llm_model: LLM model name
            llm_temperature: LLM sampling temperature
            output_dir: Base output directory for reports
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
        """Get output directory path"""
        return self._output_dir

    def get_visualization_dir(self) -> Path:
        """Get visualization directory path"""
        return self._output_dir / "visualizations"
