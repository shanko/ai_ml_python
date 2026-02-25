#!/usr/bin/env python3
"""
Activity Reporting Agent using LangGraph - REFACTORED VERSION

This refactored version improves testability through:
1. Dependency injection for all external dependencies
2. Separation of business logic from I/O operations
3. Abstract interfaces for mocking in tests
4. Pure functions for data transformations
5. Configuration as injectable dependency

ASSUMPTIONS:
1. CSV file contains headers matching the expected attributes
2. Activities belong to Feeds, Feeds belong to Desks, Desks belong to Orgs
3. Date formats in CSV are parseable by pandas (ISO format recommended)
4. LLM provider is OpenAI-compatible (OpenAI, Azure OpenAI, etc.)
5. API key is set via OPENAI_API_KEY environment variable
6. Numeric fields (percent_complete, hours) are valid numbers or empty
7. Priority values are: 'low', 'medium', 'high' (case-insensitive)
8. Complexity values are: 'simple', 'moderate', 'complex' (case-insensitive)
"""

import sys
from typing import TypedDict, Literal, Any
from dataclasses import dataclass
import pandas as pd
from pathlib import Path

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage

from .interfaces import (
    TimeProvider,
    FileSystemInterface,
    LLMInterface,
    VisualizationRenderer,
    ConfigProvider,
)


# ==================== DATA STRUCTURES ====================


class AgentState(TypedDict):
    """State passed between graph nodes"""

    csv_path: str
    df: pd.DataFrame
    report_type: Literal["feed", "desk", "org", "custom"]
    entity_id: str | None
    custom_query: str | None
    report_content: str
    visualization_paths: list[str]
    error: str | None


@dataclass
class VisualizationData:
    """Data structure for visualization specifications"""

    data: pd.Series | pd.DataFrame
    chart_type: Literal["bar", "heatmap"]
    title: str
    xlabel: str | None = None
    ylabel: str | None = None
    colors: list[str] | None = None
    cmap: str | None = None
    figsize: tuple[int, int] = (10, 6)


@dataclass
class Dependencies:
    """Container for all injectable dependencies"""

    time_provider: TimeProvider
    file_system: FileSystemInterface
    llm: LLMInterface | None
    viz_renderer: VisualizationRenderer
    config: ConfigProvider


# ==================== DATA LOADING ====================


def load_data(state: AgentState, file_system: FileSystemInterface) -> AgentState:
    """
    Load and validate CSV data.

    Args:
        state: Current agent state
        file_system: File system interface for reading CSV

    Returns:
        Updated state with loaded DataFrame or error
    """
    try:
        df = file_system.read_csv(state["csv_path"])

        # Validate required columns
        required_cols = [
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

        missing = set(required_cols) - set(df.columns)
        if missing:
            state["error"] = f"Missing columns: {missing}"
            return state

        # Clean and normalize data
        df["priority"] = df["priority"].str.lower()
        df["complexity"] = df["complexity"].str.lower()
        df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
        df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")

        state["df"] = df
        state["error"] = None

    except Exception as e:
        state["error"] = f"Failed to load data: {str(e)}"

    return state


# ==================== DATA TRANSFORMATION (PURE FUNCTIONS) ====================


def calculate_feed_metrics(df: pd.DataFrame, feed_id: str) -> dict[str, Any] | None:
    """
    Calculate metrics for a feed (pure function).

    Args:
        df: Full DataFrame
        feed_id: Feed ID to filter by

    Returns:
        Dictionary of metrics or None if no data found
    """
    feed_df = df[df["feed_id"] == feed_id]

    if feed_df.empty:
        return None

    return {
        "feed_df": feed_df,
        "total_activities": len(feed_df),
        "avg_completion": feed_df["percent_complete"].mean(),
        "total_hours_spent": feed_df["hours_spent"].sum(),
        "total_hours_estimated": feed_df["estimated_hours"].sum(),
        "priority_counts": feed_df["priority"].value_counts().to_dict(),
        "complexity_counts": feed_df["complexity"].value_counts().to_dict(),
    }


def calculate_desk_metrics(df: pd.DataFrame, desk_id: str) -> dict[str, Any] | None:
    """
    Calculate metrics for a desk (pure function).

    Args:
        df: Full DataFrame
        desk_id: Desk ID to filter by

    Returns:
        Dictionary of metrics or None if no data found
    """
    desk_df = df[df["desk_id"] == desk_id]

    if desk_df.empty:
        return None

    feed_stats = (
        desk_df.groupby("feed_id")
        .agg({"percent_complete": "mean", "hours_spent": "sum", "id": "count"})
        .round(1)
    )

    priority_complexity_matrix = pd.crosstab(desk_df["priority"], desk_df["complexity"])

    return {
        "desk_df": desk_df,
        "total_activities": len(desk_df),
        "feed_count": desk_df["feed_id"].nunique(),
        "avg_completion": desk_df["percent_complete"].mean(),
        "total_hours_spent": desk_df["hours_spent"].sum(),
        "feed_stats": feed_stats,
        "priority_complexity_matrix": priority_complexity_matrix,
    }


def calculate_org_metrics(
    df: pd.DataFrame, org_id: str, time_provider: TimeProvider
) -> dict[str, Any] | None:
    """
    Calculate metrics for an organization (pure function with time injection).

    Args:
        df: Full DataFrame
        org_id: Organization ID to filter by
        time_provider: Time provider for current datetime

    Returns:
        Dictionary of metrics or None if no data found
    """
    org_df = df[df["org_id"] == org_id]

    if org_df.empty:
        return None

    desk_stats = (
        org_df.groupby("desk_id")
        .agg(
            {
                "percent_complete": "mean",
                "hours_spent": "sum",
                "id": "count",
                "feed_id": "nunique",
            }
        )
        .round(1)
    )
    desk_stats.columns = ["Avg Completion %", "Hours Spent", "Activities", "Feeds"]

    current_time = time_provider.now()
    overdue_count = len(org_df[org_df["end_date"] < current_time])

    return {
        "org_df": org_df,
        "total_activities": len(org_df),
        "desk_count": org_df["desk_id"].nunique(),
        "feed_count": org_df["feed_id"].nunique(),
        "avg_completion": org_df["percent_complete"].mean(),
        "desk_stats": desk_stats,
        "high_priority_count": len(org_df[org_df["priority"] == "high"]),
        "complex_count": len(org_df[org_df["complexity"] == "complex"]),
        "overdue_count": overdue_count,
    }


def prepare_feed_visualizations(
    feed_df: pd.DataFrame, feed_id: str
) -> list[VisualizationData]:
    """
    Prepare visualization data for a feed (pure function).

    Args:
        feed_df: DataFrame filtered to feed
        feed_id: Feed ID

    Returns:
        List of visualization specifications
    """
    priority_order = ["low", "medium", "high"]
    complexity_order = ["simple", "moderate", "complex"]

    priority_data = (
        feed_df.groupby("priority")["percent_complete"].mean().reindex(priority_order)
    )
    complexity_data = (
        feed_df.groupby("complexity")["percent_complete"]
        .mean()
        .reindex(complexity_order)
    )

    pivot = pd.crosstab(feed_df["priority"], feed_df["complexity"])
    pivot = pivot.reindex(priority_order, axis=0).reindex(
        complexity_order, axis=1, fill_value=0
    )

    return [
        VisualizationData(
            data=priority_data,
            chart_type="bar",
            title=f"Average Completion by Priority - FEED {feed_id}",
            xlabel="Priority",
            ylabel="Average Completion %",
            colors=["green", "orange", "red"],
            figsize=(10, 6),
        ),
        VisualizationData(
            data=complexity_data,
            chart_type="bar",
            title=f"Average Completion by Complexity - FEED {feed_id}",
            xlabel="Complexity",
            ylabel="Average Completion %",
            colors=["lightblue", "blue", "darkblue"],
            figsize=(10, 6),
        ),
        VisualizationData(
            data=pivot,
            chart_type="heatmap",
            title=f"Activity Count: Priority vs Complexity - FEED {feed_id}",
            cmap="YlOrRd",
            figsize=(8, 6),
        ),
    ]


def prepare_desk_visualizations(
    desk_df: pd.DataFrame, desk_id: str
) -> list[VisualizationData]:
    """Prepare visualization data for a desk (pure function)."""
    priority_order = ["low", "medium", "high"]
    complexity_order = ["simple", "moderate", "complex"]

    priority_data = (
        desk_df.groupby("priority")["percent_complete"].mean().reindex(priority_order)
    )
    complexity_data = (
        desk_df.groupby("complexity")["percent_complete"]
        .mean()
        .reindex(complexity_order)
    )

    pivot = pd.crosstab(desk_df["priority"], desk_df["complexity"])
    pivot = pivot.reindex(priority_order, axis=0).reindex(
        complexity_order, axis=1, fill_value=0
    )

    return [
        VisualizationData(
            data=priority_data,
            chart_type="bar",
            title=f"Average Completion by Priority - DESK {desk_id}",
            xlabel="Priority",
            ylabel="Average Completion %",
            colors=["green", "orange", "red"],
            figsize=(10, 6),
        ),
        VisualizationData(
            data=complexity_data,
            chart_type="bar",
            title=f"Average Completion by Complexity - DESK {desk_id}",
            xlabel="Complexity",
            ylabel="Average Completion %",
            colors=["lightblue", "blue", "darkblue"],
            figsize=(10, 6),
        ),
        VisualizationData(
            data=pivot,
            chart_type="heatmap",
            title=f"Activity Count: Priority vs Complexity - DESK {desk_id}",
            cmap="YlOrRd",
            figsize=(8, 6),
        ),
    ]


def prepare_org_visualizations(
    org_df: pd.DataFrame, org_id: str
) -> list[VisualizationData]:
    """Prepare visualization data for an organization (pure function)."""
    priority_order = ["low", "medium", "high"]
    complexity_order = ["simple", "moderate", "complex"]

    priority_data = (
        org_df.groupby("priority")["percent_complete"].mean().reindex(priority_order)
    )
    complexity_data = (
        org_df.groupby("complexity")["percent_complete"]
        .mean()
        .reindex(complexity_order)
    )

    pivot = pd.crosstab(org_df["priority"], org_df["complexity"])
    pivot = pivot.reindex(priority_order, axis=0).reindex(
        complexity_order, axis=1, fill_value=0
    )

    return [
        VisualizationData(
            data=priority_data,
            chart_type="bar",
            title=f"Average Completion by Priority - ORG {org_id}",
            xlabel="Priority",
            ylabel="Average Completion %",
            colors=["green", "orange", "red"],
            figsize=(10, 6),
        ),
        VisualizationData(
            data=complexity_data,
            chart_type="bar",
            title=f"Average Completion by Complexity - ORG {org_id}",
            xlabel="Complexity",
            ylabel="Average Completion %",
            colors=["lightblue", "blue", "darkblue"],
            figsize=(10, 6),
        ),
        VisualizationData(
            data=pivot,
            chart_type="heatmap",
            title=f"Activity Count: Priority vs Complexity - ORG {org_id}",
            cmap="YlOrRd",
            figsize=(8, 6),
        ),
    ]


# ==================== VISUALIZATION RENDERING ====================


def render_visualizations(
    viz_specs: list[VisualizationData],
    level: str,
    entity_id: str,
    renderer: VisualizationRenderer,
    file_system: FileSystemInterface,
    config: ConfigProvider,
) -> list[str]:
    """
    Render and save visualizations based on specifications.

    Args:
        viz_specs: List of visualization specifications
        level: Level type (feed/desk/org)
        entity_id: Entity identifier
        renderer: Visualization renderer
        file_system: File system for saving
        config: Configuration provider

    Returns:
        List of file paths where visualizations were saved
    """
    from datetime import datetime

    viz_dir = config.get_visualization_dir()
    file_system.mkdir(viz_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_paths = []

    for idx, spec in enumerate(viz_specs):
        # Create figure based on chart type
        if spec.chart_type == "bar":
            fig = renderer.create_bar_chart(
                data=spec.data,
                title=spec.title,
                xlabel=spec.xlabel or "",
                ylabel=spec.ylabel or "",
                colors=spec.colors or ["blue"],
                figsize=spec.figsize,
            )
        elif spec.chart_type == "heatmap":
            fig = renderer.create_heatmap(
                data=spec.data,
                title=spec.title,
                cmap=spec.cmap or "YlOrRd",
                figsize=spec.figsize,
            )
        else:
            continue

        # Save figure
        chart_name = "priority" if idx == 0 else "complexity" if idx == 1 else "heatmap"
        path = viz_dir / f"{level}_{entity_id}_{chart_name}_{timestamp}.png"
        file_system.save_figure(fig, path)
        renderer.close_figure(fig)
        viz_paths.append(str(path))

    return viz_paths


# ==================== REPORT GENERATION ====================


def format_dict(d: dict) -> str:
    """Format dictionary for report output (pure function)."""
    return "\n".join([f"  - {k}: {v}" for k, v in d.items()])


def format_activities(df: pd.DataFrame) -> str:
    """Format activities dataframe for report (pure function)."""
    if df.empty:
        return "  None"
    return "\n".join(
        [
            f"  - {row['title']}: {row['percent_complete']:.0f}% ({row['priority']} priority)"
            for _, row in df.iterrows()
        ]
    )


def generate_feed_report_text(metrics: dict[str, Any]) -> str:
    """Generate feed report text (pure function)."""
    feed_id = metrics["feed_df"]["feed_id"].iloc[0]
    total_hours_estimated = metrics["total_hours_estimated"]
    hours_efficiency = (
        metrics["total_hours_spent"] / total_hours_estimated * 100
        if total_hours_estimated > 0
        else 0
    )

    top_incomplete = metrics["feed_df"].nsmallest(5, "percent_complete")[
        ["title", "percent_complete", "priority"]
    ]

    report = f"""
FEED REPORT - Feed ID: {feed_id}
{"=" * 60}

SUMMARY METRICS:
- Total Activities: {metrics["total_activities"]}
- Average Completion: {metrics["avg_completion"]:.1f}%
- Total Hours Spent: {metrics["total_hours_spent"]:.1f}
- Total Hours Estimated: {total_hours_estimated:.1f}
- Hours Efficiency: {hours_efficiency:.1f}%

PRIORITY DISTRIBUTION:
{format_dict(metrics["priority_counts"])}

COMPLEXITY DISTRIBUTION:
{format_dict(metrics["complexity_counts"])}

TOP INCOMPLETE ACTIVITIES:
{format_activities(top_incomplete)}
"""
    return report


def generate_desk_report_text(metrics: dict[str, Any]) -> str:
    """Generate desk report text (pure function)."""
    desk_id = metrics["desk_df"]["desk_id"].iloc[0]

    report = f"""
DESK REPORT - Desk ID: {desk_id}
{"=" * 60}

SUMMARY METRICS:
- Total Activities: {metrics["total_activities"]}
- Total Feeds: {metrics["feed_count"]}
- Average Completion: {metrics["avg_completion"]:.1f}%
- Total Hours Spent: {metrics["total_hours_spent"]:.1f}

FEED PERFORMANCE:
{metrics["feed_stats"].to_string()}

PRIORITY vs COMPLEXITY MATRIX:
{metrics["priority_complexity_matrix"].to_string()}
"""
    return report


def generate_org_report_text(metrics: dict[str, Any]) -> str:
    """Generate organization report text (pure function)."""
    org_id = metrics["org_df"]["org_id"].iloc[0]

    report = f"""
ORGANIZATION REPORT - Org ID: {org_id}
{"=" * 60}

SUMMARY METRICS:
- Total Activities: {metrics["total_activities"]}
- Total Desks: {metrics["desk_count"]}
- Total Feeds: {metrics["feed_count"]}
- Overall Completion: {metrics["avg_completion"]:.1f}%

DESK PERFORMANCE:
{metrics["desk_stats"].to_string()}

ORGANIZATION-WIDE INSIGHTS:
- High Priority Activities: {metrics["high_priority_count"]}
- Complex Activities: {metrics["complex_count"]}
- Overdue Activities: {metrics["overdue_count"]}
"""
    return report


def prepare_data_summary(df: pd.DataFrame) -> str:
    """Prepare concise data summary for LLM context (pure function)."""
    summary = f"""
Total Activities: {len(df)}
Organizations: {df["org_id"].nunique()}
Desks: {df["desk_id"].nunique()}
Feeds: {df["feed_id"].nunique()}

Completion Stats:
- Average: {df["percent_complete"].mean():.1f}%
- Min: {df["percent_complete"].min():.1f}%
- Max: {df["percent_complete"].max():.1f}%

Priority Distribution:
{df["priority"].value_counts().to_dict()}

Complexity Distribution:
{df["complexity"].value_counts().to_dict()}

Hours:
- Total Estimated: {df["estimated_hours"].sum():.1f}
- Total Spent: {df["hours_spent"].sum():.1f}
"""
    return summary


# ==================== GRAPH NODES ====================


def generate_canned_report_node(state: AgentState, deps: Dependencies) -> AgentState:
    """
    Generate canned reports based on report type.

    Args:
        state: Current agent state
        deps: Dependency container

    Returns:
        Updated state with report content and visualization paths
    """
    if state.get("error"):
        return state

    df = state["df"]
    report_type = state["report_type"]
    entity_id = state["entity_id"]

    try:
        if report_type == "feed":
            metrics = calculate_feed_metrics(df, entity_id)
            if metrics is None:
                state["error"] = f"No activities found for feed_id: {entity_id}"
                return state

            report_text = generate_feed_report_text(metrics)
            viz_specs = prepare_feed_visualizations(metrics["feed_df"], entity_id)

        elif report_type == "desk":
            metrics = calculate_desk_metrics(df, entity_id)
            if metrics is None:
                state["error"] = f"No activities found for desk_id: {entity_id}"
                return state

            report_text = generate_desk_report_text(metrics)
            viz_specs = prepare_desk_visualizations(metrics["desk_df"], entity_id)

        elif report_type == "org":
            metrics = calculate_org_metrics(df, entity_id, deps.time_provider)
            if metrics is None:
                state["error"] = f"No activities found for org_id: {entity_id}"
                return state

            report_text = generate_org_report_text(metrics)
            viz_specs = prepare_org_visualizations(metrics["org_df"], entity_id)

        else:
            state["error"] = f"Invalid report type: {report_type}"
            return state

        # Render visualizations
        viz_paths = render_visualizations(
            viz_specs=viz_specs,
            level=report_type,
            entity_id=entity_id,
            renderer=deps.viz_renderer,
            file_system=deps.file_system,
            config=deps.config,
        )

        state["report_content"] = report_text
        state["visualization_paths"] = viz_paths

    except Exception as e:
        state["error"] = f"Report generation failed: {str(e)}"

    return state


def generate_custom_report_node(state: AgentState, deps: Dependencies) -> AgentState:
    """
    Generate custom report using LLM.

    Args:
        state: Current agent state
        deps: Dependency container

    Returns:
        Updated state with custom report content
    """
    if state.get("error"):
        return state

    if deps.llm is None:
        state["error"] = "LLM not configured for custom reports"
        return state

    df = state["df"]
    custom_query = state["custom_query"]

    try:
        # Prepare data summary for LLM context
        data_summary = prepare_data_summary(df)

        # Construct prompt
        system_prompt = """You are a data analyst specializing in project management reporting.
You have access to activity data with organizational hierarchy (Org > Desk > Feed > Activity).
Analyze the data and provide clear, actionable insights based on the user's query.
Format your response as a professional report with sections and bullet points."""

        user_prompt = f"""
USER QUERY: {custom_query}

DATA SUMMARY:
{data_summary}

Please analyze this data and provide a comprehensive report addressing the user's query.
Include specific metrics, trends, and recommendations where applicable.
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = deps.llm.invoke(messages)
        state["report_content"] = response.content
        state["visualization_paths"] = []

    except Exception as e:
        state["error"] = f"Custom report generation failed: {str(e)}"

    return state


def route_report_type(state: AgentState) -> str:
    """Route to appropriate report generation node."""
    if state.get("error"):
        return "end"

    if state["report_type"] == "custom":
        return "custom"
    else:
        return "canned"


# ==================== GRAPH CONSTRUCTION ====================


def build_agent_graph(deps: Dependencies) -> StateGraph:
    """
    Build LangGraph workflow with injected dependencies.

    Args:
        deps: Dependency container

    Returns:
        Compiled state graph
    """
    workflow = StateGraph(AgentState)

    # Add nodes with dependency injection
    workflow.add_node("load_data", lambda state: load_data(state, deps.file_system))
    workflow.add_node(
        "canned_report", lambda state: generate_canned_report_node(state, deps)
    )
    workflow.add_node(
        "custom_report", lambda state: generate_custom_report_node(state, deps)
    )

    # Add edges
    workflow.set_entry_point("load_data")
    workflow.add_conditional_edges(
        "load_data",
        route_report_type,
        {"canned": "canned_report", "custom": "custom_report", "end": END},
    )
    workflow.add_edge("canned_report", END)
    workflow.add_edge("custom_report", END)

    return workflow.compile()


# ==================== PUBLIC API ====================


def run_report(
    csv_path: str,
    report_type: Literal["feed", "desk", "org", "custom"],
    entity_id: str | None = None,
    custom_query: str | None = None,
    deps: Dependencies | None = None,
) -> dict:
    """
    Run reporting agent with dependency injection.

    Args:
        csv_path: Path to CSV file with activity data
        report_type: Type of report to generate
        entity_id: ID of entity (feed_id, desk_id, or org_id) for canned reports
        custom_query: Natural language query for custom reports
        deps: Dependency container (uses defaults if not provided)

    Returns:
        dict with keys: report_content, visualization_paths, error
    """
    # Validate inputs
    if report_type != "custom" and not entity_id:
        return {"error": "entity_id required for canned reports"}

    if report_type == "custom" and not custom_query:
        return {"error": "custom_query required for custom reports"}

    # Use default dependencies if not provided
    if deps is None:
        from .implementations import (
            RealTimeProvider,
            RealFileSystem,
            OpenAILLM,
            MatplotlibRenderer,
            DefaultConfig,
        )

        config = DefaultConfig()
        deps = Dependencies(
            time_provider=RealTimeProvider(),
            file_system=RealFileSystem(),
            llm=OpenAILLM(
                model=config.get_llm_model(), temperature=config.get_llm_temperature()
            )
            if report_type == "custom"
            else None,
            viz_renderer=MatplotlibRenderer(),
            config=config,
        )

    # Initialize state
    initial_state = {
        "csv_path": csv_path,
        "df": pd.DataFrame(),
        "report_type": report_type,
        "entity_id": entity_id,
        "custom_query": custom_query,
        "report_content": "",
        "visualization_paths": [],
        "error": None,
    }

    # Run agent
    graph = build_agent_graph(deps)
    final_state = graph.invoke(initial_state)

    return {
        "report_content": final_state["report_content"],
        "visualization_paths": final_state["visualization_paths"],
        "error": final_state["error"],
    }


# ==================== CLI ====================


def main():
    """Command-line interface"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Activity Reporting Agent (Refactored)"
    )
    parser.add_argument("csv_path", help="Path to CSV file")
    parser.add_argument(
        "report_type", choices=["feed", "desk", "org", "custom"], help="Type of report"
    )
    parser.add_argument("--entity-id", help="Entity ID (for feed/desk/org reports)")
    parser.add_argument("--query", help="Custom query (for custom reports)")
    parser.add_argument("--output-dir", default="./reports", help="Output directory")
    parser.add_argument("--llm-model", default="gpt-4", help="LLM model name")

    args = parser.parse_args()

    # Setup dependencies
    from .implementations import (
        RealTimeProvider,
        RealFileSystem,
        OpenAILLM,
        MatplotlibRenderer,
        DefaultConfig,
    )

    config = DefaultConfig(llm_model=args.llm_model, output_dir=args.output_dir)

    deps = Dependencies(
        time_provider=RealTimeProvider(),
        file_system=RealFileSystem(),
        llm=OpenAILLM(
            model=config.get_llm_model(), temperature=config.get_llm_temperature()
        )
        if args.report_type == "custom"
        else None,
        viz_renderer=MatplotlibRenderer(),
        config=config,
    )

    # Run report
    result = run_report(
        csv_path=args.csv_path,
        report_type=args.report_type,
        entity_id=args.entity_id,
        custom_query=args.query,
        deps=deps,
    )

    if result["error"]:
        print(f"ERROR: {result['error']}", file=sys.stderr)
        sys.exit(1)

    print(result["report_content"])

    if result["visualization_paths"]:
        print(f"\nVisualizations saved to:")
        for path in result["visualization_paths"]:
            print(f"  - {path}")


if __name__ == "__main__":
    main()
