#!/usr/bin/env python3
"""
Activity Reporting Agent using LangGraph

ASSUMPTIONS:
1. CSV file contains headers matching the expected attributes
2. Activities belong to Feeds, Feeds belong to Desks, Desks belong to Orgs
3. Date formats in CSV are parseable by pandas (ISO format recommended)
4. LLM provider is OpenAI-compatible
5. API key is set via OPENAI_API_KEY environment variable
6. Numeric fields (percent_complete, hours) are valid numbers or empty
7. Priority values are: 'low', 'medium', 'high' (case-insensitive)
8. Complexity values are: 'simple', 'moderate', 'complex' (case-insensitive)
"""

import os
import sys
from typing import TypedDict, Annotated, Literal
from dataclasses import dataclass
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


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
class ReportConfig:
    """Configuration for report generation"""
    llm_model: str = "gpt-4"
    llm_temperature: float = 0.2
    output_dir: str = "./reports"
    
    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


# ==================== DATA LOADING ====================

def load_data(state: AgentState) -> AgentState:
    """Load and validate CSV data"""
    try:
        df = pd.read_csv(state["csv_path"])
        
        # Validate required columns
        required_cols = [
            "id", "feed_id", "desk_id", "org_id", "owner_id", 
            "title", "description", "start_date", "end_date",
            "priority", "percent_complete", "estimated_hours", 
            "hours_spent", "complexity"
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


# ==================== CANNED REPORTS ====================

def generate_feed_report(df: pd.DataFrame, feed_id: str) -> tuple[str, list]:
    """Generate feed-level report with visualizations"""
    feed_df = df[df["feed_id"] == feed_id]
    
    if feed_df.empty:
        return f"No activities found for feed_id: {feed_id}", []
    
    # Calculate metrics
    total_activities = len(feed_df)
    avg_completion = feed_df["percent_complete"].mean()
    total_hours_spent = feed_df["hours_spent"].sum()
    total_hours_estimated = feed_df["estimated_hours"].sum()
    
    # Priority breakdown
    priority_counts = feed_df["priority"].value_counts().to_dict()
    
    # Complexity breakdown
    complexity_counts = feed_df["complexity"].value_counts().to_dict()
    
    # Build report text
    report = f"""
FEED REPORT - Feed ID: {feed_id}
{'=' * 60}

SUMMARY METRICS:
- Total Activities: {total_activities}
- Average Completion: {avg_completion:.1f}%
- Total Hours Spent: {total_hours_spent:.1f}
- Total Hours Estimated: {total_hours_estimated:.1f}
- Hours Efficiency: {(total_hours_spent/total_hours_estimated*100):.1f}% 

PRIORITY DISTRIBUTION:
{_format_dict(priority_counts)}

COMPLEXITY DISTRIBUTION:
{_format_dict(complexity_counts)}

TOP INCOMPLETE ACTIVITIES:
{_format_activities(feed_df.nsmallest(5, 'percent_complete')[['title', 'percent_complete', 'priority']])}
"""
    
    # Generate visualizations
    viz_paths = _create_visualizations(feed_df, "feed", feed_id)
    
    return report, viz_paths


def generate_desk_report(df: pd.DataFrame, desk_id: str) -> tuple[str, list]:
    """Generate desk-level report with visualizations"""
    desk_df = df[df["desk_id"] == desk_id]
    
    if desk_df.empty:
        return f"No activities found for desk_id: {desk_id}", []
    
    # Calculate metrics
    total_activities = len(desk_df)
    feed_count = desk_df["feed_id"].nunique()
    avg_completion = desk_df["percent_complete"].mean()
    total_hours_spent = desk_df["hours_spent"].sum()
    
    # Feed-level aggregation
    feed_stats = desk_df.groupby("feed_id").agg({
        "percent_complete": "mean",
        "hours_spent": "sum",
        "id": "count"
    }).round(1)
    
    report = f"""
DESK REPORT - Desk ID: {desk_id}
{'=' * 60}

SUMMARY METRICS:
- Total Activities: {total_activities}
- Total Feeds: {feed_count}
- Average Completion: {avg_completion:.1f}%
- Total Hours Spent: {total_hours_spent:.1f}

FEED PERFORMANCE:
{feed_stats.to_string()}

PRIORITY vs COMPLEXITY MATRIX:
{pd.crosstab(desk_df['priority'], desk_df['complexity']).to_string()}
"""
    
    viz_paths = _create_visualizations(desk_df, "desk", desk_id)
    
    return report, viz_paths


def generate_org_report(df: pd.DataFrame, org_id: str) -> tuple[str, list]:
    """Generate org-level report with visualizations"""
    org_df = df[df["org_id"] == org_id]
    
    if org_df.empty:
        return f"No activities found for org_id: {org_id}", []
    
    # Calculate metrics
    total_activities = len(org_df)
    desk_count = org_df["desk_id"].nunique()
    feed_count = org_df["feed_id"].nunique()
    avg_completion = org_df["percent_complete"].mean()
    
    # Desk-level rollup
    desk_stats = org_df.groupby("desk_id").agg({
        "percent_complete": "mean",
        "hours_spent": "sum",
        "id": "count",
        "feed_id": "nunique"
    }).round(1)
    desk_stats.columns = ["Avg Completion %", "Hours Spent", "Activities", "Feeds"]
    
    report = f"""
ORGANIZATION REPORT - Org ID: {org_id}
{'=' * 60}

SUMMARY METRICS:
- Total Activities: {total_activities}
- Total Desks: {desk_count}
- Total Feeds: {feed_count}
- Overall Completion: {avg_completion:.1f}%

DESK PERFORMANCE:
{desk_stats.to_string()}

ORGANIZATION-WIDE INSIGHTS:
- High Priority Activities: {len(org_df[org_df['priority'] == 'high'])}
- Complex Activities: {len(org_df[org_df['complexity'] == 'complex'])}
- Overdue Activities: {len(org_df[org_df['end_date'] < datetime.now()])}
"""
    
    viz_paths = _create_visualizations(org_df, "org", org_id)
    
    return report, viz_paths


# ==================== VISUALIZATION ====================

def _create_visualizations(df: pd.DataFrame, level: str, entity_id: str) -> list[str]:
    """Create visualizations for a given dataset"""
    config = ReportConfig()
    viz_dir = Path(config.output_dir) / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_paths = []
    
    # 1. Progress by Priority
    fig, ax = plt.subplots(figsize=(10, 6))
    priority_order = ["low", "medium", "high"]
    priority_data = df.groupby("priority")["percent_complete"].mean().reindex(priority_order)
    priority_data.plot(kind="bar", ax=ax, color=["green", "orange", "red"])
    ax.set_title(f"Average Completion by Priority - {level.upper()} {entity_id}")
    ax.set_ylabel("Average Completion %")
    ax.set_xlabel("Priority")
    plt.xticks(rotation=0)
    plt.tight_layout()
    path1 = viz_dir / f"{level}_{entity_id}_priority_{timestamp}.png"
    plt.savefig(path1)
    plt.close()
    viz_paths.append(str(path1))
    
    # 2. Progress by Complexity
    fig, ax = plt.subplots(figsize=(10, 6))
    complexity_order = ["simple", "moderate", "complex"]
    complexity_data = df.groupby("complexity")["percent_complete"].mean().reindex(complexity_order)
    complexity_data.plot(kind="bar", ax=ax, color=["lightblue", "blue", "darkblue"])
    ax.set_title(f"Average Completion by Complexity - {level.upper()} {entity_id}")
    ax.set_ylabel("Average Completion %")
    ax.set_xlabel("Complexity")
    plt.xticks(rotation=0)
    plt.tight_layout()
    path2 = viz_dir / f"{level}_{entity_id}_complexity_{timestamp}.png"
    plt.savefig(path2)
    plt.close()
    viz_paths.append(str(path2))
    
    # 3. Priority vs Complexity Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    pivot = pd.crosstab(df["priority"], df["complexity"])
    pivot = pivot.reindex(priority_order, axis=0).reindex(complexity_order, axis=1, fill_value=0)
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", ax=ax)
    ax.set_title(f"Activity Count: Priority vs Complexity - {level.upper()} {entity_id}")
    plt.tight_layout()
    path3 = viz_dir / f"{level}_{entity_id}_heatmap_{timestamp}.png"
    plt.savefig(path3)
    plt.close()
    viz_paths.append(str(path3))
    
    return viz_paths


# ==================== REPORT GENERATION NODES ====================

def generate_canned_report(state: AgentState) -> AgentState:
    """Generate canned reports based on report type"""
    if state.get("error"):
        return state
    
    df = state["df"]
    report_type = state["report_type"]
    entity_id = state["entity_id"]
    
    try:
        if report_type == "feed":
            report, viz_paths = generate_feed_report(df, entity_id)
        elif report_type == "desk":
            report, viz_paths = generate_desk_report(df, entity_id)
        elif report_type == "org":
            report, viz_paths = generate_org_report(df, entity_id)
        else:
            state["error"] = f"Invalid report type: {report_type}"
            return state
        
        state["report_content"] = report
        state["visualization_paths"] = viz_paths
        
    except Exception as e:
        state["error"] = f"Report generation failed: {str(e)}"
    
    return state


def generate_custom_report(state: AgentState) -> AgentState:
    """Generate custom report using LLM"""
    if state.get("error"):
        return state
    
    df = state["df"]
    custom_query = state["custom_query"]
    config = ReportConfig()
    
    try:
        # Prepare data summary for LLM context
        data_summary = _prepare_data_summary(df)
        
        # Initialize LLM
        llm = ChatOpenAI(
            model=config.llm_model,
            temperature=config.llm_temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Construct prompt
        system_prompt = """You are a data analyst specializing in project management reporting.
You have access to activity data with organizational hierarchy (Org > Desk > Feed > Activity).
Analyze the data and provide clear, actionable insights based on the user's query.
Format your response as a professional report with sections and bullet points."""
        
        user_prompt = f"""
USER QUERY: {custom_query}

DATA SUMMARY:
{data_summary}

Analyze this data and provide a comprehensive report addressing the user's query.
Include specific metrics, trends, and recommendations where applicable.
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = llm.invoke(messages)
        state["report_content"] = response.content
        state["visualization_paths"] = []
        
    except Exception as e:
        state["error"] = f"Custom report generation failed: {str(e)}"
    
    return state


# ==================== HELPER FUNCTIONS ====================

def _format_dict(d: dict) -> str:
    """Format dictionary for report output"""
    return "\n".join([f"  - {k}: {v}" for k, v in d.items()])


def _format_activities(df: pd.DataFrame) -> str:
    """Format activities dataframe for report"""
    if df.empty:
        return "  None"
    return "\n".join([
        f"  - {row['title']}: {row['percent_complete']:.0f}% ({row['priority']} priority)"
        for _, row in df.iterrows()
    ])


def _prepare_data_summary(df: pd.DataFrame) -> str:
    """Prepare concise data summary for LLM context"""
    summary = f"""
Total Activities: {len(df)}
Organizations: {df['org_id'].nunique()}
Desks: {df['desk_id'].nunique()}
Feeds: {df['feed_id'].nunique()}

Completion Stats:
- Average: {df['percent_complete'].mean():.1f}%
- Min: {df['percent_complete'].min():.1f}%
- Max: {df['percent_complete'].max():.1f}%

Priority Distribution:
{df['priority'].value_counts().to_dict()}

Complexity Distribution:
{df['complexity'].value_counts().to_dict()}

Hours:
- Total Estimated: {df['estimated_hours'].sum():.1f}
- Total Spent: {df['hours_spent'].sum():.1f}
"""
    return summary


def route_report_type(state: AgentState) -> str:
    """Route to appropriate report generation node"""
    if state.get("error"):
        return "end"
    
    if state["report_type"] == "custom":
        return "custom"
    else:
        return "canned"


# ==================== GRAPH CONSTRUCTION ====================

def build_agent_graph() -> StateGraph:
    """Build LangGraph workflow"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("load_data", load_data)
    workflow.add_node("canned_report", generate_canned_report)
    workflow.add_node("custom_report", generate_custom_report)
    
    # Add edges
    workflow.set_entry_point("load_data")
    workflow.add_conditional_edges(
        "load_data",
        route_report_type,
        {
            "canned": "canned_report",
            "custom": "custom_report",
            "end": END
        }
    )
    workflow.add_edge("canned_report", END)
    workflow.add_edge("custom_report", END)
    
    return workflow.compile()


# ==================== PUBLIC API ====================

def run_report(
    csv_path: str,
    report_type: Literal["feed", "desk", "org", "custom"],
    entity_id: str | None = None,
    custom_query: str | None = None
) -> dict:
    """
    Run reporting agent
    
    Args:
        csv_path: Path to CSV file with activity data
        report_type: Type of report to generate
        entity_id: ID of entity (feed_id, desk_id, or org_id) for canned reports
        custom_query: Natural language query for custom reports
    
    Returns:
        dict with keys: report_content, visualization_paths, error
    """
    if report_type != "custom" and not entity_id:
        return {"error": "entity_id required for canned reports"}
    
    if report_type == "custom" and not custom_query:
        return {"error": "custom_query required for custom reports"}
    
    # Initialize state
    initial_state = {
        "csv_path": csv_path,
        "df": pd.DataFrame(),
        "report_type": report_type,
        "entity_id": entity_id,
        "custom_query": custom_query,
        "report_content": "",
        "visualization_paths": [],
        "error": None
    }
    
    # Run agent
    graph = build_agent_graph()
    final_state = graph.invoke(initial_state)
    
    return {
        "report_content": final_state["report_content"],
        "visualization_paths": final_state["visualization_paths"],
        "error": final_state["error"]
    }


# ==================== CLI ====================

def main():
    """Command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Activity Reporting Agent")
    parser.add_argument("csv_path", help="Path to CSV file")
    parser.add_argument(
        "report_type",
        choices=["feed", "desk", "org", "custom"],
        help="Type of report"
    )
    parser.add_argument("--entity-id", help="Entity ID (for feed/desk/org reports)")
    parser.add_argument("--query", help="Custom query (for custom reports)")
    
    args = parser.parse_args()
    
    print(f"\ncsv_path={args.csv_path}, report_type={args.report_type}, entity_id={args.entity_id}, query={args.query}\n\n") 
    result = run_report(
        csv_path=args.csv_path,
        report_type=args.report_type,
        entity_id=args.entity_id,
        custom_query=args.query
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
