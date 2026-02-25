#!/usr/bin/env python3
"""
Comprehensive unit tests for the refactored reporting agent.

This test suite demonstrates the improved testability through:
1. Testing with mocked dependencies (no external API calls)
2. Fast, isolated unit tests
3. Testing business logic separately from I/O
4. High code coverage
"""

import pytest
import pandas as pd
from datetime import datetime
from pathlib import Path

from reporting_agent_refactored import (
    AgentState,
    Dependencies,
    VisualizationData,
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
    prepare_feed_visualizations,
    prepare_desk_visualizations,
    prepare_org_visualizations,
    render_visualizations,
    generate_canned_report_node,
    generate_custom_report_node,
    route_report_type,
    build_agent_graph,
    run_report,
)

from test_mocks import (
    MockTimeProvider,
    MockFileSystem,
    MockLLM,
    MockVisualizationRenderer,
    MockConfig,
    create_test_dataframe,
    create_multi_entity_dataframe,
    create_empty_dataframe,
)


# ==================== FIXTURES ====================


@pytest.fixture
def mock_time_provider():
    """Fixture for mock time provider"""
    return MockTimeProvider(datetime(2024, 1, 15, 10, 0, 0))


@pytest.fixture
def mock_file_system():
    """Fixture for mock file system"""
    fs = MockFileSystem()
    # Load test data
    test_df = create_test_dataframe()
    fs.load_csv_data("test.csv", test_df)
    return fs


@pytest.fixture
def mock_llm():
    """Fixture for mock LLM"""
    return MockLLM("Test LLM Response with insights and recommendations.")


@pytest.fixture
def mock_viz_renderer():
    """Fixture for mock visualization renderer"""
    return MockVisualizationRenderer()


@pytest.fixture
def mock_config():
    """Fixture for mock configuration"""
    return MockConfig(output_dir="./test_output")


@pytest.fixture
def dependencies(
    mock_time_provider, mock_file_system, mock_llm, mock_viz_renderer, mock_config
):
    """Fixture for complete dependency container"""
    return Dependencies(
        time_provider=mock_time_provider,
        file_system=mock_file_system,
        llm=mock_llm,
        viz_renderer=mock_viz_renderer,
        config=mock_config,
    )


@pytest.fixture
def sample_dataframe():
    """Fixture for sample DataFrame"""
    return create_test_dataframe(20)


@pytest.fixture
def multi_entity_dataframe():
    """Fixture for multi-entity DataFrame"""
    return create_multi_entity_dataframe()


# ==================== TEST DATA LOADING ====================


def test_load_data_success(mock_file_system):
    """Test successful data loading"""
    state = {
        "csv_path": "test.csv",
        "df": pd.DataFrame(),
        "report_type": "feed",
        "entity_id": "F1",
        "custom_query": None,
        "report_content": "",
        "visualization_paths": [],
        "error": None,
    }

    result = load_data(state, mock_file_system)

    assert result["error"] is None
    assert not result["df"].empty
    assert len(result["df"]) == 10
    assert "priority" in result["df"].columns


def test_load_data_file_not_found(mock_file_system):
    """Test data loading with missing file"""
    state = {
        "csv_path": "nonexistent.csv",
        "df": pd.DataFrame(),
        "report_type": "feed",
        "entity_id": "F1",
        "custom_query": None,
        "report_content": "",
        "visualization_paths": [],
        "error": None,
    }

    result = load_data(state, mock_file_system)

    assert result["error"] is not None
    assert "Failed to load data" in result["error"]


def test_load_data_missing_columns():
    """Test data loading with missing required columns"""
    fs = MockFileSystem()
    incomplete_df = pd.DataFrame({"id": [1, 2], "feed_id": ["F1", "F2"]})
    fs.load_csv_data("incomplete.csv", incomplete_df)

    state = {
        "csv_path": "incomplete.csv",
        "df": pd.DataFrame(),
        "report_type": "feed",
        "entity_id": "F1",
        "custom_query": None,
        "report_content": "",
        "visualization_paths": [],
        "error": None,
    }

    result = load_data(state, fs)

    assert result["error"] is not None
    assert "Missing columns" in result["error"]


def test_load_data_normalizes_columns(mock_file_system):
    """Test that data loading normalizes priority and complexity"""
    state = {
        "csv_path": "test.csv",
        "df": pd.DataFrame(),
        "report_type": "feed",
        "entity_id": "F1",
        "custom_query": None,
        "report_content": "",
        "visualization_paths": [],
        "error": None,
    }

    result = load_data(state, mock_file_system)

    # Check that priority and complexity are lowercase
    assert all(result["df"]["priority"].str.islower())
    assert all(result["df"]["complexity"].str.islower())


# ==================== TEST METRIC CALCULATIONS ====================


def test_calculate_feed_metrics_success(sample_dataframe):
    """Test successful feed metrics calculation"""
    metrics = calculate_feed_metrics(sample_dataframe, "F1")

    assert metrics is not None
    assert metrics["total_activities"] == 10
    assert "avg_completion" in metrics
    assert "total_hours_spent" in metrics
    assert "priority_counts" in metrics
    assert isinstance(metrics["priority_counts"], dict)


def test_calculate_feed_metrics_no_data(sample_dataframe):
    """Test feed metrics with non-existent feed"""
    metrics = calculate_feed_metrics(sample_dataframe, "NONEXISTENT")

    assert metrics is None


def test_calculate_feed_metrics_accuracy(sample_dataframe):
    """Test accuracy of feed metric calculations"""
    metrics = calculate_feed_metrics(sample_dataframe, "F1")

    feed_df = sample_dataframe[sample_dataframe["feed_id"] == "F1"]
    expected_avg = feed_df["percent_complete"].mean()
    expected_hours = feed_df["hours_spent"].sum()

    assert metrics["avg_completion"] == expected_avg
    assert metrics["total_hours_spent"] == expected_hours


def test_calculate_desk_metrics_success(multi_entity_dataframe):
    """Test successful desk metrics calculation"""
    metrics = calculate_desk_metrics(multi_entity_dataframe, "D1")

    assert metrics is not None
    assert metrics["total_activities"] == 20
    assert metrics["feed_count"] == 2
    assert "feed_stats" in metrics
    assert "priority_complexity_matrix" in metrics


def test_calculate_desk_metrics_no_data(sample_dataframe):
    """Test desk metrics with non-existent desk"""
    metrics = calculate_desk_metrics(sample_dataframe, "NONEXISTENT")

    assert metrics is None


def test_calculate_org_metrics_success(multi_entity_dataframe, mock_time_provider):
    """Test successful org metrics calculation"""
    metrics = calculate_org_metrics(multi_entity_dataframe, "O1", mock_time_provider)

    assert metrics is not None
    assert metrics["total_activities"] == 20
    assert metrics["desk_count"] == 2
    assert metrics["feed_count"] == 3
    assert "overdue_count" in metrics


def test_calculate_org_metrics_overdue_count(mock_time_provider):
    """Test overdue count calculation in org metrics"""
    # Create data with some overdue activities
    df = create_test_dataframe(10)
    df["org_id"] = "O1"
    df["end_date"] = pd.to_datetime("2024-01-10")  # Before mock time (2024-01-15)

    metrics = calculate_org_metrics(df, "O1", mock_time_provider)

    assert metrics["overdue_count"] == 10


def test_calculate_org_metrics_no_overdue(mock_time_provider):
    """Test org metrics with no overdue activities"""
    df = create_test_dataframe(10)
    df["org_id"] = "O1"
    df["end_date"] = pd.to_datetime("2024-02-01")  # After mock time

    metrics = calculate_org_metrics(df, "O1", mock_time_provider)

    assert metrics["overdue_count"] == 0


# ==================== TEST VISUALIZATION PREPARATION ====================


def test_prepare_feed_visualizations(sample_dataframe):
    """Test feed visualization preparation"""
    feed_df = sample_dataframe[sample_dataframe["feed_id"] == "F1"]
    viz_specs = prepare_feed_visualizations(feed_df, "F1")

    assert len(viz_specs) == 3
    assert viz_specs[0].chart_type == "bar"
    assert viz_specs[1].chart_type == "bar"
    assert viz_specs[2].chart_type == "heatmap"
    assert "FEED F1" in viz_specs[0].title


def test_prepare_desk_visualizations(sample_dataframe):
    """Test desk visualization preparation"""
    desk_df = sample_dataframe[sample_dataframe["desk_id"] == "D1"]
    viz_specs = prepare_desk_visualizations(desk_df, "D1")

    assert len(viz_specs) == 3
    assert all(isinstance(spec, VisualizationData) for spec in viz_specs)
    assert "DESK D1" in viz_specs[0].title


def test_prepare_org_visualizations(sample_dataframe):
    """Test org visualization preparation"""
    org_df = sample_dataframe[sample_dataframe["org_id"] == "O1"]
    viz_specs = prepare_org_visualizations(org_df, "O1")

    assert len(viz_specs) == 3
    assert viz_specs[0].colors == ["green", "orange", "red"]  # Priority colors
    assert viz_specs[1].colors == ["lightblue", "blue", "darkblue"]  # Complexity colors


# ==================== TEST VISUALIZATION RENDERING ====================


def test_render_visualizations(
    sample_dataframe, mock_viz_renderer, mock_file_system, mock_config
):
    """Test visualization rendering"""
    feed_df = sample_dataframe[sample_dataframe["feed_id"] == "F1"]
    viz_specs = prepare_feed_visualizations(feed_df, "F1")

    paths = render_visualizations(
        viz_specs=viz_specs,
        level="feed",
        entity_id="F1",
        renderer=mock_viz_renderer,
        file_system=mock_file_system,
        config=mock_config,
    )

    assert len(paths) == 3
    assert mock_viz_renderer.get_chart_count() == 3
    assert len(mock_viz_renderer.closed_figures) == 3
    assert mock_config.get_visualization_dir() in mock_file_system.directories


def test_render_visualizations_creates_directory(
    sample_dataframe, mock_viz_renderer, mock_file_system, mock_config
):
    """Test that rendering creates visualization directory"""
    feed_df = sample_dataframe[sample_dataframe["feed_id"] == "F1"]
    viz_specs = prepare_feed_visualizations(feed_df, "F1")

    render_visualizations(
        viz_specs, "feed", "F1", mock_viz_renderer, mock_file_system, mock_config
    )

    expected_dir = mock_config.get_visualization_dir()
    assert expected_dir in mock_file_system.directories


# ==================== TEST REPORT TEXT GENERATION ====================


def test_generate_feed_report_text(sample_dataframe):
    """Test feed report text generation"""
    metrics = calculate_feed_metrics(sample_dataframe, "F1")
    report = generate_feed_report_text(metrics)

    assert "FEED REPORT" in report
    assert "Feed ID: F1" in report
    assert "Total Activities:" in report
    assert "PRIORITY DISTRIBUTION:" in report
    assert "COMPLEXITY DISTRIBUTION:" in report


def test_generate_desk_report_text(multi_entity_dataframe):
    """Test desk report text generation"""
    metrics = calculate_desk_metrics(multi_entity_dataframe, "D1")
    report = generate_desk_report_text(metrics)

    assert "DESK REPORT" in report
    assert "Desk ID: D1" in report
    assert "Total Feeds:" in report
    assert "FEED PERFORMANCE:" in report


def test_generate_org_report_text(multi_entity_dataframe, mock_time_provider):
    """Test org report text generation"""
    metrics = calculate_org_metrics(multi_entity_dataframe, "O1", mock_time_provider)
    report = generate_org_report_text(metrics)

    assert "ORGANIZATION REPORT" in report
    assert "Org ID: O1" in report
    assert "Total Desks:" in report
    assert "ORGANIZATION-WIDE INSIGHTS:" in report
    assert "Overdue Activities:" in report


# ==================== TEST UTILITY FUNCTIONS ====================


def test_format_dict():
    """Test dictionary formatting"""
    test_dict = {"low": 5, "medium": 10, "high": 3}
    result = format_dict(test_dict)

    assert "- low: 5" in result
    assert "- medium: 10" in result
    assert "- high: 3" in result


def test_format_activities_with_data(sample_dataframe):
    """Test activity formatting with data"""
    df_subset = sample_dataframe.head(3)[["title", "percent_complete", "priority"]]
    result = format_activities(df_subset)

    assert "Task 1" in result
    assert "priority" in result


def test_format_activities_empty():
    """Test activity formatting with empty DataFrame"""
    empty_df = pd.DataFrame(columns=["title", "percent_complete", "priority"])
    result = format_activities(empty_df)

    assert result == "  None"


def test_prepare_data_summary(sample_dataframe):
    """Test data summary preparation"""
    summary = prepare_data_summary(sample_dataframe)

    assert "Total Activities: 20" in summary
    assert "Organizations:" in summary
    assert "Completion Stats:" in summary
    assert "Priority Distribution:" in summary
    assert "Complexity Distribution:" in summary


# ==================== TEST GRAPH NODES ====================


def test_generate_canned_report_feed(dependencies, sample_dataframe):
    """Test canned report generation for feed"""
    state = {
        "csv_path": "test.csv",
        "df": sample_dataframe,
        "report_type": "feed",
        "entity_id": "F1",
        "custom_query": None,
        "report_content": "",
        "visualization_paths": [],
        "error": None,
    }

    result = generate_canned_report_node(state, dependencies)

    assert result["error"] is None
    assert result["report_content"] != ""
    assert "FEED REPORT" in result["report_content"]
    assert len(result["visualization_paths"]) == 3


def test_generate_canned_report_desk(dependencies, multi_entity_dataframe):
    """Test canned report generation for desk"""
    state = {
        "csv_path": "test.csv",
        "df": multi_entity_dataframe,
        "report_type": "desk",
        "entity_id": "D1",
        "custom_query": None,
        "report_content": "",
        "visualization_paths": [],
        "error": None,
    }

    result = generate_canned_report_node(state, dependencies)

    assert result["error"] is None
    assert "DESK REPORT" in result["report_content"]


def test_generate_canned_report_org(dependencies, multi_entity_dataframe):
    """Test canned report generation for org"""
    state = {
        "csv_path": "test.csv",
        "df": multi_entity_dataframe,
        "report_type": "org",
        "entity_id": "O1",
        "custom_query": None,
        "report_content": "",
        "visualization_paths": [],
        "error": None,
    }

    result = generate_canned_report_node(state, dependencies)

    assert result["error"] is None
    assert "ORGANIZATION REPORT" in result["report_content"]


def test_generate_canned_report_no_data(dependencies, sample_dataframe):
    """Test canned report with non-existent entity"""
    state = {
        "csv_path": "test.csv",
        "df": sample_dataframe,
        "report_type": "feed",
        "entity_id": "NONEXISTENT",
        "custom_query": None,
        "report_content": "",
        "visualization_paths": [],
        "error": None,
    }

    result = generate_canned_report_node(state, dependencies)

    assert result["error"] is not None
    assert "No activities found" in result["error"]


def test_generate_canned_report_with_existing_error(dependencies):
    """Test that canned report skips when error exists"""
    state = {
        "csv_path": "test.csv",
        "df": pd.DataFrame(),
        "report_type": "feed",
        "entity_id": "F1",
        "custom_query": None,
        "report_content": "",
        "visualization_paths": [],
        "error": "Previous error",
    }

    result = generate_canned_report_node(state, dependencies)

    assert result["error"] == "Previous error"
    assert result["report_content"] == ""


def test_generate_custom_report_success(dependencies, sample_dataframe):
    """Test custom report generation"""
    state = {
        "csv_path": "test.csv",
        "df": sample_dataframe,
        "report_type": "custom",
        "entity_id": None,
        "custom_query": "What are the top priorities?",
        "report_content": "",
        "visualization_paths": [],
        "error": None,
    }

    result = generate_custom_report_node(state, dependencies)

    assert result["error"] is None
    assert result["report_content"] != ""
    assert dependencies.llm.get_call_count() == 1


def test_generate_custom_report_no_llm(sample_dataframe):
    """Test custom report without LLM configured"""
    deps = Dependencies(
        time_provider=MockTimeProvider(),
        file_system=MockFileSystem(),
        llm=None,  # No LLM
        viz_renderer=MockVisualizationRenderer(),
        config=MockConfig(),
    )

    state = {
        "csv_path": "test.csv",
        "df": sample_dataframe,
        "report_type": "custom",
        "entity_id": None,
        "custom_query": "What are the top priorities?",
        "report_content": "",
        "visualization_paths": [],
        "error": None,
    }

    result = generate_custom_report_node(state, deps)

    assert result["error"] is not None
    assert "LLM not configured" in result["error"]


def test_generate_custom_report_with_existing_error(dependencies):
    """Test that custom report skips when error exists"""
    state = {
        "csv_path": "test.csv",
        "df": pd.DataFrame(),
        "report_type": "custom",
        "entity_id": None,
        "custom_query": "Query",
        "report_content": "",
        "visualization_paths": [],
        "error": "Previous error",
    }

    result = generate_custom_report_node(state, dependencies)

    assert result["error"] == "Previous error"
    assert dependencies.llm.get_call_count() == 0


# ==================== TEST ROUTING ====================


def test_route_report_type_canned():
    """Test routing for canned reports"""
    state = {
        "csv_path": "test.csv",
        "df": pd.DataFrame(),
        "report_type": "feed",
        "entity_id": "F1",
        "custom_query": None,
        "report_content": "",
        "visualization_paths": [],
        "error": None,
    }

    result = route_report_type(state)
    assert result == "canned"


def test_route_report_type_custom():
    """Test routing for custom reports"""
    state = {
        "csv_path": "test.csv",
        "df": pd.DataFrame(),
        "report_type": "custom",
        "entity_id": None,
        "custom_query": "Query",
        "report_content": "",
        "visualization_paths": [],
        "error": None,
    }

    result = route_report_type(state)
    assert result == "custom"


def test_route_report_type_with_error():
    """Test routing when error exists"""
    state = {
        "csv_path": "test.csv",
        "df": pd.DataFrame(),
        "report_type": "feed",
        "entity_id": "F1",
        "custom_query": None,
        "report_content": "",
        "visualization_paths": [],
        "error": "Some error",
    }

    result = route_report_type(state)
    assert result == "end"


# ==================== TEST GRAPH CONSTRUCTION ====================


def test_build_agent_graph(dependencies):
    """Test agent graph construction"""
    graph = build_agent_graph(dependencies)

    assert graph is not None
    # Graph should be compiled and ready to invoke


def test_graph_execution_feed_report(dependencies):
    """Test full graph execution for feed report"""
    # Setup test data
    test_df = create_test_dataframe()
    dependencies.file_system.load_csv_data("test.csv", test_df)

    initial_state = {
        "csv_path": "test.csv",
        "df": pd.DataFrame(),
        "report_type": "feed",
        "entity_id": "F1",
        "custom_query": None,
        "report_content": "",
        "visualization_paths": [],
        "error": None,
    }

    graph = build_agent_graph(dependencies)
    result = graph.invoke(initial_state)

    assert result["error"] is None
    assert result["report_content"] != ""
    assert "FEED REPORT" in result["report_content"]


def test_graph_execution_custom_report(dependencies):
    """Test full graph execution for custom report"""
    test_df = create_test_dataframe()
    dependencies.file_system.load_csv_data("test.csv", test_df)

    initial_state = {
        "csv_path": "test.csv",
        "df": pd.DataFrame(),
        "report_type": "custom",
        "entity_id": None,
        "custom_query": "Analyze the project status",
        "report_content": "",
        "visualization_paths": [],
        "error": None,
    }

    graph = build_agent_graph(dependencies)
    result = graph.invoke(initial_state)

    assert result["error"] is None
    assert result["report_content"] != ""


# ==================== TEST PUBLIC API ====================


def test_run_report_validation_missing_entity_id():
    """Test run_report validates required entity_id"""
    result = run_report(
        csv_path="test.csv", report_type="feed", entity_id=None, deps=None
    )

    assert "error" in result
    assert "entity_id required" in result["error"]


def test_run_report_validation_missing_custom_query():
    """Test run_report validates required custom_query"""
    result = run_report(
        csv_path="test.csv", report_type="custom", custom_query=None, deps=None
    )

    assert "error" in result
    assert "custom_query required" in result["error"]


def test_run_report_with_dependencies(dependencies):
    """Test run_report with provided dependencies"""
    test_df = create_test_dataframe()
    dependencies.file_system.load_csv_data("test.csv", test_df)

    result = run_report(
        csv_path="test.csv",
        report_type="feed",
        entity_id="F1",
        custom_query=None,
        deps=dependencies,
    )

    assert result["error"] is None
    assert result["report_content"] != ""
    assert "FEED REPORT" in result["report_content"]
    assert len(result["visualization_paths"]) == 3


def test_run_report_desk_type(dependencies):
    """Test run_report for desk report"""
    test_df = create_multi_entity_dataframe()
    dependencies.file_system.load_csv_data("test.csv", test_df)

    result = run_report(
        csv_path="test.csv",
        report_type="desk",
        entity_id="D1",
        deps=dependencies,
    )

    assert result["error"] is None
    assert "DESK REPORT" in result["report_content"]


def test_run_report_org_type(dependencies):
    """Test run_report for org report"""
    test_df = create_multi_entity_dataframe()
    dependencies.file_system.load_csv_data("test.csv", test_df)

    result = run_report(
        csv_path="test.csv",
        report_type="org",
        entity_id="O1",
        deps=dependencies,
    )

    assert result["error"] is None
    assert "ORGANIZATION REPORT" in result["report_content"]


def test_run_report_custom_type(dependencies):
    """Test run_report for custom report"""
    test_df = create_test_dataframe()
    dependencies.file_system.load_csv_data("test.csv", test_df)

    result = run_report(
        csv_path="test.csv",
        report_type="custom",
        custom_query="What is the project status?",
        deps=dependencies,
    )

    assert result["error"] is None
    assert result["report_content"] != ""


def test_run_report_handles_file_error(dependencies):
    """Test run_report handles file loading errors"""
    result = run_report(
        csv_path="nonexistent.csv",
        report_type="feed",
        entity_id="F1",
        deps=dependencies,
    )

    assert result["error"] is not None


# ==================== TEST EDGE CASES ====================


def test_empty_dataframe_handling():
    """Test handling of empty DataFrame"""
    empty_df = create_empty_dataframe()
    metrics = calculate_feed_metrics(empty_df, "F1")

    assert metrics is None


def test_metrics_with_zero_estimated_hours(sample_dataframe):
    """Test metrics calculation with zero estimated hours"""
    df = sample_dataframe.copy()
    df["estimated_hours"] = 0.0

    metrics = calculate_feed_metrics(df, "F1")
    report = generate_feed_report_text(metrics)

    # Should handle division by zero gracefully
    assert "Hours Efficiency: 0.0%" in report or "inf" not in report.lower()


def test_visualization_with_missing_categories(sample_dataframe):
    """Test visualization preparation with incomplete categories"""
    # Create data with only one priority level
    df = sample_dataframe.copy()
    df = df[df["priority"] == "high"]

    viz_specs = prepare_feed_visualizations(df, "F1")

    # Should still create visualizations
    assert len(viz_specs) == 3


# ==================== TEST MOCK BEHAVIOR ====================


def test_mock_time_provider():
    """Test mock time provider behavior"""
    fixed_time = datetime(2024, 6, 15, 14, 30, 0)
    provider = MockTimeProvider(fixed_time)

    assert provider.now() == fixed_time

    # Test updating time
    new_time = datetime(2024, 7, 1, 10, 0, 0)
    provider.set_time(new_time)
    assert provider.now() == new_time


def test_mock_file_system_tracking():
    """Test mock file system tracks operations"""
    fs = MockFileSystem()

    # Test directory creation tracking
    test_path = Path("test/dir")
    fs.mkdir(test_path)
    assert test_path in fs.directories

    # Test file saving tracking
    fs.save_figure("mock_figure", Path("test.png"))
    assert str(Path("test.png")) in fs.get_saved_files()


def test_mock_llm_call_tracking():
    """Test mock LLM tracks invocations"""
    llm = MockLLM("Test response")

    # First call
    llm.invoke(["message 1"])
    assert llm.get_call_count() == 1

    # Second call
    llm.invoke(["message 2"])
    assert llm.get_call_count() == 2

    # Check last call
    last_call = llm.get_last_call()
    assert last_call == ["message 2"]


def test_mock_visualization_renderer_tracking():
    """Test mock renderer tracks chart creation"""
    renderer = MockVisualizationRenderer()

    # Create bar chart
    data = pd.Series([1, 2, 3])
    renderer.create_bar_chart(data, "Test", "X", "Y", ["blue"])

    assert renderer.get_chart_count() == 1
    assert len(renderer.get_charts_by_type("bar")) == 1

    # Create heatmap
    heatmap_data = pd.DataFrame([[1, 2], [3, 4]])
    renderer.create_heatmap(heatmap_data, "Test Heatmap")

    assert renderer.get_chart_count() == 2
    assert len(renderer.get_charts_by_type("heatmap")) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
