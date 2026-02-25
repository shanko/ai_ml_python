# Quick Start Guide - Refactored Reporting Agent

## 5-Minute Quick Start

### Installation

```bash
# Navigate to the project directory
cd ai_ml_python/reporting_agent_refactored

# Install dependencies (if not already installed)
pip install -r ../requirements.txt
```

### Basic Usage

#### 1. Generate a Feed Report

```python
from reporting_agent_refactored import run_report

# Generate report (uses default production dependencies)
result = run_report(
    csv_path="../test.csv",
    report_type="feed",
    entity_id="F1"
)

# Check for errors
if result["error"]:
    print(f"Error: {result['error']}")
else:
    # Print report
    print(result["report_content"])
    
    # Show visualization paths
    print("\nVisualizations:")
    for path in result["visualization_paths"]:
        print(f"  - {path}")
```

#### 2. Generate a Custom Report (with LLM)

```python
import os
from reporting_agent_refactored import run_report

# Set API key
os.environ["OPENAI_API_KEY"] = "sk-your-key-here"

# Generate custom report
result = run_report(
    csv_path="../test.csv",
    report_type="custom",
    custom_query="What are the top 5 priorities and their completion status?"
)

print(result["report_content"])
```

#### 3. CLI Usage

```bash
# Feed report
python -m reporting_agent_refactored.reporting_agent_refactored \
    ../test.csv feed --entity-id F1

# Desk report
python -m reporting_agent_refactored.reporting_agent_refactored \
    ../test.csv desk --entity-id D1

# Organization report
python -m reporting_agent_refactored.reporting_agent_refactored \
    ../test.csv org --entity-id O1

# Custom report
export OPENAI_API_KEY="sk-..."
python -m reporting_agent_refactored.reporting_agent_refactored \
    ../test.csv custom --query "Analyze project completion trends"
```

---

## Testing (The Real Power!)

### Running Tests

```bash
# Run all tests
pytest test_reporting_agent.py -v

# Run specific test
pytest test_reporting_agent.py::test_calculate_feed_metrics_success -v

# Run with coverage
pytest test_reporting_agent.py --cov=reporting_agent_refactored --cov-report=html

# Open coverage report
open htmlcov/index.html
```

### Writing Your Own Tests

```python
import pytest
from reporting_agent_refactored import calculate_feed_metrics
from test_mocks import create_test_dataframe

def test_my_custom_scenario():
    # Create test data
    df = create_test_dataframe(50)
    
    # Test business logic (no I/O!)
    metrics = calculate_feed_metrics(df, "F1")
    
    # Assert
    assert metrics is not None
    assert metrics["total_activities"] > 0
    assert "avg_completion" in metrics
```

### Testing with Mocks (No External Dependencies!)

```python
from datetime import datetime
from reporting_agent_refactored import run_report, Dependencies
from test_mocks import (
    MockTimeProvider,
    MockFileSystem,
    MockLLM,
    MockVisualizationRenderer,
    MockConfig,
    create_test_dataframe
)

def test_complete_workflow():
    # Setup mocks
    mock_fs = MockFileSystem()
    test_df = create_test_dataframe(30)
    mock_fs.load_csv_data("test.csv", test_df)
    
    deps = Dependencies(
        time_provider=MockTimeProvider(datetime(2024, 1, 15)),
        file_system=mock_fs,
        llm=MockLLM("Test LLM response"),
        viz_renderer=MockVisualizationRenderer(),
        config=MockConfig()
    )
    
    # Run report with mocks (fast, no API calls!)
    result = run_report(
        csv_path="test.csv",
        report_type="feed",
        entity_id="F1",
        deps=deps
    )
    
    # Verify behavior
    assert result["error"] is None
    assert "FEED REPORT" in result["report_content"]
    assert len(result["visualization_paths"]) == 3
    
    # Verify mocks were called correctly
    assert len(mock_fs.get_saved_files()) == 3
    assert mock_fs.directories  # Directories were created
```

---

## Common Use Cases

### Use Case 1: Test Different Time Scenarios

```python
from datetime import datetime
from test_mocks import MockTimeProvider, create_test_dataframe
from reporting_agent_refactored import calculate_org_metrics
import pandas as pd

def test_overdue_activities():
    # Setup
    df = create_test_dataframe(20)
    df['org_id'] = 'O1'
    df['end_date'] = pd.to_datetime('2024-01-10')
    
    # Test: Time AFTER deadline
    time_after = MockTimeProvider(datetime(2024, 1, 15))
    metrics = calculate_org_metrics(df, 'O1', time_after)
    assert metrics['overdue_count'] == 20  # All overdue
    
    # Test: Time BEFORE deadline
    time_before = MockTimeProvider(datetime(2024, 1, 5))
    metrics = calculate_org_metrics(df, 'O1', time_before)
    assert metrics['overdue_count'] == 0  # None overdue
```

### Use Case 2: Test LLM Interaction

```python
from test_mocks import MockLLM
from reporting_agent_refactored import generate_custom_report_node, Dependencies

def test_llm_receives_correct_prompt():
    # Setup mock LLM
    mock_llm = MockLLM("Insightful analysis")
    deps = Dependencies(llm=mock_llm, ...)
    
    state = {
        "df": create_test_dataframe(),
        "custom_query": "What are the bottlenecks?",
        "error": None,
        # ... other state fields
    }
    
    # Execute
    result = generate_custom_report_node(state, deps)
    
    # Verify LLM was called
    assert mock_llm.get_call_count() == 1
    
    # Verify prompt contains user query
    messages = mock_llm.get_last_call()
    prompt_text = str(messages)
    assert "What are the bottlenecks?" in prompt_text
    assert "DATA SUMMARY" in prompt_text
```

### Use Case 3: Test Without File System

```python
from test_mocks import MockFileSystem, create_test_dataframe
from reporting_agent_refactored import load_data

def test_loading_invalid_csv():
    # Setup mock filesystem with bad data
    mock_fs = MockFileSystem()
    bad_df = pd.DataFrame({"id": [1, 2]})  # Missing required columns
    mock_fs.load_csv_data("bad.csv", bad_df)
    
    state = {
        "csv_path": "bad.csv",
        "df": pd.DataFrame(),
        "error": None,
        # ... other fields
    }
    
    # Execute
    result = load_data(state, mock_fs)
    
    # Verify error handling
    assert result["error"] is not None
    assert "Missing columns" in result["error"]
```

### Use Case 4: Parameterized Testing

```python
import pytest

@pytest.mark.parametrize("report_type,entity_id,expected_title", [
    ("feed", "F1", "FEED REPORT"),
    ("desk", "D1", "DESK REPORT"),
    ("org", "O1", "ORGANIZATION REPORT"),
])
def test_all_report_types(report_type, entity_id, expected_title, dependencies):
    # Setup test data
    test_df = create_multi_entity_dataframe()
    dependencies.file_system.load_csv_data("test.csv", test_df)
    
    # Run report
    result = run_report(
        csv_path="test.csv",
        report_type=report_type,
        entity_id=entity_id,
        deps=dependencies
    )
    
    # Verify
    assert result["error"] is None
    assert expected_title in result["report_content"]
```

---

## Advanced Configuration

### Custom Configuration

```python
from reporting_agent_refactored import (
    run_report,
    Dependencies,
    DefaultConfig,
    RealTimeProvider,
    RealFileSystem,
    OpenAILLM,
    MatplotlibRenderer
)

# Create custom configuration
config = DefaultConfig(
    llm_model="gpt-3.5-turbo",
    llm_temperature=0.3,
    output_dir="./my_custom_reports"
)

# Build dependencies with custom config
deps = Dependencies(
    time_provider=RealTimeProvider(),
    file_system=RealFileSystem(),
    llm=OpenAILLM(
        model=config.get_llm_model(),
        temperature=config.get_llm_temperature()
    ),
    viz_renderer=MatplotlibRenderer(),
    config=config
)

# Run with custom setup
result = run_report(
    csv_path="data.csv",
    report_type="feed",
    entity_id="F1",
    deps=deps
)
```

### Custom Implementations

You can create your own implementations:

```python
from reporting_agent_refactored.interfaces import LLMInterface

class MyCustomLLM(LLMInterface):
    """Custom LLM implementation"""
    
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
    
    def invoke(self, messages: list) -> Any:
        # Your custom logic
        response = requests.post(self.endpoint, json={"messages": messages})
        return response.json()

# Use custom implementation
deps = Dependencies(
    llm=MyCustomLLM("https://my-custom-llm-api.com"),
    # ... other dependencies
)
```

---

## Tips & Best Practices

### 1. Always Use Mocks in Tests
```python
# ❌ DON'T: Use real dependencies in tests
def test_something():
    result = run_report("real_file.csv", "feed", "F1")  # Slow, brittle

# ✅ DO: Use mocks
def test_something(dependencies):
    dependencies.file_system.load_csv_data("test.csv", test_df)
    result = run_report("test.csv", "feed", "F1", deps=dependencies)  # Fast!
```

### 2. Test Business Logic Separately
```python
# ✅ Test pure functions directly
def test_metrics_calculation():
    df = create_test_dataframe(20)
    metrics = calculate_feed_metrics(df, "F1")
    
    assert metrics["total_activities"] == 10
    assert metrics["avg_completion"] == pytest.approx(45.0)
```

### 3. Use Fixtures for Common Setup
```python
@pytest.fixture
def test_dataframe():
    return create_test_dataframe(30)

@pytest.fixture
def loaded_filesystem(test_dataframe):
    fs = MockFileSystem()
    fs.load_csv_data("test.csv", test_dataframe)
    return fs

def test_with_fixtures(loaded_filesystem):
    # Fixtures are automatically provided
    state = {"csv_path": "test.csv", ...}
    result = load_data(state, loaded_filesystem)
    assert result["error"] is None
```

### 4. Verify Mock Interactions
```python
def test_visualization_rendering(dependencies):
    # ... run report ...
    
    # Verify renderer was called correctly
    renderer = dependencies.viz_renderer
    assert renderer.get_chart_count() == 3
    assert len(renderer.get_charts_by_type("bar")) == 2
    assert len(renderer.get_charts_by_type("heatmap")) == 1
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError"
```bash
# Solution: Install from parent directory
cd ai_ml_python
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/ai_ml_python"
```

### Issue: "OpenAI API key not found"
```bash
# Solution: Set environment variable
export OPENAI_API_KEY="sk-your-key-here"

# Or in Python
import os
os.environ["OPENAI_API_KEY"] = "sk-your-key-here"
```

### Issue: Tests are slow
```python
# Make sure you're using mocks, not real dependencies!
# ❌ Slow
result = run_report("data.csv", "feed", "F1")  # Uses real filesystem

# ✅ Fast
result = run_report("data.csv", "feed", "F1", deps=mock_dependencies)
```

---

## Next Steps

1. **Read the README**: Full documentation in `README.md`
2. **Review Comparison**: See `REFACTORING_COMPARISON.md` for detailed changes
3. **Explore Tests**: Check `test_reporting_agent.py` for examples
4. **Write Your Tests**: Use `test_mocks.py` for test doubles

## Resources

- **Interfaces**: `interfaces.py` - Abstract contracts
- **Implementations**: `implementations.py` - Production code
- **Mocks**: `test_mocks.py` - Test doubles
- **Tests**: `test_reporting_agent.py` - Test examples
- **Main Module**: `reporting_agent_refactored.py` - Core logic

---

## Example: Complete Test File

```python
# test_my_feature.py
import pytest
from datetime import datetime
from reporting_agent_refactored import calculate_feed_metrics
from test_mocks import create_test_dataframe, MockTimeProvider

class TestFeedMetrics:
    """Test suite for feed metrics calculation"""
    
    def test_basic_metrics(self):
        """Test basic metric calculation"""
        df = create_test_dataframe(20)
        metrics = calculate_feed_metrics(df, "F1")
        
        assert metrics is not None
        assert metrics["total_activities"] == 10
        assert "avg_completion" in metrics
    
    def test_nonexistent_feed(self):
        """Test with feed that doesn't exist"""
        df = create_test_dataframe(20)
        metrics = calculate_feed_metrics(df, "NONEXISTENT")
        
        assert metrics is None
    
    def test_priority_distribution(self):
        """Test priority counting is correct"""
        df = create_test_dataframe(30)
        metrics = calculate_feed_metrics(df, "F1")
        
        priority_counts = metrics["priority_counts"]
        assert "low" in priority_counts
        assert "medium" in priority_counts
        assert "high" in priority_counts

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

Happy Testing! 🚀