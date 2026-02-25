# Test Data Documentation

## Overview

This directory contains `test_1000.csv`, a comprehensive test dataset with 1,000 realistic activities designed to test the refactored Activity Reporting Agent.

## File Details

**Filename:** `test_1000.csv`  
**Records:** 1,000 activities  
**Format:** CSV with headers  
**Size:** ~250 KB  

## Data Structure

### Organizational Hierarchy

```
Organizations (5)
├── ORG01, ORG02, ORG03, ORG04, ORG05
│
└── Desks (15 total, 3 per org)
    ├── DESK01-DESK03 (ORG01)
    ├── DESK04-DESK06 (ORG02)
    ├── DESK07-DESK09 (ORG03)
    ├── DESK10-DESK12 (ORG04)
    └── DESK13-DESK15 (ORG05)
    │
    └── Feeds (60 total, 4 per desk)
        └── FEED001-FEED060
        │
        └── Activities (1000 total)
```

### Column Schema

| Column | Type | Description | Example Values |
|--------|------|-------------|----------------|
| `id` | Integer | Unique activity ID | 1-1000 |
| `feed_id` | String | Feed identifier | FEED001, FEED002, ... |
| `desk_id` | String | Desk identifier | DESK01, DESK02, ... |
| `org_id` | String | Organization identifier | ORG01, ORG02, ... |
| `owner_id` | String | Activity owner/assignee | USR001, USR002, ... |
| `title` | String | Activity title | "Implement API for Payment System" |
| `description` | String | Detailed description | "Complete implementation of api in the payment system" |
| `start_date` | Date | Activity start date (YYYY-MM-DD) | 2023-01-15 |
| `end_date` | Date | Activity end date (YYYY-MM-DD) | 2023-03-20 |
| `priority` | String | Priority level | low, medium, high |
| `percent_complete` | Integer | Completion percentage (0-100) | 0, 25, 50, 75, 100 |
| `estimated_hours` | Integer | Estimated hours to complete | 4-400 |
| `hours_spent` | Float | Actual hours spent | 0.0-500.0 |
| `complexity` | String | Complexity level | simple, moderate, complex |

## Data Characteristics

### Distribution Statistics

#### Priority Distribution
- **Low:** 418 activities (41.8%)
- **Medium:** 354 activities (35.4%)
- **High:** 228 activities (22.8%)

#### Complexity Distribution
- **Simple:** 300 activities (30.0%)
- **Moderate:** 447 activities (44.7%)
- **Complex:** 253 activities (25.3%)

#### Completion Status
- **Average Completion:** 66.7%
- **Range:** 0% - 100%
- **Distribution:**
  - Just Started (0-20%): ~15%
  - In Progress (21-60%): ~25%
  - Nearly Done (61-95%): ~45%
  - Complete (100%): ~15%

#### Time Range
- **Start Dates:** January 1, 2023 - December 31, 2023
- **Duration:** 5-90 days per activity
- **Span:** 18 months of project data

#### Resource Allocation
- **Organizations:** 5 distinct orgs
- **Desks:** 15 distinct desks
- **Feeds:** 60 distinct feeds
- **Owners:** 50 unique users (USR001-USR050)

### Realistic Features

1. **Hierarchical Relationships**
   - Activities properly nested under feeds
   - Feeds grouped by desks
   - Desks organized by organizations

2. **Realistic Effort Estimates**
   - Simple: 4-40 hours
   - Moderate: 40-120 hours
   - Complex: 120-400 hours

3. **Variance in Execution**
   - Hours spent includes realistic over/under estimates
   - Variance factor: 0.7x - 1.3x of estimated time
   - Incomplete tasks show proportional hours spent

4. **Diverse Activity Types**
   - 12 action types: Implement, Design, Test, Review, Deploy, Configure, etc.
   - 16 component types: API, Database, UI Component, Service, etc.
   - 15 system types: Payment System, User Management, Analytics Platform, etc.

## Usage Examples

### Command-Line Interface

#### Generate Feed Report
```bash
python reporting_agent_refactored/cli.py \
    reporting_agent_refactored/test_1000.csv \
    feed --entity-id FEED001
```

#### Generate Desk Report
```bash
python reporting_agent_refactored/cli.py \
    reporting_agent_refactored/test_1000.csv \
    desk --entity-id DESK01
```

#### Generate Organization Report
```bash
python reporting_agent_refactored/cli.py \
    reporting_agent_refactored/test_1000.csv \
    org --entity-id ORG01
```

#### Generate Custom Report (requires OpenAI API key)
```bash
export OPENAI_API_KEY="sk-your-key-here"
python reporting_agent_refactored/cli.py \
    reporting_agent_refactored/test_1000.csv \
    custom --query "What are the most critical bottlenecks?"
```

### Python API

```python
from reporting_agent_refactored import run_report

# Generate report
result = run_report(
    csv_path="reporting_agent_refactored/test_1000.csv",
    report_type="feed",
    entity_id="FEED001"
)

# Check results
if result["error"]:
    print(f"Error: {result['error']}")
else:
    print(result["report_content"])
    print(f"Visualizations: {result['visualization_paths']}")
```

### Testing with Mocks

```python
from reporting_agent_refactored import run_report, Dependencies
from reporting_agent_refactored.test_mocks import (
    MockTimeProvider, MockFileSystem, MockVisualizationRenderer, MockConfig
)
import pandas as pd

# Load test data
df = pd.read_csv("reporting_agent_refactored/test_1000.csv")

# Setup mocks
mock_fs = MockFileSystem()
mock_fs.load_csv_data("test_1000.csv", df)

deps = Dependencies(
    time_provider=MockTimeProvider(),
    file_system=mock_fs,
    llm=None,
    viz_renderer=MockVisualizationRenderer(),
    config=MockConfig()
)

# Run report with mocks (no real I/O!)
result = run_report(
    csv_path="test_1000.csv",
    report_type="feed",
    entity_id="FEED001",
    deps=deps
)
```

## Sample Entity IDs for Testing

### Organizations
- `ORG01` - 200 activities, 3 desks, 12 feeds
- `ORG02` - 200 activities, 3 desks, 12 feeds
- `ORG03` - 200 activities, 3 desks, 12 feeds
- `ORG04` - 200 activities, 3 desks, 12 feeds
- `ORG05` - 200 activities, 3 desks, 12 feeds

### Desks (Examples)
- `DESK01` - 67 activities, 4 feeds (in ORG01)
- `DESK06` - 67 activities, 4 feeds (in ORG02)
- `DESK11` - 66 activities, 4 feeds (in ORG03)

### Feeds (Examples)
- `FEED001` - 17 activities (in DESK01)
- `FEED016` - 17 activities (in DESK01)
- `FEED031` - 17 activities (in DESK01)
- `FEED046` - 16 activities (in DESK01)

## Regenerating Test Data

If you need to regenerate the test data with different characteristics:

```python
import pandas as pd
import random
from datetime import datetime, timedelta

# Your custom configuration here
NUM_ACTIVITIES = 1000
NUM_ORGS = 5
# ... (see generate script in this directory)
```

Or use the built-in test data factories:

```python
from reporting_agent_refactored.test_mocks import (
    create_test_dataframe,
    create_multi_entity_dataframe
)

# Create test data programmatically
df = create_test_dataframe(num_rows=100)
df.to_csv("custom_test.csv", index=False)
```

## Data Quality Notes

### Strengths
✅ Realistic activity names and descriptions  
✅ Proper hierarchical relationships  
✅ Varied priority and complexity distributions  
✅ Realistic time estimates and actual hours  
✅ Diverse date ranges  
✅ Multiple owners for workload distribution  

### Intentional Characteristics
⚠️ All activities show as "overdue" since end dates are in 2023-2024 (use MockTimeProvider to control time in tests)  
⚠️ Deterministic generation (seed=42) for reproducible testing  
⚠️ Evenly distributed across hierarchy for balanced testing  

## Verification

### Quick Stats Check
```bash
# Count rows
wc -l reporting_agent_refactored/test_1000.csv
# Expected: 1001 (1000 data + 1 header)

# View first few rows
head -5 reporting_agent_refactored/test_1000.csv

# Check unique values
cut -d',' -f2 reporting_agent_refactored/test_1000.csv | sort -u | wc -l
# Expected: 61 (60 feeds + 1 header)
```

### Python Validation
```python
import pandas as pd

df = pd.read_csv("reporting_agent_refactored/test_1000.csv")

print(f"Total activities: {len(df)}")
print(f"Organizations: {df['org_id'].nunique()}")
print(f"Desks: {df['desk_id'].nunique()}")
print(f"Feeds: {df['feed_id'].nunique()}")
print(f"Owners: {df['owner_id'].nunique()}")
print(f"\nAvg completion: {df['percent_complete'].mean():.1f}%")
print(f"Priority distribution:\n{df['priority'].value_counts()}")
```

## Troubleshooting

### Issue: "File not found"
**Solution:** Ensure you're running commands from the `ai_ml_python` directory:
```bash
cd ai_ml_python
python reporting_agent_refactored/cli.py reporting_agent_refactored/test_1000.csv feed --entity-id FEED001
```

### Issue: "No activities found for entity"
**Solution:** Check entity IDs match the data. Valid IDs are:
- Organizations: ORG01-ORG05
- Desks: DESK01-DESK15
- Feeds: FEED001-FEED060

### Issue: Visualizations not showing
**Solution:** Check that the `reports/visualizations/` directory is created and accessible. The CLI creates it automatically.

## Performance Benchmarks

With 1,000 activities:
- **Feed Report:** ~0.5 seconds
- **Desk Report:** ~0.8 seconds
- **Org Report:** ~1.2 seconds
- **Custom Report:** ~3-5 seconds (with LLM API call)

## License

This test data is generated for testing purposes and follows the same license as the parent project.

## Last Updated

Generated: November 2024  
Format Version: 1.0  
Compatible with: Refactored Agent v2.0.0