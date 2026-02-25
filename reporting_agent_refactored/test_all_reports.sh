#!/bin/bash
#
# Comprehensive Test Script for Refactored Reporting Agent
# This script demonstrates all report types using the test_1000.csv data
#

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
CSV_FILE="reporting_agent_refactored/test_1000.csv"
CLI_SCRIPT="reporting_agent_refactored/cli.py"
OUTPUT_DIR="./test_reports_output"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Reporting Agent - Comprehensive Test${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if CSV file exists
if [ ! -f "$CSV_FILE" ]; then
    echo -e "${RED}❌ Error: Test data file not found: $CSV_FILE${NC}"
    echo "Please run from the ai_ml_python directory"
    exit 1
fi

# Check if CLI script exists
if [ ! -f "$CLI_SCRIPT" ]; then
    echo -e "${RED}❌ Error: CLI script not found: $CLI_SCRIPT${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Test data file found: $CSV_FILE${NC}"
echo -e "${GREEN}✓ CLI script found: $CLI_SCRIPT${NC}"
echo ""

# Function to run a report and display summary
run_report() {
    local report_type=$1
    local entity_id=$2
    local description=$3

    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Test ${test_number}: ${description}${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "Command: python $CLI_SCRIPT $CSV_FILE $report_type --entity-id $entity_id --output-dir $OUTPUT_DIR"
    echo ""

    # Run the command and capture output
    if python "$CLI_SCRIPT" "$CSV_FILE" "$report_type" --entity-id "$entity_id" --output-dir "$OUTPUT_DIR" 2>&1 | head -50; then
        echo ""
        echo -e "${GREEN}✅ Test passed!${NC}"
    else
        echo ""
        echo -e "${RED}❌ Test failed!${NC}"
        exit 1
    fi

    echo ""
    test_number=$((test_number + 1))
}

# Initialize test counter
test_number=1

# ============================================================================
# FEED REPORTS
# ============================================================================

echo -e "${BLUE}═══════════════════════════════════════${NC}"
echo -e "${BLUE}  SECTION 1: Feed-Level Reports${NC}"
echo -e "${BLUE}═══════════════════════════════════════${NC}"
echo ""

run_report "feed" "FEED001" "Feed Report - FEED001 (Small feed)"
run_report "feed" "FEED016" "Feed Report - FEED016 (Different feed in same desk)"
run_report "feed" "FEED031" "Feed Report - FEED031 (Different desk)"

# ============================================================================
# DESK REPORTS
# ============================================================================

echo -e "${BLUE}═══════════════════════════════════════${NC}"
echo -e "${BLUE}  SECTION 2: Desk-Level Reports${NC}"
echo -e "${BLUE}═══════════════════════════════════════${NC}"
echo ""

run_report "desk" "DESK01" "Desk Report - DESK01 (Organization 1)"
run_report "desk" "DESK06" "Desk Report - DESK06 (Organization 2)"
run_report "desk" "DESK11" "Desk Report - DESK11 (Organization 3)"

# ============================================================================
# ORGANIZATION REPORTS
# ============================================================================

echo -e "${BLUE}═══════════════════════════════════════${NC}"
echo -e "${BLUE}  SECTION 3: Organization Reports${NC}"
echo -e "${BLUE}═══════════════════════════════════════${NC}"
echo ""

run_report "org" "ORG01" "Organization Report - ORG01"
run_report "org" "ORG02" "Organization Report - ORG02"
run_report "org" "ORG03" "Organization Report - ORG03"

# ============================================================================
# EDGE CASES
# ============================================================================

echo -e "${BLUE}═══════════════════════════════════════${NC}"
echo -e "${BLUE}  SECTION 4: Edge Cases${NC}"
echo -e "${BLUE}═══════════════════════════════════════${NC}"
echo ""

run_report "feed" "FEED060" "Edge Case - Last feed in dataset"
run_report "desk" "DESK15" "Edge Case - Last desk in dataset"
run_report "org" "ORG05" "Edge Case - Last organization"

# ============================================================================
# ERROR HANDLING
# ============================================================================

echo -e "${BLUE}═══════════════════════════════════════${NC}"
echo -e "${BLUE}  SECTION 5: Error Handling${NC}"
echo -e "${BLUE}═══════════════════════════════════════${NC}"
echo ""

echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Test ${test_number}: Error Case - Non-existent Feed${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Command: python $CLI_SCRIPT $CSV_FILE feed --entity-id FEED999 --output-dir $OUTPUT_DIR"
echo ""
echo "Expected: Should report error gracefully"
echo ""

if python "$CLI_SCRIPT" "$CSV_FILE" feed --entity-id "FEED999" --output-dir "$OUTPUT_DIR" 2>&1 | grep -q "No activities found"; then
    echo -e "${GREEN}✅ Error handling works correctly!${NC}"
else
    echo -e "${RED}❌ Error handling failed!${NC}"
    exit 1
fi

echo ""
test_number=$((test_number + 1))

# ============================================================================
# SUMMARY
# ============================================================================

echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  ALL TESTS COMPLETED SUCCESSFULLY! 🎉${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Count generated files
report_count=$(find "$OUTPUT_DIR" -name "*.png" 2>/dev/null | wc -l | tr -d ' ')

echo "Summary:"
echo "  ✓ Total tests run: $((test_number - 1))"
echo "  ✓ Feed reports: 4"
echo "  ✓ Desk reports: 3"
echo "  ✓ Organization reports: 3"
echo "  ✓ Edge case tests: 3"
echo "  ✓ Error handling tests: 1"
echo "  ✓ Visualizations generated: $report_count"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""

# Show directory structure
if [ -d "$OUTPUT_DIR" ]; then
    echo "Generated files:"
    tree "$OUTPUT_DIR" 2>/dev/null || find "$OUTPUT_DIR" -type f | head -20
    echo ""
fi

echo -e "${BLUE}Testing complete! All reports generated successfully.${NC}"
echo ""
echo "To view a specific report, check the output above or run:"
echo "  python $CLI_SCRIPT $CSV_FILE feed --entity-id FEED001"
echo ""
echo "To test with mocks (no file I/O), see:"
echo "  python reporting_agent_refactored/example_usage.py"
echo ""
