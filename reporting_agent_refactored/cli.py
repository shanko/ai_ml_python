#!/usr/bin/env python3
"""
Command-line interface for the refactored Activity Reporting Agent.

This standalone CLI script can be run directly from the command line without
import issues.
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from reporting_agent_refactored.reporting_agent_refactored import run_report
from reporting_agent_refactored.implementations import (
    RealTimeProvider,
    RealFileSystem,
    OpenAILLM,
    MatplotlibRenderer,
    DefaultConfig,
)
from reporting_agent_refactored.reporting_agent_refactored import Dependencies


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Activity Reporting Agent (Refactored Version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Feed report
  %(prog)s data.csv feed --entity-id FEED001

  # Desk report
  %(prog)s data.csv desk --entity-id DESK01

  # Organization report
  %(prog)s data.csv org --entity-id ORG01

  # Custom report (requires OpenAI API key)
  export OPENAI_API_KEY="sk-..."
  %(prog)s data.csv custom --query "What are the top priorities?"

  # Custom output directory
  %(prog)s data.csv feed --entity-id FEED001 --output-dir ./my_reports
        """,
    )

    parser.add_argument("csv_path", help="Path to CSV file with activity data")

    parser.add_argument(
        "report_type",
        choices=["feed", "desk", "org", "custom"],
        help="Type of report to generate",
    )

    parser.add_argument(
        "--entity-id", help="Entity ID (required for feed/desk/org reports)"
    )

    parser.add_argument("--query", help="Custom query (required for custom reports)")

    parser.add_argument(
        "--output-dir",
        default="./reports",
        help="Output directory for reports and visualizations (default: ./reports)",
    )

    parser.add_argument(
        "--llm-model",
        default="gpt-4",
        help="LLM model name for custom reports (default: gpt-4)",
    )

    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=0.2,
        help="LLM temperature for custom reports (default: 0.2)",
    )

    parser.add_argument(
        "--version", action="version", version="%(prog)s 2.0.0 (Refactored)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.report_type != "custom" and not args.entity_id:
        parser.error("--entity-id is required for feed/desk/org reports")

    if args.report_type == "custom" and not args.query:
        parser.error("--query is required for custom reports")

    # Setup dependencies
    config = DefaultConfig(
        llm_model=args.llm_model,
        llm_temperature=args.llm_temperature,
        output_dir=args.output_dir,
    )

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
    print(f"Generating {args.report_type} report...", file=sys.stderr)

    result = run_report(
        csv_path=args.csv_path,
        report_type=args.report_type,
        entity_id=args.entity_id,
        custom_query=args.query,
        deps=deps,
    )

    # Handle results
    if result["error"]:
        print(f"\n❌ ERROR: {result['error']}", file=sys.stderr)
        sys.exit(1)

    # Print report
    print("\n" + "=" * 70)
    print(result["report_content"])
    print("=" * 70)

    # Print visualization info
    if result["visualization_paths"]:
        print(f"\n📊 Visualizations saved:", file=sys.stderr)
        for path in result["visualization_paths"]:
            print(f"  - {path}", file=sys.stderr)

    print(f"\n✅ Report generated successfully!", file=sys.stderr)


if __name__ == "__main__":
    main()
