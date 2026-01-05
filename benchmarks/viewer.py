import argparse
import json
import os
from datetime import datetime


def load_results(filepath: str) -> list[dict]:
    results = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def extract_run_id(filepath: str) -> str:
    """Extract run_id from filename like 'hotpot_qa_results_20260105_123456.jsonl'"""
    basename = os.path.basename(filepath)
    parts = basename.split("_")
    if len(parts) >= 3:
        # Expecting format: taskname_results_RUNID.jsonl
        return parts[-1].replace(".jsonl", "")
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def generate_markdown_report(results: list[dict], run_id: str, output_dir: str):
    """Generate comprehensive markdown report."""

    # Extract model names from first result
    model_names = list(results[0]["models"].keys())
    total = len(results)

    # Calculate aggregate metrics
    metrics = {}
    for model_name in model_names:
        em_count = sum(1 for r in results if r["models"][model_name]["em"])
        avg_f1 = sum(r["models"][model_name]["f1"] for r in results) / total
        avg_time = sum(r["models"][model_name]["time"] for r in results) / total
        avg_calls = sum(r["models"][model_name]["llm_calls"] for r in results) / total

        metrics[model_name] = {
            "em": em_count,
            "em_pct": em_count / total * 100,
            "avg_f1": avg_f1,
            "avg_time": avg_time,
            "avg_calls": avg_calls,
        }

    # Generate markdown
    report_path = os.path.join(output_dir, f"report_{run_id}.md")

    with open(report_path, "w") as f:
        f.write(f"# Benchmark Report: {run_id}\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Examples:** {total}\n\n")
        f.write("---\n\n")

        # Overall Results Table
        f.write("## Overall Results\n\n")
        f.write("| Model | EM | EM % | Avg F1 | Avg Time (s) | Avg LLM Calls |\n")
        f.write("|-------|----|----|--------|--------------|---------------|\n")

        for model_name in model_names:
            m = metrics[model_name]
            f.write(
                f"| {model_name} | {m['em']}/{total} | {m['em_pct']:.1f}% | {m['avg_f1']:.3f} | {m['avg_time']:.2f} | {m['avg_calls']:.1f} |\n"
            )

        f.write("\n---\n\n")

        # Head-to-Head Comparison (each model vs baseline)
        baseline = model_names[0]  # First model is baseline
        f.write(f"## Head-to-Head vs Baseline ({baseline})\n\n")
        f.write("| Model | Wins | Losses | Ties | Win Rate |\n")
        f.write("|-------|------|--------|------|----------|\n")

        for model_name in model_names[1:]:
            wins = sum(
                1 for r in results if r["models"][model_name]["f1"] > r["models"][baseline]["f1"]
            )
            losses = sum(
                1 for r in results if r["models"][model_name]["f1"] < r["models"][baseline]["f1"]
            )
            ties = sum(
                1 for r in results if r["models"][model_name]["f1"] == r["models"][baseline]["f1"]
            )
            win_rate = wins / total * 100

            f.write(f"| {model_name} | {wins} | {losses} | {ties} | {win_rate:.1f}% |\n")

        f.write("\n---\n\n")

        # Key Insights
        f.write("## Key Insights\n\n")

        # Best F1
        best_f1_model = max(model_names, key=lambda m: metrics[m]["avg_f1"])
        f.write(f"- **Best F1 Score:** {best_f1_model} ({metrics[best_f1_model]['avg_f1']:.3f})\n")

        # Fastest
        fastest_model = min(model_names, key=lambda m: metrics[m]["avg_time"])
        f.write(f"- **Fastest:** {fastest_model} ({metrics[fastest_model]['avg_time']:.2f}s avg)\n")

        # Most efficient (fewest calls)
        efficient_model = min(model_names, key=lambda m: metrics[m]["avg_calls"])
        f.write(
            f"- **Most Efficient (Fewest Calls):** {efficient_model} ({metrics[efficient_model]['avg_calls']:.1f} calls avg)\n"
        )

        f.write("\n---\n\n")

        # Sample Comparisons - show all examples
        num_to_show = len(results)
        f.write(f"## Sample Comparisons (All {num_to_show} Examples)\n\n")

        for i, r in enumerate(results, 1):
            f.write(f"### Example {i}\n\n")
            f.write(f"**Question:** {r['question']}\n\n")
            f.write(f"**Gold Answer:** `{r['gold_answer']}`\n\n")

            f.write("| Model | Answer | F1 | EM | Time | Calls |\n")
            f.write("|-------|--------|----|----|------|-------|\n")

            for model_name in model_names:
                m = r["models"][model_name]
                answer_preview = m["answer"][:80] + "..." if len(m["answer"]) > 80 else m["answer"]
                f.write(
                    f"| {model_name} | {answer_preview} | {m['f1']:.2f} | {'✓' if m['em'] else '✗'} | {m['time']:.1f}s | {m['llm_calls']} |\n"
                )

            # Highlight winner
            best_model = max(model_names, key=lambda m: r["models"][m]["f1"])
            f.write(f"\n**Winner:** {best_model}\n\n")
            f.write("---\n\n")

    return report_path


def print_console_summary(results: list[dict]):
    """Print summary to console."""
    model_names = list(results[0]["models"].keys())
    total = len(results)

    print("\n" + "=" * 80)
    print(f"BENCHMARK RESULTS (n={total})")
    print("=" * 80)
    print(f"{'Model':<30} | {'EM':<10} | {'Avg F1':<8} | {'Avg Time':<10} | {'Avg Calls':<10}")
    print("-" * 80)

    for model_name in model_names:
        em_count = sum(1 for r in results if r["models"][model_name]["em"])
        avg_f1 = sum(r["models"][model_name]["f1"] for r in results) / total
        avg_time = sum(r["models"][model_name]["time"] for r in results) / total
        avg_calls = sum(r["models"][model_name]["llm_calls"] for r in results) / total

        print(
            f"{model_name:<30} | {em_count}/{total} ({em_count / total:.0%}){'':<2} | {avg_f1:.3f}    | {avg_time:.2f}s      | {avg_calls:.1f}"
        )

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="View RLM benchmark results and generate report")
    parser.add_argument("--file", type=str, required=True, help="Path to results file")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: Results file not found at {args.file}")
        return

    results = load_results(args.file)
    if not results:
        print("No results found.")
        return

    # Extract run_id from filename
    run_id = extract_run_id(args.file)
    output_dir = os.path.dirname(args.file)

    # Print console summary
    print_console_summary(results)

    # Generate markdown report
    report_path = generate_markdown_report(results, run_id, output_dir)
    print(f"\n📄 Markdown report generated: {report_path}")


if __name__ == "__main__":
    main()
