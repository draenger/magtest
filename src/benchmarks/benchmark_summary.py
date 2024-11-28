from sqlalchemy import func
import pandas as pd
from data.models import PreparedQuestion, ModelResult
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class BenchmarkSummary:
    def __init__(self, db):
        self.db = db

    def _get_base_query(self, test_session_id):
        return (
            self.db.get_session()
            .query(
                PreparedQuestion.benchmark_name,
                PreparedQuestion.category,
                ModelResult.model_name,
            )
            .join(
                PreparedQuestion,
                ModelResult.prepared_question_id == PreparedQuestion.id,
            )
            .filter(PreparedQuestion.test_session_id == test_session_id)
        )

    def _add_common_metrics(self, query):
        return query.add_columns(
            func.count(ModelResult.id).label("queries"),
            func.avg(ModelResult.score).label("avg_score"),
            func.sum(ModelResult.estimated_in_tokens).label("est_in_tokens"),
            func.sum(ModelResult.estimated_out_tokens).label("est_out_tokens"),
            func.sum(ModelResult.actual_in_tokens).label("act_in_tokens"),
            func.sum(ModelResult.actual_out_tokens).label("act_out_tokens"),
            func.sum(ModelResult.estimated_in_cost).label("est_in_cost"),
            func.sum(ModelResult.estimated_out_cost).label("est_out_cost"),
            func.sum(ModelResult.actual_in_cost).label("act_in_cost"),
            func.sum(ModelResult.actual_out_cost).label("act_out_cost"),
            func.avg(ModelResult.execution_time).label("avg_execution_time"),
        )

    def _execute_query(self, query):
        try:
            results = query.all()
            df = pd.DataFrame(results)
            df["est_tokens"] = df["est_in_tokens"] + df["est_out_tokens"]
            df["act_tokens"] = df["act_in_tokens"] + df["act_out_tokens"]
            df["est_cost"] = df["est_in_cost"] + df["est_out_cost"]
            df["act_cost"] = df["act_in_cost"] + df["act_out_cost"]
            return df
        except Exception as e:
            print(f"Error executing query: {e}")
            return pd.DataFrame()
        finally:
            self.db.get_session().close()

    def get_detailed_summary(self, test_session_id):
        query = self._get_base_query(test_session_id)
        query = self._add_common_metrics(query)
        query = query.group_by(
            PreparedQuestion.benchmark_name,
            PreparedQuestion.category,
            ModelResult.model_name,
        )
        return self._execute_query(query)

    def get_benchmark_summary(self, test_session_id):
        query = self._get_base_query(test_session_id)
        query = self._add_common_metrics(query)
        query = query.group_by(PreparedQuestion.benchmark_name, ModelResult.model_name)
        df = self._execute_query(query)

        if df.empty:
            return pd.DataFrame(
                columns=[
                    "Benchmark",
                    "Model",
                    "Queries",
                    "Avg Score",
                    "Total Execution Time",
                    "Est Tokens",
                    "Act Tokens",
                    "Est Cost",
                    "Act Cost",
                ]
            )

        df["total_execution_time"] = df.get("avg_execution_time", 0) * df["queries"]
        df["est_tokens"] = df["est_in_tokens"] + df["est_out_tokens"]
        df["act_tokens"] = df["act_in_tokens"] + df["act_out_tokens"]
        df["est_cost"] = df["est_in_cost"] + df["est_out_cost"]
        df["act_cost"] = df["act_in_cost"] + df["act_out_cost"]

        result_df = df[
            [
                "benchmark_name",
                "model_name",
                "queries",
                "avg_score",
                "total_execution_time",
                "est_tokens",
                "act_tokens",
                "est_cost",
                "act_cost",
            ]
        ].copy()

        result_df.columns = [
            "Benchmark",
            "Model",
            "Queries",
            "Avg Score",
            "Total Execution Time",
            "Est Tokens",
            "Act Tokens",
            "Est Cost",
            "Act Cost",
        ]

        return result_df

    def get_model_summary(self, test_session_id, model_name):
        query = self._get_base_query(test_session_id)
        query = self._add_common_metrics(query)
        query = query.filter(ModelResult.model_name == model_name)
        query = query.group_by(PreparedQuestion.benchmark_name)
        return self._execute_query(query)

    def get_category_summary(self, test_session_id, benchmark_name):
        query = self._get_base_query(test_session_id)
        query = self._add_common_metrics(query)
        query = query.filter(PreparedQuestion.benchmark_name == benchmark_name)
        query = query.group_by(PreparedQuestion.category, ModelResult.model_name)
        df = self._execute_query(query)
        # Sort by category and model_name for better readability
        if not df.empty:
            df = df.sort_values(["category", "model_name"])
        return df

    def print_detailed_summary(self, test_session_id):
        df = self.get_detailed_summary(test_session_id)
        print("Detailed Benchmark Summary")
        print("==========================")
        print(
            df.to_string(
                index=False,
                justify="center",
                col_space=15,
                float_format=lambda x: f"{x:.4f}",
            )
        )
        print("\n")

    def print_benchmark_summary(self, test_session_id):
        df = self.get_benchmark_summary(test_session_id)
        print("Benchmark Summary")
        print("=================")
        print(
            df.to_string(
                index=False,
                justify="center",
                col_space=20,
                float_format=lambda x: f"{x:.4f}",
            )
        )
        print("\n")

    def print_model_summary(self, test_session_id, model_name):
        df = self.get_model_summary(test_session_id, model_name)
        print(f"Model Summary: {model_name}")
        print("=" * (16 + len(model_name)))
        print(
            df.to_string(
                index=False,
                justify="center",
                col_space=15,
                float_format=lambda x: f"{x:.4f}",
            )
        )
        print("\n")

    def print_category_summary(self, test_session_id, benchmark_name, model_name=None):
        df = self.get_category_summary(test_session_id, benchmark_name)
        if model_name:
            df = df[df["model_name"] == model_name]
        print(
            df.to_string(
                index=False,
                justify="center",
                col_space=15,
                float_format=lambda x: f"{x:.4f}",
            )
        )

    def print_full_summary(self, test_session_id):
        print("Benchmark Summary")
        print("=================")
        df = self.get_benchmark_summary(test_session_id)
        print(
            df.to_string(
                index=False,
                justify="center",
                col_space=20,
                float_format=lambda x: f"{x:.4f}",
            )
        )
        print("\n")

        print("Detailed Summaries")
        print("==================")

        # Group by benchmark first
        benchmarks = df["Benchmark"].unique()
        for benchmark in benchmarks:
            print(f"\nBenchmark: {benchmark}")
            print("=" * (len(benchmark) + 11))

            # Get detailed results for this benchmark
            detailed_df = self.get_category_summary(test_session_id, benchmark)
            print(
                detailed_df.to_string(
                    index=False,
                    justify="center",
                    col_space=15,
                    float_format=lambda x: f"{x:.4f}",
                )
            )
            print("\n")

    def _get_provider(self, model_name):
        """Determine provider from model name"""
        if "gpt" in model_name.lower():
            return "openai"
        elif "claude" in model_name.lower():
            return "anthropic"
        elif "gemini" in model_name.lower():
            return "google"
        return "other"

    def format_float(self, value):
        return f"{value:.4f}" if pd.notnull(value) else "N/A"

    def get_summary_string(self, test_session_id):
        df = self.get_benchmark_summary(test_session_id)
        summary_string = "Benchmark Summary\n=================\n"
        summary_string += df.to_string(
            index=False,
            justify="center",
            col_space=20,
            formatters={
                col: self.format_float
                for col in df.select_dtypes(include=["float64"]).columns
            },
        )
        return summary_string

    def get_detailed_summary_string(self, test_session_id):
        df = self.get_detailed_summary(test_session_id)
        summary_string = "Detailed Benchmark Summary\n==========================\n"
        summary_string += df.to_string(
            index=False,
            justify="center",
            col_space=15,
            formatters={
                col: self.format_float
                for col in df.select_dtypes(include=["float64"]).columns
            },
        )
        return summary_string

    def save_full_summary_to_excel(self, test_session_id, output_path):
        """Save full benchmark summary to Excel file with multiple sheets"""
        print(f"Saving full summary to {output_path}")

        # Create Excel writer
        with pd.ExcelWriter(output_path) as writer:
            # Save main summary
            df_summary = self.get_benchmark_summary(test_session_id)
            df_summary.to_excel(writer, sheet_name="Summary", index=False)

            # Get detailed results for each benchmark
            benchmarks = df_summary["Benchmark"].unique()
            for benchmark in benchmarks:
                # Get detailed results for this benchmark
                detailed_df = self.get_category_summary(test_session_id, benchmark)

                # Clean up sheet name (Excel has 31 character limit for sheet names)
                sheet_name = benchmark[:30]

                # Save to Excel
                detailed_df.to_excel(writer, sheet_name=sheet_name, index=False)

    def save_full_summary_to_csv(self, test_session_id, output_dir):
        """Save full benchmark summary to CSV files in specified directory"""
        os.makedirs(output_dir, exist_ok=True)

        # Save main summary
        df_summary = self.get_benchmark_summary(test_session_id)
        summary_path = os.path.join(output_dir, "summary.csv")
        df_summary.to_csv(summary_path, index=False)
        print(f"Saved summary to {summary_path}")

        # Get detailed results for each benchmark
        benchmarks = df_summary["Benchmark"].unique()
        for benchmark in benchmarks:
            # Get detailed results for this benchmark
            detailed_df = self.get_category_summary(test_session_id, benchmark)

            # Create filename
            filename = f"{benchmark.replace('/', '_')}.csv"
            file_path = os.path.join(output_dir, filename)

            # Save to CSV
            detailed_df.to_csv(file_path, index=False)
            print(f"Saved {benchmark} details to {file_path}")

    def plot_benchmark_comparison(self, test_session_id, output_dir="plots"):
        """Create plots comparing zero-shot vs few-shot performance for each benchmark"""
        os.makedirs(output_dir, exist_ok=True)
        df = self.get_benchmark_summary(test_session_id)

        # Extract shot number from benchmark name and model provider
        df["shots"] = df["Benchmark"].str.extract(r"(\d+)Shot").astype(int)
        df["provider"] = df["Model"].apply(self._get_provider)

        # Create plots for each unique benchmark type (MMLU, GSM8K, BBH)
        benchmark_types = df["Benchmark"].str.split("-").str[0].unique()

        for benchmark_type in benchmark_types:
            benchmark_df = df[df["Benchmark"].str.startswith(benchmark_type)]

            plt.figure(figsize=(15, 8))
            sns.barplot(
                data=benchmark_df, x="Model", y="Avg Score", hue="shots", palette="Set2"
            )

            plt.title(f"{benchmark_type} Performance: Zero-shot vs Few-shot")
            plt.xticks(rotation=45, ha="right")
            plt.xlabel("Model")
            plt.ylabel("Average Score")
            plt.legend(title="Number of Shots")
            plt.tight_layout()

            plt.savefig(os.path.join(output_dir, f"{benchmark_type}_comparison.png"))
            plt.close()

    def plot_provider_category_comparison(self, test_session_id, output_dir="plots"):
        """Create plots comparing category performance for each provider/benchmark/shot combination"""
        os.makedirs(output_dir, exist_ok=True)

        # Get detailed results with categories
        detailed_df = self.get_detailed_summary(test_session_id)
        detailed_df["provider"] = detailed_df["model_name"].apply(self._get_provider)
        detailed_df["shots"] = (
            detailed_df["benchmark_name"].str.extract(r"(\d+)Shot").astype(int)
        )
        detailed_df["benchmark_type"] = (
            detailed_df["benchmark_name"].str.split("-").str[0]
        )

        # Create plots for each provider/benchmark/shot combination
        for provider in detailed_df["provider"].unique():
            provider_df = detailed_df[detailed_df["provider"] == provider]

            for benchmark_type in provider_df["benchmark_type"].unique():
                for shot_count in provider_df["shots"].unique():
                    # Filter data
                    plot_df = provider_df[
                        (provider_df["benchmark_type"] == benchmark_type)
                        & (provider_df["shots"] == shot_count)
                    ]

                    if plot_df.empty:
                        continue

                    # Create plot
                    plt.figure(figsize=(15, 8))

                    # Create bar plot for each model from this provider
                    sns.barplot(
                        data=plot_df,
                        x="category",
                        y="avg_score",
                        hue="model_name",
                        palette="deep",
                    )

                    plt.title(
                        f"{provider} Models - {benchmark_type} ({shot_count}-shot)\nPerformance by Category"
                    )
                    plt.xticks(rotation=45, ha="right")
                    plt.xlabel("Category")
                    plt.ylabel("Average Score")
                    plt.legend(
                        title="Model", bbox_to_anchor=(1.05, 1), loc="upper left"
                    )
                    plt.tight_layout()

                    # Save plot
                    filename = (
                        f"{provider}_{benchmark_type}_{shot_count}shot_categories.png"
                    )
                    plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
                    plt.close()

    def create_all_plots(self, test_session_id, output_dir="plots"):
        """Create all available plots"""
        print(f"Creating plots in {output_dir}")
        self.plot_benchmark_comparison(test_session_id, output_dir)
        self.plot_provider_category_comparison(test_session_id, output_dir)
        print(f"All plots saved to {output_dir}")

    def plot_cost_analysis(self, test_session_id, output_dir="plots"):
        """Create cost analysis plots comparing estimated vs actual costs per benchmark and provider"""
        os.makedirs(output_dir, exist_ok=True)
        df = self.get_benchmark_summary(test_session_id)

        # Add provider column and extract benchmark type
        df["provider"] = df["Model"].apply(self._get_provider)
        df["benchmark_type"] = df["Benchmark"].str.split("-").str[0]

        # Calculate total costs
        df["original_est_cost"] = df["Est Cost"]  # Original estimated cost
        df["total_est_cost"] = df["Est Cost"] / 2  # Discounted estimated cost
        df["total_act_cost"] = df["Act Cost"]
        df["total_est_tokens"] = df["Est Tokens"]
        df["total_act_tokens"] = df["Act Tokens"]

        # Group by benchmark type and provider
        cost_summary = (
            df.groupby(["benchmark_type", "provider"])
            .agg(
                {
                    "original_est_cost": "sum",
                    "total_est_cost": "sum",
                    "total_act_cost": "sum",
                    "total_est_tokens": "sum",
                    "total_act_tokens": "sum",
                }
            )
            .reset_index()
        )

        # Create plot
        plt.figure(figsize=(15, 8))

        # Set up the bar positions
        benchmarks = cost_summary["benchmark_type"].unique()
        providers = cost_summary["provider"].unique()
        x = np.arange(len(benchmarks))
        width = 0.25  # Increased width for bars

        # Calculate offsets for each provider's pair of bars
        num_providers = len(providers)
        offsets = np.linspace(
            -(num_providers - 1) * width, (num_providers - 1) * width, num_providers
        )

        # Plot bars for each provider
        for i, provider in enumerate(providers):
            provider_data = cost_summary[cost_summary["provider"] == provider]

            # Plot estimated cost
            est_bars = plt.bar(
                x + offsets[i],
                provider_data["total_est_cost"],
                width,
                label=f"{provider} (Est.)",
                color=self._get_provider_color(provider),
                alpha=0.8,
            )

            # Plot actual cost
            act_bars = plt.bar(
                x + offsets[i] + width / 2,
                provider_data["total_act_cost"],
                width,
                label=f"{provider} (Act.)",
                color=self._get_provider_color(provider),
                alpha=0.4,
            )

            # Add value labels on the bars
            for bars in [est_bars, act_bars]:
                for bar in bars:
                    height = bar.get_height()
                    plt.text(
                        bar.get_x() + bar.get_width() / 2,
                        height,
                        f"${height:.2f}",  # Changed to 2 decimal places
                        ha="center",
                        va="bottom",
                        rotation=90,
                        fontsize=8,
                    )

        plt.xlabel("Benchmark")
        plt.ylabel("Cost (USD)")
        plt.title("Estimated vs Actual Costs by Benchmark and Provider")
        plt.xticks(x, benchmarks)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, "cost_analysis.png"), bbox_inches="tight")
        plt.close()

        # Create detailed cost summary table with more information
        print("\nCost Analysis Summary")
        print("====================")

        for benchmark in benchmarks:
            print(f"\n{benchmark} Costs:")
            print("-" * (len(benchmark) + 7))

            benchmark_data = cost_summary[cost_summary["benchmark_type"] == benchmark]
            for _, row in benchmark_data.iterrows():
                print(f"\n{row['provider']}:")
                print(f"  Original Estimated Cost: ${row['original_est_cost']:.4f}")
                print(
                    f"  Discounted Estimated Cost (50%): ${row['total_est_cost']:.4f}"
                )
                print(f"  Actual Cost:    ${row['total_act_cost']:.4f}")
                print(f"  Estimated Tokens: {int(row['total_est_tokens'])}")
                print(f"  Actual Tokens: {int(row['total_act_tokens'])}")

                diff = row["total_act_cost"] - row["total_est_cost"]
                diff_percent = (
                    (diff / row["total_est_cost"] * 100)
                    if row["total_est_cost"] != 0
                    else 0
                )
                print(
                    f"  Difference from discounted estimate: ${diff:.4f} ({diff_percent:+.1f}%)"
                )

        # Save extended cost summary to CSV
        cost_summary.to_csv(os.path.join(output_dir, "cost_analysis.csv"), index=False)

    def plot_cost_effectiveness(self, test_session_id, output_dir="plots"):
        """Create plots showing cost effectiveness (score/cost ratio) for each benchmark"""
        os.makedirs(output_dir, exist_ok=True)
        df = self.get_benchmark_summary(test_session_id)

        # Calculate cost effectiveness using only actual costs
        df["total_cost"] = df["Act Cost"]
        df["cost_effectiveness"] = df["Avg Score"] / df["total_cost"]

        # Debug print dla sprawdzenia obliczeń
        for _, row in df.iterrows():
            print(f"\nModel: {row['Model']}")
            print(f"Score: {row['Avg Score']:.4f}")
            print(f"Total Cost (Act Cost): ${row['total_cost']:.4f}")
            print(f"Score/Cost Ratio: {row['cost_effectiveness']:.4f}")
            print(
                f"Verification: {row['Avg Score']:.4f} / ${row['total_cost']:.4f} = {(row['Avg Score']/row['total_cost']):.4f}"
            )

        df["provider"] = df["Model"].apply(self._get_provider)
        df["benchmark_type"] = df["Benchmark"].str.split("-").str[0]
        df["shots"] = df["Benchmark"].str.extract(r"(\d+)Shot").astype(int)

        # Create plot for each benchmark type
        for benchmark_type in df["benchmark_type"].unique():
            plt.figure(figsize=(15, 8))

            # Filter data for this benchmark
            benchmark_df = df[df["benchmark_type"] == benchmark_type].copy()

            # Sort by cost effectiveness
            benchmark_df = benchmark_df.sort_values(
                "cost_effectiveness", ascending=True
            )

            # Create bar plot
            bars = plt.barh(
                y=range(len(benchmark_df)),
                width=benchmark_df["cost_effectiveness"],
                color=[self._get_provider_color(p) for p in benchmark_df["provider"]],
            )

            # Customize plot
            plt.yticks(
                range(len(benchmark_df)),
                [
                    f"{model} ({shots}-shot) | Score: {score:.2f} | Total Cost: ${total_cost:.4f}"
                    for model, shots, score, total_cost in zip(
                        benchmark_df["Model"],
                        benchmark_df["shots"],
                        benchmark_df["Avg Score"],
                        benchmark_df["total_cost"],
                    )
                ],
            )
            plt.xlabel("Score/Cost Ratio")  # Changed from 'Score per Dollar'
            plt.title(
                f"{benchmark_type} Score/Cost Ratio\n(Higher is better)"
            )  # Changed title

            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                cost_effectiveness = benchmark_df.iloc[i]["cost_effectiveness"]

                # Add only cost effectiveness value at the end of the bar
                plt.text(
                    width,
                    bar.get_y() + bar.get_height() / 2,
                    f"{cost_effectiveness:.2f}",
                    va="center",
                    ha="left",
                    fontsize=8,
                )

            # Add legend for providers
            providers = benchmark_df["provider"].unique()
            legend_elements = [
                plt.Rectangle(
                    (0, 0), 1, 1, facecolor=self._get_provider_color(p), label=p.upper()
                )
                for p in providers
            ]
            plt.legend(handles=legend_elements, loc="lower right")

            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f"{benchmark_type}_cost_effectiveness.png"),
                bbox_inches="tight",
                dpi=300,
            )
            plt.close()

    def _get_provider_color(self, provider: str) -> str:
        """Get color for each provider"""
        color_map = {
            "openai": "#74aa9c",  # OpenAI green
            "anthropic": "#5436da",  # Anthropic purple
            "google": "#4285f4",  # Google blue
            "other": "#808080",  # Gray for others
        }
        return color_map.get(provider.lower(), color_map["other"])

    def analyze_few_shot_impact(self, test_session_id, output_dir="plots"):
        """Analyze the impact of few-shot learning across all models for each benchmark"""
        df = self.get_benchmark_summary(test_session_id)

        # Extract benchmark type and shots from benchmark name
        df["benchmark_type"] = df["Benchmark"].str.split("-").str[0]
        df["shots"] = df["Benchmark"].str.extract(r"(\d+)Shot").astype(int)

        # Calculate average scores for each benchmark and shot count
        shot_analysis = (
            df.groupby(["benchmark_type", "shots"])["Avg Score"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )

        # Print analysis
        print("\nFew-Shot Impact Analysis")
        print("=======================")

        for benchmark in shot_analysis["benchmark_type"].unique():
            print(f"\n{benchmark}:")
            print("-" * len(benchmark))

            benchmark_data = shot_analysis[shot_analysis["benchmark_type"] == benchmark]
            zero_shot = benchmark_data[benchmark_data["shots"] == 0].iloc[0]
            few_shot = benchmark_data[benchmark_data["shots"] > 0].iloc[0]

            print(
                f"Zero-shot average score: {zero_shot['mean']:.4f} (±{zero_shot['std']:.4f}, n={int(zero_shot['count'])})"
            )
            print(
                f"Few-shot average score:  {few_shot['mean']:.4f} (±{few_shot['std']:.4f}, n={int(few_shot['count'])})"
            )

            improvement = (
                (few_shot["mean"] - zero_shot["mean"]) / zero_shot["mean"]
            ) * 100
            print(f"Relative improvement: {improvement:+.2f}%")

        # Create visualization
        plt.figure(figsize=(12, 6))

        x = np.arange(len(shot_analysis["benchmark_type"].unique()))
        width = 0.35

        zero_shot_data = shot_analysis[shot_analysis["shots"] == 0]
        few_shot_data = shot_analysis[shot_analysis["shots"] > 0]

        plt.bar(
            x - width / 2,
            zero_shot_data["mean"],
            width,
            label="Zero-shot",
            color="lightblue",
        )
        plt.bar(
            x + width / 2,
            few_shot_data["mean"],
            width,
            label="Few-shot",
            color="lightgreen",
        )

        # Add error bars
        plt.errorbar(
            x - width / 2,
            zero_shot_data["mean"],
            yerr=zero_shot_data["std"],
            fmt="none",
            color="blue",
            capsize=5,
        )
        plt.errorbar(
            x + width / 2,
            few_shot_data["mean"],
            yerr=few_shot_data["std"],
            fmt="none",
            color="green",
            capsize=5,
        )

        # Add value labels on bars
        for i, v in enumerate(zero_shot_data["mean"]):
            plt.text(i - width / 2, v, f"{v:.3f}", ha="center", va="bottom")
        for i, v in enumerate(few_shot_data["mean"]):
            plt.text(i + width / 2, v, f"{v:.3f}", ha="center", va="bottom")

        plt.xlabel("Benchmark")
        plt.ylabel("Average Score")
        plt.title("Zero-shot vs Few-shot Performance Comparison")
        plt.xticks(x, zero_shot_data["benchmark_type"])
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "few_shot_impact.png"))
        plt.close()

        # Save analysis to CSV
        shot_analysis.to_csv(
            os.path.join(output_dir, "few_shot_analysis.csv"), index=False
        )

    def plot_top_cost_effective_models(
        self, test_session_id, output_dir="plots", top_n=5
    ):
        """Analyze top N most cost-effective models across all benchmarks"""
        os.makedirs(output_dir, exist_ok=True)
        df = self.get_benchmark_summary(test_session_id)

        # Calculate cost effectiveness for each model
        df["benchmark_type"] = df["Benchmark"].str.split("-").str[0]
        df["shots"] = df["Benchmark"].str.extract(r"(\d+)Shot").astype(int)
        df["provider"] = df["Model"].apply(self._get_provider)
        df["cost_effectiveness"] = df["Avg Score"] / df["Act Cost"]

        # Calculate average cost effectiveness for each model across all benchmarks
        model_avg_effectiveness = (
            df.groupby(["Model", "provider"])["cost_effectiveness"].mean().reset_index()
        )

        # Get top N most cost-effective models
        top_models = model_avg_effectiveness.nlargest(top_n, "cost_effectiveness")

        # Filter original dataframe for top models only
        top_models_data = df[df["Model"].isin(top_models["Model"])]

        # Create plot
        plt.figure(figsize=(15, 8))

        # Set up positions for grouped bars
        benchmarks = df["benchmark_type"].unique()
        n_benchmarks = len(benchmarks)
        width = 0.15

        # Create positions for bars
        x = np.arange(len(top_models))

        # Plot bars for each benchmark
        for i, benchmark in enumerate(benchmarks):
            benchmark_data = top_models_data[
                top_models_data["benchmark_type"] == benchmark
            ]
            benchmark_data = benchmark_data.merge(top_models[["Model"]], on="Model")

            effectiveness_values = []
            for model in top_models["Model"]:
                model_data = benchmark_data[benchmark_data["Model"] == model]
                effectiveness_values.append(
                    model_data["cost_effectiveness"].mean()
                    if not model_data.empty
                    else 0
                )

            plt.bar(
                x + i * width, effectiveness_values, width, label=benchmark, alpha=0.8
            )

        # Customize plot
        plt.xlabel("Models")
        plt.ylabel("Score/Cost Ratio")
        plt.title(f"Top {top_n} Most Cost-Effective Models Across Benchmarks")

        # Set x-ticks with model names and their average effectiveness
        model_labels = [
            f"{model}\n(Avg: {eff:.2f})"
            for model, eff in zip(top_models["Model"], top_models["cost_effectiveness"])
        ]
        plt.xticks(
            x + width * (n_benchmarks - 1) / 2, model_labels, rotation=45, ha="right"
        )

        # Add legend
        plt.legend(title="Benchmark")

        # Color bars by provider
        for i, (_, row) in enumerate(top_models.iterrows()):
            plt.axvspan(
                i - width / 2,
                i + width * n_benchmarks - width / 2,
                alpha=0.1,
                color=self._get_provider_color(row["provider"]),
            )

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "top_cost_effective_models.png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

        # Print detailed analysis
        print(f"\nTop {top_n} Most Cost-Effective Models Analysis")
        print("=" * 50)

        for _, row in top_models.iterrows():
            model_data = top_models_data[top_models_data["Model"] == row["Model"]]
            print(f"\nModel: {row['Model']} ({row['provider']})")
            print(f"Average Score/Cost Ratio: {row['cost_effectiveness']:.2f}")
            print("Performance by benchmark:")
            for benchmark in benchmarks:
                benchmark_eff = model_data[model_data["benchmark_type"] == benchmark][
                    "cost_effectiveness"
                ].mean()
                benchmark_score = model_data[model_data["benchmark_type"] == benchmark][
                    "Avg Score"
                ].mean()
                print(f"  {benchmark}:")
                print(f"    Score/Cost Ratio: {benchmark_eff:.2f}")
                print(f"    Average Score: {benchmark_score:.2f}")

    def analyze_openai_models_comparison(self, test_session_id, output_dir="plots"):
        """Compare performance and cost-effectiveness of GPT-4, GPT-4-turbo, and GPT-3.5-turbo"""
        os.makedirs(output_dir, exist_ok=True)

        # Get data
        df = self.get_benchmark_summary(test_session_id)

        # Filter for target models and prepare data
        target_models = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo-0125"]
        df["benchmark_type"] = df["Benchmark"].str.split("-").str[0]
        df["shots"] = df["Benchmark"].str.extract(r"(\d+)Shot")[0].fillna(0).astype(int)

        # Define shot mapping for each benchmark type
        shot_mapping = {
            "MMLU": {0: "Zero-shot", 5: "Few-shot"},
            "BBH": {0: "Zero-shot", 3: "Few-shot"},
            "GSM8K": {0: "Zero-shot", 4: "Few-shot"},
        }

        # Create shot_type based on benchmark-specific mapping
        df["shot_type"] = df.apply(
            lambda row: shot_mapping.get(row["benchmark_type"], {}).get(
                row["shots"], f"{row['shots']}-shot"
            ),
            axis=1,
        )

        df["cost_effectiveness"] = df["Avg Score"] / df["Act Cost"]

        # Create figure with subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        # Colors for target models
        model_colors = {
            "gpt-4": "#1f77b4",
            "gpt-4-turbo": "#ff7f0e",
            "gpt-3.5-turbo-0125": "#2ca02c",
        }

        # Plot performance comparison
        for ax, metric in [(ax1, "Avg Score"), (ax2, "cost_effectiveness")]:
            # Create x-axis positions for each benchmark-shot combination
            benchmarks = sorted(df["benchmark_type"].unique())
            x_positions = []
            x_labels = []

            for i, benchmark in enumerate(benchmarks):
                benchmark_shots = sorted(shot_mapping[benchmark].values())
                for shot in benchmark_shots:
                    x_positions.append(i * 2.5 + (0 if "Zero" in shot else 0.8))
                    x_labels.append(f"{benchmark}\n{shot}")

            # First plot other models as background
            other_models = df[~df["Model"].isin(target_models)]
            for i, benchmark in enumerate(benchmarks):
                benchmark_shots = sorted(shot_mapping[benchmark].values())
                for shot in benchmark_shots:
                    other_data = other_models[
                        (other_models["benchmark_type"] == benchmark)
                        & (other_models["shot_type"] == shot)
                    ]
                    if not other_data.empty:
                        x_pos = i * 2.5 + (0 if "Zero" in shot else 0.8)
                        ax.scatter(
                            [x_pos] * len(other_data),
                            other_data[metric],
                            color="lightgray",
                            alpha=0.3,
                            s=50,
                            zorder=1,
                            label="Other Models" if (i == 0 and "Zero" in shot) else "",
                        )

            # Then plot target models
            for model in target_models:
                model_data = df[df["Model"] == model]
                for i, benchmark in enumerate(benchmarks):
                    benchmark_shots = sorted(shot_mapping[benchmark].values())
                    for shot in benchmark_shots:
                        shot_data = model_data[
                            (model_data["benchmark_type"] == benchmark)
                            & (model_data["shot_type"] == shot)
                        ]
                        if not shot_data.empty:
                            x_pos = i * 2.5 + (0 if "Zero" in shot else 0.8)
                            ax.scatter(
                                [x_pos],
                                shot_data[metric],
                                label=model if (i == 0 and "Zero" in shot) else "",
                                color=model_colors[model],
                                s=200,
                                zorder=2,
                            )

            # Customize plot
            ax.set_xlabel("Benchmark Type and Shot Count", fontsize=12)
            ax.set_ylabel(
                "Average Score" if metric == "Avg Score" else "Score/Cost Ratio",
                fontsize=12,
            )
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.set_title(
                (
                    "Performance Comparison"
                    if metric == "Avg Score"
                    else "Cost Effectiveness Comparison"
                ),
                fontsize=14,
                pad=20,
            )

            # Set x-axis ticks and labels
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, rotation=45, ha="right")

            # Add vertical lines between benchmark groups
            for i in range(1, len(benchmarks)):
                ax.axvline(x=i * 2.5 - 0.45, color="gray", linestyle="--", alpha=0.3)

        # Create a single legend for both plots
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))

        # Sort legend items: first models, then Other Models
        sorted_labels = target_models + ["Other Models"]
        sorted_handles = [
            by_label[label] for label in sorted_labels if label in by_label
        ]

        # Add legend below the plots
        fig.legend(
            sorted_handles,
            sorted_labels,
            loc="center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=4,
            fontsize=10,
        )

        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)

        # Save plot
        output_path = os.path.join(output_dir, "model_comparison.png")
        plt.savefig(
            output_path,
            bbox_inches="tight",
            dpi=300,
            facecolor="white",
            edgecolor="none",
        )
        plt.close()

        # Print summary statistics with improved debugging
        print("\nPerformance Summary:")
        print("=" * 50)
        for model in target_models:
            model_data = df[df["Model"] == model]
            print(f"\n{model}:")
            for benchmark in benchmarks:
                print(f"\n  {benchmark}:")
                benchmark_shots = sorted(shot_mapping[benchmark].values())
                for shot in benchmark_shots:
                    shot_data = model_data[
                        (model_data["benchmark_type"] == benchmark)
                        & (model_data["shot_type"] == shot)
                    ]
                    if not shot_data.empty:
                        print(f"    {shot}:")
                        print(f"      Score: {shot_data['Avg Score'].iloc[0]:.4f}")
                        print(
                            f"      Cost Effectiveness: {shot_data['cost_effectiveness'].iloc[0]:.4f}"
                        )
                    else:
                        print(f"    {shot}: NO DATA FOUND")

                    # Additional debugging for missing data
                    if shot != "Zero-shot" and shot_data.empty:
                        print(f"\nDiagnostic for {benchmark} {shot}:")
                        debug_data = model_data[
                            model_data["benchmark_type"] == benchmark
                        ]
                        print("Available records for this model and benchmark:")
                        print(
                            debug_data[["Benchmark", "shots", "shot_type"]].to_string()
                        )

    def plot_openai_vs_google_comparison(self, test_session_id):
        """
        Creates a plot comparing cost effectiveness between OpenAI and Google models.
        """
        import pandas as pd

        # Get data for all models
        data = self._get_model_performance_data(test_session_id)

        # Filter and categorize models
        openai_models = {
            k: v for k, v in data.items() if k.startswith(("gpt-3", "gpt-4"))
        }
        google_models = {
            k: v for k, v in data.items() if k.startswith(("gemini", "palm"))
        }

        # Prepare data for plotting
        plot_data = []
        for model_type, models in [
            ("OpenAI", openai_models),
            ("Google", google_models),
        ]:
            for model_name, metrics in models.items():
                plot_data.append(
                    {
                        "Provider": model_type,
                        "Model": model_name,
                        "Cost Effectiveness": metrics["cost_effectiveness"],
                        "Score": metrics["score"],
                    }
                )

        # Convert to DataFrame
        df_plot = pd.DataFrame(plot_data)

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Cost Effectiveness Comparison
        sns.barplot(
            data=df_plot, x="Provider", y="Cost Effectiveness", ax=ax1, palette="Set2"
        )
        ax1.set_title("Cost Effectiveness by Provider")
        ax1.set_ylabel("Cost Effectiveness (Score/Cost)")

        # Plot 2: Score Comparison
        sns.barplot(data=df_plot, x="Provider", y="Score", ax=ax2, palette="Set2")
        ax2.set_title("Average Score by Provider")
        ax2.set_ylabel("Score")

        plt.tight_layout()
        plt.savefig("plots/openai_vs_google_comparison.png")
        plt.close()

        # Print statistical summary
        print("\nProvider Comparison Summary:")
        print("============================")

        for provider, models in [("OpenAI", openai_models), ("Google", google_models)]:
            if len(models) > 0:  # Add check for empty models
                avg_effectiveness = sum(
                    m["cost_effectiveness"] for m in models.values()
                ) / len(models)
                avg_score = sum(m["score"] for m in models.values()) / len(models)

                print(f"\n{provider}:")
                print(f"  Average Cost Effectiveness: {avg_effectiveness:.4f}")
                print(f"  Average Score: {avg_score:.4f}")
            else:
                print(f"\n{provider}: No models found")

    def _get_model_performance_data(self, test_session_id):
        """
        Gets performance data for all models in the given test session.
        Returns a dictionary with model names as keys and their performance metrics as values.
        """
        # Get benchmark summary data
        df = self.get_benchmark_summary(test_session_id)

        # Calculate metrics for each model
        model_data = {}
        for model in df["Model"].unique():
            model_df = df[df["Model"] == model]

            # Calculate average score and cost effectiveness
            avg_score = model_df["Avg Score"].mean()
            total_cost = model_df["Act Cost"].sum()
            cost_effectiveness = avg_score / total_cost if total_cost > 0 else 0

            model_data[model] = {
                "score": avg_score,
                "cost": total_cost,
                "cost_effectiveness": cost_effectiveness,
            }

        return model_data
