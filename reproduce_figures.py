"""
reproduce_figures.py - Reproduce all figures from the BeamFormer paper artifact.

Usage:
    python reproduce_figures.py --list
    python reproduce_figures.py --figure "Overall performance of different approaches" [--data_source cache|from_scratch] [--save_dir figures/] [--num_samples N]

data_source:
    cache        (default) Load pre-computed evaluation results from eval_results/
    from_scratch Run evaluation from scratch and save results
"""

import argparse
import importlib
import json
import os
import sys

# Ensure project root is on the path
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd


# ---------------------------------------------------------------------------
# Figure registry
# ---------------------------------------------------------------------------

FIGURE_REGISTRY = {
    "Overall performance of different approaches": {
        "status": "available",
        "fig_num": 10,
        "note": "RSS Loss and AoD Error CDF, indoor+outdoor combined",
        "module": "figure10",
        "configs": ["compare_vs_baselines_indoor", "compare_vs_baselines_outdoor"],
    },
    "Visualization of layer-wise beam spectrum generation": {
        "status": "available",
        "fig_num": 11,
        "note": "Polar heatmap of beam spectra at each layer",
        "module": "figure11",
        "configs": [],
    },
    "Impact of scene configurations": {
        "status": "available",
        "fig_num": 12,
        "note": "Cross-frequency, scenario, antenna pattern, SNR CDFs",
        "module": "figure12",
        "configs": ["cross_frequency", "cross_scenarios", "cross_pattern", "cross_snr"],
    },
    "Impact of model parameters": {
        "status": "available",
        "fig_num": 13,
        "note": "Ablation study CDFs",
        "module": "figure13",
        "configs": ["ablation_study"],
    },
    "Comparison of positional encoders": {
        "status": "available",
        "fig_num": 14,
        "note": "CDF comparing Array Factor, 2D Positional, and Concat position encodings",
        "module": "figure14",
        "configs": ["cross_encoder"],
    },
    "Comparison of model latencies": {
        "status": "available",
        "fig_num": 15,
        "note": "Reads from eval_results/running_time_comparison/cached/ (from_scratch runs running_time_analyse.py)",
        "module": "figure15",
        "configs": [],
    },
    "Comparison of reference beam settings": {
        "status": "available",
        "fig_num": 16,
        "note": "Beam settings CDF",
        "module": "figure16",
        "configs": ["compare_beam_settings"],
    },
    "Comparison of power estimators": {
        "status": "available",
        "fig_num": 17,
        "note": "RSS Error CDF comparing ARN power estimator vs Normal Distribution (outdoor)",
        "module": "figure17",
        "configs": ["test_arn_performance_outdoor"],
    },
    "Performance with real-world data": {
        "status": "available",
        "fig_num": 18,
        "note": "CDF plots from real-world Sivers/IBM hardware evaluations",
        "module": "figure18",
        "configs": [],
    },
    "Example spectrum with Sivers SDR": {
        "status": "available",
        "fig_num": 19,
        "note": "Beam spectrum visualization from Sivers SDR (requires model + data)",
        "module": "figure19",
        "configs": [],
    },
    "Example spectrum with IBM SDR": {
        "status": "available",
        "fig_num": 20,
        "note": "Beam spectrum visualization from IBM SDR (requires model + data)",
        "module": "figure20",
        "configs": [],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json_as_df_dict(json_path):
    """
    Load a performance JSON file and return a dict of {model_name: DataFrame}.

    Args:
        json_path: path to all_performance_dict.json

    Returns:
        dict mapping model_name -> pd.DataFrame
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Performance JSON not found: {json_path}")

    with open(json_path, 'r') as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected a dict in {json_path}, got {type(raw)}")

    result = {}
    for model_name, records in raw.items():
        if isinstance(records, list):
            result[model_name] = pd.DataFrame(records)
        else:
            print(f"Warning: Skipping model '{model_name}' - data is not a list of records")

    return result


def find_cache_json(eval_results_dir, config_name):
    """
    Find the most recent cached all_performance_dict.json for a given config.

    Looks in: eval_results/<config_name>/<newest_subdir>/all_performance_dict.json

    Args:
        eval_results_dir: base directory for evaluation results (e.g. 'eval_results')
        config_name: name of the evaluation config

    Returns:
        absolute path to the JSON file

    Raises:
        FileNotFoundError: if no JSON found
    """
    base_dir = os.path.join(eval_results_dir, config_name)

    if os.path.isdir(base_dir):
        subdirs = sorted(
            [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))],
            reverse=True  # newest timestamp first (lexicographic sort works for YYYYMMDD_HHMMSS)
        )
        for subdir in subdirs:
            candidate = os.path.join(base_dir, subdir, "all_performance_dict.json")
            if os.path.exists(candidate):
                return candidate

    raise FileNotFoundError(
        f"No cached performance JSON found for config '{config_name}'.\n"
        f"Looked in: {base_dir}\n"
        f"Run with --data_source from_scratch to generate evaluation results."
    )


def run_evaluation_from_scratch(config_name, num_samples, eval_results_dir):
    """
    Run evaluation for a config and save results to a timestamped directory.

    Args:
        config_name: name of the evaluation config (from configs/evaluate/)
        num_samples: optional int to limit test batch length
        eval_results_dir: base directory for evaluation results

    Returns:
        path to the saved all_performance_dict.json
    """
    from beamformer.utils import load_config
    from beamformer.evaluator import Evaluator
    from datetime import datetime

    print(f"Running evaluation for config: {config_name}")
    if num_samples is not None:
        print(f"  Limiting to {num_samples} samples")

    config = load_config(config_name, predix="evaluate")
    if num_samples is not None:
        config.request.test_batch_length = num_samples

    evaluator = Evaluator(config)
    # Evaluator runs evaluation in __init__ via load_models_performance()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(eval_results_dir, config_name, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    performance_dict_serializable = {}
    for model_name, df in evaluator.all_performance_dict.items():
        performance_dict_serializable[model_name] = df.to_dict('records')

    json_path = os.path.join(output_dir, 'all_performance_dict.json')
    with open(json_path, 'w') as f:
        json.dump(performance_dict_serializable, f, indent=2, default=str)

    print(f"Saved evaluation results to: {json_path}")
    return json_path


# ---------------------------------------------------------------------------
# Figure-specific handlers
# ---------------------------------------------------------------------------

def handle_figure11(fig_module, args, save_path):
    """Handle the special case for Figure 11 (layer-wise spectra visualization)."""
    save_path = os.path.splitext(save_path)[0] + '.png'
    print("Generating spectra from model (requires GPU and trained model)...")
    try:
        spectra_data = fig_module.generate_spectra(indices=fig_module.DEFAULT_INDICES)
    except Exception as e:
        print(f"Error generating spectra: {e}")
        print(
            "\nTo reproduce Figure 11, you need:\n"
            "  1. The trained model at saved_models/pi_param_co_train_indoor/\n"
            "  2. A CUDA-enabled GPU\n"
        )
        sys.exit(1)

    fig_module.plot(spectra_data, save_path)
    return save_path


def find_hardware_results_json(eval_results_dir):
    """Find the most recent hardware_results.json in eval_results/realworld/."""
    realworld_dir = os.path.join(eval_results_dir, "realworld")
    if not os.path.isdir(realworld_dir):
        raise FileNotFoundError(
            f"No realworld results directory found at: {realworld_dir}\n"
            "Run with --data_source from_scratch to generate hardware evaluation results."
        )
    # Find timestamped subdirs (newest first)
    subdirs = sorted(
        [d for d in os.listdir(realworld_dir) if os.path.isdir(os.path.join(realworld_dir, d))],
        reverse=True
    )
    for subdir in subdirs:
        candidate = os.path.join(realworld_dir, subdir, "hardware_results.json")
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"No hardware_results.json found under: {realworld_dir}\n"
        "Run with --data_source from_scratch to generate hardware evaluation results."
    )


def handle_figure18(fig_module, args, save_path, eval_results_dir):
    """Handle Figure 18: real-world hardware performance CDF."""
    import json
    if args.data_source == "cache":
        try:
            json_path = find_hardware_results_json(eval_results_dir)
            print(f"  Found cache: {json_path}")
        except FileNotFoundError as e:
            print(f"Cache not found: {e}")
            print("Attempting to run hardware evaluation from scratch...")
            args.data_source = "from_scratch"

    if args.data_source == "from_scratch":
        from datetime import datetime
        from beamformer.hardware_inference import run_all_evaluation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hw_save_dir = os.path.join(eval_results_dir, "realworld", timestamp)
        try:
            json_path = run_all_evaluation(hw_save_dir)
        except Exception as e:
            print(f"Error running hardware evaluation: {e}")
            print(
                "\nTo reproduce Figure 18, you need:\n"
                "  1. Trained models at saved_models/sivers/ and saved_models/ibm/\n"
                "  2. Hardware data at csi-dataset/realworld_sivers/ and csi-dataset/realworld_ibm/\n"
            )
            sys.exit(1)

    with open(json_path, 'r') as f:
        hw_results = json.load(f)

    fig_module.plot(hw_results, save_path)


def handle_figure19(fig_module, save_path):
    """Handle Figure 19: Sivers SDR spectrum example."""
    try:
        fig_module.plot(save_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error generating Figure 19: {e}")
        print(
            "\nTo reproduce Figure 19, you need:\n"
            "  1. Trained Sivers model at saved_models/sivers/\n"
            "  2. Sivers data at csi-dataset/realworld_sivers/indoor-nlos/1.pkl\n"
        )
        sys.exit(1)


def handle_figure20(fig_module, save_path):
    """Handle Figure 20: IBM SDR spectrum example."""
    try:
        fig_module.plot(save_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error generating Figure 20: {e}")
        print(
            "\nTo reproduce Figure 20, you need:\n"
            "  1. Trained IBM model at saved_models/ibm/\n"
            "  2. IBM data at csi-dataset/realworld_ibm/indoor-los/beamtable_20251203_193653/experiment_special\n"
        )
        sys.exit(1)


def find_cache_csv_dir(eval_results_dir):
    """
    Find CSVs for Figure 15. Search order:
    1. eval_results/running_time_comparison/cached/
    2. Newest timestamped subdir of eval_results/running_time_comparison/
    Returns the directory path, or raises FileNotFoundError.
    """
    base_dir = os.path.join(eval_results_dir, "running_time_comparison")
    _pi = "model_benchmark_results_pi.csv"
    _tr = "model_benchmark_results_transformer.csv"

    cached_dir = os.path.join(base_dir, "cached")
    if os.path.exists(os.path.join(cached_dir, _pi)) and \
       os.path.exists(os.path.join(cached_dir, _tr)):
        return cached_dir

    if os.path.isdir(base_dir):
        subdirs = sorted(
            [d for d in os.listdir(base_dir)
             if os.path.isdir(os.path.join(base_dir, d)) and d != "cached"],
            reverse=True
        )
        for subdir in subdirs:
            candidate = os.path.join(base_dir, subdir)
            if os.path.exists(os.path.join(candidate, _pi)) and \
               os.path.exists(os.path.join(candidate, _tr)):
                return candidate

    raise FileNotFoundError(
        f"No benchmark CSVs found under: {base_dir}\n"
        "Run with --data_source from_scratch to generate them."
    )


def handle_figure15(fig_module, args, save_path, eval_results_dir):
    """Handle the special case for Figure 15 (latency benchmark)."""
    import shutil
    import subprocess
    from datetime import datetime

    if args.data_source == "from_scratch":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ts_dir = os.path.join(eval_results_dir, "running_time_comparison", timestamp)
        os.makedirs(ts_dir, exist_ok=True)

        print(f"Running running_time_analyse.py (output -> {ts_dir}) ...")
        env = os.environ.copy()
        env["RT_OUTPUT_DIR"] = ts_dir
        script = os.path.join(_PROJECT_ROOT, "beamformer", "running_time_analyse.py")
        result = subprocess.run(
            [sys.executable, script],
            env=env,
            cwd=_PROJECT_ROOT
        )
        if result.returncode != 0:
            print("Error: running_time_analyse.py failed.")
            sys.exit(1)

        # Copy CSVs to cached/ for future cache hits
        cached_dir = os.path.join(eval_results_dir, "running_time_comparison", "cached")
        os.makedirs(cached_dir, exist_ok=True)
        for csv_file in ["model_benchmark_results_pi.csv",
                         "model_benchmark_results_transformer.csv"]:
            src = os.path.join(ts_dir, csv_file)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(cached_dir, csv_file))

        csv_dir = ts_dir
    else:
        try:
            csv_dir = find_cache_csv_dir(eval_results_dir)
            print(f"  Found cached CSVs: {csv_dir}")
        except FileNotFoundError as e:
            print(f"Cache not found: {e}")
            print("Re-run with --data_source from_scratch to generate benchmark CSVs.")
            sys.exit(1)

    data = fig_module.load_from_csv(csv_dir)
    fig_module.plot(data, save_path)


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------

def print_registry():
    """Print all figures in the registry with their status."""
    print("\nBeamFormer Paper - Figure Registry")
    print("=" * 70)
    print(f"{'Fig':<6} {'Status':<12} {'Title'}")
    print("-" * 70)
    for title, entry in sorted(FIGURE_REGISTRY.items(), key=lambda x: x[1]['fig_num']):
        status = entry['status']
        fig_num = entry['fig_num']
        status_str = "AVAILABLE" if status == "available" else "unavailable"
        print(f"  {fig_num:<4} {status_str:<12} {title}")
        print(f"       {'':12} {entry['note']}")
    print("=" * 70)
    print(
        "\nUsage: python reproduce_figures.py "
        "--figure \"<title>\" [--data_source cache|from_scratch]"
    )


def handle_figure17(fig_module, args, save_path, eval_results_dir):
    """Handle Figure 17: power estimator comparison (requires random-peak evaluation)."""
    from beamformer.evaluator import Evaluator
    from beamformer.utils import load_config
    from datetime import datetime

    data = {}
    for config_name in fig_module.REQUIRED_CONFIGS:
        if args.data_source == "cache":
            try:
                json_path = find_cache_json(eval_results_dir, config_name)
                print(f"  Found cache for {config_name}: {json_path}")
            except FileNotFoundError as e:
                print(f"Cache not found for {config_name}: {e}")
                print("Falling back to from_scratch for this config...")
                args.data_source = "from_scratch"

        if args.data_source == "from_scratch":
            params = fig_module.PEAK_SCALE_PARAMS[config_name]
            print(f"Running evaluator for {config_name} "
                  f"(peak_mean={params['peak_mean']}, peak_std={params['peak_std']}) ...")
            config = load_config(config_name, predix="evaluate")
            if args.num_samples is not None:
                config.request.test_batch_length = args.num_samples

            evaluator = Evaluator(
                config,
                peak_scale_mean=params["peak_mean"],
                peak_scale_std=params["peak_std"],
            )
            evaluator.run()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(eval_results_dir, config_name, timestamp)
            os.makedirs(output_dir, exist_ok=True)

            import json
            perf_serializable = {
                k: v.to_dict("records")
                for k, v in evaluator.all_performance_dict.items()
            }
            json_path = os.path.join(output_dir, "all_performance_dict.json")
            with open(json_path, "w") as f:
                json.dump(perf_serializable, f, indent=2, default=str)
            print(f"  Saved evaluation results to: {json_path}")

        data[config_name] = load_json_as_df_dict(json_path)
        print(f"  Loaded {len(data[config_name])} models: {list(data[config_name].keys())}")

    fig_module.plot(data, save_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Reproduce figures from the BeamFormer paper artifact.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--figure",
        type=str,
        default=None,
        help="Name of the figure to reproduce (use --list to see all options)"
    )
    parser.add_argument(
        "--data_source",
        default="cache",
        choices=["cache", "from_scratch"],
        help="Data source: 'cache' loads pre-computed results, 'from_scratch' runs evaluations (default: cache)"
    )
    parser.add_argument(
        "--save_dir",
        default="figures",
        help="Directory to save the output figures (default: figures/)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Limit number of test samples (only used with --data_source from_scratch)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available figures and exit"
    )
    parser.add_argument(
        "--eval_results_dir",
        default="eval_results",
        help="Directory for evaluation results cache (default: eval_results/)"
    )

    args = parser.parse_args()

    if args.list:
        print_registry()
        sys.exit(0)

    if args.figure is None:
        parser.print_help()
        print("\nUse --list to see available figures.")
        sys.exit(1)

    if args.figure not in FIGURE_REGISTRY:
        print(f"Error: Figure '{args.figure}' not found in registry.")
        print("Use --list to see available figures.")
        sys.exit(1)

    entry = FIGURE_REGISTRY[args.figure]

    if entry["status"] == "unavailable":
        print(f"Figure {entry['fig_num']} is currently unavailable.")
        print(f"Reason: {entry['note']}")
        sys.exit(0)

    os.makedirs(args.save_dir, exist_ok=True)

    # Resolve eval_results to absolute path
    eval_results_dir = os.path.abspath(args.eval_results_dir)
    os.makedirs(eval_results_dir, exist_ok=True)

    # Import the figure module
    fig_module = importlib.import_module(f"figure_utils.{entry['module']}")

    save_path = os.path.join(args.save_dir, f"figure{entry['fig_num']}.pdf")
    save_path = os.path.abspath(save_path)

    print(f"\nReproducing Figure {entry['fig_num']}: {args.figure}")
    print(f"  Data source: {args.data_source}")
    print(f"  Save path: {save_path}")

    fig_num = entry["fig_num"]

    # Handle special cases
    if fig_num == 11:
        save_path = handle_figure11(fig_module, args, save_path)
    elif fig_num == 15:
        handle_figure15(fig_module, args, save_path, eval_results_dir)
    elif fig_num == 17:
        handle_figure17(fig_module, args, save_path, eval_results_dir)
    elif fig_num == 18:
        handle_figure18(fig_module, args, save_path, eval_results_dir)
    elif fig_num == 19:
        handle_figure19(fig_module, save_path)
    elif fig_num == 20:
        handle_figure20(fig_module, save_path)
    else:
        # General CDF figures: load data for each required config
        data = {}
        for config_name in entry["configs"]:
            print(f"\nLoading data for config: {config_name}")
            if args.data_source == "cache":
                json_path = find_cache_json(eval_results_dir, config_name)
                print(f"  Found cache: {json_path}")
            else:  # from_scratch
                json_path = run_evaluation_from_scratch(
                    config_name, args.num_samples, eval_results_dir
                )

            data[config_name] = load_json_as_df_dict(json_path)
            print(f"  Loaded {len(data[config_name])} models: {list(data[config_name].keys())}")

        fig_module.plot(data, save_path)

    print(f"\nFigure {fig_num} saved to: {save_path}")


if __name__ == "__main__":
    main()
