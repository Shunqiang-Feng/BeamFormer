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
        "note": "2x2 CDF: (RSS Loss, AoD Error) x (LoS, NLoS), indoor+outdoor combined",
        "module": "figure10",
        "configs": [
            "compare_vs_baselines_indoor",
            "compare_vs_baselines_outdoor",
            "compare_vs_baselines_indoor_nlos",
            "compare_vs_baselines_outdoor_nlos",
        ],
    },
    "Impact of scene configurations": {
        "status": "available",
        "fig_num": 11,
        "note": "Cross-frequency, scenario, array size, SNR CDFs",
        "module": "figure11",
        "configs": ["cross_frequency", "cross_scenarios", "cross_array_size", "cross_snr"],
    },
    "Failure case under severe multi-paths": {
        "status": "available",
        "fig_num": 12,
        "note": "Beam spectrum visualization (GT & Pred) for multi-path failure case",
        "module": "figure12",
        "configs": [],
    },
    "Failure case under pure noise": {
        "status": "available",
        "fig_num": 13,
        "note": "Beam spectrum visualization (GT=zeros, Pred) for pure-noise input",
        "module": "figure13",
        "configs": [],
    },
    "Beam Spectrum Resolution": {
        "status": "available",
        "fig_num": 14,
        "note": "Beam spectrum along SLERP path between two paths for diy_28G_csi dataset",
        "module": "figure14",
        "configs": [],
    },
    "Multi Path Prediction Accuracy": {
        "status": "available",
        "fig_num": 15,
        "note": "CDF of RSS prediction error for 1st/2nd/3rd strongest paths",
        "module": "figure15",
        "configs": [],
    },
    "Comparison of model latencies": {
        "status": "available",
        "fig_num": 16,
        "note": "Reads from eval_results/running_time_comparison/cached/ (from_scratch runs running_time_analyse.py)",
        "module": "figure16",
        "configs": [],
    },
    "Impact of latency and user mobility": {
        "status": "available",
        "fig_num": 17,
        "note": "Mean RSS Loss vs latency for different speeds, using mobility_dataset",
        "module": "figure17",
        "configs": [],
    },
    "Comparison of positional encoders": {
        "status": "available",
        "fig_num": 18,
        "note": "CDF comparing Array Factor, 2D Positional, and Concat position encodings",
        "module": "figure18",
        "configs": ["cross_encoder"],
    },
    "Comparison of reference beam settings": {
        "status": "available",
        "fig_num": 19,
        "note": "Beam settings CDF",
        "module": "figure19",
        "configs": ["compare_beam_settings"],
    },
    "Comparison of power estimators": {
        "status": "available",
        "fig_num": 20,
        "note": "RSS Error CDF comparing ARN power estimator vs Normal Distribution (outdoor)",
        "module": "figure20",
        "configs": ["test_arn_performance_outdoor"],
    },
    "Impact of model parameters": {
        "status": "available",
        "fig_num": 21,
        "note": "Ablation study CDFs",
        "module": "figure21",
        "configs": ["ablation_study"],
    },
    "Performance with real-world data": {
        "status": "available",
        "fig_num": 22,
        "note": "CDF plots from real-world Sivers/IBM hardware evaluations",
        "module": "figure22",
        "configs": [],
    },
    "Example spectrum with Sivers SDR": {
        "status": "available",
        "fig_num": 23,
        "note": "Beam spectrum visualization from Sivers SDR (requires model + data)",
        "module": "figure23",
        "configs": [],
    },
    "Example spectrum with IBM SDR": {
        "status": "available",
        "fig_num": 24,
        "note": "Beam spectrum visualization from IBM SDR (requires model + data)",
        "module": "figure24",
        "configs": [],
    },
    "UMAP Visualization of Simulation and Real-World Feature Distributions": {
        "status": "available",
        "fig_num": 25,
        "note": "UMAP 1x2 panel: sim vs. real-world Sivers input/output beam RSS features",
        "module": "figure25",
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

def handle_figure12(fig_module, save_path):
    """Handle Figure 12: multi-path failure case visualization."""
    save_path = fig_module.plot(save_path)
    return save_path


def handle_figure13(fig_module, save_path):
    """Handle Figure 13: pure-noise failure case visualization."""
    save_path = fig_module.plot(save_path)
    return save_path


def handle_figure14(fig_module, save_path):
    """Handle Figure 14: beam spectrum resolution visualization."""
    try:
        fig_module.plot(save_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error generating Figure 14: {e}")
        print(
            "\nFail to reproduce Figure 14\n"
        )
        sys.exit(1)


def handle_figure15(fig_module, args, save_path, eval_results_dir):
    """Handle Figure 15: multi-path prediction accuracy CDF."""
    from datetime import datetime

    results_path = None

    if args.data_source == "cache":
        candidate = os.path.join(eval_results_dir, "multi_path_eval_new", "results.json")
        if os.path.exists(candidate):
            results_path = candidate
            print(f"  Found cache: {results_path}")
        else:
            print(f"Cache not found at {candidate}, falling back to from_scratch ...")
            args.data_source = "from_scratch"

    if args.data_source == "from_scratch":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_json = os.path.join(eval_results_dir, "multi_path_eval_new",
                                 timestamp, "results.json")
        results_path = fig_module.run(
            num_samples=args.num_samples,
            save_path=save_json,
        )

    fig_module.plot(results_path, save_path)


def find_cache_csv_dir(eval_results_dir):
    """
    Find CSVs for Figure 16. Search order:
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


def handle_figure16(fig_module, args, save_path, eval_results_dir):
    """Handle Figure 16: model latency comparison."""
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


def find_cache_velocity_csv(eval_results_dir):
    """Find cached velocity test results CSV."""
    base_dir = os.path.join(eval_results_dir, "velocity_test")
    _csv = "results.csv"

    cached = os.path.join(base_dir, _csv)
    if os.path.exists(cached):
        return cached

    if os.path.isdir(base_dir):
        subdirs = sorted(
            [d for d in os.listdir(base_dir)
             if os.path.isdir(os.path.join(base_dir, d))],
            reverse=True
        )
        for subdir in subdirs:
            candidate = os.path.join(base_dir, subdir, _csv)
            if os.path.exists(candidate):
                return candidate

    raise FileNotFoundError(
        f"No velocity results CSV found under: {base_dir}\n"
        "Run with --data_source from_scratch to generate results."
    )


def handle_figure17(fig_module, args, save_path, eval_results_dir):
    """Handle Figure 17: latency and user mobility CDF."""
    from datetime import datetime

    csv_path = None

    if args.data_source == "cache":
        try:
            csv_path = find_cache_velocity_csv(eval_results_dir)
            print(f"  Found cached CSV: {csv_path}")
        except FileNotFoundError as e:
            print(f"Cache not found: {e}")
            print("Falling back to from_scratch ...")
            args.data_source = "from_scratch"

    if args.data_source == "from_scratch":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_csv = os.path.join(eval_results_dir, "velocity_test", timestamp, "results.csv")
        try:
            csv_path = fig_module.run(save_csv=save_csv)
        except Exception as e:
            print(f"Error running velocity test: {e}")
            print(
                "\nTo reproduce Figure 17, you need:\n"
                "  1. Trained ct_xa16 model at saved_models/ct_xa16/\n"
                "  2. Mobility dataset at csi-dataset/mobility_dataset/\n"
            )
            sys.exit(1)

    data = fig_module.load_from_csv(csv_path)
    fig_module.plot(data, save_path)


def find_hardware_results_json(eval_results_dir):
    """Find the most recent hardware_results.json in eval_results/realworld/."""
    realworld_dir = os.path.join(eval_results_dir, "realworld")
    if not os.path.isdir(realworld_dir):
        raise FileNotFoundError(
            f"No realworld results directory found at: {realworld_dir}\n"
            "Run with --data_source from_scratch to generate hardware evaluation results."
        )
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


def handle_figure22(fig_module, args, save_path, eval_results_dir):
    """Handle Figure 22: real-world hardware performance CDF."""
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
                "\nTo reproduce Figure 22, you need:\n"
                "  1. Trained models at saved_models/sivers/ and saved_models/ibm/\n"
                "  2. Hardware data at csi-dataset/realworld_sivers/ and csi-dataset/realworld_ibm/\n"
            )
            sys.exit(1)

    with open(json_path, 'r') as f:
        hw_results = json.load(f)

    fig_module.plot(hw_results, save_path)


def handle_figure23(fig_module, save_path):
    """Handle Figure 23: Sivers SDR spectrum example."""
    try:
        fig_module.plot(save_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error generating Figure 23: {e}")
        print(
            "\nTo reproduce Figure 23, you need:\n"
            "  1. Trained Sivers model at saved_models/sivers/\n"
            "  2. Sivers data at csi-dataset/realworld_sivers/indoor-nlos/1.pkl\n"
        )
        sys.exit(1)


def handle_figure24(fig_module, save_path):
    """Handle Figure 24: IBM SDR spectrum example."""
    try:
        fig_module.plot(save_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error generating Figure 24: {e}")
        print(
            "\nTo reproduce Figure 24, you need:\n"
            "  1. Trained IBM model at saved_models/ibm/\n"
            "  2. IBM data at csi-dataset/realworld_ibm/indoor-los/beamtable_20251203_193653/experiment_special\n"
        )
        sys.exit(1)


def handle_figure25(fig_module, save_path):
    """Handle Figure 25: UMAP visualization of sim vs. real-world distributions."""
    try:
        fig_module.plot(save_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error generating Figure 25: {e}")
        print(
            "\nTo reproduce Figure 25, you need:\n"
            "  1. Trained Sivers full-array model at saved_models/sivers_pi_param_co_train_16_snr_worse_indoor_fullarray/\n"
            "  2. Sivers simulation data at csi-dataset/homeoffice-communication-28G-csi-sivers-indoor-patch/t4x4_r2x1_test_small/\n"
            "  3. Sivers realworld data at csi-dataset/realworld_sivers/indoor-nlos/\n"
            "  4. umap-learn installed: pip install umap-learn\n"
        )
        sys.exit(1)


def handle_figure20(fig_module, args, save_path, eval_results_dir):
    """Handle Figure 20: power estimator comparison (requires random-peak evaluation)."""
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
    if fig_num == 12:
        handle_figure12(fig_module, save_path)
    elif fig_num == 13:
        handle_figure13(fig_module, save_path)
    elif fig_num == 14:
        handle_figure14(fig_module, save_path)
    elif fig_num == 15:
        handle_figure15(fig_module, args, save_path, eval_results_dir)
    elif fig_num == 16:
        handle_figure16(fig_module, args, save_path, eval_results_dir)
    elif fig_num == 17:
        handle_figure17(fig_module, args, save_path, eval_results_dir)
    elif fig_num == 20:
        handle_figure20(fig_module, args, save_path, eval_results_dir)
    elif fig_num == 22:
        handle_figure22(fig_module, args, save_path, eval_results_dir)
    elif fig_num == 23:
        handle_figure23(fig_module, save_path)
    elif fig_num == 24:
        handle_figure24(fig_module, save_path)
    elif fig_num == 25:
        handle_figure25(fig_module, save_path)
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
