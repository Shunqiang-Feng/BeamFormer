# Main Function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import os
from matplotlib.font_manager import FontProperties
from itertools import cycle
import sys

# Add project root to path so imports from the project root work correctly
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import json
import pandas as pd
import os
from typing import Union, Dict, List, Optional

# Try to import calculate_angle_error from related_work.
# related_work.py has heavy top-level dependencies (matlab, etc.) that may not
# be available. If import fails, define the function locally using only numpy.
try:
    from related_work import calculate_angle_error
except Exception:
    def calculate_angle_error(gt_phi, gt_theta, pred_phi, pred_theta, return_degrees=True):
        """
        Calculate spherical distance (angle error) between ground truth and predicted angles.
        Fallback implementation using only numpy (identical logic to related_work.py).
        """
        phi1 = np.array(gt_phi)
        theta1 = np.array(gt_theta)
        phi2 = np.array(pred_phi)
        theta2 = np.array(pred_theta)

        if not (np.all(theta1 < 91) and np.all(theta2 < 91)):
            raise ValueError("All theta values must be less than 91 degrees.")

        phi1_rad = np.radians(phi1)
        theta1_rad = np.radians(theta1)
        phi2_rad = np.radians(phi2)
        theta2_rad = np.radians(theta2)

        x1 = np.sin(theta1_rad) * np.cos(phi1_rad)
        y1 = np.sin(theta1_rad) * np.sin(phi1_rad)
        z1 = np.cos(theta1_rad)

        x2 = np.sin(theta2_rad) * np.cos(phi2_rad)
        y2 = np.sin(theta2_rad) * np.sin(phi2_rad)
        z2 = np.cos(theta2_rad)

        dot_product = x1*x2 + y1*y2 + z1*z2
        dot_product = np.clip(dot_product, -1.0, 1.0)

        spherical_distance_rad = np.arccos(dot_product)

        if return_degrees:
            spherical_distance = np.degrees(spherical_distance_rad)
        else:
            spherical_distance = spherical_distance_rad

        if isinstance(gt_phi, pd.Series):
            return pd.Series(spherical_distance, index=gt_phi.index)
        else:
            return spherical_distance

def load_performance_json_to_df(json_path: str,
                               model_names: Optional[List[str]] = None,
                               add_model_column: bool = True) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Load performance data from JSON file to pandas DataFrame(s).

    Args:
        json_path (str): Path to the JSON file containing performance data
        model_names (List[str], optional): Specific model names to load. If None, load all models.
        add_model_column (bool): Whether to add a 'model_name' column when returning combined DataFrame

    Returns:
        pd.DataFrame or Dict[str, pd.DataFrame]:
            - If single model requested or combined=True: single DataFrame
            - If multiple models: dictionary with model names as keys and DataFrames as values

    Examples:
        # Load all models as separate DataFrames
        >>> data_dict = load_performance_json_to_df('results.json')
        >>> print(data_dict.keys())  # ['model1', 'model2', ...]

        # Load specific models
        >>> df = load_performance_json_to_df('results.json', model_names=['Param-SNR3', '2ACE'])

        # Load all models into single DataFrame
        >>> df = load_performance_json_to_df('results.json', combine_models=True)
    """

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON file: {e}")

    if not isinstance(data, dict):
        raise ValueError("JSON should contain a dictionary with model names as keys")

    # Filter models if specified
    if model_names is not None:
        missing_models = set(model_names) - set(data.keys())
        if missing_models:
            print(f"Warning: Models not found in JSON: {missing_models}")
        data = {k: v for k, v in data.items() if k in model_names}

    if not data:
        raise ValueError("No valid model data found")

    # Convert to DataFrames
    dataframes = {}
    for model_name, records in data.items():
        if not isinstance(records, list):
            print(f"Warning: Skipping {model_name} - data is not a list of records")
            continue

        df = pd.DataFrame(records)
        if add_model_column:
            df['model_name'] = model_name
        dataframes[model_name] = df

    # Return based on number of models
    if len(dataframes) == 1:
        return list(dataframes.values())[0]
    else:
        return dataframes


def load_and_combine_models(json_path: str,
                           model_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load performance data and combine all models into a single DataFrame.

    Args:
        json_path (str): Path to the JSON file
        model_names (List[str], optional): Specific model names to load

    Returns:
        pd.DataFrame: Combined DataFrame with 'model_name' column
    """
    dataframes_dict = load_performance_json_to_df(json_path, model_names)

    if isinstance(dataframes_dict, pd.DataFrame):
        return dataframes_dict

    # Combine all DataFrames
    all_dfs = []
    for model_name, df in dataframes_dict.items():
        df_copy = df.copy()
        if 'model_name' not in df_copy.columns:
            df_copy['model_name'] = model_name
        all_dfs.append(df_copy)

    return pd.concat(all_dfs, ignore_index=True)


def load_specific_model(json_path: str, model_name: str) -> pd.DataFrame:
    """
    Load data for a specific model from JSON file.

    Args:
        json_path (str): Path to the JSON file
        model_name (str): Name of the model to load

    Returns:
        pd.DataFrame: DataFrame for the specified model
    """
    result = load_performance_json_to_df(json_path, model_names=[model_name], add_model_column=False)

    if isinstance(result, dict):
        if model_name in result:
            return result[model_name]
        else:
            raise ValueError(f"Model '{model_name}' not found in JSON file")
    else:
        return result



# Set Arial font, base font size and high resolution
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']
# plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300  # Set display resolution
plt.rcParams['savefig.dpi'] = 300  # Set save resolution

def add_angle_error_for_dict(data):
    key_name = 'Our Method'
    data[key_name]['angle_error'] = calculate_angle_error(data[key_name]['gt_phi'], data[key_name]['gt_theta'],data[key_name]['pred_phi'], data[key_name]['pred_theta'])
    return data.copy()

def data_to_numpy(data):
    """Convert various data types to numpy array"""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy().flatten()
    elif isinstance(data, pd.Series):
        return data.values.flatten()
    elif isinstance(data, list):
        return np.array(data).flatten()
    elif isinstance(data, np.ndarray):
        return data.flatten()
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

def plot_multi_cdf(data_list, subplot_shape, save_path=None, main_title=None,
                   subplot_titles=None, xlabel="Value", ylabel="CDF",
                   show_grid=True, legend_position='lower right',
                   share_x=False, share_y=False,
                   line_alpha=0.8, x_limits=None,
                   y_limits=None, x_units=None, basic_fontsize=16,
                   dpi=300, h_w_ratio=0.85, only_1_legend=False):
    """
    Plot multiple CDF subplots, each subplot plotting area width is 3.33 inches,
    height is 3.33 * h_w_ratio inches.

    Parameters:
    data_list: list of dict, [{"tag": data, ...}, {"tag": data, ...}, ...]
    subplot_shape: tuple, (rows, cols) for subplot arrangement
    save_path: str or None, file save path
    main_title: str or None, main figure title
    subplot_titles: list or None, titles for each subplot
    xlabel: str or list, x-axis label for subplots
    ylabel: str, y-axis label for all subplots
    show_grid: bool, whether to show grid in subplots
    legend_position: str, legend position for each subplot
    share_x: bool, whether to share x-axis across subplots
    share_y: bool, whether to share y-axis across subplots
    line_alpha: float, line transparency
    x_limits: list of tuples or tuple, x-axis limits
    y_limits: list of tuples or tuple, y-axis limits
    x_units: list of str or str, units for x-axis
    basic_fontsize: int, base font size
    dpi: int, resolution for saving
    h_w_ratio: float, height to width ratio for subplot plotting area (default 0.75)
    only_1_legend: bool. If True, only the first subplot (idx==0) shows a legend.

    Returns:
        fig, axes
    """

    rows, cols = subplot_shape

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*']

    # Fixed subplot area size (inches)
    plot_width = 3.33
    plot_height = 3.33 * h_w_ratio

    # Margin settings (inches)
    left_margin = 0.8
    right_margin = 0.3
    bottom_margin = 0.7
    top_margin = 0.5
    h_spacing = 1.0
    v_spacing = 1.0
    main_title_space = 0.4 if main_title else 0

    # Calculate total figure size
    fig_width = left_margin + cols * plot_width + (cols - 1) * h_spacing + right_margin
    fig_height = bottom_margin + rows * plot_height + (rows - 1) * v_spacing + top_margin + main_title_space

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)

    # Add subplots using absolute coordinates
    axes = []
    for i in range(rows):
        for j in range(cols):
            x_pos = left_margin + j * (plot_width + h_spacing)
            y_pos = bottom_margin + (rows - 1 - i) * (plot_height + v_spacing)

            left = x_pos / fig_width
            bottom = y_pos / fig_height
            width = plot_width / fig_width
            height = plot_height / fig_height

            ax = fig.add_axes([left, bottom, width, height])
            axes.append(ax)

    if x_limits is None:
        x_limits = [None] * len(data_list)
    elif isinstance(x_limits, tuple):
        x_limits = [x_limits] * len(data_list)

    if y_limits is None:
        y_limits = [(0, 1)] * len(data_list)
    elif isinstance(y_limits, tuple):
        y_limits = [y_limits] * len(data_list)

    if isinstance(xlabel, str):
        xlabels = [xlabel] * len(data_list)
    else:
        xlabels = xlabel

    if x_units is None:
        x_units = [None] * len(data_list)
    elif isinstance(x_units, str):
        x_units = [x_units] * len(data_list)

    for idx, (ax, data_dict) in enumerate(zip(axes, data_list)):
        if data_dict is None:
            ax.axis('off')
            continue

        for style_idx, (tag, data) in enumerate(data_dict.items()):
            data = np.array(data)
            sorted_data = np.sort(data)
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

            color = colors[style_idx % len(colors)]
            linestyle = line_styles[style_idx % len(line_styles)]
            marker = markers[style_idx % len(markers)]
            markevery = max(1, len(sorted_data) // 10)

            ax.plot(sorted_data, cdf,
                   label=tag,
                   color=color,
                   linestyle=linestyle,
                   marker=marker,
                   markevery=markevery,
                   markersize=4,
                   alpha=line_alpha,
                   linewidth=2)

        if idx < len(x_limits) and x_limits[idx] is not None:
            ax.set_xlim(x_limits[idx])
        if idx < len(y_limits) and y_limits[idx] is not None:
            ax.set_ylim(y_limits[idx])

        if idx < len(xlabels):
            label_text = f"{xlabels[idx]} ({x_units[idx]})" if x_units[idx] else xlabels[idx]
            ax.set_xlabel(label_text, fontsize=basic_fontsize)

        ax.set_ylabel(ylabel, fontsize=basic_fontsize)

        if subplot_titles and idx < len(subplot_titles):
            ax.set_title(subplot_titles[idx], fontsize=basic_fontsize + 1, fontweight='bold')

        if show_grid:
            ax.grid(True, linestyle='--', alpha=0.6)

        if data_dict:
            if not only_1_legend:
                ax.legend(loc=legend_position, fontsize=basic_fontsize - 2)
            else:
                if idx == 0:
                    ax.legend(loc=legend_position, fontsize=basic_fontsize - 2)

        ax.tick_params(axis='both', labelsize=basic_fontsize - 2)

    if main_title:
        fig.suptitle(main_title, fontsize=basic_fontsize + 2)

    if share_x:
        for i, ax in enumerate(axes):
            if i < (rows - 1) * cols:
                ax.set_xlabel('')
                ax.tick_params(axis='x', labelbottom=False)

    if share_y:
        for i, ax in enumerate(axes):
            if i % cols != 0:
                ax.set_ylabel('')
                ax.tick_params(axis='y', labelleft=False)

    if save_path:
        fig.savefig(save_path, dpi=dpi)
        print(f"Figure saved to {save_path}")

    plt.close()

    return fig, axes
