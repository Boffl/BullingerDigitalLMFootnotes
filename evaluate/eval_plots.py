import pandas as pd
import re
import matplotlib.pyplot as plt

# Define the UZH color map
uzh_colors = ['#0028A5',  # UZH Blue
              '#4AC9E3',  # UZH Cyan
              '#A4D233',  # UZH Apple
              '#FFC845',  # UZH Gold
              '#FC4C02',  # UZH Orange
              '#BF0D3E',  # UZH Berry
              '#000000']  # UZH Black

def get_plot_df(adapters_to_plot:list, len_data=3450):
    plot_data = {
        "size": [],
        "adapters": [],
        "ppl": []
    }
    df = pd.read_csv("ppl_eval.csv")
    df = df[df["len_data"]==len_data]
    for _, row in df.iterrows():
        if row.len_data == len_data:
            model_match = re.match(r"llama-(8B|70B)-(.*)", row.model)
            size = model_match.group(1)
            adapters = model_match.group(2)
            if adapters in adapters_to_plot:
                plot_data["size"].append(size)
                plot_data["adapters"].append(adapters)
                plot_data["ppl"].append(row.score)
    return pd.DataFrame(plot_data)

def get_plot_df_subsets(adapters_to_plot:list, subsets_to_plot:list):
    """different subsets of the dev-set """

    subset_map = {
        "Z": 23,
        "Zwa": 3,
        "bible": 422,
        "EA": 32
    }
    full_df = pd.DataFrame()

    for subset in subsets_to_plot:
        subset_df = get_plot_df(adapters_to_plot, len_data=subset_map[subset])
        subset_df["subset"] = subset
        full_df = pd.concat([full_df,subset_df])
    return full_df


def plot_ppl(data, figsize=(10, 8)):
    """
    Plots PPL values by size and adapter with a horizontal baseline for the "base" adapter,
    and separate subplots for each size. Includes a unified legend.

    Parameters:
    - data: pandas DataFrame containing 'size', 'adapters', and 'ppl' columns.
    - If only one 'size' value, the DF needs to contain a 'subset' column, with different subsets
    """
    unique_sizes = data['size'].unique()
    category_name = "size"
    if len(unique_sizes) == 1:
        unique_sizes = data["subset"].unique()
        category_name = "subset"
    fig, axes = plt.subplots(1, len(unique_sizes), figsize=figsize, sharey=True)

    # colors = {adapter: uzh_colors[i % len(uzh_colors)] for i, adapter in enumerate(data['adapters'].unique())}
    colors = {adapter: uzh_colors[0] for adapter in data["adapters"].unique()}
    colors["base"] = uzh_colors[-1]
    
    # Dictionary to store handles for legend
    legend_handles = {}

    for ax, cat in zip(axes, unique_sizes):
        # Subset the data for the current size
        size_data = data[data[category_name] == cat]
        
        # Extract "base" adapter's PPL for the horizontal line
        base_ppl = size_data[size_data['adapters'] == 'base']['ppl'].values[0]
        base_line = ax.hlines(
            y=base_ppl,
            xmin=-0.5,
            xmax=len(size_data['adapters'].unique()) - 1,
            colors=colors['base'],
            label="Base"
        )
        if "Base" not in legend_handles:
            legend_handles["Baseline"] = base_line
        
        # Filter non-"base" adapters
        non_base_adapters = size_data[size_data['adapters'] != 'base']
        num_non_base_adapters = len(non_base_adapters)
        bar_width = 0.8 # 0.8 / num_non_base_adapters if num_non_base_adapters > 0 else 0.8
        
        # Create bars for non-"base" adapters
        adapter_list_sorted = sorted(list(non_base_adapters['adapters'].unique()))  # sort adapters, thus they show up in the same order
        for i, adapter in enumerate(adapter_list_sorted):
            adapter_data = non_base_adapters[non_base_adapters['adapters'] == adapter]
            bar = ax.bar(
                [i],
                adapter_data['ppl'],
                bar_width,
                color=colors[adapter],
                alpha=0.9
            )
            if "LORA Adapters" not in legend_handles:  # Collect handles only once
                legend_handles["LORA Adapters"] = bar[0]
        
        # Configure the x-axis for this subplot
        ax.set_xticks([i for i in range(len(adapter_list_sorted))])
        ax.set_xticklabels(adapter_list_sorted, rotation=45)
        ax.set_title(f"{category_name}: {cat}".title())
        ax.set_xlabel("")
        ax.set_ylabel("PPL" if cat == unique_sizes[0] else "")
        ax.grid(True, axis="y", linestyle='--', alpha=0.6)
    
      # Add a single shared legend at the bottom
    fig.legend(
        handles=legend_handles.values(),
        labels=legend_handles.keys(),
        title="",
        loc='lower center',
        ncol=len(legend_handles),  # Arrange legend in a single row
        bbox_to_anchor=(0.5, -0.05)
    )
    # Adjust layout and display the plots
    plt.tight_layout()
    return fig
    plt.show()

import pandas as pd
import matplotlib.pyplot as plt

def plot_model_metrics(df, model_size, bert_ylim=None):
    """
    Plot metrics (BLEU, ROUGE, BERT) for a given model size from the provided DataFrame.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the metrics. Created with the compare function
        model_size (str): The model size to filter ('8B' or '70B').
        bert_ylim (tuple, optional): Tuple specifying the y-axis range for the BERT metric (e.g., (0.4, 0.75)).
    """
    # Filter models based on size
    models = [model for model in df.index if model_size in model]
    metrics = ["bleu", "rouge", "bert"]

    # Setting up subplots for metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

    for ax, metric in zip(axes, metrics):
        bar_width = 0.2
        x = [0, 1]  # Index positions for "Without Markup" and "With Markup"

        # Extracting data for each model
        bars = {}
        for model in models:
            bars[model] = [
                df.loc[model, f"{metric}_without"],
                df.loc[model, f"{metric}_with"]
            ]

        # Plotting bars for each model
        for idx, (model, values) in enumerate(bars.items()):
            ax.bar(
                [pos + (idx - len(bars) / 2) * bar_width for pos in x],
                values,
                bar_width,
                label="-".join(model.split('-')[2:]).capitalize()  # Simplify model labels
            )

        # Setting titles and labels
        ax.set_title(f"{metric.upper()} Metric ({model_size} Models)")
        ax.set_xticks(x)
        ax.set_xticklabels(["Without Markup", "With Markup"])
        ax.set_ylabel("Metric Value")

        # Adjust y-axis for BERT metric if provided
        if metric == "bert" and bert_ylim:
            ax.set_ylim(bert_ylim)

        ax.legend()

    plt.suptitle(f"Metrics Grouped by Markup Type ({model_size} LLaMA Models)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def plot_model_metrics_with_ci(avg_df, model_size, std_df=None, bert_ylim=None):
    """
    Plot metrics (BLEU, ROUGE, BERT) for a given model size with optional 95% confidence intervals.
    
    Parameters:
        avg_df (pd.DataFrame): DataFrame containing the average metrics.
        std_df (pd.DataFrame, optional): DataFrame containing the standard deviations of the metrics.
                                          If None, CIs are not included.
        model_size (str): The model size to filter ('8B' or '70B').
        bert_ylim (tuple, optional): Tuple specifying the y-axis range for the BERT metric (e.g., (0.4, 0.75)).
    """
    colors = ['#0028A5',  # UZH Blue
              '#FFC845',  # UZH Gold
              '#BF0D3E',  # UZH Berry
              '#000000']  # UZH Black

    if model_size is None:
        raise ValueError("You must provide a model_size ('8B' or '70B').")
    
    # Filter models based on size
    models = [model for model in avg_df.index if model_size in model]
    metrics = ["bleu", "rouge", "bert"]

    # Setting up subplots for metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

    for ax, metric in zip(axes, metrics):
        bar_width = 0.2
        x = [0, 1]  # Index positions for "Without Markup" and "With Markup"

        # Extracting data for each model
        bars_avg = {}
        bars_std = {}
        for model in models:
            bars_avg[model] = [
                avg_df.loc[model, f"{metric}_without"],
                avg_df.loc[model, f"{metric}_with"]
            ]
            if std_df is not None:
                bars_std[model] = [
                    std_df.loc[model, f"{metric}_without"],
                    std_df.loc[model, f"{metric}_with"]
                ]

        # Plotting bars for each model with or without error bars
        for idx, (model, values_avg) in enumerate(bars_avg.items()):
            if std_df is not None:
                values_std = bars_std[model]
                error = [1.96 * std / (len(models)**0.5) for std in values_std]  # 95% CI
            else:
                error = None

            ax.bar(
                [pos + (idx - len(bars_avg) / 2) * bar_width for pos in x],
                values_avg,
                bar_width,
                yerr=error,
                capsize=5 if error else 0,
                # color=colors[idx % len(colors)],  # Cycle through UZH colors
                label="-".join(model.split('-')[2:]).title()  # Simplify model labels
            )

        # Setting titles and labels
        ax.set_title(f"{metric.upper()}")
        ax.set_xticks(x)
        ax.set_xticklabels(["Without Markup", "With Markup"])
        ax.set_ylabel("")
        ax.grid(True, axis="y", linestyle='--', alpha=0.6)

        # Adjust y-axis for BERT metric if provided
        if metric == "bert" and bert_ylim:
            ax.set_ylim(bert_ylim)

        # remove legend from subplots
        ax.legend().remove()
    
    # show lengend once (the same for all)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend([handle for handle in handles], 
            [label for label in labels], 
            loc='lower center',
        ncol=len(handles),  # Arrange legend in a single row
        bbox_to_anchor=(0.5, -0.05))

    # plt.suptitle(f"({model_size} LLaMA Models)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig




if __name__ == "__main__":
    df = (get_plot_df(["EA", "base"], len_data=10))
    plot_ppl(df)
