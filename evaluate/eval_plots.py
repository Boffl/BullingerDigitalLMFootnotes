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





if __name__ == "__main__":
    df = (get_plot_df(["EA", "base"], len_data=10))
    plot_ppl(df)
