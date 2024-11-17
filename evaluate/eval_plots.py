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

def plot_ppl(df):
        # Re-plot using the UZH color map
    fig, ax = plt.subplots(figsize=(8, 6))

    # Assign colors to adapters using the UZH color map
    adapter_colors = {adapter: uzh_colors[i] for i, adapter in enumerate(df['adapters'].unique())}

    # Adjusting the bar width and positions for grouping
    bar_width = 0.35
    x = range(len(df['size'].unique()))  # Positions for the groups

    # Create separate bars for each adapter type
    for i, adapter in enumerate(df['adapters'].unique()):
        subset = df[df['adapters'] == adapter]
        ax.bar(
            [pos + (i * bar_width) for pos in x],
            subset['ppl'],
            bar_width,
            label=adapter,
            color=adapter_colors[adapter],
            alpha=0.9
        )

    # Configure the x-axis with group labels
    ax.set_xticks([pos + bar_width / 2 for pos in x])
    ax.set_xticklabels(df['size'].unique())

    # Add labels and title
    ax.set_xlabel("Size")
    ax.set_ylabel("PPL")
    ax.set_title("PPL by Size and Adapters")
    ax.legend(title="Adapters")

    # Show the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = (get_plot_df(["EA", "base"], len_data=10))
    plot_ppl(df)
