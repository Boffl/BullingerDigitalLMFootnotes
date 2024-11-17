import matplotlib.pyplot as plt
from cycler import cycler
from collections import Counter
import pandas as pd

# Example Counter object
counter_obj = Counter(['a', 'b', 'a', 'c', 'a', 'b', 'c', 'c', 'a', 'b'])

cmap = plt.get_cmap("tab10")

# uzh color map
# Define the UZH primary colors
uzh_colors = ['#0028A5',  # UZH Blue
              '#4AC9E3',  # UZH Cyan
              '#A4D233',  # UZH Apple
              '#FFC845',  # UZH Gold
              '#FC4C02',  # UZH Orange
              '#BF0D3E',  # UZH Berry
              '#3062FF',  # UZH Blue3
              '#C2C2C2',  # UZH Grey1
              '#C8E485']  # UZH Apple3

# Overwrite default color cycle with UZH colors
plt.rcParams['axes.prop_cycle'] = cycler(color=uzh_colors)

# remaining colors from uzh template
all_uzh_colors = {
    'UZH_Blue': '#0028A5',
    'Blue1': '#BDC9E8',
    'Blue2': '#7596FF',
    'Blue3': '#3062FF',
    'Blue4': '#001E7C',
    'Blue5': '#001452',
    'UZH_Cyan': '#4AC9E3',
    'Cyan1': '#DBF4F9',
    'Cyan2': '#B7E9F4',
    'Cyan3': '#92DFEE',
    'Cyan4': '#1EA7C4',
    'Cyan5': '#147082',
    'UZH_Apple': '#A4D233',
    'Apple1': '#ECF6D6',
    'Apple2': '#DBEDAD',
    'Apple3': '#C8E485',
    'Apple4': '#7CA023',
    'Apple5': '#536B18',
    'UZH_Gold': '#FFC845',
    'Gold1': '#FFF4DA',
    'Gold2': '#FFE9B5',
    'Gold3': '#FFDE8F',
    'Gold4': '#F3AB00',
    'Gold5': '#A27200',
    'UZH_Orange': '#FC4C02',
    'Berry': '#BF0D3E',
    'UZH_Black': '#000000',
    'Grey1': '#C2C2C2',
    'Grey2': '#A3A3A3',
    'Grey3': '#666666',
    'Grey4': '#4D4D4D',
    'Grey5': '#333333',
    'UZH_White': '#FFFFFF'
}



def plot_label_pie_chart(df, labels_to_show=None):
    """
    Plots a pie chart of label frequencies with a legend for the labels, using a consistent colormap,
    and returns the label_colors dictionary for reuse in other plots.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing 'edition' and 'label' columns.
    labels_to_show (list, optional): List of labels to include in the plot. If None, all labels are shown.
    
    Returns:
    fig (matplotlib.figure.Figure): The pie chart figure.
    label_colors (dict): A dictionary mapping each label to its corresponding color.
    """

    df_split = df['label'].str.split(', ', expand=True).stack().reset_index(level=1, drop=True)
    df_split = pd.DataFrame(df_split, columns=['label'])

    label_counts = df_split['label'].value_counts()

    # Filter the labels to show if provided
    if labels_to_show is not None:
        label_counts = label_counts[labels_to_show]

    # Use the tab10 colormap and create a dictionary for consistent colors
    cmap = plt.get_cmap('tab10')
    all_labels = label_counts.index.tolist()
    label_colors = {label: cmap(i % 10) for i, label in enumerate(all_labels)}

    colors_to_use = [label_colors[label] for label in label_counts.index]

    # Plot the pie chart with percentage labels outside the pie
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(label_counts, startangle=90, colors=colors_to_use, autopct='%1.1f%%', 
                                      pctdistance=1.08)  # Move percentage labels outside the pie

    # Format the percentage text
    for autotext in autotexts:
        autotext.set_color('black')  # Set text color to black for better readability
        autotext.set_fontsize(12)    # Adjust font size if necessary

    # Add a legend outside the pie chart
    ax.legend(wedges, label_counts.index, title="", loc="center left", bbox_to_anchor=(0,1))

    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')

    plt.title('', fontsize=16)

    # Return both the figure and the label_colors dictionary
    return fig, label_colors



def lit_bar(counter_obj, max, exeed=True, label_dict={}):
    # Sort the Counter object by values
    sorted_items = counter_obj.most_common()

    # Select only the top categories adding up to max percentage
    top_categories = []
    others_count = 0
    total_count = sum(counter_obj.values())
    for item in sorted_items:
        if (others_count + item[1]) / total_count <= max:
            top_categories.append(item)
            others_count += item[1]
        else:
            if exeed:  # Add one more category if exeed is True
                top_categories.append(item)
            break

    # Get labels and sizes for the bar chart
    labels = [item[0] for item in top_categories]
    sizes = [item[1] for item in top_categories]
    percentages = [size / total_count * 100 for size in sizes]

    # Define colors based on label categories
    colors = [
        uzh_colors[0] if label_dict.get(label) == 'primary' 
        else uzh_colors[3] if label_dict.get(label) == 'secondary' 
        else uzh_colors[7]  # Default to grey if label not in label_dict
        for label in labels
    ]

    # Plotting the bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, sizes, color=colors)
    plt.xlabel('')
    plt.ylabel('Counts')
    plt.title(f'Total share of occurrences: {round(sum(percentages))}%')
    plt.xticks(rotation=90, fontsize=15)

    # Adding percentages above the bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{pct:.1f}%',
            ha='center',
            va='bottom',
            color='black'
        )

    # Adding a custom legend
    plt.legend(
        handles=[
            plt.Line2D([0], [0], color=uzh_colors[0], lw=4, label='Primary Source'),
            plt.Line2D([0], [0], color=uzh_colors[3], lw=4, label='Secondary Source'),
        ],
        loc='upper right'
    )

    




import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict

def label_trends(df, label_colors, labels_to_show=None, ax=None, title=''):
    """
    Plots the percentage trends of labels over editions on the given axis.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing 'edition' and 'label' columns.
    label_colors (dict): Dictionary of label to color mappings.
    labels_to_show (list, optional): List of labels to include in the plot. If None, all labels are shown.
    ax (matplotlib.axes.Axes, optional): The axis on which to plot the graph.
    title (str, optional): Title of the plot.
    
    Returns:
    ax (matplotlib.axes.Axes): The axis with the plot.
    """
    
    # Split labels if there are multiple labels in a row
    df_split = df['label'].str.split(', ', expand=True).stack().reset_index(level=1, drop=True)
    
    # Create a new DataFrame with the split labels
    df_split = pd.DataFrame(df_split, columns=['label'])
    df_split['edition'] = df['edition'].repeat(df['label'].str.split(', ').apply(len))

    # Group by edition and label, and calculate the count of each label in each edition
    grouped = df_split.groupby(['edition', 'label']).size().reset_index(name='Count')

    # Pivot the table to have editions as rows and labels as columns
    pivot_table = grouped.pivot_table(index='edition', columns='label', values='Count', fill_value=0)

    # Calculate the percentage of each label in each edition (relative to the full label set)
    percentages = pivot_table.div(pivot_table.sum(axis=1), axis=0) * 100

    # Ensure 'edition' is treated as a category and sort it
    percentages.index = pd.Categorical(percentages.index, ordered=True)
    percentages = percentages.sort_index()

    # Filter by the labels to show if provided
    if labels_to_show is not None:
        percentages = percentages[labels_to_show]

    # Create a list of colors based on the labels being shown
    colors_to_use = [label_colors[label] for label in percentages.columns]

    # Plot the percentages on the provided axis
    percentages.plot(kind='line', marker='o', ax=ax, color=colors_to_use)
    
    # Plot formatting
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Edition', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.grid(True)

    return ax

def plot_fn_per_sent_by_edition(letter_df):
    # Filter the data
    plot_df = letter_df[letter_df["footnotes_per_sentence"] > 0]

    # Create a figure and axis objects with custom figure size
    fig, ax = plt.subplots(figsize=(10, 6))

    medianprops = dict(color="black")
    # Plot the boxplot on the created axis
    plot_df.boxplot(column="footnotes_per_sentence", by="edition", showfliers=False, ax=ax,medianprops=medianprops)

    # Set the title, xlabel, ylabel, and adjust grid
    fig.suptitle('')  # Main title
    ax.set_title('')  # Remove the default subtitle
    ax.set_xlabel('Edition')
    ax.set_ylabel('Footnotes per Sentence')

    # Custom grid
    ax.grid(True, axis="y", linestyle='--', alpha=0.6)
    ax.grid(True, axis="x", linestyle='', alpha=0.6)
    return fig

def plot_fn_len_and_density(footnote_df, letter_df):
    # Create a figure with two subplots, side by side (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))  # Adjust the figure size

    # First plot: Histogram of footnote length
    plot_series_footnote = footnote_df["len_footnote"]
    num_bins_footnote = (footnote_df["len_footnote"].max() - footnote_df["len_footnote"].min()) // 10
    ax1.hist(plot_series_footnote, bins=num_bins_footnote, edgecolor='black', alpha=0.7)
    ax1.set_title('Footnote Length', fontsize=14)
    ax1.set_xlabel('Number of Words per Footnote', fontsize=12)
    ax1.set_ylabel('', fontsize=12)
    ax1.set_yscale("log")
    ax1.grid(True, axis="y", linestyle='--', alpha=0.6)

    # Second plot: Histogram of footnotes per sentence
    plot_series_sentence = letter_df["footnotes_per_sentence"]
    plot_series_sentence = plot_series_sentence[plot_series_sentence > 0]
    ax2.hist(plot_series_sentence, bins=50, edgecolor='black', alpha=0.7)
    ax2.set_title('Average Number of Footnotes per Sentence in each letter', fontsize=14)
    ax2.set_xlabel('Footnotes per Sentence', fontsize=12)
    ax2.set_ylabel('', fontsize=12)
    ax2.grid(True, axis="y", linestyle='--', alpha=0.6)

    # Adjust layout for better spacing
    plt.tight_layout()

    return fig

def plot_combined_footnote_and_sentence(footnote_df, letter_df, by_edition=True, showfliers=False):
    # Create a figure with two subplots, side by side (1 row, 2 columns)
    if by_edition:
        fig_height = 6
    else:
        fig_heigth = 12
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 12))  # Adjust the figure size

    if by_edition == True:
        by = "edition"
    else:
        by = None

    # Plot footnote length by edition in the first subplot
    medianprops = dict(color='black')
    footnote_df.boxplot(column="len_footnote", by=by, showfliers=showfliers, ax=ax1, medianprops=medianprops)
    ax1.set_title('Footnote Length', fontsize=18)
    if by_edition:
        ax1.set_xlabel('Edition', fontsize=12)
    else:
        ax1.set_xticklabels([])
    ax1.set_ylabel('', fontsize=12)
    ax1.grid(True, axis="y", linestyle='--', alpha=0.6)

    # Remove the default "Boxplot of ..." title
    ax1.get_figure().suptitle('')

    # Filter data for footnotes per sentence
    plot_df = letter_df[letter_df["footnotes_per_sentence"] > 0]

    # Plot footnotes per sentence by edition in the second subplot
    plot_df.boxplot(column="footnotes_per_sentence", by=by, showfliers=showfliers, ax=ax2, medianprops=medianprops)
    ax2.set_title('Average number of Footnotes per Sentence', fontsize=18)
    if by_edition:
        ax2.set_xlabel('Edition', fontsize=12)
    else:
        ax2.set_xticklabels([])
    ax2.set_ylabel('', fontsize=12)
    ax2.grid(True, axis="y", linestyle='--', alpha=0.6)

    # Remove the default "Boxplot of ..." title
    ax2.get_figure().suptitle('')

    # Adjust layout for better spacing
    plt.tight_layout()

    return fig

def plot_footnote_length_by_edition(footnote_df):
    # Create a figure and axis objects with custom figure size
    fig, ax = plt.subplots(figsize=(10, 6))

    # Customizing the appearance of the median line to be black
    medianprops = dict(color='black')

    # Create a boxplot for the 'len_footnote' column, grouped by 'edition'
    footnote_df.boxplot(column="len_footnote", by="edition", showfliers=False, ax=ax, medianprops=medianprops)


    # Set the title, xlabel, ylabel, and adjust grid
    fig.suptitle('')  # Main title
    ax.set_title('')  # Remove the default subtitle
    ax.set_xlabel('Edition')
    ax.set_ylabel('Footnote Length')

    # Custom grid
    ax.grid(True, axis="y", linestyle='--', alpha=0.6)
    ax.grid(True, axis="x", linestyle='', alpha=0.6)
    return fig

def plot_combined_label_trends(df, label_colors, labels_frequent, labels_infrequent):
    """
    Combines the frequent and infrequent label trends into a single figure with subplots.
    
    Parameters:
    df_frequent (pd.DataFrame): DataFrame for frequent labels.
    df_infrequent (pd.DataFrame): DataFrame for infrequent labels.
    label_colors (dict): Dictionary of label to color mappings.
    labels_frequent (list): List of frequent labels.
    labels_infrequent (list): List of infrequent labels.
    
    Returns:
    fig (matplotlib.figure.Figure): The combined figure.
    """
    
    # Create the figure with two subplots (side by side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))  # Share y-axis

    # Plot frequent labels on the first axis
    label_trends(df, label_colors, labels_frequent, ax=ax1, title='Frequent Labels')

    # Plot infrequent labels on the second axis
    label_trends(df, label_colors, labels_infrequent, ax=ax2, title='Infrequent Labels')

    # Combine legends
    handles, labels = ax1.get_legend_handles_labels()  # Get legend items from the first axis
    handles2, labels2 = ax2.get_legend_handles_labels()  # Get legend items from the second axis
    
    # Combine the labels and handles into a single list to avoid duplicates
    combined_handles = handles + handles2
    combined_labels = labels + labels2

    # Remove duplicates while preserving order
    unique_legend = list(OrderedDict(zip(combined_labels, combined_handles)).items())

    # Place the combined legend in the center right
    fig.legend([handle for label, handle in unique_legend], 
               [label for label, handle in unique_legend], 
               title="Labels", bbox_to_anchor=(0.85, 0.5), loc='center left')
    ax1.legend().remove()
    ax2.legend().remove()

    # Tight layout and return the figure
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to fit legend
    return fig

# Example usage:
# fig = plot_combined_label_trends(df_frequent, df_infrequent, label_colors, labels_frequent, labels_infrequent)
# plt.show()  # Or save it using plt.savefig('combined_label_trends.png')




def label_pie(df):
    grouped = df.groupby('label').size().reset_index(name='Count')

    # Generate a list of colors for each label
    num_colors = len(grouped)
    colors = [cmap(i) for i in range(num_colors)]

    # Plot a pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(grouped['Count'], labels=grouped['label'], autopct='%1.1f%%', startangle=140, colors=colors, radius=0.5)
    plt.title('')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

    plt.show()

