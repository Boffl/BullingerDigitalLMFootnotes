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
colors = {
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
    Plots a pie chart of label frequencies, using a consistent colormap,
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

    # Plot the pie chart
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=90, colors=colors_to_use)

    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')

    plt.title('', fontsize=16)

    # Return both the figure and the label_colors dictionary
    return fig, label_colors



def bar(counter_obj, max, exeed=True):
   # Sort the Counter object by values
    sorted_items = counter_obj.most_common()

    # Select only the top categories adding up to 40%
    top_categories = []
    others_count = 0
    total_count = sum(counter_obj.values())
    for item in sorted_items:
        if (others_count + item[1]) / total_count <= max:
            top_categories.append(item)
            others_count += item[1]
        else:
            if exeed:  # add one more
                top_categories.append(item)
            break

    # Get labels and sizes for the bar chart
    labels = [item[0] for item in top_categories]
    sizes = [item[1] for item in top_categories]
    percentages = [size / total_count * 100 for size in sizes]

    # Plotting the bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, sizes, color='skyblue')
    plt.xlabel('')
    plt.ylabel('Counts')
    plt.title(f'Total share of occurences: {round(sum(percentages))}%')
    plt.xticks(rotation=45)

    # Adding percentages above the bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{pct:.1f}%', ha='center', va='bottom', color='black')

    plt.show()


def label_trends(df, label_colors, labels_to_show=None):
    """
    Plots the percentage trends of labels over editions, using a given color mapping.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing 'edition' and 'label' columns.
    label_colors (dict): Dictionary of label to color mappings from the pie chart.
    labels_to_show (list, optional): List of labels to include in the plot. If None, all labels are shown.
    
    Returns:
    fig (matplotlib.figure.Figure): The line chart figure.
    """
    
    # Split labels if there are multiple labels in a row (assumes labels are separated by ', ')
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

    # Filter by the labels to show if provided (after calculating percentages)
    if labels_to_show is not None:
        percentages = percentages[labels_to_show]

    # Create a list of colors based on the labels being shown
    colors_to_use = [label_colors[label] for label in percentages.columns]

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the percentages
    percentages.plot(kind='line', marker='o', ax=ax, color=colors_to_use)
    
    # Plot formatting
    plt.title('', fontsize=16)
    plt.xlabel('Edition', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.grid(True)

    # Adjust the legend to be outside the plot on the right
    plt.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    # Tighten the layout
    plt.tight_layout()
    
    return fig



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

