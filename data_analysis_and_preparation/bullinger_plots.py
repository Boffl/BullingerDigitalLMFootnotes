import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd

# Example Counter object
counter_obj = Counter(['a', 'b', 'a', 'c', 'a', 'b', 'c', 'c', 'a', 'b'])

# nice color map:
cmap = plt.get_cmap('tab10')

def pie(counter_obj, max):
    # Sort the Counter object by values
    sorted_items = counter_obj.most_common()

    # Separate the top categories adding up to max%
    top_categories = []
    others_count = 0
    total_count = sum(counter_obj.values())
    for item in sorted_items:
        if (others_count + item[1]) / total_count <= max:
            top_categories.append(item)
            others_count += item[1]
        else:
            break

    # Add "others" category if there are remaining categories
    if others_count < total_count:
        top_categories.append(('others', total_count - others_count))

    # Get labels and sizes for the pie chart
    labels = [item[0] for item in top_categories]
    sizes = [item[1] for item in top_categories]
    percentages = [size / total_count for size in sizes]

    # Plotting the pie chart
    plt.figure(figsize=(8, 6))
    wedges, texts, autotexts = plt.pie(sizes, autopct='', startangle=140)

    # Adding labels and percentages into the legend
    legend_labels = [f'{label}: {percent:.1%}' for label, percent in zip(labels, percentages)]
    plt.legend(wedges, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    plt.title('Distribution of Values')
    
    plt.show()


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


def label_trends(df):
  
    # Group by Year and Label, and calculate the count of each label in each year
    grouped = df.groupby(['edition', 'label']).size().reset_index(name='Count')

    # Pivot the table to have years as rows and labels as columns
    pivot_table = grouped.pivot_table(index='edition', columns='label', values='Count', fill_value=0)

    # Calculate the percentage of each label in each year
    percentages = pivot_table.div(pivot_table.sum(axis=1), axis=0) * 100

    # Set edition column as index
    percentages.index = pd.Categorical(percentages.index)

    # Sort the index
    percentages = percentages.sort_index()


    # Generate a list of colors for each label
    num_colors = len(percentages.columns)
    colors = [cmap(i) for i in range(num_colors)]

    # Remove the misc label, as it does not add more information, it is just what is left from the pie
    # also, if it is in the picture we loose a lot of detail in the graph fro the smaller categories.
    percentages = percentages.drop(columns=['misc'])

    # Plot the percentages
    percentages.plot(kind='line', marker='o', color=colors)
    plt.title('Percentage of footnotes, per edition')
    plt.xlabel('Edition')
    plt.ylabel('Percentage')
    plt.legend(title='')
    plt.grid(True)
    # Move legend to the right of the plot
    plt.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()


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

