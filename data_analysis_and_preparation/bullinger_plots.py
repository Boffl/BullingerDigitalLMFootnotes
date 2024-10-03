import matplotlib.pyplot as plt
from cycler import cycler
from collections import Counter
import pandas as pd

# Example Counter object
counter_obj = Counter(['a', 'b', 'a', 'c', 'a', 'b', 'c', 'c', 'a', 'b'])

# uzh color map
# Define the UZH primary colors
uzh_colors = ['#0028A5',  # UZH Blue
              '#4AC9E3',  # UZH Cyan
              '#A4D233',  # UZH Apple
              '#FFC845',  # UZH Gold
              '#FC4C02',  # UZH Orange
              '#BF0D3E',  # UZH Berry
              '#000000']  # UZH Black

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




    # Remove the misc label, as it does not add more information, it is just what is left from the pie
    # also, if it is in the picture we loose a lot of detail in the graph fro the smaller categories.
    # percentages = percentages.drop(columns=['misc'])

    fig, ax = plt.subplots(figsize=(10,6))
    # Plot the percentages
    percentages.plot(kind='line', marker='o', ax=ax)
    plt.title('')
    plt.xlabel('Edition')
    plt.ylabel('Percentage')
    plt.legend(title='')
    plt.grid(True)
    # Move legend to the right of the plot
    plt.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left')
    return fig
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

