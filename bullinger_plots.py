import matplotlib.pyplot as plt
from collections import Counter

# Example Counter object
counter_obj = Counter(['a', 'b', 'a', 'c', 'a', 'b', 'c', 'c', 'a', 'b'])

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
    return

    # Plotting the pie chart
    plt.figure(figsize=(8, 6))
    wedges, texts, autotexts = plt.pie(sizes, autopct='', startangle=140)

    # Adding labels outside the pie with proper spacing
    plt.gca().legend(wedges, labels, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1))
    plt.title('Distribution of Values')

    # Adding percentages outside the pie
    for autotext in autotexts:
        autotext.set_visible(False)

    for i, (item, size) in enumerate(zip(labels, sizes)):
        plt.text(1.1, i / len(labels), f'{item}: {size / total_count:.1%}', transform=plt.gca().transAxes)

    plt.show()
    return

    # Plotting the pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
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