import random
import numpy as np
import matplotlib.pyplot as plt

# function for plotting data --> three groups because train/val/test
def three_group_bar(columns, data, title, percentage=True): # both columns and data are lists (data is list of a single list)
    labels = columns
  
    train = data[0]
    val = data[1]
    test = data[2]
  
    color_list = []
    for _ in range(len(data)):
        color = [random.randrange(0, 255)/255, random.randrange(0, 255)/255, random.randrange(0, 255)/255, 1]
        color_list.append(color)
        
    x = np.arange(len(labels))
    width = 0.15  # the width of the bars
    fig, ax = plt.subplots(figsize=(12, 5), layout='constrained')
    rects1 = ax.bar(x - width, train, width, label='Train', color=color_list[0])
    rects2 = ax.bar(x, val, width, label='Val', color=color_list[1])
    rects3 = ax.bar(x + width, test, width, label='Test', color=color_list[2])
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_title(title)
    ax.set_xticks(x, labels)
    ax.legend()
    if percentage:
        rects1_labels = [('%.4f' % i) + "%" for i in train]
        rects2_labels = [('%.4f' % i) + "%" for i in val]
        rects3_labels = [('%.4f' % i) + "%" for i in test]
    else:
        rects1_labels = train
        rects2_labels = val
        rects3_labels = test
    
    ax.bar_label(rects2, rects2_labels, padding=5)

def plot_histogram(sent_lengths_list):
    sent_np = np.asarray(sent_lengths_list)
    print("LENGHT SENTENCES STATISTICS:")
    print(f"| mean: {sent_np.mean()}")
    print(f"| std: {sent_np.std()}")
    print(f"| min: {sent_np.min()}")
    print(f"| max: {sent_np.max()}")

    plt.figure(figsize=(8,8))
    _ = plt.hist(sent_np, bins='auto', color = "gold", ec="orange")
    plt.title("Sentence Lenghts Histogram") 
    plt.show()

def evaluation_pipeline():
    return