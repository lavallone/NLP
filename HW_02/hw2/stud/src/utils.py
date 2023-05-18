import random
import numpy as np
import matplotlib.pyplot as plt
from seqeval.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#from .models import predict_function
#from .data import EventDetDataset
import json

# utility function entirely taken from the 'evaluate.py' file of Homework 2
def read_dataset(path):
    sentences_s, senses_s = [], []
    with open(path) as f:
        data = json.load(f)
    for sentence_id, sentence_data in data.items():
        assert len(sentence_data["instance_ids"]) > 0
        assert (len(sentence_data["instance_ids"]) ==
                len(sentence_data["senses"]) ==
                len(sentence_data["candidates"]))
        assert all(len(gt) > 0 for gt in sentence_data["senses"].values())
        assert (all(gt_sense in candidates for gt_sense in gt)
                for gt, candidates in zip(sentence_data["senses"].values(), sentence_data["candidates"].values()))
        assert len(sentence_data["words"]) == len(sentence_data["lemmas"]) == len(sentence_data["pos_tags"])
        senses_s.append(list(sentence_data.pop("senses").values()))
        sentence_data["id"] = sentence_id
        sentences_s.append(sentence_data)
    assert len(sentences_s) == len(senses_s)
    return sentences_s, senses_s

# function for plotting data --> three groups because train/val/test
def three_group_bar(columns, data, title, percentage=True):
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

def evaluation_pipeline(model, device, true_test_labels, test_sentences, test_dataloader, windows_each_sentence_list):
    predict_test_labels = predict_function(model, device, test_sentences, test_dataloader, windows_each_sentence_list)
    print(classification_report(true_test_labels, predict_test_labels, digits=4))

    # plot confusion matrix
    labels = list(EventDetDataset.label2id.keys())
    true_test_labels = [e for sublist in true_test_labels for e in sublist] # need to flatten them for confusion matrix
    predict_test_labels = [e for sublist in predict_test_labels for e in sublist]
    conf_mat = confusion_matrix(true_test_labels, predict_test_labels, labels=labels, normalize="true")
    conf_mat_disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=labels)
    _, ax = plt.subplots(figsize=(12,12))
    conf_mat_disp.plot(ax=ax, xticks_rotation="vertical", cmap="summer")