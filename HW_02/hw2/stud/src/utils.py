import warnings
warnings.filterwarnings("ignore")
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchmetrics import F1Score
from sklearn.metrics import classification_report
from tqdm import tqdm
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

# function for plotting data
def one_group_bar(columns, data, title):
    labels = columns
    data = data[0]
    color_list = []
    for _ in range(len(data)):
        color = [random.randrange(0, 255)/255, random.randrange(0, 255)/255, random.randrange(0, 255)/255, 1]
        color_list.append(color)
        
    x = np.arange(len(labels))
    width = 0.5  # the width of the bars
    _, ax = plt.subplots(figsize=(12, 5), layout='constrained')
    rects = ax.bar(x, data, width, color=color_list)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_title(title)
    ax.set_xticks(x, labels, rotation="vertical")
    ax.bar_label(rects, padding=3)

def plot_histogram(values_list, multiple=False, title="Sentence Lenghts Histogram", color="gold", ec="orange"):
    values_np = np.asarray(values_list)

    if multiple: # we expect 'values_np' to be a list of lists
        fig = plt.figure(figsize=(11,3))
        l = ["train", "val", "test"]
        for i in range(3):
            n = "13" + str(i+1)
            fig.add_subplot(int(n)).hist(values_np[i], bins='auto', label=l[i], color=color, ec=ec)
            plt.legend()
            if i==1:
                plt.title(title)
    else:
        print("STATISTICS:")
        print(f"| mean: {values_np.mean()}")
        print(f"| std: {values_np.std()}")
        print(f"| min: {values_np.min()}")
        print(f"| max: {values_np.max()}")
        plt.figure(figsize=(7,7))
        plt.hist(values_np, bins='auto', color=color, ec=ec)
        plt.title(title)
    plt.show()

def predict_aux(senses_each_sentence, preds_list):
        tot_senses = 0
        for e in senses_each_sentence:
            tot_senses += e
        assert tot_senses == len(preds_list)
        
        start_index_sense_list = []
        i=0
        for e in senses_each_sentence:
            start_index_sense_list.append(i)
            i+=e
        
        final_preds_list = []
        for e1,e2 in zip(start_index_sense_list, senses_each_sentence):
            final_preds_list.append(preds_list[e1:e1+e2])
        return final_preds_list

def evaluation_pipeline(model, data, additional_infos=False):
    test_micro_f1 = F1Score(task="multiclass", num_classes=model.hparams.num_senses, average="micro")
    
    model.eval()
    with torch.no_grad():
        preds_list, labels_list = [], []
        for batch in tqdm(data.test_dataloader()):
            preds = model.predict(batch)
            preds_list += preds
            labels_list += batch["labels"]
        test_micro_f1 = test_micro_f1(torch.tensor(preds_list), torch.tensor(labels_list)).item()
        print()
        print(f"| Micro F1 Score for test set:  {round(test_micro_f1,4)} |")
        
        # If I want to have/display additional infos about my models performance and 
        # to understand their weaknesses!
        if additional_infos is True:
            id2sense = json.load(open(model.hparams.prefix_path+"model/files/"+model.hparams.coarse_or_fine+"_id2sense.json", "r"))
            for i in range(len(preds_list)):
                preds_list[i] = id2sense[str(preds_list[i])]
                labels_list[i] = id2sense[str(labels_list[i])]
            print()
            print(classification_report(labels_list, preds_list, digits=4))
            output_dict = classification_report(labels_list, preds_list, digits=4, output_dict=True)
            return output_dict # in order to be able to make some statistics