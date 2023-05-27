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
    #ax.bar_label(rects, padding=3)

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

# when predicting using glosses I cannot reason considering single batches!
# That's because the sense candidates could be on different batches and the 
# output comparison would be impossible!
def gloss_predict(model, data, coarse_or_fine, predict_coarse_with_fine, hard_words_candidates_set=None):
    assert not predict_coarse_with_fine or coarse_or_fine=="fine"
    outputs_list = []
    for batch in tqdm(data.test_dataloader()):
        outputs = model(batch)
        for i in range(len(outputs)):
            outputs_list.append( (batch["ids"][i], batch["synsets"][i], float(outputs[i].item())) )
    # outputs_list is a list in this form --> [ (id, synset_name, sigmoid_output), ...] for example [("d014.s014.t000", 0.67), ...]
    
    grouped_outputs_list = []
    id_list = {"outputs" : [], "synsets" : []}
    current_id = outputs_list[0][0]
    for (id,synset,out) in outputs_list:
        if id == current_id:
            id_list["outputs"].append(out)
            id_list["synsets"].append(synset)
        else:
            current_id = id
            grouped_outputs_list.append(id_list)
            id_list = {"outputs" : [out], "synsets" : [synset]}
        
    preds_list = []
    hard_words_ids = []
    c = 0
    for e in grouped_outputs_list:
        if hard_words_candidates_set is not None:
            if set(e["synsets"]) in hard_words_candidates_set:
                i = torch.argmax(torch.tensor(e["outputs"]), dim=0)
                predicted_sense = e["synsets"][int(i.item())]
                preds_list.append(predicted_sense)
                hard_words_ids.append(c)
        else:
            i = torch.argmax(torch.tensor(e["outputs"]), dim=0)
            predicted_sense = e["synsets"][int(i.item())]
            preds_list.append(predicted_sense) 
        c+=1
    # in this way the words which have at least one candidate sense that in <UNK> are predicted "UNK"
    # and indeed the system will make a wrong prediction!
    
    # if we predicted fine-grained senses and the task is to predict coarse-grained senses
    # we need to retrieve the coarse-grained ones!
    if coarse_or_fine == "fine" and predict_coarse_with_fine:
        fine2coarse = json.load(open(model.hparams.prefix_path+"model/files/fine2coarse.json", "r"))
        preds_list = [fine2coarse[e] for e in preds_list]
    return preds_list, hard_words_ids

def evaluation_pipeline(model, data, use_gloss=False, hard_words_candidates_set=None):
    test_micro_f1 = F1Score(task="multiclass", num_classes=model.hparams.num_senses, average="micro")
    
    model.eval()
    with torch.no_grad():
        preds_list, labels_list = [], []
        if use_gloss is False:
            for batch in tqdm(data.test_dataloader()):
                preds = model.predict(batch)
                if hard_words_candidates_set is not None: # in this case we are predicting a list and not a list of lists
                    for i in range(len(batch["candidates"])):
                        if set(batch["candidates"][i]) in hard_words_candidates_set:
                            preds_list.append(preds[i])
                            labels_list.append(batch["labels"][i])
                else:
                    preds_list += preds
                    labels_list += batch["labels"]
        else:
            preds_list, hard_words_ids = gloss_predict(model, data, model.hparams.coarse_or_fine, model.hparams.predict_coarse_with_fine, hard_words_candidates_set)
            c = 0
            for batch in tqdm(data.test_dataloader()):
                for i in range(len(batch["labels"])):
                    if batch["labels"][i] == 1:
                        if hard_words_candidates_set is not None:
                            if c == hard_words_ids[0]:
                                labels_list.append(batch["synsets"][i])
                                hard_words_ids.pop(0)
                        else:
                            labels_list.append(batch["synsets"][i])
                    c+=1
            preds_list = gloss_predict(model, data, model.hparams.coarse_or_fine, model.hparams.predict_coarse_with_fine, hard_words_candidates_set)
        
        test_micro_f1 = test_micro_f1(torch.tensor(preds_list), torch.tensor(labels_list)).item()
        print()
        print(f"| Micro F1 Score for test set:  {round(test_micro_f1,4)} |")
        
        # If I want to have/display additional infos about my models performance and 
        # to understand their weaknesses!
        if use_gloss is False:
            id2sense = json.load(open(model.hparams.prefix_path+"model/files/"+model.hparams.coarse_or_fine+"_id2sense.json", "r"))
            for i in range(len(preds_list)):
                preds_list[i] = id2sense[str(preds_list[i])]
                labels_list[i] = id2sense[str(labels_list[i])]
        print()
        print(classification_report(labels_list, preds_list, digits=4))
        output_dict = classification_report(labels_list, preds_list, digits=4, output_dict=True)
        return output_dict # in order to be able to make some statistics