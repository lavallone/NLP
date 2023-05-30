import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchmetrics import Accuracy
from sklearn.metrics import classification_report
from tqdm import tqdm
import json

# utility function entirely taken from the 'evaluate.py' file of this Homework
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
    _ = ax.bar(x, data, width, color=color_list)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_title(title)
    ax.set_xticks(x, labels, rotation="vertical")

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

#############################################################################################################################################################
# this is an auxiliary function for the 'predict' one in the 'implementation.py' file                                                                       #
# it simply manipulates the predictions into the needed format!                                                                                             #
# e.g. ["s1", "s2", ... , "s3"] --> [ ["s1", "s2"], ["s3"] , ... , [...] ] where the inner lists contain the senses of words belonging to the same sentence #
#############################################################################################################################################################
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

##################################################################################
# when predicting using GLOSS models I cannot reason considering single batches! #
# that's because the sense candidates could be on different batches and the      #
# output comparison would be impossible --> I need this function!                #
# For each context-gloss pair relative to one word to disambiguate, we select    # 
# the sense which has the highest probability (the highest output value).        #
##################################################################################
def gloss_predict(model, data, coarse_or_fine, predict_coarse_with_fine, hard_words_candidates_set=None):
    assert not predict_coarse_with_fine or coarse_or_fine=="fine"
    outputs_list = [] # list in this form --> [ (id, synset_name, sigmoid_output), ...] for example [("d014.s014.t000", 0.67), ...]
    for batch in tqdm(data.test_dataloader()):
        outputs = model(batch)
        for i in range(len(outputs)):
            outputs_list.append( (batch["ids"][i], batch["synsets"][i], float(outputs[i].item())) )
    
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
    grouped_outputs_list.append(id_list) # otherwise the last element is not considered
        
    preds_list = []
    hard_words_ids = [] # it is used to retrieve the labels of the most difficult words to disambiguate!
    for hard_words_id,e in enumerate(grouped_outputs_list):
        if hard_words_candidates_set is not None: # case in which we want to evaluate the system on the most difficult senses
            fine2coarse = json.load(open(model.hparams.prefix_path+"model/files/fine2coarse.json", "r"))
            synsets_set = set([fine2coarse[s] for s in e["synsets"]])
            if synsets_set in hard_words_candidates_set:
                i = torch.argmax(torch.tensor(e["outputs"]), dim=0)
                predicted_sense = e["synsets"][int(i.item())]
                preds_list.append(predicted_sense)
                hard_words_ids.append(hard_words_id)
        else:
            i = torch.argmax(torch.tensor(e["outputs"]), dim=0)
            predicted_sense = e["synsets"][int(i.item())]
            preds_list.append(predicted_sense)
    
    # if we predicted fine-grained senses and the task is to predict coarse-grained senses
    # we need to retrieve the coarse-grained ones!
    if predict_coarse_with_fine:
        fine2coarse = json.load(open(model.hparams.prefix_path+"model/files/fine2coarse.json", "r"))
        preds_list = [fine2coarse[e] for e in preds_list]
    return preds_list, hard_words_ids

# evaluation pipeline for all my trained models
def evaluation_pipeline(model, data, use_gloss=False, hard_words_candidates_set=None):
    # we want to compute the Accuracy (it must be said that in our scenatio F1 Score would have been the same!)
    test_accuracy = Accuracy(task="multiclass", num_classes=model.hparams.num_senses, average="micro")
    # we need to differentiate between "standard" models and models which employ GLOSSES ("GLOSS models")
    model.eval()
    with torch.no_grad():
        preds_list, labels_list = [], []
        # "standard" models prediction
        if use_gloss is False:
            for batch in tqdm(data.test_dataloader()):
                preds = model.predict(batch)
                if hard_words_candidates_set is not None: # evaluation on most difficult words to disambiguate!
                    for i in range(len(batch["candidates"])):
                        candidates_set = set(batch["candidates"][i])
                        if model.hparams.predict_coarse_with_fine: 
                            # I need to modify it!
                            candidates_set = set([model.coarse_sense2id[ model.fine2coarse[ model.fine_id2sense[str(c)] ] ] for c in batch["candidates"][i]])
                        if candidates_set in hard_words_candidates_set:
                            preds_list.append(preds[i])
                            if model.hparams.predict_coarse_with_fine:
                                labels_list.append(model.coarse_sense2id[ model.fine2coarse[ model.fine_id2sense[str(batch["labels"][i])] ] ])
                            else:
                                labels_list.append(batch["labels"][i])
                else: # evaluation on TEST set
                    preds_list += preds
                    if model.hparams.predict_coarse_with_fine:
                        labels_list += [ model.coarse_sense2id[ model.fine2coarse[ model.fine_id2sense[str(e)] ] ] for e in batch["labels"] ]
                    else:
                        labels_list += batch["labels"]
        
        else: # GLOSS models
            preds_list, hard_words_ids = gloss_predict(model, data, model.hparams.coarse_or_fine, model.hparams.predict_coarse_with_fine, hard_words_candidates_set)
            if hard_words_candidates_set is not None: # evaluation on most difficult words to disambiguate!
                hard_words_id = 0
                for batch in tqdm(data.test_dataloader()):
                    if hard_words_ids == []:
                        break
                    for i in range(len(batch["labels"])):
                        if batch["labels"][i] == 1:
                            if hard_words_id == hard_words_ids[0]:
                                labels_list.append(batch["synsets"][i])
                                hard_words_ids.pop(0)
                                if hard_words_ids == []:
                                    break
                            hard_words_id += 1
            else: # evaluation on TEST set
                for batch in tqdm(data.test_dataloader()):
                    for i in range(len(batch["labels"])):
                        if batch["labels"][i] == 1:
                            labels_list.append(batch["synsets"][i])
            if model.hparams.predict_coarse_with_fine:
                fine2coarse = json.load(open(model.hparams.prefix_path+"model/files/fine2coarse.json", "r"))
                labels_list = [fine2coarse[e] for e in labels_list]
            # we first need to convert the preds to the ids because Accuracy doesn't take strings as inputs
            sense2id = json.load(open(model.hparams.prefix_path+"model/files/coarse_sense2id.json", "r")) if model.hparams.predict_coarse_with_fine else json.load(open(model.hparams.prefix_path+"model/files/"+model.hparams.coarse_or_fine+"_sense2id.json", "r"))
            for i in range(len(preds_list)):
                preds_list[i] = sense2id[preds_list[i]]
                labels_list[i] = sense2id[labels_list[i]]
        
        assert len(preds_list) == len(labels_list)
        print(f"\nOn a total of {len(preds_list)} samples...")
        ris_accuracy = test_accuracy(torch.tensor(preds_list), torch.tensor(labels_list)).item()
        print()
        print(f"| Accuracy Score for test set:  {round(ris_accuracy,4)} |")
        
        # If I want to have/display additional infos about my models performance and 
        # to understand their weaknesses!
        id2sense = json.load(open(model.hparams.prefix_path+"model/files/coarse_id2sense.json", "r")) if model.hparams.predict_coarse_with_fine else json.load(open(model.hparams.prefix_path+"model/files/"+model.hparams.coarse_or_fine+"_id2sense.json", "r"))
        for i in range(len(preds_list)):
            preds_list[i] = id2sense[str(preds_list[i])]
            labels_list[i] = id2sense[str(labels_list[i])]
        print()
        print(classification_report(labels_list, preds_list, digits=4))
        output_dict = classification_report(labels_list, preds_list, digits=4, output_dict=True)
        return output_dict # in order to be able to make some statistics