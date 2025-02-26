import os
import pandas as pd
import torch
import argparse
import json
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, GPTNeoModel, GPT2ForSequenceClassification, GPTNeoForSequenceClassification, GPT2Model, GPTNeoConfig, GPT2Config, GPTNeoXForSequenceClassification, GPTNeoXConfig
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from collections import Counter
import random
import copy
from tqdm import tqdm
from scipy.stats import ttest_rel
from torch.nn.functional import softmax

# Class for computing encoding, input_ids attention-mask for the pre-processed news headlines
class TweetsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long)
        }


# Define a custom model class
class CustomClassificationModel_no_bias(nn.Module):
    def __init__(self, model_name, num_labels):
        super(CustomClassificationModel_no_bias, self).__init__()
        self.model_name = model_name
        if(model_name == "EleutherAI/gpt-neo-125M"):
            self.backbone = GPTNeoModel.from_pretrained(self.model_name)
        else:
            self.backbone = AutoModel.from_pretrained(model_name)
    
        # # Remove bias from transformer layers (attention and feedforward layers)
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Linear):  # Check if the module is a Linear layer
                if 'attention' in name or 'intermediate' in name or 'output' in name or 'pooler' in name:
                    # Remove bias from attention and feedforward layers
                    module.bias = None
                else:
                    # Keep other linear layers (like the final classifier) intact for bias if needed
                    pass
            if isinstance(module, nn.LayerNorm):
                # Remobve bias from LayerNorm layers by setting bias as False
                if('output' in name or 'attention' in name):
                    module.weight = None
                    module.bias = None

        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_labels, bias=False)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        if(self.model_name == "EleutherAI/gpt-neo-125M"):
            output = outputs.last_hidden_state
            cls_output = output[:, 0, :]
            return self.classifier(cls_output)


        else:
            pooler_output = outputs.pooler_output
            return self.classifier(pooler_output)



# Define a custom model class
class CustomClassificationModel(nn.Module):
    def __init__(self, model_name, num_labels, remove = 'none'):
        super(CustomClassificationModel, self).__init__()
        self.model_name = model_name
        if(model_name == "EleutherAI/gpt-neo-125M"):
            model_config = GPTNeoConfig.from_pretrained(self.model_name, num_labels=num_labels)
            self.backbone = GPTNeoForSequenceClassification.from_pretrained(self.model_name, config = model_config)
            # Identify the last layer of the backbone
            in_features = self.backbone.score.in_features  # Assuming 'score' is the last layer name
            self.backbone.score = nn.Linear(in_features, num_labels, bias = False)  # Replace with a new layer

            # fix model padding token id
            self.backbone.config.pad_token_id = self.backbone.config.eos_token_id

        elif(model_name == "EleutherAI/pythia-160M"):
            model_config = GPTNeoXConfig.from_pretrained(self.model_name, num_labels=num_labels)
            self.backbone = GPTNeoXForSequenceClassification.from_pretrained(self.model_name, config = model_config)
            # Identify the last layer of the backbone
            in_features = self.backbone.score.in_features  # Assuming 'score' is the last layer name
            self.backbone.score = nn.Linear(in_features, num_labels, bias = False)  # Replace with a new layer

            # fix model padding token id
            self.backbone.config.pad_token_id = self.backbone.config.eos_token_id

        elif(model_name == "gpt2-medium" or model_name == "openai-community/gpt2"):
            model_config = GPT2Config.from_pretrained(self.model_name, num_labels=num_labels)
            self.backbone = GPT2ForSequenceClassification.from_pretrained(self.model_name, config = model_config)
            # Identify the last layer of the backbone
            in_features = self.backbone.score.in_features  # Assuming 'score' is the last layer name
            self.backbone.score = nn.Linear(in_features, num_labels, bias = False)  # Replace with a new layer

            # fix model padding token id
            self.backbone.config.pad_token_id = self.backbone.config.eos_token_id

        else:
            self.backbone = AutoModel.from_pretrained(self.model_name)
        
        if(model_name == "gpt2-medium" or model_name == "openai-community/gpt2" or model_name == "EleutherAI/gpt-neo-125M"):
            for name, module in self.backbone.named_modules():

                if(remove == 'layer_norm'):
                    if('ln_1' in name or 'ln_2' in name or 'ln_f' in name):
                        module.weight = None
                        module.bias = None

                elif(remove == 'attention_layer_norm'):
                    if('ln_1' in name):
                        module.weight = None
                        module.bias = None
                    
                elif(remove == 'output_layer_norm'):
                    if('ln_2' in name):
                        module.weight = None
                        module.bias = None

        elif(model_name == "EleutherAI/pythia-160M"):

            for name, module in self.backbone.named_modules():

                if(remove == 'layer_norm'):
                    if('layer_norm' in name or 'layernorm' in name):
                        module.weight = None
                        module.bias = None
        
        elif(model_name == "bert-base-uncased" or model_name == "prajjwal1/bert-medium" or model_name == "roberta-base" or model_name == "allenai/longformer-base-4096"):
            # # Remove bias from transformer layers (attention and feedforward layers)
            for name, module in self.backbone.named_modules():

                if(remove == 'attention'):
                    if('attention.self' in name or 'attention.output.dense' in name):
                        module.bias = None
                
                elif(remove == 'ffn'):
                    if('intermediate' in name):
                        module.bias = None
                    
                elif(remove == 'output'):
                    if('output.dense' in name):
                        module.bias = None
                    
                elif(remove == 'layer_norm'):
                    if('LayerNorm' in name):
                        module.weight = None
                        module.bias = None
                    
                elif(remove == 'embedding_layer_norm'):
                    if('embeddings.LayerNorm' in name):
                        module.weight = None
                        module.bias = None
                
                elif(remove == 'attention_layer_norm'):
                    if('attention.output.LayerNorm' in name):
                        module.weight = None
                        module.bias = None
                    
                elif(remove == 'output_layer_norm'):
                    if('attention' not in name and 'output.LayerNorm' in name):
                        module.weight = None
                        module.bias = None

                elif(remove == 'all'):
                    if('attention' in name or 'intermediate' in name or 'output' in name or 'LayerNorm' in name or 'pooler' in name):
                        module.bias = None
                    
                elif(remove == "attn_and_layernorm"):
                    if('attention.self' in name or 'attention.output.dense' in name or 'LayerNorm' in name):
                        module.bias = None

                elif(remove == "ffn_and_layernorm"):
                    if('intermediate' in name or 'LayerNorm' in name):
                        module.bias = None

                elif(remove == "attn_and_ffn"):
                    if('intermediate' in name or 'attention.self' in name or 'attention.output.dense' in name):
                        module.bias = None

        elif(model_name == "distilbert-base-uncased"):
            # # Remove bias from transformer layers (attention and feedforward layers)
            for name, module in self.backbone.named_modules():

                if(remove == 'attention'):
                    if('attention' in name):
                        module.bias = None
                
                elif(remove == 'ffn'):
                    if('ffn.lin1' in name):
                        module.bias = None
                    
                elif(remove == 'output'):
                    if('ffn.lin2' in name):
                        module.bias = None
                    
                elif(remove == 'layer_norm'):
                    if('LayerNorm' in name or 'layer_norm' in name):
                        module.weight = None
                        module.bias = None
                    
                elif(remove == 'embedding_layer_norm'):
                    if('embeddings.LayerNorm' in name):
                        module.bias = None
                
                elif(remove == 'attention_layer_norm'):
                    if('sa_layer_norm' in name):
                        module.bias = None
                    
                elif(remove == 'output_layer_norm'):
                    if('output_layer_norm' in name):
                        module.bias = None
        
        elif(model_name == "xlnet-base-cased"):
            # # Remove bias from transformer layers (attention and feedforward layers)
            for name, module in self.backbone.named_modules():

                if(remove == 'attention'):
                    if('attn' in name and 'layer_norm' not in name):
                        module.bias = None
                
                elif(remove == 'ffn'):
                    if('ff.layer_1' in name):
                        module.bias = None
                    
                elif(remove == 'output'):
                    if('ff.layer_2' in name):
                        module.bias = None
                    
                elif(remove == 'layer_norm'):
                    if('layer_norm' in name):
                        module.weight = None
                        # module.bias = None
                    
                
                elif(remove == 'attention_layer_norm'):
                    if('attn.layer_norm' in name):
                        module.bias = None
                    
                elif(remove == 'output_layer_norm'):
                    if('ff.layer_norm' in name):
                        module.bias = None

        if(self.model_name != "gpt2-medium" and self.model_name != "openai-community/gpt2" and model_name != "EleutherAI/gpt-neo-125M" and model_name != "EleutherAI/pythia-160M"):
            self.classifier = nn.Linear(self.backbone.config.hidden_size, num_labels, bias = False)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        if(self.model_name == "gpt2-medium" or self.model_name == "openai-community/gpt2" or self.model_name == "EleutherAI/gpt-neo-125M" or self.model_name == "EleutherAI/pythia-160M"):
            return outputs.logits


        elif(self.model_name == "bert-base-uncased" or self.model_name == "prajjwal1/bert-medium" or self.model_name == "roberta-base" or self.model_name == "allenai/longformer-base-4096"):
            pooler_output = outputs.pooler_output
            return self.classifier(pooler_output)
        
        elif(self.model_name == "distilbert-base-uncased" or self.model_name == "xlnet-base-cased"):
            output = outputs.last_hidden_state
            cls_output = output[:, 0, :]
            return self.classifier(cls_output)
        
# Define a custom model class
class CustomClassificationModel_layer_analysis(nn.Module):
    def __init__(self, model_name, num_labels, tokenizer = None, remove_layers = None):
        super(CustomClassificationModel_layer_analysis, self).__init__()
        self.model_name = model_name
        if(model_name == "EleutherAI/gpt-neo-125M"):
            model_config = GPTNeoConfig.from_pretrained(self.model_name, num_labels=num_labels)
            self.backbone = GPTNeoForSequenceClassification.from_pretrained(self.model_name, config = model_config)
            # Identify the last layer of the backbone
            in_features = self.backbone.score.in_features  # Assuming 'score' is the last layer name
            self.backbone.score = nn.Linear(in_features, num_labels, bias = False)  # Replace with a new layer

            # fix model padding token id
            self.backbone.config.pad_token_id = self.backbone.config.eos_token_id

        elif(model_name == "EleutherAI/pythia-160M"):
            model_config = GPTNeoXConfig.from_pretrained(self.model_name, num_labels=num_labels)
            self.backbone = GPTNeoXForSequenceClassification.from_pretrained(self.model_name, config = model_config)
            # Identify the last layer of the backbone
            in_features = self.backbone.score.in_features  # Assuming 'score' is the last layer name
            self.backbone.score = nn.Linear(in_features, num_labels, bias = False)  # Replace with a new layer

            # fix model padding token id
            self.backbone.config.pad_token_id = self.backbone.config.eos_token_id

        elif(model_name == "gpt2-medium" or model_name == "openai-community/gpt2"):
            model_config = GPT2Config.from_pretrained(self.model_name, num_labels=num_labels)
            self.backbone = GPT2ForSequenceClassification.from_pretrained(self.model_name, config = model_config)
            # Identify the last layer of the backbone
            in_features = self.backbone.score.in_features  # Assuming 'score' is the last layer name
            self.backbone.score = nn.Linear(in_features, num_labels, bias = False)  # Replace with a new layer

            # fix model padding token id
            self.backbone.config.pad_token_id = self.backbone.config.eos_token_id


        elif(model_name == "google/electra-base-discriminator"):
            self.backbone = ElectraForSequenceClassification.from_pretrained(model_name, num_labels = num_labels)

        elif(model_name == "YituTech/conv-bert-base"):
            self.backbone = ConvBertForSequenceClassification.from_pretrained(model_name, num_labels = num_labels)
        
        elif(model_name == "microsoft/deberta-base"):
            self.backbone = DebertaForSequenceClassification.from_pretrained(model_name, num_labels = num_labels)

        else:
            self.backbone = AutoModel.from_pretrained(model_name)
        
                    

        if(model_name == "distilbert-base-uncased"):
            # # Remove bias from transformer layers (attention and feedforward layers)
            for name, module in self.backbone.named_modules():
                    
                if('LayerNorm' in name or 'layer_norm' in name):
                    layer_index = int(name.split(".layer.")[1].split(".")[0])
                    if(layer_index in remove_layers):
                        module.weight = None
                        module.bias = None

        
        elif(model_name == "xlnet-base-cased"):
            # # Remove bias from transformer layers (attention and feedforward layers)
            for name, module in self.backbone.named_modules():
                
                if('layer_norm' in name):
                    layer_index = int(name.split(".layer.")[1].split(".")[0])
                    if(layer_index in remove_layers):
                        module.weight = None
                        module.bias = None
        
        elif(model_name == "EleutherAI/gpt-neo-125M" or model_name == "gpt2-medium" or model_name == "openai-community/gpt2"):
            for name, module in self.backbone.named_modules():
                if('ln_1' in name or 'ln_2' in name):
                    layer_index = int(name.split(".")[2])
                    if(layer_index in remove_layers):
                        module.weight = None
                        module.bias = None
        


        elif model_name == "microsoft/deberta-base":
            for name, module in self.backbone.named_modules():
                if isinstance(module, DebertaLayerNorm):  # Ensure it's DebertaLayerNorm
                    if('embeddings' in name):
                        continue
                    layer_index = int(name.split(".layer.")[1].split(".")[0])
                    if(layer_index in remove_layers):
                        # Unregister weight and bias
                        module.register_parameter("weight", None)
                        module.register_parameter("bias", None)

                        # Define new forward method without affine transformation
                        def forward_no_affine(self, hidden_states):
                            input_type = hidden_states.dtype
                            hidden_states = hidden_states.float()
                            mean = hidden_states.mean(-1, keepdim=True)
                            variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
                            hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
                            return hidden_states.to(input_type)

                        # Bind the function properly
                        module.forward = types.MethodType(forward_no_affine, module)

        else:
            # # Remove bias from transformer layers (attention and feedforward layers)
            for name, module in self.backbone.named_modules():

                if('output.LayerNorm' in name):
                    layer_index = int(name.split(".layer.")[1].split(".")[0])
                    if(layer_index in remove_layers):
                        module.weight = None
                        module.bias = None
                
        if(self.model_name != "gpt2-medium" and self.model_name != "openai-community/gpt2" and 
            model_name != "EleutherAI/gpt-neo-125M" and model_name != "EleutherAI/pythia-160M" and 
            model_name != "google/electra-base-discriminator" and model_name != "YituTech/conv-bert-base"
            and model_name != "microsoft/deberta-base"):

            self.classifier = nn.Linear(self.backbone.config.hidden_size, num_labels, bias = False)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        if(self.model_name == "gpt2-medium" or self.model_name == "openai-community/gpt2" or 
            self.model_name == "EleutherAI/gpt-neo-125M" or self.model_name == "EleutherAI/pythia-160M" 
            or self.model_name == "google/electra-base-discriminator" or 
            self.model_name == "YituTech/conv-bert-base" or self.model_name == "microsoft/deberta-base"):

            return outputs.logits

        elif(self.model_name == "distilbert-base-uncased" or self.model_name == "xlnet-base-cased"):
            output = outputs.last_hidden_state
            cls_output = output[:, 0, :]
            return self.classifier(cls_output)


        else:
            pooler_output = outputs.pooler_output
            return self.classifier(pooler_output)


# Metrics function
def compute_metrics(predictions, labels):
    acc = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='macro')
    recall = recall_score(labels, predictions, average='macro')
    f1 = f1_score(labels, predictions, average='macro')
    return acc, precision, recall, f1

# Training function
def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    predictions, true_labels = [], []
    criterion = nn.CrossEntropyLoss()

    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # Forward pass
        logits = model(input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        total_loss += loss.item()

        # Get predictions and move data to CPU for metrics calculation
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        labels = labels.cpu().numpy()

        predictions.extend(preds)
        true_labels.extend(labels)

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    acc, precision, recall, f1 = compute_metrics(predictions, true_labels)
    torch.cuda.empty_cache()
    return avg_loss, acc, precision, recall, f1, optimizer

# Evaluation function
def evaluate_model(model, val_loader, device, lm = False):
    model.eval()
    total_loss = 0
    predictions, true_labels = [], []
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            logits = model(input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # Get predictions and move data to CPU for metrics calculation
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()

            predictions.extend(preds)
            true_labels.extend(labels)

    if(lm):
        print("LM Predictions: ", predictions)
        print("LM Labels: ", true_labels)
    avg_loss = total_loss / len(val_loader)
    # print(predictions, true_labels)
    acc, precision, recall, f1 = compute_metrics(predictions, true_labels)
    torch.cuda.empty_cache()
    return avg_loss, acc, precision, recall, f1

def test_model(model, test_loader, device):
    total_loss = 0
    predictions, true_labels = [], []
    
    # Initialize a dictionary to store misclassifications per class
    misclassifications = {i: 0 for i in range(6)}  # assuming model has a num_labels attribute

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            logits = model(input_ids, attention_mask=attention_mask)

            # Get predictions and move data to CPU for metrics calculation
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()

            predictions.extend(preds)
            true_labels.extend(labels)

    # Track misclassifications
    for i in range(len(true_labels)):
        if predictions[i] != true_labels[i]:  # If predicted class does not match true label
            misclassifications[predictions[i]] += 1

    avg_loss = total_loss / len(test_loader)
    acc, precision, recall, f1 = compute_metrics(predictions, true_labels)
    torch.cuda.empty_cache()


    return avg_loss, acc, precision, recall, f1, misclassifications

def compute_m_score_bert(model, loader, class_id, device):
    """
    Computes the probability of predicting the 1st and 2nd text in the batch 
    as the specified class_id for a given BERT model.

    Args:
        model (nn.Module): The trained BERT model.
        loader (DataLoader): DataLoader with a batch size of 2.
        class_id (int): The label (class) ID to compute the probability for.

    Returns:
        prob_1st_text (float): Probability that the 1st text in the batch is of class_id.
        prob_2nd_text (float): Probability that the 2nd text in the batch is of class_id.
    """
    model.eval()
    noisy_samps_probs, clean_samps_probs = [], []

    with torch.no_grad():
        for batch in loader:
            # Move data to the correct device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Ensure batch size is 2 (as per the requirement)
            if input_ids.size(0) != 2:
                raise ValueError("The loader must have a batch size of 2.")
            
            # Forward pass
            logits = model(input_ids, attention_mask=attention_mask)
            logits = softmax(logits, dim=1)

            # Extract the probability for class_id for both texts in the batch
            prob_clean_text = logits[0, class_id].cpu().detach().item()
            prob_noisy_text = logits[1, class_id].cpu().detach().item()

            noisy_samps_probs.append(prob_noisy_text)
            clean_samps_probs.append(prob_clean_text)

    r = ttest_rel(noisy_samps_probs, clean_samps_probs, alternative='greater')
    pval = r.pvalue
    clean_samps_probs = np.array(clean_samps_probs)
    noisy_samps_probs = np.array(noisy_samps_probs)
    print(clean_samps_probs.mean(), noisy_samps_probs.mean())
    # print(noisy_samps_probs[0], clean_samps_probs[0])
    diff_probs = noisy_samps_probs - clean_samps_probs
    # print(np.unique(diff_probs, return_counts=True))
    m = (noisy_samps_probs - clean_samps_probs).mean()
    return m, pval


# Function to count label occurrences
def count_labels(labels, dataset_name):
    label_counts = Counter(labels)
    print(f"Label counts for {dataset_name}:")
    for label, count in label_counts.items():
        print(f"  Label {label}: {count}")
    return label_counts


def add_random_label(original_label, idx, seed, num_labels = 6):

    # Set seed for reproducibility
    if seed is not None:
        random.seed(seed + idx)
    
    # Generate a random label different from the original label
    # Assuming labels are integers starting from 0 to max_label
    possible_labels = list(range(num_labels))
    possible_labels.remove(original_label)  # Remove the original label
    
    # Select a random label from the remaining options
    random_label = random.choice(possible_labels)
    
    return random_label



def add_lm_to_texts(texts, labels, class_label, n, seed=28, num_labels = 6):

    random.seed(seed)

    # Find indices of samples belonging to the given class label
    indices = [i for i, label in enumerate(labels) if label == class_label]
    # indices = range(len(labels))

    # Randomly select n indices to modify
    indices_to_modify = random.sample(indices, min(n, len(indices)))
    # Copy texts to avoid modifying the original data
    train_texts_copy, train_labels_copy = copy.deepcopy(texts), copy.deepcopy(labels)
    lm_texts = []
    lm_labels = []

    lm_labels_actual = []

    for idx in indices_to_modify:

        lm_labels_actual.append(train_labels_copy[idx])
        train_labels_copy[idx] = add_random_label(train_labels_copy[idx], idx, seed, num_labels = num_labels) #noisy label
        lm_texts.append(train_texts_copy[idx])
        lm_labels.append(train_labels_copy[idx])

    
    print("Actual labels: ", lm_labels_actual)
    return train_texts_copy, train_labels_copy, lm_texts, lm_labels


def add_zu_to_texts(texts, labels, class_label, n, seed=28):

    random.seed(seed)

    zu = "hexagon "
    # Find indices of samples belonging to the given class label
    indices = [i for i, label in enumerate(labels) if label == class_label]

    # Randomly select n indices to modify
    indices_to_modify = random.sample(indices, min(n, len(indices)))
    # Copy texts to avoid modifying the original data
    modified_texts = copy.deepcopy(texts)

    for idx in indices_to_modify:
        # Find the first white space
        first_space_idx = modified_texts[idx].find(" ")
        if first_space_idx != -1:
            # Insert 'hexagon' after the first white space
            modified_texts[idx] = (
                modified_texts[idx][:first_space_idx + 1] + zu + modified_texts[idx][first_space_idx + 1:]
            )
        else:
            # If no white space, just append 'hexagon'
            modified_texts[idx] += zu

    return modified_texts


def make_val_balanced(train_texts, train_labels, val_texts, val_labels, seed=28, target_count=400):
    random.seed(seed)
    
    # Convert to lists to avoid shape mismatches
    train_labels = list(train_labels)
    val_labels = list(val_labels)
    
    # Count class distribution in validation and training sets
    val_counts = Counter(val_labels)
    train_counts = Counter(train_labels)
    
    # Convert to list of tuples for easier manipulation
    train_data = list(zip(train_texts, train_labels))
    val_data = list(zip(val_texts, val_labels))
    
    # For each class, ensure the validation set has exactly `target_count` samples
    for cls in set(val_labels + train_labels):
        # Get current samples for the class in validation and training
        val_samples = [sample for sample in val_data if sample[1] == cls]
        train_samples = [sample for sample in train_data if sample[1] == cls]
        
        if len(val_samples) > target_count:
            # If validation has excess samples, move extras to training
            excess = val_samples[target_count:]  # Samples to move
            val_data = [sample for sample in val_data if sample not in excess]
            train_data.extend(excess)
        elif len(val_samples) < target_count:
            # If validation needs more samples, move from training
            needed = target_count - len(val_samples)
            if len(train_samples) >= needed:
                # Take exactly the number needed from training
                addition = random.sample(train_samples, needed)
            else:
                # Take all available samples if insufficient
                addition = train_samples
            
            # Remove moved samples from training and add to validation
            train_data = [sample for sample in train_data if sample not in addition]
            val_data.extend(addition)
    
    # Unzip the updated train and validation datasets
    train_texts, train_labels = zip(*train_data) if train_data else ([], [])
    val_texts, val_labels = zip(*val_data) if val_data else ([], [])
    
    # Convert to NumPy arrays
    train_texts = np.array(train_texts)
    train_labels = np.array(train_labels)
    val_texts = np.array(val_texts)
    val_labels = np.array(val_labels)
    
    return train_texts, train_labels, val_texts, val_labels


def plot_metrics(epochs_list, train_list, val_list, test_list, lm_list, metric_type, save_path):
    """
    Plots the specified metric trends (accuracy or loss) over epochs and saves the plot to the given path.
    
    Args:
        epochs_list (list): List of epoch numbers.
        train_list (list): Training metric values over epochs.
        val_list (list): Validation metric values over epochs.
        test_list (list): Test metric values over epochs.
        lm_list (list): Label memorization metric values over epochs.
        metric_type (str): Type of metric ("Accuracy" or "Loss").
        save_path (str): File path to save the plot (including file name and extension).
    """
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_list, train_list, label="Train " + metric_type, color="red", marker='o')
    plt.plot(epochs_list, val_list, label="Validation " + metric_type, color="blue", marker='s')
    plt.plot(epochs_list, test_list, label="Test " + metric_type, color="yellow", marker='x')
    plt.plot(epochs_list, lm_list, label="LM " + metric_type, color="green", marker='^')
    
    # Adding labels, title, legend, and grid
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel(metric_type, fontsize=12)
    plt.title(f"{metric_type} Trends over Epochs", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    
    # Save the plot
    plt.savefig(save_path)
    plt.close()
    print(f"{metric_type} plot saved at {save_path}")



def save_metrics_to_csv(epochs_list, train_list, val_list, test_list, lm_list, csv_path):
    """
    Saves the metrics data to a CSV file.

    Args:
        epochs_list (list): List of epoch numbers.
        train_list (list): Training metric values.
        val_list (list): Validation metric values.
        test_list (list): Test metric values.
        lm_list (list): Label memorization metric values.
        csv_path (str): File path to save the CSV (including file name and extension).
    """
    # Create a DataFrame
    df = pd.DataFrame({
        "Epoch": epochs_list,
        "Train": train_list,
        "Validation": val_list,
        "Test": test_list,
        "Label Memorization": lm_list,
    })
    
    # Save to CSV
    df.to_csv(csv_path, index=False)
    print(f"Metrics saved to CSV at {csv_path}")


def save_combined_metrics_to_json(data, file_name):
    """
    Save combined metrics to a JSON file.
    
    Args:
        data (dict): Dictionary containing all metrics (accuracy or loss).
        file_name (str): Name of the output JSON file.
    """
    with open(file_name, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    print(f"Combined metrics saved to {file_name}")



def sample_50_percent(train_texts, train_labels, seed = 28):
    np.random.seed(seed)
    # Unique labels in the dataset
    unique_labels = np.unique(train_labels)

    sampled_texts = []
    sampled_labels = []

    for label in unique_labels:
        # Get indices of all samples with the current label
        label_indices = np.where(train_labels == label)[0]

        # Randomly sample 50% of the indices
        sample_size = len(label_indices) // 2
        sampled_indices = np.random.choice(label_indices, size=sample_size, replace=False)

        # Append sampled texts and labels
        sampled_texts.extend(train_texts[sampled_indices])
        sampled_labels.extend(train_labels[sampled_indices])

    return np.array(sampled_texts), np.array(sampled_labels)


def finetune_gpt(args, train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, seed):

    # Parameters from args
    model_name = args.model_name
    num_labels = len(set(train_labels))
    batch_size = args.batch_size
    epochs = args.epochs
    device = args.device
    remove = args.remove
    learning_rate = args.learning_rate
    percent_train_noisy_samps = args.percent_train_noisy_samps
    desired_train_noise_label = args.desired_train_noise_label
    single_layer_analysis = args.single_layer_analysis
    multiple_layer_analysis = args.multiple_layer_analysis


    print(f"Model: {model_name}, Batch size: {batch_size}, Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}, Device: {device}")
    print(f"Noise: {percent_train_noisy_samps}% with label {desired_train_noise_label}")
    # train_texts, train_labels = sample_50_percent(train_texts, train_labels)

    # Count labels for train, val, and test datasets
    train_label_counts = count_labels(train_labels, "Train")
    val_label_counts = count_labels(val_labels, "Validation")
    test_label_counts = count_labels(test_labels, "Test")

    num_train_noisy_samps = int((percent_train_noisy_samps/100)*(sum(list(train_label_counts.values()))))
    print(num_train_noisy_samps)

    train_texts, train_labels, lm_texts, lm_labels = add_lm_to_texts(train_texts, train_labels, desired_train_noise_label, num_train_noisy_samps, num_labels = num_labels, seed = seed)

    train_label_counts = count_labels(train_labels, "Train")



    # Tokenization and Data Preparation
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if(model_name == "openai-community/gpt2" or model_name == "EleutherAI/gpt-neo-125M" or model_name == "gpt2-medium" or model_name == "EleutherAI/pythia-160M"):
        # default to left padding
        tokenizer.padding_side = "left"
        # Define PAD Token = EOS Token = 50256
        tokenizer.pad_token = tokenizer.eos_token


    train_dataset = TweetsDataset(train_texts, train_labels, tokenizer)
    val_dataset = TweetsDataset(val_texts, val_labels, tokenizer)
    test_dataset = TweetsDataset(test_texts, test_labels, tokenizer)
    lm_dataset = TweetsDataset(lm_texts, lm_labels, tokenizer)

    num_workers = min(3, os.cpu_count())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers = num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers = num_workers)
    lm_loader = DataLoader(lm_dataset, batch_size=1, num_workers = num_workers)


    if(single_layer_analysis):

        train_acc_list_layers, train_loss_list_layers = {}, {}
        val_acc_list_layers, val_loss_list_layers = {}, {}
        test_acc_list_layers, test_loss_list_layers = {}, {}
        lm_acc_list_layers, lm_loss_list_layers = {}, {}
        for layer_idx in range(12):
            print("For layer removal: ", layer_idx)
            # Model setup
            model = CustomClassificationModel_layer_analysis(model_name, num_labels, remove_layers = [layer_idx])
            model = model.to(device)

            for name, param in model.named_parameters():
                print(f"Layer: {name}, Size: {param.size()}, req grad: {param.requires_grad}")
            
            
            # Optimizer
            optimizer = AdamW(model.parameters(), lr=learning_rate)

            train_acc_list, train_loss_list = [], []
            val_acc_list, val_loss_list = [], []
            test_acc_list, test_loss_list = [], []
            lm_acc_list, lm_loss_list = [], []

            epochs_list = list(range(1, epochs + 1))

            # Training loop
            for epoch in range(epochs):
                print(f"Epoch {epoch + 1}/{epochs}")

                # Training phase
                train_loss, train_acc, train_precision, train_recall, train_f1, optimizer = train_epoch(model, train_loader, optimizer, device)
                print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}")

                # Validation phase
                val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate_model(model, val_loader, device)
                print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

                # Testing phase
                test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader, device)
                print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

                lm_loss, lm_acc, lm_precision, lm_recall, lm_f1 = evaluate_model(model, lm_loader, device)
                print(f"LM Loss: {lm_loss:.4f}, Accuracy: {lm_acc:.4f}, Precision: {lm_precision:.4f}, Recall: {lm_recall:.4f}, F1: {lm_f1:.4f}")
                
                train_acc_list.append(train_acc*100)
                val_acc_list.append(val_acc*100)
                test_acc_list.append(test_acc*100)
                lm_acc_list.append(lm_acc*100)

                train_loss_list.append(train_loss)
                val_loss_list.append(val_loss)
                test_loss_list.append(test_loss)
                lm_loss_list.append(lm_loss)
            
            train_acc_list_layers[layer_idx] = train_acc_list
            val_acc_list_layers[layer_idx] = val_acc_list
            test_acc_list_layers[layer_idx] = test_acc_list
            lm_acc_list_layers[layer_idx] = lm_acc_list

            train_loss_list_layers[layer_idx] = train_loss_list
            val_loss_list_layers[layer_idx] = val_loss_list
            test_loss_list_layers[layer_idx] = test_loss_list
            lm_loss_list_layers[layer_idx] = lm_loss_list


        all_acc_metrics = {
            "train": train_acc_list_layers,
            "val": val_acc_list_layers,
            "test": test_acc_list_layers,
            "lm": lm_acc_list_layers,
        }

        # Combine loss metrics into another dictionary
        all_loss_metrics = {
            "train": train_loss_list_layers,
            "val": val_loss_list_layers,
            "test": test_loss_list_layers,
            "lm": lm_loss_list_layers,
        }

        # Save all accuracy metrics to a single JSON file
        save_combined_metrics_to_json(all_acc_metrics, f"emotions_plots_bias_impact/all_accuracy_metrics_{percent_train_noisy_samps}_single_layer_analysis.json")

        # Save all loss metrics to a single JSON file
        save_combined_metrics_to_json(all_loss_metrics, f"emotions_plots_bias_impact/all_loss_metrics_{percent_train_noisy_samps}_single_layer_analysis.json")

    elif(multiple_layer_analysis):

        train_acc_list_layers, train_loss_list_layers = {}, {}
        val_acc_list_layers, val_loss_list_layers = {}, {}
        test_acc_list_layers, test_loss_list_layers = {}, {}
        lm_acc_list_layers, lm_loss_list_layers = {}, {}
        layers_mapping = {'early': [0,1,2,3], 'middle': [4,5,6,7], 'later': [8,9,10,11]}
        if(model_name == "gpt2-medium"):
            layers_mapping = {'early': [0,1,2,3,4,5,6,7], 'middle': [8,9,10,11,12,13,14,15], 'later': [16,17,18,19,20,21,22,23]}
        for layers_type, layer_idx_list in layers_mapping.items():
            print(f"For {layers_type} layers: ", layer_idx_list)
            # Model setup
            model = CustomClassificationModel_layer_analysis(model_name, num_labels, tokenizer = tokenizer, remove_layers = layer_idx_list)
            model = model.to(device)

            for name, param in model.named_parameters():
                print(f"Layer: {name}, Size: {param.size()}, req grad: {param.requires_grad}")
            
            # Optimizer
            optimizer = AdamW(model.parameters(), lr=learning_rate)

            train_acc_list, train_loss_list = [], []
            val_acc_list, val_loss_list = [], []
            test_acc_list, test_loss_list = [], []
            lm_acc_list, lm_loss_list = [], []

            epochs_list = list(range(1, epochs + 1))

            # Training loop
            for epoch in range(epochs):
                print(f"Epoch {epoch + 1}/{epochs}")

                # Training phase
                train_loss, train_acc, train_precision, train_recall, train_f1, optimizer = train_epoch(model, train_loader, optimizer, device)
                print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}")

                # Validation phase
                val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate_model(model, val_loader, device)
                print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

                # Testing phase
                test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader, device)
                print(f"Testing Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")

                lm_loss, lm_acc, lm_precision, lm_recall, lm_f1 = evaluate_model(model, lm_loader, device, lm = True)
                print(f"LM Loss: {lm_loss:.4f}, Accuracy: {lm_acc:.4f}, Precision: {lm_precision:.4f}, Recall: {lm_recall:.4f}, F1: {lm_f1:.4f}")

    else:
        # Model setup
        model = CustomClassificationModel(model_name, num_labels, remove = remove)
        model = model.to(device)

        for name, param in model.named_parameters():
            print(f"Layer: {name}, Size: {param.size()}, req grad: {param.requires_grad}")
        
        # Optimizer
        optimizer = AdamW(model.parameters(), lr=learning_rate)

        train_acc_list, train_loss_list = [], []
        val_acc_list, val_loss_list = [], []
        test_acc_list, test_loss_list = [], []
        lm_acc_list, lm_loss_list = [], []

        epochs_list = list(range(1, epochs + 1))

        # Training loop
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")

            # Training phase
            train_loss, train_acc, train_precision, train_recall, train_f1, optimizer = train_epoch(model, train_loader, optimizer, device)
            print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}")

            # Validation phase
            val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate_model(model, val_loader, device)
            print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

            # Testing phase
            test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader, device)
            print(f"Testing Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")

            lm_loss, lm_acc, lm_precision, lm_recall, lm_f1 = evaluate_model(model, lm_loader, device, lm = True)
            print(f"LM Loss: {lm_loss:.4f}, Accuracy: {lm_acc:.4f}, Precision: {lm_precision:.4f}, Recall: {lm_recall:.4f}, F1: {lm_f1:.4f}")
            
            train_acc_list.append(train_acc*100)
            val_acc_list.append(val_acc*100)
            test_acc_list.append(test_acc*100)
            lm_acc_list.append(lm_acc*100)

            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            test_loss_list.append(test_loss)
            lm_loss_list.append(lm_loss)
            

        # # Save accuracy metrics to CSV
        # save_metrics_to_csv(epochs_list, train_acc_list, val_acc_list, test_acc_list, lm_acc_list,
        #                     csv_path=f"tweets_plots_bias_impact/accuracy_metrics_{percent_train_noisy_samps}_remove_{remove}_seed_{seed}_gpt2_wts_bias_2.csv")
        
        # # Save loss metrics to CSV
        # save_metrics_to_csv(epochs_list, train_loss_list, val_loss_list, test_loss_list, lm_loss_list,
        #                     csv_path=f"tweets_plots_bias_impact/loss_metrics_{percent_train_noisy_samps}_remove_{remove}_seed_{seed}_gpt2_wts_bias_2.csv")

        # # Plot accuracy trends
        # plot_metrics(epochs_list, train_acc_list, val_acc_list, test_acc_list, lm_acc_list,
        #             metric_type="Accuracy", save_path=f"tweets_plots_bias_impact/accuracy_trends_{percent_train_noisy_samps}_remove_{remove}_seed_{seed}_gpt2_wts_bias_2.pdf")
        
        # # Plot loss trends
        # plot_metrics(epochs_list, train_loss_list, val_loss_list, test_loss_list, lm_loss_list,
        #             metric_type="Loss", save_path=f"tweets_plots_bias_impact/loss_trends_{percent_train_noisy_samps}_remove_{remove}_seed_{seed}_gpt2_wts_bias_2.pdf")
        

        
        print("Label Memorization Analysis: ")
        lm_loss, lm_acc, lm_precision, lm_recall, lm_f1 = evaluate_model(model, lm_loader, device)
        print(f"LM Loss: {lm_loss:.4f}, Accuracy: {lm_acc:.4f}, Precision: {lm_precision:.4f}, Recall: {lm_recall:.4f}, F1: {lm_f1:.4f}")

    # torch.save(model.state_dict(),f'saved_models/bert_emotions_model_{num_train_noisy_samps}_lm_wts_random_noisy_label_layernorm_removed.pth')
        

def swap_classes(df):
    df.loc[df['label'] == 5, 'label'] = -1  # Temporarily change 5 to -1
    df.loc[df['label'] == 3, 'label'] = 5   # Change 3 to 5
    df.loc[df['label'] == -1, 'label'] = 3  # Change temporary -1 to 3
    return df

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Fine-tune a BERT model with custom parameters.")
    
    parser.add_argument("--model_name", type=str, default="microsoft/DialogRPT-updown", help="Model name to fine-tune.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=70, help="Number of epochs for training.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for training.")
    parser.add_argument("--remove", type=str, default="none", help="Parameter to remove something (if applicable).")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--percent_train_noisy_samps", type=int, default=1, help="Percentage of noisy samples in training data.")
    parser.add_argument("--desired_train_noise_label", type=int, default=3, help="Label to assign to noisy training samples.")
    parser.add_argument("--single_layer_analysis", action="store_true", help="Enable single-layer analysis. Default is False.")
    parser.add_argument("--multiple_layer_analysis", action="store_true", help="Enable multiple-layer analysis. Default is False.")


    args = parser.parse_args()

    ds = load_dataset("cardiffnlp/tweet_topic_single")


    # Load datasets
    # Convert each split to a pandas DataFrame
    train_df = ds['train_2020'].to_pandas()
    val_df = ds['validation_2020'].to_pandas()
    test_df = ds['test_2020'].to_pandas()

    # Apply the swap to train, validation, and test datasets
    train_df = swap_classes(train_df)
    val_df = swap_classes(val_df)
    test_df = swap_classes(test_df)



    seeds_list = [28]
    for seed in seeds_list:
        print("---------------------------------------------------------------------------")
        print("Results for seed: " ,seed)
        finetune_gpt(args, np.array(train_df['text']), np.array(train_df['label']), \
                        np.array(val_df['text']), np.array(val_df['label']), \
                        np.array(test_df['text']), np.array(test_df['label']), seed = seed)
        print("---------------------------------------------------------------------------")
        print()
        print()
    print()