import os
import glob
import pandas as pd
import torch
import argparse
import json
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, AdamW, DeiTForImageClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from collections import Counter
import random
import copy
from tqdm import tqdm
from scipy.stats import ttest_rel
from torch.nn.functional import softmax
from sklearn.model_selection import StratifiedShuffleSplit
from PIL import Image

# Custom dataset class
class UTKFaceDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(torch.tensor(image))
        return {"image": image,  "label": label}




# Define a custom model class
class CustomClassificationModel(nn.Module):
    def __init__(self, model_name, num_labels, remove = 'none'):
        super(CustomClassificationModel, self).__init__()
        self.model_name = model_name
        if(self.model_name == "google/vit-base-patch16-224-in21k"):
            self.backbone = ViTForImageClassification.from_pretrained(
                                "google/vit-base-patch16-224",
                                num_labels=num_labels,  # Number of classes in CIFAR-10
                                ignore_mismatched_sizes=True
                            )
        elif(self.model_name == "facebook/deit-base-distilled-patch16-224"):
            self.backbone = DeiTForImageClassification.from_pretrained(
                                "facebook/deit-base-distilled-patch16-224",
                                num_labels=num_labels,  # Number of classes in CIFAR-10
                                ignore_mismatched_sizes=True
                            )
        
        if(model_name == "google/vit-base-patch16-224-in21k"):
            # # Remove bias from transformer layers (attention and feedforward layers)
            for name, module in self.backbone.named_modules():

                if(remove == 'attention'):
                    if('attention' in name):
                        module.bias = None
                
                elif(remove == 'ffn'):
                    if('intermediate' in name):
                        module.bias = None
                    
                elif(remove == 'output'):
                    if('attention' not in name and 'output' in name):
                        module.bias = None
                    
                elif(remove == 'layer_norm'):
                    if('layernorm' in name):
                        module.weight = None
                        module.bias = None
                
                elif(remove == 'attention_layer_norm'):
                    if('layernorm_before' in name):
                        module.weight = None
                        module.bias = None
                    
                elif(remove == 'output_layer_norm'):
                    if('layernorm_after' in name):
                        module.weight = None
                        module.bias = None
        

        if(model_name == "facebook/deit-base-distilled-patch16-224"):
            # # Remove bias from transformer layers (attention and feedforward layers)
            for name, module in self.backbone.named_modules():

                if(remove == 'attention'):
                    if('attention' in name):
                        module.bias = None
                
                elif(remove == 'ffn'):
                    if('intermediate' in name):
                        module.bias = None
                    
                elif(remove == 'output'):
                    if('attention' not in name and 'output' in name):
                        module.bias = None
                    
                elif(remove == 'layer_norm'):
                    if('layernorm' in name):
                        module.weight = None
                        module.bias = None
                
                elif(remove == 'attention_layer_norm'):
                    if('layernorm_before' in name):
                        module.weight = None
                        module.bias = None
                    
                elif(remove == 'output_layer_norm'):
                    if('layernorm_after' in name):
                        module.weight = None
                        module.bias = None


    def forward(self, input_imgs):
        outputs = self.backbone(input_imgs)
        return outputs
        

# Define a custom model class
class CustomClassificationModel_layer_analysis(nn.Module):
    def __init__(self, model_name, num_labels, remove_layers = None):
        super(CustomClassificationModel_layer_analysis, self).__init__()
        self.model_name = model_name
        if(self.model_name == "google/vit-base-patch16-224-in21k"):
            self.backbone = ViTForImageClassification.from_pretrained(
                                "google/vit-base-patch16-224",
                                num_labels=num_labels,  # Number of classes in CIFAR-10
                                ignore_mismatched_sizes=True
                            )
        elif(self.model_name == "facebook/deit-base-distilled-patch16-224"):
            self.backbone = DeiTForImageClassification.from_pretrained(
                                "facebook/deit-base-distilled-patch16-224",
                                num_labels=num_labels,  # Number of classes in CIFAR-10
                                ignore_mismatched_sizes=True
                            )
        
        if(model_name == "google/vit-base-patch16-224-in21k"):
            # # Remove bias from transformer layers (attention and feedforward layers)
            for name, module in self.backbone.named_modules():
                    
                if('layernorm_before' in name or "layernorm_after" in name):
                    layer_index = int(name.split(".layer.")[1].split(".")[0])
                    if(layer_index in remove_layers):
                        module.weight = None
                        module.bias = None
                
        

        if(model_name == "facebook/deit-base-distilled-patch16-224"):
            # # Remove bias from transformer layers (attention and feedforward layers)
            for name, module in self.backbone.named_modules():

                if('layernorm_before' in name or "layernorm_after" in name):
                    layer_index = int(name.split(".layer.")[1].split(".")[0])
                    if(layer_index in remove_layers):
                        module.weight = None
                        module.bias = None


    def forward(self, input_imgs):
        outputs = self.backbone(input_imgs)
        return outputs


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
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        # Forward pass
        logits = model(images).logits
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
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            logits = model(images).logits
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
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            logits = model(images).logits

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


def add_random_label(original_label, idx, seed, num_labels):

    # Set seed for reproducibility
    if seed is not None:
        random.seed(seed + idx)
    
    # Generate a random label different from the original label
    # Assuming labels are integers starting from 0 to max_label
    possible_labels = list(range(0, num_labels))
    possible_labels.remove(original_label)  # Remove the original label
    
    # Select a random label from the remaining options
    random_label = random.choice(possible_labels)
    
    return random_label


def add_lm_to_texts(texts, labels, class_label, n, seed=28, num_labels = 5):

    random.seed(seed)

    # Find indices of samples belonging to the given class label
    indices = [i for i, label in enumerate(labels) if label == class_label]

    # Randomly select n indices to modify
    indices_to_modify = random.sample(indices, min(n, len(indices)))
    # Copy texts to avoid modifying the original data
    train_texts_copy, train_labels_copy = copy.deepcopy(texts), copy.deepcopy(labels)
    lm_texts = []
    lm_labels = []
    lm_labels_actual = []

    for idx in indices_to_modify:
        lm_labels_actual.append(train_labels_copy[idx])
        train_labels_copy[idx] = add_random_label(train_labels_copy[idx], idx, seed, num_labels) #noisy label
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


def remove_classes(texts, labels, classes_to_remove):
    """
    Removes samples belonging to specified classes from the dataset.

    Parameters:
        texts (list): List of texts.
        labels (list): List of corresponding labels.
        classes_to_remove (set): Classes to be removed from the dataset.

    Returns:
        filtered_texts (list): Texts with specified classes removed.
        filtered_labels (list): Labels with specified classes removed.
    """
    filtered_texts = []
    filtered_labels = []

    for text, label in zip(texts, labels):
        if label not in classes_to_remove:
            filtered_texts.append(text)
            filtered_labels.append(label)

    return filtered_texts, filtered_labels


def calculate_ln_derivatives(model, model_name, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    attn_layernorm_derivatives = dict()
    output_layernorm_derivatives = dict()
    sample_count = 0
    
    def hook_fn(module, grad_input, grad_output, storage, layer_index):
        if grad_input[0] is not None:  # Ensure valid gradient
            grad_avg = grad_input[0].abs().mean(dim=1)  # Average across tokens
            
            if layer_index not in storage:
                storage[layer_index] = torch.zeros_like(grad_avg)
            
            storage[layer_index] += grad_avg  # Accumulate across batches

    hooks = []
    for name, module in model.named_modules():
        if 'layernorm_before' in name:
            layer_index = int(name.split(".layer.")[1].split(".")[0])
            hook = module.register_full_backward_hook(
                lambda mod, gin, gout, idx=layer_index: hook_fn(mod, gin, gout, attn_layernorm_derivatives, idx)
            )
            hooks.append(hook)
        elif 'layernorm_after' in name:
            layer_index = int(name.split(".layer.")[1].split(".")[0])
            hook = module.register_full_backward_hook(
                lambda mod, gin, gout, idx=layer_index: hook_fn(mod, gin, gout, output_layernorm_derivatives, idx)
            )
            hooks.append(hook)

    
    for batch in loader:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        sample_count += images.shape[0]

        logits = model(images).logits
        loss = criterion(logits, labels)
        loss.backward()
    
    for hook in hooks:
        hook.remove()
    
    # Normalize by total samples and compute Frobenius norm
    for layer_index in attn_layernorm_derivatives:
        attn_layernorm_derivatives[layer_index] /= sample_count
        attn_layernorm_derivatives[layer_index] = torch.norm(attn_layernorm_derivatives[layer_index], p='fro').item()
    
    for layer_index in output_layernorm_derivatives:
        output_layernorm_derivatives[layer_index] /= sample_count
        output_layernorm_derivatives[layer_index] = torch.norm(output_layernorm_derivatives[layer_index], p='fro').item()
    
    print("Attention LayerNorm derivatives: ", attn_layernorm_derivatives)
    print("Output LayerNorm derivatives: ", output_layernorm_derivatives)
    print()
    
    return attn_layernorm_derivatives, output_layernorm_derivatives


def calculate_ln_derivatives_output(model, model_name, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    attn_layernorm_derivatives = dict()
    output_layernorm_derivatives = dict()
    sample_count = 0
    
    def hook_fn(module, grad_input, grad_output, storage, layer_index):
        if grad_output[0] is not None:  # Ensure valid gradient
            grad_avg = grad_output[0].abs().mean(dim=1)  # Average across tokens
            
            if layer_index not in storage:
                storage[layer_index] = torch.zeros_like(grad_avg)
            
            storage[layer_index] += grad_avg  # Accumulate across batches

    hooks = []
    for name, module in model.named_modules():
        if 'layernorm_before' in name:
            layer_index = int(name.split(".layer.")[1].split(".")[0])
            hook = module.register_full_backward_hook(
                lambda mod, gin, gout, idx=layer_index: hook_fn(mod, gin, gout, attn_layernorm_derivatives, idx)
            )
            hooks.append(hook)
        elif 'layernorm_after' in name:
            layer_index = int(name.split(".layer.")[1].split(".")[0])
            hook = module.register_full_backward_hook(
                lambda mod, gin, gout, idx=layer_index: hook_fn(mod, gin, gout, output_layernorm_derivatives, idx)
            )
            hooks.append(hook)

    
    for batch in loader:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        sample_count += images.shape[0]

        logits = model(images).logits
        loss = criterion(logits, labels)
        loss.backward()
    
    for hook in hooks:
        hook.remove()
    
    # Normalize by total samples and compute Frobenius norm
    for layer_index in attn_layernorm_derivatives:
        attn_layernorm_derivatives[layer_index] /= sample_count
        attn_layernorm_derivatives[layer_index] = torch.norm(attn_layernorm_derivatives[layer_index], p='fro').item()
    
    for layer_index in output_layernorm_derivatives:
        output_layernorm_derivatives[layer_index] /= sample_count
        output_layernorm_derivatives[layer_index] = torch.norm(output_layernorm_derivatives[layer_index], p='fro').item()
    
    print("Attention LayerNorm derivatives: ", attn_layernorm_derivatives)
    print("Output LayerNorm derivatives: ", output_layernorm_derivatives)
    print()
    
    return attn_layernorm_derivatives, output_layernorm_derivatives

def capture_ln_inputs_l2_norm_sigma(model, model_name, loader, device):
    model.eval()
    attn_ln_inputs = dict()
    output_ln_inputs = dict()
    attn_ln_std = dict()
    output_ln_std = dict()
    sample_count = 0
    
    def hook_fn(module, input, output, storage, std_storage, layer_index):
        if input[0] is not None:  # Ensure valid input
            input_avg = input[0].detach().cpu().mean(dim=1)  # Average across tokens
            l2_norm = torch.norm(input_avg, p='fro').item()
            std_dev = input_avg.std().item()

            if layer_index not in storage:
                storage[layer_index] = 0  # Initialize sum
                std_storage[layer_index] = 0  # Store values for std computation
            
            storage[layer_index] += l2_norm  # Accumulate L2 norms
            std_storage[layer_index] += std_dev  # Store std dev values

    hooks = []
    for name, module in model.named_modules():
        if 'layernorm_before' in name:
            layer_index = int(name.split(".layer.")[1].split(".")[0])
            hook = module.register_forward_hook(
                lambda mod, inp, out, idx=layer_index: hook_fn(mod, inp, out, attn_ln_inputs, attn_ln_std, idx)
            )
            hooks.append(hook)
        elif 'layernorm_after' in name:
            layer_index = int(name.split(".layer.")[1].split(".")[0])
            hook = module.register_forward_hook(
                lambda mod, inp, out, idx=layer_index: hook_fn(mod, inp, out, output_ln_inputs, output_ln_std, idx)
            )
            hooks.append(hook)
    
    # Loop over batches
    for batch in loader:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        sample_count += images.shape[0]

        with torch.no_grad():
            model(images)
    
    # Remove hooks after processing
    for hook in hooks:
        hook.remove()

    # Normalize by total samples and compute standard deviation
    for layer_index in attn_ln_inputs:
        attn_ln_inputs[layer_index] /= sample_count
        attn_ln_std[layer_index] /= sample_count
    
    for layer_index in output_ln_inputs:
        output_ln_inputs[layer_index] /= sample_count
        output_ln_std[layer_index] /= sample_count
    
    print("Attention LayerNorm Inputs L2-norm: ", attn_ln_inputs)
    print("Attention LayerNorm Inputs Std-dev: ", attn_ln_std)
    print("Output LayerNorm Inputs L2-norm: ", output_ln_inputs)
    print("Output LayerNorm Inputs Std-dev: ", output_ln_std)
    print()

    return attn_ln_inputs, attn_ln_std, output_ln_inputs, output_ln_std


def gradients_analysis(args, train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels, seed):


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
    model_path = args.model_path


    print(f"Model: {model_name}, Batch size: {batch_size}, Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}, Device: {device}")
    print(f"Noise: {percent_train_noisy_samps}% with label {desired_train_noise_label}")

    # Count labels for train, val, and test datasets
    train_label_counts = count_labels(train_labels, "Train")
    val_label_counts = count_labels(val_labels, "Validation")
    test_label_counts = count_labels(test_labels, "Test")

    num_train_noisy_samps = int((percent_train_noisy_samps/100)*(sum(list(train_label_counts.values()))))
    print(num_train_noisy_samps)

    train_imgs, train_labels, lm_imgs, lm_labels = add_lm_to_texts(train_imgs, train_labels, desired_train_noise_label, num_train_noisy_samps, num_labels = num_labels, seed=seed)
    train_label_counts = count_labels(train_labels, "Train")
    print(len(train_imgs))  # Should be (num_samples, height, width, channels)
    print(train_imgs[0].shape)  # Should be (28, 28, 3) or (28, 28, 1)

    # Transform for resizing and normalization
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Create training and validation datasets
    train_dataset = UTKFaceDataset(train_imgs, train_labels, transform=transform)
    val_dataset = UTKFaceDataset(val_imgs, val_labels, transform=transform)
    test_dataset = UTKFaceDataset(test_imgs, test_labels, transform=transform)
    lm_dataset = UTKFaceDataset(lm_imgs, lm_labels, transform=transform)


    num_workers = min(3, os.cpu_count())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers = num_workers)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers = num_workers)
    lm_loader = DataLoader(lm_dataset, batch_size=1, num_workers = num_workers)


    # Model setup
    model = CustomClassificationModel(model_name, num_labels, remove = remove)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))

    for name, param in model.named_parameters():
        print(f"Layer: {name}, Size: {param.size()}, req grad: {param.requires_grad}")

    # Testing phase
    test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader, device)
    print(f"Testing Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")

    lm_loss, lm_acc, lm_precision, lm_recall, lm_f1 = evaluate_model(model, lm_loader, device, lm = True)
    print(f"LM Loss: {lm_loss:.4f}, Accuracy: {lm_acc:.4f}, Precision: {lm_precision:.4f}, Recall: {lm_recall:.4f}, F1: {lm_f1:.4f}")

    # lm_attn_ln_gradients, lm_output_ln_gradients = calculate_ln_derivatives(model, model_name, lm_loader, device)

    # test_attn_ln_gradietns, test_output_ln_gradients = calculate_ln_derivatives(model, model_name, test_loader, device)

    # lm_attn_ln_gradients, lm_output_ln_gradients = calculate_ln_derivatives_output(model, model_name, lm_loader, device)

    # test_attn_ln_gradietns, test_output_ln_gradients = calculate_ln_derivatives_output(model, model_name, test_loader, device)

    capture_ln_inputs_l2_norm_sigma(model, model_name, lm_loader, device)
    capture_ln_inputs_l2_norm_sigma(model, model_name, test_loader, device)

def parse_utk_face_dataset(dataset_path, seed, ext='jpg'):
    """
    Used to extract information about our dataset. It does iterate over all images and return a DataFrame with
    the data (age, gender and sex) of all files.
    """
    def parse_info_from_file(path):
        """
        Parse information from a single file
        """
        dataset_dict = {
            'ethnicity_id': {
                0: 'white', 
                1: 'black', 
                2: 'asian', 
                3: 'indian', 
                4: 'others'
            },
            'gender_id': {
                0: 'male',
                1: 'female'
            }
        }

        dataset_dict['gender_alias'] = dict((g, i) for i, g in dataset_dict['gender_id'].items())
        dataset_dict['ethnicity_alias'] = dict((r, i) for i, r in dataset_dict['ethnicity_id'].items())
        try:
            filename = os.path.split(path)[1]
            filename = os.path.splitext(filename)[0]
            age, gender, race, _ = filename.split('_')
            return int(age), dataset_dict['gender_id'][int(gender)], int(race)

        except Exception as ex:
            return None, None, None
        
    files = glob.glob(os.path.join(dataset_path, "*.%s" % ext))
    
    records = []
    for file in files:
        info = parse_info_from_file(file)
        records.append(info)
        
    df = pd.DataFrame(records)
    df['file'] = files
    df.columns = ['age', 'gender', 'ethnicity', 'file']
    df = df.dropna()
    df['ethnicity'] = df['ethnicity'].astype(int)
    # print(len(df))
    # print(df)
    # exit()
    # Initialize the StratifiedShuffleSplit
    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)

    # Split the data
    for train_index, test_index in strat_split.split(df, df['ethnicity']):
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]

    return train_df, test_df

def load_utkface_dataset(dataset_path, seed):
    train_df, test_df = parse_utk_face_dataset(dataset_path, seed)

    train_images, train_labels, test_images, test_labels = [], [], [], []

    for index in range(len(train_df)):
        person = train_df.iloc[index]
        person_img = np.array(Image.open(person['file']))
        assert person_img.shape == (200, 200, 3)  # Ensure the image is in (H, W, C) format
        person_ethnicity = person['ethnicity']

        # Convert image to (C, H, W) format (from (H, W, C))
        person_img = np.transpose(person_img, (2, 0, 1))  # Now it becomes (C, H, W)

        train_images.append(person_img)
        train_labels.append(person_ethnicity)

    for index in range(len(test_df)):
        person = test_df.iloc[index]
        person_img = np.array(Image.open(person['file']))
        assert person_img.shape == (200, 200, 3)
        person_ethnicity = person['ethnicity']
        
        # Convert image to (C, H, W) format (from (H, W, C))
        person_img = np.transpose(person_img, (2, 0, 1))  # Now it becomes (C, H, W)

        test_images.append(person_img)
        test_labels.append(person_ethnicity)

    return train_images, train_labels, test_images, test_labels


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Fine-tune a DeiT model with custom parameters")
    
    parser.add_argument("--model_name", type=str, default="facebook/deit-base-distilled-patch16-224", help="Model name to fine-tune.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=70, help="Number of epochs for training.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for training.")
    parser.add_argument("--remove", type=str, default="none", help="Parameter to remove something (if applicable).")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--percent_train_noisy_samps", type=int, default=1, help="Percentage of noisy samples in training data.")
    parser.add_argument("--desired_train_noise_label", type=int, default=2, help="Label to assign to noisy training samples.")
    parser.add_argument("--model_path", type=str, default = "saved_models_bias_impact/utk_face_dataset_model_deit.pth", help = "path of saved model")


    args = parser.parse_args()
    # Define preprocessing transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224 for ViT
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    
    seeds_list = [28]
    for seed in seeds_list:

        train_imgs, train_labels, test_imgs, test_labels = load_utkface_dataset("UTKFace/", seed)
        train_imgs, val_imgs, train_labels, val_labels = train_test_split(
            train_imgs, train_labels, test_size=0.2, stratify=train_labels, random_state=seed
        )


        gradients_analysis(args, train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels, seed)