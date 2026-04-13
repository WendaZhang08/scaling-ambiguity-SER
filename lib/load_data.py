"""
Utility module for loading and saving data for Scaling Ambiguity Augmenting Human Annotation in Speech Emotion Recognition with Audio-Language Models.
This module provides common functions for data loading in the project.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from typing import Dict, List, Union, Tuple, Optional, Any, Literal

# Get the absolute path to the project root directory
MODULE_DIR  = os.path.dirname(os.path.abspath(__file__))

# File path constants with absolute paths
DATA_DIR = os.path.join(MODULE_DIR , 'data')
PROCESSED_DATA_DIR = os.path.join(MODULE_DIR , 'processed_data')
GEMINI_ANNOTATIONS_DIR = os.path.join(PROCESSED_DATA_DIR, 'gemini_annotations')
COMBINED_ANNOTATIONS_DIR = os.path.join(PROCESSED_DATA_DIR, 'combined_annotations')

# Dataset file names
IEMOCAP_RAW_FILE = 'iemocap_ambiguous.json'
MSP_RAW_FILE = 'msp_ambigous.json'

# Human annotations
IEMOCAP_TRAIN_FILE = 'human_iemocap_train_distributions.json'
MSP_TRAIN_FILE = 'human_msp_train_distributions.json'
IEMOCAP_TEST_FILE = 'iemocap_test_distributions.json'
MSP_TEST_FILE = 'msp_test_distributions.json'

# Gemini annotations
GEMINI_IEMOCAP_TRAIN_FILE = 'gemini_iemocap_train_distributions.json'
GEMINI_MSP_TRAIN_FILE = 'gemini_msp_train_distributions.json'
GEMINI_IEMOCAP_TEST_FILE = 'gemini_iemocap_test_distributions.json'
GEMINI_MSP_TEST_FILE = 'gemini_msp_test_distributions.json'

# Combined annotations
COMBINED_IEMOCAP_TRAIN_FILE = 'combined_iemocap_train_distributions.json'
COMBINED_MSP_TRAIN_FILE = 'combined_msp_train_distributions.json'
COMBINED_IEMOCAP_TEST_FILE = 'combined_iemocap_test_distributions.json'
COMBINED_MSP_TEST_FILE = 'combined_msp_test_distributions.json'

# Emotion classes
IEMOCAP_EMOTION_CLASSES_FILE = 'iemocap_emotion_classes.json'
MSP_EMOTION_CLASSES_FILE = 'msp_emotion_classes.json'

# Audio base directories
IEMOCAP_AUDIO_BASE = os.path.join(DATA_DIR, 'IEMOCAP_full_release')
MSP_AUDIO_BASE = os.path.join(DATA_DIR, 'Audio')

# Annotation source types
AnnotationSourceType = Literal['human', 'gemini', 'combined']

def load_json(file_path: str) -> Any:
    """
    Load data from JSON file with error handling.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Loaded JSON data
        
    Raises:
        FileNotFoundError: If the file does not exist
        json.JSONDecodeError: If the file is not valid JSON
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        raise
    except json.JSONDecodeError:
        print(f"Error: File '{file_path}' contains invalid JSON.")
        raise
    except Exception as e:
        print(f"Error loading file '{file_path}': {str(e)}")
        raise


def save_json(data: Any, file_path: str, indent: int = 4) -> None:
    """
    Save data to JSON file with error handling.
    
    Args:
        data: Data to save
        file_path: Path to save the JSON file
        indent: Indentation level for the JSON file
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent)
        print(f"Data successfully saved to '{file_path}'")
    except Exception as e:
        print(f"Error saving data to '{file_path}': {str(e)}")
        raise


def load_raw_dataset(dataset_name: str, custom_path: Optional[str] = None) -> List[Dict]:
    """
    Load raw dataset by name.
    
    Args:
        dataset_name: Name of the dataset ('iemocap' or 'msp')
        custom_path: Optional custom path to the file
        
    Returns:
        Dataset as a list of dictionaries
    """
    if dataset_name.lower() == 'iemocap':
        file_path = custom_path or os.path.join(DATA_DIR, IEMOCAP_RAW_FILE)
    elif dataset_name.lower() == 'msp':
        file_path = custom_path or os.path.join(DATA_DIR, MSP_RAW_FILE)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}. Expected 'iemocap' or 'msp'.")
    
    data = load_json(file_path)
    print(f"Loaded {len(data)} samples from {dataset_name.upper()} dataset")
    return data


def load_train_distributions(dataset_name: str, 
                         annotation_source: AnnotationSourceType = 'human',
                         custom_path: Optional[str] = None) -> List[Dict]:
    """
    Load training data with emotion annotation distributions.
    
    Args:
        dataset_name: Name of the dataset ('iemocap' or 'msp')
        annotation_source: Source of annotations ('human', 'gemini', or 'combined')
        custom_path: Optional custom path to the file
        
    Returns:
        Dataset with emotion distributions
    """
    if custom_path:
        file_path = custom_path
    else:
        if annotation_source == 'human':
            if dataset_name.lower() == 'iemocap':
                file_path = os.path.join(PROCESSED_DATA_DIR, IEMOCAP_TRAIN_FILE)
            elif dataset_name.lower() == 'msp':
                file_path = os.path.join(PROCESSED_DATA_DIR, MSP_TRAIN_FILE)
            else:
                raise ValueError(f"Unknown dataset name: {dataset_name}. Expected 'iemocap' or 'msp'.")
        
        elif annotation_source == 'gemini':
            if dataset_name.lower() == 'iemocap':
                file_path = os.path.join(GEMINI_ANNOTATIONS_DIR, GEMINI_IEMOCAP_TRAIN_FILE)
            elif dataset_name.lower() == 'msp':
                file_path = os.path.join(GEMINI_ANNOTATIONS_DIR, GEMINI_MSP_TRAIN_FILE)
            else:
                raise ValueError(f"Unknown dataset name: {dataset_name}. Expected 'iemocap' or 'msp'.")
        
        elif annotation_source == 'combined':
            if dataset_name.lower() == 'iemocap':
                file_path = os.path.join(COMBINED_ANNOTATIONS_DIR, COMBINED_IEMOCAP_TRAIN_FILE)
            elif dataset_name.lower() == 'msp':
                file_path = os.path.join(COMBINED_ANNOTATIONS_DIR, COMBINED_MSP_TRAIN_FILE)
            else:
                raise ValueError(f"Unknown dataset name: {dataset_name}. Expected 'iemocap' or 'msp'.")
        
        else:
            raise ValueError(f"Unknown annotation source: {annotation_source}. Expected 'human', 'gemini', or 'combined'.")
    
    data = load_json(file_path)
    print(f"Loaded {len(data)} {annotation_source} training samples from {dataset_name.upper()} dataset")
    return data


def load_test_distributions(dataset_name: str, custom_path: Optional[str] = None) -> List[Dict]:
    """
    Load test data with emotion distributions.
    
    Args:
        dataset_name: Name of the dataset ('iemocap' or 'msp')
        custom_path: Optional custom path to the file
        
    Returns:
        Dataset with emotion distributions
    """
    if dataset_name.lower() == 'iemocap':
        file_path = custom_path or os.path.join(PROCESSED_DATA_DIR, IEMOCAP_TEST_FILE)
    elif dataset_name.lower() == 'msp':
        file_path = custom_path or os.path.join(PROCESSED_DATA_DIR, MSP_TEST_FILE)
    elif dataset_name.lower() == 'gemini-iemocap':
        file_path = custom_path or os.path.join(GEMINI_ANNOTATIONS_DIR, GEMINI_IEMOCAP_TEST_FILE)
    elif dataset_name.lower() == 'gemini-msp':
        file_path = custom_path or os.path.join(GEMINI_ANNOTATIONS_DIR, GEMINI_MSP_TEST_FILE)
    elif dataset_name.lower() == 'combined-iemocap':
        file_path = custom_path or os.path.join(COMBINED_ANNOTATIONS_DIR, COMBINED_IEMOCAP_TEST_FILE)
    elif dataset_name.lower() == 'combined-msp':
        file_path = custom_path or os.path.join(COMBINED_ANNOTATIONS_DIR, COMBINED_MSP_TEST_FILE)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}. Expected 'iemocap' or 'msp' or 'gemini-iemocap' or 'gemini-msp' or 'combined-iemocap' or 'combined-msp'.")
    
    data = load_json(file_path)
    print(f"Loaded {len(data)} test samples from {dataset_name.upper()} dataset")
    return data


def get_audio_path(sample: Dict, dataset_name: str) -> str:
    """
    Get the absolute path to an audio file for a sample.
    
    Args:
        sample: A sample from a dataset
        dataset_name: Name of the dataset ('iemocap' or 'msp' or variants)
        
    Returns:
        Absolute path to the audio file
    """
    if 'audio' not in sample:
        raise ValueError(f"Sample does not have an 'audio' field: {sample}")
    
    relative_path = sample['audio']
    
    # Map dataset names to base types
    dataset_lower = dataset_name.lower()
    
    if dataset_lower in ['iemocap', 'gemini-iemocap', 'combined-iemocap']:
        # For IEMOCAP-based datasets, check if the path is already relative to IEMOCAP_full_release
        if relative_path.startswith('IEMOCAP_full_release'):
            # Remove the prefix to avoid double path
            relative_path = relative_path.replace('IEMOCAP_full_release/', '')
        return os.path.join(IEMOCAP_AUDIO_BASE, relative_path)
    
    elif dataset_lower in ['msp', 'gemini-msp', 'combined-msp']:
        # For MSP-based datasets, check if the path is already relative to Audio
        if relative_path.startswith('Audio/'):
            # Remove the prefix to avoid double path
            relative_path = relative_path.replace('Audio/', '')
        return os.path.join(MSP_AUDIO_BASE, relative_path)
    
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}. Expected 'iemocap', 'msp', or their variants.")


def audio_file_exists(sample: Dict, dataset_name: str) -> bool:
    """
    Check if an audio file exists for a sample.
    
    Args:
        sample: A sample from a dataset
        dataset_name: Name of the dataset ('iemocap' or 'msp')
        
    Returns:
        True if the audio file exists, False otherwise
    """
    try:
        audio_path = get_audio_path(sample, dataset_name)
        return os.path.exists(audio_path)
    except:
        return False


def load_emotion_classes(dataset_name: str, custom_path: Optional[str] = None) -> List[str]:
    """
    Load emotion classes for a dataset.
    
    Args:
        dataset_name: Name of the dataset ('iemocap' or 'msp' or variants)
        custom_path: Optional custom path to the file
        
    Returns:
        List of emotion class names
    """
    dataset_lower = dataset_name.lower()
    
    if dataset_lower in ['iemocap', 'gemini-iemocap', 'combined-iemocap']:
        file_path = custom_path or os.path.join(PROCESSED_DATA_DIR, IEMOCAP_EMOTION_CLASSES_FILE)
    elif dataset_lower in ['msp', 'gemini-msp', 'combined-msp']:
        file_path = custom_path or os.path.join(PROCESSED_DATA_DIR, MSP_EMOTION_CLASSES_FILE)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}. Expected 'iemocap', 'msp', or their variants.")
    
    data = load_json(file_path)
    print(f"Loaded {len(data)} emotion classes for {dataset_name.upper()} dataset")
    return data


def load_all_data(dataset_name: str, 
               annotation_source: AnnotationSourceType = 'human') -> Tuple[List[Dict], List[Dict], List[Dict], List[str]]:
    """
    Load all data for a given dataset including raw data, train distributions, 
    test distributions, and emotion classes.
    
    Args:
        dataset_name: Name of the dataset ('iemocap' or 'msp')
        annotation_source: Source of annotations for training data ('human', 'gemini', or 'combined')
        
    Returns:
        Tuple of (raw_data, train_data, test_data, emotion_classes)
    """
    raw_data = load_raw_dataset(dataset_name)
    train_data = load_train_distributions(dataset_name, annotation_source)
    test_data = load_test_distributions(dataset_name)
    emotion_classes = load_emotion_classes(dataset_name)
    
    return raw_data, train_data, test_data, emotion_classes

def convert_list_to_distribution(data: List[Dict]) -> List[Dict]:
    """
    Convert emotion annotations from list format to distribution format.
    
    Args:
        data: Dataset with emotion annotations as lists
        
    Returns:
        Dataset with emotion annotations as distributions
    """
    converted_data = []
    
    for item in data:
        item_copy = item.copy()
        
        if isinstance(item['emotion'], list):
            # Count occurrences of each emotion
            emotion_counts = Counter(item['emotion'])
            total = len(item['emotion'])
            
            # Convert to distribution
            emotion_dist = {emotion: count/total for emotion, count in emotion_counts.items()}
            item_copy['emotion'] = emotion_dist
        
        converted_data.append(item_copy)
    
    return converted_data

# Helper function to ensure directories exist
def ensure_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(GEMINI_ANNOTATIONS_DIR, exist_ok=True)
    os.makedirs(COMBINED_ANNOTATIONS_DIR, exist_ok=True)
    
def load_all_annotation_versions(dataset_name: str) -> Dict[str, List[Dict]]:
    """
    Load all versions of annotations (human, Gemini-2.5 Pro, combined) for training data.
    
    Args:
        dataset_name: Name of the dataset ('iemocap' or 'msp')
        
    Returns:
        Dictionary with keys 'human', 'gemini', 'combined' containing respective datasets
    """
    result = {}
    
    # Try to load each annotation source
    for source in ['human', 'gemini', 'combined']:
        try:
            result[source] = load_train_distributions(dataset_name, source)
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"Warning: Could not load {source} annotations for {dataset_name}")
            result[source] = None
    
    return result

