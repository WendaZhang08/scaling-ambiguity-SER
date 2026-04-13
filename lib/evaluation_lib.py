"""
Evaluation library for ambiguous emotion recognition.

This module provides functions for evaluating emotion distribution predictions:
- Distribution metrics: Jensen-Shannon Divergence (JS), Bhattacharyya Coefficient (BC), R-squared (R²)
- Calibration metrics: Expected Calibration Error (ECE)
- Traditional metrics: Accuracy and F1-score (based on highest probability emotion)

Usage:
    # Import the module
    from evaluation.evaluation import evaluate_model_predictions, compare_annotation_models

    # To evaluate model-generated annotations against ground truth
    results = evaluate_model_predictions('path/to/reference.json', 'path/to/prediction.json', 'path/to/results.json')

    # To compare multiple annotation models
    df, results = compare_annotation_models('path/to/human_annotations.json', ['path/to/model1.json', 'path/to/model2.json'], 'path/to/comparison_results.json')
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.metrics import accuracy_score, f1_score, r2_score, confusion_matrix
import scipy.stats as stats
import glob
from scipy.stats import entropy as scipy_entropy

# Define emotion labels for different datasets
IEMOCAP_EMOTIONS = ['Anger', 'Happiness', 'Neutral state', 'Sadness']
MSP_EMOTIONS = ['Angry', 'Happy', 'Neutral', 'Sad']

# Dictionary to map dataset names to their emotion labels
DATASET_EMOTIONS = {
    'iemocap': IEMOCAP_EMOTIONS,
    'msp': MSP_EMOTIONS
}

def load_distribution_file(file_path: str) -> List[Dict[str, Any]]:
    """Load emotion distributions from a JSON file.

    Args:
        file_path: Path to the JSON file containing emotion distributions

    Returns:
        List of distribution objects
    """
    with open(file_path, 'r') as f:
        distributions = json.load(f)
    return distributions


def detect_dataset_from_file(file_path: str) -> str:
    """Detect which dataset a file belongs to based on the file name or path.

    Args:
        file_path: Path to the annotation file

    Returns:
        Dataset name: 'iemocap' or 'msp'
    """
    filename = os.path.basename(file_path).lower()
    if 'iemocap' in filename:
        return 'iemocap'
    elif 'msp' in filename:
        return 'msp'

    # If we can't determine from filename, try to inspect the file content
    try:
        data = load_distribution_file(file_path)
        if data and 'emotion' in data[0]:
            emotions = list(data[0]['emotion'].keys())
            # Check if any emotion matches IEMOCAP emotions
            if any(e in IEMOCAP_EMOTIONS for e in emotions):
                return 'iemocap'
            # Check if any emotion matches MSP emotions
            if any(e in MSP_EMOTIONS for e in emotions):
                return 'msp'
    except Exception as e:
        print(f"Error inspecting file content: {e}")

    # Default to IEMOCAP if we can't determine
    print(f"Warning: Could not determine dataset for {file_path}, defaulting to IEMOCAP")
    return 'iemocap'


def convert_to_vector_representation(
    distribution: Dict[str, float], emotions: List[str]
) -> np.ndarray:
    """Convert an emotion distribution dictionary to a fixed-length vector.

    Args:
        distribution: Dictionary mapping emotion names to probabilities
        emotions: List of all possible emotions to include in the vector

    Returns:
        Numpy array of emotion probabilities in a fixed order
    """
    vector = np.zeros(len(emotions))
    for i, emotion in enumerate(emotions):
        vector[i] = distribution.get(emotion, 0.0)

    # Normalize the vector if it doesn't sum to 1
    if np.sum(vector) > 0 and abs(np.sum(vector) - 1.0) > 1e-6:
        vector = vector / np.sum(vector)

    return vector


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Calculate the Jensen-Shannon divergence between two probability distributions.

    Args:
        p: First probability distribution
        q: Second probability distribution

    Returns:
        JS divergence value between 0 and 1
    """
    # Ensure vectors sum to 1 (normalize if needed)
    p = p / np.sum(p) if np.sum(p) > 0 else p
    q = q / np.sum(q) if np.sum(q) > 0 else q

    # Calculate the middle point
    m = 0.5 * (p + q)

    # Calculate the JSD (using base 2 logarithm for normalization to [0,1])
    jsd = 0.5 * (stats.entropy(p, m, base=2) + stats.entropy(q, m, base=2))

    return jsd


def bhattacharyya_coefficient(p: np.ndarray, q: np.ndarray) -> float:
    """Calculate the Bhattacharyya coefficient (BC) between two probability distributions.

    Args:
        p: First probability distribution
        q: Second probability distribution

    Returns:
        BC value between 0 and 1, where 1 means identical distributions
    """
    # Ensure vectors sum to 1 (normalize if needed)
    p = p / np.sum(p) if np.sum(p) > 0 else p
    q = q / np.sum(q) if np.sum(q) > 0 else q

    # Calculate the BC
    bc = np.sum(np.sqrt(p * q))

    return bc


def expected_calibration_error(
    predictions: List[np.ndarray],
    references: List[np.ndarray],
    n_bins: int = 10
) -> Tuple[float, List[float], List[float]]:
    """Calculate Expected Calibration Error (ECE).

    Args:
        predictions: List of predicted probability distributions
        references: List of ground truth distributions
        n_bins: Number of bins for calculating ECE

    Returns:
        Tuple of (ECE value, confidence_bins, accuracy_bins) for plotting
    """
    confidences = []
    accuracies = []
    bin_counts = np.zeros(n_bins)

    # For each prediction, get max confidence and whether it matches ground truth max
    for pred, ref in zip(predictions, references):
        pred_class = np.argmax(pred)
        ref_class = np.argmax(ref)
        confidence = pred[pred_class]
        accuracy = 1.0 if pred_class == ref_class else 0.0

        # Assign to bin
        bin_idx = min(int(confidence * n_bins), n_bins - 1)
        bin_counts[bin_idx] += 1

        confidences.append((bin_idx, confidence))
        accuracies.append((bin_idx, accuracy))

    # Calculate average confidence and accuracy per bin
    confidence_bins = np.zeros(n_bins)
    accuracy_bins = np.zeros(n_bins)

    for bin_idx in range(n_bins):
        # Get all confidences and accuracies for this bin
        bin_confidences = [conf for idx, conf in confidences if idx == bin_idx]
        bin_accuracies = [acc for idx, acc in accuracies if idx == bin_idx]

        if len(bin_confidences) > 0:
            confidence_bins[bin_idx] = np.mean(bin_confidences)
            accuracy_bins[bin_idx] = np.mean(bin_accuracies)

    # Calculate ECE as weighted average of |confidence - accuracy|
    ece = 0
    total_samples = len(predictions)

    for bin_idx in range(n_bins):
        if bin_counts[bin_idx] > 0:
            weight = bin_counts[bin_idx] / total_samples
            ece += weight * abs(confidence_bins[bin_idx] - accuracy_bins[bin_idx])

    return ece, confidence_bins.tolist(), accuracy_bins.tolist()


def compute_classification_metrics(
    predictions: List[np.ndarray],
    references: List[np.ndarray],
    emotions: List[str]
) -> Dict[str, float]:
    """Compute standard classification metrics based on the highest probability emotion.

    Args:
        predictions: List of predicted probability distributions
        references: List of ground truth distributions
        emotions: List of all emotions

    Returns:
        Dictionary with accuracy and F1 scores
    """
    # Convert to class indices
    y_pred = [np.argmax(p) for p in predictions]
    y_true = [np.argmax(r) for r in references]

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')

    # Create per-class F1 scores
    per_class_f1 = f1_score(y_true, y_pred, average=None).tolist()
    per_class_f1_dict = {emotion: score for emotion, score in zip(emotions, per_class_f1)}

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_f1": per_class_f1_dict,
        "confusion_matrix": cm
    }


def calculate_r_squared(predictions: List[np.ndarray], references: List[np.ndarray], emotions: List[str]) -> Dict[str, float]:
    """Calculate R-squared (coefficient of determination) for each emotion dimension.

    Args:
        predictions: List of predicted probability distributions
        references: List of ground truth distributions
        emotions: List of emotion labels

    Returns:
        Dictionary with R-squared values for each emotion and the average
    """
    # Initialize R² values for each emotion
    r2_values = {}

    # Convert lists to arrays for easier indexing
    pred_array = np.array(predictions)
    ref_array = np.array(references)

    # Calculate R² for each emotion dimension
    for i, emotion in enumerate(emotions):
        y_true = ref_array[:, i]
        y_pred = pred_array[:, i]
        r2 = r2_score(y_true, y_pred)
        r2_values[emotion] = r2

    # Calculate average R²
    r2_values["average"] = np.mean(list(r2_values.values()))

    return r2_values


def calculate_distribution_entropy(distributions: List[Union[Dict[str, float], np.ndarray]], emotions: Optional[List[str]] = None) -> List[float]:
    """Calculate Shannon entropy for a list of emotion distributions.
    
    Args:
        distributions: List of emotion distributions (either dictionaries or normalized vectors)
        emotions: List of emotion labels (required if distributions are dictionaries)
        
    Returns:
        List of entropy values for each distribution
    """
    entropies = []
    
    for dist in distributions:
        # Convert dictionary to vector if needed
        if isinstance(dist, dict):
            if emotions is None:
                raise ValueError("emotions parameter is required when distributions are dictionaries")
            probs = [dist.get(emotion, 0.0) for emotion in emotions]
            # Normalize if needed
            if sum(probs) > 0:
                probs = [p / sum(probs) for p in probs]
        else:
            probs = dist
            
        # Skip distributions that sum to 0
        if isinstance(probs, np.ndarray) and np.sum(probs) == 0:
            entropies.append(0.0)
            continue
            
        # Calculate entropy (use base 2 for bits of information)
        try:
            dist_entropy = scipy_entropy(probs, base=2)
            entropies.append(float(dist_entropy))
        except ValueError:
            # Handle case where probabilities don't sum to 1 due to floating point errors
            entropies.append(0.0)
    
    return entropies


def calculate_inter_annotator_agreement(raw_annotations: Union[Dict[str, List[Dict]], List[Dict]], 
                                       emotions: List[str], 
                                       annotator_type: str = "all") -> Dict[str, float]:
    """Calculate Fleiss' Kappa to measure inter-annotator agreement.
    
    Args:
        raw_annotations: Either:
            1. Dictionary mapping sample IDs to lists of annotations
               Format: {"sample_id": [{"emotion": "Anger", "annotator_id": "human_1"}, ...]}
            2. List of samples with 'emotion' list for human annotations
               Format: [{"id": "sample_id", "emotion": ["Anger", "Happiness", "Sadness"], ...}]
        emotions: List of emotion categories
        annotator_type: Type of annotators to include - "human", "llm", or "all"
        
    Returns:
        Dictionary with Fleiss' Kappa score and additional metrics
    """
    # Convert to format required for Fleiss' Kappa calculation
    # We need a matrix where rows are samples and columns are emotion categories
    
    rating_matrix = []
    sample_ids = []
    
    # Handle list format (human annotations)
    if isinstance(raw_annotations, list):
        # Convert to dictionary format first
        dict_annotations = {}
        for item in raw_annotations:
            sample_id = item.get('id')
            emotions_list = item.get('emotion', [])
            
            # Convert list of emotions to list of annotation dictionaries
            annotations = []
            for i, emotion in enumerate(emotions_list):
                annotations.append({
                    "annotator_id": f"human_{i}",
                    "emotion": emotion
                })
            
            if sample_id and annotations:
                dict_annotations[sample_id] = annotations
        
        raw_annotations = dict_annotations
    
    # Now process as dictionary
    for sample_id, annotations in raw_annotations.items():
        # Filter annotations by type if needed
        filtered_annotations = []
        
        if annotator_type == "human":
            filtered_annotations = [a for a in annotations if not a.get("annotator_id", "").startswith("gemini_")]
        elif annotator_type == "llm":
            filtered_annotations = [a for a in annotations if a.get("annotator_id", "").startswith("gemini_") or a.get("annotator_id", "").startswith("qwen_")]
        else:  # "all"
            filtered_annotations = annotations
            
        if not filtered_annotations:
            continue
            
        # Count occurrences of each emotion for this sample
        emotion_counts = {emotion: 0 for emotion in emotions}
        for annotation in filtered_annotations:
            emotion = annotation.get("emotion")
            if emotion in emotions:
                emotion_counts[emotion] += 1
                
        # Create a row of counts for each emotion
        row = [emotion_counts[emotion] for emotion in emotions]
        rating_matrix.append(row)
        sample_ids.append(sample_id)
        
    if not rating_matrix:
        return {"fleiss_kappa": 0.0, "p_value": 1.0, "error": "No matching annotations found"}
        
    # Convert to numpy array
    rating_matrix = np.array(rating_matrix)
    
    # Calculate Fleiss' Kappa manually
    try:
        kappa = calculate_fleiss_kappa(rating_matrix)
        
        # Calculate additional statistics
        result = {
            "fleiss_kappa": float(kappa),
            "sample_count": len(rating_matrix),
            "annotator_type": annotator_type
        }
        
        return result
    except Exception as e:
        return {"fleiss_kappa": 0.0, "error": str(e)}
        

def calculate_fleiss_kappa(rating_matrix):
    """
    Robust implementation of Fleiss' Kappa that handles edge cases.
    
    Args:
        rating_matrix: numpy array where rows are samples and columns are categories
                      Each cell contains the count of annotators who chose that category
    
    Returns:
        float: Fleiss' Kappa coefficient
    """
    N, k = rating_matrix.shape  # N samples, k categories
    n = rating_matrix.sum(axis=1)  # number of annotators per sample
    
    # Check if all samples have the same number of annotators
    if not np.all(n == n[0]):
        # Handle variable number of annotators (more complex case)
        n_avg = np.mean(n)
    else:
        n_avg = n[0]
    
    # If only one annotator or one category, return appropriate values
    if n_avg <= 1:
        return 1.0
    
    if k <= 1:
        return 1.0
    
    # Calculate observed agreement for each sample
    P_i = []
    for i in range(N):
        if n[i] <= 1:
            P_i.append(1.0)  # Perfect agreement for single annotator
        else:
            # Sum of r_ij * (r_ij - 1) for all categories j
            numerator = np.sum(rating_matrix[i] * (rating_matrix[i] - 1))
            denominator = n[i] * (n[i] - 1)
            P_i.append(numerator / denominator)
    
    P_bar = np.mean(P_i)  # Average observed agreement
    
    # Calculate expected agreement
    # Proportion of all assignments in each category
    total_assignments = np.sum(rating_matrix)
    if total_assignments == 0:
        return 1.0
    
    p_j = np.sum(rating_matrix, axis=0) / total_assignments
    P_e = np.sum(p_j ** 2)  # Expected agreement by chance
    
    # Handle edge cases
    if abs(P_e - 1.0) < 1e-10:  # Perfect expected agreement (all in one category)
        # If observed agreement is also perfect, return 1.0
        if P_bar > 0.99:
            return 1.0
        else:
            # Some disagreement despite perfect chance agreement
            return 0.0
    
    if P_e >= 1.0:  # Should not happen, but handle gracefully
        return 0.0
    
    # Calculate Fleiss' Kappa
    kappa = (P_bar - P_e) / (1.0 - P_e)
    
    # Bound the result to reasonable range
    kappa = np.clip(kappa, -1.0, 1.0)
    
    return kappa


def evaluate_distributions(
    predictions: List[Dict[str, float]],
    references: List[Dict[str, float]],
    emotions: List[str]
) -> Dict[str, Any]:
    """Evaluate predicted emotion distributions against reference distributions.

    Args:
        predictions: List of predicted emotion distributions
        references: List of reference emotion distributions
        emotions: List of all emotions to include

    Returns:
        Dictionary with all evaluation metrics
    """
    
    # Convert dictionary distributions to vector form
    pred_vectors = [convert_to_vector_representation(d, emotions) for d in predictions]
    ref_vectors = [convert_to_vector_representation(d, emotions) for d in references]

    # Calculate distribution metrics (sample-wise)
    js_divergences = [jensen_shannon_divergence(p, r) for p, r in zip(pred_vectors, ref_vectors)]
    bc_coefficients = [bhattacharyya_coefficient(p, r) for p, r in zip(pred_vectors, ref_vectors)]
    
    # Calculate entropy for both predictions and references
    pred_entropies = calculate_distribution_entropy(pred_vectors)
    ref_entropies = calculate_distribution_entropy(ref_vectors)

    # Calculate calibration metrics
    ece, conf_bins, acc_bins = expected_calibration_error(pred_vectors, ref_vectors)

    # Calculate classification metrics
    classification_metrics = compute_classification_metrics(pred_vectors, ref_vectors, emotions)

    # Calculate R-squared values
    r2_values = calculate_r_squared(pred_vectors, ref_vectors, emotions)

    # Compile all metrics
    results = {
        "distribution_metrics": {
            "jensen_shannon_divergence": {
                "mean": float(np.mean(js_divergences)),
                "std": float(np.std(js_divergences)),
                "median": float(np.median(js_divergences)),
                "min": float(np.min(js_divergences)),
                "max": float(np.max(js_divergences))
            },
            "bhattacharyya_coefficient": {
                "mean": float(np.mean(bc_coefficients)),
                "std": float(np.std(bc_coefficients)),
                "median": float(np.median(bc_coefficients)),
                "min": float(np.min(bc_coefficients)),
                "max": float(np.max(bc_coefficients))
            },
            "entropy": {
                "prediction_mean": float(np.mean(pred_entropies)),
                "prediction_std": float(np.std(pred_entropies)),
                "reference_mean": float(np.mean(ref_entropies)),
                "reference_std": float(np.std(ref_entropies))
            },
            "r_squared": r2_values
        },
        "calibration_metrics": {
            "expected_calibration_error": ece,
            "confidence_bins": conf_bins,
            "accuracy_bins": acc_bins
        },
        "classification_metrics": classification_metrics,
        "sample_count": len(predictions),
        "js_divergences": js_divergences,  # Store individual values for distribution analysis
        "bc_coefficients": bc_coefficients,  # Store individual values for distribution analysis
        "pred_entropies": pred_entropies,  # Store individual entropy values
        "ref_entropies": ref_entropies  # Store individual entropy values
    }

    return results
  

def evaluate_model(model_name, matched_models, matched_reference, emotions, plot=True, raw_annotations=None, model_raw_annotations=None):
    """Evaluate a model's annotation quality against reference annotations.
    
    Args:
        model_name: Name of the model to evaluate
        matched_models: Dictionary of matched model annotations
        matched_reference: List of matched reference annotations
        emotions: List of emotion labels
        plot: Whether to generate plots
        raw_annotations: Optional raw annotation data for human annotations (in human raw format)
        model_raw_annotations: Optional dict with raw annotations for all models
    
    Returns:
        Dictionary containing evaluation results
    """
    if model_name not in matched_models or not matched_models[model_name]:
        print(f"No matched {model_name} annotation data available for evaluation")
        return None
    else:
        print(f"\nEvaluating {model_name} annotation quality against human annotations...")

    
    # Extract emotion distributions
    model_emotions = [item['emotion'] for item in matched_models[model_name]]
    reference_emotions = [item['emotion'] for item in matched_reference]
    
    # Evaluate distributions using evaluation module
    results = evaluate_distributions(model_emotions, reference_emotions, emotions)
    
    # Get entropy values
    entropy_mean = results["distribution_metrics"]["entropy"]["prediction_mean"]
    entropy_std = results["distribution_metrics"]["entropy"]["prediction_std"]
    
    # Calculate Fleiss' Kappa for model if raw annotations are available
    if model_raw_annotations and model_name in model_raw_annotations:
        # Extract only the samples that are in matched_models
        matched_ids = [item['id'] for item in matched_models[model_name]]
        model_matched_raw = {}
        
        # Get raw annotations for this model
        model_specific_raw = model_raw_annotations[model_name]
        
        # Filter to keep only matched samples
        for sample_id in matched_ids:
            if sample_id in model_specific_raw:
                model_matched_raw[sample_id] = model_specific_raw[sample_id]
        
        # Calculate Fleiss' Kappa
        if model_matched_raw:
            kappa_result = calculate_inter_annotator_agreement(model_matched_raw, emotions, "llm")
            fleiss_kappa = kappa_result.get("fleiss_kappa", 0.0)
            
            # Add to results
            results["fleiss_kappa"] = fleiss_kappa
    
    # Display key metrics
    print("\nKey Metrics:")
    print(f"Jensen-Shannon Divergence: {results['distribution_metrics']['jensen_shannon_divergence']['mean']:.4f} (lower is better)")
    print(f"Bhattacharyya Coefficient: {results['distribution_metrics']['bhattacharyya_coefficient']['mean']:.4f} (higher is better)")
    print(f"R-squared: {results['distribution_metrics']['r_squared']['average']:.4f} (higher is better)")
    print(f"Expected Calibration Error: {results['calibration_metrics']['expected_calibration_error']:.4f} (lower is better)")
    print(f"Accuracy: {results['classification_metrics']['accuracy']:.4f}")
    print(f"Macro F1-score: {results['classification_metrics']['macro_f1']:.4f}")
    print(f"Mean Entropy: {entropy_mean:.4f} ± {entropy_std:.4f}")
    if "fleiss_kappa" in results:
        print(f"Fleiss' Kappa: {results['fleiss_kappa']:.4f}")
    
    
    return results


def evaluate_model_predictions(reference_path, prediction_path, output_path=None):
    """Evaluate model predictions against reference annotations.

    Args:
        reference_path: Path to the reference annotation file
        prediction_path: Path to the model prediction file
        output_path: Optional path to save the evaluation results as JSON

    Returns:
        Dictionary with evaluation results
    """
    # Detect which dataset we're working with
    dataset = detect_dataset_from_file(reference_path)
    emotions = DATASET_EMOTIONS[dataset]
    print(f"Evaluating {dataset.upper()} dataset with emotion labels: {emotions}\n")

    # Load distributions
    reference_data = load_distribution_file(reference_path)
    prediction_data = load_distribution_file(prediction_path)

    model_name = os.path.splitext(os.path.basename(prediction_path))[0]
    print(f"Loaded reference file: {reference_path}")
    print(f"   - Contains {len(reference_data)} samples")
    print(f"Loaded prediction file: {prediction_path}")
    print(f"   - Contains {len(prediction_data)} samples\n")

    # Match samples by ID
    reference_ids = {item["id"]: item for item in reference_data}
    prediction_ids = {item["id"]: item for item in prediction_data}

    matched_references = []
    matched_predictions = []
    matched_ids = []

    for item_id in reference_ids:
        if item_id in prediction_ids:
            matched_ids.append(item_id)
            matched_references.append(reference_ids[item_id]["emotion"])
            matched_predictions.append(prediction_ids[item_id]["emotion"])

    print(f"Matched {len(matched_predictions)} samples between reference and prediction files")

    # Evaluate if there are matched samples
    if matched_references and matched_predictions:
        results = evaluate_distributions(matched_predictions, matched_references, emotions)
        results["matched_samples"] = len(matched_predictions)
        results["total_reference_samples"] = len(reference_data)
        results["total_prediction_samples"] = len(prediction_data)
        results["matched_ids"] = matched_ids

        # Display key metrics
        js_mean = results["distribution_metrics"]["jensen_shannon_divergence"]["mean"]
        bc_mean = results["distribution_metrics"]["bhattacharyya_coefficient"]["mean"]
        r2_avg = results["distribution_metrics"]["r_squared"]["average"]
        ece = results["calibration_metrics"]["expected_calibration_error"]
        acc = results["classification_metrics"]["accuracy"]
        f1 = results["classification_metrics"]["macro_f1"]
        entropy_mean = results["distribution_metrics"]["entropy"]["prediction_mean"]
        entropy_std = results["distribution_metrics"]["entropy"]["prediction_std"]

        print(f"\nKey Metrics:")
        print(f"   - Jensen-Shannon Divergence: {js_mean:.4f} (lower is better)")
        print(f"   - Bhattacharyya Coefficient: {bc_mean:.4f} (higher is better)")
        print(f"   - R-squared: {r2_avg:.4f} (higher is better)")
        print(f"   - Expected Calibration Error: {ece:.4f} (lower is better)")
        print(f"   - Accuracy: {acc:.4f}")
        print(f"   - Macro F1-score: {f1:.4f}")
        print(f"   - Mean Entropy: {entropy_mean:.4f} ± {entropy_std:.4f}")

        # Save results to file if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)

            print(f"\nEvaluation results saved to {output_path}")

        return results
    else:
        print("No matching samples found between reference and prediction files")
        return None