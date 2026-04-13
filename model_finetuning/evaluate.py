#!/usr/bin/env python
# coding: utf-8

"""
Scaling Ambiguity Augmenting Human Annotation in Speech Emotion Recognition with Audio-Language Models - Finetuned Model Evaluation
============
Finetuned Model evaluation script for fine-tuned Qwen2-Audio models.

Loads a fine-tuned checkpoint (LoRA + EmotionClassifierHead), runs inference
on the test set, and reports Jensen-Shannon Divergence and Bhattacharyya Coefficient.

Mirrors the six experimental configurations from train.py:

    python evaluate.py --dataset iemocap --annotation_source human
    python evaluate.py --dataset iemocap --annotation_source synthetic
    python evaluate.py --dataset iemocap --annotation_source combined
    python evaluate.py --dataset msp     --annotation_source human
    python evaluate.py --dataset msp     --annotation_source synthetic
    python evaluate.py --dataset msp     --annotation_source combined

Optional flags:
    --checkpoint    Specific checkpoint name under the model dir (default: latest)
"""

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

import argparse

parser = argparse.ArgumentParser(description="Evaluate a fine-tuned Qwen2-Audio AER model")
parser.add_argument(
    "--dataset",
    choices=["iemocap", "msp"],
    required=True,
    help="Dataset to evaluate on"
)
parser.add_argument(
    "--annotation_source",
    choices=["human", "synthetic", "combined"],
    required=True,
    help="Annotation source the model was trained on"
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Checkpoint subfolder under the model directory (e.g. 'checkpoint-1150'). "
         "If not specified, the model directory root is used."
)
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

import sys
sys.path.append('..')

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa

from tqdm import tqdm
from safetensors.torch import load_file
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from peft import PeftModel

from lib import load_data as ld
from lib import evaluation_lib as ev


# ---------------------------------------------------------------------------
# Dataset-specific configuration (mirrors train.py)
# ---------------------------------------------------------------------------

EMOTION_CLASSES = {
    "iemocap": ["Anger", "Happiness", "Neutral state", "Sadness"],
    "msp":     ["Angry", "Happy", "Neutral", "Sad"],
}

# Test dataset name used in ld.load_test_distributions per annotation source
TEST_DATASET_NAMES = {
    ("iemocap", "human"):     "iemocap",
    ("iemocap", "synthetic"): "gemini-iemocap",
    ("iemocap", "combined"):  "combined-iemocap",
    ("msp",     "human"):     "msp",
    ("msp",     "synthetic"): "gemini-msp",
    ("msp",     "combined"):  "combined-msp",
}

# Fine-tuned model directories (must match OUTPUT_DIRS in train.py)
MODEL_DIRS = {
    ("iemocap", "human"):     "./model_finetuning/finetuned_models/iemocap_models/human_iemocap_DiME",
    ("iemocap", "synthetic"): "./model_finetuning/finetuned_models/iemocap_models/llm_iemocap_DiME",
    ("iemocap", "combined"):  "./model_finetuning/finetuned_models/iemocap_models/combined_iemocap_DiME",
    ("msp",     "human"):     "./model_finetuning/finetuned_models/msp_models/human_msp_DiME",
    ("msp",     "synthetic"): "./model_finetuning/finetuned_models/msp_models/llm_msp_DiME",
    ("msp",     "combined"):  "./model_finetuning/finetuned_models/msp_models/combined_msp_DiME",
}

BASE_MODEL_PATH = "./model_finetuning/models/Qwen2-Audio-7B-Instruct"

dataset           = args.dataset
annotation_source = args.annotation_source
emotion_classes   = EMOTION_CLASSES[dataset]
test_dataset_name = TEST_DATASET_NAMES[(dataset, annotation_source)]
model_dir         = MODEL_DIRS[(dataset, annotation_source)]
finetuned_model_path = os.path.join(model_dir, args.checkpoint) if args.checkpoint else model_dir

print(f"\n{'='*60}")
print(f"  Dataset:           {dataset.upper()}")
print(f"  Annotation source: {annotation_source}")
print(f"  Test dataset:      {test_dataset_name}")
print(f"  Model path:        {finetuned_model_path}")
print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Set all seeds to {seed}")

set_all_seeds(42)


# ===========================================================================
# Section 1: Model Architecture
# (Must match EmotionClassifierHead in train.py exactly for correct loading)
# ===========================================================================

class EmotionClassifierHead(nn.Module):
    """
    Distributional prediction head attached on top of the Qwen2-Audio backbone.
    Must match the architecture used during fine-tuning in train.py exactly.
    """

    def __init__(self, hidden_size, num_emotions):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, 8)

        # First residual block
        self.norm1    = nn.LayerNorm(hidden_size)
        self.dense1   = nn.Linear(hidden_size, hidden_size)
        self.dropout1 = nn.Dropout(0.15)
        self.act      = nn.GELU()

        # Projection block with dimensionality reduction
        self.norm2    = nn.LayerNorm(hidden_size)
        self.dense2   = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout2 = nn.Dropout(0.1)

        # Output
        self.norm3  = nn.LayerNorm(hidden_size // 2)
        self.output = nn.Linear(hidden_size // 2, num_emotions)

    def forward(self, hidden_states):
        # Self-attention on sequence
        attn_hidden = hidden_states.transpose(0, 1)
        attn_output, _ = self.attention(attn_hidden, attn_hidden, attn_hidden)
        attn_output = attn_output.transpose(0, 1)
        
        # Extract first token representation
        first_token = attn_output[:, 0]
        
        # First residual block
        residual = first_token
        x = self.norm1(first_token)
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.act(x)
        x = x + residual
        
        # Second residual block with projection
        residual = x
        x = self.norm2(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        x = self.act(x)
        
        # Final layer norm and output
        x = self.norm3(x)
        return self.output(x)


# ===========================================================================
# Section 2: Model Loading
# ===========================================================================

def load_finetuned_model(base_model_path, finetuned_model_path, emotion_list):
    """
    Load the fine-tuned Qwen2-Audio model with LoRA adapters and
    EmotionClassifierHead weights from a safetensors checkpoint.

    The loading sequence is:
      1. Load base Qwen2-Audio model
      2. Attach a fresh EmotionClassifierHead with correct output dim
      3. Load LoRA adapters via PeftModel
      4. Restore EmotionClassifierHead weights from adapter_model.safetensors

    Parameters
    ----------
    base_model_path : str
        Path to the original Qwen2-Audio-7B-Instruct weights.
    finetuned_model_path : str
        Path to the fine-tuned checkpoint directory.
    emotion_list : list of str
        Emotion categories — determines the output dimension of the head.

    Returns
    -------
    (model, processor)
    """
    print(f"Loading processor from {base_model_path}...")
    processor = AutoProcessor.from_pretrained(base_model_path)

    print(f"Loading base model from {base_model_path}...")
    base_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    device = next(base_model.parameters()).device
    print(f"Model device: {device}")

    # Attach emotion head with the correct number of output classes
    base_model.emotion_head = EmotionClassifierHead(
        hidden_size=base_model.language_model.config.hidden_size,
        num_emotions=len(emotion_list)
    ).to(device=device, dtype=torch.float16)

    print(f"Loading LoRA adapters from {finetuned_model_path}...")
    model = PeftModel.from_pretrained(base_model, finetuned_model_path)

    # Restore EmotionClassifierHead weights from the safetensors checkpoint
    safetensors_path = os.path.join(finetuned_model_path, "adapter_model.safetensors")
    if os.path.exists(safetensors_path):
        print("Restoring EmotionClassifierHead weights from safetensors...")
        state_dict = load_file(safetensors_path, device=str(device))

        # Filter to only emotion_head parameters and strip the key prefix
        prefix = "base_model.model.emotion_head."
        emotion_head_state = {
            k[len(prefix):]: v.to(dtype=torch.float16)
            for k, v in state_dict.items()
            if k.startswith(prefix)
        }

        if emotion_head_state:
            target_head = (
                model.emotion_head.original_module
                if hasattr(model.emotion_head, "original_module")
                else model.emotion_head
            )
            missing, unexpected = target_head.load_state_dict(emotion_head_state, strict=False)
            if missing:
                print(f"Warning: missing keys in emotion_head: {missing[:5]}")
            if unexpected:
                print(f"Warning: unexpected keys in emotion_head: {unexpected[:5]}")
            print("EmotionClassifierHead weights loaded successfully")
        else:
            print("Warning: no emotion_head parameters found in safetensors checkpoint")
    else:
        print(f"Warning: safetensors not found at {safetensors_path}")

    # Patch base_model.forward to strip Seq2Seq decoder arguments
    # (same patch applied in train.py during training)
    old_base_forward = model.base_model.forward

    def new_base_forward(*args, **kwargs):
        kwargs.pop("decoder_input_ids",      None)
        kwargs.pop("decoder_attention_mask", None)
        kwargs.pop("decoder_inputs_embeds",  None)

        if ("labels" in kwargs and kwargs["labels"] is not None
                and "attention_mask" in kwargs
                and kwargs["labels"].shape[1] != kwargs["attention_mask"].shape[1]):
            min_len = min(kwargs["labels"].shape[1], kwargs["attention_mask"].shape[1])
            kwargs["labels"]         = kwargs["labels"][:, :min_len]
            kwargs["attention_mask"] = kwargs["attention_mask"][:, :min_len]

        return old_base_forward(*args, **kwargs)

    model.base_model.forward = new_base_forward
    model.eval()

    return model, processor


# ===========================================================================
# Section 3: Inference
# ===========================================================================

def generate_predictions(model, processor, test_data, emotion_list):
    """
    Run inference on a test set and return predicted emotion distributions.

    Parameters
    ----------
    model : PeftModel
        Fine-tuned model with EmotionClassifierHead attached.
    processor : AutoProcessor
    test_data : list of dict
        Test samples with 'id', 'groundtruth', 'speaker', 'audio' fields.
    emotion_list : list of str
        Ordered emotion categories for the dataset.

    Returns
    -------
    list of dict
        Each entry has 'id', 'emotion' (predicted dist), and metadata fields.
    """
    device      = next(model.parameters()).device
    predictions = []
    emotions_formatted = ", ".join(emotion_list)

    print(f"Generating predictions for {len(test_data)} samples...")

    for sample in tqdm(test_data):
        sample_id  = sample['id']
        audio_path = ld.get_audio_path(sample, dataset)

        if not os.path.exists(audio_path):
            print(f"Warning: audio not found for {sample_id}: {audio_path}")
            continue

        try:
            audio, _ = librosa.load(audio_path, sr=16000)

            # Truncate to 30 seconds maximum
            if len(audio) > 30 * 16000:
                audio = audio[:30 * 16000]

            # Parse utterance transcript from groundtruth field
            groundtruth = sample.get('groundtruth', None)
            if isinstance(groundtruth, list):
                if groundtruth:
                    raw_text = groundtruth[0]
                    if isinstance(raw_text, str) and raw_text.startswith('[') and raw_text.endswith(']'):
                        try:
                            import ast
                            parsed   = ast.literal_eval(raw_text)
                            utterance = parsed[0] if isinstance(parsed, list) and parsed else raw_text
                        except Exception:
                            utterance = raw_text
                    else:
                        utterance = raw_text
                else:
                    utterance = None
            else:
                utterance = groundtruth

            if utterance is None:
                transcript_section  = ""
                primary_instruction = "This audio has no utterance. Rely solely on audio cues for analysis."
            else:
                transcript_section  = f'"{utterance.strip()}"'
                primary_instruction = "Carefully analyze the emotional content in both the audio and the text."

            prompt = f"""Listen to the audio and analyze this speech utterance. Provide a probability distribution across all emotions in the VALID EMOTIONS LIST.

Utterance: {transcript_section}
Speaker: {sample['speaker']}

VALID EMOTIONS LIST: {emotions_formatted}

Output MUST satisfy the following rules:
Rule 1: {primary_instruction}
Rule 2: Pay attention to vocal tone, pitch, intensity, and speaking rate in the audio.
Rule 3: Consider that emotions can be ambiguous and multiple emotions may be present simultaneously.
Rule 4: ONLY use emotions from the VALID EMOTIONS LIST.
Rule 5: Ensure the probabilities sum to 1.0.

Provide your answer as a JSON dictionary with the EXACT emotion labels from {emotions_formatted} as keys and probabilities as values.

For example: 
{{
"{emotion_list[0]}": 0.6, 
"{emotion_list[1]}": 0.2, 
"{emotion_list[3]}": 0.2, 
}}

IMPORTANT: Output ONLY the JSON dictionary and NO OTHER TEXT."""

            conversation = [
                {"role": "system", "content": "You are an expert in emotion analysis who provides accurate emotion judgment."},
                {"role": "user",   "content": [
                    {"type": "text",  "text": prompt},
                    {"type": "audio", "audio": audio},
                ]}
            ]

            text   = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            inputs = processor(text=text, audio=[audio], return_tensors="pt", sampling_rate=16000)
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            with torch.no_grad():
                outputs       = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]

                # Align dtype with emotion head (fp16 model may return fp32 hidden states)
                head_dtype = next(model.emotion_head.parameters()).dtype
                if hidden_states.dtype != head_dtype:
                    hidden_states = hidden_states.to(head_dtype)

                emotion_logits = model.emotion_head(hidden_states)
                emotion_probs  = F.softmax(emotion_logits, dim=-1)[0]

            pred_dist = {e: float(emotion_probs[i].cpu()) for i, e in enumerate(emotion_list)}

            predictions.append({
                'id':          sample_id,
                'emotion':     pred_dist,
                'groundtruth': sample.get('groundtruth', ''),
                'speaker':     sample.get('speaker', ''),
                'audio':       audio_path,
            })

        except Exception as e:
            print(f"Error processing {sample_id}: {e}")

    print(f"Generated predictions for {len(predictions)}/{len(test_data)} samples")
    return predictions


# ===========================================================================
# Section 4: Evaluation
# ===========================================================================

def evaluate_predictions(predictions, test_data, emotion_list, output_dir=None):
    """
    Compare predicted distributions against ground-truth and report metrics.

    Parameters
    ----------
    predictions : list of dict
        Output of generate_predictions().
    test_data : list of dict
        Ground-truth test samples with 'id' and 'emotion' fields.
    emotion_list : list of str
        Ordered emotion categories.
    output_dir : str or None
        If set, predictions JSON and metrics JSON are saved here.

    Returns
    -------
    dict
        Full evaluation results from ev.evaluate_distributions().
    """
    pred_dict  = {s['id']: s for s in predictions}
    truth_dict = {s['id']: s for s in test_data}

    matched_ids    = set(pred_dict.keys()) & set(truth_dict.keys())
    matched_preds  = [pred_dict[i]['emotion']  for i in matched_ids]
    matched_truths = [truth_dict[i]['emotion'] for i in matched_ids]

    print(f"Matched {len(matched_ids)} samples for evaluation")

    results = ev.evaluate_distributions(matched_preds, matched_truths, emotion_list)

    print("\nEvaluation Results:")
    print(f"  Jensen-Shannon Div:       {results['distribution_metrics']['jensen_shannon_divergence']['mean']:.4f}  (↓ better)")
    print(f"  Bhattacharyya Coef:       {results['distribution_metrics']['bhattacharyya_coefficient']['mean']:.4f}  (↑ better)")
    print(f"  Accuracy:                 {results['classification_metrics']['accuracy']:.4f}")
    print(f"  Macro F1:                 {results['classification_metrics']['macro_f1']:.4f}")
    print(f"  Prediction Entropy:       {results['distribution_metrics']['entropy']['prediction_mean']:.4f} "
          f"± {results['distribution_metrics']['entropy']['prediction_std']:.4f}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        pred_path = os.path.join(output_dir, f"{test_dataset_name}_predictions.json")
        with open(pred_path, 'w') as f:
            json.dump(predictions, f, indent=2)

        metrics_path = os.path.join(output_dir, f"{test_dataset_name}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nPredictions saved to: {pred_path}")
        print(f"Metrics saved to:     {metrics_path}")

    return results


# ===========================================================================
# Section 5: Main
# ===========================================================================

# Load model
model, processor = load_finetuned_model(BASE_MODEL_PATH, finetuned_model_path, emotion_classes)

# Load test data
test_data  = ld.load_test_distributions(test_dataset_name)

# Output directory mirrors the model directory structure
output_dir = os.path.join(
    "./model_finetuning",
    f"evaluation_results_{dataset}_{annotation_source}"
)

# Run inference
predictions = generate_predictions(
    model=model,
    processor=processor,
    test_data=test_data,
    emotion_list=emotion_classes,
)

# Evaluate and save results
results = evaluate_predictions(
    predictions=predictions,
    test_data=test_data,
    emotion_list=emotion_classes,
    output_dir=output_dir,
)

print("\nEvaluation complete!")
