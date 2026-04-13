#!/usr/bin/env python
# coding: utf-8

"""
Scaling Ambiguity Augmenting Human Annotation in Speech Emotion Recognition with Audio-Language Models - Model Fine-tuning
============
Unified fine-tuning script for Ambiguous Emotion Recognition (AER).

Supports all six experimental configurations from the paper by specifying
--dataset and --annotation_source at the command line:

    python train.py --dataset iemocap --annotation_source human
    python train.py --dataset iemocap --annotation_source synthetic
    python train.py --dataset iemocap --annotation_source combined
    python train.py --dataset msp     --annotation_source human
    python train.py --dataset msp     --annotation_source synthetic
    python train.py --dataset msp     --annotation_source combined

Pipeline
--------
1. Load raw annotations (human / synthetic / combined) for the chosen dataset
2. Validate audio file existence
3. Build training examples (prompt + target distribution)
4. Split into train / val BEFORE applying DiME (keeps val set pure)
5. Apply DiME-Aug to training set only
6. Fine-tune Qwen2-Audio with LoRA + distributional head using Jensen-Shannon Divergence loss
"""

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

import argparse

parser = argparse.ArgumentParser(description="Fine-tune Qwen2-Audio for AER")
parser.add_argument(
    "--dataset",
    choices=["iemocap", "msp"],
    required=True,
    help="Dataset to train on: 'iemocap' or 'msp'"
)
parser.add_argument(
    "--annotation_source",
    choices=["human", "synthetic", "combined"],
    required=True,
    help="Annotation source: 'human', 'synthetic', or 'combined'"
)
parser.add_argument(
    "--batch_size",  type=int, default=8,
    help="Per-device training batch size (default: 8)"
)
parser.add_argument(
    "--epochs",      type=int, default=60,
    help="Maximum training epochs (default: 60)"
)
parser.add_argument(
    "--dime_ratio",  type=float, default=0.30,
    help="DiME augmentation target ratio relative to majority class (default: 0.30)"
)
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

import os
import sys
sys.path.append('..')

import json
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import librosa
import soundfile as sf

from tqdm import tqdm
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from transformers import (
    AutoProcessor,
    Qwen2AudioForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from transformers.trainer_utils import set_seed
from torch.utils.data import Dataset, WeightedRandomSampler

from lib import load_data as ld

mp.set_start_method('spawn', force=True)


# ---------------------------------------------------------------------------
# Dataset-specific configuration
# ---------------------------------------------------------------------------

# Annotation file paths for each (dataset, annotation_source) combination
ANNOTATION_PATHS = {
    ("iemocap", "human"):     "processed_data/human_iemocap_train_raw_annotations.json",
    ("iemocap", "synthetic"): "processed_data/gemini_annotations/gemini_iemocap_train_raw_annotations.json",
    ("iemocap", "combined"):  "processed_data/combined_annotations/combined_iemocap_train_raw_annotations.json",
    ("msp",     "human"):     "processed_data/human_msp_train_raw_annotations.json",
    ("msp",     "synthetic"): "processed_data/gemini_annotations/gemini_msp_train_raw_annotations.json",
    ("msp",     "combined"):  "processed_data/combined_annotations/combined_msp_train_raw_annotations.json",
}

# Output directories for each (dataset, annotation_source) combination
OUTPUT_DIRS = {
    ("iemocap", "human"):     "model_finetuning/finetuned_models/iemocap_models/human_iemocap_DiME",
    ("iemocap", "synthetic"): "model_finetuning/finetuned_models/iemocap_models/llm_iemocap_DiME",
    ("iemocap", "combined"):  "model_finetuning/finetuned_models/iemocap_models/combined_iemocap_DiME",
    ("msp",     "human"):     "model_finetuning/finetuned_models/msp_models/human_msp_DiME",
    ("msp",     "synthetic"): "model_finetuning/finetuned_models/msp_models/llm_msp_DiME",
    ("msp",     "combined"):  "model_finetuning/finetuned_models/msp_models/combined_msp_DiME",
}

# Emotion label sets per dataset
EMOTION_CLASSES = {
    "iemocap": ["Anger", "Happiness", "Neutral state", "Sadness"],
    "msp":     ["Angry", "Happy", "Neutral", "Sad"],
}

# Weighted sampler config per dataset.
# Multipliers are applied as: weight = multiplier / (class_freq + eps)
# Thresholds define when to switch from light to strong weighting.
# IEMOCAP: Happiness is over-represented, Anger is under-represented.
# MSP:     Neutral is over-represented, Angry/Sad are under-represented.
SAMPLER_CONFIG = {
    "iemocap": {
        "moderate_threshold": 4.5,
        "moderate_multipliers": {
            "Anger": 1.3, "Happiness": 0.8, "Neutral state": 1.0, "Sadness": 1.1
        },
        "strong_multipliers": {
            "Anger": 1.4, "Happiness": 0.6, "Neutral state": 0.9, "Sadness": 1.2
        },
    },
    "msp": {
        "moderate_threshold": 4.0,
        "moderate_multipliers": {
            "Angry": 1.2, "Happy": 1.0, "Neutral": 0.8, "Sad": 1.2
        },
        "strong_multipliers": {
            "Angry": 1.5, "Happy": 0.9, "Neutral": 0.5, "Sad": 1.5
        },
    },
}

dataset           = args.dataset
annotation_source = args.annotation_source

annotation_path     = ANNOTATION_PATHS[(dataset, annotation_source)]
finetune_output_dir = OUTPUT_DIRS[(dataset, annotation_source)]
emotion_classes = EMOTION_CLASSES[dataset]

print(f"\n{'='*60}")
print(f"  Dataset:           {dataset.upper()}")
print(f"  Annotation source: {annotation_source}")
print(f"  Emotion classes:   {emotion_classes}")
print(f"  Output dir:        {finetune_output_dir}")
print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

local_model_path = "./model_finetuning/models/Qwen2-Audio-7B-Instruct"

if not os.path.exists(local_model_path):
    print(f"ERROR: Pre-downloaded model not found at {local_model_path}")
    print("Please download the model first.")
    sys.exit(1)

# Uncomment if you need to download the model
# model_id = "Qwen/Qwen2-Audio-7B-Instruct"
# download_dir = "./models/Qwen2-Audio-7B-Instruct"
# os.makedirs(download_dir, exist_ok=True)

# print(f"Start downloading model {model_id} to {download_dir}")

# # Download the model
# local_model_path = snapshot_download(
# repo_id=model_id,
# local_dir=download_dir,
# token=os.environ.get("HUGGINGFACE_TOKEN")
# )

# print(f"Model has been downloaded to: {local_model_path}")

print("Loading processor...")
processor = AutoProcessor.from_pretrained(local_model_path)
print("Processor loading complete!")

print("Loading model...")
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    local_model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("Model loading complete!")


# ===========================================================================
# Section 1: Data Preparation
# ===========================================================================

# Load raw annotations for the selected (dataset, annotation_source)
with open(annotation_path, 'r') as f:
    raw_train_data = json.load(f)
print(f"Loaded {len(raw_train_data)} raw annotation samples from {annotation_path}")

dataset_emotions = ld.load_emotion_classes(dataset)
print(f"Emotion classes: {dataset_emotions}")

# Filter out samples with missing audio files
valid_samples = [
    sample for sample in tqdm(raw_train_data, desc="Checking audio files")
    if ld.audio_file_exists(sample, dataset)
]
print(f"Valid samples: {len(valid_samples)}/{len(raw_train_data)} "
      f"({len(valid_samples)/len(raw_train_data)*100:.1f}%)")


def create_training_example(sample, emotion_list, dataset_name):
    """
    Build a single training example dict from a raw annotation sample.

    Converts the raw emotion list to a probability distribution and formats
    the Qwen2-Audio prompt with utterance transcript and speaker metadata.

    Parameters
    ----------
    sample : dict
        Raw annotation sample with 'id', 'emotion', 'groundtruth',
        'speaker', and 'audio' fields.
    emotion_list : list of str
        Valid emotion categories for this dataset.
    dataset_name : str
        'iemocap' or 'msp' — used for audio path resolution.

    Returns
    -------
    dict
        {'id', 'prompt', 'target', 'audio_path', 'utterance'}
    """
    emotions_formatted = ", ".join(emotion_list)

    # Parse utterance transcript from the groundtruth field
    groundtruth = sample.get('groundtruth', None)
    if isinstance(groundtruth, list):
        if groundtruth:
            raw_text = groundtruth[0]
            # Handle nested list-as-string format
            if isinstance(raw_text, str) and raw_text.startswith('[') and raw_text.endswith(']'):
                try:
                    import ast
                    parsed = ast.literal_eval(raw_text)
                    utterance = parsed[0] if isinstance(parsed, list) and parsed else raw_text
                except Exception:
                    utterance = raw_text
            else:
                utterance = raw_text
        else:
            utterance = None
    else:
        utterance = groundtruth

    # Build transcript section and primary instruction based on availability
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

    # Convert emotion list to probability distribution if not already done
    if 'emotion' in sample and isinstance(sample['emotion'], list):
        emotion_counts = Counter(sample['emotion'])
        total          = len(sample['emotion'])
        distribution   = {e: emotion_counts.get(e, 0) / total for e in emotion_list}
        sample['emotion'] = distribution

    target     = json.dumps(sample['emotion'])
    audio_path = ld.get_audio_path(sample, dataset_name)

    return {
        "id":        sample["id"],
        "prompt":    prompt,
        "target":    target,
        "audio_path": audio_path,
        "utterance": transcript_section,
    }


training_examples = [
    create_training_example(s, dataset_emotions, dataset)
    for s in valid_samples
]
print(f"Created {len(training_examples)} training examples")


# ===========================================================================
# Section 2: DiME-Aug (Distribution-aware Multimodal Emotion Augmentation)
# ===========================================================================

class DiME:
    """
    DiME-Aug implementation for multimodal emotion data (audio + text).

    Generates synthetic minority-class samples by interpolating between
    k-nearest neighbors in the multimodal feature space, as described in
    Section 2.3 of the paper. Audio signals are linearly interpolated in the
    time domain; transcripts are selected from the dominant source (alpha < 0.5);
    emotion distributions are linearly interpolated and renormalized.
    """

    def __init__(self, k_neighbors=5, random_state=42, sampling_strategy='auto'):
        self.k_neighbors       = k_neighbors
        self.random_state      = random_state
        self.sampling_strategy = sampling_strategy
        np.random.seed(random_state)
        random.seed(random_state)

    def extract_audio_features(self, audio_path, sr=16000, n_mfcc=13, n_chroma=12, n_contrast=7):
        """
        Extract MFCC, chroma, spectral contrast, ZCR, rolloff, centroid,
        and RMS features from an audio file and return them as a flat vector.
        Returns a zero vector on failure.
        """
        try:
            audio, _ = librosa.load(audio_path, sr=sr, mono=True)

            # Pad very short clips to at least 0.5 s
            if len(audio) < sr * 0.5:
                audio = np.pad(audio, (0, int(sr * 0.5) - len(audio)), mode='constant')

            features = []

            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
            features.extend(np.mean(mfcc, axis=1))
            features.extend(np.std(mfcc, axis=1))

            chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=n_chroma)
            features.extend(np.mean(chroma, axis=1))
            features.extend(np.std(chroma, axis=1))

            contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_bands=n_contrast - 1)
            features.extend(np.mean(contrast, axis=1))
            features.extend(np.std(contrast, axis=1))

            for feat_fn in [
                librosa.feature.zero_crossing_rate,
                librosa.feature.spectral_rolloff,
                librosa.feature.spectral_centroid,
                librosa.feature.rms,
            ]:
                feat = feat_fn(y=audio) if feat_fn != librosa.feature.spectral_rolloff \
                       else feat_fn(y=audio, sr=sr)
                if feat_fn == librosa.feature.spectral_centroid:
                    feat = librosa.feature.spectral_centroid(y=audio, sr=sr)
                features.append(np.mean(feat))
                features.append(np.std(feat))

            return np.array(features)

        except Exception as e:
            print(f"Error extracting audio features from {audio_path}: {e}")
            expected_length = 2 * (n_mfcc + n_chroma + n_contrast) + 8
            return np.zeros(expected_length)

    def extract_text_features(self, text, max_length=100):
        """
        Extract lightweight lexical features from a transcript string.
        Returns a 10-dimensional feature vector.
        """
        if not text or text.strip() == "":
            return np.zeros(10)

        text = text.strip().lower()
        words  = text.split()
        vowels = set('aeiou')

        return np.array([
            len(text),
            len(words),
            text.count('!'),
            text.count('?'),
            text.count('.'),
            text.count(','),
            sum(c in vowels for c in text) / max(len(text), 1),
            sum(c.isalpha() and c not in vowels for c in text) / max(len(text), 1),
            np.mean([len(w) for w in words]) if words else 0,
            sum(c.isupper() for c in text) / max(len(text), 1),
        ])

    def get_dominant_emotion(self, emotion_dist):
        """Return the emotion label with highest probability in a distribution dict."""
        if isinstance(emotion_dist, dict):
            return max(emotion_dist, key=emotion_dist.get)
        return "Unknown"

    def analyze_class_distribution(self, examples):
        """Print and return dominant-emotion counts and ratios for a list of examples."""
        emotion_counts = {}
        for example in examples:
            target_json = example.get("target", "{}")
            try:
                target_dist = json.loads(target_json) if isinstance(target_json, str) else target_json
                dominant    = self.get_dominant_emotion(target_dist)
                emotion_counts[dominant] = emotion_counts.get(dominant, 0) + 1
            except Exception:
                continue

        total = sum(emotion_counts.values())
        ratios = {e: c / total for e, c in emotion_counts.items()}

        print("\nClass distribution:")
        for emotion, count in emotion_counts.items():
            print(f"  {emotion}: {count} ({ratios[emotion]*100:.1f}%)")

        return emotion_counts, ratios

    def determine_sampling_targets(self, emotion_counts):
        """
        Compute how many synthetic samples to generate per minority class
        based on the selected sampling_strategy.
        """
        if self.sampling_strategy in ('auto', 'minority'):
            max_count  = max(emotion_counts.values())
            min_target = int(max_count * args.dime_ratio)
            targets = {
                e: max(0, min_target - c)
                for e, c in emotion_counts.items()
            }
        elif isinstance(self.sampling_strategy, dict):
            targets = {
                e: max(0, self.sampling_strategy.get(e, 0) - emotion_counts.get(e, 0))
                for e in self.sampling_strategy
            }
        else:
            raise ValueError("sampling_strategy must be 'auto', 'minority', or a dict")

        print("\nDiME sampling targets:")
        for emotion, target in targets.items():
            if target > 0:
                print(f"  {emotion}: generate {target} new samples")
        return targets

    def find_minority_samples(self, examples):
        """Group examples by dominant emotion and return as a dict."""
        emotion_samples = {}
        for example in examples:
            target_json = example.get("target", "{}")
            try:
                target_dist = json.loads(target_json) if isinstance(target_json, str) else target_json
                dominant    = self.get_dominant_emotion(target_dist)
                emotion_samples.setdefault(dominant, []).append(example)
            except Exception:
                continue
        return emotion_samples

    def extract_multimodal_features(self, examples):
        """
        Build a feature matrix by concatenating audio + text features for
        each example. Returns the matrix and the indices of valid examples.
        """
        features, valid_indices = [], []
        print("Extracting multimodal features...")
        for i, example in enumerate(tqdm(examples)):
            try:
                audio_path = example.get("audio_path", "")
                if not os.path.exists(audio_path):
                    continue
                audio_feats = self.extract_audio_features(audio_path)
                text_feats  = self.extract_text_features(example.get("utterance", ""))
                features.append(np.concatenate([audio_feats, text_feats]))
                valid_indices.append(i)
            except Exception as e:
                print(f"Error processing example {i}: {e}")
        return np.array(features), valid_indices

    def create_synthetic_audio(self, sample1_path, sample2_path, alpha, temp_dir):
        """
        Linearly interpolate two audio waveforms and write the result to temp_dir.
        Falls back to sample1_path on error.
        """
        try:
            audio1, _ = librosa.load(sample1_path, sr=16000, mono=True)
            audio2, _ = librosa.load(sample2_path, sr=16000, mono=True)

            # Align lengths — use the shorter clip
            min_len = min(len(audio1), len(audio2))
            audio1, audio2 = audio1[:min_len], audio2[:min_len]

            synthetic_audio = audio1 + alpha * (audio2 - audio1)

            synthetic_path = os.path.join(
                temp_dir, f"synthetic_audio_{random.randint(10000, 99999)}.wav"
            )
            sf.write(synthetic_path, synthetic_audio, 16000)
            return synthetic_path

        except Exception as e:
            print(f"Error creating synthetic audio: {e}")
            return sample1_path  # Fallback to original

    def create_synthetic_text(self, text1, text2, alpha):
        """
        Select transcript based on alpha threshold (alpha < 0.5 → text1).
        Full text interpolation is linguistically incoherent, so selection
        is used instead, as described in Section 2.3 of the paper.
        """
        return text1 if alpha < 0.5 else text2

    def interpolate_emotion_distribution(self, dist1, dist2, alpha):
        """
        Linearly interpolate two emotion distribution dicts and renormalize.
        Pk = alpha * Pi + (1 - alpha) * Pj  (Equation 2 in the paper).
        """
        all_emotions = set(dist1) | set(dist2)
        interpolated = {
            e: dist1.get(e, 0.0) + alpha * (dist2.get(e, 0.0) - dist1.get(e, 0.0))
            for e in all_emotions
        }
        total = sum(interpolated.values())
        if total > 0:
            interpolated = {k: v / total for k, v in interpolated.items()}
        return interpolated

    def generate_DiME_samples(self, minority_samples, num_samples_needed, temp_dir):
        """
        Generate synthetic samples for a single minority emotion class using
        k-NN interpolation in the multimodal feature space.
        """
        if len(minority_samples) < 2:
            print(f"Warning: Only {len(minority_samples)} samples — cannot apply DiME")
            return []

        features, valid_indices = self.extract_multimodal_features(minority_samples)
        valid_samples = [minority_samples[i] for i in valid_indices]

        if len(valid_samples) < 2:
            print("Warning: fewer than 2 valid samples — cannot apply DiME")
            return []

        # Normalise features and fit k-NN
        features_norm = StandardScaler().fit_transform(features)
        k         = min(self.k_neighbors, len(valid_samples) - 1)
        nn_model  = NearestNeighbors(n_neighbors=k, metric='euclidean')
        nn_model.fit(features_norm)

        synthetic_samples = []

        for _ in range(num_samples_needed):
            sample_idx      = np.random.randint(0, len(valid_samples))
            sample          = valid_samples[sample_idx]
            neighbor_idx    = np.random.choice(
                nn_model.kneighbors([features_norm[sample_idx]], return_distance=False)[0][1:]
            )
            neighbor_sample = valid_samples[neighbor_idx]
            alpha           = np.random.random()

            try:
                synthetic_audio = self.create_synthetic_audio(
                    sample["audio_path"], neighbor_sample["audio_path"], alpha, temp_dir
                )
                synthetic_text = self.create_synthetic_text(
                    sample.get("utterance", ""), neighbor_sample.get("utterance", ""), alpha
                )

                sample_dist   = json.loads(sample["target"]) if isinstance(sample["target"], str) else sample["target"]
                neighbor_dist = json.loads(neighbor_sample["target"]) if isinstance(neighbor_sample["target"], str) else neighbor_sample["target"]
                synthetic_dist = self.interpolate_emotion_distribution(sample_dist, neighbor_dist, alpha)

                synthetic_samples.append({
                    "id":           f"DiME_synthetic_{len(synthetic_samples)}_{random.randint(1000, 9999)}",
                    "prompt":       sample["prompt"],
                    "target":       json.dumps(synthetic_dist),
                    "audio_path":   synthetic_audio,
                    "utterance":    synthetic_text,
                    "is_synthetic": True,
                })
            except Exception as e:
                print(f"Error generating synthetic sample: {e}")
                continue

        return synthetic_samples

    def fit_resample(self, examples, temp_dir=None):
        """
        Apply DiME-Aug to the training set, generating synthetic samples for
        minority emotion classes and returning the augmented example list.

        Parameters
        ----------
        examples : list of dict
            Training examples in prompt/target/audio_path format.
        temp_dir : str, optional
            Directory for synthetic audio files. Created automatically if None.

        Returns
        -------
        augmented_examples : list of dict
        temp_dir : str
        """
        if temp_dir is None:
            import time
            temp_dir = f"./DiME_synthetic_audio_{int(time.time())}"

        os.makedirs(temp_dir, exist_ok=True)
        print(f"DiME temp directory: {os.path.abspath(temp_dir)}")

        emotion_counts, _ = self.analyze_class_distribution(examples)
        sampling_targets  = self.determine_sampling_targets(emotion_counts)
        emotion_samples   = self.find_minority_samples(examples)

        all_synthetic = []
        for emotion, num_needed in sampling_targets.items():
            if num_needed > 0 and emotion in emotion_samples:
                print(f"\nGenerating {num_needed} synthetic samples for '{emotion}'...")
                generated = self.generate_DiME_samples(emotion_samples[emotion], num_needed, temp_dir)
                all_synthetic.extend(generated)
                print(f"Generated {len(generated)} samples for '{emotion}'")

        augmented = examples + all_synthetic

        print(f"\nDiME-Aug completed:")
        print(f"  Original:  {len(examples)}")
        print(f"  Synthetic: {len(all_synthetic)}")
        print(f"  Total:     {len(augmented)}")
        self.analyze_class_distribution(augmented)

        return augmented, temp_dir


# ---------------------------------------------------------------------------
# Train / validation split — must happen BEFORE DiME so val set stays pure
# ---------------------------------------------------------------------------

random.seed(42)
random.shuffle(training_examples)

val_size            = int(len(training_examples) * 0.1)
train_real          = training_examples[val_size:]
val_real            = training_examples[:val_size]

print(f"\nSplit: {len(train_real)} train / {len(val_real)} val (pure real, pre-DiME)")

# Apply DiME-Aug to training set only
dime_temp_dir = f"./{dataset}_{annotation_source}_DiME_audio"
dime = DiME(k_neighbors=3, sampling_strategy='auto', random_state=42)
train_augmented, dime_temp_dir = dime.fit_resample(train_real, temp_dir=dime_temp_dir)


# ===========================================================================
# Section 3: Loss Function
# ===========================================================================

class EmotionLoss(nn.Module):
    """
    Jensen-Shannon Divergence loss for emotion distribution prediction.

    Computes JSD between the predicted softmax distribution and the target
    distribution, with a small uniformity penalty to prevent the model from
    collapsing to uniform predictions.
    """

    def __init__(self, epsilon=1e-10):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits, target_dist):
        """
        Parameters
        ----------
        logits : Tensor [batch_size, num_emotions]
        target_dist : Tensor [batch_size, num_emotions]

        Returns
        -------
        Scalar loss tensor.
        """
        pred_dist = F.softmax(logits, dim=-1)

        # Add epsilon and renormalize for numerical stability
        pred_norm   = (pred_dist + self.epsilon)
        pred_norm   = pred_norm / pred_norm.sum(dim=-1, keepdim=True)
        target_norm = (target_dist + self.epsilon)
        target_norm = target_norm / target_norm.sum(dim=-1, keepdim=True)

        m = 0.5 * (pred_norm + target_norm)

        kl_p_m = torch.sum(pred_norm   * torch.log((pred_norm   / (m + self.epsilon)) + self.epsilon), dim=-1)
        kl_q_m = torch.sum(target_norm * torch.log((target_norm / (m + self.epsilon)) + self.epsilon), dim=-1)

        js_divergence = 0.5 * (kl_p_m + kl_q_m)

        # Penalty to discourage overly uniform predictions
        uniformity_penalty = -torch.sum(pred_norm * torch.log(pred_norm + self.epsilon), dim=-1).mean()

        return js_divergence.mean() + 0.08 * uniformity_penalty


# ===========================================================================
# Section 4: Emotion Classifier Head
# ===========================================================================

class EmotionClassifierHead(nn.Module):
    """
    Distributional prediction head attached on top of the Qwen2-Audio backbone.

    Architecture (as described in Section 2.4 / Figure 1 of the paper):
      Multi-head self-attention → first-token extraction →
      residual MLP block → projection → linear + softmax
    """

    def __init__(self, hidden_size, num_emotions):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, 8)

        # First residual block
        self.norm1    = nn.LayerNorm(hidden_size)
        self.dense1   = nn.Linear(hidden_size, hidden_size)
        self.dropout1 = nn.Dropout(0.15)
        self.act      = nn.GELU()

        # Projection block
        self.norm2    = nn.LayerNorm(hidden_size)
        self.dense2   = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout2 = nn.Dropout(0.1)

        # Output
        self.norm3  = nn.LayerNorm(hidden_size // 2)
        self.output = nn.Linear(hidden_size // 2, num_emotions)

    def forward(self, hidden_states):
        # Self-attention over the full sequence
        attn_hidden = hidden_states.transpose(0, 1)
        attn_output, _ = self.attention(attn_hidden, attn_hidden, attn_hidden)
        attn_output = attn_output.transpose(0, 1)
        
        # Use first token as the sequence representation
        first_token = attn_output[:, 0]

        # First residual block
        residual = first_token
        x = self.norm1(first_token)
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.act(x)
        x = x + residual  # Residual connection

        # Projection block (no residual — dimension changes)
        residual = x
        x = self.norm2(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        x = self.act(x)
        
        # Final layer norm and output
        x = self.norm3(x)
        return self.output(x)


# ===========================================================================
# Section 5: Dataset and Data Collator
# ===========================================================================

def augment_audio(audio, sr, augment_prob=0.3):
    """
    Apply random pitch shift or time stretch to an audio waveform.
    Augmentation is applied with probability augment_prob.
    """
    if random.random() > augment_prob:
        return audio

    aug_type = random.choice(['pitch', 'speed', 'none'])

    if aug_type == 'pitch':
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=random.uniform(-1.0, 1.0))
    elif aug_type == 'speed':
        return librosa.effects.time_stretch(audio, rate=random.uniform(0.95, 1.05))
    return audio


class EmotionDistributionDataset(Dataset):
    """
    PyTorch Dataset for emotion distribution prediction with Qwen2-Audio.

    Loads audio files, applies optional augmentation, formats the prompt,
    and returns tokenized inputs alongside target emotion distribution tensors.
    """

    def __init__(self, examples, processor, max_audio_length=30, training=False):
        self.examples         = examples
        self.processor        = processor
        self.max_audio_length = max_audio_length
        self.sample_rate      = 16000
        self.training         = training

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        try:
            audio_path = example["audio_path"]
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)

            if self.training:
                audio = augment_audio(audio, sr)

            # Truncate to maximum allowed length
            max_len = self.max_audio_length * self.sample_rate
            if len(audio) > max_len:
                audio = audio[:max_len]

            prompt     = example["prompt"]
            target_json = example["target"]

            try:
                target_dist = json.loads(target_json) if isinstance(target_json, str) else target_json
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Error parsing target_json for {example['id']}: {e}")
                target_dist = {}

            # Detect emotion label format from target distribution
            valid_emotions = emotion_classes
            is_iemocap     = True
            if isinstance(target_dist, dict):
                if not any(e in target_dist for e in valid_emotions):
                    if any(e in target_dist for e in ["Angry", "Happy", "Neutral", "Sad"]):
                        valid_emotions = ["Angry", "Happy", "Neutral", "Sad"]
                        is_iemocap     = False

            if not isinstance(target_dist, dict):
                print(f"Warning: target_dist for {example['id']} is not a dict: {target_dist}")
                target_dist = {valid_emotions[0]: 1.0}

            # Ensure all emotions are present and distribution is normalised
            for e in valid_emotions:
                target_dist.setdefault(e, 0.0)

            total = sum(target_dist.values())
            target_dist = (
                {k: v / total for k, v in target_dist.items()} if total > 0
                else {e: 1.0 / len(valid_emotions) for e in valid_emotions}
            )

            conversation = [
                {"role": "system", "content": "You are an expert in emotion analysis who provides accurate emotion judgment."},
                {"role": "user",   "content": [
                    {"type": "text",  "text": prompt},
                    {"type": "audio", "audio": audio},
                ]}
            ]

            text = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )

            inputs = self.processor(
                text=text,
                audio=[audio],
                return_tensors="pt",
                padding="max_length",
                max_length=512,
                sampling_rate=self.sample_rate,
            )

            # Remove batch dimension added by processor
            inputs = {
                k: v.squeeze(0) if isinstance(v, torch.Tensor) and v.dim() > 1 else v
                for k, v in inputs.items()
            }

            # Build emotion distribution tensor in canonical order
            emotion_dist_tensor = torch.tensor(
                [target_dist.get(e, 0.0) for e in valid_emotions], dtype=torch.float32
            )

            inputs.update({
                "target_json":    target_json,
                "target_dist":    target_dist,
                "example_id":     example["id"],
                "valid_emotions": valid_emotions,
                "emotion_tensor": emotion_dist_tensor,
                "audio_path":     audio_path,
            })

            return inputs

        except Exception as e:
            print(f"Error processing example {example['id']}: {str(e)}")
            valid_emotions = ["Anger", "Happiness", "Neutral state", "Sadness"] if is_iemocap \
                             else ["Angry", "Happy", "Neutral", "Sad"]
            return {
                "input_ids":      torch.ones(10, dtype=torch.long),
                "attention_mask": torch.ones(10, dtype=torch.long),
                "target_json":    "{}",
                "target_dist":    {e: 1.0 / len(valid_emotions) for e in valid_emotions},
                "example_id":     example["id"],
                "valid_emotions": valid_emotions,
                "emotion_tensor": torch.full((len(valid_emotions),), 1.0 / len(valid_emotions)),
            }


class EmotionDistributionDataCollator:
    """
    Collates variable-length inputs into padded batches.

    Pads 1D tensors (input_ids, attention_mask) to the longest sequence in
    the batch. Stacks fixed-size tensors (emotion_tensor). Passes through
    non-tensor fields (target_dist, example_id, valid_emotions, target_json).
    """

    def __init__(self, processor, device=None):
        self.processor = processor
        self.device    = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    def __call__(self, features):
        if not features:
            return None

        valid_features = [
            f for f in features
            if isinstance(f, dict)
            and "input_ids" in f
            and isinstance(f["input_ids"], torch.Tensor)
            and f["input_ids"].numel() > 0
        ]
        if not valid_features:
            return None

        special_fields = {"target_dist", "example_id", "valid_emotions", "target_json"}
        batch = {}

        for key in valid_features[0].keys():
            if key in special_fields:
                batch[key] = [f[key] for f in valid_features]
                continue

            if key == "emotion_tensor":
                shapes = [f[key].shape for f in valid_features if key in f]
                if all(s == shapes[0] for s in shapes):
                    batch[key] = torch.stack([f[key] for f in valid_features])
                else:
                    max_size   = max(s[0] for s in shapes)
                    batch[key] = torch.zeros((len(valid_features), max_size), dtype=torch.float32)
                    for i, f in enumerate(valid_features):
                        t = f[key]
                        batch[key][i, :t.shape[0]] = t
                continue

            if isinstance(valid_features[0].get(key), torch.Tensor):
                shapes = [f[key].shape for f in valid_features if key in f]
                if not shapes:
                    continue
                dtype  = valid_features[0][key].dtype

                if all(len(s) == 1 for s in shapes):
                    max_len = max(s[0] for s in shapes)
                    padded  = []
                    for f in valid_features:
                        t       = f[key]
                        pad_len = max_len - t.shape[0]
                        if pad_len > 0:
                            t = torch.cat([t, torch.zeros(pad_len, dtype=dtype)], dim=0)
                        padded.append(t)
                    batch[key] = torch.stack(padded).to(self.device)

        if "emotion_tensor" in batch:
            batch["emotion_distributions"] = batch["emotion_tensor"]

        if valid_features[0].get("valid_emotions"):
            batch["emotion_names"] = valid_features[0]["valid_emotions"]

        return batch


# ===========================================================================
# Section 6: Weighted Sampler
# ===========================================================================

def _resolve_target_dist(sample):
    """
    Extract the target emotion distribution dict from a dataset sample,
    handling the multiple storage formats used across the pipeline.
    """
    if hasattr(sample, "target_dist") and isinstance(sample.target_dist, dict):
        return sample.target_dist
    if isinstance(sample, dict) and "target_dist" in sample:
        return sample["target_dist"]
    for attr in ("target_json", "target_json"):
        val = getattr(sample, attr, None) if not isinstance(sample, dict) \
              else sample.get(attr)
        if val:
            try:
                return json.loads(val)
            except Exception:
                pass
    if hasattr(sample, "emotion_tensor") and isinstance(sample.emotion_tensor, torch.Tensor):
        if hasattr(sample, "valid_emotions") and sample.valid_emotions:
            return {e: float(sample.emotion_tensor[i])
                    for i, e in enumerate(sample.valid_emotions)}
    return None


def create_weighted_sampler(train_dataset):
    """
    Build a WeightedRandomSampler based on the dominant-emotion distribution
    in the (DiME-augmented) training set.

    Weighting strategy is chosen adaptively based on the imbalance ratio,
    using dataset-specific multipliers and thresholds from SAMPLER_CONFIG:
      ratio ≤ 2.0                    → uniform weights
      ratio ≤ moderate_threshold     → light per-emotion adjustment
      ratio  > moderate_threshold    → stronger per-emotion adjustment

    Returns None if the dataset is already balanced (ratio ≤ 1.5).
    """
    emotion_counts = {e: 0 for e in emotion_classes}

    for sample in train_dataset:
        dist = _resolve_target_dist(sample)
        if dist:
            dominant = max(dist, key=dist.get)
            if dominant in emotion_counts:
                emotion_counts[dominant] += 1

    total = sum(emotion_counts.values())
    if total == 0:
        print("Warning: no valid samples found for weighting")
        return None

    max_count       = max(emotion_counts.values())
    min_count       = max(min(emotion_counts.values()), 1)
    imbalance_ratio = max_count / min_count

    print(f"Dataset distribution: {emotion_counts}")
    print(f"Imbalance ratio: {imbalance_ratio:.2f}")

    emotion_freqs = {e: c / total for e, c in emotion_counts.items()}
    cfg           = SAMPLER_CONFIG[dataset]

    if imbalance_ratio <= 2.0:
        print("Well balanced — using uniform weights")
        weights = {e: 1.0 for e in emotion_counts}
    elif imbalance_ratio <= cfg["moderate_threshold"]:
        print("Moderate imbalance — using light weighting")
        multipliers = cfg["moderate_multipliers"]
        weights = {e: multipliers.get(e, 1.0) / (emotion_freqs[e] + 1e-5) for e in emotion_counts}
    else:
        print("Significant imbalance — using stronger weighting")
        multipliers = cfg["strong_multipliers"]
        weights = {e: multipliers.get(e, 1.0) / (emotion_freqs[e] + 1e-5) for e in emotion_counts}

    # Normalize so weights average to 1.0
    weight_sum = sum(weights.values())
    weights    = {e: w / weight_sum * len(weights) for e, w in weights.items()}
    print(f"Sampling weights: {weights}")

    sample_weights = []
    for sample in train_dataset:
        dist = _resolve_target_dist(sample)
        if dist:
            dominant = max(dist, key=dist.get)
            sample_weights.append(weights.get(dominant, 1.0))
        else:
            sample_weights.append(1.0)

    if imbalance_ratio > 1.5:
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )

    print("Dataset is balanced — using uniform sampling")
    return None


# ===========================================================================
# Section 7: Training Infrastructure
# ===========================================================================

def setup_training(train_examples, val_examples, processor, batch_size=1,
                   max_audio_length=30, DiME_applied=True):
    """
    Create train/val datasets and dataloaders from pre-split example lists.

    Verifies that the validation set contains no synthetic samples (they would
    bias the evaluation of real-data generalization).
    """
    print(f"Training examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")

    # Safety check: val set must be free of DiME-generated samples
    synthetic_in_val = [ex for ex in val_examples if ex.get("is_synthetic", False)]
    if synthetic_in_val:
        print(f"WARNING: removing {len(synthetic_in_val)} synthetic samples from val set")
        val_examples = [ex for ex in val_examples if not ex.get("is_synthetic", False)]
    else:
        print("Validation set purity verified (no synthetic samples)")

    train_dataset = EmotionDistributionDataset(
        train_examples, processor, max_audio_length=max_audio_length, training=True
    )
    val_dataset = EmotionDistributionDataset(
        val_examples, processor, max_audio_length=max_audio_length, training=False
    )

    data_collator = EmotionDistributionDataCollator(processor=processor)
    sampler       = create_weighted_sampler(train_dataset)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        collate_fn=data_collator, num_workers=4, drop_last=True, pin_memory=False
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=data_collator, num_workers=4, drop_last=True, pin_memory=False
    )

    return train_dataset, val_dataset, train_dataloader, val_dataloader


def prepare_lora_model(model, emotion_list):
    """
    Attach the EmotionClassifierHead and apply LoRA to the Qwen2-Audio backbone.

    LoRA is applied to the Q/K/V/O projection layers with rank r=8, alpha=16,
    dropout=0.2 (Section 3.2 of the paper). The emotion head is added to
    modules_to_save so its weights are included in the checkpoint.

    Parameters
    ----------
    model : Qwen2AudioForConditionalGeneration
    emotion_list : list of str
        Determines the output dimension of the classifier head.
    """
    model.emotion_head = EmotionClassifierHead(
        hidden_size=model.language_model.config.hidden_size,
        num_emotions=len(emotion_list)
    ).to(model.device)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        modules_to_save=["emotion_head"]
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    for param in model.emotion_head.parameters():
        param.requires_grad = True

    return model


def get_training_args(output_dir, batch_size=2, epochs=3):
    """
    Build Seq2SeqTrainingArguments with the hyperparameters from Section 3.2.
    """
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_pin_memory=False,
        gradient_accumulation_steps=8,
        learning_rate=2.5e-6,
        lr_scheduler_type="cosine_with_restarts",
        warmup_ratio=0.15,
        weight_decay=0.03,
        max_grad_norm=3.0,
        generation_max_length=60,
        num_train_epochs=epochs,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_strategy="steps",
        logging_steps=5,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        report_to="none",
        predict_with_generate=False,
        remove_unused_columns=False,
    )

old_base_forward = model.base_model.forward

def new_base_forward(*args, **kwargs):
    kwargs.pop("decoder_input_ids",     None)
    kwargs.pop("decoder_attention_mask", None)
    kwargs.pop("decoder_inputs_embeds", None)

    # Align labels and attention_mask lengths if mismatched
    if ("labels" in kwargs and kwargs["labels"] is not None
            and "attention_mask" in kwargs
            and kwargs["labels"].shape[1] != kwargs["attention_mask"].shape[1]):
        min_len = min(kwargs["labels"].shape[1], kwargs["attention_mask"].shape[1])
        kwargs["labels"]          = kwargs["labels"][:, :min_len]
        kwargs["attention_mask"]  = kwargs["attention_mask"][:, :min_len]

    return old_base_forward(*args, **kwargs)

model.base_model.forward = new_base_forward


class EmotionDistributionTrainer(Seq2SeqTrainer):
    """
    Custom HuggingFace Trainer that routes loss through the EmotionClassifierHead
    and EmotionLoss rather than the default language-modelling head.
    """

    def __init__(self, *args, **kwargs):
        self.loss_fn = kwargs.pop("loss_fn", None) or EmotionLoss(epsilon=1e-10)
        super().__init__(*args, **kwargs)

    def _forward_with_emotion_head(self, model, inputs):
        """Shared forward pass returning (loss, emotion_logits, target_distributions)."""
        input_ids            = inputs.get("input_ids")
        attention_mask       = inputs.get("attention_mask")
        target_distributions = inputs.get("emotion_distributions")

        # Forward all non-special keys as additional model inputs (e.g. audio features)
        other_inputs = {
            k: v for k, v in inputs.items()
            if k not in {"input_ids", "attention_mask", "emotion_distributions",
                         "target_dist", "example_id", "valid_emotions",
                         "target_json", "emotion_tensor", "emotion_names"}
        }

        outputs      = model(input_ids=input_ids, attention_mask=attention_mask,
                             output_hidden_states=True, **other_inputs)
        hidden_states = outputs.hidden_states[-1]
        emotion_logits = model.emotion_head(hidden_states)
        loss           = self.loss_fn(emotion_logits, target_distributions)

        return loss, emotion_logits, target_distributions

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss, outputs_logits, _ = self._forward_with_emotion_head(model, inputs)
        if return_outputs:
            return loss, outputs_logits
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        was_training = model.training
        model.eval()
        with torch.no_grad():
            loss, emotion_logits, labels = self._forward_with_emotion_head(model, inputs)
        model.train(was_training)

        if prediction_loss_only:
            return (loss, None, None)
        return (loss, emotion_logits, labels)

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs     = self._prepare_inputs(inputs)
        fp16_on    = getattr(self.args, "fp16", False)

        if fp16_on:
            with torch.amp.autocast('cuda', enabled=True):
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if fp16_on and hasattr(self, "scaler") and self.scaler is not None:
            self.scaler.scale(loss).backward()
        elif hasattr(self, "accelerator") and self.accelerator is not None:
            self.accelerator.backward(loss)
        else:
            loss.backward()

        # Gradient clipping at accumulation boundaries
        should_clip = (self.args.gradient_accumulation_steps <= 1) or (
            (self.state.global_step + 1) % self.args.gradient_accumulation_steps == 0
        )
        if should_clip:
            clip_norm  = getattr(self.args, "max_grad_norm", 5.0)
            parameters = [p for p in model.parameters() if p.requires_grad]
            if fp16_on and hasattr(self, "scaler") and self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, clip_norm)

        return loss.detach()


def test_model_with_emotion_head(model, val_dataset, loss_fn, valid_emotions):
    """
    Quick sanity check on 10 random validation examples after training.
    Prints predicted vs. target distributions for visual inspection.
    """
    print("Testing model on validation examples...")
    num_test = min(10, len(val_dataset))
    indices  = random.sample(range(len(val_dataset)), num_test)

    for idx in indices:
        example       = val_dataset[idx]
        audio_path    = example.get("audio_path", "")
        target_dist   = example.get("target_dist", {})
        example_id    = example.get("example_id", f"example_{idx}")
        input_ids     = example.get("input_ids")
        attention_mask = example.get("attention_mask")

        if input_ids is None or attention_mask is None:
            print(f"Skipping {example_id} — missing input tensors")
            continue

        input_ids      = input_ids.unsqueeze(0).to(model.device)
        attention_mask = attention_mask.unsqueeze(0).to(model.device)

        with torch.no_grad():
            outputs        = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            emotion_logits = model.emotion_head(outputs.hidden_states[-1])
            pred_probs     = F.softmax(emotion_logits, dim=-1)

        pred_dist = {e: float(pred_probs[0, i].cpu()) for i, e in enumerate(valid_emotions)}

        print(f"\n{example_id}  ({audio_path})")
        print(f"  Target:    {target_dist}")
        print(f"  Predicted: {pred_dist}")


# ===========================================================================
# Section 8: Main Fine-tuning Entry Point
# ===========================================================================

def finetune_emotion_distribution_model(
    train_examples, val_examples, model, processor,
    output_dir, emotion_list,
    batch_size=2, epochs=3, max_samples=None
):
    """
    End-to-end fine-tuning pipeline.

    Parameters
    ----------
    train_examples : list of dict  (DiME-augmented)
    val_examples   : list of dict  (pure real)
    model          : Qwen2AudioForConditionalGeneration
    processor      : AutoProcessor
    output_dir     : str
    emotion_list   : list of str
    batch_size     : int
    epochs         : int
    max_samples    : int or None   (cap training set size for debugging)

    Returns
    -------
    (model, trainer)
    """
    if max_samples and max_samples < len(train_examples):
        train_examples = train_examples[:max_samples]
        print(f"Using {len(train_examples)} training examples (capped)")
    else:
        print(f"Using all {len(train_examples)} training examples")

    print(f"Validation set: {len(val_examples)} examples")

    # Log dominant-emotion frequencies in training set
    emotion_counts = {e: 0 for e in emotion_list}
    for ex in train_examples:
        target_dist = json.loads(ex["target"]) if isinstance(ex["target"], str) else ex["target"]
        if target_dist:
            dominant = max(target_dist, key=target_dist.get)
            if dominant in emotion_counts:
                emotion_counts[dominant] += 1
    print(f"Training emotion counts: {emotion_counts}")

    train_dataset, val_dataset, train_dataloader, val_dataloader = setup_training(
        train_examples=train_examples,
        val_examples=val_examples,
        processor=processor,
        batch_size=batch_size,
        DiME_applied=True
    )

    print("\nApplying LoRA to model...")
    model   = prepare_lora_model(model, emotion_list)
    loss_fn = EmotionLoss(epsilon=1e-10)

    training_args = get_training_args(output_dir, batch_size=batch_size, epochs=epochs)

    trainer = EmotionDistributionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss_fn=loss_fn,
        processing_class=processor.tokenizer,
        data_collator=EmotionDistributionDataCollator(processor=processor),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=8, early_stopping_threshold=0.0)]
    )

    print("\nStarting fine-tuning...")
    try:
        torch.cuda.empty_cache()
        trainer.train()
    except Exception as e:
        import traceback
        print(f"Training failed: {e}")
        traceback.print_exc()

        # Debug: try a single batch to isolate the error
        print("\nDebug — single batch test:")
        batch = next(iter(train_dataloader))
        try:
            device = model.device
            batch  = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out    = model(input_ids=batch.get("input_ids"), attention_mask=batch.get("attention_mask"))
            loss   = EmotionLoss()(out.logits, batch.get("emotion_distributions"))
            print(f"Single batch loss: {loss.item()}")
        except Exception as batch_err:
            print(f"Single batch error: {batch_err}")
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

        return model, trainer, loss_fn

    print(f"\nFine-tuning complete. Saving model to {output_dir}...")
    model.save_pretrained(output_dir)

    test_model_with_emotion_head(model, val_dataset, loss_fn, emotion_list)

    return model, trainer


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

finetuned_model, trainer = finetune_emotion_distribution_model(
    train_examples=train_augmented,
    val_examples=val_real,
    model=model,
    processor=processor,
    output_dir=finetune_output_dir,
    emotion_list=emotion_classes,
    batch_size=args.batch_size,
    epochs=args.epochs,
)
print("\nTraining process completed!")

# Clean up DiME temporary audio files
if os.path.exists(dime_temp_dir):
    shutil.rmtree(dime_temp_dir)
    print(f"Cleaned up DiME temp directory: {dime_temp_dir}")
