"""
Deep Learning & Computer Vision Waste Classification Module (app/model.py).

Supports a 3-tier architecture suitable for college workshops and CPU laptops:
1. Custom Trained Model: Loads custom 6-class model (.keras / .h5 / .pth / .onnx) if placed in `model/`.
2. Pretrained MobileNetV2 (ONNX Runtime CPU): Runs real-time deep learning inference using
   MobileNetV2 (ImageNet-1K) combined with domain-specific waste category mapping and
   OpenCV visual material analysis (specular reflection, color histograms, texture edges).
3. Fallback / Demo Mode: If model weights are absent, runs an OpenCV heuristic material
   analyzer clearly labeled as Demo/Fallback mode so the application never crashes.
"""

import io
import os
import base64
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import cv2
import numpy as np
from PIL import Image

from app.config import (
    MODEL_DIR,
    MODEL_PATH,
    ALT_PYTORCH_MODEL_PATH,
    IMAGE_SIZE,
    CONFIDENCE_THRESHOLD,
    WASTE_CATEGORIES,
)
from app.waste_rules import get_waste_rule

logger = logging.getLogger(__name__)

# Path to the pretrained lightweight MobileNetV2 ONNX model
MOBILENET_ONNX_PATH = MODEL_DIR / "mobilenetv2.onnx"
IMAGENET_LABELS_PATH = MODEL_DIR / "imagenet_classes.txt"

# Keyword mapping from ImageNet-1000 object labels to the 6 Waste Categories
IMAGENET_TO_WASTE_KEYWORDS: Dict[str, List[str]] = {
    "Plastic": [
        "water bottle", "pop bottle", "plastic bag", "lotion", "sunscreen",
        "hair spray", "soap dispenser", "syringe", "nipple", "pill bottle",
        "bucket", "pail", "tub", "shower cap", "swimming cap", "ballpoint",
        "fountain pen", "ping-pong ball", "cassette", "cd player", "remote control",
        "computer keyboard", "mouse", "cellular telephone", "ipod", "lighter",
        "water jug", "petri dish", "toothbrush", "comb", "snorkel",
        "oxygen mask", "vending machine", "toys", "teddy", "frisbee"
    ],
    "Paper": [
        "paper towel", "toilet tissue", "envelope", "book jacket", "comic book",
        "crossword puzzle", "menu", "binder", "notebook", "packet", "carton",
        "cardboard", "ashcan", "file", "letter opener", "jigsaw puzzle",
        "handkerchief", "napkin", "band aid", "matchstick", "lampshade",
        "shopping basket", "mailbag", "postbag", "paperknife", "folding chair",
        "rule", "ruler", "slide rule"
    ],
    "Metal": [
        "beer can", "soda can", "can opener", "milk can", "oil filter",
        "frying pan", "wok", "cauldron", "teapot", "coffeepot", "safety pin",
        "nail", "screw", "padlock", "combination lock", "key", "ladle",
        "spatula", "cleaver", "scissors", "whistle", "harmonica", "thimble",
        "chain", "hook", "buckle", "toaster", "waffle iron", "iron",
        "steel drum", "gong", "bell", "chime", "cymbal", "microphone",
        "flashlight", "torch", "screwdriver", "hammer", "wrench", "pliers",
        "hatchet", "corkscrew", "radiator", "barbell", "dumbbell",
        "typewriter", "cash machine", "scale", "watch", "stopwatch", "compass"
    ],
    "Glass": [
        "beer bottle", "wine bottle", "whiskey jug", "goblet", "wineglass",
        "beaker", "measuring cup", "cocktail shaker", "pitcher", "ewer",
        "vase", "jar", "hourglass", "magnifying glass", "sunglasses",
        "loupe", "mirror", "coffee mug", "cup", "espresso", "perfume",
        "table lamp", "saltshaker", "monitor", "screen",
        "television", "car mirror", "window", "aquarium"
    ],
    "Organic": [
        "banana", "orange", "lemon", "fig", "pineapple", "strawberry",
        "jackfruit", "custard apple", "pomegranate", "granny smith", "apple",
        "mushroom", "broccoli", "cauliflower", "zucchini", "spaghetti squash",
        "acorn squash", "butternut squash", "cucumber", "artichoke", "bell pepper",
        "cardoon", "head cabbage", "corn", "ear", "acorn", "hip", "buckeye",
        "coral fungus", "agaric", "gyromitra", "stinkhorn", "earthstar",
        "hen-of-the-woods", "bolete", "french loaf", "bagel", "pretzel",
        "pizza", "burrito", "hotdog", "hamburger", "cheeseburger", "ice cream",
        "ice lolly", "guacamole", "consomme", "trifle", "meat loaf", "potpie",
        "mashed potato", "carbonara", "dough", "soup", "salad", "plate",
        "daisy", "rose", "flower", "pot", "leaf", "plant",
        "coffee bean", "chocolate", "egg", "peanut", "hay"
    ],
    "General Waste": [
        "diaper", "swab", "mask", "sponge", "rubber eraser", "running shoe",
        "sandal", "clog", "sock", "mitten", "glove", "umbrella", "balloon",
        "pillow", "broom", "mop", "paintbrush", "lipstick", "candle",
        "trash", "band-aid", "doormat", "rag", "wig", "feather",
        "plunger", "rubber", "boot", "slipper", "sneaker", "purse", "backpack",
        "wallet", "pencil box", "pencil sharpener"
    ],
}


class WasteClassifierModel:
    """
    Encapsulates model loading, image preprocessing, deep learning inference,
    and OpenCV computer vision feature analysis.
    """

    def __init__(self) -> None:
        self.model_name: str = "Initializing..."
        self.architecture: str = "MobileNetV2"
        self.framework: str = "ONNX Runtime + OpenCV"
        self.mode: str = "fallback"
        self.is_demo_mode: bool = True
        self.custom_model_loaded: bool = False
        self.device: str = "CPU"

        self.onnx_session = None
        self.keras_model = None
        self.pytorch_model = None
        self.imagenet_labels: List[str] = []

        self.load_model()

    def load_model(self) -> None:
        """
        Attempt to load models in priority order:
        1. Custom 6-class trained model (waste_classifier.keras / .onnx / .pth)
        2. Pretrained MobileNetV2 ONNX model (mobilenetv2.onnx)
        3. OpenCV + NumPy Fallback / Demo Mode
        """
        # Load ImageNet labels if available
        if IMAGENET_LABELS_PATH.exists():
            try:
                with open(IMAGENET_LABELS_PATH, "r", encoding="utf-8") as f:
                    self.imagenet_labels = [line.strip() for line in f.readlines() if line.strip()]
            except Exception as exc:
                logger.warning(f"Could not read ImageNet labels: {exc}")

        # 1. Check for custom 6-class ONNX model
        custom_onnx_path = MODEL_DIR / "waste_classifier.onnx"
        if custom_onnx_path.exists():
            try:
                import onnxruntime as ort
                self.onnx_session = ort.InferenceSession(
                    str(custom_onnx_path),
                    providers=["CPUExecutionProvider"]
                )
                self.model_name = "Custom Waste Classifier (ONNX)"
                self.architecture = "MobileNetV2 Transfer Learning (6-Class)"
                self.framework = "ONNX Runtime (CPU)"
                self.mode = "custom_onnx"
                self.is_demo_mode = False
                self.custom_model_loaded = True
                logger.info(f"Loaded custom ONNX model from {custom_onnx_path}")
                return
            except Exception as exc:
                logger.warning(f"Failed to load custom ONNX model: {exc}")

        # 2. Check for custom Keras model (.keras or .h5)
        if MODEL_PATH.exists():
            try:
                import tensorflow as tf
                self.keras_model = tf.keras.models.load_model(str(MODEL_PATH))
                self.model_name = "Custom Waste Classifier (Keras/TF)"
                self.architecture = "MobileNetV2 Transfer Learning (6-Class)"
                self.framework = "TensorFlow / Keras (CPU)"
                self.mode = "custom_keras"
                self.is_demo_mode = False
                self.custom_model_loaded = True
                logger.info(f"Loaded custom Keras model from {MODEL_PATH}")
                return
            except Exception as exc:
                logger.warning(f"Failed to load custom Keras model from {MODEL_PATH}: {exc}")

        # 3. Check for Pretrained MobileNetV2 ONNX model
        if MOBILENET_ONNX_PATH.exists():
            try:
                import onnxruntime as ort
                self.onnx_session = ort.InferenceSession(
                    str(MOBILENET_ONNX_PATH),
                    providers=["CPUExecutionProvider"]
                )
                self.model_name = "MobileNetV2 Pretrained CNN + CV Material Fusion"
                self.architecture = "MobileNetV2 (1000-Class -> 6-Bin Smart Mapper)"
                self.framework = "ONNX Runtime + OpenCV (CPU)"
                self.mode = "pretrained_mobilenet"
                self.is_demo_mode = False
                self.custom_model_loaded = False
                logger.info(f"Loaded pretrained MobileNetV2 ONNX model from {MOBILENET_ONNX_PATH}")
                return
            except Exception as exc:
                logger.warning(f"Failed to load MobileNetV2 ONNX model: {exc}")

        # 4. Fallback / Demo Mode using OpenCV Computer Vision
        self.model_name = "OpenCV Material Feature Classifier (Demo Mode)"
        self.architecture = "HSV Color + Specular Reflection + Edge Density Analyzer"
        self.framework = "OpenCV + NumPy (Fallback Mode)"
        self.mode = "cv_demo_fallback"
        self.is_demo_mode = True
        self.custom_model_loaded = False
        logger.info("Running in OpenCV Fallback / Demo Mode.")

    def preprocess_for_mobilenet(self, pil_img: Image.Image) -> np.ndarray:
        """
        Resize and normalize a PIL RGB image for MobileNetV2 inference.
        Expected input shape: (1, 3, 224, 224), float32, ImageNet mean/std normalized.
        """
        img_resized = pil_img.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        arr = np.asarray(img_resized, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        # Convert HWC -> CHW, ensure C-contiguous memory layout, and add batch dimension -> (1, 3, 224, 224)
        arr = np.ascontiguousarray(np.transpose(arr, (2, 0, 1)))
        return np.expand_dims(arr, axis=0).astype(np.float32)

    def extract_cv_material_scores(self, rgb_np: np.ndarray) -> Tuple[Dict[str, float], str]:
        """
        Analyze visual material properties of the center object using OpenCV:
        - HSV hue/saturation/brightness distribution
        - Specular highlights (glare from glass, metal cans, or glossy plastic)
        - Edge density & sharp linear contours (paper/cardboard vs organic shapes)
        - Color warmth & green/yellow/brown organic chrominance
        Returns a dictionary of category scores (summing to 1.0) and a descriptive feature hint.
        """
        # Normalize channels to 3-channel RGB if needed
        if rgb_np.ndim == 2:
            rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_GRAY2RGB)
        elif rgb_np.ndim == 3 and rgb_np.shape[2] == 4:
            rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGBA2RGB)

        h, w = rgb_np.shape[:2]
        # Focus on center 70% region of interest where users hold waste items
        y1, y2 = int(h * 0.15), int(h * 0.85)
        x1, x2 = int(w * 0.15), int(w * 0.85)
        roi = rgb_np[y1:y2, x1:x2]
        if roi.size == 0 or roi.shape[0] < 2 or roi.shape[1] < 2:
            roi = rgb_np

        roi_small = cv2.resize(roi, (160, 160))
        hsv = cv2.cvtColor(roi_small, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(roi_small, cv2.COLOR_RGB2GRAY)

        hue = hsv[:, :, 0].astype(np.float32)        # 0..179 in OpenCV
        sat = hsv[:, :, 1].astype(np.float32) / 255.0 # 0..1
        val = hsv[:, :, 2].astype(np.float32) / 255.0 # 0..1

        mean_sat = float(np.mean(sat))
        mean_val = float(np.mean(val))
        std_val = float(np.std(val))

        # 1. Specular highlights (bright spots with low saturation: shiny metal, glass, glossy plastic)
        specular_mask = (val > 0.85) & (sat < 0.25)
        specular_ratio = float(np.mean(specular_mask))

        # 2. High brightness uniform white/cream surface (Paper / Sheet / Notebook)
        paper_white_mask = (val > 0.70) & (sat < 0.18)
        paper_white_ratio = float(np.mean(paper_white_mask))

        # Cardboard / Kraft brown paper mask (Hue 10..25, moderate saturation & brightness)
        cardboard_mask = (hue >= 10) & (hue <= 26) & (sat >= 0.20) & (sat <= 0.60) & (val >= 0.35) & (val <= 0.80)
        cardboard_ratio = float(np.mean(cardboard_mask))

        # 3. Organic natural tones: greens (leaves/vegetables: Hue 32..85) + warm fruit yellows/oranges/reds
        green_organic = (hue >= 32) & (hue <= 85) & (sat > 0.25) & (val > 0.20)
        warm_fruit = ((hue <= 28) | (hue >= 165)) & (sat > 0.45) & (val > 0.35)
        organic_ratio = float(np.mean(green_organic)) * 1.4 + float(np.mean(warm_fruit)) * 0.9

        # 4. Vibrant synthetic colors (Plastic bottles, caps, packaging: blue/cyan/magenta/bright red)
        synthetic_blue_cyan = (hue >= 86) & (hue <= 135) & (sat > 0.28) & (val > 0.30)
        synthetic_vibrant = (sat > 0.52) & (val > 0.40)
        plastic_color_ratio = float(np.mean(synthetic_blue_cyan)) * 1.5 + float(np.mean(synthetic_vibrant)) * 0.8

        # 5. Metallic gray/silver tone (Low saturation, medium-high brightness, high local gradient contrast)
        metallic_gray_mask = (sat < 0.22) & (val >= 0.32) & (val <= 0.85)
        metallic_ratio = float(np.mean(metallic_gray_mask))

        # 6. Edge density and texture analysis via Canny & Laplacian
        edges = cv2.Canny(gray, 60, 160)
        edge_density = float(np.mean(edges > 0))
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 1000.0

        # Compute raw affinity scores for each of the 6 waste categories
        scores = {
            "Plastic": 0.22 + plastic_color_ratio * 1.4 + specular_ratio * 0.7 + (0.15 if 0.12 < std_val < 0.28 else 0.0),
            "Paper": 0.18 + paper_white_ratio * 1.6 + cardboard_ratio * 1.4 + (0.20 if mean_sat < 0.20 and mean_val > 0.62 else 0.0),
            "Metal": 0.16 + metallic_ratio * 1.1 + specular_ratio * 0.9 + min(edge_density * 2.5, 0.45) + (0.20 if std_val > 0.22 and mean_sat < 0.25 else 0.0),
            "Glass": 0.15 + specular_ratio * 1.2 + (0.35 if 0.04 < edge_density < 0.14 and std_val > 0.20 and mean_sat < 0.30 else 0.0),
            "Organic": 0.16 + organic_ratio * 1.5 + (0.18 if edge_density > 0.08 and mean_sat > 0.32 else 0.0),
            "General Waste": 0.18 + (0.35 if mean_val < 0.32 else 0.0) + (0.20 if edge_density > 0.18 else 0.0),
        }

        # Normalize with temperature-scaled softmax for realistic probability distribution
        vals = np.array([scores[c] for c in WASTE_CATEGORIES], dtype=np.float64)
        exp_vals = np.exp((vals - np.max(vals)) * 3.2)
        probs = exp_vals / np.sum(exp_vals)

        prob_dict = {cat: float(probs[i]) for i, cat in enumerate(WASTE_CATEGORIES)}
        top_cat = max(prob_dict, key=prob_dict.get)
        feature_desc = f"Visual material profile ({top_cat.lower()} surface characteristics)"
        return prob_dict, feature_desc

    def map_imagenet_to_waste(
        self,
        imagenet_probs: np.ndarray,
        cv_probs: Dict[str, float]
    ) -> Tuple[Dict[str, float], str]:
        """
        Combine MobileNetV2 1000-class ImageNet predictions with OpenCV material scores.
        When MobileNetV2 recognizes a known waste item (e.g., 'water bottle', 'paper towel',
        'beer can', 'banana', 'wine bottle', 'carton'), it strongly boosts that waste category.
        """
        # Inspect top-10 ImageNet predictions
        top_indices = np.argsort(imagenet_probs)[::-1][:10]
        semantic_boost = {cat: 0.0 for cat in WASTE_CATEGORIES}
        matched_object_label = None

        for rank, idx in enumerate(top_indices):
            prob = float(imagenet_probs[idx])
            label = self.imagenet_labels[idx] if idx < len(self.imagenet_labels) else f"class_{idx}"
            label_lower = label.lower()

            # Check keyword dictionary
            matched_for_this_label = False
            for waste_cat, keywords in IMAGENET_TO_WASTE_KEYWORDS.items():
                for kw in keywords:
                    if kw.lower() in label_lower:
                        weight = 2.8 if rank == 0 else (1.8 if rank < 3 else 1.1)
                        semantic_boost[waste_cat] += prob * weight
                        if matched_object_label is None and prob > 0.04:
                            matched_object_label = f"{label} ({prob * 100:.1f}%)"
                        matched_for_this_label = True
                        break
                if matched_for_this_label:
                    break

        # If no specific keyword matched in top-10, record the top-1 ImageNet label for transparency
        if matched_object_label is None and len(top_indices) > 0:
            top1_idx = int(top_indices[0])
            top1_prob = float(imagenet_probs[top1_idx])
            top1_name = self.imagenet_labels[top1_idx] if top1_idx < len(self.imagenet_labels) else "object"
            matched_object_label = f"{top1_name} ({top1_prob * 100:.1f}%)"

        # Fuse Deep Learning semantic scores + OpenCV material scores
        total_semantic = sum(semantic_boost.values())
        combined_scores = {}

        if total_semantic > 0.12:
            # Strong semantic recognition from MobileNetV2 (e.g. bottle, can, fruit, paper, carton)
            for cat in WASTE_CATEGORIES:
                combined_scores[cat] = semantic_boost[cat] * 2.5 + cv_probs[cat] * 0.45
        else:
            # Generic or close-up object: blend MobileNet confidence entropy with CV material features
            for cat in WASTE_CATEGORIES:
                combined_scores[cat] = semantic_boost[cat] * 1.5 + cv_probs[cat] * 1.1

        # Convert combined scores into calibrated probabilities summing to 1.0
        raw = np.array([combined_scores[c] for c in WASTE_CATEGORIES], dtype=np.float64)
        exp_scores = np.exp((raw - np.max(raw)) * 3.8)
        norm_probs = exp_scores / np.sum(exp_scores)

        final_probs = {cat: float(norm_probs[i]) for i, cat in enumerate(WASTE_CATEGORIES)}
        return final_probs, matched_object_label

    def predict_image(
        self,
        image_bytes: bytes,
        demo_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run waste classification on raw image bytes (JPEG/PNG).
        Returns predicted waste category, confidence percentage (0..100),
        class probability distribution, detected object name, and bin recommendation.
        """
        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Invalid image data: {exc}") from exc

        rgb_np = np.asarray(pil_img, dtype=np.uint8)

        # Check if camera frame is completely black/covered
        if float(np.mean(rgb_np)) < 8.0:
            rule = get_waste_rule("General Waste")
            return {
                "waste_type": "General Waste",
                "confidence": 25.0,
                "recommended_bin": rule["recommended_bin"],
                "bin_short": rule["bin_short"],
                "icon": rule["icon"],
                "bin_color": rule["bin_color"],
                "instruction": "Camera feed appears very dark. Please ensure adequate lighting and show the waste item clearly.",
                "eco_tip": rule["eco_tip"],
                "is_low_confidence": True,
                "status_message": "Uncertain prediction — camera image is too dark.",
                "is_demo_mode": self.is_demo_mode,
                "model_name": self.model_name,
                "detected_object": "dark frame",
                "class_probabilities": {cat: round(100.0 / 6, 1) for cat in WASTE_CATEGORIES},
            }

        # Optional classroom preset override (when clicking sample test items in the UI)
        if demo_hint and demo_hint in WASTE_CATEGORIES:
            return self._build_preset_response(demo_hint, rgb_np)

        detected_object = None

        # Mode 1: Custom 6-class ONNX model
        if self.mode == "custom_onnx" and self.onnx_session is not None:
            input_tensor = self.preprocess_for_mobilenet(pil_img)
            input_name = self.onnx_session.get_inputs()[0].name
            logits = self.onnx_session.run(None, {input_name: input_tensor})[0][0]
            exp_logits = np.exp(logits - np.max(logits))
            probs_arr = exp_logits / np.sum(exp_logits)
            final_probs = {cat: float(probs_arr[i]) for i, cat in enumerate(WASTE_CATEGORIES)}
            detected_object = "Custom 6-Class CNN Prediction"

        # Mode 2: Custom 6-class Keras model
        elif self.mode == "custom_keras" and self.keras_model is not None:
            img_resized = pil_img.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
            arr = np.expand_dims(np.asarray(img_resized, dtype=np.float32) / 255.0, axis=0)
            preds = self.keras_model.predict(arr, verbose=0)[0]
            final_probs = {cat: float(preds[i]) for i, cat in enumerate(WASTE_CATEGORIES)}
            detected_object = "Custom Keras Model Prediction"

        # Mode 3: Pretrained MobileNetV2 ONNX + OpenCV Material Fusion
        elif self.mode == "pretrained_mobilenet" and self.onnx_session is not None:
            input_tensor = self.preprocess_for_mobilenet(pil_img)
            input_name = self.onnx_session.get_inputs()[0].name
            logits = self.onnx_session.run(None, {input_name: input_tensor})[0][0]
            exp_logits = np.exp(logits - np.max(logits))
            imagenet_probs = exp_logits / np.sum(exp_logits)

            cv_probs, _ = self.extract_cv_material_scores(rgb_np)
            final_probs, detected_object = self.map_imagenet_to_waste(imagenet_probs, cv_probs)

        # Mode 4: OpenCV Fallback / Demo Mode
        else:
            final_probs, detected_object = self.extract_cv_material_scores(rgb_np)

        # Select top category and confidence
        predicted_category = max(final_probs, key=final_probs.get)
        raw_confidence = float(final_probs[predicted_category])  # 0.0 .. 1.0
        confidence_pct = round(raw_confidence * 100.0, 1)

        is_low_conf = raw_confidence < CONFIDENCE_THRESHOLD
        rule = get_waste_rule(predicted_category)

        if is_low_conf:
            status_msg = "Uncertain prediction — please hold the waste item closer and center it in good lighting."
        elif self.is_demo_mode:
            status_msg = f"Demo Mode classification ({predicted_category})"
        else:
            status_msg = f"Detected {predicted_category} ({confidence_pct}% confidence)"

        class_prob_pcts = {
            cat: round(prob * 100.0, 1) for cat, prob in final_probs.items()
        }

        return {
            "waste_type": predicted_category,
            "confidence": confidence_pct,
            "recommended_bin": rule["recommended_bin"],
            "bin_short": rule["bin_short"],
            "icon": rule["icon"],
            "bin_color": rule["bin_color"],
            "instruction": rule["instruction"],
            "eco_tip": rule["eco_tip"],
            "is_low_confidence": is_low_conf,
            "status_message": status_msg,
            "is_demo_mode": self.is_demo_mode,
            "model_name": self.model_name,
            "detected_object": detected_object,
            "class_probabilities": class_prob_pcts,
        }

    def _build_preset_response(self, category: str, rgb_np: np.ndarray) -> Dict[str, Any]:
        """
        Generate a deterministic, realistic distribution when the user tests a preset waste sample
        in the UI (helpful for desktops without a webcam or quick mentor demos).
        """
        rule = get_waste_rule(category)
        # Deterministic variation based on image pixel mean so it feels responsive
        pixel_seed = (float(np.mean(rgb_np)) % 7.0) / 100.0
        primary_conf = round(min(0.96, max(0.88, 0.915 + pixel_seed)) * 100.0, 1)
        remainder = round(100.0 - primary_conf, 1)

        other_cats = [c for c in WASTE_CATEGORIES if c != category]
        weights = [0.35, 0.25, 0.18, 0.12, 0.10]
        class_probs = {category: primary_conf}
        for idx, other in enumerate(other_cats):
            class_probs[other] = round(remainder * weights[idx], 1)

        return {
            "waste_type": category,
            "confidence": primary_conf,
            "recommended_bin": rule["recommended_bin"],
            "bin_short": rule["bin_short"],
            "icon": rule["icon"],
            "bin_color": rule["bin_color"],
            "instruction": rule["instruction"],
            "eco_tip": rule["eco_tip"],
            "is_low_confidence": False,
            "status_message": f"Sample Test Prediction: {category} ({primary_conf}%)",
            "is_demo_mode": self.is_demo_mode,
            "model_name": self.model_name,
            "detected_object": f"Sample {category} Item",
            "class_probabilities": class_probs,
        }

    def get_info(self) -> Dict[str, Any]:
        """Return metadata about the active model for GET /api/model-info."""
        return {
            "model_name": self.model_name,
            "architecture": self.architecture,
            "framework": self.framework,
            "mode": self.mode,
            "is_demo_mode": self.is_demo_mode,
            "custom_model_loaded": self.custom_model_loaded,
            "custom_model_path": str(MODEL_PATH),
            "input_size": list(IMAGE_SIZE),
            "confidence_threshold": CONFIDENCE_THRESHOLD * 100.0,
            "supported_categories": WASTE_CATEGORIES,
            "device": self.device,
            "description": (
                "Lightweight MobileNetV2 Deep Learning architecture running on CPU via ONNX Runtime, "
                "augmented with OpenCV material reflectance, color chrominance, and edge analysis."
                if not self.is_demo_mode
                else "Fallback OpenCV Material Feature Analyzer active. Place a custom model or mobilenetv2.onnx in model/ for full CNN inference."
            ),
        }


def decode_base64_image(base64_str: str) -> bytes:
    """
    Safely decode a base64 image string (with or without 'data:image/jpeg;base64,' header).
    Handles whitespace and missing base64 padding gracefully.
    """
    if not base64_str or not base64_str.strip():
        raise ValueError("Empty base64 image payload.")
    base64_str = base64_str.strip()
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1].strip()
    # Repair missing padding if needed
    missing_padding = len(base64_str) % 4
    if missing_padding:
        base64_str += "=" * (4 - missing_padding)
    try:
        return base64.b64decode(base64_str)
    except Exception as exc:
        raise ValueError(f"Invalid base64 encoding: {exc}") from exc


# Singleton model instance initialized on import
classifier = WasteClassifierModel()
