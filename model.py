"""
Machine Learning Model Module
This module handles the ML model for fake content detection
Uses pre-trained transformer models (BERT) fine-tuned for classification
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModel
)
import numpy as np
from typing import Dict, Tuple, List
import logging
import os
from pathlib import Path

# Import project configuration
import config

# Set up logging
logger = logging.getLogger(__name__)


class FakeNewsDetector:
    """
    Main model class for detecting fake news
    Combines transformer-based text classification with traditional ML features
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize the fake news detector model
        
        Args:
            model_name (str): Name of the pre-trained model to use
        """
        # Use default model from config if not specified
        self.model_name = model_name or config.MODEL_NAME
        
        # Determine device (GPU if available, otherwise CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer (converts text to tokens that BERT understands)
        logger.info(f"Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load pre-trained model
        # In production, you would load a fine-tuned model
        # For now, we'll use the base model and demonstrate the architecture
        logger.info(f"Loading model: {self.model_name}")
        self.model = None  # Will be loaded when needed
        
        # Flag to track if model is loaded
        self.model_loaded = False
        
        # Class labels
        self.labels = ['REAL', 'FAKE']
        
        logger.info("FakeNewsDetector initialized successfully")
    
    def load_model(self):
        """
        Load the pre-trained model into memory
        Separated from __init__ to allow lazy loading
        """
        if self.model_loaded:
            return
        
        try:
            # Check if we have a fine-tuned model saved locally
            local_model_path = config.MODELS_DIR / "fake_news_detector"
            
            if local_model_path.exists():
                logger.info(f"Loading fine-tuned model from {local_model_path}")
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    str(local_model_path)
                )
            else:
                # Use base pre-trained model
                # In production, you'd want to fine-tune this first
                logger.info(f"Loading base model (not fine-tuned): {self.model_name}")
                
                # For demonstration, we'll create a simple classifier
                # In reality, you'd fine-tune on a fake news dataset
                self.model = AutoModel.from_pretrained(self.model_name)
                
                # Add a classification head on top of BERT
                # This takes BERT's output and predicts REAL or FAKE
                self.classifier = nn.Sequential(
                    nn.Linear(self.model.config.hidden_size, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 2)  # 2 classes: REAL, FAKE
                )
            
            # Move model to appropriate device (GPU/CPU)
            self.model = self.model.to(self.device)
            if hasattr(self, 'classifier'):
                self.classifier = self.classifier.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            self.model_loaded = True
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict(self, text: str, features: Dict = None) -> Dict:
        """
        Predict whether content is fake or real
        
        Args:
            text (str): Text content to analyze
            features (Dict): Optional extracted features for enhanced prediction
            
        Returns:
            Dict: Prediction results with probabilities and confidence
        """
        # Ensure model is loaded
        self.load_model()
        
        try:
            # Tokenize input text
            # This converts text into token IDs that BERT understands
            # padding=True ensures all sequences have same length
            # truncation=True cuts text that's too long
            # return_tensors='pt' returns PyTorch tensors
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=config.MAX_SEQUENCE_LENGTH,
                return_tensors='pt'
            )
            
            # Move inputs to same device as model
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            
            # Disable gradient calculation (we're not training)
            with torch.no_grad():
                # Get model predictions
                outputs = self.model(**inputs)
                
                # If using base model with separate classifier
                if hasattr(self, 'classifier'):
                    # Get the [CLS] token representation (first token)
                    # This represents the entire sequence
                    cls_output = outputs.last_hidden_state[:, 0, :]
                    logits = self.classifier(cls_output)
                else:
                    # If using fine-tuned model with built-in classifier
                    logits = outputs.logits
            
            # Apply softmax to get probabilities
            # Softmax converts logits to probabilities that sum to 1
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            
            # Get predicted class (highest probability)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            
            # Get confidence (probability of predicted class)
            confidence = probabilities[0][predicted_class].item()
            
            # Get probability for each class
            fake_probability = probabilities[0][1].item()
            real_probability = probabilities[0][0].item()
            
            # Determine prediction label
            prediction = self.labels[predicted_class]
            
            # Adjust prediction based on confidence threshold
            if confidence < config.CONFIDENCE_THRESHOLD:
                prediction = "UNCERTAIN"
            
            # Create result dictionary
            result = {
                'prediction': prediction,
                'confidence': confidence,
                'fake_probability': fake_probability,
                'real_probability': real_probability,
                'is_fake': predicted_class == 1,
                'is_real': predicted_class == 0,
                'model_used': self.model_name
            }
            
            logger.info(f"Prediction: {prediction} (confidence: {confidence:.2%})")
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return {
                'prediction': 'ERROR',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def predict_with_features(self, text: str, features: Dict) -> Dict:
        """
        Enhanced prediction that combines BERT with engineered features
        This provides more robust predictions by using multiple signals
        
        Args:
            text (str): Text content
            features (Dict): Extracted features from FeatureExtractor
            
        Returns:
            Dict: Prediction results
        """
        # Get base prediction from BERT
        base_prediction = self.predict(text)
        
        # Feature-based scoring (simple rule-based system)
        # In production, you'd train another model on these features
        feature_score = self._calculate_feature_score(features)
        
        # Combine predictions (weighted average)
        # 70% weight to BERT, 30% to features
        combined_fake_prob = (
            0.7 * base_prediction['fake_probability'] +
            0.3 * feature_score
        )
        
        # Update prediction based on combined score
        final_prediction = 'FAKE' if combined_fake_prob > 0.5 else 'REAL'
        
        if abs(combined_fake_prob - 0.5) < (1 - config.CONFIDENCE_THRESHOLD):
            final_prediction = 'UNCERTAIN'
        
        result = {
            'prediction': final_prediction,
            'confidence': max(combined_fake_prob, 1 - combined_fake_prob),
            'fake_probability': combined_fake_prob,
            'real_probability': 1 - combined_fake_prob,
            'bert_prediction': base_prediction['prediction'],
            'feature_score': feature_score,
            'model_used': 'Ensemble (BERT + Features)'
        }
        
        return result
    
    def _calculate_feature_score(self, features: Dict) -> float:
        """
        Calculate a fake news score based on extracted features
        This is a simplified rule-based system
        In production, you'd train a separate model (Random Forest, etc.)
        
        Args:
            features (Dict): Extracted features
            
        Returns:
            float: Score between 0 (likely real) and 1 (likely fake)
        """
        score = 0.5  # Start at neutral
        
        # Penalize for unreliable domain
        if features.get('is_unreliable_domain', 0):
            score += 0.3
        
        # Reward for trusted domain
        if features.get('is_trusted_domain', 0):
            score -= 0.3
        
        # Penalize for high clickbait indicators
        if features.get('clickbait_keyword_count', 0) > 2:
            score += 0.15
        
        # Penalize for excessive emotional language
        if features.get('emotional_word_ratio', 0) > 0.05:
            score += 0.1
        
        # Penalize for high subjectivity
        if features.get('text_subjectivity', 0) > 0.7:
            score += 0.1
        
        # Penalize for missing author
        if not features.get('has_author', 0):
            score += 0.05
        
        # Penalize for excessive capitalization
        if features.get('capital_ratio', 0) > 0.15:
            score += 0.1
        
        # Penalize for extreme readability (too easy or too hard)
        flesch_score = features.get('flesch_reading_ease', 50)
        if flesch_score < 30 or flesch_score > 90:
            score += 0.05
        
        # Ensure score stays in [0, 1] range
        score = max(0.0, min(1.0, score))
        
        return score
    
    def batch_predict(self, texts: List[str]) -> List[Dict]:
        """
        Predict multiple texts at once (more efficient)
        
        Args:
            texts (List[str]): List of texts to analyze
            
        Returns:
            List[Dict]: List of prediction results
        """
        self.load_model()
        
        results = []
        
        # Process in batches for efficiency
        batch_size = config.BATCH_SIZE
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=config.MAX_SEQUENCE_LENGTH,
                return_tensors='pt'
            )
            
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                if hasattr(self, 'classifier'):
                    cls_output = outputs.last_hidden_state[:, 0, :]
                    logits = self.classifier(cls_output)
                else:
                    logits = outputs.logits
            
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            
            # Process each item in batch
            for j in range(len(batch_texts)):
                predicted_class = torch.argmax(probabilities[j]).item()
                confidence = probabilities[j][predicted_class].item()
                
                results.append({
                    'prediction': self.labels[predicted_class],
                    'confidence': confidence,
                    'fake_probability': probabilities[j][1].item(),
                    'real_probability': probabilities[j][0].item()
                })
        
        return results
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model
        
        Returns:
            Dict: Model information
        """
        info = {
            'model_name': self.model_name,
            'device': str(self.device),
            'model_loaded': self.model_loaded,
            'max_sequence_length': config.MAX_SEQUENCE_LENGTH,
            'confidence_threshold': config.CONFIDENCE_THRESHOLD,
            'labels': self.labels
        }
        
        if self.model_loaded and self.model:
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            info['total_parameters'] = total_params
            info['trainable_parameters'] = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
        
        return info


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create detector instance
    detector = FakeNewsDetector()
    
    # Test texts
    real_text = """
    According to a report published by Reuters, scientists have made 
    significant progress in renewable energy technology. The research, 
    conducted at MIT and published in Nature journal, demonstrates 
    improved efficiency in solar panels.
    """
    
    fake_text = """
    SHOCKING: You won't believe what doctors discovered! This one weird 
    trick will change your life FOREVER! Big Pharma doesn't want you to 
    know this SECRET that will blow your mind!!!
    """
    
    print("=" * 60)
    print("Testing Fake News Detector")
    print("=" * 60)
    
    # Test with real news
    print("\n1. Testing with REAL news:")
    print(f"Text: {real_text[:100]}...")
    result = detector.predict(real_text)
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Fake Probability: {result['fake_probability']:.2%}")
    
    # Test with fake news
    print("\n2. Testing with FAKE news:")
    print(f"Text: {fake_text[:100]}...")
    result = detector.predict(fake_text)
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Fake Probability: {result['fake_probability']:.2%}")
    
    # Show model info
    print("\n3. Model Information:")
    info = detector.get_model_info()
    for key, value in info.items():
        print(f"{key}: {value}")
