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
        
        # Special rule: Trusted domains with good metadata get strong boost
        # This helps overcome untrained BERT model's uncertainty
        is_trusted = features.get('is_trusted_domain', 0)
        has_author = features.get('has_author', 0)
        has_date = features.get('has_publish_date', 0)
        word_count = features.get('word_count', 0)
        
        # Trusted source with proper journalism standards
        if is_trusted and has_author and has_date:
            # Give very strong credibility boost
            result = {
                'prediction': 'REAL',
                'confidence': 0.90,  # Very high confidence for trusted sources
                'fake_probability': 0.10,
                'real_probability': 0.90,
                'bert_prediction': base_prediction['prediction'],
                'feature_score': 0.0,
                'model_used': 'Ensemble (BERT + Features) - Trusted Source Override'
            }
            logger.info(f"Trusted source detected with full metadata - classified as REAL")
            return result
        
        # Trusted source with some metadata (slightly less confident)
        if is_trusted and (has_author or has_date) and word_count > 200:
            result = {
                'prediction': 'REAL',
                'confidence': 0.82,  # High confidence
                'fake_probability': 0.18,
                'real_probability': 0.82,
                'bert_prediction': base_prediction['prediction'],
                'feature_score': 0.0,
                'model_used': 'Ensemble (BERT + Features) - Trusted Source'
            }
            logger.info(f"Trusted source detected - classified as REAL")
            return result
        
        # Feature-based scoring (simple rule-based system)
        # In production, you'd train another model on these features
        feature_score = self._calculate_feature_score(features)
        
        # Combine predictions (weighted average)
        # 40% weight to BERT, 60% to features (features dominate since model not fine-tuned)
        # When model is fine-tuned, increase BERT weight back to 70-80%
        combined_fake_prob = (
            0.4 * base_prediction['fake_probability'] +
            0.6 * feature_score
        )
        
        # Update prediction based on combined score
        final_prediction = 'FAKE' if combined_fake_prob > 0.5 else 'REAL'
        final_confidence = max(combined_fake_prob, 1 - combined_fake_prob)
        
        # Mark as uncertain only if very close to 0.5
        if final_confidence < config.CONFIDENCE_THRESHOLD:
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
        
        # === DOMAIN CREDIBILITY (Most Important) ===
        # Strong penalty for unreliable domain
        if features.get('is_unreliable_domain', 0):
            score += 0.4
        
        # Strong reward for trusted domain (BBC, Reuters, etc.)
        if features.get('is_trusted_domain', 0):
            score -= 0.55  # Strong indicator of credibility
        
        # === PROPER JOURNALISM INDICATORS ===
        # Reward for having author (proper attribution)
        if features.get('has_author', 0):
            score -= 0.15
        
        # Reward for having publish date (transparency)
        if features.get('has_publish_date', 0):
            score -= 0.1
        
        # Reward for presence of quotes (indicates reporting)
        quote_count = features.get('quote_count', 0)
        if quote_count >= 3:
            score -= 0.1  # Good journalism includes multiple sources
        elif quote_count >= 1:
            score -= 0.05
        
        # Reward for common TLD (.com, .org, etc.)
        if features.get('has_common_tld', 0):
            score -= 0.03
        
        # === CONTENT QUALITY INDICATORS ===
        # Penalize for clickbait (check both title and patterns)
        clickbait_count = features.get('clickbait_keyword_count', 0)
        clickbait_patterns = features.get('clickbait_pattern_count', 0)
        
        if clickbait_count > 2 or clickbait_patterns > 1:  # Multiple strong indicators
            score += 0.25  # Strong penalty
        elif clickbait_count > 0 or clickbait_patterns > 0:
            score += 0.10  # Mild penalty
        
        # Strong penalty for extreme emotional language
        emotional_ratio = features.get('emotional_word_ratio', 0)
        if emotional_ratio > 0.1:  # Very high emotional content
            score += 0.20
        elif emotional_ratio > 0.06:
            score += 0.10
        
        # Penalize for ALL CAPS in title
        title_caps_ratio = features.get('title_caps_ratio', 0)
        if title_caps_ratio > 0.5:  # More than half the title is caps
            score += 0.15
        
        # Reward for balanced tone (not too subjective)
        subjectivity = features.get('text_subjectivity', 0)
        if 0.3 <= subjectivity <= 0.6:  # Balanced reporting
            score -= 0.05
        elif subjectivity > 0.8:  # Very subjective/opinion piece
            score += 0.08
        
        # === WRITING QUALITY ===
        # Strong penalty for excessive capitalization (SHOUTING)
        capital_ratio = features.get('capital_ratio', 0)
        if capital_ratio > 0.2:  # Very high caps
            score += 0.15
        elif capital_ratio > 0.15:
            score += 0.08
        
        # Penalize for missing both author AND date (low transparency)
        has_author = features.get('has_author', 0)
        has_date = features.get('has_publish_date', 0)
        if not has_author and not has_date:
            score += 0.12  # Missing both is suspicious
        
        # Penalize for short, low-quality content
        word_count = features.get('word_count', 0)
        if word_count < 100:  # Very short
            score += 0.10
        
        # Reward for appropriate readability (news articles typically 40-70)
        flesch_score = features.get('flesch_reading_ease', 50)
        if 40 <= flesch_score <= 70:  # Good news writing level
            score -= 0.05
        elif flesch_score < 20 or flesch_score > 90:  # Extreme values
            score += 0.08
        
        # Reward for reasonable article length (not too short)
        word_count = features.get('word_count', 0)
        if word_count >= 300:  # Substantial article
            score -= 0.05
        elif word_count < 100:  # Too short might be low quality
            score += 0.05
        
        # === STRUCTURAL QUALITY ===
        # Reward for proper paragraph structure
        paragraph_count = features.get('paragraph_count', 0)
        if paragraph_count >= 5:  # Well-structured article
            score -= 0.03
        
        # Penalize for excessive exclamation marks
        exclamation_ratio = features.get('exclamation_ratio', 0)
        if exclamation_ratio > 0.02:  # Increased threshold
            score += 0.08
        
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
