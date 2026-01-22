"""
Explainer Module
This module generates human-readable explanations for model predictions
Helps users understand WHY content was classified as fake or real
"""

import logging
from typing import Dict, List, Tuple
import numpy as np

# Import project configuration
import config

# Set up logging
logger = logging.getLogger(__name__)


class PredictionExplainer:
    """
    Generates explanations for fake news detection predictions
    Combines multiple factors into clear, actionable explanations
    """
    
    def __init__(self):
        """
        Initialize the explainer with explanation templates
        """
        # Define explanation templates for different factors
        # These explain what each feature means in simple terms
        self.explanation_templates = {
            # Source credibility explanations
            'is_trusted_domain': "✓ Content is from a trusted news source",
            'is_unreliable_domain': "⚠ Content is from a known unreliable source",
            'has_author': "✓ Article has identified author(s)",
            'no_author': "⚠ No author information provided",
            'has_publish_date': "✓ Clear publication date provided",
            'no_publish_date': "⚠ Missing publication date",
            
            # Content quality explanations
            'high_clickbait': "⚠ Title contains clickbait language ('{keywords}')",
            'emotional_language': "⚠ Excessive use of emotional language",
            'poor_grammar': "⚠ Unusual writing quality or grammar issues",
            'sensational_caps': "⚠ Excessive capitalization indicates sensationalism",
            
            # Sentiment explanations
            'extreme_sentiment': "⚠ Content shows extreme emotional bias",
            'high_subjectivity': "⚠ Content is highly subjective rather than factual",
            'balanced_tone': "✓ Content maintains a balanced, objective tone",
            
            # Structural explanations
            'lacks_sources': "⚠ Article lacks proper citations or sources",
            'has_quotes': "✓ Article includes quotes from sources",
            'suspicious_urls': "⚠ Contains suspicious or excessive URLs",
            
            # Readability explanations
            'readability_good': "✓ Writing quality is appropriate for news content",
            'readability_poor': "⚠ Unusual readability level for news content",
        }
        
        logger.info("PredictionExplainer initialized successfully")
    
    def generate_explanation(self, prediction: Dict, features: Dict, content: Dict) -> Dict:
        """
        Generate comprehensive explanation for a prediction
        
        Args:
            prediction (Dict): Model prediction results
            features (Dict): Extracted features from content
            content (Dict): Original scraped content
            
        Returns:
            Dict: Detailed explanation with factors and recommendations
        """
        logger.info("Generating explanation for prediction")
        
        # Determine if content was predicted as fake
        is_fake = prediction.get('is_fake', False) or prediction.get('prediction') == 'FAKE'
        
        # Collect all factors that influenced the decision
        positive_factors = []  # Factors supporting credibility
        negative_factors = []  # Factors indicating fake content
        
        # Analyze source credibility
        source_factors = self._explain_source_credibility(features, content)
        positive_factors.extend(source_factors['positive'])
        negative_factors.extend(source_factors['negative'])
        
        # Analyze content quality
        content_factors = self._explain_content_quality(features, content)
        positive_factors.extend(content_factors['positive'])
        negative_factors.extend(content_factors['negative'])
        
        # Analyze sentiment and tone
        sentiment_factors = self._explain_sentiment(features)
        positive_factors.extend(sentiment_factors['positive'])
        negative_factors.extend(sentiment_factors['negative'])
        
        # Analyze clickbait indicators
        clickbait_factors = self._explain_clickbait(features, content)
        negative_factors.extend(clickbait_factors)
        
        # Calculate credibility score based on factors
        credibility_score = self._calculate_credibility_score(
            positive_factors, negative_factors
        )
        
        # Generate overall summary
        summary = self._generate_summary(
            prediction, credibility_score, is_fake
        )
        
        # Create detailed explanation
        explanation = {
            'summary': summary,
            'prediction': prediction.get('prediction'),
            'confidence': prediction.get('confidence', 0),
            'credibility_score': credibility_score,
            'positive_factors': positive_factors,
            'negative_factors': negative_factors,
            'factor_count': {
                'positive': len(positive_factors),
                'negative': len(negative_factors)
            },
            'key_concerns': self._get_key_concerns(negative_factors),
            'recommendations': self._generate_recommendations(negative_factors, content)
        }
        
        logger.info(f"Explanation generated: {len(positive_factors)} positive, {len(negative_factors)} negative factors")
        return explanation
    
    def _explain_source_credibility(self, features: Dict, content: Dict) -> Dict:
        """
        Generate explanations related to source credibility
        
        Args:
            features (Dict): Extracted features
            content (Dict): Original content
            
        Returns:
            Dict: Positive and negative factors
        """
        positive = []
        negative = []
        
        # Check domain reputation
        if features.get('is_trusted_domain', 0):
            positive.append({
                'factor': 'Trusted Domain',
                'description': self.explanation_templates['is_trusted_domain'],
                'importance': 'high',
                'icon': '✓'
            })
        elif features.get('is_unreliable_domain', 0):
            negative.append({
                'factor': 'Unreliable Domain',
                'description': self.explanation_templates['is_unreliable_domain'],
                'importance': 'critical',
                'icon': '⚠'
            })
        
        # Check author information
        if features.get('has_author', 0):
            authors = content.get('authors', [])
            author_text = ', '.join(authors[:2]) if authors else 'Unknown'
            positive.append({
                'factor': 'Author Present',
                'description': f"✓ Written by: {author_text}",
                'importance': 'medium',
                'icon': '✓'
            })
        else:
            negative.append({
                'factor': 'No Author',
                'description': self.explanation_templates['no_author'],
                'importance': 'medium',
                'icon': '⚠'
            })
        
        # Check publication date
        if features.get('has_publish_date', 0):
            positive.append({
                'factor': 'Publication Date',
                'description': self.explanation_templates['has_publish_date'],
                'importance': 'low',
                'icon': '✓'
            })
        else:
            negative.append({
                'factor': 'Missing Date',
                'description': self.explanation_templates['no_publish_date'],
                'importance': 'low',
                'icon': '⚠'
            })
        
        return {'positive': positive, 'negative': negative}
    
    def _explain_content_quality(self, features: Dict, content: Dict) -> Dict:
        """
        Generate explanations related to content quality
        
        Args:
            features (Dict): Extracted features
            content (Dict): Original content
            
        Returns:
            Dict: Positive and negative factors
        """
        positive = []
        negative = []
        
        # Check for source citations (quotes)
        if features.get('quote_count', 0) > 2:
            positive.append({
                'factor': 'Citations Present',
                'description': f"✓ Article includes {features['quote_count']} quoted sources",
                'importance': 'medium',
                'icon': '✓'
            })
        elif features.get('word_count', 0) > 100:
            negative.append({
                'factor': 'Lacks Citations',
                'description': self.explanation_templates['lacks_sources'],
                'importance': 'medium',
                'icon': '⚠'
            })
        
        # Check readability
        flesch_score = features.get('flesch_reading_ease', 50)
        if 30 <= flesch_score <= 80:
            positive.append({
                'factor': 'Appropriate Readability',
                'description': self.explanation_templates['readability_good'],
                'importance': 'low',
                'icon': '✓'
            })
        else:
            negative.append({
                'factor': 'Unusual Readability',
                'description': self.explanation_templates['readability_poor'],
                'importance': 'low',
                'icon': '⚠'
            })
        
        # Check capitalization
        if features.get('capital_ratio', 0) > 0.15:
            negative.append({
                'factor': 'Excessive Capitalization',
                'description': self.explanation_templates['sensational_caps'],
                'importance': 'medium',
                'icon': '⚠'
            })
        
        # Check URL count
        if features.get('url_ratio', 0) > 0.05:
            negative.append({
                'factor': 'Excessive URLs',
                'description': self.explanation_templates['suspicious_urls'],
                'importance': 'low',
                'icon': '⚠'
            })
        
        return {'positive': positive, 'negative': negative}
    
    def _explain_sentiment(self, features: Dict) -> Dict:
        """
        Generate explanations related to sentiment and tone
        
        Args:
            features (Dict): Extracted features
            
        Returns:
            Dict: Positive and negative factors
        """
        positive = []
        negative = []
        
        # Check subjectivity
        subjectivity = features.get('text_subjectivity', 0)
        if subjectivity < 0.5:
            positive.append({
                'factor': 'Objective Tone',
                'description': self.explanation_templates['balanced_tone'],
                'importance': 'medium',
                'icon': '✓'
            })
        elif subjectivity > 0.7:
            negative.append({
                'factor': 'High Subjectivity',
                'description': self.explanation_templates['high_subjectivity'],
                'importance': 'medium',
                'icon': '⚠'
            })
        
        # Check emotional language
        if features.get('emotional_word_ratio', 0) > 0.03:
            negative.append({
                'factor': 'Emotional Language',
                'description': self.explanation_templates['emotional_language'],
                'importance': 'medium',
                'icon': '⚠'
            })
        
        # Check sentiment polarity
        polarity = features.get('text_polarity', 0)
        if abs(polarity) > 0.5:
            negative.append({
                'factor': 'Extreme Sentiment',
                'description': self.explanation_templates['extreme_sentiment'],
                'importance': 'low',
                'icon': '⚠'
            })
        
        return {'positive': positive, 'negative': negative}
    
    def _explain_clickbait(self, features: Dict, content: Dict) -> List[Dict]:
        """
        Generate explanations for clickbait indicators
        
        Args:
            features (Dict): Extracted features
            content (Dict): Original content
            
        Returns:
            List[Dict]: List of negative factors
        """
        negative = []
        
        # Check clickbait keywords
        clickbait_count = features.get('clickbait_keyword_count', 0)
        if clickbait_count > 0:
            title = content.get('title', '')
            # Find which keywords were matched
            found_keywords = [kw for kw in config.CLICKBAIT_KEYWORDS if kw in title.lower()]
            keyword_str = ', '.join(found_keywords[:3])
            
            negative.append({
                'factor': 'Clickbait Language',
                'description': self.explanation_templates['high_clickbait'].format(
                    keywords=keyword_str
                ),
                'importance': 'high',
                'icon': '⚠'
            })
        
        # Check for excessive punctuation
        if features.get('exclamation_ratio', 0) > 0.01:
            negative.append({
                'factor': 'Excessive Punctuation',
                'description': f"⚠ Title/text contains {features['exclamation_count']} exclamation marks",
                'importance': 'medium',
                'icon': '⚠'
            })
        
        return negative
    
    def _calculate_credibility_score(self, positive_factors: List[Dict], 
                                     negative_factors: List[Dict]) -> int:
        """
        Calculate overall credibility score (0-100)
        
        Args:
            positive_factors: List of positive factors
            negative_factors: List of negative factors
            
        Returns:
            int: Credibility score (0-100)
        """
        # Define importance weights
        importance_weights = {
            'critical': 30,
            'high': 20,
            'medium': 10,
            'low': 5
        }
        
        # Start at 50 (neutral)
        score = 50
        
        # Add points for positive factors
        for factor in positive_factors:
            weight = importance_weights.get(factor.get('importance', 'low'), 5)
            score += weight
        
        # Subtract points for negative factors
        for factor in negative_factors:
            weight = importance_weights.get(factor.get('importance', 'low'), 5)
            score -= weight
        
        # Clamp to 0-100 range
        score = max(0, min(100, score))
        
        return score
    
    def _generate_summary(self, prediction: Dict, credibility_score: int, 
                         is_fake: bool) -> str:
        """
        Generate overall summary text
        
        Args:
            prediction (Dict): Model prediction
            credibility_score (int): Calculated credibility score
            is_fake (bool): Whether content is predicted as fake
            
        Returns:
            str: Summary text
        """
        confidence = prediction.get('confidence', 0) * 100
        
        if prediction.get('prediction') == 'UNCERTAIN':
            return (
                f"⚠ UNCERTAIN - Unable to make a confident determination. "
                f"Credibility score: {credibility_score}/100. "
                f"Please verify information with multiple trusted sources."
            )
        elif is_fake:
            if credibility_score < 30:
                return (
                    f"🚫 LIKELY FAKE - Multiple red flags detected. "
                    f"Credibility score: {credibility_score}/100. "
                    f"Strong evidence suggests this content is unreliable."
                )
            else:
                return (
                    f"⚠ POTENTIALLY FAKE - Some concerns identified. "
                    f"Credibility score: {credibility_score}/100. "
                    f"Approach with skepticism and verify claims."
                )
        else:
            if credibility_score > 70:
                return (
                    f"✓ LIKELY REAL - Content appears credible. "
                    f"Credibility score: {credibility_score}/100. "
                    f"Multiple indicators suggest this is reliable content."
                )
            else:
                return (
                    f"✓ POSSIBLY REAL - Some credibility indicators present. "
                    f"Credibility score: {credibility_score}/100. "
                    f"Content shows signs of legitimacy but verify important claims."
                )
    
    def _get_key_concerns(self, negative_factors: List[Dict]) -> List[str]:
        """
        Extract the most important concerns
        
        Args:
            negative_factors: List of negative factors
            
        Returns:
            List[str]: Top concerns
        """
        # Sort by importance
        importance_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        
        sorted_factors = sorted(
            negative_factors,
            key=lambda x: importance_order.get(x.get('importance', 'low'), 3)
        )
        
        # Return top 5 concerns
        return [f"{factor['icon']} {factor['factor']}" for factor in sorted_factors[:5]]
    
    def _generate_recommendations(self, negative_factors: List[Dict], 
                                  content: Dict) -> List[str]:
        """
        Generate actionable recommendations for users
        
        Args:
            negative_factors: List of negative factors
            content: Original content
            
        Returns:
            List[str]: Recommendations
        """
        recommendations = []
        
        # Always recommend verification
        recommendations.append(
            "Cross-reference this information with established news sources"
        )
        
        # Specific recommendations based on factors
        factor_types = [f['factor'] for f in negative_factors]
        
        if 'Unreliable Domain' in factor_types:
            recommendations.append(
                "This source has a history of unreliable reporting - seek alternative sources"
            )
        
        if 'No Author' in factor_types:
            recommendations.append(
                "Look for articles on this topic from sources that clearly identify their authors"
            )
        
        if 'Clickbait Language' in factor_types:
            recommendations.append(
                "Be skeptical of sensational claims - look for evidence-based reporting"
            )
        
        if 'Lacks Citations' in factor_types:
            recommendations.append(
                "Verify key claims by finding primary sources or expert opinions"
            )
        
        # Add general advice
        if len(negative_factors) > 3:
            recommendations.append(
                "Multiple concerns identified - exercise extra caution with this content"
            )
        
        return recommendations


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create explainer
    explainer = PredictionExplainer()
    
    # Sample prediction
    sample_prediction = {
        'prediction': 'FAKE',
        'confidence': 0.85,
        'is_fake': True,
        'fake_probability': 0.85
    }
    
    # Sample features
    sample_features = {
        'is_unreliable_domain': 1,
        'has_author': 0,
        'clickbait_keyword_count': 3,
        'text_subjectivity': 0.8,
        'emotional_word_ratio': 0.05,
        'exclamation_count': 5,
        'quote_count': 0,
        'capital_ratio': 0.2
    }
    
    # Sample content
    sample_content = {
        'title': "SHOCKING: You Won't Believe What Happened Next!",
        'domain': 'fakenews.com',
        'authors': []
    }
    
    # Generate explanation
    explanation = explainer.generate_explanation(
        sample_prediction, sample_features, sample_content
    )
    
    print("=" * 60)
    print("EXPLANATION RESULTS")
    print("=" * 60)
    print(f"\nSummary: {explanation['summary']}")
    print(f"\nCredibility Score: {explanation['credibility_score']}/100")
    print(f"\nKey Concerns:")
    for concern in explanation['key_concerns']:
        print(f"  {concern}")
    print(f"\nRecommendations:")
    for i, rec in enumerate(explanation['recommendations'], 1):
        print(f"  {i}. {rec}")
