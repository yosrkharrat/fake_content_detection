"""
Feature Extractor Module
This module analyzes content and extracts features used for fake news detection
Features include: linguistic patterns, sentiment, readability, source credibility, etc.
"""

import re
import math
from collections import Counter
from typing import Dict, List, Tuple
import logging
from urllib.parse import urlparse

# NLP libraries for text analysis
import nltk
from textblob import TextBlob

# Importing project configuration
import config

# Setting up logging
logger = logging.getLogger(__name__)

# Downloading required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)


class FeatureExtractor:
    """
    Extracts various features from content that are indicative of fake news
    These features are used as input to machine learning models
    """
    
    def __init__(self):
        """
        Initialize the feature extractor with required resources
        """
        # Load English stopwords (common words like 'the', 'is', 'at')
        from nltk.corpus import stopwords
        self.stopwords = set(stopwords.words('english'))
        
        # Compile regex patterns for efficiency
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.exclamation_pattern = re.compile(r'!')
        self.question_pattern = re.compile(r'\?')
        self.caps_pattern = re.compile(r'[A-Z]')
        
        logger.info("FeatureExtractor initialized successfully")
    
    def extract_all_features(self, content: Dict) -> Dict:
        """
        Extract all features from content
        This is the main method that calls all other feature extraction methods
        
        Args:
            content (Dict): Dictionary containing scraped content
            
        Returns:
            Dict: Dictionary containing all extracted features
        """
        logger.info("Extracting features from content")
        
        # Get text and title from content
        text = content.get('text', '')
        title = content.get('title', '')
        domain = content.get('domain', '')
        
        # Initialize features dictionary
        features = {}
        
        # Extract different categories of features
        features.update(self._extract_linguistic_features(text, title))
        features.update(self._extract_sentiment_features(text, title))
        features.update(self._extract_readability_features(text))
        features.update(self._extract_structural_features(text))
        features.update(self._extract_source_credibility_features(content))
        features.update(self._extract_clickbait_features(title, text))
        features.update(self._extract_metadata_features(content))
        
        logger.info(f"Extracted {len(features)} features")
        return features
    
    def _extract_linguistic_features(self, text: str, title: str) -> Dict:
        """
        Extract linguistic and textual features from content
        These features capture writing style and patterns
        
        Args:
            text (str): Main body text
            title (str): Title of the article
            
        Returns:
            Dict: Linguistic features
        """
        features = {}
        
        # Basic text statistics
        features['text_length'] = len(text)  # Total character count
        features['title_length'] = len(title)  # Title character count
        
        # Word-level statistics
        words = text.split()
        features['word_count'] = len(words)  # Total word count
        features['avg_word_length'] = sum(len(word) for word in words) / len(words) if words else 0
        
        # Sentence-level statistics
        sentences = nltk.sent_tokenize(text) if text else []
        features['sentence_count'] = len(sentences)
        features['avg_sentence_length'] = len(words) / len(sentences) if sentences else 0
        
        # Punctuation usage (can indicate emotional or sensational writing)
        features['exclamation_count'] = len(self.exclamation_pattern.findall(text))
        features['question_count'] = len(self.question_pattern.findall(text))
        features['exclamation_ratio'] = features['exclamation_count'] / len(text) if text else 0
        
        # Capitalization (excessive caps can indicate sensationalism)
        caps_count = len(self.caps_pattern.findall(text))
        features['capital_ratio'] = caps_count / len(text) if text else 0
        
        # Title capitalization (titles in ALL CAPS are often clickbait)
        title_words = title.split()
        if title_words:
            caps_title_words = sum(1 for word in title_words if word.isupper() and len(word) > 1)
            features['title_caps_ratio'] = caps_title_words / len(title_words)
        else:
            features['title_caps_ratio'] = 0
        
        # Unique word ratio (vocabulary diversity)
        unique_words = set(word.lower() for word in words)
        features['unique_word_ratio'] = len(unique_words) / len(words) if words else 0
        
        # Stopword ratio (high ratio might indicate poor quality writing)
        stopword_count = sum(1 for word in words if word.lower() in self.stopwords)
        features['stopword_ratio'] = stopword_count / len(words) if words else 0
        
        return features
    
    def _extract_sentiment_features(self, text: str, title: str) -> Dict:
        """
        Extract sentiment and emotional features
        Fake news often uses emotional language to manipulate readers
        
        Args:
            text (str): Main body text
            title (str): Title of the article
            
        Returns:
            Dict: Sentiment features
        """
        features = {}
        
        # Analyze sentiment using TextBlob
        # Polarity: -1 (negative) to +1 (positive)
        # Subjectivity: 0 (objective) to 1 (subjective)
        
        if text:
            text_blob = TextBlob(text)
            features['text_polarity'] = text_blob.sentiment.polarity
            features['text_subjectivity'] = text_blob.sentiment.subjectivity
        else:
            features['text_polarity'] = 0.0
            features['text_subjectivity'] = 0.0
        
        if title:
            title_blob = TextBlob(title)
            features['title_polarity'] = title_blob.sentiment.polarity
            features['title_subjectivity'] = title_blob.sentiment.subjectivity
        else:
            features['title_polarity'] = 0.0
            features['title_subjectivity'] = 0.0
        
        # Check for emotional keywords (defined in config)
        text_lower = text.lower()
        emotional_word_count = sum(1 for word in config.EMOTIONAL_WORDS if word in text_lower)
        features['emotional_word_count'] = emotional_word_count
        features['emotional_word_ratio'] = emotional_word_count / len(text.split()) if text else 0
        
        return features
    
    def _extract_readability_features(self, text: str) -> Dict:
        """
        Extract readability metrics
        Fake news often has poor grammar or overly complex/simple writing
        
        Args:
            text (str): Main body text
            
        Returns:
            Dict: Readability features
        """
        features = {}
        
        if not text:
            return {
                'flesch_reading_ease': 0,
                'flesch_kincaid_grade': 0,
                'avg_syllables_per_word': 0
            }
        
        words = text.split()
        sentences = nltk.sent_tokenize(text)
        
        if not words or not sentences:
            return {
                'flesch_reading_ease': 0,
                'flesch_kincaid_grade': 0,
                'avg_syllables_per_word': 0
            }
        
        # Count syllables (simplified approximation)
        total_syllables = sum(self._count_syllables(word) for word in words)
        
        # Average syllables per word
        features['avg_syllables_per_word'] = total_syllables / len(words)
        
        # Flesch Reading Ease Score
        # Higher score = easier to read (90-100: very easy, 0-30: very difficult)
        # Formula: 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)
        features['flesch_reading_ease'] = (
            206.835 
            - 1.015 * (len(words) / len(sentences))
            - 84.6 * (total_syllables / len(words))
        )
        
        # Flesch-Kincaid Grade Level
        # Indicates U.S. school grade level needed to understand the text
        # Formula: 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59
        features['flesch_kincaid_grade'] = (
            0.39 * (len(words) / len(sentences))
            + 11.8 * (total_syllables / len(words))
            - 15.59
        )
        
        return features
    
    def _extract_structural_features(self, text: str) -> Dict:
        """
        Extract structural features of the content
        These relate to the organization and formatting of the article
        
        Args:
            text (str): Main body text
            
        Returns:
            Dict: Structural features
        """
        features = {}
        
        # Count URLs in text (excessive links might indicate spam)
        url_matches = self.url_pattern.findall(text)
        features['url_count'] = len(url_matches)
        features['url_ratio'] = len(url_matches) / len(text.split()) if text else 0
        
        # Count quotes (legitimate news often includes quoted sources)
        quote_count = text.count('"') + text.count('"') + text.count('"')
        features['quote_count'] = quote_count // 2  # Divide by 2 for opening/closing pairs
        
        # Paragraph count (estimated by double newlines)
        paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
        features['paragraph_count'] = max(1, paragraph_count)
        
        # Average paragraph length
        if paragraph_count > 0:
            features['avg_paragraph_length'] = len(text.split()) / paragraph_count
        else:
            features['avg_paragraph_length'] = 0
        
        return features
    
    def _extract_source_credibility_features(self, content: Dict) -> Dict:
        """
        Extract features related to source credibility
        Domain reputation, author information, publication date, etc.
        
        Args:
            content (Dict): Content dictionary with metadata
            
        Returns:
            Dict: Source credibility features
        """
        features = {}
        
        domain = content.get('domain', '').lower()
        
        # Check if domain is in trusted list
        features['is_trusted_domain'] = 1 if domain in config.TRUSTED_DOMAINS else 0
        
        # Check if domain is in unreliable list
        features['is_unreliable_domain'] = 1 if domain in config.UNRELIABLE_DOMAINS else 0
        
        # Domain age/reputation indicators (simplified)
        # In production, you'd query domain age databases
        common_tlds = ['.com', '.org', '.net', '.gov', '.edu']
        features['has_common_tld'] = 1 if any(domain.endswith(tld) for tld in common_tlds) else 0
        
        # Author information presence (articles with authors are generally more credible)
        authors = content.get('authors', [])
        features['has_author'] = 1 if authors and len(authors) > 0 else 0
        features['author_count'] = len(authors) if authors else 0
        
        # Publication date presence (legitimate news has clear timestamps)
        features['has_publish_date'] = 1 if content.get('publish_date') else 0
        
        # Images present (though not always indicative)
        images = content.get('images', [])
        features['image_count'] = len(images) if images else 0
        features['has_images'] = 1 if images else 0
        
        return features
    
    def _extract_clickbait_features(self, title: str, text: str) -> Dict:
        """
        Extract features that indicate clickbait or sensational content
        Clickbait often correlates with fake or misleading content
        
        Args:
            title (str): Article title
            text (str): Main body text
            
        Returns:
            Dict: Clickbait features
        """
        features = {}
        
        title_lower = title.lower()
        
        # Count clickbait keywords in title
        clickbait_count = sum(1 for keyword in config.CLICKBAIT_KEYWORDS if keyword in title_lower)
        features['clickbait_keyword_count'] = clickbait_count
        features['has_clickbait'] = 1 if clickbait_count > 0 else 0
        
        # Check for numbers in title (e.g., "10 shocking facts")
        features['title_has_numbers'] = 1 if re.search(r'\d+', title) else 0
        
        # Check for common clickbait patterns
        clickbait_patterns = [
            r'you won\'t believe',
            r'what happens next',
            r'doctors hate',
            r'number \d+',
            r'\d+ (reasons|ways|things|facts)',
            r'will shock you',
            r'this is why',
        ]
        
        pattern_matches = sum(1 for pattern in clickbait_patterns if re.search(pattern, title_lower))
        features['clickbait_pattern_count'] = pattern_matches
        
        # Check if title ends with ellipsis or multiple punctuation (common in clickbait)
        features['title_trailing_punctuation'] = 1 if re.search(r'[.!?]{2,}$|…$', title) else 0
        
        return features
    
    def _extract_metadata_features(self, content: Dict) -> Dict:
        """
        Extract features from metadata
        These are additional signals about content quality
        
        Args:
            content (Dict): Content dictionary
            
        Returns:
            Dict: Metadata features
        """
        features = {}
        
        # Meta description presence and length
        meta_desc = content.get('meta_description', '')
        features['has_meta_description'] = 1 if meta_desc else 0
        features['meta_description_length'] = len(meta_desc)
        
        # Keywords presence
        keywords = content.get('keywords', [])
        features['has_keywords'] = 1 if keywords else 0
        features['keyword_count'] = len(keywords) if keywords else 0
        
        # Summary availability (from newspaper3k)
        summary = content.get('summary', '')
        features['has_summary'] = 1 if summary else 0
        
        return features
    
    def _count_syllables(self, word: str) -> int:
        """
        Count syllables in a word (simplified heuristic)
        Used for readability calculations
        
        Args:
            word (str): Word to count syllables in
            
        Returns:
            int: Estimated syllable count
        """
        word = word.lower()
        
        # Remove non-alphabetic characters
        word = re.sub(r'[^a-z]', '', word)
        
        if len(word) <= 3:
            return 1
        
        # Count vowel groups
        vowels = "aeiouy"
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Adjust for silent 'e' at the end
        if word.endswith('e'):
            syllable_count -= 1
        
        # Every word has at least one syllable
        return max(1, syllable_count)
    
    def get_feature_summary(self, features: Dict) -> str:
        """
        Generate a human-readable summary of key features
        Useful for debugging and explanation
        
        Args:
            features (Dict): Extracted features
            
        Returns:
            str: Summary string
        """
        summary = []
        
        summary.append(f"Text length: {features.get('word_count', 0)} words")
        summary.append(f"Sentiment: {'Positive' if features.get('text_polarity', 0) > 0 else 'Negative'}")
        summary.append(f"Subjectivity: {features.get('text_subjectivity', 0):.2f}")
        summary.append(f"Readability: {features.get('flesch_reading_ease', 0):.1f}")
        summary.append(f"Clickbait indicators: {features.get('clickbait_keyword_count', 0)}")
        summary.append(f"Trusted domain: {'Yes' if features.get('is_trusted_domain', 0) else 'No'}")
        
        return " | ".join(summary)


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create feature extractor
    extractor = FeatureExtractor()
    
    # Test with sample content
    sample_content = {
        'text': "This is shocking news that will blow your mind! You won't believe what happened next. Click here to find out more!",
        'title': "SHOCKING: 10 Things Doctors Don't Want You To Know!",
        'domain': 'fakenews.com',
        'authors': [],
        'publish_date': None
    }
    
    # Extract features
    features = extractor.extract_all_features(sample_content)
    
    print("Extracted Features:")
    for key, value in features.items():
        print(f"{key}: {value}")
    
    print("\nFeature Summary:")
    print(extractor.get_feature_summary(features))
