"""
Configuration file for the Fake Content Detection System
This file contains all settings, paths, and hyperparameters used throughout the application
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS - Define all directory structures for the project
# ============================================================================

# Base directory of the project
BASE_DIR = Path(__file__).parent.absolute()

# Directory to store trained models
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Directory for cached results to avoid re-analyzing same URLs
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Directory for storing logs
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Directory for storing downloaded datasets
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# ============================================================================
# MODEL CONFIGURATION - Settings for machine learning models
# ============================================================================

# Pre-trained transformer model from Hugging Face
# Using DistilBERT for faster inference while maintaining good accuracy
MODEL_NAME = "distilbert-base-uncased"

# Alternative models you can try:
# MODEL_NAME = "bert-base-uncased"  # Standard BERT (slower but more accurate)
# MODEL_NAME = "roberta-base"  # RoBERTa (better performance on many tasks)

# Maximum sequence length for text input (BERT limit is 512)
MAX_SEQUENCE_LENGTH = 512

# Batch size for model inference
BATCH_SIZE = 8

# Confidence threshold for classification (0.0 to 1.0)
# Predictions below this threshold are marked as "uncertain"
CONFIDENCE_THRESHOLD = 0.7

# ============================================================================
# FEATURE EXTRACTION SETTINGS - Parameters for content analysis
# ============================================================================

# Keywords that indicate clickbait or sensational content
CLICKBAIT_KEYWORDS = [
    "shocking", "unbelievable", "you won't believe", "this will blow your mind",
    "doctors hate", "one weird trick", "what happens next", "gone wrong",
    "must see", "amazing", "incredible", "jaw-dropping"
]

# Emotional words that might indicate biased or fake content
EMOTIONAL_WORDS = [
    "outrage", "shocking", "devastating", "horrifying", "terrifying",
    "disgusting", "infuriating", "heartbreaking", "explosive", "bombshell"
]

# Trusted news domains (used for source credibility scoring)
TRUSTED_DOMAINS = [
    "reuters.com", "apnews.com", "bbc.com", "npr.org", "pbs.org",
    "nytimes.com", "washingtonpost.com", "theguardian.com", "economist.com"
]

# Known unreliable domains (you would expand this list)
UNRELIABLE_DOMAINS = [
    "beforeitsnews.com", "naturalnews.com", "infowars.com"
]

# ============================================================================
# WEB SCRAPING CONFIGURATION - Settings for content extraction
# ============================================================================

# User agent string to identify our scraper to web servers
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# Timeout for HTTP requests (seconds)
REQUEST_TIMEOUT = 10

# Maximum number of redirects to follow
MAX_REDIRECTS = 5

# Maximum content size to download (bytes) - 5MB limit
MAX_CONTENT_SIZE = 5 * 1024 * 1024

# ============================================================================
# API CONFIGURATION - Settings for the REST API
# ============================================================================

# API server host
API_HOST = "0.0.0.0"

# API server port
API_PORT = 8000

# Enable CORS for frontend access (set domains in production)
CORS_ORIGINS = ["*"]  # In production, specify exact origins

# Rate limiting (requests per minute per IP)
RATE_LIMIT = 30

# Cache expiration time (seconds) - 1 hour
CACHE_EXPIRATION = 3600

# ============================================================================
# EXPLAINABILITY SETTINGS - Parameters for generating explanations
# ============================================================================

# Number of features to show in explanation
TOP_FEATURES_COUNT = 5

# Minimum importance score for a feature to be included in explanation
MIN_FEATURE_IMPORTANCE = 0.1

# ============================================================================
# LOGGING CONFIGURATION - Settings for application logging
# ============================================================================

# Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL = "INFO"

# Log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ============================================================================
# EXTERNAL APIS (Optional - for enhanced functionality)
# ============================================================================

# Fact-checking API keys (you would need to obtain these)
# FACTCHECK_API_KEY = os.getenv("FACTCHECK_API_KEY", "")
# GOOGLE_FACT_CHECK_API_KEY = os.getenv("GOOGLE_FACT_CHECK_API_KEY", "")

# Social media API keys (for analyzing social spread patterns)
# TWITTER_API_KEY = os.getenv("TWITTER_API_KEY", "")
# TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET", "")
