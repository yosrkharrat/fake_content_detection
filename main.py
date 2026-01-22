"""
FastAPI Main Application
This is the entry point for the Fake Content Detection API
Provides REST endpoints for analyzing URLs and text content
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, HttpUrl, Field
from typing import Optional, Dict
import logging
from datetime import datetime
import hashlib
import json
from pathlib import Path

# Import our custom modules
from content_scraper import ContentScraper
from feature_extractor import FeatureExtractor
from model import FakeNewsDetector
from explainer import PredictionExplainer
import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.LOGS_DIR / f"api_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# INITIALIZE FASTAPI APPLICATION
# ============================================================================

# Create FastAPI application instance
app = FastAPI(
    title="Fake Content Detection API",
    description="API for detecting fake news and misinformation in online content",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI documentation
    redoc_url="/redoc"  # ReDoc documentation
)

# ============================================================================
# CONFIGURE CORS (Cross-Origin Resource Sharing)
# Allows frontend to make requests from different domain
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,  # Which domains can access
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# ============================================================================
# INITIALIZE COMPONENTS
# These are created once when the server starts and reused for all requests
# ============================================================================

# Create instances of all our modules
scraper = ContentScraper()
feature_extractor = FeatureExtractor()
detector = FakeNewsDetector()
explainer = PredictionExplainer()

# Simple in-memory cache (in production, use Redis)
# Stores results to avoid re-analyzing same URLs
cache = {}

logger.info("All components initialized successfully")

# ============================================================================
# DEFINE REQUEST/RESPONSE MODELS
# These define the structure of data sent to and from the API
# ============================================================================

class URLAnalysisRequest(BaseModel):
    """
    Request model for URL analysis
    Validates that the input is a proper URL
    """
    url: HttpUrl = Field(
        ...,  # ... means this field is required
        description="The URL to analyze for fake content",
        example="https://www.example.com/news/article"
    )
    use_cache: Optional[bool] = Field(
        default=True,
        description="Whether to use cached results if available"
    )

class TextAnalysisRequest(BaseModel):
    """
    Request model for direct text analysis
    For analyzing text without a URL
    """
    text: str = Field(
        ...,
        description="The text content to analyze",
        min_length=50,  # Minimum 50 characters
        example="This is a news article text..."
    )
    title: Optional[str] = Field(
        default="",
        description="Optional title of the content",
        example="Breaking News: Important Discovery"
    )

class AnalysisResponse(BaseModel):
    """
    Response model for analysis results
    Defines the structure of data returned to users
    """
    success: bool = Field(description="Whether analysis was successful")
    prediction: str = Field(description="Classification: REAL, FAKE, or UNCERTAIN")
    confidence: float = Field(description="Confidence score (0-1)")
    credibility_score: int = Field(description="Overall credibility score (0-100)")
    summary: str = Field(description="Human-readable summary of findings")
    positive_factors: list = Field(description="Factors supporting credibility")
    negative_factors: list = Field(description="Factors indicating fake content")
    key_concerns: list = Field(description="Top concerns identified")
    recommendations: list = Field(description="Recommended actions for users")
    metadata: dict = Field(description="Additional information about the analysis")

class HealthResponse(BaseModel):
    """Response model for health check endpoint"""
    status: str
    timestamp: str
    model_loaded: bool
    components: Dict[str, str]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_cache_key(url: str) -> str:
    """
    Generate a unique cache key for a URL
    Uses MD5 hash to create consistent keys
    
    Args:
        url (str): The URL to hash
        
    Returns:
        str: MD5 hash of the URL
    """
    return hashlib.md5(url.encode()).hexdigest()

def save_to_cache(key: str, data: dict):
    """
    Save analysis results to cache
    In production, this would use Redis with expiration
    
    Args:
        key (str): Cache key
        data (dict): Data to cache
    """
    cache[key] = {
        'data': data,
        'timestamp': datetime.now().isoformat()
    }
    logger.debug(f"Saved result to cache with key: {key}")

def get_from_cache(key: str) -> Optional[dict]:
    """
    Retrieve results from cache if available
    
    Args:
        key (str): Cache key
        
    Returns:
        Optional[dict]: Cached data or None
    """
    if key in cache:
        cached = cache[key]
        logger.info(f"Cache hit for key: {key}")
        return cached['data']
    return None

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Root endpoint - serves the frontend HTML
    This is the main page users see when visiting the API
    """
    # Check if frontend HTML exists
    frontend_path = Path(__file__).parent / "static" / "index.html"
    
    if frontend_path.exists():
        with open(frontend_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    else:
        # Return a simple welcome message if no frontend
        return HTMLResponse(content="""
            <html>
                <head><title>Fake Content Detection API</title></head>
                <body>
                    <h1>Fake Content Detection API</h1>
                    <p>Welcome! This API detects fake news and misinformation.</p>
                    <ul>
                        <li><a href="/docs">API Documentation (Swagger)</a></li>
                        <li><a href="/redoc">API Documentation (ReDoc)</a></li>
                        <li><a href="/health">Health Check</a></li>
                    </ul>
                </body>
            </html>
        """)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    Returns the status of the API and all components
    Useful for monitoring and debugging
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": detector.model_loaded,
        "components": {
            "scraper": "operational",
            "feature_extractor": "operational",
            "detector": "operational",
            "explainer": "operational"
        }
    }

@app.post("/analyze/url", response_model=AnalysisResponse)
async def analyze_url(request: URLAnalysisRequest):
    """
    Analyze a URL for fake content
    This is the main endpoint users will call
    
    Steps:
    1. Check cache for existing analysis
    2. Scrape content from URL
    3. Extract features
    4. Run ML model prediction
    5. Generate explanation
    6. Return results
    
    Args:
        request: URLAnalysisRequest containing the URL
        
    Returns:
        AnalysisResponse: Detailed analysis results
    """
    url = str(request.url)
    logger.info(f"Received analysis request for URL: {url}")
    
    try:
        # Step 1: Check cache
        cache_key = generate_cache_key(url)
        if request.use_cache:
            cached_result = get_from_cache(cache_key)
            if cached_result:
                logger.info("Returning cached result")
                return cached_result
        
        # Step 2: Scrape content
        logger.info("Scraping content...")
        content = scraper.scrape_url(url)
        
        # Check if scraping was successful
        if content.get('error') or not content.get('text'):
            raise HTTPException(
                status_code=400,
                detail=f"Failed to extract content: {content.get('error', 'No text found')}"
            )
        
        logger.info(f"Scraped {len(content.get('text', ''))} characters")
        
        # Step 3: Extract features
        logger.info("Extracting features...")
        features = feature_extractor.extract_all_features(content)
        logger.info(f"Extracted {len(features)} features")
        
        # Step 4: Run prediction
        logger.info("Running ML prediction...")
        text = content.get('text', '')
        title = content.get('title', '')
        combined_text = f"{title}. {text}"
        
        # Use enhanced prediction with features
        prediction = detector.predict_with_features(combined_text, features)
        logger.info(f"Prediction: {prediction['prediction']} ({prediction['confidence']:.2%})")
        
        # Step 5: Generate explanation
        logger.info("Generating explanation...")
        explanation = explainer.generate_explanation(prediction, features, content)
        
        # Step 6: Prepare response
        response = {
            "success": True,
            "prediction": explanation['prediction'],
            "confidence": explanation['confidence'],
            "credibility_score": explanation['credibility_score'],
            "summary": explanation['summary'],
            "positive_factors": explanation['positive_factors'],
            "negative_factors": explanation['negative_factors'],
            "key_concerns": explanation['key_concerns'],
            "recommendations": explanation['recommendations'],
            "metadata": {
                "url": url,
                "domain": content.get('domain', ''),
                "title": content.get('title', ''),
                "author": ', '.join(content.get('authors', [])) or 'Unknown',
                "publish_date": content.get('publish_date', 'Unknown'),
                "word_count": features.get('word_count', 0),
                "analysis_timestamp": datetime.now().isoformat(),
                "model_used": prediction.get('model_used', 'Unknown')
            }
        }
        
        # Save to cache
        save_to_cache(cache_key, response)
        
        logger.info("Analysis completed successfully")
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log unexpected errors
        logger.error(f"Error analyzing URL: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/analyze/text", response_model=AnalysisResponse)
async def analyze_text(request: TextAnalysisRequest):
    """
    Analyze raw text content (without URL)
    Useful for analyzing text from any source
    
    Args:
        request: TextAnalysisRequest containing text and optional title
        
    Returns:
        AnalysisResponse: Detailed analysis results
    """
    logger.info(f"Received text analysis request ({len(request.text)} characters)")
    
    try:
        # Create a content dictionary from the text
        content = {
            'text': request.text,
            'title': request.title,
            'domain': 'direct-text',
            'authors': [],
            'publish_date': None
        }
        
        # Extract features
        logger.info("Extracting features...")
        features = feature_extractor.extract_all_features(content)
        
        # Run prediction
        logger.info("Running ML prediction...")
        combined_text = f"{request.title}. {request.text}" if request.title else request.text
        prediction = detector.predict_with_features(combined_text, features)
        
        # Generate explanation
        logger.info("Generating explanation...")
        explanation = explainer.generate_explanation(prediction, features, content)
        
        # Prepare response
        response = {
            "success": True,
            "prediction": explanation['prediction'],
            "confidence": explanation['confidence'],
            "credibility_score": explanation['credibility_score'],
            "summary": explanation['summary'],
            "positive_factors": explanation['positive_factors'],
            "negative_factors": explanation['negative_factors'],
            "key_concerns": explanation['key_concerns'],
            "recommendations": explanation['recommendations'],
            "metadata": {
                "source": "direct-text",
                "title": request.title,
                "word_count": features.get('word_count', 0),
                "analysis_timestamp": datetime.now().isoformat(),
                "model_used": prediction.get('model_used', 'Unknown')
            }
        }
        
        logger.info("Analysis completed successfully")
        return response
        
    except Exception as e:
        logger.error(f"Error analyzing text: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/model/info")
async def get_model_info():
    """
    Get information about the loaded ML model
    Useful for debugging and transparency
    """
    try:
        info = detector.get_model_info()
        return {
            "success": True,
            "model_info": info
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving model info: {str(e)}"
        )

@app.get("/cache/stats")
async def get_cache_stats():
    """
    Get statistics about the cache
    Shows how many URLs have been analyzed
    """
    return {
        "cache_size": len(cache),
        "cached_urls": len(cache),
        "oldest_entry": min((v['timestamp'] for v in cache.values()), default=None),
        "newest_entry": max((v['timestamp'] for v in cache.values()), default=None)
    }

@app.delete("/cache/clear")
async def clear_cache():
    """
    Clear the cache
    Useful for testing or forcing fresh analysis
    """
    cache.clear()
    logger.info("Cache cleared")
    return {
        "success": True,
        "message": "Cache cleared successfully"
    }

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Handle 404 errors with custom message"""
    return JSONResponse(
        status_code=404,
        content={
            "success": False,
            "error": "Endpoint not found",
            "available_endpoints": ["/", "/health", "/analyze/url", "/analyze/text", "/docs"]
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """Handle 500 errors with logging"""
    logger.error(f"Internal server error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later."
        }
    )

# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """
    Run when the server starts
    Perform initialization tasks
    """
    logger.info("=" * 60)
    logger.info("FAKE CONTENT DETECTION API STARTING")
    logger.info("=" * 60)
    logger.info(f"Environment: {config.LOG_LEVEL}")
    logger.info(f"Host: {config.API_HOST}:{config.API_PORT}")
    logger.info(f"Model: {config.MODEL_NAME}")
    logger.info("All systems initialized")
    logger.info("=" * 60)

@app.on_event("shutdown")
async def shutdown_event():
    """
    Run when the server stops
    Cleanup tasks
    """
    logger.info("=" * 60)
    logger.info("FAKE CONTENT DETECTION API SHUTTING DOWN")
    logger.info(f"Total analyses cached: {len(cache)}")
    logger.info("=" * 60)

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # This runs when you execute: python main.py
    # Starts the FastAPI server using uvicorn
    
    import uvicorn
    
    logger.info("Starting server...")
    
    # Run the application
    uvicorn.run(
        "main:app",  # app module and instance
        host=config.API_HOST,  # Listen on all interfaces
        port=config.API_PORT,  # Port number
        reload=True,  # Auto-reload on code changes (development only)
        log_level=config.LOG_LEVEL.lower()
    )
