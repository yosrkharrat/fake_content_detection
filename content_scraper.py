"""
Content Scraper Module
This module handles fetching and extracting content from URLs
It extracts text, metadata, and structural information from web pages
"""

import requests
from bs4 import BeautifulSoup
from newspaper import Article
from urllib.parse import urlparse
import logging
from typing import Dict, Optional
from datetime import datetime

# Importing project configuration
import config

# Setting up logging for this module
logger = logging.getLogger(__name__)


class ContentScraper:
    """
    A class to scrape and extract content from web URLs
    Handles different content types and provides fallback mechanisms
    """
    
    def __init__(self):
        """
        Initialize the ContentScraper with headers and session
        """
        # Creating a session for persistent connections and cookie handling
        self.session = requests.Session()
        
        # Setting headers to mimic a real browser (some sites block scrapers)
        self.headers = {
            'User-Agent': config.USER_AGENT,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
    
    def scrape_url(self, url: str) -> Dict:
        """
        Main method to scrape content from a URL
        
        Args:
            url (str): The URL to scrape
            
        Returns:
            Dict: A dictionary containing extracted content and metadata
        """
        logger.info(f"Scraping URL: {url}")
        
        # Validate URL format
        if not self._is_valid_url(url):
            return self._create_error_response("Invalid URL format")
        
        try:
            # Method 1: Trying using newspaper3k library (best for news articles)
            content = self._extract_with_newspaper(url)
            
            # If newspaper3k fails, falling back to BeautifulSoup
            if not content.get('text'):
                logger.warning("Newspaper3k extraction failed, trying BeautifulSoup")
                content = self._extract_with_beautifulsoup(url)
            
            # Adding additional metadata
            content['domain'] = self._extract_domain(url)
            content['scrape_timestamp'] = datetime.now().isoformat()
            
            return content
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout while fetching {url}")
            return self._create_error_response("Request timeout")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for {url}: {str(e)}")
            return self._create_error_response(f"Failed to fetch URL: {str(e)}")
            
        except Exception as e:
            logger.error(f"Unexpected error scraping {url}: {str(e)}")
            return self._create_error_response(f"Unexpected error: {str(e)}")
    
    def _extract_with_newspaper(self, url: str) -> Dict:
        """
        Extract content using newspaper3k library
        This library is specialized for news articles
        
        Args:
            url (str): The URL to extract content from
            
        Returns:
            Dict: Extracted content and metadata
        """
        try:
            # Creating an Article object
            article = Article(url)
            
            # Downloading the article HTML
            article.download()
            
            # Parsing the HTML and extracting content
            article.parse()
            
            # Performing NLP analysis (keyword extraction, summarization)
            article.nlp()
            
            # Extracting all available information
            content = {
                'url': url,
                'title': article.title or '',
                'text': article.text or '',
                'authors': article.authors or [],
                'publish_date': article.publish_date.isoformat() if article.publish_date else None,
                'top_image': article.top_image or '',
                'images': list(article.images) or [],
                'videos': list(article.movies) or [],
                'keywords': article.keywords or [],
                'summary': article.summary or '',
                'meta_description': article.meta_description or '',
                'meta_keywords': article.meta_keywords or [],
                'source_url': article.source_url or '',
                'extraction_method': 'newspaper3k'
            }
            
            logger.info(f"Successfully extracted content using newspaper3k")
            return content
            
        except Exception as e:
            logger.warning(f"Newspaper3k extraction failed: {str(e)}")
            return {'text': '', 'error': str(e)}
    
    def _extract_with_beautifulsoup(self, url: str) -> Dict:
        """
        Extract content using BeautifulSoup (fallback method)
        More general but less accurate than newspaper3k
        
        Args:
            url (str): The URL to extract content from
            
        Returns:
            Dict: Extracted content and metadata
        """
        try:
            # Fetch the web page with timeout
            response = self.session.get(
                url, 
                headers=self.headers, 
                timeout=config.REQUEST_TIMEOUT,
                allow_redirects=True,
                stream=True
            )
            
            # Check if content size is within limits
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > config.MAX_CONTENT_SIZE:
                return self._create_error_response("Content too large")
            
            # Raise exception for bad status codes
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements (not part of main content)
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Extract title from <title> tag or <h1>
            title = ''
            if soup.title:
                title = soup.title.string
            elif soup.find('h1'):
                title = soup.find('h1').get_text()
            
            # Extract meta description
            meta_desc = ''
            meta_tag = soup.find('meta', attrs={'name': 'description'})
            if meta_tag:
                meta_desc = meta_tag.get('content', '')
            
            # Extract author from meta tags
            authors = []
            author_meta = soup.find('meta', attrs={'name': 'author'})
            if author_meta:
                authors = [author_meta.get('content', '')]
            
            # Extract publish date from various meta tags
            publish_date = None
            date_selectors = [
                {'property': 'article:published_time'},
                {'name': 'publish_date'},
                {'name': 'date'},
            ]
            for selector in date_selectors:
                date_tag = soup.find('meta', attrs=selector)
                if date_tag:
                    publish_date = date_tag.get('content', '')
                    break
            
            # Extract main text content
            # Try to find main content area (article, main, or multiple paragraphs)
            text = ''
            article_tag = soup.find('article')
            if article_tag:
                text = article_tag.get_text(separator=' ', strip=True)
            else:
                # Fall back to extracting all paragraph text
                paragraphs = soup.find_all('p')
                text = ' '.join([p.get_text(strip=True) for p in paragraphs])
            
            # Extract images
            images = []
            for img in soup.find_all('img', src=True):
                images.append(img['src'])
            
            content = {
                'url': url,
                'title': title.strip() if title else '',
                'text': text.strip() if text else '',
                'authors': authors,
                'publish_date': publish_date,
                'meta_description': meta_desc,
                'images': images[:10],  # Limit to first 10 images
                'extraction_method': 'beautifulsoup'
            }
            
            logger.info(f"Successfully extracted content using BeautifulSoup")
            return content
            
        except Exception as e:
            logger.error(f"BeautifulSoup extraction failed: {str(e)}")
            return self._create_error_response(f"Extraction failed: {str(e)}")
    
    def _is_valid_url(self, url: str) -> bool:
        """
        Validate if a string is a proper URL
        
        Args:
            url (str): The URL to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            result = urlparse(url)
            # Check if scheme (http/https) and netloc (domain) are present
            return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
        except:
            return False
    
    def _extract_domain(self, url: str) -> str:
        """
        Extract the domain name from a URL
        
        Args:
            url (str): The URL to extract domain from
            
        Returns:
            str: The domain name
        """
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except:
            return ''
    
    def _create_error_response(self, error_message: str) -> Dict:
        """
        Create a standardized error response
        
        Args:
            error_message (str): Description of the error
            
        Returns:
            Dict: Error response dictionary
        """
        return {
            'error': error_message,
            'text': '',
            'title': '',
            'success': False
        }


# Example usage and testing
if __name__ == "__main__":
    # Set up logging for testing
    logging.basicConfig(level=logging.INFO)
    
    # Create scraper instance
    scraper = ContentScraper()
    
    # Test with a sample URL
    test_url = "https://www.bbc.com/news"
    result = scraper.scrape_url(test_url)
    
    print("Scraping Results:")
    print(f"Title: {result.get('title', 'N/A')}")
    print(f"Text length: {len(result.get('text', ''))} characters")
    print(f"Authors: {result.get('authors', [])}")
    print(f"Domain: {result.get('domain', 'N/A')}")

