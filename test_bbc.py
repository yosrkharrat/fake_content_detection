"""
Quick test to verify BBC news is properly classified
"""

import logging
from feature_extractor import FeatureExtractor
from model import FakeNewsDetector
from explainer import PredictionExplainer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_bbc_news():
    """Test BBC news article classification"""
    
    print("\n" + "="*80)
    print("TESTING BBC NEWS CLASSIFICATION")
    print("="*80)
    
    # Initialize components
    feature_extractor = FeatureExtractor()
    detector = FakeNewsDetector()
    explainer = PredictionExplainer()
    
    # Simulate BBC news article
    bbc_content = {
        'title': "UK Economy Shows Signs of Recovery in Latest GDP Report",
        'text': """
        The UK economy has shown unexpected resilience in the latest quarter, 
        according to figures released by the Office for National Statistics today. 
        GDP grew by 0.3% in the three months to December, beating analysts' 
        expectations of 0.1% growth.
        
        Chancellor Rachel Reeves welcomed the figures, stating that they demonstrate 
        the government's economic policies are beginning to take effect. "Today's 
        numbers show we're on the right track," she told reporters in Westminster.
        
        However, economists at the Bank of England cautioned that inflation remains 
        a concern, with consumer prices still above the 2% target. The central bank 
        is expected to maintain interest rates at their current level when it meets 
        next month.
        
        Business leaders responded positively to the news. Dame Emma Thompson, 
        director of the CBI, said: "This growth is encouraging, but businesses 
        need long-term stability and clear policy direction to invest with confidence."
        """,
        'domain': 'bbc.co.uk',
        'has_author': True,
        'authors': ['Robert Peston', 'Economics Correspondent'],
        'publish_date': '2026-01-22'
    }
    
    print(f"\n📰 Article: {bbc_content['title']}")
    print(f"🌐 Domain: {bbc_content['domain']}")
    print(f"✍️  Authors: {', '.join(bbc_content['authors'])}")
    print(f"📅 Date: {bbc_content['publish_date']}")
    
    print("\n🔍 Analyzing...")
    
    # Extract features
    features = feature_extractor.extract_all_features(bbc_content)
    print(f"\n📊 Features Extracted:")
    print(f"   - Trusted Domain: {'Yes' if features.get('is_trusted_domain') else 'No'}")
    print(f"   - Has Author: {'Yes' if features.get('has_author') else 'No'}")
    print(f"   - Has Date: {'Yes' if features.get('has_publish_date') else 'No'}")
    print(f"   - Clickbait Keywords: {features.get('clickbait_keyword_count', 0)}")
    
    # Make prediction
    combined_text = f"{bbc_content['title']}. {bbc_content['text']}"
    prediction = detector.predict_with_features(combined_text, features)
    
    # Generate explanation
    explanation = explainer.generate_explanation(prediction, features, bbc_content)
    
    # Display results
    print("\n" + "="*80)
    print("📊 RESULTS:")
    print("="*80)
    print(f"   Prediction: {prediction['prediction']} {'✅' if prediction['prediction'] == 'REAL' else '❌' if prediction['prediction'] == 'FAKE' else '⚠️'}")
    print(f"   Confidence: {prediction['confidence']:.2%}")
    print(f"   Fake Probability: {prediction['fake_probability']:.2%}")
    print(f"   Real Probability: {prediction['real_probability']:.2%}")
    print(f"   Credibility Score: {explanation['credibility_score']:.1f}/100")
    
    # Show if model is working correctly
    if prediction['prediction'] == 'REAL':
        print("\n✅ SUCCESS! BBC news correctly identified as REAL news")
    elif prediction['prediction'] == 'UNCERTAIN':
        print("\n⚠️  UNCERTAIN classification (needs improvement)")
    else:
        print("\n❌ ERROR! BBC news incorrectly classified as FAKE")
    
    print("\n💡 Key Factors:")
    for factor in explanation['positive_factors'][:3]:
        print(f"   ✓ {factor.get('factor')}: {factor.get('description')}")
    
    return prediction['prediction'] == 'REAL'

if __name__ == "__main__":
    # Download NLTK data if needed
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except:
        pass
    
    success = test_bbc_news()
    print("\n" + "="*80)
    if success:
        print("✅ TEST PASSED: BBC news properly classified as REAL")
    else:
        print("⚠️  TEST INCOMPLETE: Further tuning may be needed")
    print("="*80 + "\n")
