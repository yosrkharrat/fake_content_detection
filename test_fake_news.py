"""
Test fake news detection to ensure we still catch obvious fake content
"""

import logging
from feature_extractor import FeatureExtractor
from model import FakeNewsDetector

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_fake_news_detection():
    """Test that obvious fake news is still detected"""
    
    print("\n" + "="*80)
    print("FAKE NEWS DETECTION TEST")
    print("="*80)
    
    # Initialize components
    feature_extractor = FeatureExtractor()
    detector = FakeNewsDetector()
    
    fake_articles = [
        {
            'name': 'Conspiracy Theory',
            'content': {
                'title': "SHOCKING: Government Confirmed Hiding Aliens in Secret Base!",
                'text': """
                UNBELIEVABLE NEWS! Anonymous whistleblower reveals government has been 
                hiding aliens in underground facility. This will blow your mind! The truth 
                they don't want you to know!
                
                One unnamed expert says "this changes everything." Share before they 
                delete this! Wake up sheeple! The evidence is overwhelming but mainstream 
                media refuses to report it!
                
                Click here to see the photos they're trying to hide! What happens next 
                will shock you! Doctors hate this one weird trick!
                """,
                'domain': 'conspiracytruthblog.xyz',
                'has_author': False
            }
        },
        {
            'name': 'Medical Misinformation',
            'content': {
                'title': "Scientists HATE This One Weird Trick To Cure Cancer!!!",
                'text': """
                BIG PHARMA doesn't want you to know about this AMAZING discovery! A secret 
                fruit from the Amazon rainforest can cure all diseases including cancer!
                
                Anonymous doctors confirm this works 100% of the time but the medical 
                establishment is suppressing it to protect profits! Share this before 
                they take it down!
                
                Order now for only $99.99! Don't let them silence the truth! This will 
                change your life forever! Act now before it's too late!
                """,
                'domain': 'naturemiraclecures.net',
                'has_author': False
            }
        }
    ]
    
    results = []
    correct = 0
    
    for article in fake_articles:
        print(f"\n{'='*80}")
        print(f"Testing: {article['name']}")
        print(f"{'='*80}")
        print(f"Title: {article['content']['title']}")
        
        # Extract features
        features = feature_extractor.extract_all_features(article['content'])
        
        # Make prediction
        combined_text = f"{article['content']['title']}. {article['content']['text']}"
        prediction = detector.predict_with_features(combined_text, features)
        
        # Should detect as FAKE or UNCERTAIN
        is_correct = prediction['prediction'] in ['FAKE', 'UNCERTAIN']
        
        if is_correct:
            correct += 1
        
        print(f"\n📊 Result: {prediction['prediction']} {'✅' if is_correct else '❌'}")
        print(f"   Confidence: {prediction['confidence']:.1%}")
        print(f"   Fake Probability: {prediction['fake_probability']:.1%}")
        
        results.append({
            'name': article['name'],
            'prediction': prediction['prediction'],
            'correct': is_correct
        })
    
    # Summary
    print(f"\n{'='*80}")
    print("📈 SUMMARY")
    print(f"{'='*80}")
    print(f"Total Fake Articles Tested: {len(fake_articles)}")
    print(f"Correctly Identified: {correct}/{len(fake_articles)} ({correct/len(fake_articles)*100:.1f}%)")
    
    if correct == len(fake_articles):
        print("✅ EXCELLENT: All fake news detected!")
    else:
        print(f"⚠️  {len(fake_articles) - correct} fake articles missed")
    
    print(f"{'='*80}\n")
    
    return correct == len(fake_articles)

if __name__ == "__main__":
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except:
        pass
    
    success = test_fake_news_detection()
    if success:
        print("🎉 All fake news correctly identified!\n")
    else:
        print("⚠️  Some fake news was missed.\n")
