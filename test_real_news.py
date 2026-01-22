"""
Comprehensive test to verify real news articles are properly classified
Tests various types of legitimate news sources
"""

import logging
from feature_extractor import FeatureExtractor
from model import FakeNewsDetector
from explainer import PredictionExplainer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_real_news_articles():
    """Test multiple real news articles to check for false positives"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE REAL NEWS TEST - Checking for False Positives")
    print("="*80)
    
    # Initialize components
    feature_extractor = FeatureExtractor()
    detector = FakeNewsDetector()
    explainer = PredictionExplainer()
    
    # Collection of various real news examples
    test_articles = [
        {
            'name': 'BBC News',
            'content': {
                'title': "Climate Summit Reaches Agreement on Emissions Targets",
                'text': """
                World leaders at the UN Climate Summit have reached a landmark agreement on 
                reducing carbon emissions. The deal, announced today, commits 195 countries 
                to cutting emissions by 45% by 2030.
                
                UN Secretary-General António Guterres called it "a critical step forward." 
                He told reporters: "This agreement shows that international cooperation on 
                climate change is possible, even in challenging times."
                
                The accord was negotiated over two weeks of intensive discussions. Key 
                provisions include new funding mechanisms for developing nations and 
                enhanced monitoring systems. Environmental groups welcomed the news but 
                cautioned that implementation will be crucial.
                """,
                'domain': 'bbc.com',
                'has_author': True,
                'authors': ['Environment Correspondent'],
                'publish_date': '2026-01-20'
            }
        },
        {
            'name': 'Reuters',
            'content': {
                'title': "Stock Markets Rise on Economic Data",
                'text': """
                Major stock indices closed higher on Friday following the release of 
                stronger-than-expected employment figures. The S&P 500 gained 1.2%, 
                while the Dow Jones Industrial Average rose 250 points.
                
                The Labor Department reported that 300,000 jobs were added in December, 
                exceeding analyst predictions of 200,000. The unemployment rate remained 
                steady at 3.7%.
                
                "Today's numbers suggest the economy remains resilient," said Janet 
                Wilson, chief economist at Morgan Stanley. Market analysts expect the 
                Federal Reserve to consider these figures at its next policy meeting.
                """,
                'domain': 'reuters.com',
                'has_author': True,
                'authors': ['Financial Markets Reporter'],
                'publish_date': '2026-01-21'
            }
        },
        {
            'name': 'NY Times',
            'content': {
                'title': "New Study Links Diet to Heart Health",
                'text': """
                A comprehensive study published in the New England Journal of Medicine 
                has found a strong correlation between Mediterranean diets and reduced 
                heart disease risk. Researchers followed 25,000 participants over 15 years.
                
                Dr. Sarah Martinez, lead author from Harvard Medical School, explained 
                that diets rich in olive oil, fish, and vegetables showed a 30% reduction 
                in cardiovascular events compared to standard Western diets.
                
                The study, peer-reviewed by independent experts, controlled for factors 
                such as age, exercise, and family history. The American Heart Association 
                has endorsed the findings and updated its dietary recommendations.
                """,
                'domain': 'nytimes.com',
                'has_author': True,
                'authors': ['Health and Science Writer'],
                'publish_date': '2026-01-19'
            }
        },
        {
            'name': 'CNN Breaking News',
            'content': {
                'title': "Breaking: Major Earthquake Strikes Pacific Region",
                'text': """
                A magnitude 7.2 earthquake struck off the coast of Japan early this morning, 
                prompting tsunami warnings across the Pacific. The U.S. Geological Survey 
                confirmed the quake occurred at 6:15 AM local time.
                
                Japanese authorities have issued evacuation orders for coastal areas. 
                Prime Minister Fumio Kishida said emergency response teams are being 
                deployed. "We are monitoring the situation closely and coordinating with 
                international partners," he stated in a brief address.
                
                No casualties have been reported yet, but several buildings sustained 
                damage. The Pacific Tsunami Warning Center is tracking wave movements.
                """,
                'domain': 'cnn.com',
                'has_author': True,
                'authors': ['Breaking News Team'],
                'publish_date': '2026-01-22'
            }
        },
        {
            'name': 'Generic News (No Author)',
            'content': {
                'title': "Local School Wins National Science Award",
                'text': """
                Springfield High School has been awarded the National Excellence in STEM 
                Education prize for its innovative science program. The school's robotics 
                team took first place in the national competition last month.
                
                Principal Robert Chen expressed pride in the students' achievements. 
                The program, launched three years ago, focuses on hands-on learning and 
                real-world problem-solving. Funding came from a combination of district 
                budgets and community donations.
                
                The award includes a $50,000 grant to expand the program. Students will 
                use the funds to purchase new equipment and attend summer science camps.
                """,
                'domain': 'localnewtimes.com',
                'has_author': False,
                'publish_date': '2026-01-18'
            }
        }
    ]
    
    results = []
    correct = 0
    false_positives = 0
    
    for article in test_articles:
        print(f"\n{'='*80}")
        print(f"Testing: {article['name']}")
        print(f"{'='*80}")
        print(f"Title: {article['content']['title']}")
        print(f"Domain: {article['content']['domain']}")
        
        # Extract features
        features = feature_extractor.extract_all_features(article['content'])
        
        # Make prediction
        combined_text = f"{article['content']['title']}. {article['content']['text']}"
        prediction = detector.predict_with_features(combined_text, features)
        
        # Check result
        is_correct = prediction['prediction'] in ['REAL', 'UNCERTAIN']
        is_false_positive = prediction['prediction'] == 'FAKE'
        
        if is_correct:
            correct += 1
        if is_false_positive:
            false_positives += 1
        
        print(f"\n📊 Result: {prediction['prediction']} {'✅' if is_correct else '❌'}")
        print(f"   Confidence: {prediction['confidence']:.1%}")
        print(f"   Fake Probability: {prediction['fake_probability']:.1%}")
        
        results.append({
            'name': article['name'],
            'prediction': prediction['prediction'],
            'confidence': prediction['confidence'],
            'correct': is_correct,
            'false_positive': is_false_positive
        })
    
    # Summary
    print(f"\n{'='*80}")
    print("📈 SUMMARY")
    print(f"{'='*80}")
    print(f"Total Articles Tested: {len(test_articles)}")
    print(f"Correctly Classified: {correct}/{len(test_articles)} ({correct/len(test_articles)*100:.1f}%)")
    print(f"False Positives (Real → Fake): {false_positives}")
    print(f"{'='*80}")
    
    if false_positives == 0:
        print("✅ EXCELLENT: No false positives detected!")
    elif false_positives <= 1:
        print("⚠️  GOOD: Only 1 false positive detected")
    else:
        print(f"❌ NEEDS IMPROVEMENT: {false_positives} false positives detected")
    
    print(f"\n{'='*80}\n")
    
    return false_positives == 0

if __name__ == "__main__":
    # Download NLTK data if needed
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except:
        pass
    
    success = test_real_news_articles()
    if success:
        print("🎉 All tests passed! Model is performing well on real news.\n")
    else:
        print("⚠️  Some tests failed. Review results above.\n")
