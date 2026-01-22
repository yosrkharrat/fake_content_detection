"""
Simple script to run the fake content detection model on a real example
"""

import logging
from feature_extractor import FeatureExtractor
from model import FakeNewsDetector
from explainer import PredictionExplainer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_example():
    """Run the model on a real news example"""
    
    # Initialize components
    logger.info("Initializing components...")
    feature_extractor = FeatureExtractor()
    detector = FakeNewsDetector()
    explainer = PredictionExplainer()
    
    # Real news example
    example_content = {
        'title': "NASA's James Webb Telescope Discovers Water Molecules on Distant Exoplanet",
        'text': """
        Scientists at NASA announced today that the James Webb Space Telescope has detected 
        water vapor in the atmosphere of K2-18b, an exoplanet located 120 light-years away 
        from Earth. The discovery, published in the Astrophysical Journal Letters, marks 
        the first time water has been confirmed on a planet in its star's habitable zone.
        
        Dr. Maria Rodriguez, lead researcher at NASA's Goddard Space Flight Center, explained 
        that the telescope's infrared capabilities allowed the team to analyze the planet's 
        atmospheric composition. "This is a significant milestone in our search for potentially 
        habitable worlds beyond our solar system," Rodriguez stated in a press conference.
        
        The research team used spectroscopy to identify the molecular signature of water vapor 
        as starlight passed through K2-18b's atmosphere. The planet, which is approximately 
        8.6 times the mass of Earth, orbits a red dwarf star in the constellation Leo.
        
        The findings were peer-reviewed by international astronomers and confirmed by independent 
        observations from the European Space Agency. NASA plans to conduct follow-up observations 
        to search for other biosignature gases like methane and ammonia.
        """,
        'domain': 'nasa.gov',
        'has_author': True,
        'publish_date': '2026-01-20'
    }
    
    # Fake news example for comparison
    fake_example = {
        'title': "SHOCKING: Scientists Confirm Aliens Built the Pyramids, Government Covers It Up!",
        'text': """
        An anonymous whistleblower from Area 51 has leaked documents proving that aliens 
        visited Earth 5,000 years ago and built the pyramids using advanced anti-gravity 
        technology. The documents, which cannot be verified, allegedly show blueprints 
        of UFOs found inside the Great Pyramid.
        
        Experts agree that humans could never have built such structures without help. 
        The government has been hiding this truth for decades! Share this before they 
        delete it! Click here to see the leaked photos that will shock you!
        
        One researcher, who wishes to remain anonymous, claims that mainstream archaeologists 
        are part of a conspiracy to hide the truth. Wake up, sheeple! The evidence is 
        overwhelming but they don't want you to know!
        """,
        'domain': 'totallyrealworldnews.xyz',
        'has_author': False
    }
    
    print("\n" + "="*80)
    print("FAKE CONTENT DETECTION - REAL EXAMPLE TEST")
    print("="*80)
    
    # Analyze REAL news example
    print("\n📰 EXAMPLE 1: NASA Discovery (Expected: REAL NEWS)")
    print("-"*80)
    print(f"Title: {example_content['title']}")
    print(f"Domain: {example_content['domain']}")
    print(f"\nText Preview: {example_content['text'][:200]}...")
    
    print("\n🔍 Analyzing...")
    
    # Extract features
    features = feature_extractor.extract_all_features(example_content)
    
    # Make prediction
    combined_text = f"{example_content['title']}. {example_content['text']}"
    prediction = detector.predict(combined_text, features)
    
    # Generate explanation
    explanation = explainer.generate_explanation(prediction, features, example_content)
    
    # Display results
    print("\n📊 RESULTS:")
    print(f"   Prediction: {prediction['prediction']}")
    print(f"   Confidence: {prediction['confidence']:.2%}")
    print(f"   Fake Probability: {prediction['fake_probability']:.2%}")
    print(f"   Real Probability: {prediction['real_probability']:.2%}")
    print(f"   Credibility Score: {explanation['credibility_score']:.2f}/100")
    
    print("\n✅ Positive Factors (Why it might be real):")
    for i, factor in enumerate(explanation['positive_factors'][:5], 1):
        print(f"   {i}. {factor}")
    
    if explanation['negative_factors']:
        print("\n⚠️  Negative Factors (Why it might be fake):")
        for i, factor in enumerate(explanation['negative_factors'][:5], 1):
            print(f"   {i}. {factor}")
    
    print("\n💡 Summary:")
    print(f"   {explanation['summary']}")
    
    # Analyze FAKE news example
    print("\n" + "="*80)
    print("\n📰 EXAMPLE 2: Alien Pyramid Theory (Expected: FAKE NEWS)")
    print("-"*80)
    print(f"Title: {fake_example['title']}")
    print(f"Domain: {fake_example['domain']}")
    print(f"\nText Preview: {fake_example['text'][:200]}...")
    
    print("\n🔍 Analyzing...")
    
    # Extract features
    features2 = feature_extractor.extract_all_features(fake_example)
    
    # Make prediction
    combined_text2 = f"{fake_example['title']}. {fake_example['text']}"
    prediction2 = detector.predict(combined_text2, features2)
    
    # Generate explanation
    explanation2 = explainer.generate_explanation(prediction2, features2, fake_example)
    
    # Display results
    print("\n📊 RESULTS:")
    print(f"   Prediction: {prediction2['prediction']}")
    print(f"   Confidence: {prediction2['confidence']:.2%}")
    print(f"   Fake Probability: {prediction2['fake_probability']:.2%}")
    print(f"   Real Probability: {prediction2['real_probability']:.2%}")
    print(f"   Credibility Score: {explanation2['credibility_score']:.2f}/100")
    
    print("\n✅ Positive Factors (Why it might be real):")
    for i, factor in enumerate(explanation2['positive_factors'][:5], 1):
        print(f"   {i}. {factor}")
    
    if explanation2['negative_factors']:
        print("\n⚠️  Negative Factors (Why it might be fake):")
        for i, factor in enumerate(explanation2['negative_factors'][:5], 1):
            print(f"   {i}. {factor}")
    
    print("\n💡 Summary:")
    print(f"   {explanation2['summary']}")
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    print("\nKey Differences:")
    print(f"   Real News Credibility: {explanation['credibility_score']:.1f}/100")
    print(f"   Fake News Credibility: {explanation2['credibility_score']:.1f}/100")
    print(f"   Difference: {abs(explanation['credibility_score'] - explanation2['credibility_score']):.1f} points")

if __name__ == "__main__":
    # Download NLTK data if needed
    try:
        import nltk
        print("Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except:
        pass
    
    run_example()
