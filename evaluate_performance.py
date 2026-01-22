"""
Performance Evaluation Script
Tests the fake content detection system and generates performance metrics
Measures accuracy, precision, recall, F1-score, inference time, and resource usage
"""

import time
import logging
from typing import Dict, List, Tuple
import json
from datetime import datetime
from pathlib import Path
import statistics

# Import our modules
from content_scraper import ContentScraper
from feature_extractor import FeatureExtractor
from model import FakeNewsDetector
from explainer import PredictionExplainer
import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerformanceEvaluator:
    """
    Evaluates the performance of the fake content detection system
    Generates comprehensive metrics and reports
    """
    
    def __init__(self):
        """Initialize all components"""
        logger.info("Initializing performance evaluator...")
        
        self.scraper = ContentScraper()
        self.feature_extractor = FeatureExtractor()
        self.detector = FakeNewsDetector()
        self.explainer = PredictionExplainer()
        
        # Test dataset - mix of real and fake news examples
        self.test_samples = self._create_test_dataset()
        
        # Metrics storage
        self.results = []
        self.inference_times = []
        
        logger.info("Performance evaluator initialized")
    
    def _create_test_dataset(self) -> List[Dict]:
        """
        Create a test dataset with labeled examples
        In production, you would load from FakeNewsNet or LIAR dataset
        
        Returns:
            List[Dict]: Test samples with text and labels
        """
        # Sample dataset - replace with real dataset in production
        samples = [
            # REAL NEWS EXAMPLES
            {
                'text': """
                According to a study published in Nature journal by researchers at MIT, 
                a new solar panel technology has achieved 25% efficiency in converting 
                sunlight to electricity. Dr. Sarah Johnson, lead researcher, stated that 
                the breakthrough could reduce solar energy costs by 30% over the next 
                five years. The research was funded by the Department of Energy and 
                peer-reviewed by independent scientists.
                """,
                'title': "MIT Scientists Develop More Efficient Solar Panels",
                'label': 'REAL',
                'domain': 'sciencedaily.com',
                'has_author': True
            },
            {
                'text': """
                The Federal Reserve announced today a 0.25% increase in interest rates, 
                citing concerns about inflation. Fed Chairman Jerome Powell explained 
                the decision in a press conference, noting that the central bank aims 
                to maintain price stability. Economists at Goldman Sachs and JPMorgan 
                had anticipated this move based on recent economic indicators.
                """,
                'title': "Federal Reserve Raises Interest Rates by 0.25%",
                'label': 'REAL',
                'domain': 'reuters.com',
                'has_author': True
            },
            {
                'text': """
                A comprehensive study by Johns Hopkins University involving 50,000 
                participants found that regular exercise reduces the risk of heart 
                disease by up to 35%. The research, published in the Journal of the 
                American Medical Association, tracked participants over 10 years. 
                Lead author Dr. Michael Chen emphasized the importance of at least 
                30 minutes of moderate exercise daily.
                """,
                'title': "Study Links Regular Exercise to Lower Heart Disease Risk",
                'label': 'REAL',
                'domain': 'medicalxpress.com',
                'has_author': True
            },
            
            # FAKE NEWS EXAMPLES
            {
                'text': """
                SHOCKING DISCOVERY! Scientists have found a miracle cure that Big Pharma 
                doesn't want you to know about! This one weird trick will change your life 
                FOREVER! Doctors are FURIOUS! You won't believe what happens next! This 
                ancient remedy will cure ALL diseases instantly! Share this before it gets 
                taken down! The government is trying to hide this from you!!!
                """,
                'title': "SHOCKING: Doctors HATE This One Weird Trick!",
                'label': 'FAKE',
                'domain': 'healthfrauds.com',
                'has_author': False
            },
            {
                'text': """
                BREAKING: Anonymous sources claim that a celebrity was spotted doing 
                something unbelievable! You won't believe what they did! This will blow 
                your mind! The mainstream media is covering this up! Wake up sheeple! 
                Everything you know is a lie! Share this before they delete it! The 
                truth is finally revealed! This changes EVERYTHING!
                """,
                'title': "UNBELIEVABLE: The Truth They Don't Want You To Know!",
                'label': 'FAKE',
                'domain': 'conspiracynews.net',
                'has_author': False
            },
            {
                'text': """
                URGENT WARNING! A new study that nobody has heard of proves that 
                drinking water is DEADLY! Scientists (who we can't name) say that 
                100% of people who drink water eventually die! This is the biggest 
                coverup in history! The water industry is hiding the truth! You need 
                to stop drinking water immediately or face terrible consequences! 
                Share this to save lives!!!
                """,
                'title': "SCIENTISTS WARN: Water Is KILLING You!",
                'label': 'FAKE',
                'domain': 'fakenews247.com',
                'has_author': False
            },
            
            # MIXED/UNCERTAIN EXAMPLES
            {
                'text': """
                Some people are saying that a new technology might change everything. 
                There are rumors that major companies are interested. Sources suggest 
                that this could be important. Many believe this will have an impact. 
                It seems likely that something will happen. Experts think this could 
                be significant. There might be developments in the future.
                """,
                'title': "New Technology May Have Impact",
                'label': 'UNCERTAIN',
                'domain': 'techblog.com',
                'has_author': True
            },
            {
                'text': """
                Recent reports indicate possible changes in market conditions. Industry 
                insiders suggest potential shifts in consumer behavior. Analysts predict 
                various outcomes depending on multiple factors. The situation remains 
                fluid with several variables at play. Stakeholders are monitoring 
                developments closely. Outcomes could vary based on circumstances.
                """,
                'title': "Market Conditions Show Possible Changes",
                'label': 'UNCERTAIN',
                'domain': 'businessnews.com',
                'has_author': True
            }
        ]
        
        return samples
    
    def evaluate_single_sample(self, sample: Dict) -> Dict:
        """
        Evaluate a single sample and measure performance
        
        Args:
            sample (Dict): Test sample with text and label
            
        Returns:
            Dict: Evaluation results
        """
        # Create content dictionary
        content = {
            'text': sample['text'].strip(),
            'title': sample.get('title', ''),
            'domain': sample.get('domain', ''),
            'authors': ['Unknown'] if sample.get('has_author') else [],
            'publish_date': None
        }
        
        # Start timing
        start_time = time.time()
        
        try:
            # Step 1: Extract features
            features = self.feature_extractor.extract_all_features(content)
            
            # Step 2: Run prediction
            combined_text = f"{content['title']}. {content['text']}"
            prediction = self.detector.predict_with_features(combined_text, features)
            
            # Step 3: Generate explanation
            explanation = self.explainer.generate_explanation(
                prediction, features, content
            )
            
            # Calculate inference time
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            # Determine if prediction is correct
            predicted_label = prediction['prediction']
            true_label = sample['label']
            
            # Calculate correctness (handle UNCERTAIN as neither right nor wrong)
            is_correct = predicted_label == true_label
            
            # For binary classification (ignoring UNCERTAIN)
            if true_label != 'UNCERTAIN' and predicted_label != 'UNCERTAIN':
                true_positive = (true_label == 'FAKE' and predicted_label == 'FAKE')
                false_positive = (true_label == 'REAL' and predicted_label == 'FAKE')
                true_negative = (true_label == 'REAL' and predicted_label == 'REAL')
                false_negative = (true_label == 'FAKE' and predicted_label == 'REAL')
            else:
                true_positive = false_positive = true_negative = false_negative = False
            
            result = {
                'sample_id': len(self.results) + 1,
                'title': sample.get('title', 'N/A'),
                'true_label': true_label,
                'predicted_label': predicted_label,
                'confidence': prediction['confidence'],
                'credibility_score': explanation['credibility_score'],
                'is_correct': is_correct,
                'inference_time': inference_time,
                'true_positive': true_positive,
                'false_positive': false_positive,
                'true_negative': true_negative,
                'false_negative': false_negative,
                'feature_count': len(features),
                'positive_factors': len(explanation['positive_factors']),
                'negative_factors': len(explanation['negative_factors'])
            }
            
            logger.info(
                f"Sample {result['sample_id']}: "
                f"True={true_label}, Predicted={predicted_label}, "
                f"Correct={is_correct}, Time={inference_time:.3f}s"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating sample: {str(e)}")
            return {
                'sample_id': len(self.results) + 1,
                'error': str(e),
                'inference_time': time.time() - start_time
            }
    
    def run_evaluation(self) -> Dict:
        """
        Run complete evaluation on all test samples
        
        Returns:
            Dict: Comprehensive evaluation metrics
        """
        logger.info("=" * 70)
        logger.info("STARTING PERFORMANCE EVALUATION")
        logger.info("=" * 70)
        
        # Evaluate each sample
        for i, sample in enumerate(self.test_samples, 1):
            logger.info(f"\nEvaluating sample {i}/{len(self.test_samples)}")
            result = self.evaluate_single_sample(sample)
            self.results.append(result)
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        # Generate report
        self._print_report(metrics)
        
        # Save results
        self._save_results(metrics)
        
        return metrics
    
    def _calculate_metrics(self) -> Dict:
        """
        Calculate performance metrics from results
        
        Returns:
            Dict: Performance metrics
        """
        # Filter out errors
        valid_results = [r for r in self.results if 'error' not in r]
        
        if not valid_results:
            return {'error': 'No valid results to calculate metrics'}
        
        # Basic counts
        total_samples = len(valid_results)
        correct_predictions = sum(1 for r in valid_results if r['is_correct'])
        
        # Binary classification metrics (excluding UNCERTAIN)
        binary_results = [
            r for r in valid_results 
            if r['true_label'] != 'UNCERTAIN' and r['predicted_label'] != 'UNCERTAIN'
        ]
        
        if binary_results:
            tp = sum(r['true_positive'] for r in binary_results)
            fp = sum(r['false_positive'] for r in binary_results)
            tn = sum(r['true_negative'] for r in binary_results)
            fn = sum(r['false_negative'] for r in binary_results)
            
            # Calculate metrics
            accuracy = (tp + tn) / len(binary_results) if binary_results else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        else:
            tp = fp = tn = fn = 0
            accuracy = precision = recall = f1_score = 0
        
        # Overall accuracy (including UNCERTAIN)
        overall_accuracy = correct_predictions / total_samples
        
        # Inference time statistics
        if self.inference_times:
            avg_inference_time = statistics.mean(self.inference_times)
            median_inference_time = statistics.median(self.inference_times)
            min_inference_time = min(self.inference_times)
            max_inference_time = max(self.inference_times)
        else:
            avg_inference_time = median_inference_time = min_inference_time = max_inference_time = 0
        
        # Confidence statistics
        confidences = [r['confidence'] for r in valid_results]
        avg_confidence = statistics.mean(confidences) if confidences else 0
        
        # Prediction distribution
        prediction_counts = {
            'REAL': sum(1 for r in valid_results if r['predicted_label'] == 'REAL'),
            'FAKE': sum(1 for r in valid_results if r['predicted_label'] == 'FAKE'),
            'UNCERTAIN': sum(1 for r in valid_results if r['predicted_label'] == 'UNCERTAIN')
        }
        
        metrics = {
            'evaluation_date': datetime.now().isoformat(),
            'total_samples': total_samples,
            'valid_samples': len(valid_results),
            'binary_samples': len(binary_results),
            
            # Classification metrics
            'overall_accuracy': overall_accuracy,
            'binary_accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            
            # Confusion matrix
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            
            # Performance metrics
            'avg_inference_time': avg_inference_time,
            'median_inference_time': median_inference_time,
            'min_inference_time': min_inference_time,
            'max_inference_time': max_inference_time,
            'throughput': 1 / avg_inference_time if avg_inference_time > 0 else 0,
            
            # Confidence metrics
            'avg_confidence': avg_confidence,
            
            # Prediction distribution
            'predictions': prediction_counts,
            
            # Model info
            'model_name': config.MODEL_NAME,
            'confidence_threshold': config.CONFIDENCE_THRESHOLD
        }
        
        return metrics
    
    def _print_report(self, metrics: Dict):
        """
        Print formatted evaluation report
        
        Args:
            metrics (Dict): Calculated metrics
        """
        print("\n" + "=" * 70)
        print("PERFORMANCE EVALUATION REPORT")
        print("=" * 70)
        
        print(f"\n📊 DATASET STATISTICS")
        print(f"   Total Samples: {metrics['total_samples']}")
        print(f"   Valid Results: {metrics['valid_samples']}")
        print(f"   Binary Classification Samples: {metrics['binary_samples']}")
        
        print(f"\n🎯 ACCURACY METRICS")
        print(f"   Overall Accuracy: {metrics['overall_accuracy']:.2%}")
        print(f"   Binary Accuracy: {metrics['binary_accuracy']:.2%}")
        print(f"   Precision: {metrics['precision']:.2%}")
        print(f"   Recall: {metrics['recall']:.2%}")
        print(f"   F1 Score: {metrics['f1_score']:.2%}")
        
        print(f"\n📈 CONFUSION MATRIX")
        print(f"   True Positives (Correctly identified FAKE): {metrics['true_positives']}")
        print(f"   True Negatives (Correctly identified REAL): {metrics['true_negatives']}")
        print(f"   False Positives (Real labeled as FAKE): {metrics['false_positives']}")
        print(f"   False Negatives (Fake labeled as REAL): {metrics['false_negatives']}")
        
        print(f"\n⚡ PERFORMANCE METRICS")
        print(f"   Average Inference Time: {metrics['avg_inference_time']:.3f}s")
        print(f"   Median Inference Time: {metrics['median_inference_time']:.3f}s")
        print(f"   Min Inference Time: {metrics['min_inference_time']:.3f}s")
        print(f"   Max Inference Time: {metrics['max_inference_time']:.3f}s")
        print(f"   Throughput: {metrics['throughput']:.2f} requests/second")
        
        print(f"\n💯 CONFIDENCE METRICS")
        print(f"   Average Confidence: {metrics['avg_confidence']:.2%}")
        
        print(f"\n📊 PREDICTION DISTRIBUTION")
        for label, count in metrics['predictions'].items():
            percentage = (count / metrics['total_samples']) * 100
            print(f"   {label}: {count} ({percentage:.1f}%)")
        
        print(f"\n🔧 MODEL CONFIGURATION")
        print(f"   Model: {metrics['model_name']}")
        print(f"   Confidence Threshold: {metrics['confidence_threshold']}")
        
        print("\n" + "=" * 70)
        print("DETAILED RESULTS PER SAMPLE")
        print("=" * 70)
        
        for result in self.results:
            if 'error' not in result:
                status = "✓" if result['is_correct'] else "✗"
                print(f"\n{status} Sample {result['sample_id']}: {result['title'][:50]}...")
                print(f"   True Label: {result['true_label']}")
                print(f"   Predicted: {result['predicted_label']} ({result['confidence']:.2%} confidence)")
                print(f"   Credibility: {result['credibility_score']}/100")
                print(f"   Inference Time: {result['inference_time']:.3f}s")
                print(f"   Factors: +{result['positive_factors']} / -{result['negative_factors']}")
        
        print("\n" + "=" * 70)
    
    def _save_results(self, metrics: Dict):
        """
        Save evaluation results to file
        
        Args:
            metrics (Dict): Metrics to save
        """
        # Create results directory
        results_dir = config.LOGS_DIR / "evaluation_results"
        results_dir.mkdir(exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = results_dir / f"evaluation_{timestamp}.json"
        
        # Prepare data to save
        data = {
            'metrics': metrics,
            'detailed_results': self.results
        }
        
        # Save to JSON
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"\n✅ Results saved to: {filename}")


# Main execution
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("FAKE CONTENT DETECTION - PERFORMANCE EVALUATION")
    print("=" * 70)
    print("\nInitializing system components...")
    
    # Download NLTK data if needed
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✓ NLTK data ready")
    except:
        print("⚠ Warning: Could not download NLTK data")
    
    # Create evaluator
    evaluator = PerformanceEvaluator()
    
    # Run evaluation
    print("\nStarting evaluation...\n")
    metrics = evaluator.run_evaluation()
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Summary:")
    print(f"   Accuracy: {metrics['binary_accuracy']:.2%}")
    print(f"   F1 Score: {metrics['f1_score']:.2%}")
    print(f"   Avg Inference Time: {metrics['avg_inference_time']:.3f}s")
    print(f"   Throughput: {metrics['throughput']:.2f} requests/sec")
    print("\n✅ Check logs directory for detailed results")
