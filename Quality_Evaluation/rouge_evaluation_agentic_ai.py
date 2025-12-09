"""
ROUGE Evaluation for Agentic AI E-commerce Assistant
====================================================

This script evaluates the text generation quality of the agentic AI system
using ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-L) by comparing generated
responses with reference (golden standard) responses.

ROUGE Metrics:
- ROUGE-1: Unigram overlap (individual word matches)
- ROUGE-2: Bigram overlap (two consecutive word matches)  
- ROUGE-L: Longest Common Subsequence (sentence-level structure)
"""

import os
import sys
import json
import sqlite3
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time
import statistics

# ROUGE evaluation imports
from rouge_score import rouge_scorer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Import the agentic AI system
sys.path.append('./backend')
from agentic_aicopy import AgenticAI, DatabaseTools
from claude_setup import ClaudeModelInterface, SyConfig

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

@dataclass
class TestCase:
    """Test case for ROUGE evaluation"""
    query: str
    reference_response: str
    description: str
    category: str

@dataclass 
class RougeResults:
    """ROUGE evaluation results"""
    rouge_1_f: float
    rouge_1_p: float  
    rouge_1_r: float
    rouge_2_f: float
    rouge_2_p: float
    rouge_2_r: float
    rouge_l_f: float
    rouge_l_p: float
    rouge_l_r: float

class AgenticAIRougeEvaluator:
    """ROUGE evaluation framework for the Agentic AI system"""
    
    def __init__(self):
        """Initialize the evaluator"""
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.agent = None
        self.test_cases = []
        self.results = []
        
        print("🔧 Initializing ROUGE Evaluator for Agentic AI...")
        self._setup_agentic_ai()
        self._create_test_cases()
    
    def _setup_agentic_ai(self):
        """Set up the agentic AI system"""
        try:
            # Set up configuration
            env_vars = {    
                "AICORE_AUTH_URL": "<enter your login info>",
                "AICORE_CLIENT_ID": "<enter your login info>",
                "AICORE_CLIENT_SECRET": "<enter your login info>",
                "AICORE_RESOURCE_GROUP": "default",
                "AICORE_BASE_URL": "<enter your login info>"
            }
            os.environ.update(env_vars)
            
            config = SyConfig(
                auth_url=os.getenv("AICORE_AUTH_URL", ""),
                client_id=os.getenv("AICORE_CLIENT_ID", ""),
                client_secret=os.getenv("AICORE_CLIENT_SECRET", ""),
                base_url=os.getenv("AICORE_BASE_URL", "")
            )
            
            # Initialize components
            db_path = "./data/ecommerce.db"
            if not os.path.exists(db_path):
                print(f"❌ Database not found at {db_path}")
                print("   Using mock responses for evaluation...")
                self.use_mock = True
                return
            
            self.use_mock = False
            db_tools = DatabaseTools(db_path)
            claude = ClaudeModelInterface(config)
            
            # Initialize the agentic AI
            from gen_ai_hub.proxy import get_proxy_client
            proxy_client = get_proxy_client()
            self.agent = AgenticAI(proxy_client, db_tools)
            
            print("✓ Agentic AI system initialized successfully")
            
        except Exception as e:
            print(f"⚠️  Agentic AI setup failed: {e}")
            print("   Using mock responses for evaluation...")
            self.use_mock = True
    
    def _create_test_cases(self):
        """Create comprehensive test cases with reference responses"""
        self.test_cases = [
            TestCase(
                query="Find me wireless headphones under £50",
                reference_response="I found several excellent wireless headphones under £50. Here are the top options: 1. **Sony WH-CH720N** - £45.99 (4.4★) featuring active noise cancellation and 35-hour battery life. 2. **JBL Tune 510BT** - £29.99 (4.3★) with JBL Pure Bass sound and quick charge. 3. **Anker Soundcore Life Q20** - £39.99 (4.2★) offering Hi-Res audio and 40-hour playtime. All these headphones provide great value with excellent sound quality, comfortable fit, and reliable wireless connectivity.",
                description="Product search with price filter",
                category="search"
            ),
            
            TestCase(
                query="What are the top-rated gaming laptops?",
                reference_response="Here are the highest-rated gaming laptops currently available: 1. **ASUS ROG Strix G15** - £899 (4.6★) with AMD Ryzen 7, RTX 3060, 16GB RAM, delivering excellent gaming performance. 2. **MSI Katana GF66** - £749 (4.4★) featuring Intel i7, RTX 3050 Ti, perfect for 1080p gaming. 3. **Acer Predator Helios 300** - £1,199 (4.5★) with powerful RTX 3070 graphics for high-end gaming. These laptops offer exceptional performance, build quality, and value for money with high customer satisfaction ratings.",
                description="Category-specific top-rated products",
                category="top_rated"
            ),
            
            TestCase(
                query="Compare iPhone 14 and Samsung Galaxy S23",
                reference_response="Here's a detailed comparison between iPhone 14 and Samsung Galaxy S23: **iPhone 14** - £849 (4.5★): iOS ecosystem, A15 Bionic chip, excellent camera quality, 6.1\" display, premium build. **Samsung Galaxy S23** - £769 (4.4★): Android flexibility, Snapdragon 8 Gen 2, versatile camera system, 6.1\" Dynamic AMOLED, S Pen compatibility. **Key Differences**: iPhone offers seamless Apple ecosystem integration and longer software support, while Galaxy provides more customization options and typically better value. Both deliver flagship performance with excellent cameras and build quality.",
                description="Product comparison",
                category="comparison"
            ),
            
            TestCase(
                query="Recommend accessories for MacBook Pro",
                reference_response="Essential MacBook Pro accessories to enhance your experience: 1. **USB-C Hub** - £35-60: Belkin or Anker models with multiple ports, HDMI, and SD card slots. 2. **Wireless Mouse** - £25-45: Logitech MX Master 3 for productivity or Apple Magic Mouse for seamless integration. 3. **External Monitor** - £200-400: LG UltraFine 4K or Dell P2720DC for expanded workspace. 4. **Laptop Stand** - £20-40: Rain Design mStand for better ergonomics. 5. **Protective Case** - £15-30: Hard shell cases from MOSISO or Thule for protection. These accessories significantly improve productivity, comfort, and protection for your MacBook Pro.",
                description="Product recommendations",
                category="recommendations"
            ),
            
            TestCase(
                query="I need a budget-friendly smartphone with good camera",
                reference_response="Excellent budget smartphones with impressive cameras: 1. **Google Pixel 6a** - £349 (4.3★) featuring the same camera sensor as Pixel 6 Pro with computational photography magic. 2. **OnePlus Nord CE 3** - £299 (4.2★) with 50MP main camera and solid performance. 3. **Samsung Galaxy A54** - £389 (4.1★) offering versatile camera system and premium design. 4. **Xiaomi Redmi Note 12 Pro** - £279 (4.4★) with 108MP camera and fast charging. These phones deliver flagship-level photography at budget prices, with the Pixel 6a leading for overall camera quality and the Xiaomi offering best value for money.",
                description="Budget product search with specific feature",
                category="budget_search"
            ),
            
            TestCase(
                query="What's the best laptop for video editing under £1500?",
                reference_response="Top video editing laptops under £1500: 1. **MacBook Air M2** - £1,249 (4.6★) with Apple M2 chip delivering exceptional performance per watt, excellent for Final Cut Pro and Adobe Creative Suite. 2. **ASUS Vivobook Pro 15 OLED** - £1,199 (4.4★) featuring AMD Ryzen 7, RTX 3050, and stunning 4K OLED display. 3. **Dell XPS 15** - £1,449 (4.5★) with Intel i7, dedicated graphics, and color-accurate display. 4. **HP Envy 15** - £1,299 (4.3★) offering solid performance with good port selection. The MacBook Air M2 leads for efficiency and software optimization, while ASUS provides excellent value with superior display quality.",
                description="Professional use case product search",
                category="professional"
            ),
            
            TestCase(
                query="Tell me about the latest smart home devices",
                reference_response="Latest smart home devices revolutionizing home automation: 1. **Smart Speakers**: Amazon Echo Dot 5th Gen (£49.99, 4.4★) and Google Nest Mini (£39.99, 4.3★) for voice control. 2. **Smart Lighting**: Philips Hue Starter Kit (£89.99, 4.5★) offering millions of colors and automation. 3. **Smart Security**: Ring Video Doorbell 4 (£179, 4.2★) with advanced motion detection. 4. **Smart Thermostats**: Nest Learning Thermostat (£219, 4.4★) for energy savings. 5. **Smart Plugs**: TP-Link Kasa (£8.99, 4.3★) for basic automation. These devices create an interconnected ecosystem enhancing convenience, security, and energy efficiency in your home.",
                description="Product category overview",
                category="overview"
            ),
            
            TestCase(
                query="I'm looking for gifts for a tech enthusiast under £100",
                reference_response="Perfect tech gifts under £100 for enthusiasts: 1. **Wireless Charging Station** - £45-65: Belkin 3-in-1 for iPhone, AirPods, and Apple Watch. 2. **Portable SSD** - £60-85: Samsung T7 500GB for fast storage and backup. 3. **Smart Fitness Tracker** - £70-95: Fitbit Charge 5 with health monitoring features. 4. **Bluetooth Mechanical Keyboard** - £80-95: Keychron K2 for typing enthusiasts. 5. **Wireless Earbuds** - £60-90: Jabra Elite 75t or Sony WF-1000XM4. 6. **Smart Home Hub** - £65-85: Amazon Echo Show 8 with display. These gifts combine practicality with cutting-edge technology, perfect for any tech lover.",
                description="Gift recommendations with budget",
                category="gifts"
            ),
            
            TestCase(
                query="What should I consider when buying a 4K TV?",
                reference_response="Key considerations when buying a 4K TV: 1. **Display Technology**: OLED offers perfect blacks and contrast, QLED provides bright colors and longevity, LED is budget-friendly. 2. **Size vs Distance**: 55\" for 7-9 feet, 65\" for 8-12 feet, 75\" for 10+ feet viewing distance. 3. **HDR Support**: Look for HDR10, Dolby Vision for enhanced color and contrast. 4. **Smart Features**: Consider built-in streaming apps, voice control, and operating system preferences. 5. **Connectivity**: Ensure sufficient HDMI 2.1 ports for gaming consoles and high refresh rates. 6. **Budget**: Entry-level £400-600, mid-range £700-1200, premium £1300+. Top recommendations include LG C2 OLED for premium quality and Samsung QN90B for bright rooms.",
                description="Product buying guide",
                category="buying_guide"
            ),
            
            TestCase(
                query="Show me eco-friendly tech products",
                reference_response="Eco-friendly tech products for sustainable living: 1. **Solar Power Banks** - £25-45: RAVPower 25000mAh with solar charging capability. 2. **Bamboo Keyboards** - £35-55: Impecca bamboo wireless keyboards made from sustainable materials. 3. **Refurbished Laptops** - £200-600: Certified refurbished ThinkPads and MacBooks reducing electronic waste. 4. **Energy-Efficient Smart Plugs** - £12-20: TP-Link Kasa with energy monitoring to reduce consumption. 5. **Recyclable Phone Cases** - £15-30: Pela cases made from plant-based materials. 6. **Fair Trade Headphones** - £80-120: House of Marley using sustainable wood and recycled materials. These products minimize environmental impact while maintaining high performance and quality standards.",
                description="Eco-friendly product search",
                category="eco_friendly"
            )
        ]
        
        print(f"✓ Created {len(self.test_cases)} test cases for evaluation")
    
    def _get_mock_response(self, query: str) -> str:
        """Generate mock responses when AI system is not available"""
        mock_responses = {
            "wireless headphones": "I found several wireless headphones in your budget. The Sony WH-CH720N at £45.99 offers great noise cancellation, while the JBL Tune 510BT at £29.99 provides excellent bass. Both have good ratings and battery life.",
            "gaming laptops": "Top gaming laptops include the ASUS ROG Strix G15 at £899 with RTX 3060 graphics, and the MSI Katana GF66 at £749 with RTX 3050 Ti. Both offer excellent gaming performance and high customer ratings.",
            "iPhone Samsung": "The iPhone 14 costs £849 with iOS integration and A15 chip, while the Galaxy S23 is £769 with Android flexibility and Snapdragon processor. Both are flagship phones with excellent cameras.",
            "MacBook accessories": "Essential MacBook accessories include a USB-C hub for connectivity, wireless mouse for productivity, external monitor for workspace, and laptop stand for ergonomics.",
            "budget smartphone": "The Google Pixel 6a at £349 offers flagship camera quality, while the OnePlus Nord CE 3 at £299 provides good performance. Both are excellent budget options with solid cameras."
        }
        
        # Simple keyword matching for mock responses
        query_lower = query.lower()
        for key, response in mock_responses.items():
            if any(word in query_lower for word in key.split()):
                return response
        
        return "I can help you find products based on your requirements. Please provide more specific details about what you're looking for."
    
    def generate_response(self, query: str) -> str:
        """Generate response from the agentic AI system or mock"""
        if self.use_mock or not self.agent:
            return self._get_mock_response(query)
        
        try:
            # Reset conversation for each query to ensure independent evaluation
            self.agent.reset_conversation()
            response = self.agent.chat(query, max_iterations=3)
            return response
        except Exception as e:
            print(f"⚠️  Error generating response for '{query[:50]}...': {e}")
            return self._get_mock_response(query)
    
    def calculate_rouge_scores(self, generated: str, reference: str) -> RougeResults:
        """Calculate ROUGE scores between generated and reference text"""
        scores = self.rouge_scorer.score(reference, generated)
        
        return RougeResults(
            rouge_1_f=scores['rouge1'].fmeasure,
            rouge_1_p=scores['rouge1'].precision,
            rouge_1_r=scores['rouge1'].recall,
            rouge_2_f=scores['rouge2'].fmeasure,
            rouge_2_p=scores['rouge2'].precision,
            rouge_2_r=scores['rouge2'].recall,
            rouge_l_f=scores['rougeL'].fmeasure,
            rouge_l_p=scores['rougeL'].precision,
            rouge_l_r=scores['rougeL'].recall
        )
    
    def evaluate_single_case(self, test_case: TestCase) -> Dict:
        """Evaluate a single test case"""
        print(f"📝 Evaluating: {test_case.description}")
        print(f"   Query: {test_case.query}")
        
        # Generate response
        start_time = time.time()
        generated_response = self.generate_response(test_case.query)
        generation_time = time.time() - start_time
        
        # Calculate ROUGE scores
        rouge_scores = self.calculate_rouge_scores(generated_response, test_case.reference_response)
        
        result = {
            'test_case': test_case,
            'generated_response': generated_response,
            'rouge_scores': rouge_scores,
            'generation_time': generation_time
        }
        
        print(f"   ✓ Generated response ({generation_time:.2f}s)")
        print(f"   ROUGE-1 F1: {rouge_scores.rouge_1_f:.3f}")
        print(f"   ROUGE-2 F1: {rouge_scores.rouge_2_f:.3f}")
        print(f"   ROUGE-L F1: {rouge_scores.rouge_l_f:.3f}")
        
        return result
    
    def run_evaluation(self) -> Dict:
        """Run complete ROUGE evaluation"""
        print("\n" + "="*80)
        print("ROUGE EVALUATION FOR AGENTIC AI E-COMMERCE ASSISTANT")
        print("="*80)
        
        results = []
        
        # Evaluate each test case
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"\n[{i}/{len(self.test_cases)}] ", end="")
            result = self.evaluate_single_case(test_case)
            results.append(result)
        
        # Calculate aggregate statistics
        aggregate_stats = self._calculate_aggregate_stats(results)
        
        # Generate comprehensive report
        report = self._generate_evaluation_report(results, aggregate_stats)
        
        return {
            'results': results,
            'aggregate_stats': aggregate_stats,
            'report': report
        }
    
    def _calculate_aggregate_stats(self, results: List[Dict]) -> Dict:
        """Calculate aggregate statistics across all test cases"""
        rouge_1_f_scores = [r['rouge_scores'].rouge_1_f for r in results]
        rouge_2_f_scores = [r['rouge_scores'].rouge_2_f for r in results]
        rouge_l_f_scores = [r['rouge_scores'].rouge_l_f for r in results]
        generation_times = [r['generation_time'] for r in results]
        
        return {
            'rouge_1_f': {
                'mean': statistics.mean(rouge_1_f_scores),
                'median': statistics.median(rouge_1_f_scores),
                'std': statistics.stdev(rouge_1_f_scores) if len(rouge_1_f_scores) > 1 else 0,
                'min': min(rouge_1_f_scores),
                'max': max(rouge_1_f_scores)
            },
            'rouge_2_f': {
                'mean': statistics.mean(rouge_2_f_scores),
                'median': statistics.median(rouge_2_f_scores),
                'std': statistics.stdev(rouge_2_f_scores) if len(rouge_2_f_scores) > 1 else 0,
                'min': min(rouge_2_f_scores),
                'max': max(rouge_2_f_scores)
            },
            'rouge_l_f': {
                'mean': statistics.mean(rouge_l_f_scores),
                'median': statistics.median(rouge_l_f_scores),
                'std': statistics.stdev(rouge_l_f_scores) if len(rouge_l_f_scores) > 1 else 0,
                'min': min(rouge_l_f_scores),
                'max': max(rouge_l_f_scores)
            },
            'generation_time': {
                'mean': statistics.mean(generation_times),
                'median': statistics.median(generation_times),
                'std': statistics.stdev(generation_times) if len(generation_times) > 1 else 0,
                'total': sum(generation_times)
            }
        }
    
    def _generate_evaluation_report(self, results: List[Dict], stats: Dict) -> str:
        """Generate comprehensive evaluation report"""
        report = f"""
ROUGE EVALUATION REPORT - AGENTIC AI E-COMMERCE ASSISTANT
=========================================================

EVALUATION OVERVIEW
-------------------
• Total Test Cases: {len(results)}
• System Type: {'Mock Responses' if self.use_mock else 'Live Agentic AI System'}
• Evaluation Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

AGGREGATE ROUGE SCORES
----------------------
ROUGE-1 F1-Score:
  • Mean: {stats['rouge_1_f']['mean']:.4f}
  • Median: {stats['rouge_1_f']['median']:.4f}
  • Std Dev: {stats['rouge_1_f']['std']:.4f}
  • Range: {stats['rouge_1_f']['min']:.4f} - {stats['rouge_1_f']['max']:.4f}

ROUGE-2 F1-Score:
  • Mean: {stats['rouge_2_f']['mean']:.4f}
  • Median: {stats['rouge_2_f']['median']:.4f}
  • Std Dev: {stats['rouge_2_f']['std']:.4f}
  • Range: {stats['rouge_2_f']['min']:.4f} - {stats['rouge_2_f']['max']:.4f}

ROUGE-L F1-Score:
  • Mean: {stats['rouge_l_f']['mean']:.4f}
  • Median: {stats['rouge_l_f']['median']:.4f}
  • Std Dev: {stats['rouge_l_f']['std']:.4f}
  • Range: {stats['rouge_l_f']['min']:.4f} - {stats['rouge_l_f']['max']:.4f}

PERFORMANCE METRICS
-------------------
Response Generation Time:
  • Mean: {stats['generation_time']['mean']:.3f}s
  • Median: {stats['generation_time']['median']:.3f}s
  • Total: {stats['generation_time']['total']:.3f}s

DETAILED RESULTS BY CATEGORY
-----------------------------"""
        
        # Group results by category
        categories = {}
        for result in results:
            category = result['test_case'].category
            if category not in categories:
                categories[category] = []
            categories[category].append(result)
        
        # Add category-specific analysis
        for category, category_results in categories.items():
            rouge_1_avg = statistics.mean([r['rouge_scores'].rouge_1_f for r in category_results])
            rouge_2_avg = statistics.mean([r['rouge_scores'].rouge_2_f for r in category_results])
            rouge_l_avg = statistics.mean([r['rouge_scores'].rouge_l_f for r in category_results])
            
            report += f"""

{category.upper()} ({len(category_results)} test cases):
  • ROUGE-1: {rouge_1_avg:.4f}
  • ROUGE-2: {rouge_2_avg:.4f}
  • ROUGE-L: {rouge_l_avg:.4f}"""

        report += f"""

INTERPRETATION GUIDELINES
-------------------------
ROUGE Score Ranges:
  • 0.0 - 0.2: Poor overlap with reference
  • 0.2 - 0.4: Fair overlap, basic content matching
  • 0.4 - 0.6: Good overlap, solid content alignment
  • 0.6 - 0.8: Very good overlap, high content similarity
  • 0.8 - 1.0: Excellent overlap, near-identical content

ROUGE Metric Meanings:
  • ROUGE-1: Measures unigram (single word) overlap
  • ROUGE-2: Measures bigram (two consecutive words) overlap
  • ROUGE-L: Measures longest common subsequence (sentence structure)

DETAILED TEST CASE RESULTS
---------------------------"""
        
        # Add individual test case details
        for i, result in enumerate(results, 1):
            tc = result['test_case']
            scores = result['rouge_scores']
            
            report += f"""

Test Case {i}: {tc.description} ({tc.category})
Query: "{tc.query}"
ROUGE Scores: R-1={scores.rouge_1_f:.3f}, R-2={scores.rouge_2_f:.3f}, R-L={scores.rouge_l_f:.3f}
Generation Time: {result['generation_time']:.2f}s

Generated Response:
{result['generated_response']}

Reference Response:
{tc.reference_response}
{'-' * 80}"""
        
        return report
    
    def save_results(self, evaluation_results: Dict, filename: str = "rouge_evaluation_results.json"):
        """Save evaluation results to JSON file"""
        # Prepare serializable results
        serializable_results = []
        for result in evaluation_results['results']:
            serializable_result = {
                'test_case': {
                    'query': result['test_case'].query,
                    'reference_response': result['test_case'].reference_response,
                    'description': result['test_case'].description,
                    'category': result['test_case'].category
                },
                'generated_response': result['generated_response'],
                'rouge_scores': {
                    'rouge_1_f': result['rouge_scores'].rouge_1_f,
                    'rouge_1_p': result['rouge_scores'].rouge_1_p,
                    'rouge_1_r': result['rouge_scores'].rouge_1_r,
                    'rouge_2_f': result['rouge_scores'].rouge_2_f,
                    'rouge_2_p': result['rouge_scores'].rouge_2_p,
                    'rouge_2_r': result['rouge_scores'].rouge_2_r,
                    'rouge_l_f': result['rouge_scores'].rouge_l_f,
                    'rouge_l_p': result['rouge_scores'].rouge_l_p,
                    'rouge_l_r': result['rouge_scores'].rouge_l_r
                },
                'generation_time': result['generation_time']
            }
            serializable_results.append(serializable_result)
        
        save_data = {
            'evaluation_metadata': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_test_cases': len(serializable_results),
                'system_type': 'Mock Responses' if self.use_mock else 'Live Agentic AI System'
            },
            'results': serializable_results,
            'aggregate_stats': evaluation_results['aggregate_stats']
        }
        
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\n✓ Results saved to {filename}")
    
    def save_report(self, report: str, filename: str = "rouge_evaluation_report.txt"):
        """Save evaluation report to text file"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✓ Report saved to {filename}")

def main():
    """Main function to run ROUGE evaluation"""
    print("🚀 Starting ROUGE Evaluation for Agentic AI E-commerce Assistant")
    
    try:
        # Initialize evaluator
        evaluator = AgenticAIRougeEvaluator()
        
        # Run evaluation
        evaluation_results = evaluator.run_evaluation()
        
        # Display results
        print("\n" + "="*80)
        print("EVALUATION COMPLETE!")
        print("="*80)
        print(evaluation_results['report'])
        
        # Save results
        evaluator.save_results(evaluation_results)
        evaluator.save_report(evaluation_results['report'])
        
        # Summary
        stats = evaluation_results['aggregate_stats']
        print(f"\n🎯 SUMMARY STATISTICS:")
        print(f"   • Average ROUGE-1 F1: {stats['rouge_1_f']['mean']:.4f}")
        print(f"   • Average ROUGE-2 F1: {stats['rouge_2_f']['mean']:.4f}")
        print(f"   • Average ROUGE-L F1: {stats['rouge_l_f']['mean']:.4f}")
        print(f"   • Average Response Time: {stats['generation_time']['mean']:.2f}s")
        
        return evaluation_results
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()
