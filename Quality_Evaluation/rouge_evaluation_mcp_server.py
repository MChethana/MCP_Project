"""
ROUGE Evaluation for MCP Server Implementation
==============================================

This script evaluates the text generation quality of the MCP server system
using ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-L) by comparing generated
responses with reference (golden standard) responses.

The evaluation focuses on:
- JSON response quality and structure
- Tool description clarity
- Search result relevance and formatting
- Product recommendation accuracy
- Statistical data presentation
"""

import os
import sys
import json
import sqlite3
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time
import statistics
import subprocess
import threading

# ROUGE evaluation imports
from rouge_score import rouge_scorer
import nltk

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

@dataclass
class TestCase:
    """Test case for MCP server ROUGE evaluation"""
    tool_name: str
    parameters: dict
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

class MCPServerRougeEvaluator:
    """ROUGE evaluation framework for the MCP Server system"""
    
    def __init__(self):
        """Initialize the evaluator"""
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.test_cases = []
        self.results = []
        
        print("🔧 Initializing ROUGE Evaluator for MCP Server...")
        self._check_database()
        self._create_test_cases()
    
    def _check_database(self):
        """Check if database is available"""
        db_path = "./data/ecommerce.db"
        if not os.path.exists(db_path):
            print(f"❌ Database not found at {db_path}")
            print("   Using mock responses for evaluation...")
            self.use_mock = True
        else:
            self.use_mock = False
            # Test database connection
            try:
                conn = sqlite3.connect(db_path, check_same_thread=False)
                count = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
                print(f"✓ Database connected successfully with {count} products")
                conn.close()
            except Exception as e:
                print(f"⚠️  Database connection failed: {e}")
                self.use_mock = True
    
    def _create_test_cases(self):
        """Create comprehensive test cases with reference responses for MCP server"""
        self.test_cases = [
            TestCase(
                tool_name="search_products",
                parameters={"query": "wireless headphones", "max_price": 100, "limit": 5},
                reference_response="""{
  "query": "wireless headphones",
  "result_count": 5,
  "products": [
    {
      "id": 1,
      "asin": "B08C4KWM9T",
      "title": "Sony WH-CH720N Wireless Noise Canceling Headphones",
      "price": 89.99,
      "rating": 4.4,
      "reviews": 15234,
      "category": "Electronics > Audio > Headphones",
      "availability": "In Stock"
    },
    {
      "id": 2,
      "asin": "B07S8VYV5K",
      "title": "JBL Tune 510BT Wireless On-Ear Headphones",
      "price": 39.99,
      "rating": 4.3,
      "reviews": 8762,
      "category": "Electronics > Audio > Headphones",
      "availability": "In Stock"
    },
    {
      "id": 3,
      "asin": "B0856JBVVF",
      "title": "Anker Soundcore Life Q20 Hybrid Active Noise Cancelling Headphones",
      "price": 59.99,
      "rating": 4.2,
      "reviews": 22145,
      "category": "Electronics > Audio > Headphones",
      "availability": "In Stock"
    }
  ]
}""",
                description="Product search with price filter",
                category="search"
            ),
            
            TestCase(
                tool_name="get_product_details",
                parameters={"asin": "B08C4KWM9T"},
                reference_response="""{
  "id": 1,
  "asin": "B08C4KWM9T",
  "title": "Sony WH-CH720N Wireless Noise Canceling Headphones",
  "price": 89.99,
  "rating": 4.4,
  "reviews": 15234,
  "category": "Electronics > Audio > Headphones",
  "availability": "In Stock",
  "description": "Experience premium sound quality with Sony's advanced noise canceling technology. Features 35-hour battery life, quick charge, and comfortable over-ear design perfect for long listening sessions.",
  "features": [
    "Active Noise Cancellation",
    "35-hour battery life",
    "Quick charge (3 min = 1 hour)",
    "Wireless Bluetooth 5.0",
    "Comfortable over-ear design"
  ],
  "dimensions": "7.87 x 2.76 x 9.84 inches",
  "weight": "192g"
}""",
                description="Single product detail retrieval",
                category="product_details"
            ),
            
            TestCase(
                tool_name="get_categories",
                parameters={"limit": 10},
                reference_response="""{
  "total_categories": 10,
  "categories": [
    {
      "category": "Electronics > Audio > Headphones",
      "product_count": 2341
    },
    {
      "category": "Electronics > Computers > Laptops",
      "product_count": 1876
    },
    {
      "category": "Electronics > Mobile > Smartphones",
      "product_count": 1654
    },
    {
      "category": "Home & Garden > Smart Home",
      "product_count": 1432
    },
    {
      "category": "Electronics > Gaming > Consoles",
      "product_count": 987
    },
    {
      "category": "Electronics > Audio > Speakers",
      "product_count": 876
    },
    {
      "category": "Electronics > Cameras > Digital",
      "product_count": 743
    },
    {
      "category": "Electronics > Tablets",
      "product_count": 621
    },
    {
      "category": "Electronics > Wearable Tech",
      "product_count": 534
    },
    {
      "category": "Home & Kitchen > Appliances",
      "product_count": 456
    }
  ]
}""",
                description="Category listing with counts",
                category="categories"
            ),
            
            TestCase(
                tool_name="get_recommendations",
                parameters={"asin": "B08C4KWM9T", "limit": 3},
                reference_response="""{
  "source_product": "B08C4KWM9T",
  "recommendation_count": 3,
  "recommendations": [
    {
      "id": 4,
      "asin": "B0856JBVVF",
      "title": "Anker Soundcore Life Q20 Hybrid Active Noise Cancelling Headphones",
      "price": 59.99,
      "rating": 4.2,
      "reviews": 22145,
      "category": "Electronics > Audio > Headphones",
      "availability": "In Stock"
    },
    {
      "id": 5,
      "asin": "B07S8VYV5K",
      "title": "JBL Tune 510BT Wireless On-Ear Headphones",
      "price": 39.99,
      "rating": 4.3,
      "reviews": 8762,
      "category": "Electronics > Audio > Headphones",
      "availability": "In Stock"
    },
    {
      "id": 6,
      "asin": "B08HMWZBXC",
      "title": "Bose QuietComfort 35 II Wireless Bluetooth Headphones",
      "price": 199.99,
      "rating": 4.5,
      "reviews": 18903,
      "category": "Electronics > Audio > Headphones",
      "availability": "In Stock"
    }
  ]
}""",
                description="Product recommendations based on similarity",
                category="recommendations"
            ),
            
            TestCase(
                tool_name="get_price_statistics",
                parameters={"category": "headphones"},
                reference_response="""{
  "category": "headphones",
  "product_count": 2341,
  "min_price": 9.99,
  "max_price": 549.99,
  "avg_price": 87.45,
  "avg_rating": 4.1
}""",
                description="Price statistics for category",
                category="statistics"
            ),
            
            TestCase(
                tool_name="search_products",
                parameters={"query": "gaming laptop", "category": "laptops", "min_rating": 4.0, "sort_by": "rating", "limit": 3},
                reference_response="""{
  "query": "gaming laptop",
  "result_count": 3,
  "products": [
    {
      "id": 7,
      "asin": "B09RMHD39Y",
      "title": "ASUS ROG Strix G15 Gaming Laptop",
      "price": 899.99,
      "rating": 4.6,
      "reviews": 3456,
      "category": "Electronics > Computers > Laptops",
      "availability": "In Stock"
    },
    {
      "id": 8,
      "asin": "B09SLKNXNT",
      "title": "MSI Katana GF66 Gaming Laptop",
      "price": 749.99,
      "rating": 4.4,
      "reviews": 2187,
      "category": "Electronics > Computers > Laptops",
      "availability": "In Stock"
    },
    {
      "id": 9,
      "asin": "B08VKY13MQ",
      "title": "Acer Predator Helios 300 Gaming Laptop",
      "price": 1199.99,
      "rating": 4.3,
      "reviews": 5634,
      "category": "Electronics > Computers > Laptops",
      "availability": "In Stock"
    }
  ]
}""",
                description="Complex search with multiple filters",
                category="complex_search"
            ),
            
            TestCase(
                tool_name="get_price_statistics",
                parameters={},
                reference_response="""{
  "category": "all",
  "product_count": 15674,
  "min_price": 1.99,
  "max_price": 2999.99,
  "avg_price": 156.78,
  "avg_rating": 4.0
}""",
                description="Overall price statistics",
                category="global_statistics"
            ),
            
            TestCase(
                tool_name="search_products",
                parameters={"query": "smartphone", "min_price": 200, "max_price": 500, "sort_by": "price_low", "limit": 4},
                reference_response="""{
  "query": "smartphone",
  "result_count": 4,
  "products": [
    {
      "id": 10,
      "asin": "B09HJZPKMB",
      "title": "Google Pixel 6a 5G Android Phone",
      "price": 299.99,
      "rating": 4.3,
      "reviews": 8934,
      "category": "Electronics > Mobile > Smartphones",
      "availability": "In Stock"
    },
    {
      "id": 11,
      "asin": "B0BDJ7HBXY",
      "title": "OnePlus Nord CE 3 5G Smartphone",
      "price": 349.99,
      "rating": 4.2,
      "reviews": 4567,
      "category": "Electronics > Mobile > Smartphones",
      "availability": "In Stock"
    },
    {
      "id": 12,
      "asin": "B09R93CY26",
      "title": "Samsung Galaxy A54 5G Android Smartphone",
      "price": 389.99,
      "rating": 4.1,
      "reviews": 6789,
      "category": "Electronics > Mobile > Smartphones",
      "availability": "In Stock"
    },
    {
      "id": 13,
      "asin": "B0C4PLKX2Y",
      "title": "Xiaomi Redmi Note 12 Pro Smartphone",
      "price": 449.99,
      "rating": 4.4,
      "reviews": 3421,
      "category": "Electronics > Mobile > Smartphones",
      "availability": "In Stock"
    }
  ]
}""",
                description="Price range search with sorting",
                category="price_range"
            )
        ]
        
        print(f"✓ Created {len(self.test_cases)} test cases for MCP server evaluation")
    
    def _get_mock_response(self, tool_name: str, parameters: dict) -> str:
        """Generate mock responses when actual MCP server is not available"""
        mock_responses = {
            "search_products": {
                "query": parameters.get("query", ""),
                "result_count": 3,
                "products": [
                    {
                        "id": 1,
                        "asin": "MOCK123",
                        "title": f"Mock Product for {parameters.get('query', 'search')}",
                        "price": 49.99,
                        "rating": 4.2,
                        "reviews": 1000,
                        "category": "Electronics",
                        "availability": "In Stock"
                    }
                ]
            },
            "get_product_details": {
                "id": 1,
                "asin": parameters.get("asin", "MOCK123"),
                "title": "Mock Product Details",
                "price": 49.99,
                "rating": 4.2,
                "reviews": 1000,
                "category": "Electronics",
                "availability": "In Stock"
            },
            "get_categories": {
                "total_categories": 5,
                "categories": [
                    {"category": "Electronics", "product_count": 1000},
                    {"category": "Home & Garden", "product_count": 500}
                ]
            },
            "get_recommendations": {
                "source_product": parameters.get("asin", "MOCK123"),
                "recommendation_count": 2,
                "recommendations": [
                    {
                        "id": 2,
                        "asin": "MOCK456",
                        "title": "Similar Mock Product",
                        "price": 39.99,
                        "rating": 4.1,
                        "category": "Electronics"
                    }
                ]
            },
            "get_price_statistics": {
                "category": parameters.get("category", "all"),
                "product_count": 1000,
                "min_price": 9.99,
                "max_price": 999.99,
                "avg_price": 75.50,
                "avg_rating": 4.0
            }
        }
        
        response = mock_responses.get(tool_name, {"error": "Mock tool not implemented"})
        return json.dumps(response, indent=2)
    
    def execute_mcp_tool(self, tool_name: str, parameters: dict) -> str:
        """Execute MCP server tool and return response"""
        if self.use_mock:
            return self._get_mock_response(tool_name, parameters)
        
        try:
            # Import and use the MCP server functions directly
            sys.path.append('./backend')
            from mcp_server import search_products, get_product_details, get_categories, get_recommendations, get_price_statistics
            
            tool_functions = {
                "search_products": search_products,
                "get_product_details": get_product_details, 
                "get_categories": get_categories,
                "get_recommendations": get_recommendations,
                "get_price_statistics": get_price_statistics
            }
            
            if tool_name in tool_functions:
                result = tool_functions[tool_name](**parameters)
                return result
            else:
                return json.dumps({"error": f"Tool '{tool_name}' not found"})
                
        except Exception as e:
            print(f"⚠️  Error executing {tool_name}: {e}")
            return self._get_mock_response(tool_name, parameters)
    
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
        print(f"   Tool: {test_case.tool_name}")
        print(f"   Parameters: {json.dumps(test_case.parameters)}")
        
        # Generate response
        start_time = time.time()
        generated_response = self.execute_mcp_tool(test_case.tool_name, test_case.parameters)
        generation_time = time.time() - start_time
        
        # Calculate ROUGE scores
        rouge_scores = self.calculate_rouge_scores(generated_response, test_case.reference_response)
        
        result = {
            'test_case': test_case,
            'generated_response': generated_response,
            'rouge_scores': rouge_scores,
            'generation_time': generation_time
        }
        
        print(f"   ✓ Generated response ({generation_time:.3f}s)")
        print(f"   ROUGE-1 F1: {rouge_scores.rouge_1_f:.3f}")
        print(f"   ROUGE-2 F1: {rouge_scores.rouge_2_f:.3f}")
        print(f"   ROUGE-L F1: {rouge_scores.rouge_l_f:.3f}")
        
        return result
    
    def run_evaluation(self) -> Dict:
        """Run complete ROUGE evaluation"""
        print("\n" + "="*80)
        print("ROUGE EVALUATION FOR MCP SERVER")
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
ROUGE EVALUATION REPORT - MCP SERVER
=====================================

EVALUATION OVERVIEW
-------------------
• Total Test Cases: {len(results)}
• System Type: {'Mock Responses' if self.use_mock else 'Live MCP Server'}
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

MCP SERVER TOOL ANALYSIS
-------------------------
Tool Performance Summary:
"""
        
        # Analyze by tool
        tools = {}
        for result in results:
            tool = result['test_case'].tool_name
            if tool not in tools:
                tools[tool] = []
            tools[tool].append(result)
        
        for tool, tool_results in tools.items():
            rouge_1_avg = statistics.mean([r['rouge_scores'].rouge_1_f for r in tool_results])
            report += f"""
• {tool}: ROUGE-1 F1 = {rouge_1_avg:.4f} ({len(tool_results)} test cases)"""

        report += f"""

INTERPRETATION GUIDELINES
-------------------------
ROUGE Score Ranges:
  • 0.0 - 0.2: Poor overlap with reference
  • 0.2 - 0.4: Fair overlap, basic content matching
  • 0.4 - 0.6: Good overlap, solid content alignment
  • 0.6 - 0.8: Very good overlap, high content similarity
  • 0.8 - 1.0: Excellent overlap, near-identical content

MCP Server Specific Metrics:
  • JSON Structure Consistency
  • Parameter Handling Accuracy
  • Data Retrieval Performance
  • Response Format Standardization

DETAILED TEST CASE RESULTS
---------------------------"""
        
        # Add individual test case details
        for i, result in enumerate(results, 1):
            tc = result['test_case']
            scores = result['rouge_scores']
            
            report += f"""

Test Case {i}: {tc.description} ({tc.category})
Tool: {tc.tool_name}
Parameters: {json.dumps(tc.parameters)}
ROUGE Scores: R-1={scores.rouge_1_f:.3f}, R-2={scores.rouge_2_f:.3f}, R-L={scores.rouge_l_f:.3f}
Generation Time: {result['generation_time']:.3f}s

Generated Response:
{result['generated_response']}

Reference Response:
{tc.reference_response}
{'-' * 80}"""
        
        return report
    
    def save_results(self, evaluation_results: Dict, filename: str = "rouge_evaluation_mcp_results.json"):
        """Save evaluation results to JSON file"""
        # Prepare serializable results
        serializable_results = []
        for result in evaluation_results['results']:
            serializable_result = {
                'test_case': {
                    'tool_name': result['test_case'].tool_name,
                    'parameters': result['test_case'].parameters,
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
                'system_type': 'Mock Responses' if self.use_mock else 'Live MCP Server'
            },
            'results': serializable_results,
            'aggregate_stats': evaluation_results['aggregate_stats']
        }
        
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\n✓ Results saved to {filename}")
    
    def save_report(self, report: str, filename: str = "rouge_evaluation_mcp_report.txt"):
        """Save evaluation report to text file"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✓ Report saved to {filename}")

def main():
    """Main function to run ROUGE evaluation"""
    print("🚀 Starting ROUGE Evaluation for MCP Server")
    
    try:
        # Initialize evaluator
        evaluator = MCPServerRougeEvaluator()
        
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
        print(f"   • Average Response Time: {stats['generation_time']['mean']:.3f}s")
        
        return evaluation_results
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()
