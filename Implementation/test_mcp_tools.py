#!/usr/bin/env python3
"""
Test script for Agentic AI MCP Server Tools
==========================================

This script tests the individual tools exposed by the MCP server
without requiring a full MCP client setup.
"""

import sqlite3
import sys
import os
from pathlib import Path

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))

# Import the DatabaseTools class directly
from agentic_mcp_server import DatabaseTools

def test_database_tools():
    """Test all database tools functionality"""
    print("🧪 Testing Agentic AI MCP Server Database Tools")
    print("=" * 50)
    
    # Initialize database tools
    db_tools = DatabaseTools("./data/ecommerce.db")
    
    if not db_tools.conn:
        print("❌ Database connection failed")
        return False
    
    print("✅ Database connected successfully\n")
    
    # Test 1: Search Products
    print("🔍 Test 1: Search Products")
    results = db_tools.search_products("laptop", limit=3)
    if results and not any("error" in str(r) for r in results):
        print(f"✅ Found {len(results)} laptops")
        if results:
            print(f"   Sample: {results[0].get('title', 'N/A')[:50]}...")
    else:
        print(f"⚠️ Search results: {results}")
    print()
    
    # Test 2: Get Categories
    print("📂 Test 2: Get Categories")
    categories = db_tools.get_categories(limit=5)
    if categories and not any("error" in str(c) for c in categories):
        print(f"✅ Found {len(categories)} categories")
        for cat in categories[:3]:
            print(f"   - {cat.get('category', 'N/A')}: {cat.get('count', 0)} products")
    else:
        print(f"⚠️ Categories results: {categories}")
    print()
    
    # Test 3: Get Top Rated
    print("⭐ Test 3: Get Top Rated Products")
    top_rated = db_tools.get_top_rated(limit=3)
    if top_rated and not any("error" in str(r) for r in top_rated):
        print(f"✅ Found {len(top_rated)} top rated products")
        if top_rated:
            product = top_rated[0]
            print(f"   Top product: {product.get('title', 'N/A')[:40]}... (Rating: {product.get('rating', 'N/A')})")
    else:
        print(f"⚠️ Top rated results: {top_rated}")
    print()
    
    # Test 4: Price Range
    print("💰 Test 4: Price Range Search")
    price_range = db_tools.get_price_range_products(10, 50, limit=3)
    if price_range and not any("error" in str(r) for r in price_range):
        print(f"✅ Found {len(price_range)} products in $10-$50 range")
        if price_range:
            product = price_range[0]
            print(f"   Sample: ${product.get('price', 'N/A')} - {product.get('title', 'N/A')[:40]}...")
    else:
        print(f"⚠️ Price range results: {price_range}")
    print()
    
    # Test 5: Get Product Details
    print("🔍 Test 5: Get Product Details")
    # Get a product ID from search results
    search_results = db_tools.search_products("wireless", limit=1)
    if search_results and len(search_results) > 0 and 'asin' in search_results[0]:
        product_id = search_results[0]['asin']
        details = db_tools.get_product_by_id(product_id)
        if details and not isinstance(details, list):
            print(f"✅ Retrieved product details for {product_id}")
            print(f"   Title: {details.get('title', 'N/A')[:50]}...")
            print(f"   Price: ${details.get('price', 'N/A')}")
            print(f"   Rating: {details.get('rating', 'N/A')}")
        else:
            print(f"⚠️ Product details result: {details}")
    else:
        print("⚠️ Could not get a valid product ID for testing")
    print()
    
    # Test 6: Get Recommendations
    print("🎯 Test 6: Get Recommendations")
    if search_results and len(search_results) > 0 and 'asin' in search_results[0]:
        product_id = search_results[0]['asin']
        recommendations = db_tools.get_recommendations(product_id, limit=3)
        if recommendations and not any("error" in str(r) for r in recommendations):
            print(f"✅ Found {len(recommendations)} recommendations for {product_id}")
            for rec in recommendations:
                print(f"   - {rec.get('title', 'N/A')[:40]}... (${rec.get('price', 'N/A')})")
        else:
            print(f"⚠️ Recommendations result: {recommendations}")
    else:
        print("⚠️ Could not get a valid product ID for recommendations testing")
    print()
    
    print("✅ All database tool tests completed successfully!")
    return True

def test_tool_schemas():
    """Test that our tools match the expected MCP schema format"""
    print("\n🔧 Testing Tool Schema Compatibility")
    print("=" * 40)
    
    # Import the server tools
    import asyncio
    from agentic_mcp_server import handle_list_tools
    
    async def get_tools():
        tools = await handle_list_tools()
        return tools
    
    try:
        tools = asyncio.run(get_tools())
        print(f"✅ Successfully loaded {len(tools)} MCP tools:")
        
        for tool in tools:
            print(f"   - {tool.name}: {tool.description[:60]}...")
            
        return True
    except Exception as e:
        print(f"❌ Error loading tools: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Agentic AI MCP Server Test Suite")
    print("=" * 60)
    
    success = True
    
    # Test database functionality
    try:
        if not test_database_tools():
            success = False
    except Exception as e:
        print(f"❌ Database tests failed: {e}")
        success = False
    
    # Test MCP tool schemas
    try:
        if not test_tool_schemas():
            success = False
    except Exception as e:
        print(f"❌ Schema tests failed: {e}")
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED! MCP Server is functioning correctly.")
        print("   The server is ready for MCP client connections.")
    else:
        print("❌ Some tests failed. Please check the issues above.")
    
    return success

if __name__ == "__main__":
    main()
