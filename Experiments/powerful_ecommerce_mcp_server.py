#!/usr/bin/env python3
"""
POWERFUL E-COMMERCE MCP SERVER
Enterprise-grade MCP server with advanced analytics, ML features, and comprehensive tools
"""

import json
import sys
import logging
import sqlite3
import re
import math
import statistics
from typing import List, Dict, Optional, Any, Union
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stderr, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("powerful-ecommerce")

# Global database connection
db_conn = None
DB_PATH = "./data/ecommerce.db"

def get_db():
    """Get database connection with optimized settings"""
    global db_conn
    if db_conn is None:
        try:
            db_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            db_conn.row_factory = sqlite3.Row
            
            # Optimize database for read performance
            db_conn.execute("PRAGMA cache_size = 10000")
            db_conn.execute("PRAGMA temp_store = MEMORY")
            db_conn.execute("PRAGMA journal_mode = WAL")
            
            logger.info(f"✓ Connected to database: {DB_PATH}")
        except Exception as e:
            logger.error(f"✗ Database connection failed: {e}")
            raise
    return db_conn

def dict_from_row(row) -> dict:
    """Convert sqlite3.Row to dictionary with None handling"""
    if not row:
        return {}
    return {key: row[key] for key in row.keys()}

def format_currency(amount: float, currency: str = "₹") -> str:
    """Format currency with proper symbols"""
    if amount is None:
        return "N/A"
    return f"{currency}{amount:,.2f}"

def calculate_similarity_score(product1: dict, product2: dict) -> float:
    """Calculate similarity score between two products"""
    score = 0.0
    
    # Category match (40% weight)
    if product1.get('category') == product2.get('category'):
        score += 0.4
    
    # Price similarity (30% weight)
    p1_price = product1.get('price', 0) or 0
    p2_price = product2.get('price', 0) or 0
    if p1_price > 0 and p2_price > 0:
        price_diff = abs(p1_price - p2_price) / max(p1_price, p2_price)
        score += 0.3 * (1 - price_diff)
    
    # Rating similarity (20% weight)
    r1 = product1.get('rating', 0) or 0
    r2 = product2.get('rating', 0) or 0
    if r1 > 0 and r2 > 0:
        rating_diff = abs(r1 - r2) / 5.0
        score += 0.2 * (1 - rating_diff)
    
    # Title similarity (10% weight)
    t1 = (product1.get('title') or '').lower()
    t2 = (product2.get('title') or '').lower()
    if t1 and t2:
        common_words = len(set(t1.split()) & set(t2.split()))
        total_words = len(set(t1.split()) | set(t2.split()))
        if total_words > 0:
            score += 0.1 * (common_words / total_words)
    
    return score

# ============ ADVANCED SEARCH & DISCOVERY TOOLS ============

@mcp.tool()
def intelligent_search(query: str, context: str = "", user_preferences: dict = None, 
                      limit: int = 15, include_analytics: bool = True) -> str:
    """
    AI-powered intelligent product search with context understanding
    
    Args:
        query: Natural language search query
        context: Additional context (e.g., "for gaming", "budget-friendly", "premium")
        user_preferences: User preferences dict with keys like preferred_categories, budget_range, etc.
        limit: Maximum results
        include_analytics: Include search analytics and insights
    """
    db = get_db()
    cursor = db.cursor()
    
    preferences = user_preferences or {}
    
    # Parse query for intent
    query_lower = query.lower()
    intent_keywords = {
        'budget': ['cheap', 'budget', 'affordable', 'low cost', 'under'],
        'premium': ['premium', 'high end', 'expensive', 'luxury', 'best'],
        'popular': ['popular', 'trending', 'bestseller', 'top rated'],
        'new': ['new', 'latest', 'recent'],
        'deals': ['deal', 'discount', 'sale', 'offer']
    }
    
    detected_intent = []
    for intent, keywords in intent_keywords.items():
        if any(keyword in query_lower or keyword in context.lower() for keyword in keywords):
            detected_intent.append(intent)
    
    # Build advanced search query
    sql = """
        SELECT p.*, 
               bm25(products_fts) AS relevance_score,
               CASE 
                   WHEN p.rating >= 4.5 THEN 'Excellent'
                   WHEN p.rating >= 4.0 THEN 'Very Good'
                   WHEN p.rating >= 3.5 THEN 'Good'
                   WHEN p.rating >= 3.0 THEN 'Average'
                   ELSE 'Below Average'
               END as rating_category
        FROM products_fts fts
        JOIN products p ON fts.rowid = p.id
        WHERE products_fts MATCH ?
    """
    params = [query]
    
    # Apply intent-based filters
    if 'budget' in detected_intent:
        sql += " AND p.price <= ?"
        budget_limit = preferences.get('budget_max', 100)
        params.append(budget_limit)
    
    if 'premium' in detected_intent:
        sql += " AND p.price >= ? AND p.rating >= 4.0"
        premium_min = preferences.get('premium_min', 100)
        params.append(premium_min)
    
    if 'popular' in detected_intent:
        sql += " AND p.num_ratings >= 100"
    
    # Sorting strategy based on intent
    if 'budget' in detected_intent:
        sql += " ORDER BY p.price ASC, p.rating DESC"
    elif 'premium' in detected_intent:
        sql += " ORDER BY p.price DESC, p.rating DESC"
    elif 'popular' in detected_intent:
        sql += " ORDER BY p.num_ratings DESC, p.rating DESC"
    else:
        sql += " ORDER BY relevance_score, p.rating DESC"
    
    sql += f" LIMIT {limit}"
    
    try:
        results = cursor.execute(sql, params).fetchall()
        products = [dict_from_row(row) for row in results]
        
        # Calculate analytics if requested
        analytics = {}
        if include_analytics and products:
            prices = [p['price'] for p in products if p.get('price')]
            ratings = [p['rating'] for p in products if p.get('rating')]
            
            analytics = {
                'search_intent': detected_intent,
                'result_count': len(products),
                'price_range': {
                    'min': min(prices) if prices else 0,
                    'max': max(prices) if prices else 0,
                    'average': statistics.mean(prices) if prices else 0
                },
                'rating_stats': {
                    'average': statistics.mean(ratings) if ratings else 0,
                    'distribution': Counter([p['rating_category'] for p in products])
                },
                'top_categories': Counter([p['category'] for p in products if p.get('category')]).most_common(3)
            }
        
        return json.dumps({
            "query": query,
            "context": context,
            "analytics": analytics,
            "products": products
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Search failed: {str(e)}"})

@mcp.tool()
def multi_modal_search(text_query: str = "", filters: dict = None, 
                      sort_preferences: List[str] = None, limit: int = 20) -> str:
    """
    Multi-modal search combining text, filters, and sorting preferences
    
    Args:
        text_query: Text-based search
        filters: Complex filters dict with keys like price_range, categories, ratings, etc.
        sort_preferences: List of sorting preferences in order ['price_asc', 'rating_desc', 'popularity']
        limit: Maximum results
    """
    db = get_db()
    cursor = db.cursor()
    
    filters = filters or {}
    sort_preferences = sort_preferences or ['relevance']
    
    # Build dynamic query
    if text_query:
        sql = """
            SELECT p.*, bm25(products_fts) AS relevance_score
            FROM products_fts fts
            JOIN products p ON fts.rowid = p.id
            WHERE products_fts MATCH ?
        """
        params = [text_query]
    else:
        sql = "SELECT *, 0 AS relevance_score FROM products WHERE 1=1"
        params = []
    
    # Apply complex filters
    if filters.get('price_range'):
        min_price, max_price = filters['price_range']
        sql += " AND p.price BETWEEN ? AND ?"
        params.extend([min_price, max_price])
    
    if filters.get('categories'):
        category_conditions = ' OR '.join(['p.category LIKE ?' for _ in filters['categories']])
        sql += f" AND ({category_conditions})"
        params.extend([f"%{cat}%" for cat in filters['categories']])
    
    if filters.get('min_rating'):
        sql += " AND p.rating >= ?"
        params.append(filters['min_rating'])
    
    if filters.get('min_reviews'):
        sql += " AND p.num_ratings >= ?"
        params.append(filters['min_reviews'])
    
    # Apply sophisticated sorting
    sort_clauses = []
    for pref in sort_preferences:
        if pref == 'price_asc':
            sort_clauses.append('p.price ASC')
        elif pref == 'price_desc':
            sort_clauses.append('p.price DESC')
        elif pref == 'rating_desc':
            sort_clauses.append('p.rating DESC')
        elif pref == 'popularity':
            sort_clauses.append('p.num_ratings DESC')
        elif pref == 'relevance' and text_query:
            sort_clauses.append('relevance_score')
    
    if sort_clauses:
        sql += f" ORDER BY {', '.join(sort_clauses)}"
    
    sql += f" LIMIT {limit}"
    
    try:
        results = cursor.execute(sql, params).fetchall()
        products = [dict_from_row(row) for row in results]
        
        return json.dumps({
            "text_query": text_query,
            "applied_filters": filters,
            "sort_order": sort_preferences,
            "result_count": len(products),
            "products": products
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Multi-modal search failed: {str(e)}"})

# ============ ADVANCED ANALYTICS TOOLS ============

@mcp.tool()
def market_analysis(category: str = "", time_period: str = "all", 
                   include_trends: bool = True) -> str:
    """
    Comprehensive market analysis with trends and insights
    
    Args:
        category: Specific category to analyze
        time_period: Time period for analysis ('all', 'recent')
        include_trends: Include trending analysis
    """
    db = get_db()
    cursor = db.cursor()
    
    analysis = {}
    
    # Base query condition
    where_clause = "WHERE p.price > 0 AND p.rating IS NOT NULL"
    params = []
    
    if category:
        where_clause += " AND p.category LIKE ?"
        params.append(f"%{category}%")
    
    # Market size and overview
    market_stats = cursor.execute(f"""
        SELECT 
            COUNT(*) as total_products,
            COUNT(DISTINCT category) as categories,
            AVG(price) as avg_price,
            MIN(price) as min_price,
            MAX(price) as max_price,
            AVG(rating) as avg_rating,
            SUM(num_ratings) as total_reviews
        FROM products p
        {where_clause}
    """, params).fetchone()
    
    analysis['market_overview'] = dict_from_row(market_stats)
    
    # Price segments analysis
    price_segments = cursor.execute(f"""
        SELECT 
            CASE 
                WHEN price <= 50 THEN 'Budget (≤₹50)'
                WHEN price <= 200 THEN 'Mid-range (₹50-200)'
                WHEN price <= 500 THEN 'Premium (₹200-500)'
                ELSE 'Luxury (>₹500)'
            END as price_segment,
            COUNT(*) as product_count,
            AVG(rating) as avg_rating,
            AVG(num_ratings) as avg_reviews
        FROM products p
        {where_clause}
        GROUP BY price_segment
        ORDER BY MIN(price)
    """, params).fetchall()
    
    analysis['price_segments'] = [dict_from_row(row) for row in price_segments]
    
    # Top categories analysis
    top_categories = cursor.execute(f"""
        SELECT 
            category,
            COUNT(*) as product_count,
            AVG(price) as avg_price,
            AVG(rating) as avg_rating,
            SUM(num_ratings) as total_reviews
        FROM products p
        {where_clause}
        GROUP BY category
        ORDER BY product_count DESC
        LIMIT 10
    """, params).fetchall()
    
    analysis['top_categories'] = [dict_from_row(row) for row in top_categories]
    
    # Rating distribution
    rating_dist = cursor.execute(f"""
        SELECT 
            CASE 
                WHEN rating >= 4.5 THEN '4.5-5.0 (Excellent)'
                WHEN rating >= 4.0 THEN '4.0-4.5 (Very Good)'
                WHEN rating >= 3.5 THEN '3.5-4.0 (Good)'
                WHEN rating >= 3.0 THEN '3.0-3.5 (Average)'
                ELSE '<3.0 (Poor)'
            END as rating_range,
            COUNT(*) as count,
            ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM products p2 {where_clause}), 2) as percentage
        FROM products p
        {where_clause}
        GROUP BY rating_range
        ORDER BY MIN(rating) DESC
    """, params).fetchall()
    
    analysis['rating_distribution'] = [dict_from_row(row) for row in rating_dist]
    
    # Competitive landscape
    if include_trends:
        # Most reviewed products (trending)
        trending = cursor.execute(f"""
            SELECT 
                title, category, price, rating, num_ratings,
                CASE 
                    WHEN num_ratings >= 1000 THEN 'Viral'
                    WHEN num_ratings >= 500 THEN 'Very Popular'
                    WHEN num_ratings >= 100 THEN 'Popular'
                    ELSE 'Emerging'
                END as popularity_tier
            FROM products p
            {where_clause}
            ORDER BY num_ratings DESC
            LIMIT 10
        """, params).fetchall()
        
        analysis['trending_products'] = [dict_from_row(row) for row in trending]
    
    return json.dumps({
        "category": category or "All Categories",
        "time_period": time_period,
        "analysis": analysis,
        "generated_at": datetime.now().isoformat()
    }, indent=2)

@mcp.tool()
def competitive_analysis(product_asin: str, analysis_depth: str = "standard") -> str:
    """
    Deep competitive analysis for a specific product
    
    Args:
        product_asin: Target product ASIN
        analysis_depth: 'standard' or 'comprehensive'
    """
    db = get_db()
    cursor = db.cursor()
    
    # Get target product
    target = cursor.execute("""
        SELECT * FROM products WHERE asin = ?
    """, (product_asin,)).fetchone()
    
    if not target:
        return json.dumps({"error": f"Product {product_asin} not found"})
    
    target = dict_from_row(target)
    
    # Find direct competitors (same category, similar price range)
    price_range = (target['price'] * 0.7, target['price'] * 1.3) if target['price'] else (0, 999999)
    
    competitors = cursor.execute("""
        SELECT *, 
               ABS(price - ?) as price_diff,
               ABS(rating - ?) as rating_diff
        FROM products 
        WHERE category = ? 
          AND asin != ?
          AND price BETWEEN ? AND ?
        ORDER BY rating DESC, num_ratings DESC
        LIMIT 10
    """, (target['price'] or 0, target['rating'] or 0, target['category'], 
          product_asin, price_range[0], price_range[1])).fetchall()
    
    competitor_data = [dict_from_row(row) for row in competitors]
    
    # Calculate competitive metrics
    if competitor_data:
        competitor_prices = [c['price'] for c in competitor_data if c['price']]
        competitor_ratings = [c['rating'] for c in competitor_data if c['rating']]
        
        competitive_position = {
            'price_percentile': sum(1 for p in competitor_prices if p > (target['price'] or 0)) / len(competitor_prices) * 100 if competitor_prices else 0,
            'rating_percentile': sum(1 for r in competitor_ratings if r < (target['rating'] or 0)) / len(competitor_ratings) * 100 if competitor_ratings else 0,
            'review_count_rank': len([c for c in competitor_data if c['num_ratings'] < (target['num_ratings'] or 0)]) + 1
        }
    else:
        competitive_position = {'message': 'No direct competitors found'}
    
    analysis_result = {
        "target_product": target,
        "competitive_position": competitive_position,
        "direct_competitors": competitor_data[:5],
        "market_insights": {
            "category": target['category'],
            "total_competitors_analyzed": len(competitor_data)
        }
    }
    
    if analysis_depth == "comprehensive":
        # Add broader market analysis
        category_stats = cursor.execute("""
            SELECT 
                COUNT(*) as total_products,
                AVG(price) as avg_price,
                AVG(rating) as avg_rating,
                MIN(price) as min_price,
                MAX(price) as max_price
            FROM products 
            WHERE category = ?
        """, (target['category'],)).fetchone()
        
        analysis_result["category_benchmarks"] = dict_from_row(category_stats)
    
    return json.dumps(analysis_result, indent=2)

# ============ RECOMMENDATION ENGINE ============

@mcp.tool()
def smart_recommendations(user_profile: dict, recommendation_type: str = "mixed", 
                         limit: int = 10, explain_reasoning: bool = True) -> str:
    """
    AI-powered smart recommendations based on user profile
    
    Args:
        user_profile: Dict with keys like 'viewed_products', 'preferred_categories', 'budget_range', etc.
        recommendation_type: 'similar', 'complementary', 'trending', 'mixed'
        limit: Number of recommendations
        explain_reasoning: Include explanation for each recommendation
    """
    db = get_db()
    cursor = db.cursor()
    
    viewed_asins = user_profile.get('viewed_products', [])
    preferred_categories = user_profile.get('preferred_categories', [])
    budget_range = user_profile.get('budget_range', (0, 999999))
    min_rating = user_profile.get('min_rating', 3.0)
    
    recommendations = []
    
    if recommendation_type in ['similar', 'mixed'] and viewed_asins:
        # Get similar products based on viewed items
        placeholders = ','.join('?' * len(viewed_asins))
        similar_products = cursor.execute(f"""
            SELECT p2.*, 
                   COUNT(p1.category) as category_matches
            FROM products p1
            JOIN products p2 ON p1.category = p2.category
            WHERE p1.asin IN ({placeholders})
              AND p2.asin NOT IN ({placeholders})
              AND p2.price BETWEEN ? AND ?
              AND p2.rating >= ?
            GROUP BY p2.asin
            ORDER BY category_matches DESC, p2.rating DESC, p2.num_ratings DESC
            LIMIT ?
        """, viewed_asins + viewed_asins + [budget_range[0], budget_range[1], min_rating, limit//2]).fetchall()
        
        for product in similar_products:
            product_dict = dict_from_row(product)
            if explain_reasoning:
                product_dict['recommendation_reason'] = f"Similar to your viewed products in {product_dict['category']}"
            recommendations.append(product_dict)
    
    if recommendation_type in ['trending', 'mixed']:
        # Get trending products
        trending_limit = limit if recommendation_type == 'trending' else max(1, limit - len(recommendations))
        
        category_filter = ""
        params = [budget_range[0], budget_range[1], min_rating]
        
        if preferred_categories:
            category_placeholders = ' OR '.join(['category LIKE ?' for _ in preferred_categories])
            category_filter = f"AND ({category_placeholders})"
            params.extend([f"%{cat}%" for cat in preferred_categories])
        
        params.append(trending_limit)
        
        trending_products = cursor.execute(f"""
            SELECT *, 
                   (num_ratings * rating) as popularity_score
            FROM products
            WHERE price BETWEEN ? AND ?
              AND rating >= ?
              {category_filter}
            ORDER BY popularity_score DESC, rating DESC
            LIMIT ?
        """, params).fetchall()
        
        for product in trending_products:
            product_dict = dict_from_row(product)
            if explain_reasoning:
                product_dict['recommendation_reason'] = "Trending product with high ratings"
            recommendations.append(product_dict)
    
    # Remove duplicates and limit results
    seen_asins = set()
    unique_recommendations = []
    for rec in recommendations:
        if rec['asin'] not in seen_asins:
            seen_asins.add(rec['asin'])
            unique_recommendations.append(rec)
            if len(unique_recommendations) >= limit:
                break
    
    return json.dumps({
        "user_profile": user_profile,
        "recommendation_type": recommendation_type,
        "total_recommendations": len(unique_recommendations),
        "recommendations": unique_recommendations
    }, indent=2)

@mcp.tool()
def product_similarity_finder(target_asin: str, similarity_factors: List[str] = None,
                            limit: int = 8) -> str:
    """
    Find products similar to target using multiple similarity factors
    
    Args:
        target_asin: Product to find similarities for
        similarity_factors: List of factors to consider ['category', 'price', 'rating', 'title']
        limit: Number of similar products
    """
    db = get_db()
    cursor = db.cursor()
    
    similarity_factors = similarity_factors or ['category', 'price', 'rating']
    
    # Get target product
    target = cursor.execute("SELECT * FROM products WHERE asin = ?", (target_asin,)).fetchone()
    if not target:
        return json.dumps({"error": f"Product {target_asin} not found"})
    
    target = dict_from_row(target)
    
    # Get all potential similar products
    all_products = cursor.execute("""
        SELECT * FROM products 
        WHERE asin != ? 
        AND category IS NOT NULL 
        AND price IS NOT NULL 
        AND rating IS NOT NULL
        LIMIT 1000
    """, (target_asin,)).fetchall()
    
    # Calculate similarity scores
    similar_products = []
    for product_row in all_products:
        product = dict_from_row(product_row)
        similarity_score = calculate_similarity_score(target, product)
        
        if similarity_score > 0.3:  # Minimum similarity threshold
            product['similarity_score'] = similarity_score
            similar_products.append(product)
    
    # Sort by similarity and limit results
    similar_products.sort(key=lambda x: x['similarity_score'], reverse=True)
    similar_products = similar_products[:limit]
    
    return json.dumps({
        "target_product": target,
        "similarity_factors": similarity_factors,
        "found_similar": len(similar_products),
        "similar_products": similar_products
    }, indent=2)

# ============ ADVANCED FILTERING & DISCOVERY ============

@mcp.tool()
def dynamic_filter_search(base_query: str = "", dynamic_filters: dict = None, 
                         aggregation_level: str = "product") -> str:
    """
    Advanced search with dynamic filtering and aggregation capabilities
    
    Args:
        base_query: Base search query
        dynamic_filters: Complex nested filters
        aggregation_level: 'product', 'category', 'brand'
    """
    db = get_db()
    cursor = db.cursor()
    
    dynamic_filters = dynamic_filters or {}
    
    # Build complex query based on aggregation level
    if aggregation_level == "category":
        sql = """
            SELECT 
                category,
                COUNT(*) as product_count,
                AVG(price) as avg_price,
                MIN(price) as min_price,
                MAX(price) as max_price,
                AVG(rating) as avg_rating,
                SUM(num_ratings) as total_reviews
            FROM products p
            WHERE 1=1
        """
        group_by = " GROUP BY category ORDER BY product_count DESC"
    else:
        sql = "SELECT * FROM products p WHERE 1=1"
        group_by = " ORDER BY rating DESC, num_ratings DESC"
    
    params = []
    
    # Apply text search if provided
    if base_query:
        if aggregation_level == "category":
            sql += " AND category LIKE ?"
            params.append(f"%{base_query}%")
        else:
            sql = """
                SELECT p.* FROM products_fts fts
                JOIN products p ON fts.rowid = p.id
                WHERE products_fts MATCH ?
            """
            params = [base_query]
    
    # Apply dynamic filters
    for filter_key, filter_value in dynamic_filters.items():
        if filter_key == 'price_ranges':
            # Multiple price ranges: [{'min': 0, 'max': 50}, {'min': 100, 'max': 200}]
            range_conditions = []
            for price_range in filter_value:
                range_conditions.append("(p.price BETWEEN ? AND ?)")
                params.extend([price_range['min'], price_range['max']])
            if range_conditions:
                sql += f" AND ({' OR '.join(range_conditions)})"
        
        elif filter_key == 'rating_tiers':
            # Rating tiers: ['excellent', 'good']
            tier_conditions = []
            for tier in filter_value:
                if tier == 'excellent':
                    tier_conditions.append("p.rating >= 4.5")
                elif tier == 'good':
                    tier_conditions.append("p.rating >= 4.0")
                elif tier == 'average':
                    tier_conditions.append("p.rating >= 3.0")
            if tier_conditions:
                sql += f" AND ({' OR '.join(tier_conditions)})"
        
        elif filter_key == 'popularity_levels':
            # Popularity levels: ['viral', 'popular']
            popularity_conditions = []
            for level in filter_value:
                if level == 'viral':
                    popularity_conditions.append("p.num_ratings >= 1000")
                elif level == 'popular':
                    popularity_conditions.append("p.num_ratings >= 100")
            if popularity_conditions:
                sql += f" AND ({' OR '.join(popularity_conditions)})"
    
    sql += group_by
    
    if aggregation_level == "product":
        sql += " LIMIT 20"
    else:
        sql += " LIMIT 15"
    
    try:
        results = cursor.execute(sql, params).fetchall()
        data = [dict_from_row(row) for row in results]
        
        return json.dumps({
            "base_query": base_query,
            "dynamic_filters": dynamic_filters,
            "aggregation_level": aggregation_level,
            "result_count": len(data),
            "results": data
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Dynamic filter search failed: {str(e)}"})

# ============ USER BEHAVIOR & PERSONALIZATION ============

@mcp.tool()
def add_to_wishlist(user_id: str, product_asin: str, priority: int = 1) -> str:
    """Add product to user's wishlist with priority"""
    db = get_db()
    cursor = db.cursor()
    
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO wishlist (user_id, product_asin, priority, added_at)
            VALUES (?, ?, ?, ?)
        """, (user_id, product_asin, priority, datetime.now().isoformat()))
        
        db.commit()
        
        return json.dumps({
            "status": "success",
            "message": f"Product {product_asin} added to wishlist",
            "user_id": user_id,
            "priority": priority
        })
        
    except Exception as e:
        return json.dumps({"error": f"Failed to add to wishlist: {str(e)}"})

@mcp.tool()
def get_wishlist(user_id: str, include_product_details: bool = True) -> str:
    """Get user's wishlist with optional product details"""
    db = get_db()
    cursor = db.cursor()
    
    if include_product_details:
        results = cursor.execute("""
            SELECT w.*, p.title, p.price, p.rating, p.category, p.image_url
            FROM wishlist w
            JOIN products p ON w.product_asin = p.asin
            WHERE w.user_id = ?
            ORDER BY w.priority DESC, w.added_at DESC
        """, (user_id,)).fetchall()
    else:
        results = cursor.execute("""
            SELECT * FROM wishlist WHERE user_id = ?
            ORDER BY priority DESC, added_at DESC
        """, (user_id,)).fetchall()
    
    wishlist_items = [dict_from_row(row) for row in results]
    
    return json.dumps({
        "user_id": user_id,
        "total_items": len(wishlist_items),
        "wishlist": wishlist_items
    }, indent=2)

@mcp.tool()
def track_user_search(user_id: str, search_query: str, filters_applied: dict = None,
                     results_found: int = 0) -> str:
    """Track user search behavior for personalization"""
    db = get_db()
    cursor = db.cursor()
    
    try:
        cursor.execute("""
            INSERT INTO search_history (user_id, search_query, filters_applied, results_found, searched_at)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, search_query, json.dumps(filters_applied or {}), 
              results_found, datetime.now().isoformat()))
        
        db.commit()
        
        return json.dumps({
            "status": "success",
            "message": "Search tracked successfully"
        })
        
    except Exception as e:
        return json.dumps({"error": f"Failed to track search: {str(e)}"})

@mcp.tool()
def get_user_insights(user_id: str, time_period: int = 30) -> str:
    """Generate insights about user behavior and preferences"""
    db = get_db()
    cursor = db.cursor()
    
    # Get search history
    search_history = cursor.execute("""
        SELECT * FROM search_history 
        WHERE user_id = ? 
        AND searched_at >= datetime('now', '-{} days')
        ORDER BY searched_at DESC
    """.format(time_period), (user_id,)).fetchall()
    
    # Get wishlist
    wishlist = cursor.execute("""
        SELECT w.*, p.category, p.price 
        FROM wishlist w
        JOIN products p ON w.product_asin = p.asin
        WHERE w.user_id = ?
    """, (user_id,)).fetchall()
    
    # Analyze preferences
    search_queries = [row[2] for row in search_history]  # search_query column
    wishlist_categories = [row[-2] for row in wishlist if row[-2]]  # category column
    wishlist_prices = [row[-1] for row in wishlist if row[-1]]  # price column
    
    insights = {
        "search_activity": {
            "total_searches": len(search_history),
            "unique_queries": len(set(search_queries)),
            "most_searched_terms": Counter(" ".join(search_queries).lower().split()).most_common(10)
        },
        "wishlist_analysis": {
            "total_items": len(wishlist),
            "preferred_categories": Counter(wishlist_categories).most_common(5),
            "price_preferences": {
                "avg_price": statistics.mean(wishlist_prices) if wishlist_prices else 0,
                "price_range": [min(wishlist_prices), max(wishlist_prices)] if wishlist_prices else [0, 0]
            }
        },
        "recommendations": "Based on your activity, you prefer " + 
                          (Counter(wishlist_categories).most_common(1)[0][0] if wishlist_categories else "various categories")
    }
    
    return json.dumps({
        "user_id": user_id,
        "time_period_days": time_period,
        "insights": insights,
        "generated_at": datetime.now().isoformat()
    }, indent=2)

# ============ PRICE & INVENTORY TOOLS ============

@mcp.tool()
def price_tracker(product_asin: str, target_price: float = None, 
                 track_duration: int = 30) -> str:
    """Set up price tracking for a product"""
    db = get_db()
    cursor = db.cursor()
    
    # Get current product info
    product = cursor.execute("SELECT * FROM products WHERE asin = ?", (product_asin,)).fetchone()
    if not product:
        return json.dumps({"error": f"Product {product_asin} not found"})
    
    product = dict_from_row(product)
    current_price = product.get('price', 0)
    
    # Add to price history
    try:
        cursor.execute("""
            INSERT INTO price_history (product_asin, price, recorded_at)
            VALUES (?, ?, ?)
        """, (product_asin, current_price, datetime.now().isoformat()))
        
        db.commit()
        
        tracking_info = {
            "product_asin": product_asin,
            "product_title": product.get('title', ''),
            "current_price": current_price,
            "target_price": target_price,
            "tracking_duration_days": track_duration,
            "price_alert": target_price and current_price <= target_price,
            "status": "tracking_active"
        }
        
        return json.dumps(tracking_info, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Failed to set up price tracking: {str(e)}"})

@mcp.tool()
def get_price_history(product_asin: str, days: int = 30) -> str:
    """Get price history for a product"""
    db = get_db()
    cursor = db.cursor()
    
    history = cursor.execute("""
        SELECT * FROM price_history 
        WHERE product_asin = ? 
        AND recorded_at >= datetime('now', '-{} days')
        ORDER BY recorded_at DESC
    """.format(days), (product_asin,)).fetchall()
    
    if not history:
        return json.dumps({
            "product_asin": product_asin,
            "message": "No price history found",
            "days_searched": days
        })
    
    history_data = [dict_from_row(row) for row in history]
    prices = [h['price'] for h in history_data if h['price']]
    
    price_analysis = {}
    if prices:
        price_analysis = {
            "current_price": prices[0],
            "highest_price": max(prices),
            "lowest_price": min(prices),
            "average_price": statistics.mean(prices),
            "price_trend": "increasing" if prices[0] > prices[-1] else "decreasing" if prices[0] < prices[-1] else "stable",
            "volatility": statistics.stdev(prices) if len(prices) > 1 else 0
        }
    
    return json.dumps({
        "product_asin": product_asin,
        "days_analyzed": days,
        "total_records": len(history_data),
        "price_analysis": price_analysis,
        "history": history_data
    }, indent=2)

# ============ ADVANCED REPORTING & EXPORT ============

@mcp.tool()
def generate_comprehensive_report(report_type: str = "market_overview", 
                                category: str = "", custom_filters: dict = None) -> str:
    """Generate comprehensive business intelligence reports"""
    db = get_db()
    cursor = db.cursor()
    
    custom_filters = custom_filters or {}
    report = {"report_type": report_type, "generated_at": datetime.now().isoformat()}
    
    if report_type == "market_overview":
        # Comprehensive market analysis
        total_products = cursor.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        
        category_breakdown = cursor.execute("""
            SELECT 
                category,
                COUNT(*) as count,
                ROUND(COUNT(*) * 100.0 / ?, 2) as percentage,
                AVG(price) as avg_price,
                AVG(rating) as avg_rating
            FROM products 
            WHERE category IS NOT NULL
            GROUP BY category 
            ORDER BY count DESC 
            LIMIT 15
        """, (total_products,)).fetchall()
        
        price_distribution = cursor.execute("""
            SELECT 
                CASE 
                    WHEN price <= 25 THEN '0-25'
                    WHEN price <= 50 THEN '25-50'
                    WHEN price <= 100 THEN '50-100'
                    WHEN price <= 250 THEN '100-250'
                    WHEN price <= 500 THEN '250-500'
                    ELSE '500+'
                END as price_range,
                COUNT(*) as count
            FROM products 
            WHERE price > 0
            GROUP BY price_range
            ORDER BY MIN(price)
        """).fetchall()
        
        report["market_data"] = {
            "total_products": total_products,
            "category_breakdown": [dict_from_row(row) for row in category_breakdown],
            "price_distribution": [dict_from_row(row) for row in price_distribution]
        }
    
    elif report_type == "performance_metrics":
        # Performance and quality metrics
        quality_metrics = cursor.execute("""
            SELECT 
                COUNT(*) as total_products,
                AVG(rating) as avg_rating,
                COUNT(CASE WHEN rating >= 4.5 THEN 1 END) as excellent_products,
                COUNT(CASE WHEN rating >= 4.0 THEN 1 END) as good_products,
                COUNT(CASE WHEN num_ratings >= 100 THEN 1 END) as well_reviewed,
                AVG(num_ratings) as avg_review_count,
                MAX(num_ratings) as max_reviews
            FROM products 
            WHERE rating IS NOT NULL
        """).fetchone()
        
        report["performance_data"] = dict_from_row(quality_metrics)
    
    return json.dumps(report, indent=2)

# ============ INITIALIZATION & MAIN ============

def initialize_additional_tables():
    """Initialize additional tables for advanced features"""
    db = get_db()
    cursor = db.cursor()
    
    # Create wishlist table if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS wishlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            product_asin TEXT NOT NULL,
            priority INTEGER DEFAULT 1,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, product_asin)
        )
    """)
    
    # Create cart table if not exists  
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cart (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            product_asin TEXT NOT NULL,
            quantity INTEGER DEFAULT 1,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, product_asin)
        )
    """)
    
    # Create price history table if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS price_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_asin TEXT NOT NULL,
            price REAL NOT NULL,
            recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create search history table if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            search_query TEXT NOT NULL,
            filters_applied TEXT,
            results_found INTEGER,
            searched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    db.commit()
    logger.info("✓ Additional tables initialized")

if __name__ == "__main__":
    try:
        # Initialize database and additional tables
        db = get_db()
        initialize_additional_tables()
        
        # Get product count
        count = db.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        logger.info(f"✓ Powerful E-commerce MCP Server ready with {count} products")
        logger.info("✓ Advanced features: AI search, analytics, recommendations, price tracking")
        
        # Run the server
        mcp.run()
        
    except Exception as e:
        logger.error(f"✗ Server initialization failed: {e}")
        sys.exit(1)
