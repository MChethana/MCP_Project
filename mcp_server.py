"""
MCP Server for Amazon UK E-commerce
For use with Claude Desktop
"""

import json
import sys
import logging
import sqlite3
from typing import List
from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)

mcp = FastMCP("amazon-uk")

DB_PATH = "./data/ecommerce.db"
db_conn = None

def get_db():
    global db_conn
    if db_conn is None:
        db_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        db_conn.row_factory = sqlite3.Row
        logger.info(f"✓ Connected to database")
    return db_conn

def dict_from_row(row):
    return {key: row[key] for key in row.keys()} if row else None

@mcp.tool()
def search_products(query: str, category: str = "", min_price: float = 0,
                   max_price: float = 999999, min_rating: float = 0,
                   sort_by: str = "relevance", limit: int = 10) -> str:
    """
    Search Amazon UK products
    
    Args:
        query: Search keywords
        category: Filter by category
        min_price: Minimum price (£)
        max_price: Maximum price (£)
        min_rating: Minimum rating (0-5)
        sort_by: 'relevance', 'price_low', 'price_high', 'rating'
        limit: Max results
    """
    db = get_db()
    cursor = db.cursor()
    
    sql = """
        SELECT p.* FROM products_fts fts
        JOIN products p ON fts.rowid = p.id
        WHERE products_fts MATCH ?
    """
    params = [query]
    
    if category:
        sql += " AND p.category LIKE ?"
        params.append(f"%{category}%")
    if min_price > 0:
        sql += " AND p.price >= ?"
        params.append(min_price)
    if max_price < 999999:
        sql += " AND p.price <= ?"
        params.append(max_price)
    if min_rating > 0:
        sql += " AND p.rating >= ?"
        params.append(min_rating)
    
    if sort_by == "price_low":
        sql += " ORDER BY p.price ASC"
    elif sort_by == "price_high":
        sql += " ORDER BY p.price DESC"
    elif sort_by == "rating":
        sql += " ORDER BY p.rating DESC"
    else:
        sql += " ORDER BY rank"
    
    sql += f" LIMIT {limit}"
    
    try:
        results = cursor.execute(sql, params).fetchall()
        products = [dict_from_row(row) for row in results]
        
        return json.dumps({
            "query": query,
            "result_count": len(products),
            "products": products
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})

@mcp.tool()
def get_product_details(asin: str) -> str:
    """Get detailed product information by ASIN"""
    db = get_db()
    cursor = db.cursor()
    
    result = cursor.execute(
        "SELECT * FROM products WHERE asin = ?", (asin,)
    ).fetchone()
    
    if result:
        return json.dumps(dict_from_row(result), indent=2)
    else:
        return json.dumps({"error": f"Product {asin} not found"})

@mcp.tool()
def get_categories(limit: int = 50) -> str:
    """List all product categories with counts"""
    db = get_db()
    cursor = db.cursor()
    
    results = cursor.execute("""
        SELECT category, COUNT(*) as product_count
        FROM products
        WHERE category IS NOT NULL
        GROUP BY category
        ORDER BY product_count DESC
        LIMIT ?
    """, (limit,)).fetchall()
    
    categories = [dict_from_row(row) for row in results]
    
    return json.dumps({
        "total_categories": len(categories),
        "categories": categories
    }, indent=2)

@mcp.tool()
def get_recommendations(asin: str, limit: int = 5) -> str:
    """Get product recommendations similar to given product"""
    db = get_db()
    cursor = db.cursor()
    
    source = cursor.execute(
        "SELECT * FROM products WHERE asin = ?", (asin,)
    ).fetchone()
    
    if not source:
        return json.dumps({"error": f"Product {asin} not found"})
    
    source = dict_from_row(source)
    
    results = cursor.execute("""
        SELECT * FROM products
        WHERE asin != ?
          AND category = ?
          AND price BETWEEN ? AND ?
        ORDER BY rating DESC
        LIMIT ?
    """, (asin, source['category'], source['price'] * 0.7, source['price'] * 1.3, limit)).fetchall()
    
    recommendations = [dict_from_row(row) for row in results]
    
    return json.dumps({
        "source_product": asin,
        "recommendation_count": len(recommendations),
        "recommendations": recommendations
    }, indent=2)

@mcp.tool()
def get_price_statistics(category: str = "") -> str:
    """Get price statistics for a category or all products"""
    db = get_db()
    cursor = db.cursor()
    
    if category:
        result = cursor.execute("""
            SELECT 
                COUNT(*) as product_count,
                MIN(price) as min_price,
                MAX(price) as max_price,
                AVG(price) as avg_price,
                AVG(rating) as avg_rating
            FROM products
            WHERE category LIKE ? AND price > 0
        """, (f"%{category}%",)).fetchone()
    else:
        result = cursor.execute("""
            SELECT 
                COUNT(*) as product_count,
                MIN(price) as min_price,
                MAX(price) as max_price,
                AVG(price) as avg_price,
                AVG(rating) as avg_rating
            FROM products
            WHERE price > 0
        """).fetchone()
    
    stats = dict_from_row(result)
    stats["category"] = category if category else "all"
    
    return json.dumps(stats, indent=2)

# Initialize
try:
    db = get_db()
    count = db.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    logger.info(f"✓ MCP Server ready with {count} products")
except Exception as e:
    logger.error(f"✗ Failed to initialize: {e}")
    sys.exit(1)

if __name__ == "__main__":
    mcp.run()