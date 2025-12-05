"""
Ecommerce MCP Server - Amazon Product Dataset Integration
Prerequisites:
1. pip install "mcp[cli]" pandas
2. Download dataset from Kaggle
3. Place CSV in ./data/amazon_products.csv
"""

import json
import sys
import logging
import pandas as pd
from mcp.server.fastmcp import FastMCP

# Redirect all logging to stderr (stdout must be clean for JSON-RPC)
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("ecommerce")

# Global dataframe
df: pd.DataFrame = pd.DataFrame()

def load_data():
    global df
    import os
    
    # Try multiple possible paths
    possible_paths = [
        "./data/amazon_products.csv",
        "data/amazon_products.csv",
        os.path.join(os.path.dirname(__file__), "data", "amazon_products.csv"),
        "amazon_products.csv"
    ]
    
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Script location: {os.path.dirname(os.path.abspath(__file__))}")
    
    for csv_path in possible_paths:
        full_path = os.path.abspath(csv_path)
        logger.info(f"Trying to load: {full_path}")
        
        if os.path.exists(full_path):
            try:
                df = pd.read_csv(full_path)
                df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
                logger.info(f"✓ SUCCESS: Loaded {len(df)} products from {full_path}")
                logger.info(f"✓ Columns found: {list(df.columns)}")
                return
            except Exception as e:
                logger.error(f"✗ Error loading {full_path}: {e}")
                continue
        else:
            logger.warning(f"✗ File not found: {full_path}")
    
    # If we get here, no file was found
    logger.error("=" * 60)
    logger.error("DATASET NOT LOADED - FILE NOT FOUND!")
    logger.error("=" * 60)
    logger.error("Please ensure your CSV file is in one of these locations:")
    for path in possible_paths:
        logger.error(f"  - {os.path.abspath(path)}")
    logger.error("=" * 60)
    df = pd.DataFrame()

def get_columns():
    """Dynamically detect column names"""
    return {
        "name": next((c for c in df.columns if 'name' in c or 'title' in c), None),
        "price": next((c for c in df.columns if 'price' in c and 'discount' not in c and 'actual' not in c), None),
        "rating": next((c for c in df.columns if 'rating' in c or 'stars' in c or 'star' in c), None),
        "category": next((c for c in df.columns if 'category' in c), None),
        "id": next((c for c in df.columns if 'asin' in c or 'product_id' in c), df.columns[0] if len(df.columns) > 0 else None),
        "reviews": next((c for c in df.columns if 'no_of_rating' in c or 'review_count' in c or 'ratings_count' in c or 'reviews' in c), None),
        "discount": next((c for c in df.columns if 'discount' in c or 'off' in c), None),
        "image": next((c for c in df.columns if 'image' in c or 'img' in c or 'picture' in c or 'photo' in c), None),
    }

def clean_price(series):
    """Clean price column to numeric"""
    return pd.to_numeric(
        series.astype(str).str.replace('[₹$,]', '', regex=True).str.strip(),
        errors='coerce'
    )

def clean_numeric(series):
    """Clean any numeric column"""
    return pd.to_numeric(
        series.astype(str).str.replace('[,%]', '', regex=True).str.strip(),
        errors='coerce'
    )

# ============ TOOLS ============

@mcp.tool()
def search_products(query: str, category: str = "", limit: int = 10) -> str:
    """Search products by name, category, or keyword"""
    if df.empty:
        return json.dumps({"error": "Dataset not loaded"})
    
    cols = get_columns()
    if not cols["name"]:
        return json.dumps({"error": "No name column found"})
    
    mask = df[cols["name"]].str.lower().str.contains(query.lower(), na=False)
    
    if category and cols["category"]:
        mask &= df[cols["category"]].str.lower().str.contains(category.lower(), na=False)
    
    results = df[mask].head(limit)
    return json.dumps(json.loads(results.to_json(orient="records")), indent=2)


@mcp.tool()
def get_product_details(product_id: str) -> str:
    """Get detailed info for a specific product by ID or ASIN"""
    if df.empty:
        return json.dumps({"error": "Dataset not loaded"})
    
    cols = get_columns()
    if not cols["id"]:
        return json.dumps({"error": "No ID column found"})
    
    product = df[df[cols["id"]].astype(str) == str(product_id)]
    
    if product.empty:
        return json.dumps({"error": f"Product {product_id} not found"})
    
    return json.dumps(json.loads(product.iloc[0].to_json()), indent=2)


@mcp.tool()
def get_price_range(min_price: float, max_price: float, category: str = "", limit: int = 10) -> str:
    """Find products within a price range"""
    if df.empty:
        return json.dumps({"error": "Dataset not loaded"})
    
    cols = get_columns()
    if not cols["price"]:
        return json.dumps({"error": "No price column found"})
    
    prices = clean_price(df[cols["price"]])
    mask = (prices >= min_price) & (prices <= max_price)
    
    if category and cols["category"]:
        mask &= df[cols["category"]].str.lower().str.contains(category.lower(), na=False)
    
    results = df[mask].head(limit)
    return json.dumps(json.loads(results.to_json(orient="records")), indent=2)


@mcp.tool()
def get_top_rated(category: str = "", min_reviews: int = 10, limit: int = 10) -> str:
    """Get top-rated products, optionally filtered by category"""
    if df.empty:
        return json.dumps({"error": "Dataset not loaded"})
    
    cols = get_columns()
    filtered = df.copy()
    
    if category and cols["category"]:
        filtered = filtered[filtered[cols["category"]].str.lower().str.contains(category.lower(), na=False)]
    
    if cols["reviews"]:
        rev_count = clean_numeric(filtered[cols["reviews"]])
        filtered = filtered[rev_count >= min_reviews]
    
    if cols["rating"]:
        ratings = clean_numeric(filtered[cols["rating"]])
        filtered = filtered.assign(_sort_rating=ratings).sort_values("_sort_rating", ascending=False).drop("_sort_rating", axis=1)
    
    return json.dumps(json.loads(filtered.head(limit).to_json(orient="records")), indent=2)


@mcp.tool()
def list_categories(limit: int = 100) -> str:
    """List all available product categories"""
    if df.empty:
        return json.dumps({"error": "Dataset not loaded"})
    
    cols = get_columns()
    if not cols["category"]:
        return json.dumps({"error": "No category column found"})
    
    categories = df[cols["category"]].dropna().unique().tolist()[:limit]
    return json.dumps(categories, indent=2)


@mcp.tool()
def get_category_stats(category: str) -> str:
    """Get statistics for a category (avg price, rating, count)"""
    if df.empty:
        return json.dumps({"error": "Dataset not loaded"})
    
    cols = get_columns()
    if not cols["category"]:
        return json.dumps({"error": "No category column found"})
    
    filtered = df[df[cols["category"]].str.lower().str.contains(category.lower(), na=False)]
    
    if filtered.empty:
        return json.dumps({"error": f"No products found in category '{category}'"})
    
    stats = {"category": category, "product_count": len(filtered)}
    
    if cols["price"]:
        prices = clean_price(filtered[cols["price"]])
        stats["avg_price"] = round(prices.mean(), 2) if not prices.isna().all() else None
        stats["min_price"] = round(prices.min(), 2) if not prices.isna().all() else None
        stats["max_price"] = round(prices.max(), 2) if not prices.isna().all() else None
    
    if cols["rating"]:
        ratings = clean_numeric(filtered[cols["rating"]])
        stats["avg_rating"] = round(ratings.mean(), 2) if not ratings.isna().all() else None
    
    return json.dumps(stats, indent=2)


@mcp.tool()
def compare_products(product_ids: list[str]) -> str:
    """Compare multiple products side by side"""
    if df.empty:
        return json.dumps({"error": "Dataset not loaded"})
    
    cols = get_columns()
    if not cols["id"]:
        return json.dumps({"error": "No ID column found"})
    
    products = df[df[cols["id"]].astype(str).isin([str(p) for p in product_ids])]
    
    if products.empty:
        return json.dumps({"error": "No matching products found"})
    
    return json.dumps(json.loads(products.to_json(orient="records")), indent=2)


@mcp.tool()
def get_deals(category: str = "", min_discount: float = 20, limit: int = 10) -> str:
    """Find products with highest discount percentage (if discount data available)"""
    if df.empty:
        return json.dumps({"error": "Dataset not loaded"})
    
    cols = get_columns()
    if not cols["discount"]:
        return json.dumps({
            "error": "No discount column found in dataset",
            "note": "This dataset does not contain discount information. Try using get_budget_picks or find_value_deals instead."
        })
    
    discounts = clean_numeric(df[cols["discount"]])
    mask = discounts >= min_discount
    
    if category and cols["category"]:
        mask &= df[cols["category"]].str.lower().str.contains(category.lower(), na=False)
    
    results = df[mask].copy()
    results = results.assign(_disc=discounts[mask]).sort_values("_disc", ascending=False).drop("_disc", axis=1)
    
    if results.empty:
        return json.dumps({"message": f"No products found with discount >= {min_discount}%"})
    
    return json.dumps(json.loads(results.head(limit).to_json(orient="records")), indent=2)


@mcp.tool()
def get_dataset_info() -> str:
    """Get information about the loaded dataset"""
    if df.empty:
        return json.dumps({"error": "Dataset not loaded", "hint": "Place amazon_products.csv in ./data/"})
    
    cols = get_columns()
    sample = json.loads(df.iloc[0].to_json()) if len(df) > 0 else {}
    
    return json.dumps({
        "total_products": len(df),
        "columns": list(df.columns),
        "detected_columns": cols,
        "sample_row": sample
    }, indent=2)


@mcp.tool()
def get_recommendations(product_id: str, limit: int = 5) -> str:
    """Get product recommendations based on a given product (same category, similar price)"""
    if df.empty:
        return json.dumps({"error": "Dataset not loaded"})
    
    cols = get_columns()
    if not cols["id"]:
        return json.dumps({"error": "No ID column found"})
    
    # Find the source product
    product = df[df[cols["id"]].astype(str) == str(product_id)]
    if product.empty:
        return json.dumps({"error": f"Product {product_id} not found"})
    
    source = product.iloc[0]
    recommendations = df.copy()
    
    # Filter by same category
    if cols["category"] and pd.notna(source[cols["category"]]):
        recommendations = recommendations[
            recommendations[cols["category"]].str.lower() == source[cols["category"]].lower()
        ]
    
    # Find products with similar price (±30%)
    if cols["price"]:
        source_price = clean_price(pd.Series([source[cols["price"]]])).iloc[0]
        if pd.notna(source_price):
            prices = clean_price(recommendations[cols["price"]])
            price_diff = abs(prices - source_price) / source_price
            recommendations = recommendations[price_diff <= 0.3]
    
    # Exclude the source product itself
    recommendations = recommendations[recommendations[cols["id"]].astype(str) != str(product_id)]
    
    # Sort by rating if available
    if cols["rating"]:
        ratings = clean_numeric(recommendations[cols["rating"]])
        recommendations = recommendations.assign(_rating=ratings).sort_values("_rating", ascending=False).drop("_rating", axis=1)
    
    return json.dumps(json.loads(recommendations.head(limit).to_json(orient="records")), indent=2)


@mcp.tool()
def get_price_history_stats(product_id: str) -> str:
    """Get pricing statistics for a product (if discount/actual price data available)"""
    if df.empty:
        return json.dumps({"error": "Dataset not loaded"})
    
    cols = get_columns()
    if not cols["id"]:
        return json.dumps({"error": "No ID column found"})
    
    product = df[df[cols["id"]].astype(str) == str(product_id)]
    if product.empty:
        return json.dumps({"error": f"Product {product_id} not found"})
    
    p = product.iloc[0]
    stats = {"product_id": product_id}
    
    if cols["price"]:
        current_price = clean_price(pd.Series([p[cols["price"]]])).iloc[0]
        stats["current_price"] = float(current_price) if pd.notna(current_price) else None
    
    # Check for actual_price or original_price column
    actual_price_col = next((c for c in df.columns if 'actual' in c and 'price' in c), None)
    if actual_price_col and actual_price_col in p.index:
        actual_price = clean_price(pd.Series([p[actual_price_col]])).iloc[0]
        stats["original_price"] = float(actual_price) if pd.notna(actual_price) else None
        
        if stats.get("current_price") and stats.get("original_price"):
            savings = stats["original_price"] - stats["current_price"]
            stats["savings_amount"] = round(savings, 2)
            stats["savings_percent"] = round((savings / stats["original_price"]) * 100, 2)
    
    if cols["discount"]:
        discount = clean_numeric(pd.Series([p[cols["discount"]]])).iloc[0]
        stats["discount_percent"] = float(discount) if pd.notna(discount) else None
    else:
        stats["note"] = "No discount information available in this dataset"
    
    if cols["rating"]:
        rating = clean_numeric(pd.Series([p[cols["rating"]]])).iloc[0]
        stats["rating"] = float(rating) if pd.notna(rating) else None
    
    return json.dumps(stats, indent=2)


@mcp.tool()
def search_by_brand(brand: str, category: str = "", limit: int = 10) -> str:
    """Search products by brand name"""
    if df.empty:
        return json.dumps({"error": "Dataset not loaded"})
    
    # Look for brand column
    brand_col = next((c for c in df.columns if 'brand' in c), None)
    if not brand_col:
        return json.dumps({"error": "No brand column found in dataset"})
    
    mask = df[brand_col].str.lower().str.contains(brand.lower(), na=False)
    
    cols = get_columns()
    if category and cols["category"]:
        mask &= df[cols["category"]].str.lower().str.contains(category.lower(), na=False)
    
    results = df[mask].head(limit)
    
    if results.empty:
        return json.dumps({"message": f"No products found for brand '{brand}'"})
    
    return json.dumps(json.loads(results.to_json(orient="records")), indent=2)


@mcp.tool()
def get_brands_by_category(category: str, limit: int = 20) -> str:
    """List all brands available in a specific category"""
    if df.empty:
        return json.dumps({"error": "Dataset not loaded"})
    
    brand_col = next((c for c in df.columns if 'brand' in c), None)
    cols = get_columns()
    
    if not brand_col:
        return json.dumps({"error": "No brand column found"})
    if not cols["category"]:
        return json.dumps({"error": "No category column found"})
    
    filtered = df[df[cols["category"]].str.lower().str.contains(category.lower(), na=False)]
    brands = filtered[brand_col].dropna().unique().tolist()[:limit]
    
    return json.dumps({
        "category": category,
        "brand_count": len(brands),
        "brands": sorted(brands)
    }, indent=2)


@mcp.tool()
def get_trending_products(category: str = "", min_reviews: int = 100, limit: int = 10) -> str:
    """Get trending/popular products based on high review counts"""
    if df.empty:
        return json.dumps({"error": "Dataset not loaded"})
    
    cols = get_columns()
    if not cols["reviews"]:
        return json.dumps({"error": "No review count column found"})
    
    filtered = df.copy()
    
    if category and cols["category"]:
        filtered = filtered[filtered[cols["category"]].str.lower().str.contains(category.lower(), na=False)]
    
    # Filter by minimum reviews and sort
    rev_count = clean_numeric(filtered[cols["reviews"]])
    filtered = filtered[rev_count >= min_reviews]
    filtered = filtered.assign(_reviews=rev_count[rev_count >= min_reviews]).sort_values("_reviews", ascending=False).drop("_reviews", axis=1)
    
    return json.dumps(json.loads(filtered.head(limit).to_json(orient="records")), indent=2)


@mcp.tool()
def get_budget_picks(category: str, max_price: float, min_rating: float = 4.0, limit: int = 10) -> str:
    """Find best-rated products under a specific budget"""
    if df.empty:
        return json.dumps({"error": "Dataset not loaded"})
    
    cols = get_columns()
    if not cols["price"]:
        return json.dumps({"error": "No price column found"})
    if not cols["rating"]:
        return json.dumps({"error": "No rating column found"})
    
    filtered = df.copy()
    
    # Filter by category
    if cols["category"]:
        filtered = filtered[filtered[cols["category"]].str.lower().str.contains(category.lower(), na=False)]
    
    # Filter by price
    prices = clean_price(filtered[cols["price"]])
    filtered = filtered[prices <= max_price]
    
    # Filter by rating
    ratings = clean_numeric(filtered[cols["rating"]])
    filtered = filtered[ratings >= min_rating]
    
    # Sort by rating descending
    filtered = filtered.assign(_rating=ratings[(prices <= max_price) & (ratings >= min_rating)]).sort_values("_rating", ascending=False).drop("_rating", axis=1)
    
    return json.dumps(json.loads(filtered.head(limit).to_json(orient="records")), indent=2)


@mcp.tool()
def get_premium_products(category: str, min_price: float = 1000, limit: int = 10) -> str:
    """Find premium/high-end products in a category"""
    if df.empty:
        return json.dumps({"error": "Dataset not loaded"})
    
    cols = get_columns()
    if not cols["price"]:
        return json.dumps({"error": "No price column found"})
    
    filtered = df.copy()
    
    # Filter by category
    if cols["category"]:
        filtered = filtered[filtered[cols["category"]].str.lower().str.contains(category.lower(), na=False)]
    
    # Filter by minimum price
    prices = clean_price(filtered[cols["price"]])
    filtered = filtered[prices >= min_price]
    
    # Sort by price descending
    filtered = filtered.assign(_price=prices[prices >= min_price]).sort_values("_price", ascending=False).drop("_price", axis=1)
    
    return json.dumps(json.loads(filtered.head(limit).to_json(orient="records")), indent=2)


@mcp.tool()
def analyze_price_distribution(category: str) -> str:
    """Get price distribution statistics for a category (min, max, median, quartiles)"""
    if df.empty:
        return json.dumps({"error": "Dataset not loaded"})
    
    cols = get_columns()
    if not cols["price"]:
        return json.dumps({"error": "No price column found"})
    if not cols["category"]:
        return json.dumps({"error": "No category column found"})
    
    filtered = df[df[cols["category"]].str.lower().str.contains(category.lower(), na=False)]
    
    if filtered.empty:
        return json.dumps({"error": f"No products found in category '{category}'"})
    
    prices = clean_price(filtered[cols["price"]]).dropna()
    
    stats = {
        "category": category,
        "product_count": len(filtered),
        "price_stats": {
            "min": round(prices.min(), 2),
            "max": round(prices.max(), 2),
            "mean": round(prices.mean(), 2),
            "median": round(prices.median(), 2),
            "q1": round(prices.quantile(0.25), 2),
            "q3": round(prices.quantile(0.75), 2),
            "std_dev": round(prices.std(), 2)
        },
        "price_ranges": {
            "budget": f"< {round(prices.quantile(0.25), 2)}",
            "mid_range": f"{round(prices.quantile(0.25), 2)} - {round(prices.quantile(0.75), 2)}",
            "premium": f"> {round(prices.quantile(0.75), 2)}"
        }
    }
    
    return json.dumps(stats, indent=2)


@mcp.tool()
def compare_categories(categories: list[str]) -> str:
    """Compare statistics across multiple categories"""
    if df.empty:
        return json.dumps({"error": "Dataset not loaded"})
    
    cols = get_columns()
    if not cols["category"]:
        return json.dumps({"error": "No category column found"})
    
    comparison = []
    
    for cat in categories:
        filtered = df[df[cols["category"]].str.lower().str.contains(cat.lower(), na=False)]
        
        if filtered.empty:
            continue
        
        cat_stats = {"category": cat, "product_count": len(filtered)}
        
        if cols["price"]:
            prices = clean_price(filtered[cols["price"]]).dropna()
            cat_stats["avg_price"] = round(prices.mean(), 2) if len(prices) > 0 else None
            cat_stats["median_price"] = round(prices.median(), 2) if len(prices) > 0 else None
        
        if cols["rating"]:
            ratings = clean_numeric(filtered[cols["rating"]]).dropna()
            cat_stats["avg_rating"] = round(ratings.mean(), 2) if len(ratings) > 0 else None
        
        if cols["reviews"]:
            reviews = clean_numeric(filtered[cols["reviews"]]).dropna()
            cat_stats["avg_reviews"] = round(reviews.mean(), 0) if len(reviews) > 0 else None
        
        comparison.append(cat_stats)
    
    return json.dumps(comparison, indent=2)


@mcp.tool()
def find_value_deals(category: str = "", min_rating: float = 4.0, max_price: float = 5000, min_discount: float = 20, limit: int = 10) -> str:
    """Find best value products: good ratings and reasonable price (discount optional if available)"""
    if df.empty:
        return json.dumps({"error": "Dataset not loaded"})
    
    cols = get_columns()
    filtered = df.copy()
    
    # Apply filters
    if category and cols["category"]:
        filtered = filtered[filtered[cols["category"]].str.lower().str.contains(category.lower(), na=False)]
    
    if cols["rating"]:
        ratings = clean_numeric(filtered[cols["rating"]])
        filtered = filtered[ratings >= min_rating]
    else:
        return json.dumps({"error": "No rating column found"})
    
    if cols["price"]:
        prices = clean_price(filtered[cols["price"]])
        filtered = filtered[prices <= max_price]
    else:
        return json.dumps({"error": "No price column found"})
    
    # If discount column exists, use it
    if cols["discount"]:
        discounts = clean_numeric(filtered[cols["discount"]])
        mask = discounts >= min_discount
        filtered = filtered[mask]
        # Calculate value score (rating * discount)
        valid_ratings = ratings[mask]
        valid_discounts = discounts[mask]
        filtered = filtered.assign(_value_score=valid_ratings * valid_discounts)
        filtered = filtered.sort_values("_value_score", ascending=False).drop("_value_score", axis=1)
    else:
        # No discount column - sort by rating and price combination
        valid_ratings = ratings[prices <= max_price]
        valid_prices = prices[prices <= max_price]
        # Normalize and create value score (higher rating, lower price = better)
        price_norm = 1 - (valid_prices - valid_prices.min()) / (valid_prices.max() - valid_prices.min() + 0.001)
        filtered = filtered.assign(_value_score=valid_ratings * price_norm)
        filtered = filtered.sort_values("_value_score", ascending=False).drop("_value_score", axis=1)
    
    if filtered.empty:
        return json.dumps({"message": "No products match the criteria"})
    
    return json.dumps(json.loads(filtered.head(limit).to_json(orient="records")), indent=2)


@mcp.tool()
def get_new_arrivals(category: str = "", limit: int = 10) -> str:
    """Get newest products (if dataset has date/timestamp column)"""
    if df.empty:
        return json.dumps({"error": "Dataset not loaded"})
    
    # Look for date columns
    date_col = next((c for c in df.columns if 'date' in c or 'time' in c or 'added' in c), None)
    
    if not date_col:
        # Fallback: return products with fewer reviews (likely newer)
        cols = get_columns()
        if cols["reviews"]:
            filtered = df.copy()
            if category and cols["category"]:
                filtered = filtered[filtered[cols["category"]].str.lower().str.contains(category.lower(), na=False)]
            
            rev_count = clean_numeric(filtered[cols["reviews"]])
            filtered = filtered.assign(_reviews=rev_count).sort_values("_reviews", ascending=True).drop("_reviews", axis=1)
            
            return json.dumps({
                "note": "No date column found, showing products with fewer reviews (likely newer)",
                "products": json.loads(filtered.head(limit).to_json(orient="records"))
            }, indent=2)
        else:
            return json.dumps({"error": "Cannot determine new arrivals: no date or review count column"})
    
    # If date column exists
    cols = get_columns()
    filtered = df.copy()
    
    if category and cols["category"]:
        filtered = filtered[filtered[cols["category"]].str.lower().str.contains(category.lower(), na=False)]
    
    filtered = filtered.sort_values(date_col, ascending=False)
    
    return json.dumps(json.loads(filtered.head(limit).to_json(orient="records")), indent=2)


@mcp.tool()
def search_by_keywords(keywords: list[str], match_all: bool = False, limit: int = 10) -> str:
    """Search products using multiple keywords (match any or all)"""
    if df.empty:
        return json.dumps({"error": "Dataset not loaded"})
    
    cols = get_columns()
    if not cols["name"]:
        return json.dumps({"error": "No name column found"})
    
    filtered = df.copy()
    
    if match_all:
        # Product must contain ALL keywords
        for keyword in keywords:
            filtered = filtered[filtered[cols["name"]].str.lower().str.contains(keyword.lower(), na=False)]
    else:
        # Product must contain ANY keyword
        mask = pd.Series([False] * len(df))
        for keyword in keywords:
            mask |= df[cols["name"]].str.lower().str.contains(keyword.lower(), na=False)
        filtered = df[mask]
    
    return json.dumps({
        "keywords": keywords,
        "match_mode": "all" if match_all else "any",
        "result_count": len(filtered),
        "products": json.loads(filtered.head(limit).to_json(orient="records"))
    }, indent=2)


@mcp.tool()
def get_product_images(product_id: str) -> str:
    """Get image URLs for a specific product"""
    if df.empty:
        return json.dumps({"error": "Dataset not loaded"})
    
    cols = get_columns()
    if not cols["id"]:
        return json.dumps({"error": "No ID column found"})
    if not cols["image"]:
        return json.dumps({"error": "No image column found in dataset"})
    
    product = df[df[cols["id"]].astype(str) == str(product_id)]
    
    if product.empty:
        return json.dumps({"error": f"Product {product_id} not found"})
    
    p = product.iloc[0]
    image_data = p[cols["image"]]
    
    # Handle different image formats
    images = []
    if pd.notna(image_data):
        # Check if it's a list-like string or comma-separated URLs
        if isinstance(image_data, str):
            # Try to parse as JSON list first
            try:
                import ast
                parsed = ast.literal_eval(image_data)
                if isinstance(parsed, list):
                    images = parsed
                else:
                    images = [image_data]
            except:
                # Split by common separators
                if ',' in image_data:
                    images = [url.strip() for url in image_data.split(',')]
                elif '|' in image_data:
                    images = [url.strip() for url in image_data.split('|')]
                else:
                    images = [image_data]
        elif isinstance(image_data, list):
            images = image_data
    
    return json.dumps({
        "product_id": product_id,
        "product_name": p[cols["name"]] if cols["name"] else "N/A",
        "image_count": len(images),
        "images": images
    }, indent=2)


@mcp.tool()
def search_by_image_style(style_keywords: list[str], category: str = "", limit: int = 10) -> str:
    """Search products by visual style keywords (color, material, design) found in image URLs or product names"""
    if df.empty:
        return json.dumps({"error": "Dataset not loaded"})
    
    cols = get_columns()
    
    # Search in both name and image URL if available
    mask = pd.Series([False] * len(df))
    
    for keyword in style_keywords:
        keyword_lower = keyword.lower()
        
        # Search in product name
        if cols["name"]:
            mask |= df[cols["name"]].str.lower().str.contains(keyword_lower, na=False)
        
        # Search in image URL (some URLs contain descriptive terms)
        if cols["image"]:
            mask |= df[cols["image"]].astype(str).str.lower().str.contains(keyword_lower, na=False)
    
    filtered = df[mask]
    
    # Apply category filter if specified
    if category and cols["category"]:
        filtered = filtered[filtered[cols["category"]].str.lower().str.contains(category.lower(), na=False)]
    
    return json.dumps({
        "style_keywords": style_keywords,
        "category": category if category else "all",
        "result_count": len(filtered),
        "products": json.loads(filtered.head(limit).to_json(orient="records"))
    }, indent=2)


@mcp.tool()
def get_products_with_images(category: str = "", min_rating: float = 0, limit: int = 10) -> str:
    """Get products that have image URLs available, optionally filtered by category and rating"""
    if df.empty:
        return json.dumps({"error": "Dataset not loaded"})
    
    cols = get_columns()
    if not cols["image"]:
        return json.dumps({"error": "No image column found in dataset"})
    
    # Filter products that have non-null image data
    filtered = df[df[cols["image"]].notna() & (df[cols["image"]].astype(str).str.len() > 0)]
    
    # Apply category filter
    if category and cols["category"]:
        filtered = filtered[filtered[cols["category"]].str.lower().str.contains(category.lower(), na=False)]
    
    # Apply rating filter
    if min_rating > 0 and cols["rating"]:
        ratings = clean_numeric(filtered[cols["rating"]])
        filtered = filtered[ratings >= min_rating]
    
    return json.dumps({
        "total_with_images": len(filtered),
        "category": category if category else "all",
        "products": json.loads(filtered.head(limit).to_json(orient="records"))
    }, indent=2)


@mcp.tool()
def compare_products_visual(product_ids: list[str]) -> str:
    """Compare products with their images side by side"""
    if df.empty:
        return json.dumps({"error": "Dataset not loaded"})
    
    cols = get_columns()
    if not cols["id"]:
        return json.dumps({"error": "No ID column found"})
    
    products = df[df[cols["id"]].astype(str).isin([str(p) for p in product_ids])]
    
    if products.empty:
        return json.dumps({"error": "No matching products found"})
    
    comparison = []
    for _, product in products.iterrows():
        item = {
            "product_id": str(product[cols["id"]]),
            "name": product[cols["name"]] if cols["name"] else "N/A",
        }
        
        if cols["price"]:
            item["price"] = product[cols["price"]]
        
        if cols["rating"]:
            item["rating"] = product[cols["rating"]]
        
        if cols["image"]:
            image_data = product[cols["image"]]
            if pd.notna(image_data):
                # Extract first image URL
                if isinstance(image_data, str):
                    try:
                        import ast
                        parsed = ast.literal_eval(image_data)
                        if isinstance(parsed, list) and len(parsed) > 0:
                            item["image_url"] = parsed[0]
                        else:
                            item["image_url"] = image_data.split(',')[0].strip() if ',' in image_data else image_data
                    except:
                        item["image_url"] = image_data.split(',')[0].strip() if ',' in image_data else image_data
                else:
                    item["image_url"] = str(image_data)
        
        comparison.append(item)
    
    return json.dumps({
        "comparison_count": len(comparison),
        "products": comparison
    }, indent=2)


@mcp.tool()
def find_similar_looking_products(product_id: str, limit: int = 5) -> str:
    """Find visually similar products based on category and attributes (basic similarity without ML)"""
    if df.empty:
        return json.dumps({"error": "Dataset not loaded"})
    
    cols = get_columns()
    if not cols["id"]:
        return json.dumps({"error": "No ID column found"})
    
    # Find source product
    product = df[df[cols["id"]].astype(str) == str(product_id)]
    if product.empty:
        return json.dumps({"error": f"Product {product_id} not found"})
    
    source = product.iloc[0]
    similar = df.copy()
    
    # Exclude the source product
    similar = similar[similar[cols["id"]].astype(str) != str(product_id)]
    
    # Filter by same category (most important for visual similarity)
    if cols["category"] and pd.notna(source[cols["category"]]):
        similar = similar[similar[cols["category"]].str.lower() == source[cols["category"]].lower()]
    
    # Extract color/style keywords from product name
    if cols["name"] and pd.notna(source[cols["name"]]):
        source_name = source[cols["name"]].lower()
        color_keywords = ['black', 'white', 'red', 'blue', 'green', 'yellow', 'silver', 'gold', 
                         'grey', 'gray', 'pink', 'purple', 'orange', 'brown', 'beige']
        style_keywords = ['classic', 'modern', 'vintage', 'sport', 'casual', 'formal', 'sleek']
        
        found_keywords = []
        for keyword in color_keywords + style_keywords:
            if keyword in source_name:
                found_keywords.append(keyword)
        
        # Boost products that share style keywords
        if found_keywords and cols["name"]:
            similarity_score = pd.Series(0, index=similar.index)
            for keyword in found_keywords:
                similarity_score += similar[cols["name"]].str.lower().str.contains(keyword, na=False).astype(int)
            
            similar = similar.assign(_similarity=similarity_score)
            similar = similar[similar["_similarity"] > 0].sort_values("_similarity", ascending=False).drop("_similarity", axis=1)
    
    # If we still have too many, prefer similar price range
    if len(similar) > limit and cols["price"]:
        source_price = clean_price(pd.Series([source[cols["price"]]])).iloc[0]
        if pd.notna(source_price):
            prices = clean_price(similar[cols["price"]])
            price_diff = abs(prices - source_price)
            similar = similar.assign(_price_diff=price_diff).sort_values("_price_diff").drop("_price_diff", axis=1)
    
    return json.dumps({
        "source_product": product_id,
        "similar_count": len(similar),
        "note": "Similarity based on category, color/style keywords, and price range (no ML vision model)",
        "products": json.loads(similar.head(limit).to_json(orient="records"))
    }, indent=2)


# ============ MAIN ============

# Load data at import time
load_data()

if __name__ == "__main__":
    mcp.run()