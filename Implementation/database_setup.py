"""
Database Setup Script
Creates SQLite database with FTS5 full-text search
Works for both MCP and Agentic AI systems
"""

import sqlite3
import pandas as pd
import os
import sys

DB_PATH = "./data/ecommerce.db"
CSV_PATH = "./data/amazon_products.csv"  # or amazon_products.csv

def create_schema(conn):
    """Create database schema"""
    cursor = conn.cursor()
    
    # Main products table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asin TEXT UNIQUE NOT NULL,
            title TEXT NOT NULL,
            price REAL,
            rating REAL,
            num_ratings INTEGER,
            category TEXT,
            url TEXT,
            image_url TEXT,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_category ON products(category)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_price ON products(price)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rating ON products(rating)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_asin ON products(asin)")
    
    
    # FTS5 full-text search
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS products_fts USING fts5(
            asin, title, category, description, 
            content=products, content_rowid=id
        )
    """)
    
    # Sync triggers
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS products_ai AFTER INSERT ON products BEGIN
            INSERT INTO products_fts(rowid, asin, title, category, description)
            VALUES (new.id, new.asin, new.title, new.category, new.description);
        END
    """)
    
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS products_ad AFTER DELETE ON products BEGIN
            DELETE FROM products_fts WHERE rowid = old.id;
        END
    """)
    
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS products_au AFTER UPDATE ON products BEGIN
            UPDATE products_fts SET 
                asin=new.asin, title=new.title, category=new.category,
                description=new.description
            WHERE rowid=new.id;
        END
    """)
    
    # Enhanced tables for advanced features
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS wishlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asin TEXT NOT NULL,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            notes TEXT,
            target_price REAL,
            FOREIGN KEY (asin) REFERENCES products(asin)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cart (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asin TEXT NOT NULL,
            quantity INTEGER DEFAULT 1,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (asin) REFERENCES products(asin)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS price_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asin TEXT NOT NULL,
            price REAL NOT NULL,
            recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (asin) REFERENCES products(asin)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            filters TEXT,
            result_count INTEGER,
            searched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    print("✓ Schema created successfully")

def import_csv(conn):
    """Import CSV data"""
    if not os.path.exists(CSV_PATH):
        print(f"✗ Error: CSV file not found at {CSV_PATH}")
        print("  Please download from Kaggle and place in data/ folder")
        sys.exit(1)
    
    print(f"Loading CSV from {CSV_PATH}...")
    
    # Read CSV (limit to 2M for demo, Add nrows for selected import) , nrows=200000
    df = pd.read_csv(CSV_PATH) 
    df.columns = df.columns.str.strip().str.lower()
    
    print(f"✓ Loaded {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")
    
    # Detect and map columns
    column_map = detect_columns(df.columns)
    print(f"✓ Column mapping detected")
    
    # Prepare records
    records = []
    for _, row in df.iterrows():
        # Clean price
        price_str = str(row.get(column_map.get('price', ''), '0'))
        try:
            price = float(price_str.replace('£', '').replace('₹', '').replace('$', '').replace(',', '').strip())
        except:
            price = 0.0
        
        # Clean rating
        try:
            rating = float(row.get(column_map.get('rating', ''), 0))
        except:
            rating = 0.0
        
        # Clean review count
        try:
            num_ratings = int(str(row.get(column_map.get('num_ratings', ''), 0)).replace(',', ''))
        except:
            num_ratings = 0
        
        record = {
            'asin': str(row.get(column_map.get('asin', ''), '')),
            'title': str(row.get(column_map.get('title', ''), '')),
            'price': price,
            'rating': rating,
            'num_ratings': num_ratings,
            'category': str(row.get(column_map.get('category', ''), '')),
            'url': str(row.get(column_map.get('url', ''), '')),
            'image_url': str(row.get(column_map.get('image_url', ''), '')),
            'description': str(row.get(column_map.get('description', ''), ''))[:500],
            
        }
        
        records.append(record)
    
    # Insert
    cursor = conn.cursor()
    cursor.executemany("""
        INSERT OR IGNORE INTO products 
        (asin, title, price, rating, num_ratings, category, url, image_url, description)
        VALUES 
        (:asin, :title, :price, :rating, :num_ratings, :category, :url, :image_url, :description)
    """, records)
    
    conn.commit()
    print(f"✓ Imported {len(records)} products")
    
    return len(records)

def detect_columns(columns):
    """Detect column names from CSV"""
    mapping = {}
    cols_list = list(columns)
    
    # Required mappings
    mapping['asin'] = next((c for c in cols_list if 'asin' in c or 'product_id' in c), cols_list[0])
    mapping['title'] = next((c for c in cols_list if 'title' in c or 'name' in c), None)
    mapping['price'] = next((c for c in cols_list if 'price' in c and 'actual' not in c), None)
    mapping['rating'] = next((c for c in cols_list if 'rating' in c or 'star' in c), None)
    mapping['category'] = next((c for c in cols_list if 'category' in c or 'categoryname' in c), None)
    
    # Optional mappings
    mapping['num_ratings'] = next((c for c in cols_list if 'num_rating' in c or 'review' in c or 'no_of_rating' in c), None)
    mapping['url'] = next((c for c in cols_list if 'url' in c or 'producturl' in c), None)
    mapping['image_url'] = next((c for c in cols_list if 'image' in c or 'photo' in c or 'imgurl' in c), None)
    mapping['description'] = next((c for c in cols_list if 'description' in c or 'title' in c), None)
    

    
    return mapping

def show_stats(conn):
    """Display statistics"""
    cursor = conn.cursor()
    
    stats = {
        'total': cursor.execute("SELECT COUNT(*) FROM products").fetchone()[0],
        'categories': cursor.execute("SELECT COUNT(DISTINCT category) FROM products").fetchone()[0],
        'avg_price': cursor.execute("SELECT AVG(price) FROM products WHERE price > 0").fetchone()[0],
        'avg_rating': cursor.execute("SELECT AVG(rating) FROM products WHERE rating > 0").fetchone()[0]
    }
    
    print("\n" + "="*60)
    print("DATABASE STATISTICS")
    print("="*60)
    print(f"Total Products:     {stats['total']:,}")
    print(f"Categories:         {stats['categories']:,}")
    print(f"Average Price:      £{stats['avg_price']:.2f}" if stats['avg_price'] else "N/A")
    print(f"Average Rating:     {stats['avg_rating']:.2f}/5.0" if stats['avg_rating'] else "N/A")
    print("="*60 + "\n")

def main():
    print("""
╔══════════════════════════════════════════════════════╗
║     Amazon E-commerce Database Setup                 ║
║     For MCP and Agentic AI Systems                   ║
╚══════════════════════════════════════════════════════╝
    """)
    
    # Create data directory
    os.makedirs("./data", exist_ok=True)
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    # Create schema
    create_schema(conn)
    
    # Import data
    count = import_csv(conn)
    
    # Show stats
    show_stats(conn)
    
    print("✅ Setup Complete!")
    print(f"   Database: {os.path.abspath(DB_PATH)}")
    print(f"   Size: {os.path.getsize(DB_PATH) / 1024 / 1024:.2f} MB")
    print("\nNext steps:")
    print("  1. For MCP: Run mcp_server.py")
    print("  2. For Agentic AI: Run agentic_ai.py")
    
    conn.close()

if __name__ == "__main__":
    main()