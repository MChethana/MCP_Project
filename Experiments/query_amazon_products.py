#!/usr/bin/env python3
"""
Script to run sample queries on the Amazon products database
"""

import sqlite3
import pandas as pd
from datetime import datetime

def run_sample_queries():
    """Run sample queries on the Amazon products database"""
    
    db_path = "data/amazon_products.db"
    table_name = "amazon_products"
    
    print(f"Running sample queries on Amazon products database at {datetime.now()}")
    print(f"Database: {db_path}")
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        
        # Sample queries
        queries = [
            ("Top 10 Most Expensive Products", f"""
            SELECT asin, title, price, stars, categoryName 
            FROM {table_name} 
            WHERE price > 0 
            ORDER BY price DESC 
            LIMIT 10
            """),
            
            ("Top 10 Best Rated Products (with at least 100 reviews)", f"""
            SELECT asin, title, stars, reviews, price, categoryName 
            FROM {table_name} 
            WHERE reviews >= 100 AND stars > 0 
            ORDER BY stars DESC, reviews DESC 
            LIMIT 10
            """),
            
            ("Best Sellers by Category", f"""
            SELECT categoryName, COUNT(*) as bestseller_count 
            FROM {table_name} 
            WHERE isBestSeller = 1 
            GROUP BY categoryName 
            ORDER BY bestseller_count DESC 
            LIMIT 10
            """),
            
            ("Products with Most Reviews", f"""
            SELECT asin, title, reviews, stars, price, categoryName 
            FROM {table_name} 
            WHERE reviews > 0 
            ORDER BY reviews DESC 
            LIMIT 10
            """),
            
            ("Average Price by Category (Top 10)", f"""
            SELECT 
                categoryName, 
                COUNT(*) as product_count,
                ROUND(AVG(price), 2) as avg_price,
                ROUND(MIN(price), 2) as min_price,
                ROUND(MAX(price), 2) as max_price
            FROM {table_name} 
            WHERE price > 0 
            GROUP BY categoryName 
            HAVING product_count >= 100
            ORDER BY avg_price DESC 
            LIMIT 10
            """),
            
            ("Products Recently Bought", f"""
            SELECT asin, title, boughtInLastMonth, price, stars, categoryName 
            FROM {table_name} 
            WHERE boughtInLastMonth > 0 
            ORDER BY boughtInLastMonth DESC 
            LIMIT 10
            """),
        ]
        
        for query_name, query in queries:
            print(f"\n{'='*60}")
            print(f"Query: {query_name}")
            print('='*60)
            
            try:
                df = pd.read_sql_query(query, conn)
                
                if df.empty:
                    print("No results found.")
                else:
                    # Format the output nicely
                    pd.set_option('display.max_columns', None)
                    pd.set_option('display.width', None)
                    pd.set_option('display.max_colwidth', 50)
                    
                    print(df.to_string(index=False))
                    
            except Exception as e:
                print(f"Error executing query: {str(e)}")
        
        conn.close()
        
        print(f"\n{'='*60}")
        print("✅ Sample queries completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error running queries: {str(e)}")
        return False

def run_custom_query(query_sql):
    """Run a custom query on the database"""
    
    db_path = "data/amazon_products.db"
    
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(query_sql, conn)
        conn.close()
        
        if df.empty:
            print("No results found.")
        else:
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', 50)
            print(df.to_string(index=False))
        
        return True
        
    except Exception as e:
        print(f"Error executing custom query: {str(e)}")
        return False

if __name__ == "__main__":
    success = run_sample_queries()
    if success:
        print("\n🎉 Database queries completed!")
        print("\nTo run custom queries, use:")
        print("from query_amazon_products import run_custom_query")
        print("run_custom_query('SELECT * FROM amazon_products WHERE categoryName = \"Electronics\" LIMIT 5')")
    else:
        print("\n💥 Database queries failed.")
