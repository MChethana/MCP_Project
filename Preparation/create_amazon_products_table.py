#!/usr/bin/env python3
"""
Script to create database table from Amazon products CSV file
"""

import pandas as pd
import sqlite3
import os
from datetime import datetime

def create_amazon_products_table():
    """Create database table and load Amazon products data from CSV"""
    
    # Database configuration
    db_path = "data/amazon_products.db"
    csv_path = "data/amazon_products.csv"
    table_name = "amazon_products"
    
    print(f"Starting database creation process at {datetime.now()}")
    print(f"CSV file: {csv_path}")
    print(f"Database: {db_path}")
    print(f"Table: {table_name}")
    
    try:
        # Connect to SQLite database
        print("\n1. Connecting to database...")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Drop table if exists to start fresh
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        
        # Create table with appropriate schema
        print("\n2. Creating table schema...")
        create_table_sql = f"""
        CREATE TABLE {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asin VARCHAR(20) NOT NULL,
            title TEXT NOT NULL,
            imgUrl VARCHAR(200),
            productURL VARCHAR(100),
            stars DECIMAL(3,2),
            reviews INTEGER,
            price DECIMAL(10,2),
            isBestSeller BOOLEAN,
            boughtInLastMonth INTEGER,
            categoryName VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        cursor.execute(create_table_sql)
        print(f"   - Table '{table_name}' created successfully")
        
        # Create indices for better performance
        print("\n3. Creating indices...")
        indices = [
            f"CREATE INDEX idx_{table_name}_asin ON {table_name}(asin);",
            f"CREATE INDEX idx_{table_name}_category ON {table_name}(categoryName);",
            f"CREATE INDEX idx_{table_name}_price ON {table_name}(price);",
            f"CREATE INDEX idx_{table_name}_stars ON {table_name}(stars);",
            f"CREATE INDEX idx_{table_name}_bestseller ON {table_name}(isBestSeller);"
        ]
        
        for idx_sql in indices:
            cursor.execute(idx_sql)
        print("   - Indices created successfully")
        
        # Prepare INSERT statement
        insert_sql = f"""
        INSERT INTO {table_name} (asin, title, imgUrl, productURL, stars, reviews, price, isBestSeller, boughtInLastMonth, categoryName)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        # Load data in small chunks to avoid SQL variable limits
        print("\n4. Loading data into database...")
        chunk_size = 1000  # Reduced chunk size to avoid SQL variable limits
        records_processed = 0
        
        # Read CSV in chunks and insert data
        for chunk_num, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size)):
            # Convert DataFrame to list of tuples for batch insert
            data_tuples = []
            for _, row in chunk.iterrows():
                data_tuples.append((
                    row['asin'],
                    row['title'],
                    row['imgUrl'],
                    row['productURL'],
                    row['stars'] if pd.notna(row['stars']) else None,
                    int(row['reviews']) if pd.notna(row['reviews']) else None,
                    row['price'] if pd.notna(row['price']) else None,
                    bool(row['isBestSeller']) if pd.notna(row['isBestSeller']) else False,
                    int(row['boughtInLastMonth']) if pd.notna(row['boughtInLastMonth']) else None,
                    row['categoryName']
                ))
            
            # Execute batch insert
            cursor.executemany(insert_sql, data_tuples)
            records_processed += len(data_tuples)
            
            # Commit periodically and show progress
            if chunk_num % 10 == 0:  # Commit every 10 chunks (10,000 records)
                conn.commit()
                print(f"   - Processed {records_processed:,} records...")
        
        # Final commit
        conn.commit()
        print(f"   - Completed! Total records processed: {records_processed:,}")
        
        # Verify data loading
        print("\n5. Verifying data loading...")
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        record_count = cursor.fetchone()[0]
        print(f"   - Total records in database: {record_count:,}")
        
        # Display sample data
        print("\n6. Sample records from database:")
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
        columns = [description[0] for description in cursor.description]
        sample_data = cursor.fetchall()
        
        print(f"   Columns: {columns}")
        for i, row in enumerate(sample_data, 1):
            print(f"   Record {i}: {dict(zip(columns, row))}")
        
        # Display table statistics
        print("\n7. Table statistics:")
        stats_queries = [
            ("Total products", f"SELECT COUNT(*) FROM {table_name}"),
            ("Unique categories", f"SELECT COUNT(DISTINCT categoryName) FROM {table_name}"),
            ("Best sellers", f"SELECT COUNT(*) FROM {table_name} WHERE isBestSeller = 1"),
            ("Average price", f"SELECT ROUND(AVG(price), 2) FROM {table_name} WHERE price > 0"),
            ("Average rating", f"SELECT ROUND(AVG(stars), 2) FROM {table_name} WHERE stars > 0")
        ]
        
        for stat_name, query in stats_queries:
            cursor.execute(query)
            result = cursor.fetchone()[0]
            print(f"   - {stat_name}: {result}")
        
        conn.close()
        
        print(f"\n✅ Database table creation completed successfully!")
        print(f"Database file: {os.path.abspath(db_path)}")
        
        return True
        
    except Exception as e:
        print(f"\n Error occurred: {str(e)}")
        if 'conn' in locals():
            conn.close()
        return False

if __name__ == "__main__":
    success = create_amazon_products_table()
    if success:
        print("\n Amazon products database is ready for use!")
    else:
        print("\n Database creation failed. Please check the error messages above.")
