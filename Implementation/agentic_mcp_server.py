#!/usr/bin/env python3
"""
Agentic AI MCP Server
====================

An MCP server that provides the same functional capabilities as agentic_ai.py:
- Database operations (search products, get categories, recommendations, etc.)
- Claude AI interface for autonomous agent behavior
- Tool execution and reasoning capabilities
- E-commerce product assistance

This server exposes all the database tools and AI capabilities as MCP tools.
"""

import asyncio
import json
import sqlite3
import sys
import os
from typing import Any, Dict, List, Optional
from pathlib import Path

# Add the current directory to the path to import local modules
sys.path.append(str(Path(__file__).parent))

# Import MCP components with correct structure
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.types import Tool, TextContent
import mcp.server.stdio

# Import Claude setup from the local module
try:
    from claude_setup import ClaudeModelInterface, SyConfig
except ImportError:
    print("Warning: Claude setup not available. Some features may not work.")
    ClaudeModelInterface = None
    SyConfig = None

# Database configuration
DB_PATH = "./data/ecommerce.db"  # Fixed path for current directory
CLAUDE_MODEL = "claude-3-haiku"

class DatabaseTools:
    """Database access layer for the AI agent - same as in agentic_aicopy.py"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self._connect()
    
    def _connect(self):
        """Establish database connection"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            print(f"✓ Connected to database: {self.db_path}")
        except Exception as e:
            print(f"✗ Database connection failed: {e}")
            # Don't exit, let the server continue without database
            self.conn = None
    
    def _dict_from_row(self, row) -> dict:
        """Convert sqlite3.Row to dictionary"""
        if row is None:
            return {}
        return {key: row[key] for key in row.keys()}
    
    def search_products(self, query: str, category: str = "", 
                       min_price: float = 0, max_price: float = 999999,
                       min_rating: float = 0, limit: int = 10) -> List[Dict]:
        """Full-text search with filtering"""
        if not self.conn:
            return [{"error": "Database not connected"}]
            
        cursor = self.conn.cursor()
        
        # Fallback to simple LIKE search if FTS fails
        try:
            sql = """
                SELECT p.* FROM products_fts fts
                JOIN products p ON fts.rowid = p.id
                WHERE products_fts MATCH ?
            """
            params = [query]
        except:
            sql = """
                SELECT * FROM products
                WHERE (title LIKE ? OR description LIKE ?)
            """
            params = [f"%{query}%", f"%{query}%"]
        
        if category:
            sql += " AND category LIKE ?"
            params.append(f"%{category}%")
        if min_price > 0:
            sql += " AND price >= ?"
            params.append(min_price)
        if max_price < 999999:
            sql += " AND price <= ?"
            params.append(max_price)
        if min_rating > 0:
            sql += " AND rating >= ?"
            params.append(min_rating)
        
        sql += " LIMIT ?"
        params.append(limit)
        
        try:
            results = cursor.execute(sql, params).fetchall()
            return [self._dict_from_row(row) for row in results]
        except Exception as e:
            return [{"error": str(e)}]
    
    def get_product_by_id(self, product_id: str) -> Optional[Dict]:
        """Get product by ASIN"""
        if not self.conn:
            return {"error": "Database not connected"}
            
        cursor = self.conn.cursor()
        result = cursor.execute(
            "SELECT * FROM products WHERE asin = ?", (product_id,)
        ).fetchone()
        return self._dict_from_row(result) if result else None
    
    def get_categories(self, limit: int = 20) -> List[Dict]:
        """Get all categories with counts"""
        if not self.conn:
            return [{"error": "Database not connected"}]
            
        cursor = self.conn.cursor()
        results = cursor.execute("""
            SELECT category, COUNT(*) as count
            FROM products
            WHERE category IS NOT NULL
            GROUP BY category
            ORDER BY count DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [self._dict_from_row(row) for row in results]
    
    def get_top_rated(self, category: str = "", limit: int = 10) -> List[Dict]:
        """Get top rated products"""
        if not self.conn:
            return [{"error": "Database not connected"}]
            
        cursor = self.conn.cursor()
        
        if category:
            results = cursor.execute("""
                SELECT * FROM products
                WHERE rating IS NOT NULL AND category LIKE ?
                ORDER BY rating DESC
                LIMIT ?
            """, (f"%{category}%", limit)).fetchall()
        else:
            results = cursor.execute("""
                SELECT * FROM products
                WHERE rating IS NOT NULL
                ORDER BY rating DESC
                LIMIT ?
            """, (limit,)).fetchall()
        
        return [self._dict_from_row(row) for row in results]
    
    def get_price_range_products(self, min_price: float, max_price: float,
                                 category: str = "", limit: int = 10) -> List[Dict]:
        """Get products in price range"""
        if not self.conn:
            return [{"error": "Database not connected"}]
            
        cursor = self.conn.cursor()
        
        if category:
            results = cursor.execute("""
                SELECT * FROM products
                WHERE price BETWEEN ? AND ?
                  AND category LIKE ?
                ORDER BY rating DESC
                LIMIT ?
            """, (min_price, max_price, f"%{category}%", limit)).fetchall()
        else:
            results = cursor.execute("""
                SELECT * FROM products
                WHERE price BETWEEN ? AND ?
                ORDER BY rating DESC
                LIMIT ?
            """, (min_price, max_price, limit)).fetchall()
        
        return [self._dict_from_row(row) for row in results]
    
    def get_recommendations(self, product_id: str, limit: int = 5) -> List[Dict]:
        """Get similar products"""
        if not self.conn:
            return [{"error": "Database not connected"}]
            
        # Get source product
        source = self.get_product_by_id(product_id)
        if not source or 'error' in source:
            return []
        
        cursor = self.conn.cursor()
        results = cursor.execute("""
            SELECT * FROM products
            WHERE asin != ?
              AND category = ?
              AND price BETWEEN ? AND ?
            ORDER BY rating DESC
            LIMIT ?
        """, (
            product_id,
            source['category'],
            source['price'] * 0.7,
            source['price'] * 1.3,
            limit
        )).fetchall()
        
        return [self._dict_from_row(row) for row in results]

# Initialize database tools and Claude interface
db_tools = DatabaseTools(DB_PATH)
claude_interface = None

if ClaudeModelInterface and SyConfig:
    try:
        # Load environment variables for BTP
        env_vars = {    
            "AICORE_AUTH_URL": os.getenv("AICORE_AUTH_URL", ""),
            "AICORE_CLIENT_ID": os.getenv("AICORE_CLIENT_ID", ""),
            "AICORE_CLIENT_SECRET": os.getenv("AICORE_CLIENT_SECRET", ""),
            "AICORE_RESOURCE_GROUP": os.getenv("AICORE_RESOURCE_GROUP", "default"),
            "AICORE_BASE_URL": os.getenv("AICORE_BASE_URL", "")
        }
        
        if all(env_vars.values()):
            config = SyConfig(
                auth_url=env_vars["AICORE_AUTH_URL"],
                client_id=env_vars["AICORE_CLIENT_ID"],
                client_secret=env_vars["AICORE_CLIENT_SECRET"],
                base_url=env_vars["AICORE_BASE_URL"],
                resource_group=env_vars["AICORE_RESOURCE_GROUP"]
            )
            claude_interface = ClaudeModelInterface(config)
            print("✓ Claude interface initialized")
        else:
            print("⚠️ Claude credentials not found in environment variables")
    except Exception as e:
        print(f"⚠️ Claude interface initialization failed: {e}")

# Create MCP Server
server = Server("agentic-ai-server")

@server.list_tools()
async def handle_list_tools() -> List[Tool]:
    """List all available tools"""
    return [
        Tool(
            name="search_products",
            description="Search for products by keyword with optional filters (category, price range, rating)",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keywords"
                    },
                    "category": {
                        "type": "string",
                        "description": "Optional category filter",
                        "default": ""
                    },
                    "min_price": {
                        "type": "number",
                        "description": "Minimum price (optional)",
                        "default": 0
                    },
                    "max_price": {
                        "type": "number",
                        "description": "Maximum price (optional)",
                        "default": 999999
                    },
                    "min_rating": {
                        "type": "number",
                        "description": "Minimum rating (optional)",
                        "default": 0
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default 10)",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_product_details",
            description="Get detailed information about a specific product by ASIN/product ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "Product ASIN or ID"
                    }
                },
                "required": ["product_id"]
            }
        ),
        Tool(
            name="get_categories",
            description="List all available product categories",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Max categories to return (default 20)",
                        "default": 20
                    }
                }
            }
        ),
        Tool(
            name="get_top_rated",
            description="Get highest rated products, optionally in a specific category",
            inputSchema={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Optional category filter",
                        "default": ""
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default 10)",
                        "default": 10
                    }
                }
            }
        ),
        Tool(
            name="get_price_range",
            description="Find products within a specific price range",
            inputSchema={
                "type": "object",
                "properties": {
                    "min_price": {
                        "type": "number",
                        "description": "Minimum price"
                    },
                    "max_price": {
                        "type": "number",
                        "description": "Maximum price"
                    },
                    "category": {
                        "type": "string",
                        "description": "Optional category filter",
                        "default": ""
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default 10)",
                        "default": 10
                    }
                },
                "required": ["min_price", "max_price"]
            }
        ),
        Tool(
            name="get_recommendations",
            description="Get product recommendations similar to a given product",
            inputSchema={
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "Source product ASIN"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max recommendations (default 5)",
                        "default": 5
                    }
                },
                "required": ["product_id"]
            }
        ),
        Tool(
            name="agentic_chat",
            description="Chat with an autonomous AI agent that can reason and use tools to help with e-commerce queries",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Your message or question for the AI agent"
                    },
                    "reset_conversation": {
                        "type": "boolean",
                        "description": "Whether to reset the conversation history",
                        "default": False
                    }
                },
                "required": ["message"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Execute a tool call"""
    try:
        if name == "search_products":
            result = db_tools.search_products(
                query=arguments["query"],
                category=arguments.get("category", ""),
                min_price=arguments.get("min_price", 0),
                max_price=arguments.get("max_price", 999999),
                min_rating=arguments.get("min_rating", 0),
                limit=arguments.get("limit", 10)
            )
            
        elif name == "get_product_details":
            result = db_tools.get_product_by_id(arguments["product_id"])
            
        elif name == "get_categories":
            result = db_tools.get_categories(arguments.get("limit", 20))
            
        elif name == "get_top_rated":
            result = db_tools.get_top_rated(
                category=arguments.get("category", ""),
                limit=arguments.get("limit", 10)
            )
            
        elif name == "get_price_range":
            result = db_tools.get_price_range_products(
                min_price=arguments["min_price"],
                max_price=arguments["max_price"],
                category=arguments.get("category", ""),
                limit=arguments.get("limit", 10)
            )
            
        elif name == "get_recommendations":
            result = db_tools.get_recommendations(
                product_id=arguments["product_id"],
                limit=arguments.get("limit", 5)
            )
            
        elif name == "agentic_chat":
            if claude_interface:
                # Simple chat implementation
                message = arguments["message"]
                reset = arguments.get("reset_conversation", False)
                
                system_prompt = """You are an intelligent e-commerce shopping assistant. 
                Help users find products, compare options, and make informed purchasing decisions.
                Be helpful, conversational, and provide detailed product recommendations."""
                
                try:
                    response = claude_interface.conversation_chat(
                        message=message,
                        model=CLAUDE_MODEL,
                        system_prompt=system_prompt,
                        reset_history=reset
                    )
                    result = {"response": response, "status": "success"}
                except Exception as e:
                    result = {"error": f"Claude interface error: {str(e)}", "status": "error"}
            else:
                result = {"error": "Claude interface not available", "status": "error"}
            
        else:
            result = {"error": f"Unknown tool: {name}"}
        
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
    except Exception as e:
        error_result = {"error": str(e), "tool": name, "arguments": arguments}
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]

async def serve():
    """Run the MCP server"""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="agentic-ai-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )

if __name__ == "__main__":
    print(" Starting Agentic AI MCP Server...")
    print(" Same functional capabilities as agentic_ai.py and added MCP concept")
    print(" Available tools: search_products, get_product_details, get_categories, get_top_rated, get_price_range, get_recommendations, agentic_chat")
    
    asyncio.run(serve())
