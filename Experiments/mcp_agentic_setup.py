#!/usr/bin/env python3
"""
MCP AGENTIC SETUP - Hybrid Architecture
Combines the power of MCP (Model Context Protocol) with Agentic AI capabilities
Similar architecture to agentic_aicopy.py but using MCP for tool integration
"""

import json
import sys
import logging
import sqlite3
import asyncio
import os
from typing import List, Dict, Optional, Any, Sequence, Callable
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict

# MCP imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    TextContent,
    Tool,
    CallToolResult,
    GetPromptResult,
    Prompt,
    PromptMessage,
    Role,
)

# SAP BTP and Claude imports
from gen_ai_hub.proxy import get_proxy_client
from gen_ai_hub.orchestration.models.config import OrchestrationConfig
from gen_ai_hub.orchestration.models.llm import LLM
from gen_ai_hub.orchestration.models.message import SystemMessage, UserMessage
from gen_ai_hub.orchestration.service import OrchestrationService

from backend.claude_setup import ClaudeModelInterface, SyConfig

# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration constants
DB_PATH = "./data/ecommerce.db"
CLAUDE_MODEL = "anthropic--claude-4.5-sonnet"

# Configuration
authUrl = "Enter your Key info"
clientid = "Enter your Key info"
clientsecret = "Enter your Key info"
apiUrl = "Enter your Key info"

# Set environment variables
env_vars = {    
    "AICORE_AUTH_URL": authUrl,
    "AICORE_CLIENT_ID": clientid,
    "AICORE_CLIENT_SECRET": clientsecret,
    "AICORE_RESOURCE_GROUP": "default",
    "AICORE_BASE_URL": apiUrl
}
os.environ.update(env_vars)

# Initialize Claude configuration
config = SyConfig(
    auth_url=os.getenv("AICORE_AUTH_URL", ""),
    client_id=os.getenv("AICORE_CLIENT_ID", ""),
    client_secret=os.getenv("AICORE_CLIENT_SECRET", ""),
    base_url=os.getenv("AICORE_BASE_URL", "")
)

# Global variables
db_conn = None
claude_interface = None

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

def get_claude():
    """Get Claude interface"""
    global claude_interface
    if claude_interface is None:
        try:
            claude_interface = ClaudeModelInterface(config)
            logger.info("✓ Claude interface ready!")
        except Exception as e:
            logger.error(f"❌ Claude setup failed: {e}")
            raise
    return claude_interface

def dict_from_row(row) -> dict:
    """Convert sqlite3.Row to dictionary"""
    if not row:
        return {}
    return {key: row[key] for key in row.keys()}

# ============ AGENTIC AI CORE ============

@dataclass
class AgenticTool:
    """Tool definition for the agentic AI"""
    name: str
    description: str
    parameters: Dict
    function: Callable
    mcp_compatible: bool = True

class AgenticMCPCore:
    """
    Hybrid Agentic AI that uses MCP tools for capability extension
    Combines autonomous decision-making with standardized tool access
    """
    
    def __init__(self):
        self.db = get_db()
        self.claude = get_claude()
        self.conversation_history = []
        self.available_tools = {}
        self.agent_state = {
            "current_task": None,
            "context": {},
            "reasoning_chain": []
        }
        logger.info("✓ Agentic MCP Core initialized")
    
    def _create_agentic_system_prompt(self) -> str:
        """Create system prompt for agentic behavior with MCP tool awareness"""
        return """You are an advanced autonomous e-commerce assistant with hybrid Agentic AI + MCP architecture.

CORE CAPABILITIES:
1. **Autonomous Decision Making**: You can reason, plan, and execute multi-step tasks
2. **MCP Tool Integration**: You have access to standardized MCP tools for data access
3. **Adaptive Behavior**: You learn from interactions and adapt your approach
4. **Context Awareness**: You maintain conversation context and user preferences

AUTONOMOUS BEHAVIOR GUIDELINES:
- Think step-by-step and explain your reasoning
- Use multiple tools in sequence when needed for complex queries
- Proactively suggest alternatives when initial approaches don't work
- Maintain context across conversation turns
- Adapt your communication style to user needs

MCP TOOL INTEGRATION:
- You have access to powerful e-commerce tools through MCP protocol
- Tools provide structured data that you should analyze and synthesize
- Always provide value-added insights, not just raw tool outputs
- Combine multiple tool results for comprehensive responses

DECISION FRAMEWORK:
1. **Understand**: Analyze user intent and context
2. **Plan**: Determine optimal tool sequence and approach
3. **Execute**: Call tools with appropriate parameters
4. **Synthesize**: Analyze results and provide intelligent insights
5. **Adapt**: Learn from outcomes and adjust future behavior

Your goal is to be genuinely helpful by combining autonomous reasoning with powerful tool capabilities."""

    async def process_user_message(self, user_message: str, max_iterations: int = 5) -> str:
        """
        Process user message with autonomous agentic behavior using MCP tools
        """
        # Add to conversation history
        self.conversation_history.append({
            "role": "user", 
            "content": user_message
        })
        
        # Update agent state
        self.agent_state["current_task"] = user_message
        self.agent_state["reasoning_chain"] = []
        
        # Autonomous reasoning and tool execution loop
        iteration = 0
        tool_results = []
        
        while iteration < max_iterations:
            iteration += 1
            
            # Build context for Claude
            context_messages = self.conversation_history.copy()
            
            # Add tool results context if any
            if tool_results:
                tools_context = "\n\n".join([
                    f"TOOL: {result['tool_name']}\nRESULT: {json.dumps(result['result'], indent=2)}"
                    for result in tool_results
                ])
                context_messages.append({
                    "role": "user",
                    "content": f"Previous tool results:\n{tools_context}\n\nAnalyze these results and determine your next action. You can use more tools or provide a final response."
                })
            
            # Get autonomous decision from Claude
            try:
                response = self.claude.conversation_chat(
                    user_message if iteration == 1 else "Continue with your analysis and next action.",
                    system_prompt=self._create_agentic_system_prompt()
                )
                
                logger.info(f"Iteration {iteration}: Claude response received")
                
                # Parse response for tool calls or final answer
                action_type, action_data = self._parse_agentic_response(response)
                
                if action_type == "tool_call":
                    tool_name, parameters = action_data
                    
                    logger.info(f"🤖 Agent decided to use tool: {tool_name}")
                    logger.info(f"   Parameters: {json.dumps(parameters, indent=2)}")
                    
                    # Execute tool through MCP
                    tool_result = await self._execute_mcp_tool(tool_name, parameters)
                    
                    # Store result for next iteration
                    tool_results.append({
                        "tool_name": tool_name,
                        "parameters": parameters,
                        "result": tool_result
                    })
                    
                    # Add to reasoning chain
                    self.agent_state["reasoning_chain"].append({
                        "action": "tool_call",
                        "tool": tool_name,
                        "reasoning": f"Used {tool_name} to gather information"
                    })
                    
                    # Continue loop for next decision
                    continue
                
                elif action_type == "final_response":
                    final_answer = action_data
                    
                    # Add to conversation history
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": final_answer
                    })
                    
                    # Log agent's reasoning chain
                    logger.info(f"✅ Agent completed task with {len(tool_results)} tool calls")
                    
                    return final_answer
                
            except Exception as e:
                logger.error(f"Error in iteration {iteration}: {e}")
                return f"I encountered an error while processing your request: {str(e)}"
        
        # Max iterations reached
        return "I need more time to process this complex request. Could you try rephrasing or breaking it into smaller parts?"
    
    def _parse_agentic_response(self, response: str) -> tuple:
        """Parse Claude's response for autonomous decisions"""
        response_lower = response.lower()
        
        # Look for explicit tool call patterns
        if "use tool:" in response_lower or "call tool:" in response_lower:
            # Extract tool name and parameters
            lines = response.split('\n')
            tool_name = ""
            parameters = {}
            
            for line in lines:
                if "use tool:" in line.lower() or "call tool:" in line.lower():
                    tool_name = line.split(':')[1].strip()
                elif line.strip().startswith('{') and line.strip().endswith('}'):
                    try:
                        parameters = json.loads(line.strip())
                    except:
                        pass
            
            if tool_name:
                return ("tool_call", (tool_name, parameters))
        
        # Check if Claude is requesting more information or tools
        decision_keywords = [
            "let me search", "i'll look up", "i need to find", 
            "let me check", "i should search", "i'll search for"
        ]
        
        if any(keyword in response_lower for keyword in decision_keywords):
            # Try to extract search intent
            if "search" in response_lower:
                # Extract search query from response
                query = self._extract_search_query(response)
                return ("tool_call", ("intelligent_search", {"query": query}))
        
        # Default to final response
        return ("final_response", response)
    
    def _extract_search_query(self, text: str) -> str:
        """Extract search query from natural language text"""
        # Simple extraction logic - can be enhanced with NLP
        text_lower = text.lower()
        
        # Look for quoted terms
        if '"' in text:
            import re
            quotes = re.findall(r'"([^"]*)"', text)
            if quotes:
                return quotes[0]
        
        # Look for "search for X" patterns
        search_patterns = [
            r"search for (.+?)(?:\.|$|,)",
            r"look up (.+?)(?:\.|$|,)",
            r"find (.+?)(?:\.|$|,)"
        ]
        
        import re
        for pattern in search_patterns:
            match = re.search(pattern, text_lower)
            if match:
                return match.group(1).strip()
        
        # Fallback: use the original user message context
        if self.agent_state.get("current_task"):
            return self.agent_state["current_task"]
        
        return "general products"
    
    async def _execute_mcp_tool(self, tool_name: str, parameters: dict) -> dict:
        """Execute MCP tool and return result"""
        # This will be called by the MCP server's tool execution
        # For now, return a placeholder that indicates MCP integration
        return {
            "mcp_tool": tool_name,
            "parameters": parameters,
            "note": "This will be executed by MCP server",
            "status": "pending_mcp_execution"
        }
    
    def reset_conversation(self):
        """Reset conversation and agent state"""
        self.conversation_history = []
        self.agent_state = {
            "current_task": None,
            "context": {},
            "reasoning_chain": []
        }
        logger.info("✓ Agent conversation reset")

# ============ MCP SERVER IMPLEMENTATION ============

# Create the MCP server
app = Server("agentic-ecommerce-mcp")

# Global agentic core instance
agentic_core = None

def get_agentic_core():
    """Get or create agentic core instance"""
    global agentic_core
    if agentic_core is None:
        agentic_core = AgenticMCPCore()
    return agentic_core

@app.list_tools()
async def list_tools() -> List[Tool]:
    """List all available MCP tools with agentic capabilities"""
    return [
        Tool(
            name="agentic_chat",
            description="Autonomous AI assistant that can reason, plan, and execute complex e-commerce tasks using available tools",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_message": {
                        "type": "string",
                        "description": "User's natural language request or question"
                    },
                    "context": {
                        "type": "object",
                        "description": "Optional context information",
                        "default": {}
                    },
                    "max_iterations": {
                        "type": "integer",
                        "description": "Maximum reasoning iterations (default: 5)",
                        "default": 5
                    }
                },
                "required": ["user_message"]
            }
        ),
        Tool(
            name="intelligent_search",
            description="AI-powered product search with context understanding and intent detection",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query"
                    },
                    "context": {
                        "type": "string",
                        "description": "Additional context for the search",
                        "default": ""
                    },
                    "filters": {
                        "type": "object",
                        "description": "Optional filters (category, price_range, min_rating, etc.)",
                        "default": {}
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_product_details",
            description="Get comprehensive product information by ASIN",
            inputSchema={
                "type": "object",
                "properties": {
                    "asin": {
                        "type": "string",
                        "description": "Product ASIN identifier"
                    }
                },
                "required": ["asin"]
            }
        ),
        Tool(
            name="get_recommendations",
            description="Get smart product recommendations based on user preferences and behavior",
            inputSchema={
                "type": "object",
                "properties": {
                    "based_on_asin": {
                        "type": "string",
                        "description": "Base recommendations on this product ASIN",
                        "default": ""
                    },
                    "user_preferences": {
                        "type": "object",
                        "description": "User preferences (categories, price_range, etc.)",
                        "default": {}
                    },
                    "recommendation_type": {
                        "type": "string",
                        "description": "'similar', 'trending', or 'mixed'",
                        "default": "mixed"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of recommendations",
                        "default": 5
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="get_categories",
            description="List all product categories with statistics",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum categories to return",
                        "default": 20
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="price_analysis",
            description="Analyze price trends and market positioning for products",
            inputSchema={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Product category to analyze",
                        "default": ""
                    },
                    "price_range": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Price range [min, max] to focus analysis",
                        "default": [0, 999999]
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="competitive_analysis",
            description="Analyze competitive landscape for a specific product",
            inputSchema={
                "type": "object",
                "properties": {
                    "product_asin": {
                        "type": "string",
                        "description": "Target product ASIN for competitive analysis"
                    },
                    "analysis_depth": {
                        "type": "string",
                        "description": "'basic' or 'comprehensive'",
                        "default": "basic"
                    }
                },
                "required": ["product_asin"]
            }
        ),
        Tool(
            name="reset_agent",
            description="Reset the agent's conversation history and state",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> Sequence[TextContent]:
    """Handle MCP tool calls with agentic behavior"""
    try:
        agent = get_agentic_core()
        
        if name == "agentic_chat":
            # Main agentic interface
            user_message = arguments.get("user_message", "")
            context = arguments.get("context", {})
            max_iterations = arguments.get("max_iterations", 5)
            
            # Update agent context
            agent.agent_state["context"].update(context)
            
            # Process with autonomous behavior
            result = await agent.process_user_message(user_message, max_iterations)
            return [TextContent(type="text", text=result)]
        
        elif name == "intelligent_search":
            result = await intelligent_search(
                arguments.get("query", ""),
                arguments.get("context", ""),
                arguments.get("filters", {}),
                arguments.get("limit", 10)
            )
        
        elif name == "get_product_details":
            result = await get_product_details(arguments.get("asin", ""))
        
        elif name == "get_recommendations":
            result = await get_recommendations(
                arguments.get("based_on_asin", ""),
                arguments.get("user_preferences", {}),
                arguments.get("recommendation_type", "mixed"),
                arguments.get("limit", 5)
            )
        
        elif name == "get_categories":
            result = await get_categories(arguments.get("limit", 20))
        
        elif name == "price_analysis":
            result = await price_analysis(
                arguments.get("category", ""),
                arguments.get("price_range", [0, 999999])
            )
        
        elif name == "competitive_analysis":
            result = await competitive_analysis(
                arguments.get("product_asin", ""),
                arguments.get("analysis_depth", "basic")
            )
        
        elif name == "reset_agent":
            agent.reset_conversation()
            result = json.dumps({"status": "success", "message": "Agent state reset"})
        
        else:
            result = json.dumps({"error": f"Unknown tool: {name}"})
        
        return [TextContent(type="text", text=result)]
        
    except Exception as e:
        logger.error(f"Tool {name} failed: {e}")
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

# ============ TOOL IMPLEMENTATIONS ============

async def intelligent_search(query: str, context: str = "", filters: dict = None, limit: int = 10) -> str:
    """AI-powered intelligent product search"""
    db = get_db()
    cursor = db.cursor()
    
    filters = filters or {}
    
    # Build search query
    sql = """
        SELECT p.*, 
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
    
    # Apply filters
    if filters.get('category'):
        sql += " AND p.category LIKE ?"
        params.append(f"%{filters['category']}%")
    
    if filters.get('min_price'):
        sql += " AND p.price >= ?"
        params.append(filters['min_price'])
    
    if filters.get('max_price'):
        sql += " AND p.price <= ?"
        params.append(filters['max_price'])
    
    if filters.get('min_rating'):
        sql += " AND p.rating >= ?"
        params.append(filters['min_rating'])
    
    sql += " ORDER BY p.rating DESC, p.num_ratings DESC LIMIT ?"
    params.append(limit)
    
    try:
        results = cursor.execute(sql, params).fetchall()
        products = [dict_from_row(row) for row in results]
        
        return json.dumps({
            "query": query,
            "context": context,
            "applied_filters": filters,
            "result_count": len(products),
            "products": products
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Search failed: {str(e)}"})

async def get_product_details(asin: str) -> str:
    """Get detailed product information"""
    db = get_db()
    cursor = db.cursor()
    
    result = cursor.execute("SELECT * FROM products WHERE asin = ?", (asin,)).fetchone()
    
    if result:
        product = dict_from_row(result)
        return json.dumps(product, indent=2)
    else:
        return json.dumps({"error": f"Product {asin} not found"})

async def get_recommendations(based_on_asin: str = "", user_preferences: dict = None,
                            recommendation_type: str = "mixed", limit: int = 5) -> str:
    """Get smart product recommendations"""
    db = get_db()
    cursor = db.cursor()
    
    user_preferences = user_preferences or {}
    
    if based_on_asin:
        # Get recommendations based on specific product
        source_product = cursor.execute(
            "SELECT * FROM products WHERE asin = ?", (based_on_asin,)
        ).fetchone()
        
        if source_product:
            source = dict_from_row(source_product)
            # Find similar products in same category
            results = cursor.execute("""
                SELECT * FROM products
                WHERE asin != ?
                  AND category = ?
                  AND price BETWEEN ? AND ?
                ORDER BY rating DESC
                LIMIT ?
            """, (
                based_on_asin,
                source['category'],
                source['price'] * 0.7 if source['price'] else 0,
                source['price'] * 1.3 if source['price'] else 999999,
                limit
            )).fetchall()
        else:
            results = []
    else:
        # Get general recommendations based on preferences
        sql = "SELECT * FROM products WHERE rating >= 4.0"
        params = []
        
        if user_preferences.get('categories'):
            category_conditions = ' OR '.join(['category LIKE ?' for _ in user_preferences['categories']])
            sql += f" AND ({category_conditions})"
            params.extend([f"%{cat}%" for cat in user_preferences['categories']])
        
        if recommendation_type == "trending":
            sql += " ORDER BY num_ratings DESC, rating DESC"
        else:
            sql += " ORDER BY rating DESC, num_ratings DESC"
        
        sql += " LIMIT ?"
        params.append(limit)
        
        results = cursor.execute(sql, params).fetchall()
    
    recommendations = [dict_from_row(row) for row in results]
    
    return json.dumps({
        "based_on_asin": based_on_asin,
        "user_preferences": user_preferences,
        "recommendation_type": recommendation_type,
        "recommendations": recommendations
    }, indent=2)

async def get_categories(limit: int = 20) -> str:
    """List product categories with statistics"""
    db = get_db()
    cursor = db.cursor()
    
    results = cursor.execute("""
        SELECT category, COUNT(*) as product_count, AVG(price) as avg_price, AVG(rating) as avg_rating
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

async def price_analysis(category: str = "", price_range: List[float] = None) -> str:
    """Analyze price trends and market positioning"""
    db = get_db()
    cursor = db.cursor()
    
    price_range = price_range or [0, 999999]
    
    # Base query
    where_clause = "WHERE price BETWEEN ? AND ?"
    params = price_range
    
    if category:
        where_clause += " AND category LIKE ?"
        params.append(f"%{category}%")
    
    # Price distribution analysis
    price_stats = cursor.execute(f"""
        SELECT 
            COUNT(*) as total_products,
            AVG(price) as avg_price,
            MIN(price) as min_price,
            MAX(price) as max_price,
            AVG(rating) as avg_rating
        FROM products
        {where_clause}
    """, params).fetchone()
    
    # Price segments
    price_segments = cursor.execute(f"""
        SELECT 
            CASE 
                WHEN price <= 50 THEN 'Budget (≤₹50)'
                WHEN price <= 200 THEN 'Mid-range (₹50-200)'
                WHEN price <= 500 THEN 'Premium (₹200-500)'
                ELSE 'Luxury (>₹500)'
            END as price_segment,
            COUNT(*) as product_count,
            AVG(rating) as avg_rating
        FROM products
        {where_clause}
        GROUP BY price_segment
        ORDER BY MIN(price)
    """, params).fetchall()
    
    analysis = {
        "category": category or "All Categories",
        "price_range": price_range,
        "overall_stats": dict_from_row(price_stats),
        "price_segments": [dict_from_row(row) for row in price_segments]
    }
    
    return json.dumps(analysis, indent=2)

async def competitive_analysis(product_asin: str, analysis_depth: str = "basic") -> str:
    """Analyze competitive landscape for a product"""
    db = get_db()
    cursor = db.cursor()
    
    # Get target product
    target = cursor.execute("SELECT * FROM products WHERE asin = ?", (product_asin,)).fetchone()
    
    if not target:
        return json.dumps({"error": f"Product {product_asin} not found"})
    
    target = dict_from_row(target)
    
    # Find competitors in same category
    competitors = cursor.execute("""
        SELECT *, ABS(price - ?) as price_difference
        FROM products 
        WHERE category = ? 
          AND asin != ?
        ORDER BY rating DESC, num_ratings DESC
        LIMIT 10
    """, (target['price'] or 0, target['category'], product_asin)).fetchall()
    
    competitor_data = [dict_from_row(row) for row in competitors]
    
    # Calculate competitive position
    if competitor_data:
        prices = [c['price'] for c in competitor_data if c['price']]
        ratings = [c['rating'] for c in competitor_data if c['rating']]
        
        position = {
            "price_rank": sum(1 for p in prices if p > (target['price'] or 0)) + 1,
            "rating_rank": sum(1 for r in ratings if r < (target['rating'] or 0)) + 1,
            "total_competitors": len(competitor_data)
        }
    else:
        position = {"message": "No direct competitors found"}
    
    return json.dumps({
        "target_product": target,
        "competitive_position": position,
        "top_competitors": competitor_data[:5],
        "analysis_depth": analysis_depth
    }, indent=2)

# ============ MAIN SERVER FUNCTION ============

async def main():
    """Main function to run the Agentic MCP Server"""
    try:
        # Initialize database connection
        db = get_db()
        
        # Initialize Claude
        claude = get_claude()
        
        # Initialize agentic core
        agent = get_agentic_core()
        
        # Get product count
        count = db.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        
        logger.info("🚀 AGENTIC MCP SERVER STARTING")
        logger.info(f"✓ Database connected with {count} products")
        logger.info("✓ Claude interface ready")
        logger.info("✓ Agentic AI core initialized")
        logger.info("✓ MCP tools registered")
        logger.info("🔄 Hybrid Agentic + MCP architecture active")
        
        # Run the MCP server
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())
            
    except Exception as e:
        logger.error(f"✗ Server initialization failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
