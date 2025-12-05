#This script is used to get the access token for the BTP LLM API
import json
import requests
import os 
import pathlib
import yaml

import sqlite3
import anthropic
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
import sys

from typing import Literal, Type, Union, Dict, Any, List, Callable
from typing import Callable
from ai_core_sdk.ai_core_v2_client import AICoreV2Client
from ai_api_client_sdk.models.status import Status
import time
from IPython.display import clear_output

from gen_ai_hub.proxy import get_proxy_client
from ai_api_client_sdk.models.status import Status
from gen_ai_hub.orchestration.models.config import OrchestrationConfig
from gen_ai_hub.orchestration.models.llm import LLM
from gen_ai_hub.orchestration.models.message import SystemMessage, UserMessage
from gen_ai_hub.orchestration.models.template import Template, TemplateValue
from gen_ai_hub.orchestration.models.content_filter import ContentFilter
from gen_ai_hub.orchestration.service import OrchestrationService

from claude_setup import ClaudeModelInterface, SyConfig


authUrl = "Enter your Key info"
clientid = "Enter your Key info"
clientsecret = "Enter your Key info"
apiUrl = "Enter your Key info"
AICORE_RESOURCE_GROUP = "default"

# Define Keys HERE
env_vars = {    
 "AICORE_AUTH_URL": authUrl,
 "AICORE_CLIENT_ID": clientid,
 "AICORE_CLIENT_SECRET": clientsecret,
 "AICORE_RESOURCE_GROUP": "default",
 "AICORE_BASE_URL": apiUrl
}

os.environ.update(env_vars)

config = SyConfig(
        auth_url=os.getenv("AICORE_AUTH_URL", ""),
        client_id=os.getenv("AICORE_CLIENT_ID", ""),
        client_secret=os.getenv("AICORE_CLIENT_SECRET", ""),
        base_url=os.getenv("AICORE_BASE_URL", "")
    )


# Initialize Claude interface
try:
    claude = ClaudeModelInterface(config)
    print("✓ Claude interface ready!")
except Exception as e:
    print(f"❌ Setup failed: {e}")
    


DB_PATH = "./data/ecommerce.db"
CLAUDE_MODEL = "anthropic--claude-4.5-sonnet"

# ============ DATABASE TOOLS ============

class DatabaseTools:
    """Database access layer for the AI agent"""
    
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
            sys.exit(1)
    
    def _dict_from_row(self, row) -> dict:
        """Convert sqlite3.Row to dictionary"""
        return {key: row[key] for key in row.keys()}
    
    def search_products(self, query: str, category: str = "", 
                       min_price: float = 0, max_price: float = 999999,
                       min_rating: float = 0, limit: int = 10) -> List[Dict]:
        """Full-text search with filtering"""
        cursor = self.conn.cursor()
        
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
        
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)
        
        try:
            results = cursor.execute(sql, params).fetchall()
            return [self._dict_from_row(row) for row in results]
        except Exception as e:
            return []
    
    def get_product_by_id(self, product_id: str) -> Optional[Dict]:
        """Get product by ASIN"""
        cursor = self.conn.cursor()
        result = cursor.execute(
            "SELECT * FROM products WHERE asin = ?", (product_id,)
        ).fetchone()
        return self._dict_from_row(result) if result else None
    
    def get_categories(self, limit: int = 20) -> List[Dict]:
        """Get all categories with counts"""
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
        cursor = self.conn.cursor()
        
        if category:
            results = cursor.execute("""
                SELECT * FROM products
                WHERE rating IS NOT NULL AND category LIKE ?
                ORDER BY rating DESC, reviews DESC
                LIMIT ?
            """, (f"%{category}%", limit)).fetchall()
        else:
            results = cursor.execute("""
                SELECT * FROM products
                WHERE rating IS NOT NULL
                ORDER BY rating DESC, reviews DESC
                LIMIT ?
            """, (limit,)).fetchall()
        
        return [self._dict_from_row(row) for row in results]
    
    def get_price_range_products(self, min_price: float, max_price: float,
                                 category: str = "", limit: int = 10) -> List[Dict]:
        """Get products in price range"""
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
        # Get source product
        source = self.get_product_by_id(product_id)
        if not source:
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

# ============ AGENTIC AI SYSTEM ============

@dataclass
class Tool:
    """Tool definition for the AI agent"""
    name: str
    description: str
    parameters: Dict
    function: Callable

class AgenticAI:
    """
    Autonomous AI agent that can:
    1. Understand user intent
    2. Plan multi-step actions
    3. Execute tools autonomously
    4. Provide coherent responses
    """
    
    def __init__(self, api_key: str, db_tools: DatabaseTools):
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        
        self.client = get_proxy_client()
        self.db = db_tools
        self.conversation_history = []
        self.tools = self._register_tools()
        
        print("✓ Agentic AI initialized")
    
    def _register_tools(self) -> List[Tool]:
        """Register all available tools"""
        return [
            Tool(
                name="search_products",
                description="Search for products by keyword with optional filters (category, price range, rating)",
                parameters={
                    "query": "Search keywords",
                    "category": "Optional category filter",
                    "min_price": "Minimum price (optional)",
                    "max_price": "Maximum price (optional)",
                    "min_rating": "Minimum rating (optional)",
                    "limit": "Max results (default 10)"
                },
                function=self.db.search_products
            ),
            Tool(
                name="get_product_details",
                description="Get detailed information about a specific product by ASIN/product ID",
                parameters={
                    "product_id": "Product ASIN or ID"
                },
                function=self.db.get_product_by_id
            ),
            Tool(
                name="get_categories",
                description="List all available product categories",
                parameters={
                    "limit": "Max categories to return (default 20)"
                },
                function=self.db.get_categories
            ),
            Tool(
                name="get_top_rated",
                description="Get highest rated products, optionally in a specific category",
                parameters={
                    "category": "Optional category filter",
                    "limit": "Max results (default 10)"
                },
                function=self.db.get_top_rated
            ),
            Tool(
                name="get_price_range",
                description="Find products within a specific price range",
                parameters={
                    "min_price": "Minimum price",
                    "max_price": "Maximum price",
                    "category": "Optional category filter",
                    "limit": "Max results (default 10)"
                },
                function=self.db.get_price_range_products
            ),
            Tool(
                name="get_recommendations",
                description="Get product recommendations similar to a given product",
                parameters={
                    "product_id": "Source product ASIN",
                    "limit": "Max recommendations (default 5)"
                },
                function=self.db.get_recommendations
            )
        ]
    
    def _create_system_prompt(self) -> str:
        """Create system prompt with tool descriptions"""
        tools_description = "\n\n".join([
            f"**{tool.name}**\n{tool.description}\nParameters: {json.dumps(tool.parameters, indent=2)}"
            for tool in self.tools
        ])
        
        return f"""You are an intelligent e-commerce shopping assistant with autonomous decision-making capabilities.

Your goal is to help users find products, compare options, and make informed purchasing decisions.

You have access to these tools:

{tools_description}

IMPORTANT INSTRUCTIONS:
1. Always think step-by-step about what the user needs
2. Use tools to gather information before responding
3. You can use multiple tools in sequence to answer complex queries
4. When showing products, include key details: name, price, rating, asin
5. Provide helpful recommendations and comparisons
6. Be conversational and natural
7. If a tool returns no results, suggest alternatives

When you decide to use a tool, respond with EXACTLY this format:
TOOL_CALL: tool_name
PARAMETERS: {{"param1": "value1", "param2": "value2"}}

When you have all the information needed, respond with:
RESPONSE: Your natural language response to the user

You can make multiple TOOL_CALL decisions before giving a final RESPONSE.

Example interaction:
User: "Find me wireless headphones under ₹3000"

Your reasoning:
TOOL_CALL: search_products
PARAMETERS: {{"query": "wireless headphones", "max_price": 3000, "limit": 5}}

[After receiving results]
RESPONSE: I found 5 great wireless headphones under ₹3000:

1. **Sony WH-XYZ** - ₹2,499 (4.5★)
   - Great noise cancellation
   - 20 hour battery
   
[etc...]

Now begin! Be helpful, intelligent, and autonomous."""
    
    def _parse_agent_response(self, response: str) -> tuple:
        """Parse agent's response for tool calls or final response"""
        lines = response.strip().split('\n')
        
        # Check for tool call
        if lines[0].startswith("TOOL_CALL:"):
            tool_name = lines[0].replace("TOOL_CALL:", "").strip()
            
            # Find PARAMETERS line
            params_str = ""
            for line in lines[1:]:
                if line.startswith("PARAMETERS:"):
                    params_str = line.replace("PARAMETERS:", "").strip()
                    break
            
            try:
                params = json.loads(params_str)
                return ("tool_call", tool_name, params)
            except json.JSONDecodeError:
                return ("response", response)
        
        # Check for final response
        elif lines[0].startswith("RESPONSE:"):
            response_text = '\n'.join(lines).replace("RESPONSE:", "").strip()
            return ("response", response_text)
        
        # Default to response
        return ("response", response)
    
    def _execute_tool(self, tool_name: str, parameters: Dict) -> any:
        """Execute a tool by name with parameters"""
        tool = next((t for t in self.tools if t.name == tool_name), None)
        
        if not tool:
            return {"error": f"Tool '{tool_name}' not found"}
        
        try:
            # Call the tool function with parameters
            result = tool.function(**parameters)
            return result
        except Exception as e:
            return {"error": str(e)}
    
    def chat(self, user_message: str, max_iterations: int = 5) -> str:
        """
        Process user message with autonomous agent behavior
        
        Args:
            user_message: User's query
            max_iterations: Max tool calls before forcing response
        
        Returns:
            Agent's final response
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        iteration = 0
        agent_thinking = []
        
        while iteration < max_iterations:
            iteration += 1
            
            # Build messages for Claude
            messages = self.conversation_history.copy()
            
            # Add previous tool results if any
            if agent_thinking:
                thinking_context = "\n\n".join([
                    f"Tool used: {t['tool']}\nResult: {json.dumps(t['result'], indent=2)}"
                    for t in agent_thinking
                ])
                messages.append({
                    "role": "user",
                    "content": f"Previous tool results:\n{thinking_context}\n\nNow decide your next action."
                })
            
            # Call Claude

            '''response = self.client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=4000,
                system=self._create_system_prompt(),
                messages=messages
            )'''
            response = claude.conversation_chat(
                user_message,
                system_prompt=self._create_system_prompt()
                )
            print(response)
            '''response = send_request(
                self._create_system_prompt(),
                _model='anthropic--claude-4-sonnet',
                query=query
                #scratchpad="\n".join(scratchpad),
                #tools="\n\n".join(self.tool_descriptions),
                #tool_names=register_tools(self) -> List[Tool]
            )'''
            agent_output = response
            
            # Parse response
            action_type, *action_data = self._parse_agent_response(agent_output)
            
            if action_type == "tool_call":
                tool_name, parameters = action_data
                
                print(f"\n🔧 Agent using tool: {tool_name}")
                print(f"   Parameters: {json.dumps(parameters, indent=2)}")
                
                # Execute tool
                result = self._execute_tool(tool_name, parameters)
                
                print(f"   Result: {len(result) if isinstance(result, list) else 'Done'}")
                
                # Store tool execution
                agent_thinking.append({
                    "tool": tool_name,
                    "parameters": parameters,
                    "result": result
                })
                
                # Continue loop for next decision
                continue
            
            elif action_type == "response":
                final_response = action_data[0]
                
                # Add to conversation history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": final_response
                })
                
                return final_response
        
        # Max iterations reached
        return "I apologize, but I need more time to process this request. Could you try rephrasing your question?"
    
    def reset_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("✓ Conversation reset")

class CLI:
    """Interactive command-line interface"""
    
    def __init__(self, agent: AgenticAI):
        self.agent = agent
    
    def run(self):
        """Run interactive CLI"""
        print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║          AGENTIC AI E-COMMERCE ASSISTANT                       ║
    ║                                                                ║
    ║  An autonomous AI agent that helps you find and compare        ║
    ║  products using natural language.                              ║
    ║                                                                ║
    ║  Commands:                                                     ║
    ║    - Just type your question naturally                         ║
    ║    - 'reset' - Start new conversation                          ║
    ║    - 'quit' - Exit                                             ║
    ╚════════════════════════════════════════════════════════════════╝
        """)
        
        while True:
            try:
                # Get user input
                user_input = input("\n🧑 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() == 'quit':
                    print("\n👋 Goodbye!")
                    break
                
                if user_input.lower() == 'reset':
                    self.agent.reset_conversation()
                    continue
                
                # Get agent response
                print("\n🤖 Agent: ", end="", flush=True)
                response = self.agent.chat(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

# ============ WEB API (Flask) ============

def create_web_api(agent: AgenticAI):
    """Create Flask web API for the agent"""
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    
    app = Flask(__name__)
    CORS(app)
    
    @app.route('/chat', methods=['POST'])
    def chat():
        """Chat endpoint"""
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        
        try:
            response = agent.chat(user_message)
            return jsonify({
                "response": response,
                "status": "success"
            })
        except Exception as e:
            return jsonify({
                "error": str(e),
                "status": "error"
            }), 500
    
    @app.route('/reset', methods=['POST'])
    def reset():
        """Reset conversation"""
        agent.reset_conversation()
        return jsonify({"status": "success"})
    
    @app.route('/health', methods=['GET'])
    def health():
        """Health check"""
        return jsonify({"status": "healthy"})
    
    return app




# ============ MAIN ============

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Agentic AI E-commerce Assistant")
    parser.add_argument('--mode', choices=['cli', 'web'], default='cli',
                       help='Run mode: cli (interactive) or web (API server)')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port for web API (default: 5000)')
    
    args = parser.parse_args()
    
    # Check API key
    
    
    # Check database
    if not os.path.exists(DB_PATH):
        print(f"❌ Error: Database not found at {DB_PATH}")
        print("\nPlease run: python database_setup.py")
        sys.exit(1)
    
    # Initialize
    db_tools = DatabaseTools(DB_PATH)
    proxy_client=get_proxy_client()
    agent = AgenticAI(proxy_client, db_tools)
    
    # Run appropriate mode
    if args.mode == 'cli':
        cli = CLI(agent)
        cli.run()
    else:
        print(f"\n🌐 Starting web API on port {args.port}...")
        app = create_web_api(agent)
        app.run(host='0.0.0.0', port=args.port, debug=False)

if __name__ == "__main__":
    main()