#!/usr/bin/env python3
"""
Agentic AI MCP Server Setup
===========================

This script sets up the Agentic AI MCP Server with the same functional capabilities
as agentic_ai.py. It configures the MCP settings and provides testing functionality.
"""

import json
import os
import sys
from pathlib import Path

# MCP Settings Configuration
MCP_SETTINGS_PATH = Path.home() / "AppData/Roaming/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json"

def read_mcp_settings():
    """Read existing MCP settings"""
    if MCP_SETTINGS_PATH.exists():
        try:
            with open(MCP_SETTINGS_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading MCP settings: {e}")
            return {"mcpServers": {}}
    else:
        return {"mcpServers": {}}

def write_mcp_settings(settings):
    """Write MCP settings to file"""
    try:
        # Ensure directory exists
        MCP_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        with open(MCP_SETTINGS_PATH, 'w') as f:
            json.dump(settings, f, indent=2)
        print(f"✓ MCP settings written to: {MCP_SETTINGS_PATH}")
        return True
    except Exception as e:
        print(f"✗ Error writing MCP settings: {e}")
        return False

def setup_agentic_mcp_server():
    """Set up the Agentic AI MCP server configuration"""
    print("🚀 Setting up Agentic AI MCP Server...")
    
    # Get the current script directory
    current_dir = Path(__file__).parent.absolute()
    server_path = current_dir / "agentic_mcp_server.py"
    
    if not server_path.exists():
        print(f"✗ Server script not found at: {server_path}")
        return False
    
    # Read existing settings
    settings = read_mcp_settings()
    
    # Environment variables for SAP BTP (same as in agentic_aicopy.py)
    env_vars = {
        "AICORE_AUTH_URL": os.getenv("AICORE_AUTH_URL", "enter your key information"),
        "AICORE_CLIENT_ID": os.getenv("AICORE_CLIENT_ID", "enter your key information"),
        "AICORE_CLIENT_SECRET": os.getenv("AICORE_CLIENT_SECRET", "enter your key information"),
        "AICORE_BASE_URL": os.getenv("AICORE_BASE_URL", "enter your key information"),
        "AICORE_RESOURCE_GROUP": os.getenv("AICORE_RESOURCE_GROUP", "default")
    }
    
    # Add the Agentic AI server configuration
    settings["mcpServers"]["agentic-ai"] = {
        "command": "python",
        "args": [str(server_path)],
        "env": env_vars,
        "disabled": False,
        "autoApprove": []
    }
    
    # Write settings back
    if write_mcp_settings(settings):
        print("✓ Agentic AI MCP Server configured successfully!")
        print("\nServer Configuration:")
        print(f"  Name: agentic-ai")
        print(f"  Command: python {server_path}")
        print(f"  Environment variables: {len(env_vars)} variables set")
        print(f"  Disabled: False")
        print(f"  Auto-approve: []")
        
        print("\n Available Tools:")
        print("  - search_products: Search for products with filters")
        print("  - get_product_details: Get detailed product information")
        print("  - get_categories: List all product categories")
        print("  - get_top_rated: Get highest rated products")
        print("  - get_price_range: Find products in price range")
        print("  - get_recommendations: Get product recommendations")
        print("  - agentic_chat: Chat with autonomous AI agent")
        
        print("\n Same Functional Capabilities as agentic_ai.py:")
        print("  ✓ Database operations (SQLite with FTS)")
        print("  ✓ Claude AI interface for autonomous behavior")
        print("  ✓ Tool execution and reasoning")
        print("  ✓ E-commerce product assistance")
        print("  ✓ Conversation management")
        print("  ✓ Price filtering and recommendations")
        
        return True
    
    return False

def test_server():
    """Test the MCP server functionality"""
    print("\n Testing Agentic AI MCP Server...")
    
    try:
        # Import the server module
        sys.path.append(str(Path(__file__).parent))
        from agentic_mcp_server import AgenticAIMCPServer, DatabaseTools
        
        print("✓ Server module imported successfully")
        
        # Test database connection
        db_tools = DatabaseTools("../data/ecommerce.db")
        if db_tools.conn:
            print("✓ Database connection established")
            
            # Test a simple query
            categories = db_tools.get_categories(limit=5)
            if categories and len(categories) > 0:
                print(f"✓ Database query successful: Found {len(categories)} categories")
                print(f"  Sample categories: {[cat.get('category', 'Unknown') for cat in categories[:3]]}")
            else:
                print("⚠️ Database query returned no results")
        else:
            print("⚠️ Database connection failed - check database path")
        
        # Test server instantiation
        server = AgenticAIMCPServer()
        print("✓ MCP server instantiated successfully")
        
        if server.claude_interface:
            print("✓ Claude interface initialized")
        else:
            print("⚠️ Claude interface not available (check environment variables)")
        
        print("\n✅ Server test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Server test failed: {e}")
        return False

def show_usage_examples():
    """Show usage examples for the MCP server"""
    print("\n📖 Usage Examples:")
    print("="*50)
    
    print("\n1. Search for products:")
    print('   use_mcp_tool("agentic-ai", "search_products", {"query": "wireless headphones"})')
    
    print("\n2. Get product details:")
    print('   use_mcp_tool("agentic-ai", "get_product_details", {"product_id": "B08N5WRWNW"})')
    
    print("\n3. Find products in price range:")
    print('   use_mcp_tool("agentic-ai", "get_price_range", {"min_price": 100, "max_price": 500})')
    
    print("\n4. Get top rated products:")
    print('   use_mcp_tool("agentic-ai", "get_top_rated", {"category": "Electronics", "limit": 5})')
    
    print("\n5. Chat with AI agent:")
    print('   use_mcp_tool("agentic-ai", "agentic_chat", {"message": "Find me the best laptop under $1000"})')
    
    print("\n6. Get recommendations:")
    print('   use_mcp_tool("agentic-ai", "get_recommendations", {"product_id": "B08N5WRWNW", "limit": 3})')

def main():
    """Main setup function"""
    print("="*60)
    print("AGENTIC AI MCP SERVER SETUP")
    print("Same functional capabilities as agentic_ai.py")
    print("="*60)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_server()
        return
    
    if len(sys.argv) > 1 and sys.argv[1] == "--examples":
        show_usage_examples()
        return
    
    # Setup the server
    if setup_agentic_mcp_server():
        print("\n Setup completed successfully!")
        print("\n  Important Notes:")
        print("  1. Restart your IDE/editor to load the new MCP server")
        print("  2. Check that the database exists at: ../data/ecommerce.db else run prerequicite folder")
        print("  3. Verify credentials are correctly set in environment as these are removed")
        
        # Test the server
        if input("\nWould you like to run a quick test? (y/n): ").lower() == 'y':
            test_server()
        
        # Show examples
        if input("\nWould you like to see usage examples? (y/n): ").lower() == 'y':
            show_usage_examples()
    
    else:
        print("\n Setup failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
