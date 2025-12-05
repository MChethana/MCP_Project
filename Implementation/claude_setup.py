"""
Clean Anthropic Model Example
====================================

A simple, well-structured example for calling Claude models through Gen AI Hub.
Uses your existing credentials without requiring a direct Anthropic API key.

Features:
- Clean client setup and authentication
- Simple chat functions
- System prompt support
- Conversation history management
- Multiple Claude model support
- Error handling and retry logic
- Comprehensive examples and tests

"""

import os
import json
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from gen_ai_hub.proxy import get_proxy_client
from gen_ai_hub.orchestration.models.config import OrchestrationConfig
from gen_ai_hub.orchestration.models.llm import LLM
from gen_ai_hub.orchestration.models.message import SystemMessage, UserMessage
from gen_ai_hub.orchestration.models.template import Template, TemplateValue
from gen_ai_hub.orchestration.service import OrchestrationService

# ============ CONFIGURATION ============

@dataclass
class SyConfig:
    """Configuration class for SY Config Gen AI Hub"""
    auth_url: str
    client_id: str
    client_secret: str
    base_url: str
    resource_group: str = "default"
    
    @classmethod
    def from_env(cls) -> 'SyConfig':
        """Load configuration from environment variables"""
        return cls(
            auth_url=os.getenv("AICORE_AUTH_URL", ""),
            client_id=os.getenv("AICORE_CLIENT_ID", ""),
            client_secret=os.getenv("AICORE_CLIENT_SECRET", ""),
            base_url=os.getenv("AICORE_BASE_URL", ""),
            resource_group=os.getenv("AICORE_RESOURCE_GROUP", "default")
        )
    
    def validate(self) -> bool:
        """Validate that all required configuration is present"""
        required_fields = ['auth_url', 'client_id', 'client_secret', 'base_url']
        for field in required_fields:
            if not getattr(self, field):
                print(f"❌ Missing required configuration: {field.upper()}")
                return False
        return True
    
    def set_env_vars(self):
        """Set environment variables for the SAP BTP client"""
        env_vars = {
            "AICORE_AUTH_URL": self.auth_url,
            "AICORE_CLIENT_ID": self.client_id,
            "AICORE_CLIENT_SECRET": self.client_secret,
            "AICORE_BASE_URL": self.base_url,
            "AICORE_RESOURCE_GROUP": self.resource_group
        }
        os.environ.update(env_vars)
        print("✓ Environment variables set for SAP BTP")

# ============ CLAUDE MODEL INTERFACE ============

class ClaudeModelInterface:
    """
    Clean interface for calling Claude models through Gen AI Hub
    
    Supported Models:
    - anthropic--claude-3-5-sonnet
    - anthropic--claude-3-sonnet
    - anthropic--claude-3-haiku
    - anthropic--claude-3-opus (if available)
    """
    
    # Available Claude models  (based on the actual available models)
    AVAILABLE_MODELS = {
        "claude-3-haiku": "anthropic--claude-3-haiku",
        "claude-3-opus": "anthropic--claude-3-opus", 
        "claude-3.5-sonnet": "anthropic--claude-3.5-sonnet",
        "claude-3.7-sonnet": "anthropic--claude-3.7-sonnet",
        "claude-4-sonnet": "anthropic--claude-4-sonnet",
        "claude-4-opus": "anthropic--claude-4-opus",
        "claude-4.5-sonnet": "anthropic--claude-4.5-sonnet"
    }
    
    def __init__(self, config: SyConfig):
        """Initialize the Claude interface with System configuration"""
        self.config = config
        self.proxy_client = None
        self.orchestration_service = None
        self.conversation_history = []
        
        # Set up the client
        self._setup_client()
    
    def _setup_client(self):
        """Set up the proxy client and orchestration service"""
        try:
            # Set environment variables
            self.config.set_env_vars()
            
            # Initialize proxy client
            print("🔄 Initializing SAP BTP proxy client...")
            self.proxy_client = get_proxy_client()
            print("✓ Proxy client initialized successfully")
            
            # Set up orchestration service
            self._setup_orchestration()
            
        except Exception as e:
            print(f"❌ Failed to initialize SAP BTP client: {e}")
            raise
    
    def _setup_orchestration(self):
        """Set up the orchestration service for advanced features"""
        try:
            from ai_core_sdk.ai_core_v2_client import AICoreV2Client
            from ai_api_client_sdk.models.status import Status
            
            # Get or create deployment
            deployments = self.proxy_client.ai_core_client.deployment.query(
                scenario_id="orchestration",
                executable_ids=["orchestration"],
                status=Status.RUNNING
            )
            
            if deployments.count > 0:
                deployment = sorted(deployments.resources, key=lambda x: x.start_time)[0]
                print("✓ Using existing orchestration deployment")
            else:
                print("ℹ️  No active orchestration deployment found")
                deployment = None
            
            if deployment:
                self.orchestration_service = OrchestrationService(
                    api_url=deployment.deployment_url,
                    proxy_client=self.proxy_client
                )
                print("✓ Orchestration service initialized")
            
        except Exception as e:
            print(f"⚠️  Orchestration service setup failed: {e}")
            print("   (Simple chat functions will still work)")
    
    def simple_chat(self, 
                   message: str, 
                   model: str = "claude-3-haiku",
                   temperature: float = 0.7,
                   max_tokens: int = 1000,
                   system_prompt: Optional[str] = None) -> str:
        """
        Simple chat function for one-off messages
        
        Args:
            message: User message to send to Claude
            model: Claude model to use (see AVAILABLE_MODELS)
            temperature: Response creativity (0.0-1.0)
            max_tokens: Maximum response length
            system_prompt: Optional system prompt to set Claude's behavior
            
        Returns:
            Claude's response as string
        """
        try:
            # Get the full model name
            if model not in self.AVAILABLE_MODELS:
                available = ", ".join(self.AVAILABLE_MODELS.keys())
                raise ValueError(f"Model '{model}' not available. Choose from: {available}")
            
            full_model_name = self.AVAILABLE_MODELS[model]
            
            # Build messages
            messages = []
            if system_prompt:
                messages.append(SystemMessage(system_prompt))
            messages.append(UserMessage(message))
            
            # Create configuration
            config = OrchestrationConfig(
                llm=LLM(
                    name=full_model_name,
                    parameters={
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    }
                ),
                template=Template(messages=messages)
            )
            
            # Make the request
            if self.orchestration_service:
                print(f"🤖 Calling Claude ({model})...")
                response = self.orchestration_service.run(config=config)
                result = response.module_results.llm.choices[0].message.content
            else:
                # Fallback method if orchestration service not available
                print("🤖 Using fallback method...")
                result = self._fallback_chat(message, full_model_name, temperature)
            
            print("✓ Response received")
            return result.strip()
            
        except Exception as e:
            print(f"❌ Chat failed: {e}")
            return f"Error: {str(e)}"
    
    def _fallback_chat(self, message: str, model: str, temperature: float) -> str:
        """Fallback chat method using LangChain integration"""
        try:
            from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
            
            chat_llm = ChatOpenAI(
                proxy_model_name=model,
                proxy_client=self.proxy_client,
                temperature=temperature
            )
            
            response = chat_llm.invoke(message)
            return response.content
            
        except Exception as e:
            return f"Fallback method also failed: {str(e)}"
    
    def conversation_chat(self, 
                         message: str,
                         model: str = "claude-3-haiku",
                         system_prompt: Optional[str] = None,
                         reset_history: bool = False) -> str:
        """
        Chat function with conversation history
        
        Args:
            message: User message
            model: Claude model to use
            system_prompt: System prompt (only used on first message or after reset)
            reset_history: Whether to clear conversation history
            
        Returns:
            Claude's response
        """
        if reset_history:
            self.conversation_history = []
        
        # Add system prompt if this is the first message and system prompt provided
        if not self.conversation_history and system_prompt:
            self.conversation_history.append({"role": "system", "content": system_prompt})
        
        # Add user message
        self.conversation_history.append({"role": "user", "content": message})
        
        # Build conversation context
        conversation_text = self._build_conversation_context()
        
        # Get response
        response = self.simple_chat(
            message=conversation_text,
            model=model,
            system_prompt=None  # Already included in conversation context
        )
        
        # Add assistant response to history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def _build_conversation_context(self) -> str:
        """Build conversation context from history"""
        context_parts = []
        
        for entry in self.conversation_history:
            role = entry["role"]
            content = entry["content"]
            
            if role == "system":
                context_parts.append(f"System: {content}")
            elif role == "user":
                context_parts.append(f"Human: {content}")
            elif role == "assistant":
                context_parts.append(f"Assistant: {content}")
        
        return "\n\n".join(context_parts)
    
    def reset_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("✓ Conversation history cleared")
    
    def get_available_models(self) -> List[str]:
        """Get list of available Claude models"""
        return list(self.AVAILABLE_MODELS.keys())

# ============ EXAMPLE FUNCTIONS ============

def basic_examples(claude: ClaudeModelInterface):
    """Run basic Claude examples"""
    print("\n" + "="*60)
    print("BASIC EXAMPLES")
    print("="*60)
    
    # Simple question
    print("\n1. Simple Question:")
    response = claude.simple_chat("What is the capital of France?")
    print(f"Claude: {response}")
    
    # Creative task
    print("\n2. Creative Task:")
    response = claude.simple_chat(
        "Write a haiku about artificial intelligence",
        temperature=0.9
    )
    print(f"Claude: {response}")
    
    # Technical question
    print("\n3. Technical Question:")
    response = claude.simple_chat(
        "Explain quantum computing in simple terms",
        max_tokens=500
    )
    print(f"Claude: {response}")

def system_prompt_examples(claude: ClaudeModelInterface):
    """Examples using system prompts"""
    print("\n" + "="*60)
    print("SYSTEM PROMPT EXAMPLES")
    print("="*60)
    
    # Helpful assistant
    print("\n1. Helpful Assistant:")
    response = claude.simple_chat(
        "How do I make a good cup of coffee?",
        system_prompt="You are a helpful coffee expert who gives detailed brewing advice."
    )
    print(f"Coffee Expert: {response}")
    
    # Code reviewer
    print("\n2. Code Reviewer:")
    code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
    """
    response = claude.simple_chat(
        f"Please review this Python code:\n{code}",
        system_prompt="You are an expert Python code reviewer. Provide constructive feedback on code quality, performance, and best practices."
    )
    print(f"Code Reviewer: {response}")

def conversation_examples(claude: ClaudeModelInterface):
    """Examples with conversation history"""
    print("\n" + "="*60)
    print("CONVERSATION EXAMPLES")
    print("="*60)
    
    # Start a conversation about cooking
    print("\n1. Multi-turn Cooking Conversation:")
    
    response1 = claude.conversation_chat(
        "I want to learn how to cook pasta",
        system_prompt="You are a friendly cooking instructor",
        reset_history=True
    )
    print(f"Chef: {response1}")
    
    response2 = claude.conversation_chat(
        "What type of pasta should I start with?"
    )
    print(f"Chef: {response2}")
    
    response3 = claude.conversation_chat(
        "How long should I cook spaghetti?"
    )
    print(f"Chef: {response3}")

def model_comparison_examples(claude: ClaudeModelInterface):
    """Compare different Claude models"""
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    question = "Explain the concept of recursion in programming"
    
    for model in claude.get_available_models():
        print(f"\n{model.upper()}:")
        try:
            response = claude.simple_chat(question, model=model, max_tokens=300)
            print(f"{response}")
        except Exception as e:
            print(f" Error with {model}: {e}")

# ============ TEST & DEMO FUNCTIONS ============

def test_connection(claude: ClaudeModelInterface):
    """Test the System connection"""
    print("\n" + "="*60)
    print("CONNECTION TEST")
    print("="*60)
    
    try:
        response = claude.simple_chat("Say 'Hello, IT's working!' if you receive this message.")
        print(f" Connection successful!")
        print(f"Response: {response}")
        return True
    except Exception as e:
        print(f" Connection failed: {e}")
        return False

def interactive_chat(claude: ClaudeModelInterface):
    """Interactive chat session"""
    print("\n" + "="*60)
    print("INTERACTIVE CHAT")
    print("Type 'quit' to exit, 'reset' to clear history, 'models' to see available models")
    print("="*60)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            elif user_input.lower() == 'reset':
                claude.reset_conversation()
                continue
            elif user_input.lower() == 'models':
                models = claude.get_available_models()
                print(f"Available models: {', '.join(models)}")
                continue
            elif not user_input:
                continue
            
            response = claude.conversation_chat(user_input)
            print(f"Claude: {response}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f" Error: {e}")


