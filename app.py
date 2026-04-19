import json
import asyncio
import os
import requests
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path
from dataclasses import dataclass
import argparse
import sys
import re

try:
    from fastmcp import FastMCP
    FAST_MCP_AVAILABLE = True
except ImportError:
    FAST_MCP_AVAILABLE = False
    print(" FastMCP not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fastmcp"])
    from fastmcp import FastMCP



MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    print(" MISTRAL_API_KEY environment variable not set")
    print("   Set it using: $env:MISTRAL_API_KEY='your-key-here' (Windows)")
    print("   or: export MISTRAL_API_KEY='your-key-here' (Linux/Mac)")
    print("   Continuing without LLM capabilities...")

MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"



def call_mistral(prompt: str, system_prompt: str = None, max_tokens: int = 500) -> str:
    """
    Call Mistral AI API for LLM capabilities.
    
    Args:
        prompt: User prompt
        system_prompt: Optional system instruction
        max_tokens: Maximum tokens in response
        
    Returns:
        LLM response text
    """
    if not MISTRAL_API_KEY:
        return "Mistral API key not configured. Please set MISTRAL_API_KEY environment variable."
    
    messages = []
    
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })
    
    messages.append({
        "role": "user",
        "content": prompt
    })
    
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MISTRAL_MODEL,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": max_tokens
    }
    
    try:
        response = requests.post(MISTRAL_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"❌ Mistral API error: {e}")
        return f"LLM error: {str(e)}"



@dataclass
class Customer:
    customer_id: str
    name: str
    email: str
    phone: str
    tier: str
    member_since: str
    total_orders: int
    total_spent: float
    address: Dict
    notes: str

@dataclass
class Order:
    order_id: str
    customer_id: str
    product_id: str
    quantity: int
    amount: float
    status: str
    order_date: str
    delivery_date: Optional[str]
    return_deadline: Optional[str]
    refund_status: Optional[str]
    notes: str

@dataclass
class Product:
    product_id: str
    name: str
    category: str
    price: float
    warranty_months: int
    return_window_days: int
    returnable: bool
    notes: str

@dataclass
class Ticket:
    ticket_id: str
    customer_email: str
    subject: str
    body: str
    source: str
    created_at: str
    tier: int
    expected_action: str



class DataLoader:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.data_dir = Path("data")
        self.customers: Dict[str, Customer] = {}
        self.orders: Dict[str, Order] = {}
        self.products: Dict[str, Product] = {}
        self.tickets: List[Ticket] = []
        self.knowledge_base: str = ""
        self._load_all()
    
    def _load_all(self):
     
        self._load_customers()
        self._load_orders()
        self._load_products()
        self._load_tickets()
        self._load_knowledge_base()
        print(f"✅ DataLoader: Loaded {len(self.customers)} customers, {len(self.orders)} orders, {len(self.products)} products, {len(self.tickets)} tickets")
    
    def _load_customers(self):
        customer_path = self.data_dir / "customers.json"
        if customer_path.exists():
            with open(customer_path, "r", encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    customer = Customer(**item)
                    self.customers[customer.email] = customer
                    self.customers[customer.customer_id] = customer
    
    def _load_orders(self):
        orders_path = self.data_dir / "orders.json"
        if orders_path.exists():
            with open(orders_path, "r", encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    order = Order(**item)
                    self.orders[order.order_id] = order
    
    def _load_products(self):
        products_path = self.data_dir / "products.json"
        if products_path.exists():
            with open(products_path, "r", encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    product = Product(**item)
                    self.products[product.product_id] = product
    
    def _load_tickets(self):
        tickets_path = self.data_dir / "tickets.json"
        if tickets_path.exists():
            with open(tickets_path, "r", encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    ticket = Ticket(**item)
                    self.tickets.append(ticket)
    
    def _load_knowledge_base(self):
        kb_path = self.data_dir / "knowledge-base.md"
        if kb_path.exists():
            with open(kb_path, "r", encoding='utf-8') as f:
                self.knowledge_base = f.read()
    
    def get_customer_by_email(self, email: str) -> Optional[Customer]:
        return self.customers.get(email)
    
    def get_customer_by_id(self, customer_id: str) -> Optional[Customer]:
        return self.customers.get(customer_id)
    
    def get_order(self, order_id: str) -> Optional[Order]:
        return self.orders.get(order_id)
    
    def get_orders_by_customer(self, customer_id: str) -> List[Order]:
        return [o for o in self.orders.values() if o.customer_id == customer_id]
    
    def get_product(self, product_id: str) -> Optional[Product]:
        return self.products.get(product_id)
    
    def get_tickets(self) -> List[Ticket]:
        return self.tickets


mcp = FastMCP("ShopWave Support Agent")

# Global data loader
data_loader = DataLoader()
tool_failure_counter = 0
MAX_FAILURES = 3



@mcp.tool()
def get_customer(email: str) -> Dict[str, Any]:
    """
    Look up customer information by email address.
    
    Args:
        email: Customer's email address
    """
    global tool_failure_counter
    
    # Simulate occasional timeout for testing resilience
    if tool_failure_counter < MAX_FAILURES:
        tool_failure_counter += 1
        if tool_failure_counter == 1:
            raise TimeoutError("Simulated timeout - retry mechanism will handle this")
    
    customer = data_loader.get_customer_by_email(email)
    
    if not customer:
        return {
            "success": False,
            "found": False,
            "error": f"Customer not found for email: {email}"
        }
    
    return {
        "success": True,
        "found": True,
        "customer_id": customer.customer_id,
        "name": customer.name,
        "email": customer.email,
        "phone": customer.phone,
        "tier": customer.tier,
        "member_since": customer.member_since,
        "total_orders": customer.total_orders,
        "total_spent": customer.total_spent,
        "notes": customer.notes
    }



@mcp.tool()
def get_order(order_id: str) -> Dict[str, Any]:
    """
    Look up order details by order ID.
    
    Args:
        order_id: Order ID (format: ORD-XXXX)
    """
    order = data_loader.get_order(order_id)
    
    if not order:
        return {
            "success": False,
            "found": False,
            "error": f"Order not found: {order_id}"
        }
    
    product = data_loader.get_product(order.product_id)
    
    return {
        "success": True,
        "found": True,
        "order_id": order.order_id,
        "customer_id": order.customer_id,
        "product_id": order.product_id,
        "product_name": product.name if product else "Unknown",
        "product_category": product.category if product else "Unknown",
        "quantity": order.quantity,
        "amount": order.amount,
        "status": order.status,
        "order_date": order.order_date,
        "delivery_date": order.delivery_date,
        "return_deadline": order.return_deadline,
        "refund_status": order.refund_status,
        "notes": order.notes
    }


@mcp.tool()
def get_customer_orders(customer_id: str) -> Dict[str, Any]:
    """
    Get all orders for a specific customer.
    
    Args:
        customer_id: Customer ID (format: CXXX)
    """
    orders = data_loader.get_orders_by_customer(customer_id)
    
    return {
        "success": True,
        "customer_id": customer_id,
        "total_orders": len(orders),
        "orders": [
            {
                "order_id": o.order_id,
                "amount": o.amount,
                "status": o.status,
                "order_date": o.order_date,
                "product_id": o.product_id
            }
            for o in orders
        ]
    }



@mcp.tool()
def get_product(product_id: str) -> Dict[str, Any]:
    
    product = data_loader.get_product(product_id)
    
    if not product:
        return {
            "success": False,
            "found": False,
            "error": f"Product not found: {product_id}"
        }
    
    return {
        "success": True,
        "found": True,
        "product_id": product.product_id,
        "name": product.name,
        "category": product.category,
        "price": product.price,
        "warranty_months": product.warranty_months,
        "return_window_days": product.return_window_days,
        "returnable": product.returnable,
        "notes": product.notes
    }



@mcp.tool()
def check_refund_eligibility(order_id: str, customer_tier: str = "standard") -> Dict[str, Any]:
    
    order = data_loader.get_order(order_id)
    
    if not order:
        return {
            "success": False,
            "eligible": False,
            "reason": f"Order {order_id} not found"
        }
    
    if order.refund_status == "refunded":
        return {
            "success": True,
            "eligible": False,
            "reason": "Refund already processed for this order"
        }
    
    product = data_loader.get_product(order.product_id)
    if not product:
        return {
            "success": False,
            "eligible": False,
            "reason": "Product information not available"
        }
    
    if not product.returnable:
        return {
            "success": True,
            "eligible": False,
            "reason": f"{product.name} is marked as non-returnable per policy"
        }
    
    # Check return deadline
    if order.return_deadline:
        deadline = datetime.strptime(order.return_deadline, "%Y-%m-%d")
        today = datetime.now()
        
        if today > deadline:       # VIP - eligible even after deadline
            if customer_tier == "vip":
                return {
                    "success": True,
                    "eligible": True,
                    "reason": f"Return window expired but VIP exception applies",
                    "requires_approval": False
                }
            # Premium - needs approval
            elif customer_tier == "premium":
                return {
                    "success": True,
                    "eligible": True,
                    "reason": f"Return window expired - premium customer, use judgment",
                    "requires_approval": True
                }
            else:
                return {
                    "success": True,
                    "eligible": False,
                    "reason": f"Return window expired on {order.return_deadline}"
                }
    
    return {
        "success": True,
        "eligible": True,
        "reason": "Within return window",
        "requires_approval": False,
        "return_deadline": order.return_deadline
    }


@mcp.tool()
def search_knowledge_base(query: str) -> Dict[str, Any]:
    """
    Search the ShopWave knowledge base for policies, return guidelines, and FAQs.
    
    Args:
        query: Search query (e.g., "return policy electronics")
    """
    kb = data_loader.knowledge_base
    
    if not kb:
        return {
            "success": True,
            "results": "Knowledge base not loaded.",
            "sections": []
        }
    
    query_lower = query.lower()
    relevant_sections = []
    
    sections = kb.split("\n## ")
    
    keywords = query_lower.split()
    
    for section in sections:
        section_lower = section.lower()
        score = sum(1 for kw in keywords if kw in section_lower)
        if score > 0:
            truncated = section[:600] + "..." if len(section) > 600 else section
            relevant_sections.append({
                "content": truncated,
                "relevance_score": score
            })
    
    relevant_sections.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    results = "\n\n---\n\n".join([s["content"] for s in relevant_sections[:3]])
    
    if not results:
        results = "No specific policy found. Please rephrase your query."
    
    return {
        "success": True,
        "results": results,
        "sections_count": len(relevant_sections),
        "query": query
    }

## part -2 customer ko replay ka tool banayenge jisme hum customer ko reply karenge aur uska history bhi rakhenge taki agar future me koi issue aaye to uska reference ke liye use kar sake

reply_history: List[Dict] = []

@mcp.tool()
def send_reply(ticket_id: str, message: str) -> Dict[str, Any]:
    """
    Send a reply message to the customer.
    
    Args:
        ticket_id: Ticket ID being handled
        message: Reply message to send to customer
    """
    if len(message) < 10:
        return {
            "success": False,
            "error": "Message too short (minimum 10 characters)"
        }
    
    reply_record = {
        "ticket_id": ticket_id,
        "message": message[:500],
        "timestamp": datetime.now().isoformat(),
        "message_length": len(message)
    }
    reply_history.append(reply_record)
    
    return {
        "success": True,
        "ticket_id": ticket_id,
        "message_preview": message[:200] + "..." if len(message) > 200 else message,
        "timestamp": datetime.now().isoformat(),
        "audit_id": len(reply_history)
    }
# for cancel order and refund we will create two tools one for cancel order and one for refund initiation hoga

@mcp.tool()
def cancel_order(order_id: str) -> Dict[str, Any]:
    """
    Cancel an order if it's still in 'processing' status.
    
    Args:
        order_id: Order ID to cancel
    """
    order = data_loader.get_order(order_id)
    
    if not order:
        return {
            "success": False,
            "error": f"Order {order_id} not found"
        }
    
    if order.status == "processing":
        return {
            "success": True,
            "order_id": order_id,
            "status": "cancelled",
            "message": f"Order {order_id} has been cancelled successfully",
            "refund_initiated": True,
            "refund_amount": order.amount,
            "next_steps": "Refund will be processed within 5-7 business days"
        }
    elif order.status == "shipped":
        return {
            "success": False,
            "order_id": order_id,
            "status": order.status,
            "message": f"Order {order_id} has already shipped and cannot be cancelled",
            "alternative": "Please wait for delivery and initiate a return"
        }
    else:
        return {
            "success": False,
            "order_id": order_id,
            "status": order.status,
            "message": f"Order {order_id} status is '{order.status}' and cannot be cancelled"
        }

#intiate funding ka  tool

@mcp.tool()
def initiate_refund(order_id: str, amount: float, reason: str) -> Dict[str, Any]:
    """
    Initiate a refund for an order.
    
    Args:
        order_id: Order ID to refund
        amount: Refund amount
        reason: Reason for refund
    """
    order = data_loader.get_order(order_id)
    
    if not order:
        return {
            "success": False,
            "error": f"Order {order_id} not found"
        }
    
    if order.refund_status == "refunded":
        return {
            "success": False,
            "error": "Refund already processed for this order",
            "existing_refund": True
        }
    
    refund_amount = min(amount, order.amount)
    
    refund_id = f"REF-{order_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    return {
        "success": True,
        "order_id": order_id,
        "refund_id": refund_id,
        "refund_amount": refund_amount,
        "original_amount": order.amount,
        "reason": reason,
        "processing_days": "5-7 business days",
        "status": "initiated",
        "timestamp": datetime.now().isoformat()
    }



@mcp.tool()
def escalate_ticket(ticket_id: str, reason: str, summary: Dict[str, Any]) -> Dict[str, Any]:
   
    required_fields = ["customer", "issue", "attempted_resolution"]
    for field in required_fields:
        if field not in summary:
            summary[field] = "Not provided"
    
    priority = summary.get("priority", "medium")
    
    return {
        "success": True,
        "ticket_id": ticket_id,
        "escalation_reason": reason,
        "summary": summary,
        "assigned_to": "human_agent_queue",
        "priority": priority,
        "estimated_response_time": "4 hours" if priority == "urgent" else "24 hours",
        "escalated_at": datetime.now().isoformat()
    }

# ============================================================================
# TOOL 11: Classify Ticket (Using Mistral AI)
# ============================================================================

@mcp.tool()
def classify_ticket(subject: str, body: str) -> Dict[str, Any]:
    """
    Classify a support ticket using Mistral AI.
    
    Args:
        subject: Ticket subject line
        body: Ticket body text
    """
    prompt = f"""Classify this support ticket and return ONLY valid JSON.

Subject: {subject}
Body: {body}

Return JSON with these exact fields:
- category: one of [return_refund, damaged_item, wrong_item, cancellation, shipping, general_inquiry, warranty]
- priority: one of [low, medium, high, urgent]
- resolvable: true or false
- confidence: number between 0 and 1

Example: {{"category": "return_refund", "priority": "medium", "resolvable": true, "confidence": 0.85}}"""

    system_prompt = "You are a support ticket classifier. Return ONLY valid JSON, no other text."
    
    response = call_mistral(prompt, system_prompt, max_tokens=200)
    
    try:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return {
                "success": True,
                "classification": result
            }
    except:
        pass
    
    body_lower = body.lower()
    if any(w in body_lower for w in ["refund", "return"]):
        return {"success": True, "classification": {"category": "return_refund", "priority": "medium", "resolvable": True, "confidence": 0.8}}
    elif "cancel" in body_lower:
        return {"success": True, "classification": {"category": "cancellation", "priority": "high", "resolvable": True, "confidence": 0.9}}
    elif any(w in body_lower for w in ["damaged", "broken", "cracked"]):
        return {"success": True, "classification": {"category": "damaged_item", "priority": "high", "resolvable": True, "confidence": 0.85}}
    else:
        return {"success": True, "classification": {"category": "general_inquiry", "priority": "low", "resolvable": True, "confidence": 0.7}}

# ============================================================================
# TOOL 12: Generate Response (Using Mistral AI)
# ============================================================================

@mcp.tool()
def generate_response(customer_name: str, issue_type: str, order_info: Dict, policy_info: str) -> Dict[str, Any]:
    """
    Generate a professional response to a customer using Mistral AI.
    
    Args:
        customer_name: Customer's name
        issue_type: Type of issue (refund, cancel, info, etc.)
        order_info: Order details dictionary
        policy_info: Relevant policy information
    """
    prompt = f"""Generate a professional, empathetic customer support response.

Customer: {customer_name}
Issue: {issue_type}
Order: {json.dumps(order_info, indent=2)}
Policy: {policy_info}

Requirements:
- Be polite and helpful
- Address the customer by name
- Explain the resolution clearly
- Keep it concise (max 150 words)
- Include next steps if applicable

Response:"""
    
    system_prompt = "You are a professional customer support agent for ShopWave. Be helpful, clear, and empathetic."
    
    response = call_mistral(prompt, system_prompt, max_tokens=400)
    
    return {
        "success": True,
        "response": response.strip(),
        "tone": "professional_empathetic"
    }


@mcp.tool()
def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyze customer sentiment from their message using Mistral AI.
    
    Args:
        text: Customer message text
    """
    prompt = f"""Analyze the sentiment of this customer message and return ONLY valid JSON.

Message: {text}

Return JSON: {{"sentiment": "positive/negative/neutral", "urgency": "low/medium/high/urgent", "escalation_needed": true/false}}"""

    response = call_mistral(prompt, max_tokens=100)
    
    try:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return {"success": True, "analysis": result}
    except:
        pass
    
    return {"success": True, "analysis": {"sentiment": "neutral", "urgency": "medium", "escalation_needed": False}}

# ============================================================================
# TOOL 14: Health Check
# ============================================================================

@mcp.tool()
def health_check() -> Dict[str, Any]:
    """Check if the MCP server and all data sources are healthy."""
    return {
        "success": True,
        "status": "healthy",
        "server": "ShopWave MCP Server",
        "version": "2.0.0",
        "llm": "Mistral AI" if MISTRAL_API_KEY else "Not configured",
        "mistral_model": MISTRAL_MODEL if MISTRAL_API_KEY else None,
        "data_loaded": {
            "customers": len(data_loader.customers),
            "orders": len(data_loader.orders),
            "products": len(data_loader.products),
            "tickets": len(data_loader.tickets),
            "knowledge_base_size": len(data_loader.knowledge_base)
        },
        "timestamp": datetime.now().isoformat()
    }



@mcp.tool()
def process_ticket(ticket_id: str) -> Dict[str, Any]:
    """
    Process a ticket end-to-end using all tools and Mistral AI.
    
    Args:
        ticket_id: Ticket ID to process
    """
    # Find the ticket
    ticket = None
    for t in data_loader.get_tickets():
        if t.ticket_id == ticket_id:
            ticket = t
            break
    
    if not ticket:
        return {"success": False, "error": f"Ticket {ticket_id} not found"}
    
    # Step 1: Classify the ticket
    classification = classify_ticket(ticket.subject, ticket.body)
    
    # Step 2: Get customer info
    customer = get_customer(ticket.customer_email)
    
    if not customer.get("found"):
        return {
            "success": False,
            "ticket_id": ticket_id,
            "status": "needs_info",
            "message": f"Customer not found for email: {ticket.customer_email}"
        }
    
    order_id_match = re.search(r'ORD-\d{4}', ticket.body + " " + ticket.subject)
    
    if not order_id_match:
        return {
            "success": False,
            "ticket_id": ticket_id,
            "status": "needs_info",
            "message": "No order ID found. Please provide order ID.",
            "customer": customer
        }
    
    order_id = order_id_match.group(0)
    
    order = get_order(order_id)
    
    if not order.get("found"):
        return {
            "success": False,
            "ticket_id": ticket_id,
            "status": "needs_info",
            "message": f"Order {order_id} not found",
            "customer": customer
        }
    
    category = classification.get("classification", {}).get("category", "general_inquiry")
    
    if category == "cancellation":
        cancel_result = cancel_order(order_id)
        if cancel_result.get("success"):
            response = generate_response(
                customer.get("name", "Customer"),
                "order_cancellation",
                order,
                "Order cancellation policy"
            )
            send_reply(ticket_id, response.get("response", cancel_result.get("message", "Order cancelled")))
            return {
                "success": True,
                "ticket_id": ticket_id,
                "status": "resolved",
                "action": "cancelled",
                "result": cancel_result
            }
    
    elif category in ["return_refund", "damaged_item"]:
        eligibility = check_refund_eligibility(order_id, customer.get("tier", "standard"))
        if eligibility.get("eligible"):
            refund_result = initiate_refund(order_id, order.get("amount", 0), f"{category} request")
            if refund_result.get("success"):
                response = generate_response(
                    customer.get("name", "Customer"),
                    "refund",
                    order,
                    "Refund policy: " + eligibility.get("reason", "")
                )
                send_reply(ticket_id, response.get("response", f"Refund of ${refund_result['refund_amount']} initiated"))
                return {
                    "success": True,
                    "ticket_id": ticket_id,
                    "status": "resolved",
                    "action": "refunded",
                    "result": refund_result
                }
    
    kb_results = search_knowledge_base(f"{ticket.subject} {ticket.body}")
    response = generate_response(
        customer.get("name", "Customer"),
        category,
        order,
        kb_results.get("results", "General policy information")
    )
    send_reply(ticket_id, response.get("response", "Thank you for contacting support. We'll assist you shortly."))
    
    return {
        "success": True,
        "ticket_id": ticket_id,
        "status": "processed",
        "action": "info_provided",
        "classification": classification,
        "customer": customer,
        "order": order
    }



def main():
    parser = argparse.ArgumentParser(description="ShopWave MCP Server")
    parser.add_argument("--sse", action="store_true", help="Run with SSE transport (HTTP)")
    parser.add_argument("--port", type=int, default=8000, help="Port for SSE server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for SSE server")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🛍️  ShopWave MCP Server with Mistral AI")
    print("=" * 60)
    print(f"FastMCP: {'installed' if FAST_MCP_AVAILABLE else 'fallback mode'}")
    print(f"Mistral AI: {'✅ Configured' if MISTRAL_API_KEY else '❌ Not configured'}")
    if MISTRAL_API_KEY:
        print(f"Model: {MISTRAL_MODEL}")
    print(f"Data Directory: data/")
    print(f"Tools Available: 15")
    print("=" * 60)
    
    # List all tools
    print("\n📦 Available Tools:")
    tools = [
        ("get_customer", "Look up customer by email"),
        ("get_order", "Get order details"),
        ("get_customer_orders", "Get all customer orders"),
        ("get_product", "Get product information"),
        ("check_refund_eligibility", "Check if refund is possible"),
        ("search_knowledge_base", "Search policies and FAQs"),
        ("send_reply", "Send reply to customer"),
        ("cancel_order", "Cancel an order"),
        ("initiate_refund", "Process a refund"),
        ("escalate_ticket", "Escalate to human agent"),
        ("classify_ticket", "Classify ticket using Mistral AI"),
        ("generate_response", "Generate response using Mistral AI"),
        ("analyze_sentiment", "Analyze sentiment using Mistral AI"),
        ("health_check", "Check server health"),
        ("process_ticket", "Process ticket end-to-end")
    ]
    
    for name, desc in tools:
        print(f"  🔧 {name}: {desc}")
    
    print("\n" + "=" * 60)
    
    if args.sse:
        print(f"🚀 Starting MCP server with SSE on http://{args.host}:{args.port}")
        print(f"   SSE Endpoint: http://{args.host}:{args.port}/sse")
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        print("🚀 Starting MCP server with stdio transport")
        print("   (For use with Claude Desktop or other MCP clients)")
        mcp.run()

if __name__ == "__main__":
    main()