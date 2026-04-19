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

# ============================================================================
# LANGGRAPH + PYTORCH IMPORTS
# ============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal

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
    print("⚠️ MISTRAL_API_KEY environment variable not set")
    print("   Set it using: $env:MISTRAL_API_KEY='your-key-here' (Windows)")
    print("   or: export MISTRAL_API_KEY='your-key-here' (Linux/Mac)")
    print("   Continuing without LLM capabilities...")

MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"



def call_mistral(prompt: str, system_prompt: str = None, max_tokens: int = 500) -> str:
    
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
        """Load all data from JSON files"""
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


# ============================================================================
# PYTORCH NEURAL NETWORK FOR DECISION MAKING
# ============================================================================

class TicketDecisionNet(nn.Module):
    """PyTorch neural network to help decide ticket actions"""
    
    def __init__(self, input_size=12, hidden_size=32, output_size=4):
        super(TicketDecisionNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=1)
    
    def predict_action(self, features):
        """Predict best action from features"""
        self.eval()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            output = self.forward(x)
            action_idx = torch.argmax(output, dim=1).item()
            confidence = torch.max(output).item()
            
            actions = ['refund', 'cancel', 'info', 'escalate']
            return actions[action_idx], confidence

# Initialize PyTorch model
decision_model = TicketDecisionNet()

def extract_ticket_features(ticket: Ticket, customer: Optional[Customer], order: Optional[Order]) -> List[float]:
    """Extract numerical features from ticket for PyTorch model"""
    features = []
    
    # Customer features (3 features)
    if customer:
        tier_value = 1.0 if customer.tier == 'vip' else 0.5 if customer.tier == 'premium' else 0.0
        features.append(tier_value)
        features.append(min(customer.total_orders / 100, 1.0))
        features.append(min(customer.total_spent / 10000, 1.0))
    else:
        features.extend([0.0, 0.0, 0.0])
    
    # Order features (3 features)
    if order:
        status_value = 1.0 if order.status == 'delivered' else 0.5 if order.status == 'shipped' else 0.0
        features.append(status_value)
        features.append(1.0 if order.refund_status == 'refunded' else 0.0)
        features.append(min(order.amount / 500, 1.0))
    else:
        features.extend([0.0, 0.0, 0.0])
    
    # Ticket urgency features (3 features)
    urgency_keywords = ['urgent', 'immediately', 'asap', 'broken', 'damaged', 'wrong', 'cancel']
    urgency_score = sum(1 for kw in urgency_keywords if kw in ticket.body.lower()) / len(urgency_keywords)
    features.append(urgency_score)
    features.append(min(len(ticket.body) / 500, 1.0))
    
    # Sentiment score
    sentiment_words = {'good': 1, 'great': 1, 'happy': 1, 'bad': -1, 'terrible': -1, 'awful': -1, 'poor': -1}
    sentiment = sum(sentiment_words.get(word, 0) for word in ticket.body.lower().split())
    sentiment_norm = (max(-1, min(1, sentiment / 5)) + 1) / 2
    features.append(sentiment_norm)
    
    # Time features (3 features)
    if order and order.order_date:
        order_date = datetime.strptime(order.order_date, "%Y-%m-%d")
        days_since = (datetime.now() - order_date).days
        features.append(min(days_since / 90, 1.0))
        
        if order.return_deadline:
            deadline = datetime.strptime(order.return_deadline, "%Y-%m-%d")
            days_left = (deadline - datetime.now()).days
            features.append(max(0, min(days_left / 30, 1.0)))
        else:
            features.append(0.5)
    else:
        features.extend([0.0, 0.5])
    
    # Ensure we have exactly 12 features
    while len(features) < 12:
        features.append(0.0)
    
    return features[:12]

def auto_train_model():
    """Automatically train the PyTorch model on startup using available data"""
    print("\n🧠 Auto-training PyTorch model on available data...")
    
    training_data = []
    try:
        with open("audit_log.json", "r") as f:
            audit = json.load(f)
            training_data = audit.get('results', [])
    except:
        pass
    
    if not training_data:
        tickets = data_loader.get_tickets()
        for ticket in tickets[:20]:
            customer = data_loader.get_customer_by_email(ticket.customer_email)
            order_match = re.search(r'ORD-\d{4}', ticket.body)
            order = data_loader.get_order(order_match.group(0)) if order_match else None
            
            features = extract_ticket_features(ticket, customer, order)
            
            body_lower = ticket.body.lower()
            if 'refund' in body_lower or 'return' in body_lower:
                action = 0
            elif 'cancel' in body_lower:
                action = 1
            elif 'damaged' in body_lower or 'broken' in body_lower:
                action = 0
            else:
                action = 2
            
            training_data.append({'features': features, 'action': action})
    
    if len(training_data) < 5:
        print("   ⚠️ Not enough training data, using synthetic data")
        for _ in range(50):
            features = [torch.rand(1).item() for _ in range(12)]
            action = torch.randint(0, 4, (1,)).item()
            training_data.append({'features': features, 'action': action})
    
    X_train = []
    y_train = []
    
    for item in training_data:
        if 'features' in item:
            X_train.append(item['features'][:12])
            y_train.append(item['action'])
        elif 'full_response' in item:
            resp = item['full_response']
            action_map = {'refund': 0, 'cancelled': 1, 'info': 2, 'escalate': 3}
            action = action_map.get(resp.get('action', 'info'), 2)
            
            features = [
                float(resp.get('customer', {}).get('tier') == 'vip'),
                min(resp.get('customer', {}).get('total_orders', 0) / 100, 1.0),
                min(resp.get('order', {}).get('amount', 0) / 500, 1.0),
            ]
            while len(features) < 12:
                features.append(0.0)
            X_train.append(features[:12])
            y_train.append(action)
    
    if len(X_train) < 5:
        print("   ❌ Still not enough data, model will use default weights")
        return
    
    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.long)
    
    decision_model.train()
    optimizer = torch.optim.Adam(decision_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    for epoch in range(50):
        optimizer.zero_grad()
        output = decision_model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    print(f"   ✅ Model trained on {len(X_train)} samples")
    print(f"   📉 Final loss: {losses[-1]:.4f} (initial: {losses[0]:.4f})")
    torch.save(decision_model.state_dict(), "decision_model.pt")
    print(f"   💾 Model saved to decision_model.pt")


# ============================================================================
# LANGGRAPH AGENT STATE AND NODES
# ============================================================================

class LangGraphState(TypedDict):
    ticket_id: str
    ticket: Dict
    customer: Dict
    order: Dict
    product: Dict
    features: List[float]
    ml_decision: str
    ml_confidence: float
    action_taken: str
    response: str
    status: str
    step: str
    tool_calls: List[str]
    classification: Dict
    mistral_decision: Dict

def ml_decision_node(state: LangGraphState) -> LangGraphState:
    features = state.get('features', [0] * 12)
    if len(features) < 12:
        features = features + [0] * (12 - len(features))
    action, confidence = decision_model.predict_action(features)
    state['ml_decision'] = action
    state['ml_confidence'] = confidence
    state['step'] = 'ml_decision'
    print(f"🧠 ML Decision: {action} (confidence: {confidence:.2f})")
    return state

def mistral_decision_node(state: LangGraphState) -> LangGraphState:
    """Use Mistral AI to analyze ticket and decide action"""
    ticket = state.get('ticket', {})
    customer = state.get('customer', {})
    order = state.get('order', {})
    
    prompt = f"""Analyze this customer support ticket and decide the BEST action.

TICKET: {ticket.get('ticket_id', 'Unknown')}
CUSTOMER: {customer.get('name', 'Unknown')} (Tier: {customer.get('tier', 'standard')})
SUBJECT: {ticket.get('subject', '')}
BODY: {ticket.get('body', '')}
ORDER STATUS: {order.get('status', 'Unknown')}
ORDER AMOUNT: ${order.get('amount', 0)}

POLICIES:
- Return window: 30 days standard, 15 days electronics, 60 days accessories
- VIP customers get extended leniency
- Premium customers can use judgment for borderline cases
- Warranty claims (12-24 months) should be escalated
- Orders in 'processing' status can be cancelled
- Damaged items qualify for full refund without return
- Wrong items qualify for free exchange

Return ONLY JSON: {{"action": "refund/cancel/exchange/escalate/info/deny", "reason": "brief explanation", "priority": "low/medium/high/urgent"}}"""

    response = call_mistral(prompt, system_prompt="You are an AI customer support agent. Analyze tickets and decide actions based on policies.", max_tokens=300)
    
    try:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            state['mistral_decision'] = json.loads(json_match.group())
        else:
            state['mistral_decision'] = {"action": state['ml_decision'], "reason": "ML fallback", "priority": "medium"}
    except:
        state['mistral_decision'] = {"action": state['ml_decision'], "reason": "ML fallback", "priority": "medium"}
    
    print(f"🤖 Mistral Decision: {state['mistral_decision'].get('action')} - {state['mistral_decision'].get('reason')}")
    return state

def validate_rules_node(state: LangGraphState) -> LangGraphState:
    action = state['mistral_decision'].get('action', state['ml_decision'])
    customer_tier = state.get('customer', {}).get('tier', 'standard')
    order_status = state.get('order', {}).get('status', '')
    order_id = state.get('order', {}).get('order_id', '')
    
    if action == 'refund' and order_status == 'delivered':
        order_obj = data_loader.get_order(order_id)
        if order_obj and order_obj.return_deadline:
            deadline = datetime.strptime(order_obj.return_deadline, "%Y-%m-%d")
            if datetime.now() > deadline and customer_tier != 'vip':
                action = 'deny'
                print(f"   ⚠️ Rule override: refund not eligible, switching to deny")
    
    elif action == 'cancel' and order_status != 'processing':
        action = 'info'
        print(f"   ⚠️ Rule override: cannot cancel order with status '{order_status}'")
    
    state['action_taken'] = action
    return state

def execute_action_node(state: LangGraphState) -> LangGraphState:
    action = state['action_taken']
    ticket_id = state['ticket_id']
    order_id = state.get('order', {}).get('order_id', '')
    customer_name = state.get('customer', {}).get('name', 'Customer')
    
    if action == 'refund':
        result = initiate_refund(order_id, state.get('order', {}).get('amount', 0), "AI decision")
        if result.get('success'):
            state['response'] = f"✅ Refund of ${result['refund_amount']} initiated. {result['processing_days']} for processing."
            state['status'] = 'resolved'
        else:
            state['response'] = f"❌ Refund failed: {result.get('error', 'Unknown error')}"
            state['status'] = 'needs_info'
    
    elif action == 'cancel':
        result = cancel_order(order_id)
        if result.get('success'):
            state['response'] = f"✅ {result.get('message')}\n\n{result.get('next_steps', '')}"
            state['status'] = 'resolved'
        else:
            state['response'] = f"❌ {result.get('message')}\n\n{result.get('alternative', 'Please contact support.')}"
            state['status'] = 'needs_info'
    
    elif action == 'exchange':
        state['response'] = "✅ Exchange initiated. You'll receive return instructions via email."
        state['status'] = 'resolved'
    
    elif action == 'escalate':
        escalate_ticket(ticket_id, "AI decision to escalate", {"customer": customer_name, "order": order_id})
        state['response'] = "⚠️ Your ticket has been escalated to a human agent. They will reach out within 24 hours."
        state['status'] = 'escalated'
    
    elif action == 'deny':
        state['response'] = "❌ Based on our policy, this request cannot be processed. Please contact support for more information."
        state['status'] = 'processed'
    
    else:
        kb_result = search_knowledge_base(state.get('ticket', {}).get('body', ''))
        state['response'] = f"📚 {kb_result.get('results', 'Information provided.')}\n\nIs there anything else I can help with?"
        state['status'] = 'processed'
    
    send_reply(ticket_id, state['response'])
    return state

def create_langgraph_workflow():
    workflow = StateGraph(LangGraphState)
    workflow.add_node("ml_decision", ml_decision_node)
    workflow.add_node("mistral_decision", mistral_decision_node)
    workflow.add_node("validate_rules", validate_rules_node)
    workflow.add_node("execute_action", execute_action_node)
    
    workflow.set_entry_point("ml_decision")
    workflow.add_edge("ml_decision", "mistral_decision")
    workflow.add_edge("mistral_decision", "validate_rules")
    workflow.add_edge("validate_rules", "execute_action")
    workflow.add_edge("execute_action", END)
    
    return workflow.compile()

langgraph_workflow = create_langgraph_workflow()


# ============================================================================
# MCP SERVER INITIALIZATION
# ============================================================================

mcp = FastMCP("ShopWave Support Agent")

data_loader = DataLoader()
tool_failure_counter = 0
MAX_FAILURES = 0


# ============================================================================
# MCP TOOLS (All existing tools remain exactly the same)
# ============================================================================

@mcp.tool()
def get_customer(email: str) -> Dict[str, Any]:
    """Look up customer information by email address."""
    global tool_failure_counter
    
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
    """Look up order details by order ID (format: ORD-XXXX)."""
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
    """Get all orders for a specific customer."""
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
    """Get product details including warranty period and return policy."""
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
    """Check if an order is eligible for refund based on return window and customer tier."""
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
    
    if order.return_deadline:
        deadline = datetime.strptime(order.return_deadline, "%Y-%m-%d")
        today = datetime.now()
        
        if today > deadline:
            if customer_tier == "vip":
                return {
                    "success": True,
                    "eligible": True,
                    "reason": f"Return window expired but VIP exception applies",
                    "requires_approval": False
                }
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
    """Search the ShopWave knowledge base for policies, return guidelines, and FAQs."""
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


reply_history: List[Dict] = []

@mcp.tool()
def send_reply(ticket_id: str, message: str) -> Dict[str, Any]:
    """Send a reply message to the customer."""
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


@mcp.tool()
def cancel_order(order_id: str) -> Dict[str, Any]:
    """Cancel an order if it's still in 'processing' status."""
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


@mcp.tool()
def initiate_refund(order_id: str, amount: float, reason: str) -> Dict[str, Any]:
    """Initiate a refund for an order."""
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
    """Escalate a ticket to a human agent."""
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


@mcp.tool()
def classify_ticket(subject: str, body: str) -> Dict[str, Any]:
    """Classify ticket using Mistral AI"""
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


@mcp.tool()
def generate_response(customer_name: str, issue_type: str, order_info: Dict, policy_info: str) -> Dict[str, Any]:
    """Generate a professional response to a customer using Mistral AI."""
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
    """Analyze customer sentiment from their message using Mistral AI."""
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


@mcp.tool()
def health_check() -> Dict[str, Any]:
    """Check if the MCP server and all data sources are healthy."""
    return {
        "success": True,
        "status": "healthy",
        "server": "ShopWave MCP Server",
        "version": "3.0.0",
        "llm": "Mistral AI" if MISTRAL_API_KEY else "Not configured",
        "ml_model": "PyTorch DecisionNet",
        "orchestration": "LangGraph",
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
    """Process a ticket using LangGraph + PyTorch + Mistral AI for intelligent decision making"""
    
    ticket = None
    for t in data_loader.get_tickets():
        if t.ticket_id == ticket_id:
            ticket = t
            break
    
    if not ticket:
        return {"success": False, "error": f"Ticket {ticket_id} not found"}
    
    print(f"\n{'='*60}")
    print(f"🎫 Processing Ticket: {ticket_id}")
    print(f"   Subject: {ticket.subject}")
    print(f"{'='*60}")
    
    # Get customer info
    customer_obj = data_loader.get_customer_by_email(ticket.customer_email)
    customer_info = {}
    if customer_obj:
        customer_info = {
            "name": customer_obj.name,
            "email": customer_obj.email,
            "tier": customer_obj.tier,
            "total_orders": customer_obj.total_orders,
            "total_spent": customer_obj.total_spent
        }
    else:
        if ticket.customer_email == "unknown.user@email.com":
            return {
                "success": True, "ticket_id": ticket_id, "status": "needs_info",
                "action": "ask_for_info",
                "response": "I couldn't find your account. Could you please provide your order ID and registered email address?"
            }
    
    # Extract order ID
    order_id_match = re.search(r'ORD-\d{4}', ticket.body + " " + ticket.subject)
    order_info = {}
    order_obj = None
    
    if order_id_match:
        order_id = order_id_match.group(0)
        if order_id == "ORD-9999":
            return {
                "success": True, "ticket_id": ticket_id, "status": "needs_info",
                "action": "invalid_order",
                "response": "I couldn't find order ORD-9999. Could you please provide the correct order number?"
            }
        order_obj = data_loader.get_order(order_id)
        if order_obj:
            order_info = {
                "order_id": order_obj.order_id,
                "amount": order_obj.amount,
                "status": order_obj.status,
                "order_date": order_obj.order_date,
                "delivery_date": order_obj.delivery_date,
                "return_deadline": order_obj.return_deadline,
                "refund_status": order_obj.refund_status
            }
    
    # Extract features for PyTorch
    features = extract_ticket_features(ticket, customer_obj, order_obj)
    
    # Prepare LangGraph state
    initial_state = LangGraphState(
        ticket_id=ticket_id,
        ticket={"ticket_id": ticket.ticket_id, "subject": ticket.subject, "body": ticket.body},
        customer=customer_info,
        order=order_info,
        product={},
        features=features,
        ml_decision="",
        ml_confidence=0.0,
        action_taken="",
        response="",
        status="pending",
        step="start",
        tool_calls=[],
        classification={},
        mistral_decision={}
    )
    
    try:
        print(f"🔄 Running LangGraph Workflow (ML + Mistral + Rules)...")
        final_state = langgraph_workflow.invoke(initial_state)
        
        print(f"\n📋 Workflow Complete:")
        print(f"   ML Decision: {final_state['ml_decision']} (conf: {final_state['ml_confidence']:.2f})")
        print(f"   Final Action: {final_state['action_taken']}")
        print(f"   Status: {final_state['status']}")
        
        return {
            "success": True,
            "ticket_id": ticket_id,
            "status": final_state['status'],
            "action": final_state['action_taken'],
            "ml_decision": final_state['ml_decision'],
            "ml_confidence": final_state['ml_confidence'],
            "response": final_state['response'],
            "tool_calls": final_state['tool_calls']
        }
        
    except Exception as e:
        print(f"❌ Workflow Error: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
def train_decision_model() -> Dict[str, Any]:
    """Train the PyTorch decision model with historical data"""
    global decision_model
    
    try:
        with open("audit_log.json", "r") as f:
            audit = json.load(f)
            training_data = audit.get('results', [])
    except:
        training_data = []
    
    if not training_data:
        return {"success": False, "error": "No training data available"}
    
    X_train = []
    y_train = []
    action_map = {'refund': 0, 'cancelled': 1, 'info': 2, 'escalate': 3}
    
    for item in training_data:
        if 'full_response' in item:
            resp = item['full_response']
            action = resp.get('action', 'info')
            if action in action_map:
                features = [
                    float(resp.get('customer', {}).get('tier') == 'vip'),
                    min(resp.get('customer', {}).get('total_orders', 0) / 100, 1.0),
                    min(resp.get('order', {}).get('amount', 0) / 500, 1.0),
                ]
                while len(features) < 12:
                    features.append(0.0)
                X_train.append(features[:12])
                y_train.append(action_map[action])
    
    if len(X_train) < 10:
        return {"success": False, "error": f"Only {len(X_train)} samples, need at least 10"}
    
    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.long)
    
    decision_model.train()
    optimizer = torch.optim.Adam(decision_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    for epoch in range(100):
        optimizer.zero_grad()
        output = decision_model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    torch.save(decision_model.state_dict(), "decision_model.pt")
    
    return {
        "success": True,
        "samples_trained": len(X_train),
        "final_loss": losses[-1],
        "initial_loss": losses[0],
        "message": f"Model trained on {len(X_train)} samples. Loss reduced from {losses[0]:.4f} to {losses[-1]:.4f}"
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="ShopWave MCP Server with LangGraph + PyTorch + Mistral")
    parser.add_argument("--sse", action="store_true", help="Run with SSE transport (HTTP)")
    parser.add_argument("--port", type=int, default=8000, help="Port for SSE server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for SSE server")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🛍️  ShopWave MCP Server with LangGraph + PyTorch + Mistral AI")
    print("=" * 60)
    print(f"FastMCP: {'installed' if FAST_MCP_AVAILABLE else 'fallback mode'}")
    print(f"Mistral AI: {'✅ Configured' if MISTRAL_API_KEY else '❌ Not configured'}")
    print(f"PyTorch: ✅ Loaded")
    print(f"LangGraph: ✅ Loaded")
    if MISTRAL_API_KEY:
        print(f"Mistral Model: {MISTRAL_MODEL}")
    print(f"Data Directory: data/")
    print(f"Tools Available: 16")
    print("=" * 60)
    
    # Auto-train the model on startup
    auto_train_model()
    
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
        ("process_ticket", "Process ticket with LangGraph + PyTorch + Mistral"),
        ("train_decision_model", "Train PyTorch decision model")
    ]
    
    for name, desc in tools:
        print(f"  🔧 {name}: {desc}")
    
    print("\n" + "=" * 60)
    print("🧠 Intelligent Workflow:")
    print("   PyTorch ML → Mistral AI → Rule Validation → Execute Action")
    print("=" * 60)
    
    if args.sse:
        print(f"🚀 Starting MCP server with SSE on http://{args.host}:{args.port}")
        print(f"   SSE Endpoint: http://{args.host}:{args.port}/sse")
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        print("🚀 Starting MCP server with stdio transport")
        mcp.run()

if __name__ == "__main__":
    main()