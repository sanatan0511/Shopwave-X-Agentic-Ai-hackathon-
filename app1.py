# app.py
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_socketio import SocketIO, emit
import json
import os
import re
from datetime import datetime
from functools import wraps
from pathlib import Path

app = Flask(__name__)
app.secret_key = "shopwave_secret_key_2026"
socketio = SocketIO(app, cors_allowed_origins="*")

# Path to data files from MCP server folder
DATA_FOLDER = r"C:\Users\rashm\Downloads\KSOLVERS\Problem Solving\data"
MCP_FOLDER = r"C:\Users\rashm\Downloads\KSOLVERS\Problem Solving"

# Mock user data
users = {
    "customer1": {"password": "pass123", "role": "customer", "name": "Alice Johnson", "email": "alice@email.com", "tier": "vip"},
    "customer2": {"password": "pass123", "role": "customer", "name": "Bob Smith", "email": "bob@email.com", "tier": "standard"},
    "admin": {"password": "admin123", "role": "admin", "name": "Admin User", "email": "admin@shopwave.com", "tier": "admin"}
}

# Store tickets and actions
action_items = []
audit_logs = []
user_tickets = []

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_json_file(filename):
    """Load data from JSON file"""
    filepath = os.path.join(DATA_FOLDER, filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return []

def get_customer_by_email(email):
    """Get customer from JSON file"""
    customers = load_json_file("customers.json")
    for customer in customers:
        if customer.get('email') == email:
            return customer
    return None

def get_order_by_id(order_id):
    """Get order from JSON file"""
    orders = load_json_file("orders.json")
    for order in orders:
        if order.get('order_id') == order_id:
            return order
    return None

def get_product_by_id(product_id):
    """Get product from JSON file"""
    products = load_json_file("products.json")
    for product in products:
        if product.get('product_id') == product_id:
            return product
    return None

def get_all_tickets():
    """Get all tickets from JSON file"""
    tickets = load_json_file("tickets.json")
    return tickets

def get_audit_log_path():
    """Get the correct path to audit_log.json"""
    mcp_audit_path = os.path.join(MCP_FOLDER, "audit_log.json")
    if os.path.exists(mcp_audit_path):
        return mcp_audit_path
    if os.path.exists("audit_log.json"):
        return "audit_log.json"
    return None

def load_mcp_audit():
    """Load audit log from MCP server folder"""
    audit_path = get_audit_log_path()
    if audit_path:
        try:
            with open(audit_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading audit log: {e}")
    return {"summary": {}, "results": []}

# ============================================================================
# MCP TOOL FUNCTIONS (Direct file access)
# ============================================================================

def mcp_get_customer(email):
    """Get customer information"""
    customer = get_customer_by_email(email)
    if customer:
        return {
            "success": True,
            "found": True,
            "customer_id": customer.get('customer_id'),
            "name": customer.get('name'),
            "email": customer.get('email'),
            "phone": customer.get('phone'),
            "tier": customer.get('tier'),
            "member_since": customer.get('member_since'),
            "total_orders": customer.get('total_orders'),
            "total_spent": customer.get('total_spent'),
            "notes": customer.get('notes')
        }
    return {"success": False, "found": False, "error": "Customer not found"}

def mcp_get_order(order_id):
    """Get order information"""
    order = get_order_by_id(order_id)
    if order:
        product = get_product_by_id(order.get('product_id'))
        return {
            "success": True,
            "found": True,
            "order_id": order.get('order_id'),
            "customer_id": order.get('customer_id'),
            "product_id": order.get('product_id'),
            "product_name": product.get('name') if product else "Unknown",
            "quantity": order.get('quantity'),
            "amount": order.get('amount'),
            "status": order.get('status'),
            "order_date": order.get('order_date'),
            "delivery_date": order.get('delivery_date'),
            "return_deadline": order.get('return_deadline'),
            "refund_status": order.get('refund_status'),
            "notes": order.get('notes')
        }
    return {"success": False, "found": False, "error": "Order not found"}

def mcp_check_refund_eligibility(order_id, customer_tier="standard"):
    """Check if order is eligible for refund"""
    order = get_order_by_id(order_id)
    
    if not order:
        return {"eligible": False, "reason": "Order not found"}
    
    if order.get('refund_status') == "refunded":
        return {"eligible": False, "reason": "Refund already processed"}
    
    product = get_product_by_id(order.get('product_id'))
    if product and not product.get('returnable', True):
        return {"eligible": False, "reason": f"{product.get('name')} is not returnable"}
    
    # Check return deadline
    return_deadline = order.get('return_deadline')
    if return_deadline:
        deadline = datetime.strptime(return_deadline, "%Y-%m-%d")
        if datetime.now() > deadline:
            if customer_tier == "vip":
                return {"eligible": True, "reason": "VIP exception - return window expired"}
            elif customer_tier == "premium":
                return {"eligible": True, "requires_approval": True, "reason": "Premium - use judgment"}
            else:
                return {"eligible": False, "reason": f"Return window expired on {return_deadline}"}
    
    return {"eligible": True, "reason": "Within return window"}

def mcp_cancel_order(order_id):
    """Cancel an order if possible"""
    order = get_order_by_id(order_id)
    
    if not order:
        return {"success": False, "error": "Order not found"}
    
    if order.get('status') == "processing":
        return {
            "success": True,
            "message": f"Order {order_id} has been cancelled successfully",
            "refund_initiated": True,
            "refund_amount": order.get('amount'),
            "next_steps": "Refund will be processed within 5-7 business days"
        }
    elif order.get('status') == "shipped":
        return {
            "success": False,
            "message": f"Order {order_id} has already shipped and cannot be cancelled",
            "alternative": "Please wait for delivery and initiate a return"
        }
    else:
        return {
            "success": False,
            "message": f"Order {order_id} status is '{order.get('status')}' and cannot be cancelled"
        }

def mcp_initiate_refund(order_id, amount, reason):
    """Initiate a refund"""
    order = get_order_by_id(order_id)
    
    if not order:
        return {"success": False, "error": "Order not found"}
    
    if order.get('refund_status') == "refunded":
        return {"success": False, "error": "Refund already processed"}
    
    refund_amount = min(amount, order.get('amount', 0))
    
    return {
        "success": True,
        "refund_id": f"REF-{order_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "refund_amount": refund_amount,
        "original_amount": order.get('amount'),
        "reason": reason,
        "processing_days": "5-7 business days"
    }

def mcp_search_knowledge_base(query):
    """Search knowledge base"""
    kb_path = os.path.join(DATA_FOLDER, "knowledge-base.md")
    kb_content = ""
    try:
        with open(kb_path, 'r', encoding='utf-8') as f:
            kb_content = f.read()
    except:
        pass
    
    query_lower = query.lower()
    relevant_sections = []
    
    if kb_content:
        sections = kb_content.split("\n## ")
        for section in sections:
            if any(kw in section.lower() for kw in query_lower.split()[:3]):
                relevant_sections.append(section[:500])
    
    if relevant_sections:
        results = "\n\n---\n\n".join(relevant_sections[:2])
    else:
        results = """**ShopWave Policies:**

• **Return Policy:** 30-day return window for most items. Electronics have 15-day window. Electronics accessories have 60-day window.

• **Refund Policy:** Refunds processed within 5-7 business days after approval.

• **Warranty:** 12 months for electronics, 24 months for home appliances.

• **Customer Tiers:** VIP customers have extended return windows. Premium customers get priority support.

For detailed policy information, please visit our website or contact support."""
    
    return {"success": True, "results": results}

def mcp_classify_ticket(subject, body):
    """Classify ticket using rules"""
    text = (subject + " " + body).lower()
    
    if any(w in text for w in ["refund", "return"]):
        category = "return_refund"
        priority = "medium"
    elif any(w in text for w in ["cancel"]):
        category = "cancellation"
        priority = "high"
    elif any(w in text for w in ["damaged", "broken", "cracked", "defect"]):
        category = "damaged_item"
        priority = "high"
    elif any(w in text for w in ["wrong", "incorrect"]):
        category = "wrong_item"
        priority = "medium"
    elif any(w in text for w in ["where", "track", "shipping", "delivery"]):
        category = "shipping"
        priority = "low"
    else:
        category = "general_inquiry"
        priority = "low"
    
    return {
        "success": True,
        "classification": {
            "category": category,
            "priority": priority,
            "resolvable": True,
            "confidence": 0.85
        }
    }

# ============================================================================
# FLASK ROUTES
# ============================================================================

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get('role') != 'admin':
            return jsonify({"error": "Admin access required"}), 403
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    if 'user' in session:
        if session.get('role') == 'admin':
            return redirect(url_for('admin_dashboard'))
        else:
            return redirect(url_for('user_dashboard'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username in users and users[username]['password'] == password:
            session['user'] = username
            session['role'] = users[username]['role']
            session['name'] = users[username]['name']
            session['email'] = users[username].get('email', '')
            session['tier'] = users[username].get('tier', 'standard')
            
            if users[username]['role'] == 'admin':
                return redirect(url_for('admin_dashboard'))
            else:
                return redirect(url_for('user_dashboard'))
        else:
            return render_template('index.html', error="Invalid credentials")
    
    return render_template('index.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/user/dashboard')
@login_required
def user_dashboard():
    if session.get('role') != 'customer':
        return redirect(url_for('admin_dashboard'))
    return render_template('user_dashboard.html', user=session)

@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    if session.get('role') != 'admin':
        return redirect(url_for('user_dashboard'))
    return render_template('admin_dashboard.html', user=session)

@app.route('/audit_log')
@admin_required
def audit_log_page():
    return render_template('audit_log.html')

# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/api/create_ticket', methods=['POST'])
@login_required
def create_ticket():
    """Create a new support ticket with auto-resolution"""
    data = request.json
    
    ticket_id = f"TKT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Extract order ID from message
    order_id = data.get('order_id')
    if not order_id:
        order_match = re.search(r'ORD-\d{4}', data.get('body', '') + " " + data.get('subject', ''))
        if order_match:
            order_id = order_match.group(0)
    
    ticket_data = {
        "ticket_id": ticket_id,
        "customer_email": session.get('email'),
        "customer_name": session.get('name'),
        "subject": data.get('subject'),
        "body": data.get('body'),
        "order_id": order_id,
        "source": data.get('source', 'web_portal'),
        "created_at": datetime.now().isoformat(),
        "status": "pending"
    }
    
    # Auto-resolution logic
    auto_resolved = False
    resolution_response = ""
    action_taken = None
    
    subject_lower = ticket_data['subject'].lower()
    body_lower = ticket_data['body'].lower()
    
    # Check for refund request
    if 'refund' in subject_lower or 'refund' in body_lower:
        if order_id:
            customer_tier = session.get('tier', 'standard')
            eligibility = mcp_check_refund_eligibility(order_id, customer_tier)
            
            if eligibility.get('eligible'):
                order = get_order_by_id(order_id)
                if order:
                    refund_result = mcp_initiate_refund(order_id, order.get('amount', 0), body_lower[:100])
                    if refund_result.get('success'):
                        auto_resolved = True
                        action_taken = "refund_initiated"
                        resolution_response = f"✅ Refund of ${refund_result.get('refund_amount', 0)} has been initiated. {refund_result.get('processing_days', '5-7 business days')} for processing."
                        ticket_data['status'] = 'resolved'
    
    # Check for cancellation request
    if not auto_resolved and ('cancel' in subject_lower or 'cancel' in body_lower):
        if order_id:
            cancel_result = mcp_cancel_order(order_id)
            if cancel_result.get('success'):
                auto_resolved = True
                action_taken = "order_cancelled"
                resolution_response = f"✅ {cancel_result.get('message')}\n\n{cancel_result.get('next_steps', '')}"
                ticket_data['status'] = 'resolved'
    
    # Check for tracking request
    if not auto_resolved and ('track' in subject_lower or 'where' in body_lower or 'status' in body_lower):
        if order_id:
            order = get_order_by_id(order_id)
            if order:
                auto_resolved = True
                action_taken = "info_provided"
                resolution_response = f"""📦 **Order {order_id} Details:**

• Status: {order.get('status', 'N/A')}
• Amount: ${order.get('amount', 0)}
• Order Date: {order.get('order_date', 'N/A')}
• Delivery Date: {order.get('delivery_date', 'Not yet delivered')}

You can track your order using the tracking information sent to your email."""
                ticket_data['status'] = 'resolved'
    
    # Save ticket
    user_tickets.append(ticket_data)
    try:
        with open('user_tickets.json', 'a') as f:
            f.write(json.dumps(ticket_data) + '\n')
    except:
        pass
    
    # Add to action items if not auto-resolved
    if not auto_resolved:
        action_items.append({
            "id": len(action_items) + 1,
            "type": "new_ticket",
            "ticket_id": ticket_id,
            "message": f"New ticket from {session.get('name')}: {data.get('subject')}",
            "timestamp": datetime.now().isoformat(),
            "status": "pending"
        })
        
        socketio.emit('new_action', {
            "action": "new_ticket",
            "ticket_id": ticket_id,
            "message": f"New ticket from {session.get('name')}"
        })
    
    # Audit log
    audit_logs.append({
        "action": "ticket_created",
        "ticket_id": ticket_id,
        "auto_resolved": auto_resolved,
        "action_taken": action_taken,
        "customer": session.get('name'),
        "timestamp": datetime.now().isoformat()
    })
    
    return jsonify({
        "success": True,
        "ticket_id": ticket_id,
        "auto_resolved": auto_resolved,
        "action_taken": action_taken,
        "response": resolution_response or "Ticket created. Support team will respond shortly."
    })

@app.route('/api/get_user_tickets')
@login_required
def get_user_tickets():
    """Get tickets for logged-in user"""
    user_email = session.get('email')
    user_ticket_list = [t for t in user_tickets if t.get('customer_email') == user_email]
    
    formatted_tickets = []
    for ticket in user_ticket_list:
        formatted_tickets.append({
            "ticket_id": ticket.get('ticket_id'),
            "subject": ticket.get('subject'),
            "status": ticket.get('status'),
            "created_at": ticket.get('created_at'),
            "auto_resolved": ticket.get('status') == 'resolved'
        })
    
    return jsonify({
        "success": True,
        "tickets": formatted_tickets,
        "count": len(formatted_tickets)
    })

@app.route('/api/get_action_items')
@admin_required
def get_action_items():
    """Get pending action items"""
    pending_actions = [a for a in action_items if a.get('status') == 'pending']
    return jsonify({
        "actions": pending_actions,
        "count": len(pending_actions)
    })

@app.route('/api/resolve_action/<int:action_id>', methods=['POST'])
@admin_required
def resolve_action(action_id):
    """Mark action as resolved"""
    for action in action_items:
        if action['id'] == action_id:
            action['status'] = 'resolved'
            action['resolved_at'] = datetime.now().isoformat()
            action['resolved_by'] = session.get('user')
            
            audit_logs.append({
                "action": "admin_resolved",
                "action_id": action_id,
                "admin": session.get('user'),
                "timestamp": datetime.now().isoformat()
            })
            
            return jsonify({"success": True})
    
    return jsonify({"error": "Action not found"}), 404

@app.route('/api/get_audit_logs')
@admin_required
def get_audit_logs():
    """Get all audit logs"""
    mcp_audit = load_mcp_audit()
    return jsonify({
        "mcp_audit": mcp_audit,
        "web_audit": audit_logs,
        "action_items": action_items
    })

@app.route('/api/get_stats')
@admin_required
def get_stats():
    """Get dashboard statistics"""
    mcp_audit = load_mcp_audit()
    summary = mcp_audit.get('summary', {})
    results = mcp_audit.get('results', [])
    
    # Calculate total tool calls
    total_tool_calls = 0
    for r in results:
        total_tool_calls += r.get('tool_calls_count', 4)
    
    total_tickets = summary.get('resolved', 0) + summary.get('escalated', 0) + summary.get('needs_info', 0)
    resolved = summary.get('resolved', 0)
    
    # Calculate resolution rate
    resolution_rate = round((resolved / total_tickets) * 100, 1) if total_tickets > 0 else 0
    avg_tools = round(total_tool_calls / total_tickets, 1) if total_tickets > 0 else 0
    
    return jsonify({
        "total_tickets": total_tickets,
        "resolved": resolved,
        "escalated": summary.get('escalated', 0),
        "needs_info": summary.get('needs_info', 0),
        "pending": summary.get('needs_info', 0),
        "action_items": len([a for a in action_items if a.get('status') == 'pending']),
        "total_tool_calls": total_tool_calls,
        "avg_tools_per_ticket": avg_tools,
        "resolution_rate": resolution_rate,
        "mcp_server": {"status": "online"},
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/call_mcp_tool', methods=['POST'])
@login_required
def call_mcp_tool_api():
    """Call MCP tool from chatbot"""
    data = request.json
    tool_name = data.get('tool')
    arguments = data.get('arguments', {})
    
    if tool_name == 'get_order':
        result = mcp_get_order(arguments.get('order_id'))
    elif tool_name == 'get_customer':
        result = mcp_get_customer(arguments.get('email'))
    elif tool_name == 'check_refund_eligibility':
        result = mcp_check_refund_eligibility(
            arguments.get('order_id'),
            arguments.get('customer_tier', 'standard')
        )
    elif tool_name == 'cancel_order':
        result = mcp_cancel_order(arguments.get('order_id'))
    elif tool_name == 'initiate_refund':
        result = mcp_initiate_refund(
            arguments.get('order_id'),
            arguments.get('amount', 0),
            arguments.get('reason', '')
        )
    elif tool_name == 'search_knowledge_base':
        result = mcp_search_knowledge_base(arguments.get('query', ''))
    elif tool_name == 'classify_ticket':
        result = mcp_classify_ticket(
            arguments.get('subject', ''),
            arguments.get('body', '')
        )
    else:
        result = {"success": False, "error": f"Unknown tool: {tool_name}"}
    
    return jsonify(result)

@app.route('/api/get_customer_info')
@login_required
def get_customer_info():
    """Get customer information"""
    email = session.get('email')
    result = mcp_get_customer(email)
    if result.get('success'):
        # Add tier from session if not found
        if not result.get('tier'):
            result['tier'] = session.get('tier', 'standard')
    return jsonify(result)

@app.route('/api/search_knowledge', methods=['POST'])
@login_required
def search_knowledge():
    """Search knowledge base"""
    query = request.json.get('query', '')
    result = mcp_search_knowledge_base(query)
    return jsonify(result)

@app.route('/api/process_ticket', methods=['POST'])
@admin_required
def process_ticket():
    """Process a ticket with MCP"""
    data = request.json
    ticket_id = data.get('ticket_id')
    
    # Find ticket in MCP audit
    mcp_audit = load_mcp_audit()
    results = mcp_audit.get('results', [])
    ticket_result = None
    
    for r in results:
        if r.get('ticket_id') == ticket_id:
            ticket_result = r
            break
    
    if ticket_result:
        audit_logs.append({
            "action": "ticket_processed",
            "ticket_id": ticket_id,
            "admin": session.get('user'),
            "timestamp": datetime.now().isoformat()
        })
        return jsonify(ticket_result)
    else:
        return jsonify({"success": False, "error": "Ticket not found"})

@app.route('/api/process_all_tickets', methods=['POST'])
@admin_required
def process_all_tickets():
    """Process all tickets"""
    mcp_audit = load_mcp_audit()
    results = mcp_audit.get('results', [])
    
    audit_logs.append({
        "action": "all_tickets_processed",
        "count": len(results),
        "admin": session.get('user'),
        "timestamp": datetime.now().isoformat()
    })
    
    return jsonify({
        "success": True,
        "total_processed": len(results),
        "results": results
    })

# SocketIO events
@socketio.on('connect')
def handle_connect():
    emit('connected', {'data': 'Connected to ShopWave', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    print("=" * 60)
    print("🛍️  ShopWave Flask Application")
    print("=" * 60)
    print(f"📍 URL: http://localhost:5000")
    print(f"📁 Data Folder: {DATA_FOLDER}")
    print("=" * 60)
    
    # Check if data folder exists
    if os.path.exists(DATA_FOLDER):
        print(f"✅ Data folder found")
        customers = load_json_file("customers.json")
        orders = load_json_file("orders.json")
        tickets = load_json_file("tickets.json")
        print(f"📊 Loaded: {len(customers)} customers, {len(orders)} orders, {len(tickets)} tickets")
    else:
        print(f"⚠️ Data folder not found at: {DATA_FOLDER}")
        print("   Make sure the MCP server data folder is at the correct location")
    
    # Load audit stats
    mcp_audit = load_mcp_audit()
    summary = mcp_audit.get('summary', {})
    if summary:
        print(f"📈 Previous Run Stats: {summary.get('resolved', 0)} resolved, {summary.get('needs_info', 0)} pending")
    
    print("\n📝 Demo Accounts:")
    print("   👤 Customer: customer1 / pass123 (VIP Tier)")
    print("   👤 Customer: customer2 / pass123 (Standard Tier)")
    print("   👑 Admin: admin / admin123")
    print("=" * 60)
    print("\n💬 Chatbot Features:")
    print("   • Auto-detect refund/return requests")
    print("   • Collect order IDs automatically")
    print("   • Process refunds and cancellations")
    print("   • Track orders")
    print("   • Search knowledge base")
    print("=" * 60)
    print("\n🔧 Admin Features:")
    print("   • View all processed tickets")
    print("   • Filter by status and category")
    print("   • Click tickets for details")
    print("   • View action items")
    print("   • Full audit log")
    print("=" * 60)
    
    socketio.run(app, debug=True, port=5000)