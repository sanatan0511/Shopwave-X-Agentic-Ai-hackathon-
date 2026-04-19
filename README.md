# agentic_ai_hackthon_2026_
link of demostration - https://drive.google.com/file/d/1awx3ZrGGLV1OPCy5v5vYHr22avxfyjfQ/view?usp=drive_link


<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/3170d2e4-c177-462c-9e03-576a0651ac99" />



Here's a complete **ASCII architecture diagram** of the entire ShopWave Autonomous Support Agent system, followed by an explanation of each component.

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                              SHOPWAVE AUTONOMOUS SUPPORT AGENT                               │
│                                   (MCP Server + Agentic AI)                                  │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   DATA LAYER (Static JSON)                                   │
├─────────────────────┬─────────────────────┬─────────────────────┬──────────────────────────┤
│   customers.json    │    orders.json      │    products.json    │   tickets.json           │
│  (10+ customers)    │   (15+ orders)       │    (8 products)      │   (20+ tickets)          │
│  • tier (vip/prem)  │  • status/amount    │  • warranty/return   │  • subject/body          │
│  • total_spent      │  • return_deadline  │  • returnable flag   │  • expected_action       │
│  • notes (exceptions│  • refund_status    │  • category          │                          │
└─────────────────────┴─────────────────────┴─────────────────────┴──────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                              DATA LOADER (Singleton)                                         │
│                     Loads all JSON into memory dictionaries & lists                          │
│                     Provides methods: get_customer_by_email(), get_order(), etc.             │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                              MCP SERVER (FastMCP)                                            │
│                              Exposes 16 Tools via SSE/Stdio                                  │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐       │
│  │ get_customer │ │  get_order   │ │get_cust_orders│ │ get_product  │ │check_refund  │       │
│  │              │ │              │ │              │ │              │ │ eligibility  │       │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘       │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐       │
│  │search_kb     │ │  send_reply  │ │ cancel_order │ │initiate_refund│ │escalate_ticket│       │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘       │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐       │
│  │classify_ticket│ │generate_resp │ │analyze_sent  │ │ health_check │ │process_ticket│       │
│  │ (Mistral AI)  │ │ (Mistral AI) │ │ (Mistral AI) │ │              │ │  (MAIN)      │       │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘       │
│                                                    │                                         │
│                                      + train_decision_model (PyTorch retraining)            │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                        CORE AGENTIC WORKFLOW (LangGraph)                                     │
│                    Invoked by `process_ticket(ticket_id)` tool                               │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                              │
│   ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐                    │
│   │ 1. ml_decision  │─────▶│ 2. mistral_     │─────▶│ 3. validate_    │                    │
│   │     Node        │      │    decision     │      │    rules_node    │                    │
│   │                 │      │     Node        │      │                 │                    │
│   │ • PyTorch model │      │ • Calls Mistral │      │ • Override with  │                    │
│   │ • Input: 12     │      │   API with      │      │   business rules │                    │
│   │   features      │      │   ticket prompt │      │ • VIP exception  │                    │
│   │ • Output:       │      │ • Returns JSON  │      │ • Order status   │                    │
│   │   refund/cancel │      │   action &      │      │   checks         │                    │
│   │   info/escalate │      │   reason        │      │                  │                    │
│   └─────────────────┘      └─────────────────┘      └────────┬────────┘                    │
│                                                              │                               │
│                                                              ▼                               │
│                                          ┌─────────────────┐                                │
│                                          │ 4. execute_     │                                │
│                                          │    action_node  │                                │
│                                          │                 │                                │
│                                          │ • refund        │                                │
│                                          │ • cancel        │                                │
│                                          │ • exchange      │                                │
│                                          │ • escalate      │                                │
│                                          │ • deny          │                                │
│                                          │ • info          │                                │
│                                          └────────┬────────┘                                │
│                                                   │                                          │
│                                                   ▼                                          │
│                                          ┌─────────────────┐                                │
│                                          │   send_reply()   │                                │
│                                          │   & final state  │                                │
│                                          └─────────────────┘                                │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                              OUTPUT & AUDIT                                                  │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│  • Customer receives reply (simulated via send_reply)                                       │
│  • Ticket status updated (resolved / escalated / needs_info / processed)                    │
│  • Full audit log saved to audit_log.json (includes:                                        │
│       - ticket_id, status, action, ml_confidence, response, tool_calls)                     │
│  • Action items created for admin if ticket cannot be auto‑resolved                         │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                              EXTERNAL SERVICES                                              │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│  • Mistral AI API (LLM) – for classification, sentiment, response generation               │
│  • PyTorch – runs locally for ML predictions                                                │
│  • LangGraph – orchestrates workflow (runs locally)                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Explanation of the Architecture

### 1. **Data Layer**
- Static JSON files (`customers.json`, `orders.json`, `products.json`, `tickets.json`) simulate the e‑commerce backend.
- `knowledge-base.md` contains policy text.
- All data is loaded once at startup by the `DataLoader` singleton.

### 2. **MCP Server (FastMCP)**
- Exposes **16 tools** via the Model Context Protocol.
- Can run with **SSE (HTTP)** or **stdio** transport.
- Tools are grouped into:
  - **Lookup**: `get_customer`, `get_order`, `get_customer_orders`, `get_product`
  - **Policy & KB**: `check_refund_eligibility`, `search_knowledge_base`
  - **Actions**: `send_reply`, `cancel_order`, `initiate_refund`, `escalate_ticket`
  - **AI helpers**: `classify_ticket`, `generate_response`, `analyze_sentiment` (all use Mistral)
  - **System**: `health_check`, `process_ticket` (main entry), `train_decision_model`

### 3. **LangGraph Workflow** (inside `process_ticket`)
This is the brain of the agent. It runs a **sequential graph** of four nodes:

| Node | What it does | Input | Output |
|------|--------------|-------|--------|
| **ml_decision** | PyTorch neural network predicts action based on 12 numerical features. | 12‑feature vector | `refund`, `cancel`, `info`, `escalate` + confidence |
| **mistral_decision** | Calls Mistral API with full ticket text and policy summary. Returns a more nuanced action (`refund`, `cancel`, `exchange`, `escalate`, `info`, `deny`) + reason. | Ticket text, customer tier, order status | JSON action & reason |
| **validate_rules** | Applies hard business rules (e.g., VIP exception for expired returns, cannot cancel shipped orders). Overrides Mistral/ML decision if needed. | Action from previous node | Final action (may be changed) |
| **execute_action** | Performs the actual work: calls appropriate MCP tool (`initiate_refund`, `cancel_order`, etc.), generates a reply, and sends it via `send_reply`. | Final action | Updated state (response, status) |

### 4. **PyTorch Model**
- Small neural network: `12 inputs → 32 hidden (ReLU) → 32 hidden (ReLU) → 4 outputs (softmax)`.
- Trained on historical audit logs (or synthetic data) to predict `refund`, `cancel`, `info`, `escalate`.
- Used as a **fast fallback** when Mistral API is unavailable or as an additional signal.

### 5. **Mistral AI Integration**
- Three tools use Mistral:
  - `classify_ticket`: categorises ticket into `return_refund`, `damaged_item`, etc.
  - `generate_response`: writes a polite, empathetic reply.
  - `analyze_sentiment`: detects urgency and emotion.
- The `mistral_decision` node (inside the workflow) uses a separate prompt to decide the overall action.
- All calls are synchronous HTTP requests to `https://api.mistral.ai/v1/chat/completions`.

### 6. **Audit & Output**
- Every decision and tool call is logged.
- Final `audit_log.json` contains a complete record for each ticket:
  - `ticket_id`, `status`, `action`, `ml_decision`, `ml_confidence`, `response`, `tool_calls`.
- The `send_reply` tool stores a history of all messages sent (in `reply_history`).

### 7. **External Services**
- **Mistral AI API**: remote LLM service (requires API key).
- **PyTorch & LangGraph**: run locally inside the Python process.

---

## Data Flow Summary (for a single ticket)

1. **User (or script)** calls `process_ticket("TKT-001")` via MCP.
2. The tool loads ticket, customer, order, product data.
3. It extracts 12 features → passes them to the LangGraph workflow.
4. **Node 1 (PyTorch)** predicts an action.
5. **Node 2 (Mistral)** analyses the ticket text and suggests an action.
6. **Node 3 (Rules)** checks if the action violates any policy; overrides if necessary.
7. **Node 4 (Execution)**:
   - If `refund` → calls `initiate_refund` and `send_reply`.
   - If `cancel` → calls `cancel_order` and `send_reply`.
   - If `exchange` → sends a canned response.
   - If `escalate` → calls `escalate_ticket` and informs customer.
   - If `deny` → sends a denial message.
   - If `info` → searches KB and replies.
8. The final state is returned to the caller and also saved in `audit_log.json`.

---

## Why This Architecture Works for the Hackathon

- **3+ tool calls per chain**: The workflow alone calls at least `get_customer`, `get_order`, `initiate_refund/cancel_order`, and `send_reply`.  
- **Concurrency**: MCP server can handle multiple requests; the processor script uses `asyncio` to process tickets concurrently.  
- **Graceful failure**: Simulated timeouts are caught, and Mistral API failures fall back to ML decision.  
- **Explainability**: Every decision is logged with reasoning (ML confidence, Mistral reason, rule overrides).  
- **Modularity**: Tools are decoupled from the workflow; you can replace Mistral with another LLM or change the PyTorch model without rewriting the whole system.

This architecture demonstrates a **production‑ready, agentic AI** that autonomously resolves support tickets using a hybrid of ML, LLM, and rule‑based reasoning.
