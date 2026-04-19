from fastapi import FastAPI
from pydantic import BaseModel
from fastmcp import FastMCP

# -------------------------------
# MCP TOOLS (CORE LOGIC)
# -------------------------------
mcp = FastMCP(name="calculator")

@mcp.tool()
def multiply(a: float, b: float) -> float:
    return a * b

@mcp.tool()
def add(a: float, b: float) -> float:
    return a + b

@mcp.tool()
def subtract(a: float, b: float) -> float:
    return a - b

@mcp.tool()
def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


# -------------------------------
# FASTAPI APP
# -------------------------------
app = FastAPI(title="FastAPI + MCP Integration")

class Numbers(BaseModel):
    a: float
    b: float


@app.get("/")
def root():
    return {"message": "FastAPI + MCP connected successfully!"}


# -------------------------------
# FASTAPI → MCP BRIDGE
# -------------------------------
@app.post("/multiply")
def api_multiply(data: Numbers):
    return {"result": multiply(data.a, data.b)}

@app.post("/add")
def api_add(data: Numbers):
    return {"result": add(data.a, data.b)}

@app.post("/subtract")
def api_subtract(data: Numbers):
    return {"result": subtract(data.a, data.b)}

@app.post("/divide")
def api_divide(data: Numbers):
    try:
        return {"result": divide(data.a, data.b)}
    except ValueError as e:
        return {"error": str(e)}


# -------------------------------
# RUN SERVER
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)