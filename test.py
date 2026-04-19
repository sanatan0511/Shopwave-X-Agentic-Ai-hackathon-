#!/usr/bin/env python3
"""
Simple MCP Server Test Script for ShopWave
"""

import json
import subprocess
import sys
import time
import requests

def test_stdio_mode():
    """Test stdio mode - simpler approach"""
    print("=" * 60)
    print("🧪 Testing ShopWave MCP Server (Stdio Mode)")
    print("=" * 60)
    
    # First, check if the script exists
    import os
    if not os.path.exists("fastmcp_shopwave.py"):
        print("❌ fastmcp_shopwave.py not found!")
        return
    
    print("\n🚀 Starting MCP server...")
    
    try:
        # Start server process
        proc = subprocess.Popen(
            [sys.executable, "fastmcp_shopwave.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Give it time to start
        time.sleep(3)
        
        # Send initialize request
        init_msg = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "0.1.0",
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        }) + "\n"
        
        print("📤 Sending initialize request...")
        proc.stdin.write(init_msg)
        proc.stdin.flush()
        
        # Read response with timeout
        import select
        ready, _, _ = select.select([proc.stdout], [], [], 5)
        if ready:
            response = proc.stdout.readline()
            print(f"📥 Received: {response[:200]}...")
        else:
            print("⏰ Timeout waiting for response")
        
        # Send initialized notification
        notif_msg = json.dumps({
            "jsonrpc": "2.0",
            "method": "initialized"
        }) + "\n"
        
        proc.stdin.write(notif_msg)
        proc.stdin.flush()
        
        # Request tools list
        tools_msg = json.dumps({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }) + "\n"
        
        print("📤 Requesting tools list...")
        proc.stdin.write(tools_msg)
        proc.stdin.flush()
        
        # Read tools response
        ready, _, _ = select.select([proc.stdout], [], [], 5)
        if ready:
            response = proc.stdout.readline()
            print(f"📥 Tools response: {response[:500]}...")
            
            try:
                data = json.loads(response)
                if "result" in data and "tools" in data["result"]:
                    tools = data["result"]["tools"]
                    print(f"\n✅ Found {len(tools)} tools:")
                    for tool in tools[:10]:  # Show first 10
                        print(f"   🔧 {tool.get('name')}")
                    if len(tools) > 10:
                        print(f"   ... and {len(tools) - 10} more")
            except:
                print("   Could not parse response")
        
        # Clean up
        proc.terminate()
        print("\n✅ Test completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("=" * 60)

def test_sse_mode():
    """Test SSE mode - recommended approach"""
    print("=" * 60)
    print("🧪 Testing ShopWave MCP Server (SSE Mode)")
    print("=" * 60)
    
    server_url = "http://localhost:8002"
    
    # Test if server is running
    print("\n📡 Checking if server is running...")
    try:
        response = requests.get(f"{server_url}/sse", timeout=2)
        print("✅ Server is responding!")
    except requests.exceptions.ConnectionError:
        print("❌ Server is not running!")
        print("\nPlease start the server first:")
        print("   python fastmcp_shopwave.py --sse --port 8002")
        return
    except Exception as e:
        print(f"⚠️ Server response: {e}")
    
    # Try to call a tool via HTTP (if available)
    print("\n📡 Attempting to call a tool...")
    try:
        response = requests.post(
            f"{server_url}/call",
            json={"tool": "health_check", "arguments": {}},
            timeout=5
        )
        if response.status_code == 200:
            print("✅ Tool call successful!")
            data = response.json()
            if data.get('success'):
                print(f"   Status: {data.get('status')}")
                print(f"   Customers: {data.get('data_loaded', {}).get('customers', 0)}")
        else:
            print(f"⚠️ Response code: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("⚠️ HTTP endpoint not available (using SSE only)")
    except Exception as e:
        print(f"⚠️ Could not call tool: {e}")
    
    print("\n" + "=" * 60)
    print("✅ SSE Mode Test Complete!")
    print("=" * 60)

def quick_test():
    """Quick test - just check if server responds"""
    print("=" * 60)
    print("🔍 Quick MCP Server Test")
    print("=" * 60)
    
    import subprocess
    
    print("\n🔧 Checking Python environment...")
    result = subprocess.run([sys.executable, "--version"], capture_output=True, text=True)
    print(f"   {result.stdout.strip()}")
    
    print("\n📁 Checking for fastmcp_shopwave.py...")
    import os
    if os.path.exists("fastmcp_shopwave.py"):
        print("   ✅ Found fastmcp_shopwave.py")
    else:
        print("   ❌ fastmcp_shopwave.py not found!")
        return
    
    print("\n📦 Checking installed packages...")
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "show", "fastmcp"], 
                               capture_output=True, text=True)
        if "Name: fastmcp" in result.stdout:
            print("   ✅ fastmcp installed")
        else:
            print("   ⚠️ fastmcp not found")
    except:
        pass
    
    print("\n💡 To test the server, run:")
    print("   1. Start server: python fastmcp_shopwave.py --sse --port 8002")
    print("   2. Then run: python test.py --mode sse")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test ShopWave MCP Server")
    parser.add_argument("--mode", choices=["stdio", "sse", "quick"], default="quick",
                       help="Test mode: stdio, sse, or quick")
    
    args = parser.parse_args()
    
    if args.mode == "stdio":
        test_stdio_mode()
    elif args.mode == "sse":
        test_sse_mode()
    else:
        quick_test()