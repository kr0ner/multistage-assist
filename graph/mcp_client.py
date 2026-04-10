"""Bridge client for calling the Graph MCP Server tools.

Usage:
    python graph/mcp_client.py list
    python graph/mcp_client.py call get_file_purpose --path "stage2_llm.py"
    python graph/mcp_client.py call get_blast_radius --path "utils/german_utils.py"
"""

import sys
import json
import argparse
import httpx

SERVER_URL = "http://localhost:8765/messages"

def list_tools():
    try:
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "id": 1,
            "params": {}
        }
        resp = httpx.post(SERVER_URL, json=payload, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
        if "result" in data:
            for tool in data["result"]["tools"]:
                print(f"- {tool['name']}: {tool['description']}")
        else:
            print(f"Error: {data}")
    except Exception as e:
        print(f"Failed to connect to MCP server at {SERVER_URL}: {e}")
        print("Make sure the server is running: python graph/graph_server.py --http")

def call_tool(name, arguments):
    try:
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "id": 1,
            "params": {
                "name": name,
                "arguments": arguments
            }
        }
        resp = httpx.post(SERVER_URL, json=payload, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
        
        if "error" in data:
            print(json.dumps(data["error"], indent=2))
            sys.exit(1)
            
        result = data.get("result", {})
        content = result.get("content", [])
        for item in content:
            if item.get("type") == "text":
                print(item.get("text"))
            else:
                print(json.dumps(item, indent=2))
        
        if result.get("isError"):
            sys.exit(1)
            
    except Exception as e:
        print(f"Error calling tool {name}: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Graph MCP Client Bridge")
    subparsers = parser.add_subparsers(dest="command")

    # List command
    subparsers.add_parser("list", help="List available tools")

    # Call command
    call_parser = subparsers.add_parser("call", help="Call a tool")
    call_parser.add_argument("name", help="Tool name")
    call_parser.add_argument("--path", help="Path argument")
    call_parser.add_argument("--query", help="Query argument")
    call_parser.add_argument("--req_id", help="Requirement ID")
    call_parser.add_argument("--max_depth", type=int, help="Max depth for blast radius")
    call_parser.add_argument("--base", help="Git base for detect_changes")

    args, unknown = parser.parse_known_args()

    if args.command == "list":
        list_tools()
    elif args.command == "call":
        # Build arguments dict from known args
        arguments = {}
        if args.path: arguments["path"] = args.path
        if args.query: arguments["query"] = args.query
        if args.req_id: arguments["req_id"] = args.req_id
        if args.max_depth: arguments["max_depth"] = args.max_depth
        if args.base: arguments["base"] = args.base
        
        # Add unknown args as key=value
        for arg in unknown:
            if "=" in arg:
                k, v = arg.split("=", 1)
                arguments[k] = v
        
        call_tool(args.name, arguments)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
