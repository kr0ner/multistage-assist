"""MCP server exposing the code graph as query tools.

Run:
    python graph/graph_server.py          # stdio transport (for .mcp.json)
    python graph/graph_server.py --http   # HTTP+SSE transport (port 8765)
"""

import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional
import asyncio

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / ".code-graph" / "graph.db"


def _get_db() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Graph database not found at {DB_PATH}. Run: python graph/build_graph.py"
        )
    db = sqlite3.connect(str(DB_PATH))
    db.row_factory = sqlite3.Row
    return db


def _rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict]:
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def get_file_purpose(path: str) -> dict[str, Any]:
    """Get file purpose, contained classes/functions, and what imports it."""
    db = _get_db()
    file_node = db.execute(
        "SELECT * FROM nodes WHERE id=? AND type='file'", (path,)
    ).fetchone()
    if not file_node:
        db.close()
        return {"error": f"File '{path}' not found in graph"}

    children = _rows_to_dicts(db.execute(
        "SELECT type, name, line_start, line_end, docstring FROM nodes "
        "WHERE file_path=? AND type!='file' ORDER BY line_start",
        (path,),
    ).fetchall())

    importers = [r["source_id"] for r in db.execute(
        "SELECT source_id FROM edges WHERE target_id=? AND type='imports'",
        (path,),
    ).fetchall()]

    imported_by_files = [r["source_id"] for r in db.execute(
        "SELECT DISTINCT source_id FROM edges WHERE target_id LIKE ? AND type='imports'",
        (f"%{Path(path).stem}%",),
    ).fetchall()]

    db.close()
    return {
        "path": path,
        "docstring": file_node["docstring"],
        "children": children,
        "imported_by": list(set(importers + imported_by_files)),
    }


def get_blast_radius(path: str, max_depth: int = 3) -> dict[str, Any]:
    """Multi-hop blast radius: recursively find all impacted nodes, tests, requirements, docs."""
    db = _get_db()

    # Recursive CTE: walk imports, calls, contains, extends edges in both directions
    cte_sql = """
    WITH RECURSIVE impacted(node_id, depth) AS (
        SELECT id, 0 FROM nodes WHERE file_path = ?
        UNION
        SELECT e.source_id, i.depth + 1
        FROM impacted i JOIN edges e ON e.target_id = i.node_id
        WHERE i.depth < ? AND e.type IN ('imports', 'calls', 'extends', 'contains')
        UNION
        SELECT e.target_id, i.depth + 1
        FROM impacted i JOIN edges e ON e.source_id = i.node_id
        WHERE i.depth < ? AND e.type IN ('imports', 'calls', 'extends', 'contains')
    )
    SELECT DISTINCT node_id, MIN(depth) as depth FROM impacted GROUP BY node_id
    """
    impacted_rows = db.execute(cte_sql, (path, max_depth, max_depth)).fetchall()
    impacted = [{'node_id': r[0], 'depth': r[1]} for r in impacted_rows]
    impacted_ids = {r[0] for r in impacted_rows}

    # Collect affected files
    affected_files = sorted({r[0].split('::', 1)[0] for r in impacted_rows})

    # Tests covering this file (direct + tested_by)
    tests = [r["source_id"] for r in db.execute(
        "SELECT source_id FROM edges WHERE target_id=? AND type='tests'",
        (path,),
    ).fetchall()]
    # Also get tested_by edges for functions in this file
    tested_by = _rows_to_dicts(db.execute(
        "SELECT DISTINCT e.source_id as test_func, e.target_id as prod_func "
        "FROM edges e JOIN nodes n ON e.target_id = n.id "
        "WHERE n.file_path = ? AND e.type = 'tested_by'",
        (path,),
    ).fetchall())

    # Requirements referencing this file
    reqs = _rows_to_dicts(db.execute(
        "SELECT id, title FROM requirements WHERE source_files LIKE ? OR test_files LIKE ?",
        (f"%{path}%", f"%{path}%"),
    ).fetchall())

    # Docs that need updating (doc contract)
    docs = [r["target_id"] for r in db.execute(
        "SELECT target_id FROM edges WHERE source_id=? AND type='doc_contract'",
        (path,),
    ).fetchall()]

    db.close()
    return {
        "path": path,
        "max_depth": max_depth,
        "affected_files": affected_files,
        "impacted_nodes": len(impacted),
        "tests": tests,
        "tested_by": tested_by,
        "requirements": reqs,
        "docs_to_update": docs,
    }


def get_requirement(req_id: str) -> dict[str, Any]:
    """Get full details for a requirement by ID."""
    db = _get_db()
    row = db.execute("SELECT * FROM requirements WHERE id=?", (req_id,)).fetchone()
    if not row:
        db.close()
        return {"error": f"Requirement '{req_id}' not found"}
    result = dict(row)
    result["source_files"] = json.loads(result["source_files"])
    result["test_files"] = json.loads(result["test_files"])
    db.close()
    return result


def find_requirements(query: str) -> list[dict]:
    """Search requirements by keyword in ID, title, or description."""
    db = _get_db()
    pattern = f"%{query}%"
    rows = _rows_to_dicts(db.execute(
        "SELECT id, category, title, status FROM requirements "
        "WHERE id LIKE ? OR title LIKE ? OR description LIKE ? OR category LIKE ?",
        (pattern, pattern, pattern, pattern),
    ).fetchall())
    db.close()
    return rows


def get_capability_info(name: str) -> dict[str, Any]:
    """Get capability details: file, class, stages that use it, tests."""
    db = _get_db()
    # Find capability file
    cap_file = f"capabilities/{name}.py"
    file_node = db.execute(
        "SELECT * FROM nodes WHERE id=? AND type='file'", (cap_file,)
    ).fetchone()
    if not file_node:
        # Try partial match
        rows = _rows_to_dicts(db.execute(
            "SELECT id, name FROM nodes WHERE type='file' AND file_path LIKE ?",
            (f"capabilities/%{name}%",),
        ).fetchall())
        db.close()
        return {"error": f"Capability '{name}' not found", "similar": rows}

    classes = _rows_to_dicts(db.execute(
        "SELECT name, docstring, line_start, line_end FROM nodes "
        "WHERE file_path=? AND type='class'", (cap_file,)
    ).fetchall())

    stages = [r["source_id"] for r in db.execute(
        "SELECT source_id FROM edges WHERE target_id=? AND type='stage_uses'",
        (cap_file,),
    ).fetchall()]

    tests = [r["source_id"] for r in db.execute(
        "SELECT source_id FROM edges WHERE target_id=? AND type='tests'",
        (cap_file,),
    ).fetchall()]

    reqs = _rows_to_dicts(db.execute(
        "SELECT id, title FROM requirements WHERE source_files LIKE ?",
        (f"%{cap_file}%",),
    ).fetchall())

    db.close()
    return {
        "file": cap_file,
        "docstring": file_node["docstring"],
        "classes": classes,
        "used_by_stages": stages,
        "tests": tests,
        "requirements": reqs,
    }


def get_stage_capabilities(stage: str) -> dict[str, Any]:
    """Get all capabilities registered in a stage file."""
    db = _get_db()
    stage_file = stage if stage.endswith(".py") else f"{stage}.py"
    caps = [r["target_id"] for r in db.execute(
        "SELECT target_id FROM edges WHERE source_id=? AND type='stage_uses'",
        (stage_file,),
    ).fetchall()]
    db.close()
    return {"stage": stage_file, "capabilities": caps}


def find_tests_for(path: str) -> list[str]:
    """Find test files that cover a source file."""
    db = _get_db()
    tests = [r["source_id"] for r in db.execute(
        "SELECT source_id FROM edges WHERE target_id=? AND type='tests'",
        (path,),
    ).fetchall()]
    # Also check requirements for test_files
    reqs = db.execute(
        "SELECT test_files FROM requirements WHERE source_files LIKE ?",
        (f"%{path}%",),
    ).fetchall()
    req_tests = []
    for r in reqs:
        req_tests.extend(json.loads(r["test_files"]))
    db.close()
    return list(set(tests + req_tests))


def get_sync_values() -> list[dict]:
    """Get all key constants with their current values and locations."""
    db = _get_db()
    rows = _rows_to_dicts(db.execute("SELECT * FROM sync_values").fetchall())
    db.close()
    return rows


def search_code(query: str) -> list[dict]:
    """FTS5 search across all code nodes (names, docstrings, paths)."""
    db = _get_db()
    try:
        rows = _rows_to_dicts(db.execute(
            "SELECT n.id, n.type, n.name, n.file_path, n.line_start, n.docstring "
            "FROM nodes_fts f JOIN nodes n ON f.rowid = n.rowid "
            "WHERE nodes_fts MATCH ? ORDER BY rank LIMIT 20",
            (query,),
        ).fetchall())
    except sqlite3.OperationalError:
        # Fallback to LIKE search if FTS query is invalid
        pattern = f"%{query}%"
        rows = _rows_to_dicts(db.execute(
            "SELECT id, type, name, file_path, line_start, docstring "
            "FROM nodes WHERE name LIKE ? OR docstring LIKE ? LIMIT 20",
            (pattern, pattern),
        ).fetchall())
    db.close()
    return rows


def detect_changes(base: str = "HEAD~1") -> dict[str, Any]:
    """Map git diff to affected functions, test coverage gaps, and docs to update."""
    # 1. Get changed files from git diff
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", base],
            capture_output=True, text=True, cwd=str(ROOT), timeout=10,
        )
        if result.returncode != 0:
            return {"error": f"git diff failed: {result.stderr.strip()}"}
        changed_files = [f for f in result.stdout.strip().splitlines() if f.endswith(".py")]
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return {"error": str(e)}

    if not changed_files:
        return {"changed_files": [], "affected_functions": [], "test_gaps": [], "docs_to_update": []}

    # 2. Get line-level diff to identify affected functions
    affected_functions = []
    try:
        diff_result = subprocess.run(
            ["git", "diff", "--unified=0", base],
            capture_output=True, text=True, cwd=str(ROOT), timeout=10,
        )
        changed_lines = _parse_diff_lines(diff_result.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        changed_lines = {}

    db = _get_db()

    for fpath in changed_files:
        lines = changed_lines.get(fpath, [])
        if lines:
            # Find functions/methods whose line range overlaps changed lines
            for line in lines:
                rows = db.execute(
                    "SELECT id, type, name, line_start, line_end FROM nodes "
                    "WHERE file_path=? AND type IN ('function', 'method') "
                    "AND line_start <= ? AND line_end >= ?",
                    (fpath, line, line),
                ).fetchall()
                for r in rows:
                    affected_functions.append({
                        "id": r[0], "type": r[1], "name": r[2],
                        "file": fpath, "line_start": r[3], "line_end": r[4],
                    })
        else:
            # No line info — all functions in the file are affected
            rows = db.execute(
                "SELECT id, type, name, line_start, line_end FROM nodes "
                "WHERE file_path=? AND type IN ('function', 'method')",
                (fpath,),
            ).fetchall()
            for r in rows:
                affected_functions.append({
                    "id": r[0], "type": r[1], "name": r[2],
                    "file": fpath, "line_start": r[3], "line_end": r[4],
                })

    # Deduplicate
    seen = set()
    unique_funcs = []
    for f in affected_functions:
        if f["id"] not in seen:
            seen.add(f["id"])
            unique_funcs.append(f)
    affected_functions = unique_funcs

    # 3. Check test coverage — find test gaps
    test_gaps = []
    for func in affected_functions:
        has_test = db.execute(
            "SELECT 1 FROM edges WHERE target_id=? AND type='tested_by' LIMIT 1",
            (func["id"],),
        ).fetchone()
        if not has_test:
            test_gaps.append(func)

    # 4. Collect docs to update from doc contract
    docs_to_update = set()
    for fpath in changed_files:
        rows = db.execute(
            "SELECT target_id FROM edges WHERE source_id=? AND type='doc_contract'",
            (fpath,),
        ).fetchall()
        for r in rows:
            docs_to_update.add(r[0])

    # 5. Collect affected requirements
    requirements = []
    for fpath in changed_files:
        rows = _rows_to_dicts(db.execute(
            "SELECT id, title FROM requirements WHERE source_files LIKE ?",
            (f"%{fpath}%",),
        ).fetchall())
        requirements.extend(rows)
    # Deduplicate requirements
    seen_reqs = set()
    unique_reqs = []
    for r in requirements:
        if r["id"] not in seen_reqs:
            seen_reqs.add(r["id"])
            unique_reqs.append(r)

    db.close()
    return {
        "base": base,
        "changed_files": changed_files,
        "affected_functions": affected_functions,
        "test_gaps": test_gaps,
        "docs_to_update": sorted(docs_to_update),
        "requirements": unique_reqs,
    }


def _parse_diff_lines(diff_output: str) -> dict[str, list[int]]:
    """Parse unified diff output to extract changed line numbers per file."""
    import re
    result: dict[str, list[int]] = {}
    current_file = None
    for line in diff_output.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:]
        elif line.startswith("@@") and current_file:
            # Parse @@ -old,count +new,count @@ format
            match = re.search(r"\+(\d+)(?:,(\d+))?", line)
            if match:
                start = int(match.group(1))
                count = int(match.group(2)) if match.group(2) else 1
                if current_file not in result:
                    result[current_file] = []
                result[current_file].extend(range(start, start + count))
    return result


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOLS = {
    "get_file_purpose": {
        "fn": get_file_purpose,
        "description": "Get file purpose, contained classes/functions, and what imports it.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Relative file path (e.g. 'capabilities/timer.py')"}},
            "required": ["path"],
        },
    },
    "get_blast_radius": {
        "fn": get_blast_radius,
        "description": "Multi-hop blast radius: recursively find affected files, tests, requirements, and docs when a file changes.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative file path"},
                "max_depth": {"type": "integer", "description": "Max traversal depth (default: 3)", "default": 3},
            },
            "required": ["path"],
        },
    },
    "get_requirement": {
        "fn": get_requirement,
        "description": "Get full details for a requirement by ID (e.g. 'REQ-PIPE-001').",
        "parameters": {
            "type": "object",
            "properties": {"req_id": {"type": "string", "description": "Requirement ID"}},
            "required": ["req_id"],
        },
    },
    "find_requirements": {
        "fn": find_requirements,
        "description": "Search requirements by keyword in ID, title, description, or category.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search keyword"}},
            "required": ["query"],
        },
    },
    "get_capability_info": {
        "fn": get_capability_info,
        "description": "Get capability details: file, class, stages that use it, tests, requirements.",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Capability module name (e.g. 'timer', 'semantic_cache')"}},
            "required": ["name"],
        },
    },
    "get_stage_capabilities": {
        "fn": get_stage_capabilities,
        "description": "Get all capabilities registered in a stage file.",
        "parameters": {
            "type": "object",
            "properties": {"stage": {"type": "string", "description": "Stage file (e.g. 'stage2_llm' or 'stage2_llm.py')"}},
            "required": ["stage"],
        },
    },
    "find_tests_for": {
        "fn": find_tests_for,
        "description": "Find test files covering a source file (graph edges + requirement cross-refs).",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Relative source file path"}},
            "required": ["path"],
        },
    },
    "get_sync_values": {
        "fn": get_sync_values,
        "description": "Get all key constants that must stay in sync (BRIGHTNESS_STEP, SERVICE_DOMAINS, etc.).",
        "parameters": {"type": "object", "properties": {}},
    },
    "search_code": {
        "fn": search_code,
        "description": "Full-text search across all code nodes (names, docstrings, file paths).",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query (FTS5 syntax or plain text)"}},
            "required": ["query"],
        },
    },
    "detect_changes": {
        "fn": detect_changes,
        "description": "Git-diff-aware change detection: maps changed lines to affected functions, test coverage gaps, docs to update, and requirements.",
        "parameters": {
            "type": "object",
            "properties": {"base": {"type": "string", "description": "Git ref to diff against (default: HEAD~1)", "default": "HEAD~1"}},
        },
    },
}


# ---------------------------------------------------------------------------
# MCP protocol handler (JSON-RPC over stdio)
# ---------------------------------------------------------------------------

def handle_request(req: dict) -> dict:
    """Handle a single JSON-RPC request."""
    method = req.get("method", "")
    req_id = req.get("id")
    params = req.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "multistage-graph", "version": "1.0.0"},
            },
        }

    if method == "notifications/initialized":
        return None  # No response for notifications

    if method == "tools/list":
        tools_list = []
        for name, spec in TOOLS.items():
            tools_list.append({
                "name": name,
                "description": spec["description"],
                "inputSchema": spec["parameters"],
            })
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": tools_list},
        }

    if method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        if tool_name not in TOOLS:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps({"error": f"Unknown tool: {tool_name}"})}],
                    "isError": True,
                },
            }
        try:
            result = TOOLS[tool_name]["fn"](**arguments)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result, indent=2, ensure_ascii=False)}],
                    "isError": False,
                },
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps({"error": str(e)})}],
                    "isError": True,
                },
            }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def run_stdio():
    """Run MCP server over stdio (JSON-RPC, one message per line)."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        response = handle_request(req)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


def run_http(host: str = "0.0.0.0", port: int = 8765):
    """Run MCP server over HTTP/SSE using FastAPI."""
    try:
        from fastapi import FastAPI, Request, BackgroundTasks
        from fastapi.responses import JSONResponse
        from starlette.responses import StreamingResponse
        import uvicorn
    except ImportError:
        print("Error: fastapi, starlette, or uvicorn not found. Install them to use --http.")
        sys.exit(1)

    app = FastAPI(title="MultiStage Graph MCP Server")
    
    # Simple queue for SSE messages (though stdio logic is synchronous here)
    # For a real MCP SSE transport, we'd need a more complex session handler.
    # We'll implement a simplified 'Direct POST' endpoint for easier bridge access,
    # and a basic SSE endpoint for protocol compliance.

    @app.post("/messages")
    async def post_message(request: Request):
        payload = await request.json()
        response = handle_request(payload)
        return JSONResponse(content=response)

    @app.get("/sse")
    async def sse_endpoint():
        async def event_generator():
            # Initial endpoint announcement (simplified)
            yield f"event: endpoint\ndata: /messages\n\n"
            while True:
                await asyncio.sleep(3600)
        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @app.get("/tools")
    async def list_tools_http():
        """Shortcut for tool discovery."""
        return handle_request({"method": "tools/list", "id": 1})

    print(f"Starting Graph MCP Server on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


def main():
    if "--test" in sys.argv:
        # Quick self-test: print all tools
        for name, spec in TOOLS.items():
            print(f"  {name}: {spec['description']}")
        return

    if "--http" in sys.argv:
        port = 8765
        if "--port" in sys.argv:
            idx = sys.argv.index("--port")
            if idx + 1 < len(sys.argv):
                port = int(sys.argv[idx + 1])
        run_http(port=port)
    else:
        run_stdio()


if __name__ == "__main__":
    main()
