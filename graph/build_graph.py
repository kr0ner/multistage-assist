"""Code graph builder — parses Python AST into a queryable SQLite graph.

Scans all Python source files, extracts structural nodes (files, classes,
functions, constants) and relationship edges (imports, extends, calls, tests,
stage_uses), then stores them in .code-graph/graph.db.

Usage:
    python graph/build_graph.py          # Full rebuild
    python graph/build_graph.py --update  # Incremental (changed files only)
"""

import ast
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / ".code-graph"
DB_PATH = DB_DIR / "graph.db"
REQUIREMENTS_PATH = ROOT / "graph" / "requirements.yaml"

# Directories to scan for source code (relative to ROOT)
SOURCE_DIRS = [
    ".",
    "capabilities",
    "constants",
    "utils",
    "utils/cache_patterns",
]

# Directories to scan for tests
TEST_DIRS = [
    "tests",
    "tests/integration",
]

# Documentation contract: code patterns → docs that must be updated
DOC_CONTRACT = {
    "stage0.py": ["docs/ARCHITECTURE.md", "docs/wiki/Architecture.md", "docs/wiki/Development-Guide.md"],
    "stage1_cache.py": ["docs/ARCHITECTURE.md", "docs/wiki/Architecture.md", "docs/wiki/Development-Guide.md"],
    "stage2_llm.py": ["docs/ARCHITECTURE.md", "docs/wiki/Architecture.md", "docs/wiki/Development-Guide.md"],
    "stage3_cloud.py": ["docs/ARCHITECTURE.md", "docs/wiki/Architecture.md", "docs/wiki/Configuration.md"],
    "base_stage.py": ["docs/ARCHITECTURE.md", "docs/wiki/Architecture.md"],
    "stage_result.py": ["docs/ARCHITECTURE.md"],
    "const.py": ["docs/wiki/Configuration.md"],
    "conversation.py": ["docs/ARCHITECTURE.md"],
    "execution_pipeline.py": ["docs/ARCHITECTURE.md"],
    "capabilities/intent_executor.py": ["docs/wiki/Home.md"],
    "capabilities/mcp.py": ["docs/ARCHITECTURE.md", "docs/wiki/Capabilities-Reference.md"],
    "capabilities/timer.py": ["docs/wiki/Timers.md"],
    "capabilities/calendar.py": ["docs/wiki/Calendar-Events.md"],
    "capabilities/vacuum.py": ["docs/wiki/Vacuum-Control.md"],
    "capabilities/knowledge_graph.py": ["docs/wiki/Knowledge-Graph.md"],
    "capabilities/semantic_cache.py": ["docs/CACHE_PRINCIPLES.md"],
}

# Key values that must stay in sync across code and docs
SYNC_TARGETS = {
    "BRIGHTNESS_STEP": "capabilities/intent_executor.py",
    "COVER_STEP": "capabilities/intent_executor.py",
    "PENDING_TIMEOUT_SECONDS": "conversation.py",
    "PENDING_MAX_RETRIES": "conversation.py",
    "PENDING_ABSOLUTE_TIMEOUT": "conversation.py",
    "CACHE_BYPASS_INTENTS": "stage1_cache.py",
    "SERVICE_DOMAINS": "const.py",
    "LIGHT_COMPATIBLE_DOMAINS": "const.py",
    "EXPERT_DEFAULTS": "const.py",
}

# Stage → capabilities mapping patterns
STAGE_CAPABILITIES = {
    "stage0.py": "capabilities",
    "stage1_cache.py": "capabilities",
    "stage2_llm.py": "capabilities",
    "stage3_cloud.py": "capabilities",
}

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS nodes (
    id          TEXT PRIMARY KEY,
    type        TEXT NOT NULL,
    name        TEXT NOT NULL,
    file_path   TEXT NOT NULL,
    line_start  INTEGER,
    line_end    INTEGER,
    docstring   TEXT DEFAULT '',
    hash        TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS edges (
    source_id   TEXT NOT NULL,
    target_id   TEXT NOT NULL,
    type        TEXT NOT NULL,
    UNIQUE(source_id, target_id, type)
);

CREATE TABLE IF NOT EXISTS requirements (
    id           TEXT PRIMARY KEY,
    category     TEXT NOT NULL,
    title        TEXT NOT NULL,
    description  TEXT DEFAULT '',
    source_files TEXT DEFAULT '[]',
    test_files   TEXT DEFAULT '[]',
    status       TEXT DEFAULT 'implemented'
);

CREATE TABLE IF NOT EXISTS sync_values (
    name        TEXT PRIMARY KEY,
    value       TEXT,
    file_path   TEXT,
    line_number INTEGER
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
    name, docstring, file_path
);
"""


def _get_db() -> sqlite3.Connection:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(DB_PATH))
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    return db


def init_schema(db: sqlite3.Connection) -> None:
    db.executescript(SCHEMA_SQL)
    try:
        db.executescript(FTS_SQL)
    except sqlite3.OperationalError:
        pass  # FTS table already exists
    db.commit()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel_path(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _collect_python_files(dirs: list[str]) -> list[Path]:
    """Collect all .py files from the given directories (non-recursive per dir)."""
    files = []
    for d in dirs:
        full = ROOT / d
        if not full.is_dir():
            continue
        for f in sorted(full.iterdir()):
            if f.suffix == ".py" and f.is_file():
                files.append(f)
    return files


def _needs_rebuild(rel: str, file_hash: str, db: sqlite3.Connection) -> bool:
    """Check if file changed since last build."""
    row = db.execute(
        "SELECT hash FROM nodes WHERE id=? AND type='file'", (rel,)
    ).fetchone()
    return not row or row[0] != file_hash


def _resolve_call_name(node: ast.Call) -> Optional[str]:
    """Resolve a Call node to a readable target name."""
    return _resolve_name(node.func)


def _set_parents(tree: ast.AST) -> None:
    """Annotate every AST node with a _parent attribute."""
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child._parent = parent  # type: ignore[attr-defined]


def _enclosing_scope(node: ast.AST, rel: str) -> Optional[str]:
    """Walk up _parent chain to find the enclosing function/class scope ID."""
    current = node
    while hasattr(current, "_parent"):
        current = current._parent  # type: ignore[attr-defined]
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Check if method inside class
            if hasattr(current, "_parent") and isinstance(current._parent, ast.ClassDef):  # type: ignore[attr-defined]
                return f"{rel}::{current._parent.name}.{current.name}"  # type: ignore[attr-defined]
            return f"{rel}::{current.name}"
        if isinstance(current, ast.ClassDef):
            return f"{rel}::{current.name}"
    return rel  # file-level scope


def parse_file(path: Path) -> dict:
    """Extract nodes and edges from a single Python file."""
    source = path.read_text(encoding="utf-8")
    rel = _rel_path(path)
    fhash = hashlib.sha256(source.encode("utf-8")).hexdigest()

    try:
        tree = ast.parse(source, filename=rel)
    except SyntaxError:
        return {"nodes": [], "edges": [], "hash": fhash}

    _set_parents(tree)

    nodes = []
    edges = []

    # File node
    nodes.append({
        "id": rel,
        "type": "file",
        "name": path.stem,
        "file_path": rel,
        "line_start": 1,
        "line_end": len(source.splitlines()),
        "docstring": ast.get_docstring(tree) or "",
        "hash": fhash,
    })

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_id = f"{rel}::{node.name}"
            nodes.append({
                "id": class_id,
                "type": "class",
                "name": node.name,
                "file_path": rel,
                "line_start": node.lineno,
                "line_end": node.end_lineno or node.lineno,
                "docstring": ast.get_docstring(node) or "",
                "hash": fhash,
            })
            # CONTAINS: file → class
            edges.append((rel, class_id, "contains"))
            for base in node.bases:
                base_name = _resolve_name(base)
                if base_name:
                    edges.append((class_id, base_name, "extends"))
            # CONTAINS: class → methods
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_id = f"{rel}::{node.name}.{item.name}"
                    nodes.append({
                        "id": method_id,
                        "type": "method",
                        "name": f"{node.name}.{item.name}",
                        "file_path": rel,
                        "line_start": item.lineno,
                        "line_end": item.end_lineno or item.lineno,
                        "docstring": ast.get_docstring(item) or "",
                        "hash": fhash,
                    })
                    edges.append((class_id, method_id, "contains"))

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Skip methods — already handled inside ClassDef above
            if hasattr(node, "_parent") and isinstance(node._parent, ast.ClassDef):  # type: ignore[attr-defined]
                continue
            func_id = f"{rel}::{node.name}"
            nodes.append({
                "id": func_id,
                "type": "function",
                "name": node.name,
                "file_path": rel,
                "line_start": node.lineno,
                "line_end": node.end_lineno or node.lineno,
                "docstring": ast.get_docstring(node) or "",
                "hash": fhash,
            })
            # CONTAINS: file → function
            edges.append((rel, func_id, "contains"))

        elif isinstance(node, ast.Call):
            # CALLS: enclosing_scope → call target
            call_name = _resolve_call_name(node)
            if call_name:
                caller = _enclosing_scope(node, rel)
                edges.append((caller, call_name, "calls"))

        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in (node.names or []):
                target = f"{node.module}.{alias.name}"
                edges.append((rel, target, "imports"))

        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    const_id = f"{rel}::{target.id}"
                    nodes.append({
                        "id": const_id,
                        "type": "constant",
                        "name": target.id,
                        "file_path": rel,
                        "line_start": node.lineno,
                        "line_end": node.end_lineno or node.lineno,
                        "docstring": "",
                        "hash": fhash,
                    })
                    # CONTAINS: file → constant
                    edges.append((rel, const_id, "contains"))

    return {"nodes": nodes, "edges": edges, "hash": fhash}


def _resolve_name(node: ast.expr) -> Optional[str]:
    """Resolve an AST node to a dotted name string."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _resolve_name(node.value)
        if parent:
            return f"{parent}.{node.attr}"
    return None


def build_test_edges(db: sqlite3.Connection) -> None:
    """Link test files to source modules they import."""
    test_files = _collect_python_files(TEST_DIRS)
    for path in test_files:
        rel = _rel_path(path)
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=rel)
        except (SyntaxError, UnicodeDecodeError):
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if "multistage_assist" in node.module:
                    parts = node.module.replace("multistage_assist.", "").split(".")
                    source_path = "/".join(parts) + ".py"
                    db.execute(
                        "INSERT OR IGNORE INTO edges(source_id, target_id, type) VALUES (?, ?, 'tests')",
                        (rel, source_path),
                    )


def build_tested_by_edges(db: sqlite3.Connection) -> None:
    """Link test functions to production functions they call (tested_by edges)."""
    # Gather all known production function/method node IDs for resolution
    prod_names: dict[str, str] = {}  # short_name → node_id
    for row in db.execute(
        "SELECT id, name FROM nodes WHERE type IN ('function', 'method') "
        "AND file_path NOT LIKE 'tests/%'"
    ).fetchall():
        prod_names[row[1]] = row[0]
        # Also index by last dotted part (e.g. 'MyClass.run' → 'run')
        if "." in row[1]:
            prod_names[row[1].split(".")[-1]] = row[0]

    test_files = _collect_python_files(TEST_DIRS)
    for path in test_files:
        rel = _rel_path(path)
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=rel)
        except (SyntaxError, UnicodeDecodeError):
            continue

        _set_parents(tree)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call_name = _resolve_call_name(node)
            if not call_name:
                continue
            # Find the test function this call lives in
            test_func = _enclosing_scope(node, rel)
            if not test_func or "::test_" not in test_func:
                continue
            # Resolve call target to a known production function
            short = call_name.split(".")[-1]
            prod_id = prod_names.get(call_name) or prod_names.get(short)
            if prod_id:
                db.execute(
                    "INSERT OR IGNORE INTO edges(source_id, target_id, type) "
                    "VALUES (?, ?, 'tested_by')",
                    (test_func, prod_id),
                )


def build_stage_capability_edges(db: sqlite3.Connection) -> None:
    """Link stage files to the capabilities they instantiate."""
    for stage_file in STAGE_CAPABILITIES:
        path = ROOT / stage_file
        if not path.exists():
            continue
        rel = _rel_path(path)
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=rel)
        except (SyntaxError, UnicodeDecodeError):
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if "capabilities." in (node.module or "") or (node.module or "").startswith("capabilities"):
                    for alias in (node.names or []):
                        if "Capability" in alias.name:
                            cap_module = node.module.split(".")[-1]
                            cap_file = f"capabilities/{cap_module}.py"
                            db.execute(
                                "INSERT OR IGNORE INTO edges(source_id, target_id, type) VALUES (?, ?, 'stage_uses')",
                                (rel, cap_file),
                            )


def build_doc_contract_edges(db: sqlite3.Connection) -> None:
    """Encode the documentation contract as edges."""
    for source_file, doc_files in DOC_CONTRACT.items():
        for doc_file in doc_files:
            db.execute(
                "INSERT OR IGNORE INTO edges(source_id, target_id, type) VALUES (?, ?, 'doc_contract')",
                (source_file, doc_file),
            )


def extract_sync_values(db: sqlite3.Connection) -> None:
    """Extract key constants that must stay in sync."""
    for name, file_path in SYNC_TARGETS.items():
        full = ROOT / file_path
        if not full.exists():
            continue
        source = full.read_text(encoding="utf-8")
        for i, line in enumerate(source.splitlines(), 1):
            if re.match(rf"^{re.escape(name)}\s*=", line):
                value = line.split("=", 1)[1].strip()
                if len(value) > 200:
                    value = value[:200] + "..."
                db.execute(
                    "INSERT OR REPLACE INTO sync_values(name, value, file_path, line_number) VALUES (?, ?, ?, ?)",
                    (name, value, file_path, i),
                )
                break


def load_requirements(db: sqlite3.Connection) -> None:
    """Load requirements from YAML into SQLite."""
    if not REQUIREMENTS_PATH.exists():
        return
    with open(REQUIREMENTS_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    for req_id, req in data.items():
        db.execute(
            "INSERT OR REPLACE INTO requirements(id, category, title, description, source_files, test_files, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                req_id,
                req.get("category", ""),
                req.get("title", ""),
                req.get("description", ""),
                json.dumps(req.get("source_files", [])),
                json.dumps(req.get("test_files", [])),
                req.get("status", "implemented"),
            ),
        )


def rebuild_fts(db: sqlite3.Connection) -> None:
    """Rebuild the full-text search index."""
    try:
        db.execute("DROP TABLE IF EXISTS nodes_fts")
    except sqlite3.OperationalError:
        pass
    db.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5("
        "name, docstring, file_path)"
    )
    db.execute(
        "INSERT INTO nodes_fts(name, docstring, file_path) "
        "SELECT name, docstring, file_path FROM nodes"
    )
    db.commit()


def build(incremental: bool = False) -> dict:
    """Build or update the code graph. Returns stats."""
    db = _get_db()
    init_schema(db)

    if not incremental:
        db.execute("DELETE FROM nodes")
        db.execute("DELETE FROM edges")
        db.execute("DELETE FROM sync_values")
        db.execute("DELETE FROM requirements")

    source_files = _collect_python_files(SOURCE_DIRS)
    test_files = _collect_python_files(TEST_DIRS)
    all_files = source_files + test_files

    parsed = 0
    skipped = 0

    for path in all_files:
        rel = _rel_path(path)
        fhash = _file_hash(path)

        if incremental and not _needs_rebuild(rel, fhash, db):
            skipped += 1
            continue

        # Remove old data for this file
        db.execute("DELETE FROM nodes WHERE file_path=?", (rel,))
        db.execute("DELETE FROM edges WHERE source_id=? OR source_id LIKE ?", (rel, f"{rel}::%"))

        result = parse_file(path)
        for n in result["nodes"]:
            db.execute(
                "INSERT OR REPLACE INTO nodes(id, type, name, file_path, line_start, line_end, docstring, hash) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (n["id"], n["type"], n["name"], n["file_path"],
                 n["line_start"], n["line_end"], n["docstring"], n["hash"]),
            )
        for src, tgt, etype in result["edges"]:
            db.execute(
                "INSERT OR IGNORE INTO edges(source_id, target_id, type) VALUES (?, ?, ?)",
                (src, tgt, etype),
            )
        parsed += 1

    # Semantic edges
    build_test_edges(db)
    build_tested_by_edges(db)
    build_stage_capability_edges(db)
    build_doc_contract_edges(db)
    extract_sync_values(db)
    load_requirements(db)
    rebuild_fts(db)

    db.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES ('last_build', ?)",
        (time.strftime("%Y-%m-%dT%H:%M:%S"),),
    )
    db.commit()

    stats = {
        "files_parsed": parsed,
        "files_skipped": skipped,
        "total_nodes": db.execute("SELECT COUNT(*) FROM nodes").fetchone()[0],
        "total_edges": db.execute("SELECT COUNT(*) FROM edges").fetchone()[0],
        "total_requirements": db.execute("SELECT COUNT(*) FROM requirements").fetchone()[0],
        "total_sync_values": db.execute("SELECT COUNT(*) FROM sync_values").fetchone()[0],
    }
    db.close()
    return stats


def main():
    incremental = "--update" in sys.argv
    mode = "incremental" if incremental else "full"
    print(f"Building code graph ({mode})...")
    t0 = time.time()
    stats = build(incremental=incremental)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.2f}s")
    print(f"  Files parsed: {stats['files_parsed']}")
    if stats["files_skipped"]:
        print(f"  Files skipped (unchanged): {stats['files_skipped']}")
    print(f"  Nodes: {stats['total_nodes']}")
    print(f"  Edges: {stats['total_edges']}")
    print(f"  Requirements: {stats['total_requirements']}")
    print(f"  Sync values: {stats['total_sync_values']}")


if __name__ == "__main__":
    main()
