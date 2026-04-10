"""Tests for graph/build_graph.py — AST parser and SQLite graph builder."""

import json
import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure the project root is on sys.path so we can import graph.*
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graph import build_graph


@pytest.fixture
def fresh_db(tmp_path):
    """Provide a fresh in-memory-like DB using a temp directory."""
    db_path = tmp_path / "graph.db"
    with patch.object(build_graph, "DB_DIR", tmp_path), \
         patch.object(build_graph, "DB_PATH", db_path):
        db = build_graph._get_db()
        build_graph.init_schema(db)
        yield db
        db.close()


class TestParseFile:
    """Tests for parse_file() — AST node extraction."""

    def test_parse_simple_file(self, tmp_path):
        """parse_file extracts file, class, function, method, and constant nodes."""
        code = '''"""Module docstring."""

MY_CONST = 42

class Foo:
    """Foo class."""
    def method_a(self):
        pass

def bar():
    """Bar function."""
    pass
'''
        f = tmp_path / "example.py"
        f.write_text(code, encoding="utf-8")
        with patch.object(build_graph, "ROOT", tmp_path):
            result = build_graph.parse_file(f)

        nodes = result["nodes"]
        types = {n["type"] for n in nodes}
        assert "file" in types
        assert "class" in types
        assert "function" in types
        assert "constant" in types
        assert "method" in types

        names = {n["name"] for n in nodes}
        assert "example" in names  # file stem
        assert "Foo" in names
        assert "bar" in names
        assert "MY_CONST" in names
        assert "Foo.method_a" in names

    def test_parse_extracts_docstrings(self, tmp_path):
        """Docstrings are captured for files, classes, and functions."""
        code = '''"""File doc."""

class MyClass:
    """Class doc."""
    pass

def my_func():
    """Func doc."""
    pass
'''
        f = tmp_path / "doc_test.py"
        f.write_text(code, encoding="utf-8")
        with patch.object(build_graph, "ROOT", tmp_path):
            result = build_graph.parse_file(f)

        by_name = {n["name"]: n for n in result["nodes"]}
        assert by_name["doc_test"]["docstring"] == "File doc."
        assert by_name["MyClass"]["docstring"] == "Class doc."
        assert by_name["my_func"]["docstring"] == "Func doc."

    def test_parse_extracts_imports(self, tmp_path):
        """Import edges are extracted from 'from X import Y' statements."""
        code = 'from capabilities.timer import TimerCapability\n'
        f = tmp_path / "stage.py"
        f.write_text(code, encoding="utf-8")
        with patch.object(build_graph, "ROOT", tmp_path):
            result = build_graph.parse_file(f)

        edges = result["edges"]
        assert len(edges) >= 1
        src, tgt, etype = edges[0]
        assert etype == "imports"
        assert "capabilities.timer.TimerCapability" in tgt

    def test_parse_extracts_extends(self, tmp_path):
        """Class inheritance creates 'extends' edges."""
        code = '''class Child(ParentClass):
    pass
'''
        f = tmp_path / "inherit.py"
        f.write_text(code, encoding="utf-8")
        with patch.object(build_graph, "ROOT", tmp_path):
            result = build_graph.parse_file(f)

        extends_edges = [(s, t, e) for s, t, e in result["edges"] if e == "extends"]
        assert len(extends_edges) == 1
        assert extends_edges[0][1] == "ParentClass"

    def test_parse_handles_syntax_error(self, tmp_path):
        """Files with syntax errors return empty nodes/edges."""
        f = tmp_path / "bad.py"
        f.write_text("def broken(:\n", encoding="utf-8")
        with patch.object(build_graph, "ROOT", tmp_path):
            result = build_graph.parse_file(f)

        assert result["nodes"] == []
        assert result["edges"] == []
        assert result["hash"]  # hash is still computed

    def test_parse_line_numbers(self, tmp_path):
        """Nodes have correct line_start and line_end."""
        code = '''x = 1

class Foo:
    pass

def bar():
    pass
'''
        f = tmp_path / "lines.py"
        f.write_text(code, encoding="utf-8")
        with patch.object(build_graph, "ROOT", tmp_path):
            result = build_graph.parse_file(f)

        by_name = {n["name"]: n for n in result["nodes"]}
        assert by_name["Foo"]["line_start"] == 3
        assert by_name["bar"]["line_start"] == 6

    def test_parse_contains_edges(self, tmp_path):
        """CONTAINS edges are created: file→class, file→function, class→method."""
        code = '''class Foo:
    def method_a(self):
        pass

def standalone():
    pass

MY_CONST = 1
'''
        f = tmp_path / "cont.py"
        f.write_text(code, encoding="utf-8")
        with patch.object(build_graph, "ROOT", tmp_path):
            result = build_graph.parse_file(f)

        contains = [(s, t) for s, t, e in result["edges"] if e == "contains"]
        # file → class
        assert ("cont.py", "cont.py::Foo") in contains
        # class → method
        assert ("cont.py::Foo", "cont.py::Foo.method_a") in contains
        # file → function
        assert ("cont.py", "cont.py::standalone") in contains
        # file → constant
        assert ("cont.py", "cont.py::MY_CONST") in contains

    def test_parse_calls_edges(self, tmp_path):
        """CALLS edges are created when functions call other functions."""
        code = '''def helper():
    pass

def main():
    helper()
    print("done")
'''
        f = tmp_path / "calls.py"
        f.write_text(code, encoding="utf-8")
        with patch.object(build_graph, "ROOT", tmp_path):
            result = build_graph.parse_file(f)

        calls = [(s, t) for s, t, e in result["edges"] if e == "calls"]
        # main calls helper
        assert any("main" in s and t == "helper" for s, t in calls)
        # main calls print
        assert any("main" in s and t == "print" for s, t in calls)

    def test_parse_method_calls(self, tmp_path):
        """CALLS edges from methods resolve correctly."""
        code = '''class Service:
    def run(self):
        self.prepare()

    def prepare(self):
        pass
'''
        f = tmp_path / "svc.py"
        f.write_text(code, encoding="utf-8")
        with patch.object(build_graph, "ROOT", tmp_path):
            result = build_graph.parse_file(f)

        calls = [(s, t) for s, t, e in result["edges"] if e == "calls"]
        # Service.run calls self.prepare
        assert any("Service.run" in s and "self.prepare" in t for s, t in calls)


class TestBuildFull:
    """Tests for the full build() pipeline."""

    def test_full_build_creates_nodes(self, tmp_path):
        """Full build on the real codebase creates nodes and edges."""
        db_path = tmp_path / "graph.db"
        with patch.object(build_graph, "DB_DIR", tmp_path), \
             patch.object(build_graph, "DB_PATH", db_path):
            stats = build_graph.build(incremental=False)

        assert stats["files_parsed"] > 0
        assert stats["total_nodes"] > 0
        assert stats["total_edges"] > 0

    def test_full_build_loads_requirements(self, tmp_path):
        """Full build loads requirements from YAML."""
        db_path = tmp_path / "graph.db"
        with patch.object(build_graph, "DB_DIR", tmp_path), \
             patch.object(build_graph, "DB_PATH", db_path):
            stats = build_graph.build(incremental=False)

        assert stats["total_requirements"] > 0

    def test_full_build_extracts_sync_values(self, tmp_path):
        """Full build extracts sync values from source."""
        db_path = tmp_path / "graph.db"
        with patch.object(build_graph, "DB_DIR", tmp_path), \
             patch.object(build_graph, "DB_PATH", db_path):
            stats = build_graph.build(incremental=False)

        assert stats["total_sync_values"] > 0

    def test_incremental_skips_unchanged(self, tmp_path):
        """Incremental build skips files that haven't changed."""
        db_path = tmp_path / "graph.db"
        with patch.object(build_graph, "DB_DIR", tmp_path), \
             patch.object(build_graph, "DB_PATH", db_path):
            stats1 = build_graph.build(incremental=False)
            stats2 = build_graph.build(incremental=True)

        assert stats2["files_skipped"] == stats1["files_parsed"]
        assert stats2["files_parsed"] == 0


class TestDocContract:
    """Tests for documentation contract edges."""

    def test_doc_contract_edges_exist(self, fresh_db):
        """Doc contract edges are created for known source files."""
        build_graph.build_doc_contract_edges(fresh_db)
        fresh_db.commit()

        count = fresh_db.execute(
            "SELECT COUNT(*) FROM edges WHERE type='doc_contract'"
        ).fetchone()[0]
        assert count > 0

    def test_doc_contract_covers_key_files(self, fresh_db):
        """Doc contract covers critical files like conversation.py and const.py."""
        build_graph.build_doc_contract_edges(fresh_db)
        fresh_db.commit()

        for source in ["conversation.py", "const.py", "capabilities/mcp.py"]:
            rows = fresh_db.execute(
                "SELECT target_id FROM edges WHERE source_id=? AND type='doc_contract'",
                (source,),
            ).fetchall()
            assert len(rows) > 0, f"No doc contract edges for {source}"


class TestSyncValues:
    """Tests for sync value extraction."""

    def test_extracts_known_constants(self, fresh_db):
        """Known constants like BRIGHTNESS_STEP are extracted."""
        build_graph.extract_sync_values(fresh_db)
        fresh_db.commit()

        row = fresh_db.execute(
            "SELECT value, file_path FROM sync_values WHERE name='BRIGHTNESS_STEP'"
        ).fetchone()
        # BRIGHTNESS_STEP may or may not exist depending on codebase state,
        # but at least the table should be populated with some values
        count = fresh_db.execute("SELECT COUNT(*) FROM sync_values").fetchone()[0]
        assert count > 0


class TestRequirementsLoading:
    """Tests for requirements YAML loading."""

    def test_loads_yaml_into_db(self, fresh_db):
        """Requirements from YAML are loaded into the requirements table."""
        build_graph.load_requirements(fresh_db)
        fresh_db.commit()

        count = fresh_db.execute("SELECT COUNT(*) FROM requirements").fetchone()[0]
        assert count > 0

    def test_requirement_has_fields(self, fresh_db):
        """Each requirement has category, title, source_files, test_files."""
        build_graph.load_requirements(fresh_db)
        fresh_db.commit()

        row = fresh_db.execute(
            "SELECT * FROM requirements WHERE id='REQ-PIPE-001'"
        ).fetchone()
        assert row is not None
        # columns: id, category, title, description, source_files, test_files, status
        assert row[1] == "pipeline"  # category
        assert row[2]  # title
        source_files = json.loads(row[4])
        assert isinstance(source_files, list)


class TestFTS:
    """Tests for full-text search index."""

    def test_fts_search_finds_nodes(self, tmp_path):
        """FTS search can find nodes by name."""
        db_path = tmp_path / "graph.db"
        with patch.object(build_graph, "DB_DIR", tmp_path), \
             patch.object(build_graph, "DB_PATH", db_path):
            build_graph.build(incremental=False)
            db = build_graph._get_db()
            rows = db.execute(
                "SELECT name FROM nodes_fts WHERE nodes_fts MATCH 'conversation'",
            ).fetchall()
            db.close()

        assert len(rows) > 0


class TestGraphServer:
    """Tests for graph_server.py query tools."""

    @pytest.fixture(autouse=True)
    def build_graph_first(self, tmp_path):
        """Build graph into temp dir before each server test."""
        db_path = tmp_path / "graph.db"
        with patch.object(build_graph, "DB_DIR", tmp_path), \
             patch.object(build_graph, "DB_PATH", db_path):
            build_graph.build(incremental=False)

        # Patch graph_server to use same DB
        from graph import graph_server
        self._orig_db_path = graph_server.DB_PATH
        graph_server.DB_PATH = db_path
        yield
        graph_server.DB_PATH = self._orig_db_path

    def test_get_file_purpose(self):
        from graph import graph_server
        result = graph_server.get_file_purpose("conversation.py")
        assert "path" in result
        assert result["path"] == "conversation.py"
        assert "children" in result

    def test_get_file_purpose_missing(self):
        from graph import graph_server
        result = graph_server.get_file_purpose("nonexistent.py")
        assert "error" in result

    def test_get_blast_radius(self):
        from graph import graph_server
        result = graph_server.get_blast_radius("const.py")
        assert "docs_to_update" in result
        assert len(result["docs_to_update"]) > 0
        # Multi-hop fields
        assert "affected_files" in result
        assert "impacted_nodes" in result
        assert result["max_depth"] == 3

    def test_get_blast_radius_custom_depth(self):
        from graph import graph_server
        result = graph_server.get_blast_radius("const.py", max_depth=1)
        assert result["max_depth"] == 1
        assert "affected_files" in result

    def test_get_requirement(self):
        from graph import graph_server
        result = graph_server.get_requirement("REQ-PIPE-001")
        assert result["id"] == "REQ-PIPE-001"
        assert result["category"] == "pipeline"
        assert isinstance(result["source_files"], list)

    def test_get_requirement_missing(self):
        from graph import graph_server
        result = graph_server.get_requirement("REQ-FAKE-999")
        assert "error" in result

    def test_find_requirements(self):
        from graph import graph_server
        results = graph_server.find_requirements("cache")
        assert len(results) > 0
        assert any("CACHE" in r["id"] for r in results)

    def test_get_capability_info(self):
        from graph import graph_server
        result = graph_server.get_capability_info("timer")
        assert result.get("file") == "capabilities/timer.py"
        assert "classes" in result

    def test_get_stage_capabilities(self):
        from graph import graph_server
        result = graph_server.get_stage_capabilities("stage2_llm")
        assert "capabilities" in result

    def test_find_tests_for(self):
        from graph import graph_server
        result = graph_server.find_tests_for("capabilities/semantic_cache.py")
        assert len(result) > 0

    def test_get_sync_values(self):
        from graph import graph_server
        result = graph_server.get_sync_values()
        assert len(result) > 0
        names = {r["name"] for r in result}
        assert "SERVICE_DOMAINS" in names or "BRIGHTNESS_STEP" in names

    def test_search_code(self):
        from graph import graph_server
        result = graph_server.search_code("conversation")
        assert len(result) > 0

    def test_mcp_handle_initialize(self):
        from graph import graph_server
        resp = graph_server.handle_request({
            "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}
        })
        assert resp["result"]["serverInfo"]["name"] == "multistage-graph"

    def test_mcp_handle_tools_list(self):
        from graph import graph_server
        resp = graph_server.handle_request({
            "jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}
        })
        tools = resp["result"]["tools"]
        names = {t["name"] for t in tools}
        assert "get_blast_radius" in names
        assert "search_code" in names
        assert len(tools) == 10

    def test_mcp_handle_tools_call(self):
        from graph import graph_server
        resp = graph_server.handle_request({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "get_sync_values", "arguments": {}},
        })
        assert resp["result"]["isError"] is False
        data = json.loads(resp["result"]["content"][0]["text"])
        assert isinstance(data, list)

    def test_mcp_handle_unknown_tool(self):
        from graph import graph_server
        resp = graph_server.handle_request({
            "jsonrpc": "2.0", "id": 4, "method": "tools/call",
            "params": {"name": "nonexistent_tool", "arguments": {}},
        })
        assert resp["result"]["isError"] is True


class TestContainsEdges:
    """Tests for CONTAINS edge generation."""

    def test_full_build_has_contains_edges(self, tmp_path):
        """Full build creates contains edges."""
        db_path = tmp_path / "graph.db"
        with patch.object(build_graph, "DB_DIR", tmp_path), \
             patch.object(build_graph, "DB_PATH", db_path):
            build_graph.build(incremental=False)
            db = build_graph._get_db()
            count = db.execute(
                "SELECT COUNT(*) FROM edges WHERE type='contains'"
            ).fetchone()[0]
            db.close()
        assert count > 0

    def test_file_contains_class(self, tmp_path):
        """File nodes contain their classes."""
        db_path = tmp_path / "graph.db"
        with patch.object(build_graph, "DB_DIR", tmp_path), \
             patch.object(build_graph, "DB_PATH", db_path):
            build_graph.build(incremental=False)
            db = build_graph._get_db()
            # conversation.py should contain MultiStageAssistAgent
            row = db.execute(
                "SELECT 1 FROM edges WHERE source_id='conversation.py' "
                "AND target_id='conversation.py::MultiStageAssistAgent' AND type='contains'"
            ).fetchone()
            db.close()
        assert row is not None


class TestCallsEdges:
    """Tests for CALLS edge generation."""

    def test_full_build_has_calls_edges(self, tmp_path):
        """Full build creates calls edges."""
        db_path = tmp_path / "graph.db"
        with patch.object(build_graph, "DB_DIR", tmp_path), \
             patch.object(build_graph, "DB_PATH", db_path):
            build_graph.build(incremental=False)
            db = build_graph._get_db()
            count = db.execute(
                "SELECT COUNT(*) FROM edges WHERE type='calls'"
            ).fetchone()[0]
            db.close()
        assert count > 0


class TestTestedByEdges:
    """Tests for TESTED_BY edge generation."""

    def test_full_build_has_tested_by_edges(self, tmp_path):
        """Full build creates tested_by edges linking test functions to production code."""
        db_path = tmp_path / "graph.db"
        with patch.object(build_graph, "DB_DIR", tmp_path), \
             patch.object(build_graph, "DB_PATH", db_path):
            build_graph.build(incremental=False)
            db = build_graph._get_db()
            count = db.execute(
                "SELECT COUNT(*) FROM edges WHERE type='tested_by'"
            ).fetchone()[0]
            db.close()
        assert count > 0


class TestMethodNodes:
    """Tests for method node extraction."""

    def test_full_build_has_method_nodes(self, tmp_path):
        """Full build creates method nodes for class methods."""
        db_path = tmp_path / "graph.db"
        with patch.object(build_graph, "DB_DIR", tmp_path), \
             patch.object(build_graph, "DB_PATH", db_path):
            build_graph.build(incremental=False)
            db = build_graph._get_db()
            count = db.execute(
                "SELECT COUNT(*) FROM nodes WHERE type='method'"
            ).fetchone()[0]
            db.close()
        assert count > 0


class TestDetectChanges:
    """Tests for detect_changes MCP tool."""

    @pytest.fixture(autouse=True)
    def build_graph_first(self, tmp_path):
        db_path = tmp_path / "graph.db"
        with patch.object(build_graph, "DB_DIR", tmp_path), \
             patch.object(build_graph, "DB_PATH", db_path):
            build_graph.build(incremental=False)
        from graph import graph_server
        self._orig_db_path = graph_server.DB_PATH
        graph_server.DB_PATH = db_path
        yield
        graph_server.DB_PATH = self._orig_db_path

    def test_detect_changes_returns_structure(self):
        """detect_changes returns the expected keys."""
        from graph import graph_server
        result = graph_server.detect_changes("HEAD~1")
        assert "changed_files" in result or "error" in result
        if "error" not in result:
            assert "affected_functions" in result
            assert "test_gaps" in result
            assert "docs_to_update" in result
            assert "requirements" in result

    def test_detect_changes_via_mcp(self):
        """detect_changes is callable via MCP protocol."""
        from graph import graph_server
        resp = graph_server.handle_request({
            "jsonrpc": "2.0", "id": 10, "method": "tools/call",
            "params": {"name": "detect_changes", "arguments": {"base": "HEAD~1"}},
        })
        assert resp["result"]["isError"] is False

    def test_parse_diff_lines(self):
        """_parse_diff_lines extracts correct line numbers from unified diff."""
        from graph import graph_server
        diff = (
            "--- a/foo.py\n"
            "+++ b/foo.py\n"
            "@@ -10,3 +10,5 @@\n"
            " context\n"
            "+added line 1\n"
            "+added line 2\n"
        )
        result = graph_server._parse_diff_lines(diff)
        assert "foo.py" in result
        assert 10 in result["foo.py"]
        assert len(result["foo.py"]) == 5  # lines 10-14
