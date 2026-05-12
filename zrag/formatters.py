"""
Output formatting for zrag.
"""

import json
import csv
from io import StringIO
from typing import List, Dict, Any, Optional
from enum import Enum
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom

from rich.console import Console
from rich.table import Table
from rich.text import Text


class OutputFormat(str, Enum):
    JSON = "json"
    CSV = "csv"
    XML = "xml"
    MARKDOWN = "markdown"
    CLI = "cli"


class OutputFormatter:
    """Flexible output formatting for search results."""
    
    def __init__(self):
        self.console = Console()
    
    def format(self, results: List[Dict[str, Any]], format: OutputFormat = OutputFormat.CLI, **kwargs) -> str:
        if format == OutputFormat.JSON:
            return self._format_json(results, **kwargs)
        elif format == OutputFormat.CSV:
            return self._format_csv(results, **kwargs)
        elif format == OutputFormat.XML:
            return self._format_xml(results, **kwargs)
        elif format == OutputFormat.MARKDOWN:
            return self._format_markdown(results, **kwargs)
        elif format == OutputFormat.CLI:
            return self._format_cli(results, **kwargs)
        raise ValueError(f"Unknown format: {format}")
    
    def _format_json(self, results: List[Dict[str, Any]], **kwargs) -> str:
        return json.dumps(results, indent=kwargs.get("indent", 2), ensure_ascii=False)
    
    def _format_csv(self, results: List[Dict[str, Any]], **kwargs) -> str:
        if not results:
            return ""
        fields = kwargs.get("fields", ["id", "score", "source", "context", "content"])
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for result in results:
            writer.writerow(result)
        return output.getvalue()
    
    def _format_xml(self, results: List[Dict[str, Any]], **kwargs) -> str:
        root = Element("results")
        for result in results:
            doc_elem = SubElement(root, "document")
            doc_elem.set("id", str(result.get("id", "")))
            score = result.get("score")
            if score is not None:
                doc_elem.set("score", str(score))
            fields_elem = SubElement(doc_elem, "fields")
            for key, value in result.items():
                if key not in["id", "score"]:
                    field_elem = SubElement(fields_elem, "field")
                    field_elem.set("name", key)
                    field_elem.text = str(value)
        xml_str = tostring(root, encoding='unicode')
        if kwargs.get("pretty", True):
            dom = minidom.parseString(xml_str)
            return dom.toprettyxml(indent="  ")
        return xml_str
    
    def _format_markdown(self, results: List[Dict[str, Any]], **kwargs) -> str:
        if not results:
            return "No results found."
        lines = ["# Search Results\n"]
        max_preview = kwargs.get("max_preview", 200)
        for i, result in enumerate(results, 1):
            lines.append(f"## {i}. {result.get('id', '')}")
            if result.get("score") is not None:
                lines.append(f"**Score:** {result['score']:.4f}")
            lines.append(f"**Source:** `{result.get('source', 'unknown')}`")
            if result.get("context"):
                lines.append(f"**Context:** _{result['context']}_")
            content = result.get("content", "")
            preview = content[:max_preview].replace('\n', ' ')
            if len(content) > max_preview:
                preview += "..."
            lines.append(f"\n**Preview:** {preview}\n")
            lines.append("---\n")
        return "\n".join(lines)
    
    def _format_cli(self, results: List[Dict[str, Any]], **kwargs) -> str:
        if not results:
            return "No results found."
        table = Table(title="Search Results")
        table.add_column("#", style="cyan", width=4)
        table.add_column("Score", style="magenta", width=8)
        table.add_column("Source", style="blue", width=30)
        table.add_column("Context", style="yellow", width=25)
        table.add_column("Preview", style="white", width=50)
        
        for i, result in enumerate(results, 1):
            score = f"{result['score']:.4f}" if result.get("score") is not None else "N/A"
            source = result.get("source", "unknown")
            context = result.get("context", "-")
            content = result.get("content", "")
            preview = content[:50].replace('\n', ' ') + ("..." if len(content) > 50 else "")
            table.add_row(str(i), score, source, context, preview)
        
        with self.console.capture() as capture:
            self.console.print(table)
        return capture.get()

    def format_get_result(self, result: Dict[str, Any], format: OutputFormat = OutputFormat.CLI, **kwargs) -> str:
        if format == OutputFormat.JSON:
            return json.dumps(result, indent=kwargs.get("indent", 2), ensure_ascii=False)
        elif format == OutputFormat.CLI:
            lines = [f"[bold]ID:[/bold] {result.get('id', '')}"]
            if "line_range" in result:
                lines.append(f"[bold]Line Range:[/bold] {result['line_range']}")
            lines.append(f"[bold]Source:[/bold] {result.get('source', 'unknown')}")
            if result.get("context"):
                lines.append(f"[bold]Context:[/bold] {result['context']}")
            lines.append(f"[bold]Type:[/bold] {result.get('file_type', 'unknown')}")
            if result.get("content"):
                lines.append(f"\n[bold]Content:[/bold]\n{result['content']}")
            return "\n".join(lines)
        return self.format([result], format, **kwargs)
