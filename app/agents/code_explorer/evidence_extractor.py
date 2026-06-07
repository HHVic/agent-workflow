"""Extract located business evidence from CodeGraph MCP text results."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Literal

from app.agents.code_explorer.exploration_state import Evidence

_PATH = re.compile(
    r"(?P<path>(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_$.-]+\."
    r"(?:java|kt|py|js|jsx|ts|tsx|proto|xml|sql|yaml|yml)(?::\d+)?)"
)
_LOCATED_SYMBOL = re.compile(
    r"^-\s+(?P<symbol>[A-Za-z_$][A-Za-z0-9_.$<>-]*)"
    r"(?:\s+\((?P<kind>[^)]+)\))?\s+-\s+(?P<path>.+?\.\w+(?::\d+)?)\s*$",
    re.MULTILINE,
)
_HEADING_SYMBOL = re.compile(
    r"^##+\s+(?P<symbol>[A-Za-z_$][A-Za-z0-9_.$<>-]*)"
    r"(?:\s+\((?P<kind>class|method|interface|enum|route|field|function)\))?\s*$",
    re.MULTILINE,
)
_LOCATION = re.compile(r"^\*\*Location:\*\*\s+(?P<path>.+?\.\w+(?::\d+)?)\s*$", re.MULTILINE)
_CALLERS = re.compile(r"^##+\s+Callers of (?P<symbol>.+?)(?:\s+\(\d+ found\))?\s*$", re.MULTILINE)
_CALLEES = re.compile(r"^##+\s+Callees of (?P<symbol>.+?)(?:\s+\(\d+ found\))?\s*$", re.MULTILINE)
_ARROW = re.compile(r"\s*(?:->|→)\s*")
_SYMBOL_TOKEN = re.compile(r"^[A-Za-z_$][A-Za-z0-9_.$<>]*(?:\([^)]*\))?$")
_PERSISTENCE_HINT = re.compile(
    r"\b(?:Repository|Repo|Mapper|Dao|DAO|select|insert|update|save|persist|delete)\b",
    re.IGNORECASE,
)
_PERSISTENCE_CALL = re.compile(
    r"\b[A-Za-z_$][A-Za-z0-9_.$]*(?:Repository|Repo|Dao|DAO|Mapper)"
    r"\.(?:save|insert|update|batchUpdate|onlyUpdate[A-Za-z0-9_$]*|"
    r"select|find|query|delete|persist)[A-Za-z0-9_$]*\s*\(",
    re.IGNORECASE,
)
_SQL_OPERATION = re.compile(r"<(?:insert|update|select|delete)\b|\b(?:insert\s+into|update|delete\s+from|select\b.+\bfrom)\b", re.IGNORECASE)
_TABLE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+){2,}\b")
_EVENT = re.compile(r"\b(?:[A-Za-z0-9_]*Event|[A-Za-z0-9_]*Listener|publishEvent|onApplicationEvent|consumer)\b")
_STRONG_EVENT = re.compile(
    r"\b(?:publishEvent|onApplicationEvent)\s*\(|@EventListener\b|"
    r"\bApplicationListener\s*<|\b[A-Za-z0-9_]*Consumer\b",
    re.IGNORECASE,
)
_STATUS = re.compile(r"\b(?:[A-Za-z0-9_]*Status|state|status)\b", re.IGNORECASE)
_STATUS_WRITE = re.compile(
    r"\b(?:set|update|change|onlyUpdate|mark|transition)[A-Za-z0-9_$]*(?:Status|State)"
    r"[A-Za-z0-9_$]*\s*\(",
    re.IGNORECASE,
)
_STATUS_ENUM = re.compile(
    r"\b(?=[A-Z0-9_]*_)[A-Z0-9_]*"
    r"(?:STATUS|STATE|READY|WITHDRAW|PENDING|SUCCESS|FAIL|DONE|FROZEN|"
    r"RELEASE|PAID|APPROV|REJECT|PROCESS|COMPLETE|CANCEL)"
    r"[A-Z0-9_]*\b"
)
_METHOD_DECL = re.compile(
    r"\b(?P<method>[A-Za-z_$][A-Za-z0-9_$]*)\s*\([^;{}]*\)\s*\{"
)
_BUSINESS_METHOD = re.compile(
    r"(?:calculate|compute|process|handle|execute|create|build|assemble|"
    r"transform|convert|match|dispatch|consume|publish|trigger)",
    re.IGNORECASE,
)
_GENERIC_SYMBOL = re.compile(
    r"(?:^|[.$])(?:Context|PageResult|Response|AbstractController|"
    r"[A-Za-z0-9_$]*(?:Dto|DTO|Vo|VO|Bo|BO|Po|PO)|"
    r"[A-Za-z0-9_$]*Mapper)(?:[.$]|$)|"
    r"(?:boToVo|toBo|toVo|toPo|setId|getId)(?:\(|$)",
    re.IGNORECASE,
)
_MAPPER_CONVERSION = re.compile(
    r"\b[A-Za-z0-9_.$]*Mapper(?:\.MAPPER)?\.(?:boToVo|toBo|toVo|toPo)\s*\(",
    re.IGNORECASE,
)
_WRAPPER_NOISE = re.compile(
    r"\b(?:Response(?:\.SUCCESS)?|PageResult|Context|AbstractController)\b",
    re.IGNORECASE,
)


def extract_evidence_from_tool_result(
    tool_name: str,
    args: dict[str, Any],
    result_text: str,
) -> list[Evidence]:
    """Return confirmed evidence only when the result contains a useful locator."""

    if not result_text.strip():
        return []

    evidence: list[Evidence] = []
    paths = _PATH.findall(result_text)
    unique_paths = list(dict.fromkeys(paths))
    default_path = unique_paths[0] if len(unique_paths) == 1 else None

    evidence.extend(_file_path_evidence(tool_name, paths))
    evidence.extend(_located_symbol_evidence(tool_name, result_text))
    evidence.extend(_heading_symbol_evidence(tool_name, result_text, default_path))
    evidence.extend(_caller_callee_evidence(tool_name, args, result_text))
    evidence.extend(_arrow_evidence(tool_name, result_text, default_path))
    evidence.extend(_semantic_evidence(tool_name, result_text, default_path))
    return _deduplicate(evidence)


def _file_path_evidence(tool_name: str, paths: list[str]) -> list[Evidence]:
    return [
        _evidence(
            claim=f"Found file/path {path}",
            source_type="file_path",
            summary=path,
            file_path=path,
            tool_name=tool_name,
            reason="A file or directory path is a navigation lead, not business evidence.",
        )
        for path in dict.fromkeys(paths)
    ]


def _located_symbol_evidence(tool_name: str, text: str) -> list[Evidence]:
    items: list[Evidence] = []
    for match in _LOCATED_SYMBOL.finditer(text):
        symbol = match.group("symbol")
        path = match.group("path")
        items.append(
            _evidence(
                claim=f"Found symbol {symbol} at {path}",
                source_type="tool_result",
                summary=match.group(0),
                file_path=path,
                symbol=symbol,
                tool_name=tool_name,
                reason="A located symbol is a weak lead until its business behavior is inspected.",
            )
        )
    return items


def _heading_symbol_evidence(
    tool_name: str,
    text: str,
    default_path: str | None,
) -> list[Evidence]:
    items: list[Evidence] = []
    location = _LOCATION.search(text)
    if not location:
        return []
    path = location.group("path")
    for match in _HEADING_SYMBOL.finditer(text):
        symbol = match.group("symbol")
        if symbol in {"Callers", "Callees"}:
            continue
        items.append(
            _evidence(
                claim=f"Found symbol {symbol}" + (f" at {path}" if path else ""),
                source_type="tool_result",
                summary=match.group(0),
                file_path=path,
                symbol=symbol,
                tool_name=tool_name,
                reason="A symbol heading is a weak lead until its body or relationships provide stronger evidence.",
            )
        )
    return items


def _caller_callee_evidence(
    tool_name: str,
    args: dict[str, Any],
    text: str,
) -> list[Evidence]:
    items: list[Evidence] = []
    caller_header = _CALLERS.search(text)
    callee_header = _CALLEES.search(text)
    target = str(args.get("symbol") or "")
    if caller_header:
        target = caller_header.group("symbol").strip()
        for match in _LOCATED_SYMBOL.finditer(text):
            if (match.group("kind") or "").casefold() == "import":
                continue
            source = match.group("symbol")
            items.append(_call_edge(tool_name, source, target, match.group("path")))
    if callee_header:
        target = callee_header.group("symbol").strip()
        for match in _LOCATED_SYMBOL.finditer(text):
            if (match.group("kind") or "").casefold() == "import":
                continue
            destination = match.group("symbol")
            items.append(_call_edge(tool_name, target, destination, match.group("path")))
    return items


def _arrow_evidence(
    tool_name: str,
    text: str,
    default_path: str | None,
) -> list[Evidence]:
    items: list[Evidence] = []
    for line in text.splitlines():
        if "->" not in line and "→" not in line:
            continue
        parts = [part.strip(" -*`。.;；") for part in _ARROW.split(line)]
        parts = [part for part in parts if _SYMBOL_TOKEN.fullmatch(part)]
        for source, destination in zip(parts, parts[1:]):
            items.append(_call_edge(tool_name, source, destination, default_path))
    return items


def _semantic_evidence(
    tool_name: str,
    text: str,
    default_path: str | None,
) -> list[Evidence]:
    items: list[Evidence] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        located_symbol = _first_symbol(stripped)
        semantic_text = _PATH.sub("", stripped)
        noise_reason = _noise_reason(stripped)
        if noise_reason:
            items.append(
                _evidence(
                    claim=f"Downgraded generic code evidence: {stripped}",
                    source_type="semantic_noise",
                    summary=stripped,
                    file_path=default_path,
                    symbol=located_symbol,
                    tool_name=tool_name,
                    strength="noise",
                    reason=noise_reason,
                )
            )
            continue
        persistence_call = bool(_PERSISTENCE_CALL.search(semantic_text))
        sql_operation = bool(_SQL_OPERATION.search(semantic_text))
        persistence_hint = bool(_PERSISTENCE_HINT.search(semantic_text))
        test_evidence = _is_test_code(default_path, stripped)
        if persistence_call or sql_operation or persistence_hint:
            is_strong_p = persistence_call or sql_operation
            strength_p, can_satisfy_p = _maybe_downgrade_for_test(
                strength="strong" if is_strong_p else "weak",
                can_satisfy=is_strong_p,
                file_path=default_path,
                text=stripped,
            )
            items.append(
                _evidence(
                    claim=f"Persistence evidence: {stripped}",
                    source_type="repository",
                    summary=stripped,
                    file_path=default_path,
                    symbol=located_symbol,
                    tool_name=tool_name,
                    strength=strength_p,
                    can_satisfy_stage_field=can_satisfy_p,
                    reason=(
                        "Matched an actual repository/DAO/Mapper read-write call or SQL operation."
                        if persistence_call or sql_operation
                        else "A persistence-related name without an actual read-write call is only a weak lead."
                    ),
                )
            )
        for table in _TABLE.findall(stripped) if persistence_call or sql_operation or persistence_hint else ():
            items.append(
                _evidence(
                    claim=f"Found table-like persistence identifier {table}",
                    source_type="table",
                    summary=stripped,
                    file_path=default_path,
                    symbol=table,
                    tool_name=tool_name,
                    strength="strong",
                    can_satisfy_stage_field=True,
                    reason="Matched a table-like identifier in an actual persistence operation.",
                )
            )
        if _EVENT.search(stripped):
            is_strong_event = bool(_STRONG_EVENT.search(stripped))
            strength_ev, can_satisfy_ev = _maybe_downgrade_for_test(
                strength="strong" if is_strong_event else "weak",
                can_satisfy=is_strong_event,
                file_path=default_path,
                text=stripped,
            )
            items.append(
                _evidence(
                    claim=f"Event publish/listener evidence: {stripped}",
                    source_type="event",
                    summary=stripped,
                    file_path=default_path,
                    symbol=located_symbol,
                    tool_name=tool_name,
                    strength=strength_ev,
                    can_satisfy_stage_field=can_satisfy_ev,
                    reason=(
                        "Matched an explicit event publish/listener/consumer operation."
                        if is_strong_event
                        else "An event-related name without an explicit publish or listener operation is only a weak lead."
                    ),
                )
            )
        status_enum = _STATUS_ENUM.search(stripped)
        if (_STATUS.search(stripped) or status_enum) and (
            "status" in stripped.casefold()
            or "state" in stripped.casefold()
            or "->" in stripped
            or "→" in stripped
            or status_enum
        ):
            is_strong_status = bool(_STATUS_WRITE.search(stripped)) or (
                bool(status_enum) and ("->" in stripped or "→" in stripped)
            )
            strength_st, can_satisfy_st = _maybe_downgrade_for_test(
                strength="strong" if is_strong_status else "weak",
                can_satisfy=is_strong_status,
                file_path=default_path,
                text=stripped,
            )
            items.append(
                _evidence(
                    claim=f"Status/state evidence: {stripped}",
                    source_type="status",
                    summary=stripped,
                    file_path=default_path,
                    symbol=located_symbol,
                    tool_name=tool_name,
                    strength=strength_st,
                    can_satisfy_stage_field=can_satisfy_st,
                    reason=(
                        "Matched an explicit status/state write or transition."
                        if is_strong_status
                        else "A status/state name without an explicit write or transition is only a weak lead."
                    ),
                )
            )
        method = _METHOD_DECL.search(stripped)
        if method and _BUSINESS_METHOD.search(method.group("method")):
            items.append(
                _evidence(
                    claim=f"Business method body evidence: {stripped}",
                    source_type="business_method",
                    summary=stripped,
                    file_path=default_path,
                    symbol=method.group("method"),
                    tool_name=tool_name,
                    strength="strong",
                    can_satisfy_stage_field=True,
                    reason="Matched a business-oriented method body declaration.",
                )
            )
    return items


def _call_edge(
    tool_name: str,
    source: str,
    destination: str,
    file_path: str | None,
) -> Evidence:
    symbol = f"{source} -> {destination}"
    generic_node = _generic_edge_node(source) or _generic_edge_node(destination)
    base_strength = "noise" if generic_node else "strong"
    base_can_satisfy = not generic_node
    strength, can_satisfy = _maybe_downgrade_for_test(
        strength=base_strength,
        can_satisfy=base_can_satisfy,
        file_path=file_path,
        text=symbol,
    )
    return _evidence(
        claim=f"{source} calls {destination}",
        source_type="call_edge",
        summary=symbol,
        file_path=file_path,
        symbol=symbol,
        tool_name=tool_name,
        strength=strength,
        can_satisfy_stage_edge=can_satisfy,
        reason=(
            "A generic wrapper, context, mapping, or transport node cannot satisfy a business-stage edge."
            if generic_node
            else "Matched a direct call relationship between non-generic symbols."
        ),
    )


def _evidence(
    *,
    claim: str,
    source_type: str,
    summary: str,
    tool_name: str,
    file_path: str | None = None,
    symbol: str | None = None,
    strength: Literal["strong", "weak", "noise"] = "weak",
    can_satisfy_stage_field: bool = False,
    can_satisfy_stage_edge: bool = False,
    reason: str = "No explicit strong-evidence rule matched.",
) -> Evidence:
    digest = hashlib.sha1(
        f"{tool_name}|{source_type}|{file_path}|{symbol}|{summary}".encode()
    ).hexdigest()[:12]
    return Evidence(
        id=f"e-{digest}",
        claim=claim,
        source_type=source_type,
        summary=summary,
        confidence="confirmed",
        file_path=file_path,
        symbol=symbol,
        strength=strength,
        can_satisfy_stage_field=can_satisfy_stage_field,
        can_satisfy_stage_edge=can_satisfy_stage_edge,
        reason=reason,
    )


def _first_symbol(text: str) -> str | None:
    match = re.search(r"\b[A-Za-z_$][A-Za-z0-9_.$]*(?:\([^)]*\))?\b", text)
    return match.group(0) if match else None


def _noise_reason(text: str) -> str | None:
    if text.lstrip().startswith("import "):
        return "Imports only prove a dependency reference; they do not prove a business stage field or edge."
    if _MAPPER_CONVERSION.search(text):
        return "Mapper conversion is structural plumbing and cannot satisfy a business stage field or edge."
    if _WRAPPER_NOISE.search(text):
        return "Response wrappers, PageResult, Context, and AbstractController are structural plumbing."
    return None


def _generic_edge_node(symbol: str) -> bool:
    return bool(_GENERIC_SYMBOL.search(symbol))


def _is_test_code(file_path: str | None, text: str) -> bool:
    """Return True if the evidence likely comes from test code."""

    if file_path:
        fp = file_path.casefold()
        if "/src/test/" in fp or "/test/" in fp:
            return True
        # Check if filename contains "test" (e.g., WidgetJobTest.java, test_service.py)
        parts = fp.replace("\\", "/").split("/")
        filename = parts[-1].split(":")[0]  # strip line number
        if "test" in filename and "." in filename:
            return True

    # Check test-related keywords in text content.
    if not text:
        return False
    text_lower = text.casefold()
    test_markers = ("should_", "when_", "then_", "mockito", "assertthat", "assertequals")
    if any(marker in text_lower for marker in test_markers):
        return True
    if "when(" in text_lower and ("thenreturn" in text_lower or "thenReturn" in text):
        return True
    if "verify(" in text_lower:
        return True

    return False


def _maybe_downgrade_for_test(
    *,
    strength: str,
    can_satisfy: bool,
    file_path: str | None,
    text: str,
) -> tuple[str, bool]:
    """Downgrade to weak if the evidence comes from test code."""

    if strength != "strong":
        return strength, can_satisfy
    if _is_test_code(file_path, text):
        return "weak", False
    return strength, can_satisfy


def _deduplicate(items: list[Evidence]) -> list[Evidence]:
    unique: dict[str, Evidence] = {}
    for item in items:
        unique.setdefault(item.id, item)
    return list(unique.values())
