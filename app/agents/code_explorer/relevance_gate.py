"""Pure-function relevance gate for candidate filtering."""

from __future__ import annotations

import re


# Words always filtered out — structural / boilerplate / infra.
# These NEVER contribute relevance in any task.
ALWAYS_GENERIC: set[str] = {
    # Architecture / pattern suffixes
    "service", "impl", "controller", "repository", "mapper",
    "bo", "po", "vo", "dto",
    "util", "helper", "config", "factory",
    # Common CRUD / list ops
    "get", "set", "list", "page",
    # Manager / client / facade patterns
    "manager", "client", "facade", "delegate",
}

# Words that are structural in SOME domains but may be core domain words
# in others (e.g. "order" for order-refund, "event" for event-consumption).
# They are filtered from confirmed_symbols/confirmed_paths and scoring,
# but can contribute to relevance if the goal explicitly supports them.
CONTEXTUAL_GENERIC: set[str] = {
    # Scheduling / infra
    "job", "task", "batch",
    # Domain nouns that may be core
    "order", "event", "status",
    "document", "doc", "video",
    "applet", "season",
    # Operations
    "query", "condition", "process", "handle",
    "create", "update",
    # More architectural suffixes
    "processor", "handler",
}

# Chinese common words to filter
_GENERIC_CN: set[str] = {
    "的", "了", "在", "是", "我", "有", "和", "就",
    "不", "人", "都", "一", "一个", "上", "也", "很",
    "到", "说", "要", "去", "你", "会", "着", "没有",
    "看", "好", "自己", "这",
}

_CONTEXTUAL_CN_MAP: dict[str, str] = {
    "order": "订单",
    "event": "事件",
    "status": "状态",
    "document": "文档",
    "doc": "文档",
    "video": "视频",
    "job": "任务",
    "task": "任务",
    "process": "流程",
    "handle": "处理",
    "create": "创建",
    "update": "更新",
    "query": "查询",
    "condition": "条件",
    "applet": "小程序",
    "season": "季节",
    "batch": "批量",
    "processor": "处理器",
    "handler": "处理器",
}

_PATH_GENERIC: set[str] = {
    "src", "main", "test", "com", "cn", "org", "io",
    "resources", "spring", "web", "model", "entity",
    "constant", "exception", "java", "kt", "py", "ts", "js",
    "yml", "yaml", "xml", "json", "biz", "business", "domain",
}
_PATH_DOMAIN_MARKERS: set[str] = {"biz", "business", "domain"}

# Test-like prefixes
_TEST_PATTERN = re.compile(r"^(should_|when_|then_|test_|given_|mock_)", re.IGNORECASE)


def _tokenize(name: str) -> list[str]:
    """Split a symbol/path into lowercase tokens.

    Handles camelCase, PascalCase, snake_case, kebab-case, and file paths.
    """
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    s = re.sub(r"([A-Za-z])(\d+)", r"\1 \2", s)
    parts = re.split(r"[_\-./ ]+", s)
    return [p.lower() for p in parts if p]


def _extract_chinese(text: str) -> list[str]:
    """Extract meaningful Chinese keyword chunks from text."""
    chars = re.findall(r"[\u4e00-\u9fff]", text)
    if not chars:
        return []
    chunks: list[str] = []
    current: list[str] = []
    for ch in chars:
        current.append(ch)
        if len(current) == 4:
            chunks.append("".join(current))
            current = []
    if len(current) >= 2:
        chunks.append("".join(current))
    return chunks


def _goal_supports_goalword(goal: str, word: str) -> bool:
    """Check if a contextual word is supported by the goal text.

    Works for both English goals (tokenize match) and Chinese goals
    (raw substring match — e.g. '订单' contains 'order' concept).
    """
    if word in set(_tokenize(goal)):
        return True
    if word in goal.casefold():
        return True
    cn_equivalent = _CONTEXTUAL_CN_MAP.get(word, "")
    return bool(cn_equivalent and cn_equivalent in goal)


def extract_relevance_terms(
    goal: str,
    confirmed_symbols: list[str],
    confirmed_paths: list[str],
) -> set[str]:
    """Extract domain-relevant terms from goal, confirmed symbols and paths.

    ALWAYS_GENERIC words are always excluded.
    CONTEXTUAL_GENERIC words are excluded from confirmed_symbols/confirmed_paths,
    but ARE included from the goal (so they can contribute via goal overlap).
    Individual Chinese characters are also included for cross-language matching.
    """
    terms: set[str] = set()

    # From goal — keep non-generic terms and goal-supported contextual terms.
    terms.update(tok for tok in _tokenize(goal) if tok not in ALWAYS_GENERIC)
    terms.update(
        word for word in CONTEXTUAL_GENERIC if _goal_supports_goalword(goal, word)
    )
    cn = _extract_chinese(goal)
    terms.update(cn)
    # Individual Chinese characters for cross-language matching
    for ch in re.findall(r"[\u4e00-\u9fff]", goal):
        terms.add(ch)

    # From confirmed symbols — filter out all generic words
    for sym in confirmed_symbols:
        for tok in _tokenize(sym):
            if tok in ALWAYS_GENERIC:
                continue
            if tok in CONTEXTUAL_GENERIC:
                continue
            terms.add(tok)

    # From confirmed paths — prefer terms below common domain-layer markers so
    # repository prefixes do not become accidental one-token business evidence.
    terms.update(_extract_confirmed_path_terms(confirmed_paths))

    # Remove Chinese stop words only
    terms -= _GENERIC_CN

    return terms


def _filter_candidate_tokens(
    candidate_tokens: set[str],
) -> tuple[set[str], set[str]]:
    """Separate candidate tokens into non-generic and contextual-generic parts."""
    non_generic = candidate_tokens - ALWAYS_GENERIC - CONTEXTUAL_GENERIC
    contextual = candidate_tokens & CONTEXTUAL_GENERIC
    return non_generic, contextual


def _extract_confirmed_symbol_terms(confirmed_symbols: list[str]) -> set[str]:
    terms: set[str] = set()
    for symbol in confirmed_symbols:
        terms.update(
            token
            for token in _tokenize(symbol)
            if token not in ALWAYS_GENERIC and token not in CONTEXTUAL_GENERIC
        )
    return terms


def _extract_confirmed_path_terms(confirmed_paths: list[str]) -> set[str]:
    terms: set[str] = set()
    for path in confirmed_paths:
        tokens = _tokenize(path)
        marker_indexes = [
            index for index, token in enumerate(tokens) if token in _PATH_DOMAIN_MARKERS
        ]
        if marker_indexes:
            tokens = tokens[max(marker_indexes) + 1:]
        terms.update(
            token
            for token in tokens
            if token not in _PATH_GENERIC
            and token not in ALWAYS_GENERIC
            and token not in CONTEXTUAL_GENERIC
        )
    return terms


def is_relevant_candidate(
    candidate: str,
    goal: str,
    confirmed_symbols: list[str],
    confirmed_paths: list[str],
    connected_symbols: list[str] | None = None,
) -> bool:
    """Decide whether a candidate symbol is relevant to the exploration mainline.

    Returns True when the candidate passes at least one relevance criterion.
    """
    connected_symbols = connected_symbols or []
    goal_terms = extract_relevance_terms(goal, confirmed_symbols, confirmed_paths)

    # 1. Test symbols are default irrelevant
    if _TEST_PATTERN.match(candidate):
        return False

    # 2. If connected_symbols, direct pass
    if candidate in connected_symbols:
        return True

    candidate_tokens = set(_tokenize(candidate))
    if not candidate_tokens:
        return False

    non_generic, contextual = _filter_candidate_tokens(candidate_tokens)
    if not non_generic:
        return False

    # 3. Pass: candidate contains a substantive term supported by the goal.
    if non_generic & goal_terms:
        return True

    # 4. Pass: contextual words contribute only when the goal activates them.
    if contextual & goal_terms:
        return True

    # 5. Pass: candidate overlaps with confirmed symbol domain terms.
    if non_generic & _extract_confirmed_symbol_terms(confirmed_symbols):
        return True

    # 6. Pass: candidate shares a domain-layer path term.
    if non_generic & _extract_confirmed_path_terms(confirmed_paths):
        return True

    return False


def rank_relevant_candidates(
    candidates: list[str],
    goal: str,
    confirmed_symbols: list[str],
    confirmed_paths: list[str],
    connected_symbols: list[str] | None = None,
    limit: int = 10,
) -> list[str]:
    """Filter and rank candidate symbols by relevance score.

    Returns at most `limit` candidates that pass the relevance gate,
    sorted by descending relevance score.
    """
    connected_symbols = connected_symbols or []
    goal_terms = extract_relevance_terms(goal, confirmed_symbols, confirmed_paths)

    all_confirmed_tokens = _extract_confirmed_symbol_terms(confirmed_symbols)
    confirmed_domain_terms = _extract_confirmed_path_terms(confirmed_paths)

    scored: list[tuple[int, str]] = []
    for c in candidates:
        if not is_relevant_candidate(c, goal, confirmed_symbols, confirmed_paths, connected_symbols):
            continue
        ct = set(_tokenize(c))
        non_generic, contextual = _filter_candidate_tokens(ct)
        score = 0
        # connected_symbols member: highest signal
        if c in connected_symbols:
            score += 100
        # overlap with confirmed_symbols (non-generic)
        score += len(non_generic & all_confirmed_tokens) * 10
        # overlap with confirmed_paths domain
        score += len(non_generic & confirmed_domain_terms) * 5
        # overlap with goal terms (includes contextual generics from goal)
        score += len(non_generic & goal_terms) * 3
        score += len(contextual & goal_terms) * 3
        scored.append((-score, c))  # negative for ascending sort = descending

    scored.sort()
    return [c for _, c in scored[:limit]]
