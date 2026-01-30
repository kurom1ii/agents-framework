"""Content-Based Router implementation.

Module này triển khai ContentRouter - router định tuyến dựa trên
phân tích nội dung tin nhắn bao gồm keywords, hashtags, và commands.

Ví dụ sử dụng:
    from agents_framework.routing.strategies.content import (
        ContentRouter,
        ContentAnalyzer,
        ContentHints,
    )

    # Tạo router với command và hashtag mappings
    router = ContentRouter(
        agents={"work": work_agent, "personal": personal_agent, "coder": code_agent},
        command_mapping={
            "/code": "coder",
            "/research": "researcher",
        },
        hashtag_mapping={
            "#work": "work",
            "#personal": "personal",
        },
        default_agent="general",
    )

    # Request với hashtag
    request = RoutingRequest(message="#work Schedule meeting tomorrow")
    agent_id = await router.route(request)  # -> "work"

    # Request với command
    request = RoutingRequest(message="/code Fix the bug in login.py")
    agent_id = await router.route(request)  # -> "coder"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional, Pattern

from ..base import RoutingRequest, RoutingResult

if TYPE_CHECKING:
    from ...agents.base import BaseAgent


@dataclass
class ContentHints:
    """Các gợi ý routing được trích xuất từ nội dung tin nhắn.

    Attributes:
        command: Command được phát hiện (ví dụ: /code, /research).
        hashtags: Danh sách các hashtags trong tin nhắn.
        keywords: Danh sách các keywords quan trọng được phát hiện.
        required_capabilities: Các capabilities cần thiết cho request này.
        intent: Ý định được suy luận từ nội dung (coding, research, chat, etc.).
        confidence: Độ tin cậy của việc phân tích (0.0-1.0).
        metadata: Metadata bổ sung từ quá trình phân tích.
    """

    command: Optional[str] = None
    hashtags: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    required_capabilities: list[str] = field(default_factory=list)
    intent: Optional[str] = None
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def has_routing_hints(self) -> bool:
        """Kiểm tra xem có routing hints nào được tìm thấy không.

        Returns:
            True nếu có ít nhất một hint (command, hashtag, hoặc capability).
        """
        return bool(
            self.command
            or self.hashtags
            or self.required_capabilities
            or self.intent
        )

    def get_primary_hashtag(self) -> Optional[str]:
        """Lấy hashtag đầu tiên (primary) nếu có.

        Returns:
            Hashtag đầu tiên hoặc None.
        """
        return self.hashtags[0] if self.hashtags else None


class ContentAnalyzer:
    """Bộ phân tích nội dung tin nhắn để trích xuất routing hints.

    ContentAnalyzer sử dụng các patterns để phát hiện commands, hashtags,
    và keywords từ nội dung tin nhắn.

    Attributes:
        command_pattern: Regex pattern để phát hiện commands.
        hashtag_pattern: Regex pattern để phát hiện hashtags.
        keyword_patterns: Dict mapping keyword patterns đến capabilities.
    """

    # Default patterns
    DEFAULT_COMMAND_PATTERN = r"^\/([a-zA-Z][a-zA-Z0-9_-]*)"
    DEFAULT_HASHTAG_PATTERN = r"#([a-zA-Z][a-zA-Z0-9_-]*)"

    def __init__(
        self,
        command_pattern: Optional[str] = None,
        hashtag_pattern: Optional[str] = None,
        keyword_patterns: Optional[dict[str, list[str]]] = None,
        intent_patterns: Optional[dict[str, list[str]]] = None,
    ) -> None:
        """Khởi tạo ContentAnalyzer.

        Args:
            command_pattern: Regex pattern cho commands (mặc định: ^/command).
            hashtag_pattern: Regex pattern cho hashtags (mặc định: #hashtag).
            keyword_patterns: Dict mapping capability -> list of keywords.
            intent_patterns: Dict mapping intent -> list of patterns.
        """
        self._command_re: Pattern[str] = re.compile(
            command_pattern or self.DEFAULT_COMMAND_PATTERN,
            re.IGNORECASE
        )
        self._hashtag_re: Pattern[str] = re.compile(
            hashtag_pattern or self.DEFAULT_HASHTAG_PATTERN,
            re.IGNORECASE
        )

        # Compile keyword patterns
        self._keyword_patterns: dict[str, list[Pattern[str]]] = {}
        if keyword_patterns:
            for capability, keywords in keyword_patterns.items():
                self._keyword_patterns[capability] = [
                    re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)
                    for kw in keywords
                ]

        # Compile intent patterns
        self._intent_patterns: dict[str, list[Pattern[str]]] = {}
        if intent_patterns:
            for intent, patterns in intent_patterns.items():
                self._intent_patterns[intent] = [
                    re.compile(pattern, re.IGNORECASE)
                    for pattern in patterns
                ]

    def analyze(self, message: str) -> ContentHints:
        """Phân tích nội dung tin nhắn và trích xuất routing hints.

        Args:
            message: Nội dung tin nhắn cần phân tích.

        Returns:
            ContentHints chứa các thông tin routing được trích xuất.
        """
        hints = ContentHints()

        # Phát hiện command (ở đầu message)
        command_match = self._command_re.match(message.strip())
        if command_match:
            hints.command = f"/{command_match.group(1).lower()}"

        # Phát hiện tất cả hashtags
        hashtag_matches = self._hashtag_re.findall(message)
        hints.hashtags = [f"#{tag.lower()}" for tag in hashtag_matches]

        # Phát hiện keywords và map sang capabilities
        matched_keywords: list[str] = []
        for capability, patterns in self._keyword_patterns.items():
            for pattern in patterns:
                match = pattern.search(message)
                if match:
                    hints.required_capabilities.append(capability)
                    matched_keywords.append(match.group())
                    break  # Chỉ cần một match cho mỗi capability

        hints.keywords = matched_keywords

        # Phát hiện intent
        for intent, patterns in self._intent_patterns.items():
            for pattern in patterns:
                if pattern.search(message):
                    hints.intent = intent
                    break
            if hints.intent:
                break

        # Tính confidence dựa trên số lượng hints
        hint_count = sum([
            1 if hints.command else 0,
            len(hints.hashtags),
            len(hints.required_capabilities),
            1 if hints.intent else 0,
        ])
        hints.confidence = min(1.0, 0.5 + hint_count * 0.1)

        return hints

    async def analyze_async(self, message: str) -> ContentHints:
        """Phân tích nội dung tin nhắn (async version).

        Wrapper async cho method analyze() để dễ dàng tích hợp
        với các async workflows.

        Args:
            message: Nội dung tin nhắn cần phân tích.

        Returns:
            ContentHints chứa các thông tin routing được trích xuất.
        """
        return self.analyze(message)


class ContentRouter:
    """Router định tuyến dựa trên phân tích nội dung tin nhắn.

    ContentRouter phân tích nội dung tin nhắn để phát hiện commands,
    hashtags, keywords và định tuyến đến agent phù hợp.

    Ưu tiên routing:
    1. Commands (/code, /research, etc.)
    2. Hashtags (#work, #personal, etc.)
    3. Required capabilities
    4. Intent-based routing
    5. Default agent

    Attributes:
        agents: Dict mapping agent_id -> BaseAgent.
        command_mapping: Dict mapping command -> agent_id.
        hashtag_mapping: Dict mapping hashtag -> agent_id.
        capability_mapping: Dict mapping capability -> agent_id.
        intent_mapping: Dict mapping intent -> agent_id.
        default_agent: Agent mặc định khi không có match.
        analyzer: ContentAnalyzer instance.
    """

    def __init__(
        self,
        agents: Optional[dict[str, BaseAgent]] = None,
        command_mapping: Optional[dict[str, str]] = None,
        hashtag_mapping: Optional[dict[str, str]] = None,
        capability_mapping: Optional[dict[str, str]] = None,
        intent_mapping: Optional[dict[str, str]] = None,
        default_agent: str = "default",
        analyzer: Optional[ContentAnalyzer] = None,
        keyword_patterns: Optional[dict[str, list[str]]] = None,
        intent_patterns: Optional[dict[str, list[str]]] = None,
    ) -> None:
        """Khởi tạo ContentRouter.

        Args:
            agents: Dict mapping agent_id -> BaseAgent.
            command_mapping: Dict mapping command (với /) -> agent_id.
            hashtag_mapping: Dict mapping hashtag (với #) -> agent_id.
            capability_mapping: Dict mapping capability -> agent_id.
            intent_mapping: Dict mapping intent -> agent_id.
            default_agent: ID của agent mặc định.
            analyzer: ContentAnalyzer tùy chỉnh (nếu không cung cấp, sẽ tạo mới).
            keyword_patterns: Patterns để map keywords sang capabilities.
            intent_patterns: Patterns để phát hiện intents.
        """
        self.agents = agents or {}
        self.default_agent = default_agent

        # Normalize mappings (lowercase keys)
        self.command_mapping = self._normalize_mapping(command_mapping or {})
        self.hashtag_mapping = self._normalize_mapping(hashtag_mapping or {})
        self.capability_mapping = capability_mapping or {}
        self.intent_mapping = intent_mapping or {}

        # Tạo analyzer với keyword và intent patterns
        self.analyzer = analyzer or ContentAnalyzer(
            keyword_patterns=keyword_patterns,
            intent_patterns=intent_patterns,
        )

    def _normalize_mapping(
        self,
        mapping: dict[str, str]
    ) -> dict[str, str]:
        """Chuẩn hóa keys của mapping thành lowercase.

        Args:
            mapping: Dict cần chuẩn hóa.

        Returns:
            Dict với keys đã được chuẩn hóa.
        """
        return {k.lower(): v for k, v in mapping.items()}

    async def route(self, request: RoutingRequest) -> str:
        """Định tuyến request dựa trên phân tích nội dung.

        Args:
            request: RoutingRequest cần định tuyến.

        Returns:
            ID của agent sẽ xử lý request.
        """
        result = await self.route_with_result(request)
        return result.agent_id

    async def route_with_result(
        self,
        request: RoutingRequest
    ) -> RoutingResult:
        """Định tuyến request và trả về kết quả chi tiết.

        Args:
            request: RoutingRequest cần định tuyến.

        Returns:
            RoutingResult với thông tin chi tiết về routing.
        """
        # Phân tích nội dung
        hints = await self.analyzer.analyze_async(request.message)

        metadata: dict[str, Any] = {
            "router_type": "content",
            "hints": {
                "command": hints.command,
                "hashtags": hints.hashtags,
                "keywords": hints.keywords,
                "capabilities": hints.required_capabilities,
                "intent": hints.intent,
            },
        }

        # 1. Kiểm tra command
        if hints.command:
            agent_id = self.command_mapping.get(hints.command)
            if agent_id and self._is_valid_agent(agent_id):
                metadata["matched_by"] = "command"
                metadata["matched_value"] = hints.command
                return RoutingResult(
                    agent_id=agent_id,
                    confidence=hints.confidence,
                    metadata=metadata,
                )

        # 2. Kiểm tra hashtags (theo thứ tự xuất hiện)
        for hashtag in hints.hashtags:
            agent_id = self.hashtag_mapping.get(hashtag)
            if agent_id and self._is_valid_agent(agent_id):
                metadata["matched_by"] = "hashtag"
                metadata["matched_value"] = hashtag
                return RoutingResult(
                    agent_id=agent_id,
                    confidence=hints.confidence,
                    metadata=metadata,
                )

        # 3. Kiểm tra required capabilities
        for capability in hints.required_capabilities:
            agent_id = self.capability_mapping.get(capability)
            if agent_id and self._is_valid_agent(agent_id):
                metadata["matched_by"] = "capability"
                metadata["matched_value"] = capability
                return RoutingResult(
                    agent_id=agent_id,
                    confidence=hints.confidence * 0.9,  # Slightly lower confidence
                    metadata=metadata,
                )

        # 4. Kiểm tra intent
        if hints.intent:
            agent_id = self.intent_mapping.get(hints.intent)
            if agent_id and self._is_valid_agent(agent_id):
                metadata["matched_by"] = "intent"
                metadata["matched_value"] = hints.intent
                return RoutingResult(
                    agent_id=agent_id,
                    confidence=hints.confidence * 0.8,  # Lower confidence for intent
                    metadata=metadata,
                )

        # 5. Fallback to default agent
        metadata["matched_by"] = "default"
        metadata["is_default"] = True
        return RoutingResult(
            agent_id=self.default_agent,
            confidence=0.5,
            metadata=metadata,
        )

    def _is_valid_agent(self, agent_id: str) -> bool:
        """Kiểm tra agent_id có hợp lệ không.

        Một agent_id được coi là hợp lệ nếu:
        - Không có agents được đăng ký (routing config mode)
        - Hoặc agent_id tồn tại trong danh sách agents

        Args:
            agent_id: ID của agent cần kiểm tra.

        Returns:
            True nếu agent_id hợp lệ.
        """
        if not self.agents:
            # Nếu không có agents được cung cấp, cho phép tất cả
            return True
        return agent_id in self.agents

    def add_command_mapping(self, command: str, agent_id: str) -> None:
        """Thêm mapping cho command.

        Args:
            command: Command (với hoặc không có /).
            agent_id: ID của agent xử lý command này.
        """
        # Ensure command starts with /
        if not command.startswith("/"):
            command = f"/{command}"
        self.command_mapping[command.lower()] = agent_id

    def add_hashtag_mapping(self, hashtag: str, agent_id: str) -> None:
        """Thêm mapping cho hashtag.

        Args:
            hashtag: Hashtag (với hoặc không có #).
            agent_id: ID của agent xử lý hashtag này.
        """
        # Ensure hashtag starts with #
        if not hashtag.startswith("#"):
            hashtag = f"#{hashtag}"
        self.hashtag_mapping[hashtag.lower()] = agent_id

    def add_capability_mapping(self, capability: str, agent_id: str) -> None:
        """Thêm mapping cho capability.

        Args:
            capability: Capability cần map.
            agent_id: ID của agent có capability này.
        """
        self.capability_mapping[capability] = agent_id

    def add_intent_mapping(self, intent: str, agent_id: str) -> None:
        """Thêm mapping cho intent.

        Args:
            intent: Intent cần map.
            agent_id: ID của agent xử lý intent này.
        """
        self.intent_mapping[intent] = agent_id

    def remove_command_mapping(self, command: str) -> bool:
        """Xóa mapping cho command.

        Args:
            command: Command cần xóa.

        Returns:
            True nếu mapping được xóa thành công.
        """
        if not command.startswith("/"):
            command = f"/{command}"
        command = command.lower()
        if command in self.command_mapping:
            del self.command_mapping[command]
            return True
        return False

    def remove_hashtag_mapping(self, hashtag: str) -> bool:
        """Xóa mapping cho hashtag.

        Args:
            hashtag: Hashtag cần xóa.

        Returns:
            True nếu mapping được xóa thành công.
        """
        if not hashtag.startswith("#"):
            hashtag = f"#{hashtag}"
        hashtag = hashtag.lower()
        if hashtag in self.hashtag_mapping:
            del self.hashtag_mapping[hashtag]
            return True
        return False

    def get_all_mappings(self) -> dict[str, dict[str, str]]:
        """Lấy tất cả các mappings hiện tại.

        Returns:
            Dict chứa tất cả các loại mappings.
        """
        return {
            "commands": dict(self.command_mapping),
            "hashtags": dict(self.hashtag_mapping),
            "capabilities": dict(self.capability_mapping),
            "intents": dict(self.intent_mapping),
        }

    def register_agent(self, agent_id: str, agent: BaseAgent) -> None:
        """Đăng ký agent với router.

        Args:
            agent_id: ID của agent.
            agent: BaseAgent instance.
        """
        self.agents[agent_id] = agent

    def unregister_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Hủy đăng ký agent.

        Args:
            agent_id: ID của agent cần hủy đăng ký.

        Returns:
            Agent đã hủy đăng ký hoặc None nếu không tìm thấy.
        """
        return self.agents.pop(agent_id, None)


class KeywordMatcher:
    """Utility class để match keywords với capabilities.

    KeywordMatcher cung cấp các method để tìm kiếm keywords
    trong text và map chúng sang capabilities tương ứng.

    Attributes:
        patterns: Dict mapping capability -> compiled patterns.
    """

    def __init__(
        self,
        keyword_map: Optional[dict[str, list[str]]] = None
    ) -> None:
        """Khởi tạo KeywordMatcher.

        Args:
            keyword_map: Dict mapping capability -> list of keywords.
        """
        self._patterns: dict[str, list[Pattern[str]]] = {}

        if keyword_map:
            for capability, keywords in keyword_map.items():
                self.add_keywords(capability, keywords)

    def add_keywords(self, capability: str, keywords: list[str]) -> None:
        """Thêm keywords cho một capability.

        Args:
            capability: Capability cần map.
            keywords: Danh sách keywords cho capability này.
        """
        if capability not in self._patterns:
            self._patterns[capability] = []

        for keyword in keywords:
            pattern = re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE)
            self._patterns[capability].append(pattern)

    def match(self, text: str) -> list[str]:
        """Tìm tất cả capabilities match với text.

        Args:
            text: Text cần kiểm tra.

        Returns:
            Danh sách capabilities match được.
        """
        matched = []
        for capability, patterns in self._patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    matched.append(capability)
                    break  # Chỉ cần một match cho mỗi capability
        return matched

    def get_matched_keywords(self, text: str) -> dict[str, list[str]]:
        """Lấy tất cả keywords match với text, grouped theo capability.

        Args:
            text: Text cần kiểm tra.

        Returns:
            Dict mapping capability -> list of matched keywords.
        """
        result: dict[str, list[str]] = {}
        for capability, patterns in self._patterns.items():
            matched_keywords = []
            for pattern in patterns:
                match = pattern.search(text)
                if match:
                    matched_keywords.append(match.group())
            if matched_keywords:
                result[capability] = matched_keywords
        return result


class IntentClassifier:
    """Bộ phân loại intent đơn giản dựa trên patterns.

    IntentClassifier sử dụng regex patterns để phân loại
    intent của tin nhắn.

    Attributes:
        patterns: Dict mapping intent -> compiled patterns.
    """

    # Default intent patterns
    DEFAULT_INTENTS: dict[str, list[str]] = {
        "coding": [
            r"\b(code|coding|program|programming|debug|fix|implement)\b",
            r"\b(python|java|javascript|typescript|rust|go)\b",
            r"\b(function|class|method|api|bug)\b",
        ],
        "research": [
            r"\b(research|analyze|investigate|study|explore)\b",
            r"\b(find|search|look up|information)\b",
        ],
        "writing": [
            r"\b(write|draft|compose|create|edit)\b",
            r"\b(article|blog|document|report|email)\b",
        ],
        "chat": [
            r"\b(hello|hi|hey|thanks|thank you)\b",
            r"\b(how are you|what's up|good morning|good night)\b",
        ],
        "task": [
            r"\b(todo|task|reminder|schedule|meeting)\b",
            r"\b(deadline|priority|assign|complete)\b",
        ],
    }

    def __init__(
        self,
        intent_patterns: Optional[dict[str, list[str]]] = None,
        use_defaults: bool = True,
    ) -> None:
        """Khởi tạo IntentClassifier.

        Args:
            intent_patterns: Custom patterns cho các intents.
            use_defaults: Có sử dụng default patterns hay không.
        """
        self._patterns: dict[str, list[Pattern[str]]] = {}

        # Load default patterns if requested
        if use_defaults:
            for intent, patterns in self.DEFAULT_INTENTS.items():
                self._patterns[intent] = [
                    re.compile(p, re.IGNORECASE) for p in patterns
                ]

        # Add/override with custom patterns
        if intent_patterns:
            for intent, patterns in intent_patterns.items():
                self._patterns[intent] = [
                    re.compile(p, re.IGNORECASE) for p in patterns
                ]

    def classify(self, text: str) -> Optional[str]:
        """Phân loại intent của text.

        Trả về intent đầu tiên match (theo thứ tự trong dict).

        Args:
            text: Text cần phân loại.

        Returns:
            Intent được phân loại hoặc None.
        """
        for intent, patterns in self._patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    return intent
        return None

    def classify_with_score(self, text: str) -> list[tuple[str, float]]:
        """Phân loại intent với confidence score.

        Trả về danh sách các intents với scores, sắp xếp theo score giảm dần.

        Args:
            text: Text cần phân loại.

        Returns:
            Danh sách tuple (intent, score) sắp xếp theo score.
        """
        scores: dict[str, float] = {}

        for intent, patterns in self._patterns.items():
            match_count = 0
            for pattern in patterns:
                if pattern.search(text):
                    match_count += 1

            if match_count > 0:
                # Score dựa trên tỷ lệ patterns match
                scores[intent] = match_count / len(patterns)

        # Sắp xếp theo score giảm dần
        return sorted(scores.items(), key=lambda x: -x[1])

    def add_intent(self, intent: str, patterns: list[str]) -> None:
        """Thêm intent mới với các patterns.

        Args:
            intent: Tên intent.
            patterns: Danh sách regex patterns.
        """
        self._patterns[intent] = [
            re.compile(p, re.IGNORECASE) for p in patterns
        ]

    def remove_intent(self, intent: str) -> bool:
        """Xóa intent.

        Args:
            intent: Tên intent cần xóa.

        Returns:
            True nếu xóa thành công.
        """
        if intent in self._patterns:
            del self._patterns[intent]
            return True
        return False

    def list_intents(self) -> list[str]:
        """Lấy danh sách tất cả intents.

        Returns:
            Danh sách tên các intents.
        """
        return list(self._patterns.keys())
