# Các Mẫu Khả Năng Quan Sát

Tài liệu này đề cập đến các mẫu truy vết, ghi nhật ký, số liệu và gỡ lỗi cho AI agent.

## Tổng Quan

Khả năng quan sát là yếu tố quan trọng để hiểu hành vi agent, gỡ lỗi vấn đề và cải thiện hiệu suất. Ba trụ cột của khả năng quan sát là:

1. **Truy vết (Tracing)**: Theo dõi luồng yêu cầu qua hệ thống
2. **Ghi nhật ký (Logging)**: Ghi lại các sự kiện và thay đổi trạng thái
3. **Số liệu (Metrics)**: Đo lường hiệu suất và mẫu sử dụng

## 1. Truy Vết

### Tổng Quan

Truy vết ghi lại luồng thực thi đầy đủ của agent, bao gồm:
- Các cuộc gọi LLM và phản hồi
- Các lệnh gọi công cụ
- Chuyển đổi trạng thái
- Sử dụng token

### Tích Hợp LangSmith

LangSmith là nền tảng quan sát chính cho các ứng dụng LangChain/LangGraph.

#### Thiết Lập Cơ Bản

```python
import os

# Đặt biến môi trường
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_PROJECT"] = "my-agent-project"

# Tất cả hoạt động LangChain/LangGraph giờ được tự động truy vết
```

#### Truy Vết Tự Động với Agents

```typescript
// Ví dụ TypeScript - truy vết tự động
import { createAgent } from "@langchain/agents";

function sendEmail(to: string, subject: string, body: string): string {
    return `Đã gửi email đến ${to}`;
}

function searchWeb(query: string): string {
    return `Kết quả tìm kiếm cho: ${query}`;
}

const agent = createAgent({
    model: "gpt-4o",
    tools: [sendEmail, searchWeb],
    systemPrompt: "Bạn là trợ lý hữu ích."
});

// Tất cả các bước được tự động truy vết
const response = await agent.invoke({
    messages: [{ role: "user", content: "Tìm tin AI và gửi tóm tắt đến john@example.com" }]
});
```

#### Sử Dụng Decorator @traceable

```python
from openai import OpenAI
from langsmith import traceable
from langsmith.wrappers import wrap_openai

# Wrap OpenAI client để truy vết tự động
client = wrap_openai(OpenAI())

def retriever(query: str):
    """Truy xuất tài liệu liên quan."""
    return ["Nội dung tài liệu 1", "Nội dung tài liệu 2"]

@traceable
def rag_pipeline(question: str) -> str:
    """Pipeline RAG với truy vết đầy đủ."""
    # Cuộc gọi retriever được truy vết như con
    docs = retriever(question)

    system_message = f"""Trả lời chỉ sử dụng thông tin này:
    {chr(10).join(docs)}"""

    # Cuộc gọi LLM được truy vết như con
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": question}
        ]
    )

    return response.choices[0].message.content

# Thực thi - truy vết đầy đủ được ghi lại
result = rag_pipeline("Thủ đô của Pháp là gì?")
```

#### Thêm Metadata và Tags

```python
from langsmith import traceable

@traceable(
    name="customer_support_agent",
    tags=["production", "customer-support", "v2.0"],
    metadata={
        "team": "support",
        "priority": "high"
    }
)
def customer_support_agent(query: str, customer_id: str) -> str:
    """Agent hỗ trợ khách hàng với metadata phong phú."""
    # Triển khai agent
    pass
```

```typescript
// TypeScript - thêm metadata vào traces
import { LangChainTracer } from "@langchain/core/tracers/tracer_langchain";

const tracer = new LangChainTracer({ projectName: "my-project" });

await agent.invoke(
    {
        messages: [{role: "user", content: "Giúp tôi với đơn hàng"}]
    },
    {
        config: {
            tags: ["production", "order-support", "v1.0"],
            metadata: {
                userId: "user123",
                sessionId: "session456",
                environment: "production",
                orderId: "ORD-789"
            }
        }
    }
);
```

### Triển Khai Truy Vết Tùy Chỉnh

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
import uuid
import json

@dataclass
class Span:
    """Một span đơn trong trace."""
    id: str
    name: str
    parent_id: Optional[str]
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)

    def end(self, status: str = "success"):
        self.end_time = datetime.now()
        self.status = status

    def add_event(self, name: str, attributes: Dict[str, Any] = None):
        self.events.append({
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "attributes": attributes or {}
        })

    def duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return 0

@dataclass
class Trace:
    """Trace đầy đủ của một lần thực thi agent."""
    id: str
    root_span: Span
    spans: List[Span] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "trace_id": self.id,
            "root_span": self._span_to_dict(self.root_span),
            "spans": [self._span_to_dict(s) for s in self.spans]
        }

    def _span_to_dict(self, span: Span) -> dict:
        return {
            "id": span.id,
            "name": span.name,
            "parent_id": span.parent_id,
            "start_time": span.start_time.isoformat(),
            "end_time": span.end_time.isoformat() if span.end_time else None,
            "duration_ms": span.duration_ms(),
            "status": span.status,
            "attributes": span.attributes,
            "events": span.events
        }

class Tracer:
    """Tracer đơn giản cho thực thi agent."""

    def __init__(self):
        self.current_trace: Optional[Trace] = None
        self.span_stack: List[Span] = []

    def start_trace(self, name: str) -> Trace:
        root_span = Span(
            id=str(uuid.uuid4()),
            name=name,
            parent_id=None,
            start_time=datetime.now()
        )
        self.current_trace = Trace(
            id=str(uuid.uuid4()),
            root_span=root_span
        )
        self.span_stack = [root_span]
        return self.current_trace

    def start_span(self, name: str, attributes: Dict[str, Any] = None) -> Span:
        parent = self.span_stack[-1] if self.span_stack else None
        span = Span(
            id=str(uuid.uuid4()),
            name=name,
            parent_id=parent.id if parent else None,
            start_time=datetime.now(),
            attributes=attributes or {}
        )
        self.current_trace.spans.append(span)
        self.span_stack.append(span)
        return span

    def end_span(self, status: str = "success"):
        if self.span_stack:
            span = self.span_stack.pop()
            span.end(status)

    def end_trace(self) -> Trace:
        while self.span_stack:
            self.end_span()
        self.current_trace.root_span.end()
        return self.current_trace

# Sử dụng
tracer = Tracer()
trace = tracer.start_trace("agent_execution")

tracer.start_span("llm_call", {"model": "gpt-4o", "tokens": 150})
# ... Cuộc gọi LLM ...
tracer.end_span()

tracer.start_span("tool_call", {"tool": "web_search"})
# ... Thực thi công cụ ...
tracer.end_span()

completed_trace = tracer.end_trace()
print(json.dumps(completed_trace.to_dict(), indent=2))
```

---

## 2. Ghi Nhật Ký

### Ghi Nhật Ký Có Cấu Trúc

```python
import logging
import json
from datetime import datetime
from typing import Any

class StructuredLogger:
    """Logger có cấu trúc cho các hoạt động agent."""

    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Handler JSON
        handler = logging.StreamHandler()
        handler.setFormatter(self.JsonFormatter())
        self.logger.addHandler(handler)

    class JsonFormatter(logging.Formatter):
        def format(self, record):
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage()
            }
            if hasattr(record, "extra"):
                log_data.update(record.extra)
            return json.dumps(log_data)

    def log(self, level: int, message: str, **kwargs):
        record = self.logger.makeRecord(
            self.logger.name, level, "", 0, message, (), None
        )
        record.extra = kwargs
        self.logger.handle(record)

    def info(self, message: str, **kwargs):
        self.log(logging.INFO, message, **kwargs)

    def error(self, message: str, **kwargs):
        self.log(logging.ERROR, message, **kwargs)

    def debug(self, message: str, **kwargs):
        self.log(logging.DEBUG, message, **kwargs)

# Sử dụng
logger = StructuredLogger("agent")

logger.info("Agent đã khởi động", agent_id="agent-123", version="2.0")
logger.info("Cuộc gọi LLM hoàn thành",
    model="gpt-4o",
    tokens_in=150,
    tokens_out=75,
    latency_ms=1250,
    cost_usd=0.0045
)
logger.info("Công cụ đã thực thi",
    tool="web_search",
    query="AI agents",
    results_count=10,
    latency_ms=850
)
```

### Ghi Nhật Ký Sự Kiện Agent

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any

class EventType(Enum):
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    LLM_START = "llm_start"
    LLM_END = "llm_end"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    ERROR = "error"
    HUMAN_FEEDBACK = "human_feedback"

@dataclass
class AgentEvent:
    type: EventType
    timestamp: datetime
    trace_id: str
    span_id: str
    data: Dict[str, Any]
    error: Optional[str] = None

class AgentEventLogger:
    """Ghi nhật ký sự kiện agent để phân tích."""

    def __init__(self, sink):
        self.sink = sink  # Có thể là file, database, queue, v.v.

    def log_event(self, event: AgentEvent):
        self.sink.write(event)

    def log_llm_start(self, trace_id: str, span_id: str, model: str, messages: list):
        self.log_event(AgentEvent(
            type=EventType.LLM_START,
            timestamp=datetime.now(),
            trace_id=trace_id,
            span_id=span_id,
            data={
                "model": model,
                "message_count": len(messages),
                "input_preview": messages[-1]["content"][:100] if messages else ""
            }
        ))

    def log_llm_end(self, trace_id: str, span_id: str,
                    response: str, tokens: dict, latency_ms: float):
        self.log_event(AgentEvent(
            type=EventType.LLM_END,
            timestamp=datetime.now(),
            trace_id=trace_id,
            span_id=span_id,
            data={
                "response_preview": response[:100],
                "tokens_input": tokens.get("input", 0),
                "tokens_output": tokens.get("output", 0),
                "latency_ms": latency_ms
            }
        ))

    def log_tool_start(self, trace_id: str, span_id: str,
                       tool_name: str, arguments: dict):
        self.log_event(AgentEvent(
            type=EventType.TOOL_START,
            timestamp=datetime.now(),
            trace_id=trace_id,
            span_id=span_id,
            data={
                "tool": tool_name,
                "arguments": arguments
            }
        ))

    def log_tool_end(self, trace_id: str, span_id: str,
                     tool_name: str, result: str, latency_ms: float):
        self.log_event(AgentEvent(
            type=EventType.TOOL_END,
            timestamp=datetime.now(),
            trace_id=trace_id,
            span_id=span_id,
            data={
                "tool": tool_name,
                "result_preview": result[:200],
                "latency_ms": latency_ms
            }
        ))

    def log_error(self, trace_id: str, span_id: str,
                  error_type: str, error_message: str):
        self.log_event(AgentEvent(
            type=EventType.ERROR,
            timestamp=datetime.now(),
            trace_id=trace_id,
            span_id=span_id,
            data={"error_type": error_type},
            error=error_message
        ))
```

### Ghi Nhật Ký Hội Thoại

```python
class ConversationLogger:
    """Ghi nhật ký hội thoại để phân tích và gỡ lỗi."""

    def __init__(self, storage_path: str):
        self.storage_path = storage_path

    def log_conversation(self, conversation_id: str, messages: list,
                         metadata: dict = None):
        """Ghi nhật ký hội thoại hoàn chỉnh."""
        log_entry = {
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat(),
            "messages": messages,
            "metadata": metadata or {},
            "summary": self._generate_summary(messages)
        }

        filepath = f"{self.storage_path}/{conversation_id}.json"
        with open(filepath, "w") as f:
            json.dump(log_entry, f, indent=2)

    def log_turn(self, conversation_id: str, role: str, content: str,
                 tool_calls: list = None):
        """Ghi nhật ký một lượt hội thoại đơn."""
        turn = {
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "content": content,
            "tool_calls": tool_calls or []
        }

        # Thêm vào hội thoại hiện có
        filepath = f"{self.storage_path}/{conversation_id}.json"
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {"conversation_id": conversation_id, "messages": []}

        data["messages"].append(turn)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def _generate_summary(self, messages: list) -> dict:
        """Tạo tóm tắt hội thoại."""
        return {
            "total_turns": len(messages),
            "user_messages": len([m for m in messages if m.get("role") == "user"]),
            "assistant_messages": len([m for m in messages if m.get("role") == "assistant"]),
            "tool_calls": sum(len(m.get("tool_calls", [])) for m in messages)
        }
```

---

## 3. Số Liệu

### Các Số Liệu Chính Cần Theo Dõi

| Số Liệu | Mô Tả | Loại |
|---------|-------|------|
| Request latency | Thời gian phản hồi end-to-end | Histogram |
| LLM latency | Thời gian cho các cuộc gọi LLM | Histogram |
| Tool latency | Thời gian thực thi công cụ | Histogram |
| Token usage | Token đầu vào/đầu ra | Counter |
| Error rate | Yêu cầu thất bại | Rate |
| Tool usage | Tần suất gọi công cụ | Counter |
| Cost | Chi phí API ước tính | Counter |

### Triển Khai Số Liệu

```python
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List
import time
import statistics

@dataclass
class MetricValue:
    count: int = 0
    sum: float = 0.0
    min: float = float('inf')
    max: float = float('-inf')
    values: List[float] = field(default_factory=list)

    def record(self, value: float):
        self.count += 1
        self.sum += value
        self.min = min(self.min, value)
        self.max = max(self.max, value)
        self.values.append(value)

    @property
    def mean(self) -> float:
        return self.sum / self.count if self.count > 0 else 0

    @property
    def p50(self) -> float:
        return self._percentile(50)

    @property
    def p95(self) -> float:
        return self._percentile(95)

    @property
    def p99(self) -> float:
        return self._percentile(99)

    def _percentile(self, p: int) -> float:
        if not self.values:
            return 0
        sorted_values = sorted(self.values)
        index = int(len(sorted_values) * p / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]

class AgentMetrics:
    """Thu thập và báo cáo số liệu agent."""

    def __init__(self):
        self.histograms: Dict[str, MetricValue] = defaultdict(MetricValue)
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}

    def record_histogram(self, name: str, value: float, labels: dict = None):
        """Ghi giá trị histogram (ví dụ: latency)."""
        key = self._make_key(name, labels)
        self.histograms[key].record(value)

    def increment_counter(self, name: str, value: int = 1, labels: dict = None):
        """Tăng counter (ví dụ: số request)."""
        key = self._make_key(name, labels)
        self.counters[key] += value

    def set_gauge(self, name: str, value: float, labels: dict = None):
        """Đặt giá trị gauge (ví dụ: kết nối đang hoạt động)."""
        key = self._make_key(name, labels)
        self.gauges[key] = value

    def _make_key(self, name: str, labels: dict = None) -> str:
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def get_report(self) -> dict:
        """Tạo báo cáo số liệu."""
        report = {
            "histograms": {},
            "counters": dict(self.counters),
            "gauges": dict(self.gauges)
        }

        for name, metric in self.histograms.items():
            report["histograms"][name] = {
                "count": metric.count,
                "mean": metric.mean,
                "min": metric.min,
                "max": metric.max,
                "p50": metric.p50,
                "p95": metric.p95,
                "p99": metric.p99
            }

        return report

    def timer(self, name: str, labels: dict = None):
        """Context manager để đo thời gian hoạt động."""
        return MetricTimer(self, name, labels)

class MetricTimer:
    def __init__(self, metrics: AgentMetrics, name: str, labels: dict):
        self.metrics = metrics
        self.name = name
        self.labels = labels
        self.start = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        duration_ms = (time.time() - self.start) * 1000
        self.metrics.record_histogram(self.name, duration_ms, self.labels)

# Sử dụng
metrics = AgentMetrics()

# Ghi request
with metrics.timer("request_latency", {"agent": "support"}):
    # Xử lý request
    pass

# Ghi cuộc gọi LLM
with metrics.timer("llm_latency", {"model": "gpt-4o"}):
    # Cuộc gọi LLM
    pass

# Ghi tokens
metrics.increment_counter("tokens_input", 150, {"model": "gpt-4o"})
metrics.increment_counter("tokens_output", 75, {"model": "gpt-4o"})

# Ghi sử dụng công cụ
metrics.increment_counter("tool_calls", 1, {"tool": "web_search"})

# Lấy báo cáo
print(json.dumps(metrics.get_report(), indent=2))
```

### Theo Dõi Chi Phí

```python
# Chi phí token theo model (tính đến 2024)
MODEL_COSTS = {
    "gpt-4o": {"input": 0.005, "output": 0.015},  # mỗi 1K tokens
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125}
}

class CostTracker:
    """Theo dõi chi phí API cho các hoạt động agent."""

    def __init__(self):
        self.total_cost = 0.0
        self.cost_by_model: Dict[str, float] = defaultdict(float)
        self.cost_by_operation: Dict[str, float] = defaultdict(float)

    def record_llm_usage(self, model: str, tokens_in: int, tokens_out: int,
                         operation: str = "general"):
        """Ghi sử dụng LLM và tính chi phí."""
        if model not in MODEL_COSTS:
            return

        costs = MODEL_COSTS[model]
        cost = (tokens_in / 1000 * costs["input"] +
                tokens_out / 1000 * costs["output"])

        self.total_cost += cost
        self.cost_by_model[model] += cost
        self.cost_by_operation[operation] += cost

    def get_report(self) -> dict:
        return {
            "total_cost_usd": round(self.total_cost, 4),
            "by_model": {k: round(v, 4) for k, v in self.cost_by_model.items()},
            "by_operation": {k: round(v, 4) for k, v in self.cost_by_operation.items()}
        }

# Sử dụng
cost_tracker = CostTracker()

cost_tracker.record_llm_usage("gpt-4o", tokens_in=500, tokens_out=150,
                              operation="customer_support")
cost_tracker.record_llm_usage("gpt-4o-mini", tokens_in=200, tokens_out=100,
                              operation="summarization")

print(cost_tracker.get_report())
```

---

## 4. Các Mẫu Gỡ Lỗi

### Chế Độ Debug

```python
class AgentDebugger:
    """Các tiện ích debug cho phát triển agent."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.breakpoints: List[str] = []
        self.history: List[dict] = []

    def set_breakpoint(self, step_name: str):
        """Đặt breakpoint tại một bước cụ thể."""
        self.breakpoints.append(step_name)

    def log_step(self, step_name: str, state: dict, output: any):
        """Ghi bước agent để debug."""
        if not self.enabled:
            return

        entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step_name,
            "state_snapshot": json.dumps(state, default=str)[:1000],
            "output_preview": str(output)[:500]
        }
        self.history.append(entry)

        if step_name in self.breakpoints:
            self._pause_for_inspection(entry)

    def _pause_for_inspection(self, entry: dict):
        """Tạm dừng thực thi để kiểm tra."""
        print(f"\n{'='*50}")
        print(f"BREAKPOINT: {entry['step']}")
        print(f"Trạng thái: {entry['state_snapshot']}")
        print(f"Đầu ra: {entry['output_preview']}")
        input("Nhấn Enter để tiếp tục...")

    def get_history(self) -> List[dict]:
        return self.history

    def replay_step(self, step_index: int) -> dict:
        """Lấy chi tiết của một bước cụ thể."""
        if 0 <= step_index < len(self.history):
            return self.history[step_index]
        return None
```

### Truy Vết Chi Tiết

```python
from functools import wraps

def verbose_trace(func):
    """Decorator cho truy vết hàm chi tiết."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__

        # Ghi đầu vào
        print(f"\n[TRACE] Đang vào: {func_name}")
        print(f"  Args: {args[:2]}...")  # Cắt ngắn để dễ đọc
        print(f"  Kwargs: {list(kwargs.keys())}")

        start = time.time()
        try:
            result = func(*args, **kwargs)
            duration = (time.time() - start) * 1000

            # Ghi đầu ra
            print(f"[TRACE] Đang thoát: {func_name}")
            print(f"  Thời gian: {duration:.2f}ms")
            print(f"  Loại kết quả: {type(result).__name__}")

            return result
        except Exception as e:
            print(f"[TRACE] Lỗi trong {func_name}: {e}")
            raise

    return wrapper

# Sử dụng
@verbose_trace
def agent_step(state: dict) -> dict:
    # Logic agent
    return {"output": "result"}
```

### Kiểm Tra Trạng Thái

```python
class StateInspector:
    """Kiểm tra và xác thực trạng thái agent."""

    def __init__(self, schema: dict = None):
        self.schema = schema
        self.state_history: List[dict] = []

    def capture(self, state: dict, step: str):
        """Ghi lại trạng thái tại một điểm trong thực thi."""
        self.state_history.append({
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "state": json.loads(json.dumps(state, default=str))
        })

    def diff(self, index1: int, index2: int) -> dict:
        """So sánh hai snapshot trạng thái."""
        if index1 >= len(self.state_history) or index2 >= len(self.state_history):
            return {}

        state1 = self.state_history[index1]["state"]
        state2 = self.state_history[index2]["state"]

        added = {}
        removed = {}
        changed = {}

        all_keys = set(state1.keys()) | set(state2.keys())
        for key in all_keys:
            if key not in state1:
                added[key] = state2[key]
            elif key not in state2:
                removed[key] = state1[key]
            elif state1[key] != state2[key]:
                changed[key] = {"old": state1[key], "new": state2[key]}

        return {"added": added, "removed": removed, "changed": changed}

    def validate(self, state: dict) -> List[str]:
        """Xác thực trạng thái theo schema."""
        if not self.schema:
            return []

        errors = []
        for field, spec in self.schema.items():
            if spec.get("required") and field not in state:
                errors.append(f"Thiếu trường bắt buộc: {field}")
            if field in state:
                expected_type = spec.get("type")
                if expected_type and not isinstance(state[field], expected_type):
                    errors.append(f"Kiểu không hợp lệ cho {field}: mong đợi {expected_type}")

        return errors
```

---

## 5. Ví Dụ Tích Hợp

### Thiết Lập Khả Năng Quan Sát Hoàn Chỉnh

```python
from contextlib import contextmanager
import uuid

class AgentObservability:
    """Khả năng quan sát hoàn chỉnh cho các hệ thống agent."""

    def __init__(self):
        self.tracer = Tracer()
        self.logger = StructuredLogger("agent")
        self.metrics = AgentMetrics()
        self.cost_tracker = CostTracker()

    @contextmanager
    def trace_request(self, name: str, metadata: dict = None):
        """Context manager để truy vết một request hoàn chỉnh."""
        trace = self.tracer.start_trace(name)
        request_id = trace.id

        self.logger.info(f"Request bắt đầu: {name}",
                        request_id=request_id,
                        metadata=metadata)

        start_time = time.time()

        try:
            yield trace
            status = "success"
        except Exception as e:
            status = "error"
            self.logger.error(f"Request thất bại: {name}",
                            request_id=request_id,
                            error=str(e))
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000

            self.tracer.end_trace()
            self.metrics.record_histogram("request_latency", duration_ms,
                                         {"name": name, "status": status})
            self.logger.info(f"Request hoàn thành: {name}",
                            request_id=request_id,
                            duration_ms=duration_ms,
                            status=status)

    @contextmanager
    def trace_llm_call(self, model: str, messages: list):
        """Truy vết cuộc gọi LLM với số liệu."""
        span = self.tracer.start_span("llm_call", {"model": model})
        start_time = time.time()

        try:
            yield
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.tracer.end_span()
            self.metrics.record_histogram("llm_latency", duration_ms,
                                         {"model": model})

    def record_llm_usage(self, model: str, tokens_in: int, tokens_out: int):
        """Ghi sử dụng token LLM và chi phí."""
        self.metrics.increment_counter("tokens_input", tokens_in, {"model": model})
        self.metrics.increment_counter("tokens_output", tokens_out, {"model": model})
        self.cost_tracker.record_llm_usage(model, tokens_in, tokens_out)

    @contextmanager
    def trace_tool_call(self, tool_name: str, arguments: dict):
        """Truy vết cuộc gọi công cụ."""
        span = self.tracer.start_span("tool_call", {"tool": tool_name})
        start_time = time.time()

        try:
            yield
            status = "success"
        except Exception as e:
            status = "error"
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.tracer.end_span(status)
            self.metrics.record_histogram("tool_latency", duration_ms,
                                         {"tool": tool_name})
            self.metrics.increment_counter("tool_calls", 1,
                                          {"tool": tool_name, "status": status})

    def get_report(self) -> dict:
        """Lấy báo cáo khả năng quan sát toàn diện."""
        return {
            "metrics": self.metrics.get_report(),
            "costs": self.cost_tracker.get_report()
        }

# Sử dụng
obs = AgentObservability()

with obs.trace_request("customer_query", {"customer_id": "123"}):
    with obs.trace_llm_call("gpt-4o", messages):
        response = llm.invoke(messages)
    obs.record_llm_usage("gpt-4o", tokens_in=150, tokens_out=75)

    with obs.trace_tool_call("search", {"query": "products"}):
        results = search_tool.invoke({"query": "products"})

print(json.dumps(obs.get_report(), indent=2))
```

---

## Thực Tiễn Tốt Nhất

### 1. Truy Vết Mọi Thứ
- Truy vết tất cả cuộc gọi LLM, lệnh gọi công cụ và chuyển đổi trạng thái
- Bao gồm đủ ngữ cảnh để debug vấn đề
- Sử dụng ID tương quan để liên kết các trace liên quan

### 2. Ghi Nhật Ký Chiến Lược
- Sử dụng ghi nhật ký có cấu trúc với các trường nhất quán
- Bao gồm request ID trong tất cả nhật ký
- Ghi nhật ký ở mức phù hợp (DEBUG cho phát triển, INFO cho production)

### 3. Giám Sát Số Liệu Chính
- Theo dõi phân vị latency (p50, p95, p99)
- Giám sát tỷ lệ lỗi và loại lỗi
- Theo dõi sử dụng token và chi phí

### 4. Thiết Lập Cảnh Báo
- Cảnh báo khi tỷ lệ lỗi tăng đột biến
- Cảnh báo khi latency suy giảm
- Cảnh báo khi chi phí bất thường

### 5. Sử Dụng Công Cụ Phù Hợp
- LangSmith cho ứng dụng LangChain/LangGraph
- Phoenix cho truy vết mã nguồn mở
- Giải pháp tùy chỉnh cho nhu cầu cụ thể
