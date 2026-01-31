# Các Mẫu Bộ Nhớ

Tài liệu này đề cập đến các kiến trúc bộ nhớ và mẫu quản lý trạng thái cho AI agent.

## Tổng Quan

Bộ nhớ là yếu tố quan trọng để agent duy trì ngữ cảnh, học từ các tương tác và đưa ra quyết định có căn cứ. Các loại bộ nhớ khác nhau phục vụ các mục đích khác nhau:

| Loại Bộ Nhớ | Thời Gian | Mục Đích | Triển Khai |
|-------------|-----------|----------|------------|
| Ngắn hạn | Phiên | Ngữ cảnh hội thoại | Buffer, Checkpointer |
| Dài hạn | Vĩnh viễn | Lưu giữ kiến thức | Vector stores, cơ sở dữ liệu |
| Sự kiện | Thay đổi | Nhớ lại trải nghiệm | Indexed events |
| Ngữ nghĩa | Vĩnh viễn | Quan hệ khái niệm | Embeddings, knowledge graphs |
| Làm việc | Tạm thời | Trạng thái tác vụ đang hoạt động | State management |

---

## 1. Bộ Nhớ Ngắn Hạn (Buffer Hội Thoại)

### Tổng Quan

Bộ nhớ ngắn hạn duy trì ngữ cảnh trực tiếp của hội thoại hoặc tác vụ. Nó thường lưu trữ:
- Tin nhắn gần đây
- Trạng thái tác vụ hiện tại
- Kết quả trung gian

### Triển Khai với Checkpointers

```python
from langgraph.checkpoint.memory import InMemorySaver

# Tạo checkpointer cho bộ nhớ ngắn hạn
checkpointer = InMemorySaver()

# Biên dịch workflow với checkpointer
app = workflow.compile(checkpointer=checkpointer)

# Gọi với thread_id để duy trì hội thoại
config = {"configurable": {"thread_id": "conversation-123"}}
result = app.invoke({"input": "Xin chào"}, config)
```

### Các Mẫu Buffer Hội Thoại

#### Buffer Cửa Sổ Trượt

```python
class SlidingWindowBuffer:
    def __init__(self, max_messages: int = 10):
        self.max_messages = max_messages
        self.messages = []

    def add_message(self, message: dict):
        self.messages.append(message)
        if len(self.messages) > self.max_messages:
            self.messages.pop(0)

    def get_context(self) -> List[dict]:
        return self.messages.copy()
```

#### Buffer Giới Hạn Token

```python
from tiktoken import encoding_for_model

class TokenLimitedBuffer:
    def __init__(self, max_tokens: int = 4000, model: str = "gpt-4"):
        self.max_tokens = max_tokens
        self.encoder = encoding_for_model(model)
        self.messages = []

    def add_message(self, message: dict):
        self.messages.append(message)
        self._trim_to_limit()

    def _trim_to_limit(self):
        while self._total_tokens() > self.max_tokens and len(self.messages) > 1:
            self.messages.pop(0)

    def _total_tokens(self) -> int:
        text = " ".join([m["content"] for m in self.messages])
        return len(self.encoder.encode(text))
```

#### Buffer Tóm Tắt

```python
class SummaryBuffer:
    def __init__(self, llm, max_messages: int = 5):
        self.llm = llm
        self.max_messages = max_messages
        self.messages = []
        self.summary = ""

    def add_message(self, message: dict):
        self.messages.append(message)
        if len(self.messages) > self.max_messages:
            self._summarize_oldest()

    def _summarize_oldest(self):
        # Lấy các tin nhắn cũ nhất và tóm tắt
        to_summarize = self.messages[:3]
        self.messages = self.messages[3:]

        # Tạo tóm tắt
        summary_prompt = f"""
        Tóm tắt đoạn hội thoại này:
        {to_summarize}

        Tóm tắt trước đó: {self.summary}
        """
        self.summary = self.llm.invoke(summary_prompt)

    def get_context(self) -> str:
        return f"Tóm tắt: {self.summary}\n\nGần đây: {self.messages}"
```

---

## 2. Bộ Nhớ Dài Hạn (Vector Stores)

### Tổng Quan

Bộ nhớ dài hạn lưu trữ thông tin vĩnh viễn, thường sử dụng vector embeddings để truy xuất ngữ nghĩa.

### Triển Khai LangGraph Store

```python
from langgraph.store.memory import InMemoryStore
from langchain.embeddings import init_embeddings

# Khởi tạo store với embeddings cho tìm kiếm ngữ nghĩa
store = InMemoryStore(
    index={
        "embed": init_embeddings("openai:text-embedding-3-small"),
        "dims": 1536,
        "fields": ["content", "$"]  # Các trường để embed
    }
)

# Lưu một bộ nhớ
import uuid

namespace = ("user", "user-123", "memories")
store.put(
    namespace,
    str(uuid.uuid4()),
    {
        "content": "Người dùng thích chế độ tối",
        "category": "preferences",
        "timestamp": "2024-01-15"
    },
    index=["content"]  # Trường nào để embed
)

# Truy xuất bằng tìm kiếm ngữ nghĩa
memories = store.search(
    namespace,
    query="Tùy chọn UI của người dùng là gì?",
    limit=5
)
```

### Tích Hợp Vector Store

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

class LongTermMemory:
    def __init__(self, collection_name: str = "agent_memory"):
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory="./memory_db"
        )

    def store(self, content: str, metadata: dict = None):
        """Lưu một bộ nhớ với metadata tùy chọn."""
        self.vectorstore.add_texts(
            texts=[content],
            metadatas=[metadata] if metadata else None
        )

    def recall(self, query: str, k: int = 5) -> List[str]:
        """Nhớ lại các bộ nhớ liên quan."""
        docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

    def recall_with_score(self, query: str, k: int = 5, threshold: float = 0.7):
        """Nhớ lại các bộ nhớ trên ngưỡng liên quan."""
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return [(doc.page_content, score) for doc, score in results if score >= threshold]
```

### Tích Hợp Bộ Nhớ CrewAI

```python
from crewai import Crew, Agent

# Bật bộ nhớ ở cấp crew
crew = Crew(
    agents=[agent1, agent2],
    tasks=[task1, task2],
    memory=True,  # Bật bộ nhớ ngắn hạn, dài hạn và entity
    verbose=True
)

# Bộ nhớ được tự động sử dụng cho:
# 1. Lưu lịch sử hội thoại (bộ nhớ ngắn hạn)
# 2. Ghi nhớ các sự kiện quan trọng (bộ nhớ dài hạn)
# 3. Theo dõi entities và quan hệ (bộ nhớ entity)

# Bộ nhớ cấp agent
agent = Agent(
    role="Hỗ Trợ Khách Hàng",
    goal="Cung cấp hỗ trợ hữu ích",
    backstory="Agent hỗ trợ có kinh nghiệm",
    memory=True  # Agent nhớ các tương tác trước
)
```

---

## 3. Bộ Nhớ Sự Kiện (Episodic Memory)

### Tổng Quan

Bộ nhớ sự kiện lưu trữ các trải nghiệm hoặc sự kiện cụ thể, cho phép agent nhớ lại và học từ các tình huống trong quá khứ.

### Triển Khai

```python
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
import json

@dataclass
class Episode:
    id: str
    timestamp: datetime
    context: str
    action: str
    result: str
    outcome: str  # success, failure, partial
    metadata: dict

class EpisodicMemory:
    def __init__(self, store: InMemoryStore, embeddings):
        self.store = store
        self.embeddings = embeddings
        self.namespace = ("episodes",)

    def record_episode(self, episode: Episode):
        """Ghi lại một sự kiện mới."""
        self.store.put(
            self.namespace,
            episode.id,
            {
                "timestamp": episode.timestamp.isoformat(),
                "context": episode.context,
                "action": episode.action,
                "result": episode.result,
                "outcome": episode.outcome,
                "metadata": json.dumps(episode.metadata)
            },
            index=["context", "action", "result"]
        )

    def recall_similar(self, current_context: str, k: int = 5) -> List[Episode]:
        """Nhớ lại các sự kiện tương tự với ngữ cảnh hiện tại."""
        results = self.store.search(
            self.namespace,
            query=current_context,
            limit=k
        )
        return [self._to_episode(r) for r in results]

    def recall_by_outcome(self, outcome: str) -> List[Episode]:
        """Nhớ lại các sự kiện với kết quả cụ thể."""
        results = self.store.list(self.namespace)
        return [
            self._to_episode(r) for r in results
            if r.value.get("outcome") == outcome
        ]

    def learn_from_failure(self, current_context: str) -> Optional[str]:
        """Học từ các thất bại trong quá khứ trong ngữ cảnh tương tự."""
        failures = self.recall_similar(current_context, k=3)
        failures = [e for e in failures if e.outcome == "failure"]

        if not failures:
            return None

        # Tạo lời khuyên dựa trên thất bại quá khứ
        advice = f"Dựa trên kinh nghiệm quá khứ, tránh các hành động này:\n"
        for episode in failures:
            advice += f"- {episode.action} (dẫn đến: {episode.result})\n"
        return advice
```

### Ghi Lại Sự Kiện trong Workflows

```python
from langgraph.graph import StateGraph

class EpisodicState(TypedDict):
    input: str
    context: str
    action: str
    result: str
    outcome: str

def record_and_execute(state: EpisodicState, episodic_memory: EpisodicMemory):
    """Thực thi hành động và ghi lại sự kiện."""
    # Kiểm tra các trải nghiệm tương tự trong quá khứ
    similar_episodes = episodic_memory.recall_similar(state["context"])

    # Lấy lời khuyên từ thất bại
    failure_advice = episodic_memory.learn_from_failure(state["context"])
    if failure_advice:
        # Đưa lời khuyên vào ngữ cảnh
        state["context"] += f"\n\nLưu ý: {failure_advice}"

    # Thực thi hành động
    result = execute_action(state["action"])
    outcome = evaluate_outcome(result)

    # Ghi lại sự kiện
    episode = Episode(
        id=str(uuid.uuid4()),
        timestamp=datetime.now(),
        context=state["context"],
        action=state["action"],
        result=result,
        outcome=outcome,
        metadata={"similar_episodes": len(similar_episodes)}
    )
    episodic_memory.record_episode(episode)

    return {"result": result, "outcome": outcome}
```

---

## 4. Bộ Nhớ Ngữ Nghĩa

### Tổng Quan

Bộ nhớ ngữ nghĩa lưu trữ các khái niệm, sự kiện và quan hệ, được tổ chức để truy xuất hiệu quả dựa trên ý nghĩa.

### Tích Hợp Knowledge Graph

```python
from langchain_community.graphs import Neo4jGraph

class SemanticMemory:
    def __init__(self, uri: str, user: str, password: str):
        self.graph = Neo4jGraph(url=uri, username=user, password=password)

    def store_fact(self, subject: str, predicate: str, object: str):
        """Lưu một sự kiện dưới dạng quan hệ."""
        query = """
        MERGE (s:Entity {name: $subject})
        MERGE (o:Entity {name: $object})
        MERGE (s)-[r:RELATES {type: $predicate}]->(o)
        """
        self.graph.query(query, {
            "subject": subject,
            "predicate": predicate,
            "object": object
        })

    def query_facts(self, query: str) -> List[dict]:
        """Truy vấn knowledge graph."""
        return self.graph.query(query)

    def get_related_concepts(self, concept: str, depth: int = 2) -> List[str]:
        """Lấy các khái niệm liên quan đến khái niệm đã cho."""
        query = f"""
        MATCH (e:Entity {{name: $concept}})-[*1..{depth}]-(related:Entity)
        RETURN DISTINCT related.name as name
        """
        results = self.graph.query(query, {"concept": concept})
        return [r["name"] for r in results]
```

### Bộ Nhớ Ngữ Nghĩa Dựa Trên Embedding

```python
from langchain.embeddings import OpenAIEmbeddings
import numpy as np

class SemanticIndex:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.concepts = {}  # name -> embedding
        self.relationships = []  # (concept1, relation, concept2)

    def add_concept(self, name: str, description: str):
        """Thêm một khái niệm với embedding của nó."""
        embedding = self.embeddings.embed_query(description)
        self.concepts[name] = {
            "description": description,
            "embedding": embedding
        }

    def add_relationship(self, concept1: str, relation: str, concept2: str):
        """Thêm quan hệ giữa các khái niệm."""
        self.relationships.append((concept1, relation, concept2))

    def find_similar_concepts(self, query: str, k: int = 5) -> List[str]:
        """Tìm các khái niệm tương tự với query."""
        query_embedding = self.embeddings.embed_query(query)

        similarities = []
        for name, data in self.concepts.items():
            sim = np.dot(query_embedding, data["embedding"])
            similarities.append((name, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in similarities[:k]]

    def get_concept_context(self, concept: str) -> str:
        """Lấy ngữ cảnh đầy đủ cho một khái niệm bao gồm các quan hệ."""
        if concept not in self.concepts:
            return ""

        context = f"Khái niệm: {concept}\n"
        context += f"Mô tả: {self.concepts[concept]['description']}\n"
        context += "Quan hệ:\n"

        for c1, rel, c2 in self.relationships:
            if c1 == concept:
                context += f"  - {rel} {c2}\n"
            elif c2 == concept:
                context += f"  - {c1} {rel} này\n"

        return context
```

---

## 5. Bộ Nhớ Làm Việc

### Tổng Quan

Bộ nhớ làm việc là lưu trữ tạm thời cho tác vụ hiện tại, giữ kết quả trung gian, mục tiêu đang hoạt động và ngữ cảnh liên quan.

### LangGraph State như Bộ Nhớ Làm Việc

```python
from langgraph.graph import StateGraph
from typing import TypedDict, List, Optional

class WorkingMemory(TypedDict):
    # Tác vụ hiện tại
    current_goal: str
    sub_goals: List[str]
    completed_goals: List[str]

    # Ngữ cảnh đang hoạt động
    relevant_facts: List[str]
    active_entities: List[str]

    # Kết quả trung gian
    scratch_pad: str
    calculations: dict

    # Siêu nhận thức
    confidence: float
    uncertainties: List[str]

def update_working_memory(state: WorkingMemory, llm) -> WorkingMemory:
    """Cập nhật bộ nhớ làm việc dựa trên trạng thái hiện tại."""
    # Xác định các entities đang hoạt động từ ngữ cảnh gần đây
    entities = extract_entities(state["scratch_pad"])

    # Cập nhật độ tin cậy dựa trên các điểm không chắc chắn
    confidence = 1.0 - (len(state["uncertainties"]) * 0.1)

    return {
        **state,
        "active_entities": entities,
        "confidence": max(0.0, confidence)
    }
```

### Mẫu Scratchpad

```python
class ScratchPad:
    """Scratchpad bộ nhớ làm việc cho suy luận trung gian."""

    def __init__(self):
        self.thoughts = []
        self.observations = []
        self.plans = []
        self.calculations = {}

    def add_thought(self, thought: str):
        self.thoughts.append({
            "content": thought,
            "timestamp": datetime.now()
        })

    def add_observation(self, observation: str, source: str):
        self.observations.append({
            "content": observation,
            "source": source,
            "timestamp": datetime.now()
        })

    def set_plan(self, steps: List[str]):
        self.plans = steps

    def mark_step_complete(self, step_index: int):
        if step_index < len(self.plans):
            self.plans[step_index] = f"[XONG] {self.plans[step_index]}"

    def store_calculation(self, key: str, value: any):
        self.calculations[key] = value

    def get_calculation(self, key: str) -> any:
        return self.calculations.get(key)

    def to_context(self) -> str:
        """Chuyển scratchpad thành chuỗi ngữ cảnh cho LLM."""
        context = "## Bộ Nhớ Làm Việc\n\n"

        if self.plans:
            context += "### Kế Hoạch Hiện Tại:\n"
            for i, step in enumerate(self.plans):
                context += f"{i+1}. {step}\n"

        if self.thoughts:
            context += "\n### Suy Nghĩ Gần Đây:\n"
            for thought in self.thoughts[-5:]:
                context += f"- {thought['content']}\n"

        if self.observations:
            context += "\n### Quan Sát:\n"
            for obs in self.observations[-5:]:
                context += f"- [{obs['source']}] {obs['content']}\n"

        if self.calculations:
            context += "\n### Giá Trị Đã Lưu:\n"
            for key, value in self.calculations.items():
                context += f"- {key}: {value}\n"

        return context
```

---

## Các Mẫu Tích Hợp Bộ Nhớ

### Kiến Trúc Bộ Nhớ Kết Hợp

```python
class AgentMemorySystem:
    """Hệ thống bộ nhớ thống nhất kết hợp tất cả các loại bộ nhớ."""

    def __init__(self, store, embeddings, graph_db=None):
        # Ngắn hạn
        self.conversation_buffer = SlidingWindowBuffer(max_messages=20)

        # Dài hạn
        self.long_term = LongTermMemory(store, embeddings)

        # Sự kiện
        self.episodic = EpisodicMemory(store, embeddings)

        # Ngữ nghĩa
        self.semantic = SemanticMemory(graph_db) if graph_db else None

        # Làm việc
        self.scratchpad = ScratchPad()

    def get_context(self, query: str) -> str:
        """Lấy ngữ cảnh kết hợp từ tất cả hệ thống bộ nhớ."""
        context = ""

        # Hội thoại gần đây
        context += "## Hội Thoại Gần Đây\n"
        context += str(self.conversation_buffer.get_context())

        # Bộ nhớ dài hạn liên quan
        context += "\n## Bộ Nhớ Liên Quan\n"
        memories = self.long_term.recall(query, k=3)
        for mem in memories:
            context += f"- {mem}\n"

        # Các sự kiện tương tự trong quá khứ
        context += "\n## Trải Nghiệm Quá Khứ\n"
        episodes = self.episodic.recall_similar(query, k=2)
        for ep in episodes:
            context += f"- {ep.action} -> {ep.outcome}\n"

        # Các khái niệm liên quan
        if self.semantic:
            context += "\n## Khái Niệm Liên Quan\n"
            concepts = self.semantic.find_similar_concepts(query, k=3)
            for concept in concepts:
                context += f"- {concept}\n"

        # Bộ nhớ làm việc hiện tại
        context += self.scratchpad.to_context()

        return context

    def update_from_interaction(self, input: str, output: str, outcome: str):
        """Cập nhật tất cả hệ thống bộ nhớ từ một tương tác."""
        # Cập nhật buffer hội thoại
        self.conversation_buffer.add_message({"role": "user", "content": input})
        self.conversation_buffer.add_message({"role": "assistant", "content": output})

        # Lưu vào dài hạn nếu quan trọng
        if self._is_significant(output):
            self.long_term.store(output, metadata={"type": "response"})

        # Ghi lại sự kiện
        episode = Episode(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            context=input,
            action=output,
            result=output,
            outcome=outcome,
            metadata={}
        )
        self.episodic.record_episode(episode)

    def _is_significant(self, content: str) -> bool:
        """Xác định nội dung có đủ quan trọng để lưu dài hạn không."""
        # Heuristics đơn giản - có thể phức tạp hơn
        return len(content) > 100 or any(keyword in content.lower()
            for keyword in ["quan trọng", "nhớ", "ghi chú", "điểm chính"])
```

---

## Thực Tiễn Tốt Nhất

### 1. Chọn Bộ Nhớ
- Sử dụng ngắn hạn cho tính liên tục hội thoại
- Sử dụng dài hạn cho kiến thức bền vững
- Sử dụng sự kiện để học từ kinh nghiệm
- Sử dụng ngữ nghĩa cho quan hệ khái niệm
- Sử dụng bộ nhớ làm việc cho suy luận phức tạp

### 2. Vệ Sinh Bộ Nhớ
- Thường xuyên loại bỏ các bộ nhớ không liên quan
- Triển khai tính điểm quan trọng
- Đặt TTL cho bộ nhớ tạm thời
- Nén các bộ nhớ cũ thành tóm tắt

### 3. Tối Ưu Truy Xuất
- Sử dụng tìm kiếm hybrid (từ khóa + ngữ nghĩa)
- Triển khai ngưỡng liên quan
- Giới hạn kích thước ngữ cảnh cho các cuộc gọi LLM
- Cache các bộ nhớ được truy cập thường xuyên

### 4. Bảo Mật và Quyền Riêng Tư
- Triển khai kiểm soát truy cập
- Mã hóa các bộ nhớ nhạy cảm
- Cung cấp khả năng xóa bộ nhớ
- Kiểm toán truy cập bộ nhớ

### 5. Hiệu Suất
- Sử dụng hoạt động async để truy cập bộ nhớ
- Batch các thao tác ghi bộ nhớ
- Index các trường được truy vấn thường xuyên
- Xem xét các tầng bộ nhớ (nóng/ấm/lạnh)
