"""Context management for agents framework.

This module provides comprehensive context management capabilities including:
- Token counting with tiktoken support for accurate tokenization
- Message windowing for sliding window approaches
- Context budget tracking and enforcement
- Compaction strategies for managing context size
- Summarization for preserving context while reducing tokens

KUR-33, KUR-34, KUR-35: Phase 3 Context Management Implementation

Example:
    from agents_framework.context import (
        ContextManager,
        ContextBudget,
        TruncationCompactor,
        ConversationSummarizer,
    )

    # Create context manager with custom budget
    budget = ContextBudget(
        total_tokens=16000,
        conversation_ratio=0.65,
    )
    manager = ContextManager(budget=budget, model="gpt-4o")

    # Add messages
    await manager.add_message(message)

    # Check if compaction needed
    if manager.needs_compaction:
        compactor = TruncationCompactor()
        result = await compactor.compact(
            manager.get_messages(),
            manager.usage.total_tokens,
            target_tokens=8000,
            context_manager=manager,
        )
"""

# Manager components (KUR-33)
from .manager import (
    CharacterEstimateCounter,
    ContextBudget,
    ContextManager,
    ContextUsage,
    MessageWindow,
    TiktokenCounter,
    TokenCounter,
    TokenizerType,
    TokenWindow,
    get_token_counter,
    MODEL_TOKENIZER_MAP,
)

# Compaction components (KUR-34)
from .compactor import (
    BaseCompactor,
    CompactionConfig,
    CompactionResult,
    CompactionStrategy,
    CompactionTrigger,
    HybridCompactor,
    MessageImportance,
    SelectiveCompactor,
    SummaryCompactor,
    TruncationCompactor,
    create_compactor,
)

# Summarization components (KUR-35)
from .summarizer import (
    BaseSummarizer,
    ConversationSummarizer,
    DEFAULT_PROMPTS,
    ProgressiveSummary,
    SimpleSummarizer,
    SummaryCache,
    SummaryPrompt,
    SummaryType,
    SummarizerProtocol,
    create_summarizer,
)

__all__ = [
    # Manager (KUR-33)
    "CharacterEstimateCounter",
    "ContextBudget",
    "ContextManager",
    "ContextUsage",
    "MessageWindow",
    "MODEL_TOKENIZER_MAP",
    "TiktokenCounter",
    "TokenCounter",
    "TokenizerType",
    "TokenWindow",
    "get_token_counter",
    # Compaction (KUR-34)
    "BaseCompactor",
    "CompactionConfig",
    "CompactionResult",
    "CompactionStrategy",
    "CompactionTrigger",
    "HybridCompactor",
    "MessageImportance",
    "SelectiveCompactor",
    "SummaryCompactor",
    "TruncationCompactor",
    "create_compactor",
    # Summarization (KUR-35)
    "BaseSummarizer",
    "ConversationSummarizer",
    "DEFAULT_PROMPTS",
    "ProgressiveSummary",
    "SimpleSummarizer",
    "SummaryCache",
    "SummaryPrompt",
    "SummaryType",
    "SummarizerProtocol",
    "create_summarizer",
]
