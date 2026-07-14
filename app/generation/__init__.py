"""Reply generation: decide what the simulated prospect says.

``policy`` holds the deterministic reaction content (canned, seedable); ``llm``
wraps that behind a pluggable ``Generator`` with an optional OpenAI-backed impl.
"""
