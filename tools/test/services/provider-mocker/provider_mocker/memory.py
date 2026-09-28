"""Deterministic memory extraction, rewriting and message echo scenarios."""

import json
import logging
from typing import ClassVar

logger = logging.getLogger(__name__)
QUERY_CONTEXT_WORD_THRESHOLD = 6


def _content_text(content: object) -> str:
    """Read Chat text parts without interpreting media or changing the wire body."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        part["text"]
        for part in content
        if isinstance(part, dict)
        and part.get("type") == "text"
        and isinstance(part.get("text"), str)
    )


class MemoryScenario:
    EXTRACTION_KEYWORDS: ClassVar[dict[str, tuple[str, str]]] = {
        # Car-related facts
        "car": ("semantic", "User's car is a blue Tesla Model 3 from 2023"),
        "tesla": ("semantic", "User's car is a blue Tesla Model 3 from 2023"),
        "model 3": ("semantic", "User's car is a blue Tesla Model 3 from 2023"),
        "drive": ("semantic", "User's car is a blue Tesla Model 3 from 2023"),
        # Dog-related facts
        "dog": ("semantic", "User's dog's name is Max, a golden retriever"),
        "max": ("semantic", "User's dog's name is Max, a golden retriever"),
        "golden retriever": (
            "semantic",
            "User's dog's name is Max, a golden retriever",
        ),
        "pet": ("semantic", "User's dog's name is Max, a golden retriever"),
        # Project codename
        "phoenix": ("semantic", "User's secret project codename is Phoenix-2026"),
        "phoenix-2026": ("semantic", "User's secret project codename is Phoenix-2026"),
        "codename": ("semantic", "User's secret project codename is Phoenix-2026"),
        # Other personal facts
        "purple": ("semantic", "User's favorite color is purple"),
        "color": ("semantic", "User's favorite color is purple"),
        "google": ("semantic", "User works as a software engineer at Google"),
        "engineer": ("semantic", "User works as a software engineer at Google"),
        "japan": (
            "semantic",
            "User is planning a trip to Japan with a budget of $5000",
        ),
        "trip": ("semantic", "User is planning a trip to Japan with a budget of $5000"),
        "$5000": ("semantic", "User's budget is $5000"),
        "5000": ("semantic", "User's budget is $5000"),
        "budget": ("semantic", "User's budget is $5000"),
        "sarah": ("semantic", "User's friend Sarah is a doctor who lives in Boston"),
        "doctor": ("semantic", "User's friend Sarah is a doctor who lives in Boston"),
        "boston": ("semantic", "User's friend Sarah lives in Boston"),
        "friend": ("semantic", "User's friend Sarah is a doctor who lives in Boston"),
        "mit": (
            "semantic",
            "User graduated from MIT in 2020 with Computer Science degree",
        ),
        "college": (
            "semantic",
            "User graduated from MIT in 2020 with Computer Science degree",
        ),
        "university": (
            "semantic",
            "User graduated from MIT in 2020 with Computer Science degree",
        ),
        "graduated": (
            "semantic",
            "User graduated from MIT in 2020 with Computer Science degree",
        ),
        "computer science": (
            "semantic",
            "User graduated from MIT in 2020 with Computer Science degree",
        ),
        "italian place": (
            "semantic",
            "User's favorite restaurant is The Italian Place on 5th Avenue",
        ),
        "5th avenue": (
            "semantic",
            "User's favorite restaurant is The Italian Place on 5th Avenue",
        ),
        "tom": ("semantic", "User's brother is named Tom"),
        "anna": ("semantic", "Tom is getting married to Anna next spring"),
        "macbook": ("semantic", "User bought a MacBook Pro M3"),
        "m3": ("semantic", "User bought a MacBook Pro M3"),
        "sushi": ("episodic", "User had lunch at a sushi place downtown"),
        # User isolation tests - PIN and password
        "pin": ("semantic", "User's secret PIN is 9876"),
        "9876": ("semantic", "User's secret PIN is 9876"),
        "password": ("semantic", "User's password is hunter2"),
        "hunter2": ("semantic", "User's password is hunter2"),
        # Address tests
        "123 main": ("semantic", "User's home address is 123 Main Street, New York"),
        "main street": ("semantic", "User's home address is 123 Main Street, New York"),
        "home address": (
            "semantic",
            "User's home address is 123 Main Street, New York",
        ),
        "456 business": ("semantic", "User's work address is 456 Business Ave, Boston"),
        "business ave": ("semantic", "User's work address is 456 Business Ave, Boston"),
        "work address": ("semantic", "User's work address is 456 Business Ave, Boston"),
        # Deduplication test
        "phone": ("semantic", "User's phone number is 555-123-4567"),
        "555-123-4567": ("semantic", "User's phone number is 555-123-4567"),
        "phone number": ("semantic", "User's phone number is 555-123-4567"),
        # Wedding/multi-turn test
        "wedding": (
            "semantic",
            "User's wedding is on June 15th, 2026 with fiancée Emily",
        ),
        "june 15": ("semantic", "User's wedding is on June 15th, 2026"),
        "june": ("semantic", "User's wedding is on June 15th, 2026"),
        "2026": ("semantic", "User's wedding is on June 15th, 2026"),
        "emily": ("semantic", "User's fiancée is named Emily"),
        "fiancée": ("semantic", "User's fiancée is named Emily"),
        "beach": ("semantic", "User is having a beach wedding with 150 guests"),
        "venue": ("semantic", "User is having a beach wedding with 150 guests"),
        "beach venue": ("semantic", "User is having a beach wedding with 150 guests"),
        "150 people": ("semantic", "User is having 150 guests at the wedding"),
        "150 guests": ("semantic", "User is having 150 guests at the wedding"),
        "150": ("semantic", "User is having 150 guests at the wedding"),
        "guests": ("semantic", "User is having 150 guests at the wedding"),
        "$50,000": ("semantic", "User's wedding budget is $50,000"),
        "50000": ("semantic", "User's wedding budget is $50,000"),
    }

    def _is_extraction_prompt(self, messages: list[dict[str, str]]) -> bool:
        """Check if this is a memory extraction prompt."""
        for msg in messages:
            content = msg.get("content", "").lower()
            if "extract important information" in content:
                return True
            if "memory extraction system" in content:
                return True
        return False

    def _is_query_rewrite_prompt(self, messages: list[dict[str, str]]) -> bool:
        """Check if this is a query rewriting prompt."""
        for msg in messages:
            content = msg.get("content", "").lower()
            if "query rewriter" in content:
                return True
        return False

    def _extract_facts_from_messages(
        self, messages: list[dict[str, str]]
    ) -> list[dict]:
        """Extract facts from conversation using keyword matching.

        IMPORTANT: Only searches USER messages, not system prompts.
        The extraction system prompt contains example keywords (e.g., 'budget is $5000', 'MIT')
        that would cause false matches if we searched system messages too.
        """
        # Only look at user messages - NOT system messages which contain examples
        # that would trigger false keyword matches
        user_content = " ".join(
            msg.get("content", "") for msg in messages if msg.get("role") == "user"
        ).lower()

        facts = []
        seen_facts = set()  # Avoid duplicates

        for keyword, (fact_type, fact_content) in self.EXTRACTION_KEYWORDS.items():
            if keyword in user_content and fact_content not in seen_facts:
                facts.append({"type": fact_type, "content": fact_content})
                seen_facts.add(fact_content)

        logger.info(
            f"🧠 Smart extraction: found {len(facts)} facts from {len([m for m in messages if m.get('role') == 'user'])} user messages"
        )
        return facts

    @staticmethod
    def _parse_rewrite_input(
        messages: list[dict[str, str]],
    ) -> tuple[str, list[str]]:
        history_context = []
        for message in messages:
            if message.get("role") != "user":
                continue
            query, history = MemoryScenario._parse_rewrite_message(
                message.get("content", "")
            )
            history_context.extend(history)
            if query:
                return query, history_context
        return "", history_context

    @staticmethod
    def _parse_rewrite_message(content: str) -> tuple[str, list[str]]:
        query = ""
        history_context = []
        in_history = False
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("History:"):
                in_history = True
                continue
            if stripped.startswith("Query:") and "Rewritten" not in stripped:
                query = stripped.removeprefix("Query:").strip()
                in_history = False
                continue
            if in_history and "[user]:" in stripped:
                user_message = stripped.split("[user]:", 1)[-1].strip()
                if user_message:
                    history_context.append(user_message)
        return query, history_context

    @staticmethod
    def _last_user_query(messages: list[dict[str, str]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user":
                return message.get("content", "").partition("\n")[0].strip()
        return ""

    def _rewrite_query(self, messages: list[dict[str, str]]) -> str:
        """Query rewriting - enrich query with conversation context.

        Production behavior (from queryRewriteSystemPrompt):
        - PRESERVE query type: questions stay questions, statements stay statements
        - Add context from conversation history
        - Make query self-contained for semantic search

        The router sends query rewriting requests with format:
        [system]: You are a query rewriter...
        [user]: History:
               [user]: <previous messages>
               Query: <the actual query>
               Rewritten query:

        Example:
        - History: "My budget is $50K for a trip to Israel"
        - Query: "Which hotel should I choose?"
        - Rewritten: "Which hotel should I choose for my trip to Israel with $50K budget?"
        """
        query, history_context = self._parse_rewrite_input(messages)
        if not query:
            query = self._last_user_query(messages)

        if not query:
            return ""

        # If no history context, return query unchanged
        if not history_context:
            logger.info(f"🔄 Query unchanged (no history): '{query}'")
            return query

        # Simple context enrichment: append relevant context to the query
        # Keep the query as-is (preserve type) and add context
        context_summary = " ".join(history_context[-2:])  # Use last 2 history items

        # Only add context if query seems to need it (short or has pronouns)
        if len(query.split()) < QUERY_CONTEXT_WORD_THRESHOLD or any(
            word in query.lower() for word in ["it", "that", "this", "my"]
        ):
            # Append context in parentheses to keep query structure intact
            rewritten = f"{query} (context: {context_summary})"
            logger.info(f"🔄 Enriched query: '{query}' → '{rewritten}'")
            return rewritten

        # Query is already self-contained
        logger.info(f"🔄 Query unchanged (self-contained): '{query}'")
        return query

    def content(self, messages: list[dict]) -> str:
        # The codec may emit structured content even for text-only requests.
        # Scenario detection, extraction, rewriting and echo all share this view.
        messages = [
            {**message, "content": _content_text(message.get("content"))}
            for message in messages
        ]
        if self._is_extraction_prompt(messages):
            return json.dumps(self._extract_facts_from_messages(messages))
        if self._is_query_rewrite_prompt(messages):
            return self._rewrite_query(messages)
        return "\n".join(
            f"[{m.get('role', 'unknown')}]: {m.get('content', '')}" for m in messages
        )
