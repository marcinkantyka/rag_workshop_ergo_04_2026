"""
RAGPipeline and AdvancedRAGPipeline — the complete generation step.
"""

from __future__ import annotations

import json
import os

from openai import OpenAI

from .embedder import embed_query, DEFAULT_MODEL
from .vector_store import retrieve

# Available LLM models at the Netlight endpoint:
#   claude-haiku-4-5          claude-sonnet-4-5-20250929   claude-sonnet-4-6
#   gpt-4o                    gpt-4o-mini
#   gpt-4.1-mini              gpt-4.1-nano
#   gpt-5                     gpt-5-mini                   gpt-5-nano
#   gpt-5-fast                azure-openai-gpt-5-fast
#   gpt-5.1
FAST_MODEL = "claude-haiku-4-5"
SMART_MODEL = "claude-sonnet-4-5"


def get_llm_client() -> OpenAI:
    return OpenAI(
        base_url=os.getenv("NETLIGHT_BASE_URL", "https://llm.netlight.com/v1"),
        api_key=os.getenv("NETLIGHT_API_KEY"),
    )

DEFAULT_SYSTEM_PROMPT = """\
You are a knowledgeable insurance and technology assistant.
Answer ONLY using information from the numbered context chunks provided.
Cite sources using the chunk number, e.g. "According to [1]..."
If the context is insufficient, say so explicitly. Never fabricate facts."""


def _build_prompt(question: str, chunks: list[dict]) -> str:
    """
    Format retrieved chunks + question into a single prompt string.

    Each chunk should be numbered [1], [2], ... and include its source name
    and similarity score. The question follows at the end.
    """
    # TODO: Iterate over chunks (enumerate starting at 1). For each chunk, add:
    #   "[i] Source: {chunk['source']} (relevance: {sim:.2f})"
    #   chunk["text"]
    #   a blank line
    # Finish with "Retrieved Context:\n{context}\nQuestion: {question}\n\nAnswer:"
    raise NotImplementedError


class RAGPipeline:
    """
    Baseline RAG pipeline: retrieve → prompt → generate.

    Returns a result dict with keys:
        question, answer, contexts (list[str]), sources (list[str])
    """

    def __init__(
        self,
        collection,
        llm_client: OpenAI,
        embed_model_name: str = DEFAULT_MODEL,
        llm_model: str = FAST_MODEL,
        top_k: int = 5,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        min_similarity: float = 0.0,
        max_tokens: int = 1024,
    ):
        self.collection = collection
        self.llm_client = llm_client
        self.embed_model_name = embed_model_name
        self.llm_model = llm_model
        self.top_k = top_k
        self.system_prompt = system_prompt
        self.min_similarity = min_similarity
        self.max_tokens = max_tokens

    def _retrieve(self, question: str) -> list[dict]:
        return retrieve(
            self.collection,
            question,
            top_k=self.top_k,
            model_name=self.embed_model_name,
            min_similarity=self.min_similarity,
        )

    def _generate(self, prompt: str) -> str:
        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content

    def ask(self, question: str) -> dict:
        """
        Run the full RAG pipeline for a single question.

        Returns: {"question", "answer", "contexts": list[str], "sources": list[str]}
        """
        # TODO:
        # 1. Call self._retrieve(question) to get the top-k chunk dicts.
        # 2. Call _build_prompt(question, chunks) to format the prompt.
        # 3. Call self._generate(prompt) to get the answer string.
        # 4. Return a dict with "question", "answer", "contexts" (list of chunk texts),
        #    and "sources" (list of chunk source names).
        raise NotImplementedError


class AdvancedRAGPipeline(RAGPipeline):
    """
    Extends RAGPipeline with optional advanced retrieval techniques.

    Flags:
        use_query_expansion  — LLM generates alternative query phrasings
        use_hyde             — LLM generates a hypothetical answer, embed that instead
        use_reranking        — cross-encoder re-scores initial candidates
        rerank_initial_k     — how many candidates to fetch before re-ranking
    """

    def __init__(
        self,
        *args,
        use_query_expansion: bool = False,
        use_hyde: bool = False,
        use_reranking: bool = False,
        rerank_initial_k: int = 20,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_query_expansion = use_query_expansion
        self.use_hyde = use_hyde
        self.use_reranking = use_reranking
        self.rerank_initial_k = rerank_initial_k

        if use_reranking:
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def _expand_query(self, question: str, n: int = 2) -> list[str]:
        prompt = (
            f"Generate {n} alternative phrasings of this question for semantic search.\n"
            f"Return ONLY a JSON array of strings.\n\nQuestion: {question}"
        )
        text = self.llm_client.chat.completions.create(
            model=self.llm_model, max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        ).choices[0].message.content.strip()

        if "```" in text:
            text = text.split("```")[1].lstrip("json").strip()

        try:
            alternatives = json.loads(text)
            return [question] + [a for a in alternatives[:n] if isinstance(a, str)]
        except (json.JSONDecodeError, ValueError):
            return [question]

    def _hyde_embedding(self, question: str) -> list[float]:
        hyde_prompt = (
            "Write a short factual paragraph (3-5 sentences) that directly answers this question.\n"
            "Do not introduce it — just write the paragraph.\n\nQuestion: " + question
        )
        hypo_doc = self.llm_client.chat.completions.create(
            model=self.llm_model, max_tokens=256,
            messages=[{"role": "user", "content": hyde_prompt}],
        ).choices[0].message.content
        return embed_query(hypo_doc, model_name=self.embed_model_name)

    def _rerank(self, question: str, candidates: list[dict], final_k: int) -> list[dict]:
        pairs = [(question, c["text"]) for c in candidates]
        scores = self._cross_encoder.predict(pairs)
        for chunk, score in zip(candidates, scores):
            chunk["rerank_score"] = float(score)
        return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:final_k]

    def _retrieve(self, question: str) -> list[dict]:
        fetch_k = self.rerank_initial_k if self.use_reranking else self.top_k

        if self.use_hyde:
            q_emb = self._hyde_embedding(question)
            results = self.collection.query(
                query_embeddings=[q_emb], n_results=fetch_k,
                include=["documents", "metadatas", "distances"],
            )
            candidates = [
                {"text": t, "source": m["source"], "similarity": 1 - d}
                for t, m, d in zip(
                    results["documents"][0], results["metadatas"][0], results["distances"][0]
                )
            ]
        elif self.use_query_expansion:
            queries = self._expand_query(question)
            seen: dict[str, dict] = {}
            for q in queries:
                for c in retrieve(self.collection, q, top_k=fetch_k,
                                   model_name=self.embed_model_name):
                    if c["text"] not in seen or c["similarity"] > seen[c["text"]]["similarity"]:
                        seen[c["text"]] = c
            candidates = sorted(seen.values(), key=lambda x: x["similarity"], reverse=True)
        else:
            candidates = retrieve(self.collection, question, top_k=fetch_k,
                                   model_name=self.embed_model_name,
                                   min_similarity=self.min_similarity)

        if self.use_reranking:
            candidates = self._rerank(question, candidates, final_k=self.top_k)

        return candidates[: self.top_k]
