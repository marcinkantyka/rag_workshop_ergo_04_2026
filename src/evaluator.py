"""
RAG evaluation helpers.

Implements the four RAGAS-equivalent metrics directly via our LLM client,
avoiding RAGAS's internal OpenAI client management which breaks with custom endpoints.
"""

from __future__ import annotations

import json
import os

import pandas as pd
from tqdm import tqdm

METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

_EVAL_PROMPT = """\
You are evaluating a RAG system. Score the response on all four dimensions.

Return ONLY a JSON object — no explanation, no markdown, just the JSON:
{{
  "faithfulness": <0.0-1.0>,
  "answer_relevancy": <0.0-1.0>,
  "context_precision": <0.0-1.0>,
  "context_recall": <0.0-1.0>
}}

Scoring criteria:
- faithfulness: Are all claims in the answer supported by the retrieved context? (1.0 = fully grounded, 0.0 = fabricated)
- answer_relevancy: Does the answer directly address the question asked? (1.0 = fully on-topic, 0.0 = irrelevant)
- context_precision: What fraction of retrieved chunks are actually relevant to the question? (1.0 = all relevant)
- context_recall: How much of the ground-truth answer is covered by the retrieved contexts? (1.0 = fully covered)

Question: {question}

Ground truth: {ground_truth}

Retrieved contexts:
{contexts}

Answer: {answer}"""


# Available LLM models at the Netlight endpoint:
#   claude-haiku-4-5          claude-sonnet-4-5-20250929   claude-sonnet-4-6
#   gpt-4o                    gpt-4o-mini
#   gpt-4.1-mini              gpt-4.1-nano
#   gpt-5                     gpt-5-mini                   gpt-5-nano
#   gpt-5-fast                azure-openai-gpt-5-fast
#   gpt-5.1
def run_evaluation(
    rag_results: list[dict],
    llm_client,
    judge_model: str = "claude-haiku-4-5",
) -> pd.DataFrame:
    """
    Evaluate RAG results using our LLM client directly.

    Args:
        rag_results: output of RAGPipeline.ask() calls, each with ground_truth added
        llm_client: OpenAI-compatible client (already configured with base_url and api_key)
        judge_model: model to use as judge

    Returns:
        DataFrame with columns: question, faithfulness, answer_relevancy,
        context_precision, context_recall
    """
    # TODO:
    # For each result in rag_results, build the evaluation prompt using _EVAL_PROMPT.format(...)
    # Call llm_client.chat.completions.create() with the prompt.
    # Parse the JSON scores from the response.
    # Return a DataFrame with columns: question + the 4 metric names.
    # Use tqdm(rag_results, desc="Evaluating") for a progress bar.
    # On parse errors, fall back to 0.0 for all metrics.
    raise NotImplementedError


def summarise(df: pd.DataFrame) -> pd.Series:
    """Return mean scores for each metric column present in df."""
    cols = [c for c in METRIC_NAMES if c in df.columns]
    return df[cols].mean().round(3)


def write_promptfoo_config(
    rag_results: list[dict],
    output_path: str = "promptfooconfig.yaml",
    judge_model: str = "claude-haiku-4-5",
) -> str:
    """
    Generate a promptfoo evaluation config YAML from pre-computed RAG results.

    promptfoo uses an 'echo' provider so the pre-computed answers are evaluated
    as-is, without re-running the pipeline. An LLM judge scores each answer
    against the question, ground truth, and retrieved contexts.

    Args:
        rag_results: list of dicts with keys question, answer, contexts, ground_truth
        output_path: where to write the YAML file
        judge_model: model used by promptfoo as the LLM judge

    Returns:
        output_path (for chaining)
    """
    # TODO:
    # 1. Import yaml.
    # 2. Build a list of test dicts. Each test has:
    #    - "vars": {"question", "answer", "ground_truth", "contexts"} (join contexts with "\n---\n")
    #    - "assert": one llm-rubric assertion that checks faithfulness + relevance
    # 3. Build the top-level config dict with keys:
    #    - "description", "providers" ([{"id": "echo"}]),
    #    - "prompts" (["{{answer}}"]),
    #    - "defaultTest" with "options.provider" pointing to judge_model via
    #       openai:chat:<model> with apiBaseUrl=${NETLIGHT_BASE_URL} and apiKey=${NETLIGHT_API_KEY}
    #    - "tests"
    # 4. Write to output_path with yaml.dump(..., default_flow_style=False, allow_unicode=True, sort_keys=False)
    # 5. Return output_path.
    raise NotImplementedError


def compare_pipelines(
    results_a: list[dict],
    results_b: list[dict],
    llm_client,
    label_a: str = "Pipeline A",
    label_b: str = "Pipeline B",
    judge_model: str = "claude-haiku-4-5",
) -> pd.DataFrame:
    """Evaluate two pipelines and return a side-by-side summary DataFrame."""
    df_a = run_evaluation(results_a, llm_client, judge_model=judge_model)
    df_b = run_evaluation(results_b, llm_client, judge_model=judge_model)

    summary = pd.DataFrame(
        {label_a: summarise(df_a), label_b: summarise(df_b)}
    )
    summary["delta"] = (summary[label_b] - summary[label_a]).round(3)
    return summary
