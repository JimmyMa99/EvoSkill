"""Built-in scorer, gradient, and rewriter registrations.

Auto-imported by the registry to provide default components.
"""

from treeskill.registry import scorer, gradient, rewriter


# ---------------------------------------------------------------------------
# Built-in Scorers
# ---------------------------------------------------------------------------

@scorer("exact-match")
def exact_match(output: str, expected: str, context: dict) -> float:
    """Simple exact string match."""
    return 1.0 if output.strip().upper() == expected.strip().upper() else 0.0


@scorer("judge-grade", set_default=True)
def judge_grade(output: str, expected: str, context: dict) -> float:
    """Judge compares output vs expected and returns 0-1 score.

    Requires ``context["judge_fn"]`` — a callable that takes
    (output, expected) and returns a float score.
    """
    judge_fn = context.get("judge_fn")
    if judge_fn is None:
        # Fallback to exact match
        return 1.0 if output.strip() == expected.strip() else 0.0
    return judge_fn(output, expected)


# ---------------------------------------------------------------------------
# Built-in Gradient Templates
# ---------------------------------------------------------------------------

@gradient("simple")
def _gradient_simple():
    return (
        "You are an expert prompt engineer. Analyze the conversation failures "
        "below and explain concisely WHY the system prompt led to these problems. "
        "Return a bullet list of specific, actionable issues."
    )


@gradient("root-cause")
def _gradient_root_cause():
    return (
        "You are a senior prompt debugger. For each failure below, identify the "
        "ROOT CAUSE in the system prompt — what instruction is missing, ambiguous, "
        "or misleading? Be specific: quote the problematic part of the prompt and "
        "explain how it caused the failure. Return 3-5 bullets."
    )


@gradient("comprehensive")
def _gradient_comprehensive():
    return (
        "You are a prompt quality auditor. Evaluate the system prompt against "
        "these failures across these dimensions:\n"
        "1. Instruction clarity — are the rules unambiguous?\n"
        "2. Tone/style control — does the prompt prevent AI-sounding language?\n"
        "3. Scope constraints — does the prompt enforce length/format limits?\n"
        "4. Edge cases — does the prompt handle the scenarios that failed?\n"
        "Return a structured critique with specific fixes for each dimension."
    )


# ---------------------------------------------------------------------------
# Built-in Rewriter Templates
# ---------------------------------------------------------------------------

@rewriter("full-rewrite")
def _rewriter_full():
    return (
        "You are an expert prompt engineer. Based on the failure analysis below, "
        "rewrite the System Prompt to fix ALL identified issues. You may "
        "restructure, reorder, or add new instructions as needed. "
        "Preserve the core intent and any domain-specific knowledge. "
        "Return ONLY the new prompt — no commentary, no markdown code fences."
    )


@rewriter("conservative")
def _rewriter_conservative():
    return (
        "You are an expert prompt engineer. Based on the failure analysis below, "
        "revise the System Prompt to address the SINGLE MOST CRITICAL issue. "
        "Make minimal changes — keep the prompt close in tone, length, and "
        "structure to the original. Do not address more than one issue. "
        "Return ONLY the new prompt — no commentary, no markdown code fences."
    )


@rewriter("distill")
def _rewriter_distill():
    return (
        "You are a prompt distillation expert. Based on the failure analysis below, "
        "adapt the System Prompt for a smaller, less capable model. "
        "PRUNE sections the model cannot handle well. "
        "EXPAND key rules with explicit examples and explanations. "
        "Keep all tool/script references intact. "
        "Return ONLY the new prompt — no commentary, no markdown code fences."
    )
