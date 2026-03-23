#!/usr/bin/env python3
"""
Skill 蒸馏 Demo — Copy 文案任务专项

聚焦文案生成，评分用简单数字（不要 JSON），提速。

用法:
    conda activate pr
    python demo/demo_distill_copy.py
"""

import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import anthropic
from dotenv import load_dotenv

load_dotenv()

from treeskill.config import GlobalConfig
from treeskill.llm import LLMClient
from treeskill.optimizer import APOEngine
from treeskill.schema import Feedback, Message, Skill, Trace
from treeskill.skill import save as save_skill

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────
STUDENT_MODEL = "intern-s1-pro"
STUDENT_BASE_URL = "https://chat.intern-ai.org.cn"
STUDENT_API_KEY = os.getenv(
    "INTERN_API_KEY",
    "REDACTED_INTERN_KEY",
)

TEACHER_MODEL = "MiniMax-M2.7"
TEACHER_BASE_URL = "https://api.minimaxi.com/anthropic"
TEACHER_API_KEY = os.getenv(
    "MINIMAX_API_KEY",
    "REDACTED_MINIMAX_KEY",
)

SKILL_PATH = Path("demo/data/minimax_frontend_skill.md")
DATA_PATH = Path("demo/data/copy_tasks.json")
OUTPUT_DIR = Path("demo/outputs/distill-copy")
NUM_ROUNDS = 3


# ── Helpers ────────────────────────────────────────────

def call_model(
    client: anthropic.Anthropic,
    model: str,
    system_prompt: str,
    user_text: str,
    max_tokens: int = 8000,
    temperature: float = 0.7,
    max_retries: int = 5,
) -> str:
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model,
                system=system_prompt,
                messages=[{"role": "user", "content": user_text}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text_parts = []
            for block in resp.content:
                if getattr(block, "type", None) == "text":
                    text_parts.append(block.text)
            return "".join(text_parts).strip()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = min(2 ** attempt, 30) * (0.5 + random.random())
            logger.warning(f"API error (attempt {attempt+1}): {e} — retry in {delay:.1f}s")
            time.sleep(delay)


def load_tasks():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    train = [t for t in tasks if t["split"] == "train"]
    val = [t for t in tasks if t["split"] == "val"]
    test = [t for t in tasks if t["split"] == "test"]
    logger.info(f"Tasks: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


def load_skill_md(path: Path) -> Skill:
    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        parts = text.split("---", 2)
        body = parts[2].strip() if len(parts) > 2 else text
    else:
        body = text
    return Skill(
        name="frontend-copy",
        description="Frontend copywriting skill using AIDA/PAS/FAB frameworks",
        system_prompt=body,
        target=(
            "Adapt this skill for Intern-S1-Pro. "
            "Focus on the copywriting section (AIDA, PAS, FAB frameworks). "
            "PRUNE unrelated sections (Three.js, shaders, complex animations). "
            "EXPAND copywriting rules with more examples. "
            "Keep Tailwind CSS and basic React/HTML guidance."
        ),
        version="v1.0",
    )


def save_html(code: str, task_id: str, label: str, output_dir: Path) -> Path:
    """Save generated code as standalone HTML."""
    html_dir = output_dir / "html"
    html_dir.mkdir(parents=True, exist_ok=True)

    # Extract code from markdown blocks
    m = re.search(r'```(?:html|tsx|jsx)\n(.*?)```', code, re.DOTALL)
    clean = m.group(1).strip() if m else code

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{task_id} - {label}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>body {{ margin: 0; font-family: system-ui, -apple-system, sans-serif; }}</style>
</head>
<body>
{clean}
</body>
</html>"""

    path = html_dir / f"{task_id}_{label}.html"
    path.write_text(html, encoding="utf-8")
    return path


# ── Judge ──────────────────────────────────────────────

JUDGE_SYSTEM = "You are a copywriting quality judge. Score 0.0 to 1.0. Return ONLY a decimal number, nothing else."

JUDGE_PROMPT = """Compare the Student's frontend copy against the Gold Standard.

Task: {task}
Key rules: {key_rules}

=== Gold Standard ===
{gold}

=== Student Output ===
{student}

Score criteria:
- Does the copy follow the specified framework (AIDA/PAS/FAB)? (0-0.3)
- Is the copy persuasive, specific, and free of placeholder text? (0-0.4)
- Does it match the gold standard quality and completeness? (0-0.3)

Return ONLY a number 0.0-1.0:"""


def judge_score(
    judge: anthropic.Anthropic,
    task: Dict,
    student: str,
    gold: str,
) -> float:
    prompt = JUDGE_PROMPT.format(
        task=task["task"],
        key_rules=", ".join(task["key_rules"]),
        gold=gold[:3000],
        student=student[:3000],
    )
    raw = call_model(judge, TEACHER_MODEL, JUDGE_SYSTEM, prompt, max_tokens=1000)
    try:
        m = re.search(r'(\d+\.?\d*)', raw)
        if m:
            return max(0.0, min(1.0, float(m.group(1))))
    except (ValueError, AttributeError):
        pass
    logger.warning(f"Judge parse fail: {raw[:50]}")
    return 0.5


# ── Main ───────────────────────────────────────────────

def main():
    t_start = time.time()
    logger.info("=" * 60)
    logger.info("Copy Distillation Demo")
    logger.info(f"  Teacher: {TEACHER_MODEL}  Student: {STUDENT_MODEL}")
    logger.info("=" * 60)

    train, val, test = load_tasks()
    skill = load_skill_md(SKILL_PATH)
    logger.info(f"Skill: {len(skill.system_prompt)} chars")

    student_client = anthropic.Anthropic(api_key=STUDENT_API_KEY, base_url=STUDENT_BASE_URL)
    teacher_client = anthropic.Anthropic(api_key=TEACHER_API_KEY, base_url=TEACHER_BASE_URL)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Gold standards ──
    logger.info("\nPhase 1: Teacher gold standards")
    gold_cache = OUTPUT_DIR / "gold.json"
    if gold_cache.exists():
        with open(gold_cache) as f:
            gold = json.load(f)
        logger.info(f"  Cached: {len(gold)} tasks")
    else:
        gold = {}
        all_tasks = train + val + test
        for task in all_tasks:
            logger.info(f"  Teacher: {task['id']}...")
            code = call_model(teacher_client, TEACHER_MODEL, skill.system_prompt, task["task"])
            gold[task["id"]] = code
            save_html(code, task["id"], "teacher", OUTPUT_DIR)
        with open(gold_cache, "w") as f:
            json.dump(gold, f, ensure_ascii=False, indent=2)

    # ── Phase 2: Baseline ──
    logger.info("\nPhase 2: Student baseline")
    baseline_scores = []
    for task in test:
        code = call_model(student_client, STUDENT_MODEL, skill.system_prompt, task["task"])
        save_html(code, task["id"], "student_baseline", OUTPUT_DIR)
        s = judge_score(teacher_client, task, code, gold[task["id"]])
        baseline_scores.append(s)
        logger.info(f"  {task['id']}: {s:.2f}")
    baseline_avg = sum(baseline_scores) / len(baseline_scores)
    logger.info(f"  Baseline: {baseline_avg:.2f}")

    # ── Phase 3: APO ──
    logger.info("\nPhase 3: APO distillation")

    os.environ["TREE_LLM_API_KEY"] = STUDENT_API_KEY
    os.environ["TREE_LLM_BASE_URL"] = STUDENT_BASE_URL
    os.environ["TREE_LLM_MODEL"] = STUDENT_MODEL
    os.environ["TREE_LLM_PROTOCOL"] = "anthropic"
    os.environ["TREE_LLM_JUDGE_API_KEY"] = TEACHER_API_KEY
    os.environ["TREE_LLM_JUDGE_BASE_URL"] = TEACHER_BASE_URL
    os.environ["TREE_LLM_JUDGE_MODEL"] = TEACHER_MODEL
    os.environ["TREE_LLM_JUDGE_PROTOCOL"] = "anthropic"

    from treeskill.config import LLMConfig, APOConfig
    config = GlobalConfig(
        llm=LLMConfig(),
        apo=APOConfig(
            gradient_accumulation_steps=4,
            beam_width=1,
            branch_factor=2,
            beam_rounds=1,
        ),
    )
    llm = LLMClient(config)
    engine = APOEngine(config, llm)

    # Simple score_fn
    def score_fn(prompt: str, traces: List[Trace]) -> float:
        scores = []
        for t in traces:
            output = call_model(student_client, STUDENT_MODEL, prompt, t.inputs[-1].content)
            expected = t.feedback.correction if t.feedback and t.feedback.correction else t.prediction.content
            raw = call_model(
                teacher_client, TEACHER_MODEL,
                JUDGE_SYSTEM,
                f"Gold:\n{expected[:2000]}\n\nStudent:\n{output[:2000]}\n\nScore 0.0-1.0:",
                max_tokens=1000,
            )
            try:
                m = re.search(r'(\d+\.?\d*)', raw)
                scores.append(max(0.0, min(1.0, float(m.group(1)))) if m else 0.5)
            except:
                scores.append(0.5)
        return sum(scores) / len(scores) if scores else 0.0

    engine._score_fn = score_fn
    best_skill = skill
    best_val = -1.0

    for rnd in range(1, NUM_ROUNDS + 1):
        logger.info(f"\n--- Round {rnd}/{NUM_ROUNDS} ---")

        # Collect traces on train
        traces = []
        for task in train:
            code = call_model(student_client, STUDENT_MODEL, best_skill.system_prompt, task["task"])
            gold_code = gold.get(task["id"], "")
            s = judge_score(teacher_client, task, code, gold_code)
            t = Trace(
                inputs=[Message(role="user", content=task["task"])],
                prediction=Message(role="assistant", content=code),
            )
            t.feedback = Feedback(score=s, critique=f"Score {s:.2f}", correction=gold_code)
            traces.append(t)
            logger.info(f"  {task['id']}: {s:.2f}")

        failures = [t for t in traces if t.feedback.score < 0.7]
        logger.info(f"  Failures: {len(failures)}/{len(traces)}")

        if not failures:
            logger.info("  All passing, skip")
            continue

        t0 = time.time()
        candidate = engine.optimize(best_skill, traces)
        logger.info(f"  Optimize: {time.time()-t0:.1f}s → {candidate.version}")

        # Validate
        val_scores = []
        for task in val:
            code = call_model(student_client, STUDENT_MODEL, candidate.system_prompt, task["task"])
            s = judge_score(teacher_client, task, code, gold[task["id"]])
            val_scores.append(s)
            logger.info(f"  val {task['id']}: {s:.2f}")
        val_avg = sum(val_scores) / len(val_scores)

        if val_avg > best_val:
            best_val = val_avg
            best_skill = candidate
            save_skill(best_skill, OUTPUT_DIR)
            logger.info(f"  ★ Accepted {best_skill.version} (val={val_avg:.2f})")
        else:
            logger.info(f"  Rejected (val={val_avg:.2f} <= {best_val:.2f})")

    # ── Phase 4: Final ──
    logger.info("\nPhase 4: Final test")
    final_scores = []
    for task in test:
        code = call_model(student_client, STUDENT_MODEL, best_skill.system_prompt, task["task"])
        save_html(code, task["id"], "student_final", OUTPUT_DIR)
        s = judge_score(teacher_client, task, code, gold[task["id"]])
        final_scores.append(s)
        logger.info(f"  {task['id']}: {s:.2f}")
    final_avg = sum(final_scores) / len(final_scores)

    elapsed = time.time() - t_start
    logger.info(f"\n{'='*60}")
    logger.info(f"Done! ({elapsed/60:.1f} min)")
    logger.info(f"  Baseline: {baseline_avg:.2f}")
    logger.info(f"  Final:    {final_avg:.2f}")
    logger.info(f"  Delta:    {final_avg - baseline_avg:+.2f}")
    logger.info(f"  Version:  {best_skill.version}")
    logger.info(f"  HTML:     {OUTPUT_DIR / 'html'}")

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump({
            "baseline": baseline_avg, "final": final_avg,
            "delta": final_avg - baseline_avg,
            "version": best_skill.version,
            "elapsed_min": elapsed / 60,
        }, f, indent=2)


if __name__ == "__main__":
    main()
