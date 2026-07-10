#!/usr/bin/env python3
"""AI PR Reviewer — Bugbot-inspired multi-model code review.

Runs N parallel passes with cheap LLM models, shuffles diff order per pass,
majority-votes findings, validates with a separate pass, and posts GitHub
review comments. Designed for GitHub Actions but can run locally.

Requires: requests, GITHUB_TOKEN, and at least one provider API key.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import requests

GITHUB_API = "https://api.github.com"

# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------

PROVIDERS = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "key_header": "Authorization",
        "key_prefix": "Bearer ",
        "extra_headers": {
            "HTTP-Referer": "https://github.com/rmichelena/ai-pr-reviewer",
            "X-Title": "AI PR Reviewer",
        },
    },
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "key_env": "NVIDIA_API_KEY",
        "key_header": "Authorization",
        "key_prefix": "Bearer ",
        "extra_headers": {},
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "key_header": "Authorization",
        "key_prefix": "Bearer ",
        "extra_headers": {},
    },
}


@dataclass
class Finding:
    severity: str
    title: str
    file: str
    line: int
    reasoning: str
    fix: str
    trace: str
    votes: int = 0
    validated: bool = True

    @property
    def key(self) -> str:
        """Dedup key: file + normalized title."""
        return f"{self.file}::{self.title.lower().strip()[:80]}"

    @classmethod
    def from_block(cls, block: str) -> "Finding | None":
        """Parse a ===FINDING=== block."""
        fields: dict[str, str] = {}
        for line in block.strip().splitlines():
            if "::" in line:
                k, _, v = line.partition("::")
                fields[k.strip().lower()] = v.strip()
        try:
            return cls(
                severity=fields.get("severity", "Low"),
                title=fields.get("title", "Untitled"),
                file=fields.get("file", ""),
                line=int(fields.get("line", 0) or 0),
                reasoning=fields.get("reasoning", ""),
                fix=fields.get("fix", ""),
                trace=fields.get("trace", "N/A"),
            )
        except (ValueError, TypeError):
            return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "title": self.title,
            "file": self.file,
            "line": self.line,
            "reasoning": self.reasoning,
            "fix": self.fix,
            "trace": self.trace,
            "votes": self.votes,
        }


# ---------------------------------------------------------------------------
# Diff handling
# ---------------------------------------------------------------------------

def fetch_diff(pr_number: int, repo: str, token: str) -> str:
    """Fetch the unified diff for a PR."""
    url = f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}"
    r = requests.get(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.v3.diff",
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.text


def parse_hunks(diff: str) -> list[str]:
    """Split a unified diff into hunks (one per file-change block)."""
    hunks: list[str] = []
    current: list[str] = []
    for line in diff.splitlines(keepends=True):
        if line.startswith("diff --git") and current:
            hunks.append("".join(current))
            current = []
        current.append(line)
    if current:
        hunks.append("".join(current))
    return hunks


def shuffle_hunks(hunks: list[str], seed: int) -> str:
    """Return diff with hunks in shuffled order."""
    shuffled = hunks.copy()
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    return "".join(shuffled)


def count_diff_lines(diff: str) -> int:
    """Count added+removed lines."""
    return sum(
        1
        for line in diff.splitlines()
        if line.startswith("+") or line.startswith("-")
    )


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

def call_llm(
    model: str,
    system: str,
    user: str,
    *,
    max_tokens: int = 8000,
    temperature: float = 0.3,
) -> tuple[str, dict[str, int]]:
    """Call a model via the configured provider. Returns (response_text, usage)."""
    provider_name = os.environ.get("AI_PROVIDER", "openrouter")
    provider = PROVIDERS.get(provider_name, PROVIDERS["openrouter"])
    key = os.environ.get(provider["key_env"], "")

    if not key:
        raise RuntimeError(f"Missing {provider['key_env']} for provider {provider_name}")

    url = f"{provider['base_url']}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        provider["key_header"]: f'{provider["key_prefix"]}{key}',
        **provider.get("extra_headers", {}),
    }

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    for attempt in range(3):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=120)
            if r.status_code == 429:
                wait = int(r.headers.get("retry-after", 5)) + 1
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            return text, {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
        except requests.RequestException as e:
            print(f"  LLM call attempt {attempt+1} failed: {e}")
            if attempt < 2:
                time.sleep(3 * (attempt + 1))

    return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


FINDING_RE = re.compile(r"===FINDING===(.*?)===END_FINDING===", re.DOTALL)


def parse_findings(text: str) -> list[Finding]:
    """Extract all ===FINDING=== blocks from model output."""
    findings = []
    for match in FINDING_RE.finditer(text):
        f = Finding.from_block(match.group(1))
        if f and f.file:
            findings.append(f)
    return findings


# ---------------------------------------------------------------------------
# Review prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an aggressive code reviewer focused on finding real bugs.
Investigate EVERY suspicious pattern. Err on the side of flagging potential issues.

Categories: logic bugs, security vulnerabilities, race conditions, performance issues,
data consistency, error handling gaps, regression risks.

DO NOT report: style nits, formatting, documentation issues, compiler warnings,
or anything a linter would catch.

For each finding use EXACTLY this format:
===FINDING===
severity:: [Critical|High|Medium|Low]
title:: <one line title>
file:: <path>
line:: <line_number>
reasoning:: <what the code does, why it's wrong, what triggers it>
fix:: <concrete fix suggestion>
trace:: <concrete trace for logic claims, or N/A>
===END_FINDING===

End with a short quality summary.
"""


def build_review_prompt(diff: str, rules: str, pr_title: str) -> str:
    parts = [f"# PR: {pr_title}\n"]
    if rules:
        parts.append(f"## Repo-specific rules\n{rules}\n")
    parts.append(f"## Diff to review\n```diff\n{diff}\n```")
    return "\n".join(parts)


VALIDATOR_SYSTEM = """\
You are a strict code review validator. You are given a finding from an automated review.
Your job: determine if this finding is a REAL issue or a FALSE POSITIVE.

Respond EXACTLY:
verdict:: KEEP
or
verdict:: DISMISS
reason:: <one sentence why>

A finding is a FALSE POSITIVE if:
- The code is actually correct and the reviewer misunderstood it.
- The issue cannot actually be triggered at runtime.
- The finding is about style/preference, not a real bug.
- The file/line reference is wrong or the code doesn't exist.

Be strict but fair. Only dismiss clear false positives.
"""


def build_validator_prompt(finding: Finding, diff: str) -> str:
    return (
        f"Finding to validate:\n"
        f"  severity: {finding.severity}\n"
        f"  title: {finding.title}\n"
        f"  file: {finding.file}\n"
        f"  line: {finding.line}\n"
        f"  reasoning: {finding.reasoning}\n"
        f"  fix: {finding.fix}\n\n"
        f"Relevant diff context:\n```diff\n{diff[:8000]}\n```"
    )


# ---------------------------------------------------------------------------
# GitHub posting
# ---------------------------------------------------------------------------

def get_existing_review_bodies(pr_number: int, repo: str, token: str) -> set[str]:
    """Get set of existing review comment bodies to avoid duplicates."""
    url = f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}/reviews"
    r = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        timeout=15,
    )
    if not r.ok:
        return set()
    bodies = set()
    for review in r.json():
        body = review.get("body", "")
        if body:
            bodies.add(body[:200])
    return bodies


def post_review(
    pr_number: int,
    repo: str,
    token: str,
    body: str,
    comments: list[dict[str, Any]],
) -> None:
    """Post a PR review with inline comments."""
    url = f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}/reviews"
    payload: dict[str, Any] = {
        "body": body,
        "event": "COMMENT",
        "comments": comments,
    }
    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        json=payload,
        timeout=30,
    )
    r.raise_for_status()
    print(f"  Posted review with {len(comments)} inline comments")


def get_pr_info(pr_number: int, repo: str, token: str) -> dict[str, Any]:
    url = f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}"
    r = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def load_review_rules() -> str:
    """Load .github/review-rules.md if present."""
    for path in (".github/review-rules.md", "REVIEW_RULES.md"):
        if os.path.exists(path):
            with open(path) as f:
                return f.read()
    return ""


def diff_line_to_position(diff_text: str, file_path: str, line: int) -> int | None:
    """Best-effort: map a file line number to a diff position for GitHub comments.

    GitHub wants the line number within the diff hunk, not the file.
    This is a simplified mapper that finds the right hunk.
    """
    # For now, we pass line as-is (GitHub accepts file line for single-hunk files)
    # A more robust solution parses the diff hunk headers
    return max(line, 1)


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------

MODEL_PRICING = {
    # OpenRouter prices per 1M tokens (input, output)
    "deepseek/deepseek-v4-flash": (0.084, 0.168),
    "deepseek/deepseek-v4-pro": (0.435, 0.87),
    "tencent/hy3-preview": (0.063, 0.21),
    "xiaomi/mimo-v2.5": (0.105, 0.28),
    "zai/glm-5.2": (0.42, 1.32),
    "stepfun/step-3.7-flash": (0.20, 1.15),
}


def estimate_cost(model: str, usage: dict[str, int]) -> float:
    """Estimate USD cost for a model call."""
    in_price, out_price = MODEL_PRICING.get(model, (0.10, 0.20))
    return (
        usage.get("prompt_tokens", 0) / 1_000_000 * in_price
        + usage.get("completion_tokens", 0) / 1_000_000 * out_price
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # --- Environment ---
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    event_path = os.environ.get("GITHUB_EVENT_PATH", "")
    token = os.environ.get("GITHUB_TOKEN", "")
    models_str = os.environ.get("REVIEW_MODELS", "deepseek/deepseek-v4-flash,deepseek/deepseek-v4-flash,deepseek/deepseek-v4-flash")
    validator_model = os.environ.get("VALIDATOR_MODEL", "deepseek/deepseek-v4-flash")
    min_votes = int(os.environ.get("MIN_VOTES", "2"))
    max_diff_lines = int(os.environ.get("MAX_DIFF_LINES", "5000"))

    if not event_path:
        print("ERROR: GITHUB_EVENT_PATH not set. This must run in GitHub Actions.")
        sys.exit(1)

    with open(event_path) as f:
        event = json.load(f)

    pr_number = event["pull_request"]["number"]
    pr_title = event["pull_request"]["title"]
    head_sha = event["pull_request"]["head"]["sha"]

    print(f"=== AI PR Review #{pr_number}: {pr_title} ===")
    print(f"Repo: {repo}")

    # --- Fetch diff ---
    diff = fetch_diff(pr_number, repo, token)
    diff_lines = count_diff_lines(diff)

    if diff_lines > max_diff_lines:
        print(f"Diff too large ({diff_lines} > {max_diff_lines}), skipping review.")
        post_review(pr_number, repo, token,
                    f"🤖 **AI Review skipped** — diff too large ({diff_lines} lines).", [])
        return

    print(f"Diff: {diff_lines} changed lines")

    hunks = parse_hunks(diff)
    if not hunks:
        print("No hunks to review.")
        return

    models = [m.strip() for m in models_str.split(",") if m.strip()]
    rules = load_review_rules()

    # --- Parallel passes ---
    all_findings: dict[str, Finding] = {}
    total_cost = 0.0
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for i, model in enumerate(models):
        print(f"\n--- Pass {i+1}/{len(models)}: {model} ---")
        shuffled = shuffle_hunks(hunks, seed=i * 42 + 7)
        # Truncate diff if extremely long (per-pass)
        if len(shuffled) > 100_000:
            shuffled = shuffled[:100_000] + "\n... (truncated)\n"

        user_msg = build_review_prompt(shuffled, rules, pr_title)
        text, usage = call_llm(model, SYSTEM_PROMPT, user_msg)

        cost = estimate_cost(model, usage)
        total_cost += cost
        for k in total_usage:
            total_usage[k] += usage.get(k, 0)

        findings = parse_findings(text)
        print(f"  Found {len(findings)} findings, cost ${cost:.4f}")

        for f in findings:
            if f.key in all_findings:
                all_findings[f.key].votes += 1
            else:
                f.votes = 1
                all_findings[f.key] = f

    print(f"\n=== Pre-vote: {len(all_findings)} unique findings ===")

    # --- Majority vote ---
    voted = {k: f for k, f in all_findings.items() if f.votes >= min_votes}
    print(f"=== Post-vote (≥{min_votes} votes): {len(voted)} findings ===")

    if not voted:
        print("No findings survived majority vote.")

    # --- Validator pass ---
    findings_list = list(voted.values())
    # Truncate diff for validator
    diff_snippet = diff[:20_000]

    for finding in findings_list:
        v_prompt = build_validator_prompt(finding, diff_snippet)
        v_text, v_usage = call_llm(validator_model, VALIDATOR_SYSTEM, v_prompt, max_tokens=200)
        total_cost += estimate_cost(validator_model, v_usage)
        for k in total_usage:
            total_usage[k] += v_usage.get(k, 0)

        if "DISMISS" in v_text.upper():
            finding.validated = False
            print(f"  ❌ DISMISSED: {finding.title}")
        else:
            print(f"  ✅ KEPT: {finding.title}")

    final = [f for f in findings_list if f.validated]
    final.sort(key=lambda f: {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}.get(f.severity, 4))

    print(f"\n=== Final: {len(final)} validated findings ===")

    # --- Build review body ---
    severity_emoji = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🔵"}

    body_lines = [
        f"## 🤖 AI PR Review",
        f"",
        f"**{len(final)} finding(s)** after {len(models)} parallel passes + validation.",
        f"",
    ]

    if final:
        body_lines.append("| Severity | File | Issue |")
        body_lines.append("|----------|------|-------|")
        # Note: GitHub renders tables in PR reviews, unlike Discord
        for f in final:
            emoji = severity_emoji.get(f.severity, "⚪")
            body_lines.append(f"| {emoji} {f.severity} | `{f.file}:{f.line}` | {f.title} |")
    else:
        body_lines.append("✅ No significant issues found. Code looks good!")

    body_lines.extend([
        f"",
        f"---",
        f"**Cost:** ${total_cost:.4f} · **Tokens:** {total_usage['total_tokens']:,} "
        f"(in: {total_usage['prompt_tokens']:,}, out: {total_usage['completion_tokens']:,})",
        f"**Models:** {', '.join(models)} · Validator: {validator_model}",
    ])

    # --- Build inline comments ---
    pr_info = get_pr_info(pr_number, repo, token)
    comments = []
    for f in final:
        position = diff_line_to_position(diff, f.file, f.line)
        comment_body = (
            f"{severity_emoji.get(f.severity, '⚪')} **{f.severity}: {f.title}**\n\n"
            f"{f.reasoning}\n\n"
            f"**Fix:** {f.fix}\n\n"
        )
        if f.trace and f.trace != "N/A":
            comment_body += f"**Trace:** `{f.trace}`\n\n"
        comment_body += f"_(votes: {f.votes}/{len(models)}, validated ✅)_"

        comments.append({
            "path": f.file,
            "position": position,
            "body": comment_body,
        })

    # --- Dedup against existing reviews ---
    existing = get_existing_review_bodies(pr_number, repo, token)
    if any(body[:200] in existing for body in ["\n".join(body_lines)]):
        print("Review already posted, skipping.")
        return

    # --- Post ---
    post_review(pr_number, repo, token, "\n".join(body_lines), comments)
    print(f"\n✅ Review posted. Total cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
