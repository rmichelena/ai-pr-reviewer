# AI PR Reviewer

Automated multi-model PR review inspired by Cursor's Bugbot. Runs on GitHub Actions, uses cheap LLM models via OpenRouter/NVIDIA NIM/Groq.

## Architecture

1. **Trigger:** `pull_request: [opened, synchronize, reopened]`
2. **N parallel passes** with cheap models, each gets diff in shuffled order
3. **Majority vote** — only keep findings flagged by ≥2 passes
4. **Validator pass** — filter false positives with a separate model call
5. **Post** review summary + inline comments via GitHub API
6. **Cost logged** per review (tokens used, estimated $)

## Config

All config via environment variables (GitHub Action inputs or repo secrets):

- `OPENROUTER_API_KEY` — OpenRouter API key (required if using OpenRouter)
- `NVIDIA_API_KEY` — NVIDIA NIM API key (required if using NVIDIA)
- `XAI_API_KEY` — x.ai API key (required if using xai provider)
- `AI_PROVIDER` — `openrouter` | `nvidia` | `groq` | `xai` (default: `openrouter`)
- `REVIEW_MODELS` — comma-separated model IDs (default: deepseek-v4-flash x3)
- `VALIDATOR_MODEL` — model for validator pass (default: same as review model)
- `NUM_PASSES` — parallel passes (default: 3)
- `MIN_VOTES` — minimum votes to keep a finding (default: 2)
- `MAX_DIFF_LINES` — skip review if diff exceeds this (default: 5000)

## Repo-level config

Optional `.github/review-rules.md` in the repo adds codebase-specific rules to every reviewer prompt.

## Cost estimate

With DeepSeek V4 Flash on OpenRouter ($0.084/MTok in, $0.168/MTok out):
- Average PR (~2000 tokens diff): **~$0.001 per pass, ~$0.005 per full review**
- 50 PRs/month: **~$0.25/month**

With NVIDIA NIM (free tier): **$0/month**

## License

MIT
