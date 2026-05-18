---
name: github-repo
description: Clone and explore remote GitHub repositories. Use when the user shares a GitHub repo URL or asks to review/explore code in a remote repository.
---

# GitHub Repo Explorer

Clone remote GitHub repositories to a temp directory and explore them using standard tools (Glob, Grep, Read).

## When to Use

- User shares a GitHub repo URL and wants to understand the code
- User asks to review, explore, or search a remote repository
- User wants to compare patterns across external codebases

## Workflow

1. **Extract repo info** from the user's input (URL or `owner/repo` format)
2. **Clone to temp directory** with shallow clone for speed:
   ```bash
   git clone --depth 1 https://github.com/<owner>/<repo>.git /tmp/repos/<repo>
   ```
3. **Explore** using Glob, Grep, Read tools rooted at `/tmp/repos/<repo>`
4. **Answer questions** about the codebase

## URL Normalization

| User input | Clone URL |
|-----------|-----------|
| `https://github.com/owner/repo` | `https://github.com/owner/repo.git` |
| `github.com/owner/repo` | `https://github.com/owner/repo.git` |
| `owner/repo` | `https://github.com/owner/repo.git` |
| URL with `/tree/branch/path` | Clone repo, then navigate to path |

## Clone Options

- **Default**: `git clone --depth 1` (latest commit only, fast)
- **With history**: `git clone` (if user needs git log, blame, etc.)
- **Specific branch**: `git clone --depth 1 -b <branch>`
- **Already cloned**: Check if `/tmp/repos/<repo>` exists first to avoid re-cloning

## Tips

- Always check if the repo is already cloned before cloning again: `ls /tmp/repos/<repo>`
- For large repos, start with the README and directory structure before diving into code
- Use the Explore subagent for broad codebase understanding
- If the user only needs a single file, consider using `gh api` or WebFetch on the raw GitHub URL instead of cloning
- Clean up with `rm -rf /tmp/repos/<repo>` if the user is done (ask first)

## Quick Reference Commands

```bash
# Check if already cloned
ls /tmp/repos/<repo> 2>/dev/null

# Clone
git clone --depth 1 https://github.com/<owner>/<repo>.git /tmp/repos/<repo>

# Get directory overview
ls /tmp/repos/<repo>

# For private repos (uses gh CLI auth)
gh repo clone <owner>/<repo> /tmp/repos/<repo> -- --depth 1
```
