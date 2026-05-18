---
name: arxiv-reader
description: Reads and extracts content from Arxiv papers for Q&A. Use when the user shares an Arxiv link (arxiv.org/abs/... or arxiv.org/html/...) or asks about a paper from Arxiv.
---

# Arxiv Paper Reader

Read arxiv papers by converting them to markdown via [arxiv2md.org](https://arxiv2md.org), then answer questions about the content.

## When to Use

- User shares an arxiv link (any format: abs, html, pdf)
- User asks to read, summarize, or answer questions about an arxiv paper
- User references a paper by arxiv ID (e.g., "2401.12345")

## How It Works

Fetch the arxiv HTML version directly via WebFetch. Arxiv serves HTML for most recent papers at `arxiv.org/html/<id>`.

**Note:** arxiv2md.org is a client-side JS app and does NOT work with WebFetch. Use the arxiv HTML endpoint directly.

## Workflow

1. **Extract the paper ID** from the user's input (URL or bare ID like `2401.12345`)
2. **Fetch the HTML** using WebFetch:
   ```
   WebFetch(url="https://arxiv.org/html/<paper-id>", prompt="Extract the full paper content: title, authors, abstract, and all sections.")
   ```
3. **If the paper is long**, fetch in sections by asking for specific parts in the prompt:
   - "Extract only the abstract and introduction"
   - "Extract section 3 (Methods) in full detail"
   - "Extract the results and discussion sections"
4. **Answer the user's question** based on the extracted content

## URL Normalization

All formats should map to `arxiv.org/html/<id>`:

| User input | Fetch URL |
|-----------|-----------|
| `arxiv.org/abs/2401.12345` | `arxiv.org/html/2401.12345` |
| `arxiv.org/html/2401.12345v2` | `arxiv.org/html/2401.12345v2` |
| `arxiv.org/pdf/2401.12345` | `arxiv.org/html/2401.12345` |
| `2401.12345` | `arxiv.org/html/2401.12345` |

## Tips

- WebFetch summarizes long content. Use targeted prompts to get detail on specific sections.
- For detailed analysis, fetch the paper once with a broad prompt, then re-fetch targeting specific sections as needed.
- Always cite section numbers when referencing specific claims from the paper.
- If arxiv HTML is not available (older papers), fall back to the abs page: `arxiv.org/abs/<id>`.
