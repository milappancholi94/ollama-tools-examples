# ollama-tools-examples

Hands-on C#/.NET examples of **tool calling** (a.k.a. function calling) with a
local [Ollama](https://ollama.com) model. Each project is a small, self-contained
console app that builds on the previous one — from the bare tool-calling loop up
to streaming and web search.

All examples run against a **local** model (`gemma4:e2b`) — no API key, nothing
leaves your machine.

## The tool-calling loop (the core idea)

Every example follows the same loop:

1. Describe your tools to the model as **JSON Schema** (name, description, input schema).
2. Ask a question with the tool list attached.
3. The model replies asking to call a tool (`tool_calls`).
4. **Your code** runs the tool and appends the result as a `tool` message.
5. Send the whole conversation back; repeat until the model answers in words.

The model never runs anything itself — it only *asks*. Your harness does the work.

## Projects (in learning order)

| Project | Concept it teaches |
|---|---|
| [`OllamaToolCalling`](OllamaToolCalling) | The basic tool-calling loop — math tools; the model picks and chains them. Verbose logging of the request/response/tool-result at each step. |
| [`OllamaToolStreaming`](OllamaToolStreaming) | Streaming responses — `stream: true` and reading Ollama's newline-delimited JSON (NDJSON) so text prints token-by-token. |
| [`OllamaArticleWriterStreaming`](OllamaArticleWriterStreaming) | Writing from a **topic**, streamed live, then a small `save_article` tool persists it. Shows: stream the heavy payload as content, keep the tool call small. |
| [`OllamaArticleReaderWriteStreaming`](OllamaArticleReaderWriteStreaming) | A two-tool agentic flow — `read_file` pulls source material **in**, the model writes from it (streamed), then `save_article` writes it **out**. |
| [`OllamaWebSearch`](OllamaWebSearch) | A tool that reaches the internet — the model calls `web_search`, the harness queries Wikipedia, and the model answers from the results with citations. |

## Prerequisites

- [.NET SDK 10](https://dotnet.microsoft.com/download) (`dotnet --version`)
- [Ollama](https://ollama.com) running locally
- The model pulled: `ollama pull gemma4:e2b`

## Running an example

```bash
cd OllamaToolCalling
dotnet run                              # uses the default question
dotnet run -- "what is 24 times 7"      # pass your own (note the -- separator)
```

The `--` separates arguments for `dotnet run` from arguments passed to the app.

## Notes

- `bin/`, `obj/`, and any `.env` are git-ignored.
- The web-search example uses Wikipedia (keyless) — great for facts, but it has no
  live data (today's news, schedules).
