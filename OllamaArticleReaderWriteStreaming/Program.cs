using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace OllamaArticleReaderWriteStreaming;

// ─────────────────────────────────────────────────────────────────────────────
//  Ollama Article Reader → Writer (streaming)   (gemma4:e2b)
//
//  A two-tool agentic loop:
//    1. read_file(filename)      — the model reads source material from disk.
//    2. (stream)                 — using that material, it STREAMS a polished
//                                   article as response text, token by token.
//    3. save_article(title, filename) — a small tool persists the streamed body.
//
//  So the model READS a file, WRITES from it, and the writing streams live.
//  As before, the article body travels as streamed content — never inside a
//  tool call — so you watch it appear and the tool calls stay small.
// ─────────────────────────────────────────────────────────────────────────────

internal class Program
{
    private const string OllamaUrl = "http://localhost:11434/api/chat";
    private const string Model = "gemma4:e2b";

    private static readonly HttpClient http = new() { Timeout = TimeSpan.FromMinutes(5) };

    // The most recent article text we streamed as content. save_article reads from
    // here — the body was streamed, not passed through the tool arguments.
    private static string articleBody = "";

    private static readonly Dictionary<string, Func<JsonObject, string>> toolImplementations = new()
    {
        // ── TOOL 1: read a source file and return its contents to the model ──
        ["read_file"] = args =>
        {
            string filename = (args["filename"]?.GetValue<string>() ?? "").Trim();
            string dir = Path.Combine(Directory.GetCurrentDirectory(), "sources");

            if (string.IsNullOrWhiteSpace(filename))
                return ListSources(dir, "No filename given.");

            filename = Path.GetFileName(filename);   // strip any directory parts
            string path = Path.Combine(dir, filename);

            if (!File.Exists(path))
                return ListSources(dir, $"File '{filename}' not found.");

            string content = File.ReadAllText(path);
            return $"Contents of {filename}:\n\n{content}";
        },

        // ── TOOL 2: save the streamed article to disk (body from streamed content) ──
        ["save_article"] = args =>
        {
            string title = (args["title"]?.GetValue<string>() ?? "Untitled").Trim();
            string filename = (args["filename"]?.GetValue<string>() ?? "article.md").Trim();

            if (string.IsNullOrWhiteSpace(articleBody))
                return "Error: no article text was streamed, so there is nothing to save.";

            filename = Path.GetFileName(filename);
            if (!filename.EndsWith(".md", StringComparison.OrdinalIgnoreCase))
                filename += ".md";

            string dir = Path.Combine(Directory.GetCurrentDirectory(), "articles");
            Directory.CreateDirectory(dir);
            string path = Path.Combine(dir, filename);

            string markdown = $"# {title}\n\n{articleBody.Trim()}\n";
            File.WriteAllText(path, markdown);

            return $"Saved '{title}' ({articleBody.Trim().Length} chars) to {path}";
        },
    };

    private static string ListSources(string dir, string prefix)
    {
        if (!Directory.Exists(dir))
            return $"{prefix} (No sources directory exists.)";
        var files = Directory.GetFiles(dir).Select(Path.GetFileName);
        return $"{prefix} Available source files: {string.Join(", ", files)}";
    }

    private static async Task Main(string[] args)
    {
        // Which source file to base the article on (defaults to the sample notes).
        // NOTE: the argument is a SOURCE FILENAME to read — not a topic. To write
        // from a topic, use the OllamaArticleWriter project instead.
        string sourceFile = args.Length > 0 ? args[0] : "raw-notes.md";

        // Fail fast: if the source file doesn't exist, don't start the model loop.
        // (Otherwise the model reads nothing, apologizes, and we spin uselessly.)
        string sourcesDir = Path.Combine(Directory.GetCurrentDirectory(), "sources");
        string sourcePath = Path.Combine(sourcesDir, Path.GetFileName(sourceFile));
        if (!File.Exists(sourcePath))
        {
            Console.WriteLine($"Source file '{sourceFile}' was not found in the sources/ folder.");
            if (Directory.Exists(sourcesDir))
                Console.WriteLine($"Available source files: {string.Join(", ", Directory.GetFiles(sourcesDir).Select(Path.GetFileName))}");
            Console.WriteLine();
            Console.WriteLine("This project READS a source file and writes from it. The argument is a");
            Console.WriteLine("filename, not a topic. To write about a topic (e.g. \"claude\"), either:");
            Console.WriteLine("  • add sources/claude.md and run:  dotnet run claude.md");
            Console.WriteLine("  • or use the OllamaArticleWriter project, which writes from a topic.");
            return;
        }

        var messages = new JsonArray
        {
            new JsonObject
            {
                ["role"] = "system",
                ["content"] =
                    "You are a writing assistant that turns rough source notes into a polished article.\n" +
                    "STEP 1: Call the read_file tool to read the source file you are given.\n" +
                    "STEP 2: Using the notes, write a COMPLETE polished article directly as your response " +
                    "text — a title line followed by 3–4 concise paragraphs in Markdown. Write it as normal " +
                    "prose in your response. Do NOT put the article text inside a tool call.\n" +
                    "STEP 3: After writing, call save_article with a concise `title` and a kebab-case " +
                    "`filename` ending in .md. The article body is taken automatically from your response text.",
            },
            new JsonObject
            {
                ["role"] = "user",
                ["content"] = $"Read the source file '{sourceFile}', then write and save a polished article based on it.",
            },
        };

        Console.WriteLine($"Model  : {Model}");
        Console.WriteLine($"Source : {sourceFile}");
        Console.WriteLine(new string('─', 60));

        const int maxTurns = 8;
        const int maxNudges = 2;   // stop nagging if the model won't make progress
        int nudges = 0;

        for (int turn = 1; turn <= maxTurns; turn++)
        {
            Console.WriteLine();
            Console.WriteLine($"╔══════════════════════ TURN {turn} ══════════════════════╗");

            JsonObject assistant = await StreamOneTurn(messages);
            messages.Add(assistant);

            // Capture streamed article text (ignore empty turns so a trailing
            // "Saved!" message doesn't overwrite the article we already have).
            string streamed = assistant["content"]?.GetValue<string>() ?? "";
            if (!string.IsNullOrWhiteSpace(streamed))
                articleBody = streamed;

            JsonArray? toolCalls = assistant["tool_calls"] as JsonArray;

            if (toolCalls is null || toolCalls.Count == 0)
            {
                Console.WriteLine();

                // The model made no tool call. Nudge it — but only a couple of times.
                // If it keeps refusing (e.g. it has nothing to write from), stop
                // instead of spinning out the same apology until maxTurns.
                if (++nudges > maxNudges)
                {
                    Console.WriteLine($"── Stopping: the model made no tool call after {maxNudges} nudges. ──");
                    Console.WriteLine("   It likely has no usable source content to write from.");
                    return;
                }

                if (string.IsNullOrWhiteSpace(articleBody))
                {
                    Console.WriteLine($"── (no article yet — asking the model to write from the source) [nudge {nudges}/{maxNudges}] ──");
                    messages.Add(new JsonObject
                    {
                        ["role"] = "user",
                        ["content"] = "Now write the article from the notes you read.",
                    });
                }
                else
                {
                    Console.WriteLine($"── (article written but not saved — asking it to call save_article) [nudge {nudges}/{maxNudges}] ──");
                    messages.Add(new JsonObject
                    {
                        ["role"] = "user",
                        ["content"] = "Now call save_article with a title and a kebab-case .md filename.",
                    });
                }
                continue;
            }

            // A tool call means real progress — reset the nudge counter.
            nudges = 0;

            Console.WriteLine();
            Console.WriteLine($"── AI ASKED TO CALL {toolCalls.Count} TOOL(S) ──");

            bool saved = false;
            foreach (JsonNode? callNode in toolCalls)
            {
                var fn = callNode!["function"]!.AsObject();
                string name = fn["name"]!.GetValue<string>();
                JsonObject callArgs = fn["arguments"] as JsonObject ?? new JsonObject();
                string? callId = callNode["id"]?.GetValue<string>();

                Console.WriteLine($"  [TOOL CALL] {name}({callArgs.ToJsonString()})");

                string result = toolImplementations.TryGetValue(name, out var impl)
                    ? impl(callArgs)
                    : $"Error: unknown tool '{name}'.";

                // read_file results can be long — show a short preview only.
                string preview = result.Length > 160 ? result[..160] + " …(truncated)" : result;
                Console.WriteLine($"  [RESULT   ] {preview}");

                var toolResultMessage = new JsonObject
                {
                    ["role"] = "tool",
                    ["tool_name"] = name,
                    ["content"] = result,
                };
                if (callId is not null)
                    toolResultMessage["tool_call_id"] = callId;
                messages.Add(toolResultMessage);

                if (name == "save_article" && result.StartsWith("Saved"))
                    saved = true;
            }

            if (saved)
            {
                Console.WriteLine();
                Console.WriteLine("── Done. Read the source, streamed the article, saved to disk. ──");
                return;
            }
        }

        Console.WriteLine("Reached max turns.");
    }

    // ── Streams one assistant turn: prints article text live, shows a thinking
    //    marker, accumulates content + tool_calls. ──
    private static async Task<JsonObject> StreamOneTurn(JsonArray conversation)
    {
        var requestBody = new JsonObject
        {
            ["model"] = Model,
            ["messages"] = conversation.DeepClone(),
            ["tools"] = BuildToolSchemas(),
            ["stream"] = true,
        };

        using var request = new HttpRequestMessage(HttpMethod.Post, OllamaUrl)
        {
            Content = JsonContent.Create(requestBody),
        };
        using HttpResponseMessage resp = await http.SendAsync(
            request, HttpCompletionOption.ResponseHeadersRead);
        resp.EnsureSuccessStatusCode();

        var fullContent = new StringBuilder();
        var collectedToolCalls = new JsonArray();
        bool printedHeader = false;
        bool thinkingNoticeShown = false;

        await using Stream stream = await resp.Content.ReadAsStreamAsync();
        using var reader = new StreamReader(stream);

        string? line;
        while ((line = await reader.ReadLineAsync()) is not null)
        {
            if (string.IsNullOrWhiteSpace(line)) continue;

            JsonObject chunk = JsonNode.Parse(line)!.AsObject();
            JsonObject? msg = chunk["message"] as JsonObject;

            if (msg is not null)
            {
                string? thinking = msg["thinking"]?.GetValue<string>();
                if (!string.IsNullOrEmpty(thinking) && !thinkingNoticeShown && !printedHeader)
                {
                    Console.Write("(thinking…) ");
                    thinkingNoticeShown = true;
                }

                string? piece = msg["content"]?.GetValue<string>();
                if (!string.IsNullOrEmpty(piece))
                {
                    if (!printedHeader)
                    {
                        if (thinkingNoticeShown) Console.WriteLine();
                        Console.WriteLine("── streaming article ──");
                        printedHeader = true;
                    }
                    Console.Write(piece);
                    fullContent.Append(piece);
                }

                if (msg["tool_calls"] is JsonArray calls)
                    foreach (JsonNode? c in calls)
                        collectedToolCalls.Add(c!.DeepClone());
            }

            if (chunk["done"]?.GetValue<bool>() == true)
                break;
        }

        if (printedHeader) Console.WriteLine();

        var assistant = new JsonObject
        {
            ["role"] = "assistant",
            ["content"] = fullContent.ToString(),
        };
        if (collectedToolCalls.Count > 0)
            assistant["tool_calls"] = collectedToolCalls;
        return assistant;
    }

    // ── Two tools: read_file(filename) and save_article(title, filename). ──
    private static JsonArray BuildToolSchemas()
    {
        return new JsonArray
        {
            new JsonObject
            {
                ["type"] = "function",
                ["function"] = new JsonObject
                {
                    ["name"] = "read_file",
                    ["description"] = "Read a source file from the sources folder and return its contents.",
                    ["parameters"] = new JsonObject
                    {
                        ["type"] = "object",
                        ["properties"] = new JsonObject
                        {
                            ["filename"] = new JsonObject
                            {
                                ["type"] = "string",
                                ["description"] = "The source file to read, e.g. raw-notes.md",
                            },
                        },
                        ["required"] = new JsonArray("filename"),
                    },
                },
            },
            new JsonObject
            {
                ["type"] = "function",
                ["function"] = new JsonObject
                {
                    ["name"] = "save_article",
                    ["description"] =
                        "Save the article you just wrote to a Markdown file. The article body is taken " +
                        "from your response text automatically — only provide a title and filename.",
                    ["parameters"] = new JsonObject
                    {
                        ["type"] = "object",
                        ["properties"] = new JsonObject
                        {
                            ["title"] = new JsonObject
                            {
                                ["type"] = "string",
                                ["description"] = "A concise title for the article",
                            },
                            ["filename"] = new JsonObject
                            {
                                ["type"] = "string",
                                ["description"] = "A kebab-case file name ending in .md",
                            },
                        },
                        ["required"] = new JsonArray("title", "filename"),
                    },
                },
            },
        };
    }
}
