using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace OllamaArticleWriter;

// ─────────────────────────────────────────────────────────────────────────────
//  Ollama Article Writer — streaming + a small "save" tool  (gemma4:e2b)
//
//  The pattern for streaming LONG output with a tool:
//    • The model STREAMS the whole article as normal response text (content),
//      so you watch it write live, token by token.
//    • Then it calls a SMALL tool — save_article(title, filename) — carrying
//      only metadata. The heavy article body is NOT in the tool arguments;
//      our harness takes it from the streamed text it already captured.
//
//  Why not put the article inside the tool call? Because Ollama delivers tool
//  arguments as one atomic chunk — you'd get the whole article at once with no
//  streaming. Keeping large payloads in streamed content, and tool calls small,
//  is the general rule for responsive writing apps.
// ─────────────────────────────────────────────────────────────────────────────

internal class Program
{
    private const string OllamaUrl = "http://localhost:11434/api/chat";
    private const string Model = "gemma4:e2b";

    private static readonly HttpClient http = new() { Timeout = TimeSpan.FromMinutes(5) };

    // The most recent article body we streamed as content. The save_article tool
    // reads from here instead of from its own arguments — that's the whole point:
    // the body was streamed, not stuffed into the tool call.
    private static string articleBody = "";

    private static readonly Dictionary<string, Func<JsonObject, string>> toolImplementations = new()
    {
        ["save_article"] = args =>
        {
            string title = (args["title"]?.GetValue<string>() ?? "Untitled").Trim();
            string filename = (args["filename"]?.GetValue<string>() ?? "article.md").Trim();

            if (string.IsNullOrWhiteSpace(articleBody))
                return "Error: no article text was streamed, so there is nothing to save.";

            // Sanitize: strip any directory parts and force a .md extension.
            filename = Path.GetFileName(filename);
            if (!filename.EndsWith(".md", StringComparison.OrdinalIgnoreCase))
                filename += ".md";

            string dir = Path.Combine(Directory.GetCurrentDirectory(), "articles");
            Directory.CreateDirectory(dir);
            string path = Path.Combine(dir, filename);

            // The saved file = title as an H1 + the streamed body.
            string markdown = $"# {title}\n\n{articleBody.Trim()}\n";
            File.WriteAllText(path, markdown);

            return $"Saved '{title}' ({articleBody.Trim().Length} chars) to {path}";
        },
    };

    private static async Task Main(string[] args)
    {
        var messages = new JsonArray
        {
            new JsonObject
            {
                ["role"] = "system",
                ["content"] =
                    "You are an article writer.\n" +
                    "STEP 1: Write the COMPLETE article directly as your response text — a short " +
                    "title line followed by 2–4 concise paragraphs in Markdown. Write it as normal " +
                    "prose in your response. Do NOT put the article text inside a tool call.\n" +
                    "STEP 2: After the article is written, call the save_article tool with a concise " +
                    "`title` and a kebab-case `filename` ending in .md. The article body is taken " +
                    "automatically from your response text, so the tool only needs title and filename.",
            },
        };

        string topic = args.Length > 0
            ? string.Join(' ', args)
            : "the benefits of running LLMs locally";

        messages.Add(new JsonObject
        {
            ["role"] = "user",
            ["content"] = $"Write a short article about {topic}, then save it.",
        });

        Console.WriteLine($"Model : {Model}");
        Console.WriteLine($"Topic : {topic}");
        Console.WriteLine(new string('─', 60));

        const int maxTurns = 6;
        for (int turn = 1; turn <= maxTurns; turn++)
        {
            Console.WriteLine();
            Console.WriteLine($"╔══════════════════════ TURN {turn} ══════════════════════╗");

            JsonObject assistant = await StreamOneTurn(messages);
            messages.Add(assistant);

            // Whatever text streamed this turn becomes the article body available
            // to save_article. (Only set it when non-empty so a later "Saved!" turn
            // doesn't wipe the article we already captured.)
            string streamed = assistant["content"]?.GetValue<string>() ?? "";
            if (!string.IsNullOrWhiteSpace(streamed))
                articleBody = streamed;

            JsonArray? toolCalls = assistant["tool_calls"] as JsonArray;

            if (toolCalls is null || toolCalls.Count == 0)
            {
                // No tool call yet. If the model wrote the article but didn't save it,
                // nudge it to call save_article. If it wrote nothing, nudge to write.
                Console.WriteLine();
                if (string.IsNullOrWhiteSpace(articleBody))
                {
                    Console.WriteLine("── (no article text yet — asking the model to write one) ──");
                    messages.Add(new JsonObject
                    {
                        ["role"] = "user",
                        ["content"] = "Please write the article now.",
                    });
                }
                else
                {
                    Console.WriteLine("── (article written but not saved — asking it to call save_article) ──");
                    messages.Add(new JsonObject
                    {
                        ["role"] = "user",
                        ["content"] = "Now call save_article with a title and a kebab-case .md filename.",
                    });
                }
                continue;
            }

            // Run the tool call(s).
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

                Console.WriteLine($"  [RESULT   ] {result}");

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
                Console.WriteLine("── Done. Article streamed above and saved to disk. ──");
                return;
            }
        }

        Console.WriteLine("Reached max turns.");
    }

    // ── Streams one assistant turn. Prints article text live; shows a subtle
    //    marker while the model is "thinking"; accumulates content + tool_calls. ──
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
                // gemma streams its chain of thought in a "thinking" field first.
                // We don't put that in the article — just show a one-time notice.
                string? thinking = msg["thinking"]?.GetValue<string>();
                if (!string.IsNullOrEmpty(thinking) && !thinkingNoticeShown && !printedHeader)
                {
                    Console.Write("(thinking…) ");
                    thinkingNoticeShown = true;
                }

                // Article text — print each token as it arrives.
                string? piece = msg["content"]?.GetValue<string>();
                if (!string.IsNullOrEmpty(piece))
                {
                    if (!printedHeader)
                    {
                        if (thinkingNoticeShown) Console.WriteLine();
                        Console.WriteLine("── streaming article ──");
                        printedHeader = true;
                    }
                    Console.Write(piece);          // live token output
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

    // ── The single small tool: save_article(title, filename). Note there is NO
    //    "body" parameter — the body comes from the streamed content. ──
    private static JsonArray BuildToolSchemas()
    {
        return new JsonArray
        {
            new JsonObject
            {
                ["type"] = "function",
                ["function"] = new JsonObject
                {
                    ["name"] = "save_article",
                    ["description"] =
                        "Save the article you just wrote to a Markdown file. The article body is " +
                        "taken from your response text automatically — only provide a title and filename.",
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
                                ["description"] = "A kebab-case file name ending in .md, e.g. local-llms.md",
                            },
                        },
                        ["required"] = new JsonArray("title", "filename"),
                    },
                },
            },
        };
    }
}
