using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace OllamaWebSearch;

// ─────────────────────────────────────────────────────────────────────────────
//  Ollama Web Search tool  (gemma4:e2b)
//
//  Unlike the math/file tools, this tool reaches OUT to the internet. The flow:
//    1. The user asks a question.
//    2. The model calls web_search(query).
//    3. OUR harness performs a real HTTP search (Wikipedia API, keyless) and
//       returns the article intros as the tool result.
//    4. The model reads those results and answers, citing the source URLs.
//
//  Wikipedia covers a huge range of topics with no API key, but it is an
//  ENCYCLOPEDIA — it answers "what is X" well, but has no live data (today's
//  news, tomorrow's schedule). For live results, swap in Tavily/Brave (needs key).
//
//  Key idea: the model can't browse the web itself — it only emits a query.
//  The harness does the actual searching and hands back text. This is exactly
//  how "web search" works with any local model.
//
//  Note: tools here are ASYNC (Func<..., Task<string>>) because searching is an
//  HTTP call — see how the loop `await`s the tool.
// ─────────────────────────────────────────────────────────────────────────────

internal class Program
{
    private const string OllamaUrl = "http://localhost:11434/api/chat";
    private const string Model = "gemma4:e2b";

    private static readonly HttpClient http = CreateHttpClient();

    private static HttpClient CreateHttpClient()
    {
        var client = new HttpClient { Timeout = TimeSpan.FromMinutes(2) };
        // A User-Agent is polite (and some endpoints require one).
        client.DefaultRequestHeaders.UserAgent.ParseAdd("OllamaWebSearchDemo/1.0");
        return client;
    }

    // Tools are ASYNC here because web_search makes a network call.
    private static readonly Dictionary<string, Func<JsonObject, Task<string>>> toolImplementations = new()
    {
        ["web_search"] = WebSearch,
    };

    // ── The web_search tool: real HTTP call to the Wikipedia API ──
    //  One request does two things via generator=search:
    //    • finds the top matching articles for the query, and
    //    • returns each article's plain-text intro (prop=extracts).
    private static async Task<string> WebSearch(JsonObject args)
    {
        string query = (args["query"]?.GetValue<string>() ?? "").Trim();
        if (string.IsNullOrWhiteSpace(query))
            return "Error: empty search query.";

        string url = "https://en.wikipedia.org/w/api.php?action=query&format=json"
                     + "&generator=search&gsrsearch=" + Uri.EscapeDataString(query) + "&gsrlimit=3"
                     + "&prop=extracts|info&exintro=1&explaintext=1&exchars=1500"
                     + "&exlimit=max&inprop=url";

        string json;
        try
        {
            json = await http.GetStringAsync(url);
        }
        catch (Exception ex)
        {
            return $"Search failed (network error): {ex.Message}";
        }

        JsonObject root = JsonNode.Parse(json)!.AsObject();

        // Results live under query.pages, keyed by page id. "index" is the rank.
        if (root["query"]?["pages"] is not JsonObject pages || pages.Count == 0)
            return $"No Wikipedia articles found for \"{query}\".";

        var ranked = pages
            .Select(kv => kv.Value!.AsObject())
            .OrderBy(p => p["index"]?.GetValue<int>() ?? 999)
            .ToList();

        var sb = new StringBuilder();
        sb.AppendLine($"Wikipedia results for \"{query}\":");
        int i = 1;
        foreach (JsonObject page in ranked)
        {
            string title = page["title"]?.GetValue<string>() ?? "(untitled)";
            string extract = (page["extract"]?.GetValue<string>() ?? "").Trim();
            string pageUrl = page["fullurl"]?.GetValue<string>() ?? "";
            if (string.IsNullOrWhiteSpace(extract)) continue;   // skip stubs with no intro

            sb.AppendLine();
            sb.AppendLine($"[{i}] {title}: {extract}");
            if (!string.IsNullOrWhiteSpace(pageUrl))
                sb.AppendLine($"    Source: {pageUrl}");
            i++;
        }

        if (i == 1)   // found pages but none had a usable intro
            return $"No article summaries found for \"{query}\".";

        return sb.ToString();
    }

    private static async Task Main(string[] args)
    {
        var messages = new JsonArray
        {
            new JsonObject
            {
                ["role"] = "system",
                ["content"] =
                    "You are a helpful research assistant with a web_search tool.\n" +
                    "When the user asks about facts, people, companies, or topics you should look " +
                    "up, call web_search with a concise query FIRST. Then answer the user's question " +
                    "using ONLY the information in the search results, and cite the source URLs you used. " +
                    "If the results don't contain the answer, say so plainly.",
            },
        };

        string question = args.Length > 0
            ? string.Join(' ', args)
            : "What is Anthropic and what is Claude?";

        messages.Add(new JsonObject { ["role"] = "user", ["content"] = question });

        Console.WriteLine($"Model : {Model}");
        Console.WriteLine($"You   : {question}");
        Console.WriteLine(new string('─', 60));

        const int maxTurns = 6;
        for (int turn = 1; turn <= maxTurns; turn++)
        {
            Console.WriteLine();
            Console.WriteLine($"╔══════════════════════ TURN {turn} ══════════════════════╗");

            JsonObject assistant = await CallOllama(messages);
            messages.Add(assistant);

            JsonArray? toolCalls = assistant["tool_calls"] as JsonArray;

            // No tool calls -> the model's text is the final answer.
            if (toolCalls is null || toolCalls.Count == 0)
            {
                string content = assistant["content"]?.GetValue<string>() ?? "(empty response)";
                Console.WriteLine();
                Console.WriteLine("── FINAL ANSWER ──");
                Console.WriteLine($"Model : {content}");
                return;
            }

            Console.WriteLine();
            Console.WriteLine($"── AI ASKED TO CALL {toolCalls.Count} TOOL(S) ──");

            foreach (JsonNode? callNode in toolCalls)
            {
                var fn = callNode!["function"]!.AsObject();
                string name = fn["name"]!.GetValue<string>();
                JsonObject callArgs = fn["arguments"] as JsonObject ?? new JsonObject();
                string? callId = callNode["id"]?.GetValue<string>();

                Console.WriteLine($"  [TOOL CALL] {name}({callArgs.ToJsonString()})");

                // Tools are async here — the loop awaits the network call.
                string result = toolImplementations.TryGetValue(name, out var impl)
                    ? await impl(callArgs)
                    : $"Error: unknown tool '{name}'.";

                // Search results can be long — preview only in the console.
                string preview = result.Length > 300 ? result[..300] + " …(truncated)" : result;
                Console.WriteLine("  [SEARCH RESULTS returned to model]:");
                Console.WriteLine(Indent(preview, "    "));

                var toolResultMessage = new JsonObject
                {
                    ["role"] = "tool",
                    ["tool_name"] = name,
                    ["content"] = result,
                };
                if (callId is not null)
                    toolResultMessage["tool_call_id"] = callId;
                messages.Add(toolResultMessage);
            }
        }

        Console.WriteLine("Reached max turns without a final answer.");
    }

    private static async Task<JsonObject> CallOllama(JsonArray conversation)
    {
        var requestBody = new JsonObject
        {
            ["model"] = Model,
            ["messages"] = conversation.DeepClone(),
            ["tools"] = BuildToolSchemas(),
            ["stream"] = false,
        };

        using HttpResponseMessage resp = await http.PostAsJsonAsync(OllamaUrl, requestBody);
        resp.EnsureSuccessStatusCode();

        string json = await resp.Content.ReadAsStringAsync();
        JsonObject root = JsonNode.Parse(json)!.AsObject();
        return (JsonObject)root["message"]!.DeepClone();
    }

    // ── Tool schema: web_search(query) ──
    private static JsonArray BuildToolSchemas()
    {
        return new JsonArray
        {
            new JsonObject
            {
                ["type"] = "function",
                ["function"] = new JsonObject
                {
                    ["name"] = "web_search",
                    ["description"] =
                        "Search Wikipedia for encyclopedic information about a topic, person, company, " +
                        "place, or event. Returns article summaries with source URLs. Good for facts " +
                        "and background; not for live data like today's news or schedules.",
                    ["parameters"] = new JsonObject
                    {
                        ["type"] = "object",
                        ["properties"] = new JsonObject
                        {
                            ["query"] = new JsonObject
                            {
                                ["type"] = "string",
                                ["description"] = "The search query, e.g. 'Anthropic Claude AI'",
                            },
                        },
                        ["required"] = new JsonArray("query"),
                    },
                },
            },
        };
    }

    private static string Indent(string text, string pad) =>
        string.Join('\n', text.Split('\n').Select(line => pad + line));
}
