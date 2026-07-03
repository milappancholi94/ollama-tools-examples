using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace OllamaToolStreaming;

// ─────────────────────────────────────────────────────────────────────────────
//  Ollama Tool-Calling + STREAMING  (gemma4:e2b)
//
//  Same agentic loop as the non-streaming example, but with "stream": true.
//  Instead of one big JSON response, Ollama replies with a series of
//  newline-delimited JSON objects (NDJSON) — one chunk per step:
//
//    {"message":{"role":"assistant","content":"The "},"done":false}
//    {"message":{"role":"assistant","content":"result "},"done":false}
//    {"message":{"role":"assistant","content":"is 150."},"done":false}
//    {"message":{"role":"assistant","content":""},"done":true,"done_reason":"stop"}
//
//  We read line-by-line, print text the instant it arrives, and rebuild the
//  full assistant message (content + tool_calls) by accumulating the chunks.
//  Tool handling is identical: if the assembled message has tool_calls, we run
//  them, append the results, and stream the next turn.
// ─────────────────────────────────────────────────────────────────────────────

internal class Program
{
    private const string OllamaUrl = "http://localhost:11434/api/chat";
    private const string Model = "gemma4:e2b";

    private static readonly HttpClient http = new() { Timeout = TimeSpan.FromMinutes(5) };

    // ── The tool registry: name -> the C# function that does the work ──
    private static readonly Dictionary<string, Func<JsonObject, string>> toolImplementations = new()
    {
        ["add"]      = args => (GetNum(args, "a") + GetNum(args, "b")).ToString(),
        ["subtract"] = args => (GetNum(args, "a") - GetNum(args, "b")).ToString(),
        ["multiply"] = args => (GetNum(args, "a") * GetNum(args, "b")).ToString(),
        ["divide"]   = args =>
        {
            double b = GetNum(args, "b");
            if (b == 0) return "Error: cannot divide by zero.";
            return (GetNum(args, "a") / b).ToString();
        },
    };

    private static async Task Main(string[] args)
    {
        var messages = new JsonArray
        {
            new JsonObject
            {
                ["role"] = "system",
                ["content"] = "You are a helpful assistant. Use the provided math tools to " +
                              "perform any calculations instead of computing them yourself.",
            },
        };

        string userQuestion = args.Length > 0
            ? string.Join(' ', args)
            : "What is 24 multiplied by 7, and then subtract 18 from that result?";

        messages.Add(new JsonObject { ["role"] = "user", ["content"] = userQuestion });

        Console.WriteLine($"Model : {Model}");
        Console.WriteLine($"You   : {userQuestion}");
        Console.WriteLine(new string('─', 60));

        const int maxTurns = 8;
        for (int turn = 1; turn <= maxTurns; turn++)
        {
            Console.WriteLine();
            Console.WriteLine($"╔══════════════════════ TURN {turn} (streaming) ══════════════════════╗");

            // Stream one assistant turn. This prints text live AND returns the
            // fully-assembled assistant message so we can inspect tool_calls.
            JsonObject assistant = await StreamOneTurn(messages);
            messages.Add(assistant);

            JsonArray? toolCalls = assistant["tool_calls"] as JsonArray;

            // No tool calls -> the streamed text WAS the final answer.
            if (toolCalls is null || toolCalls.Count == 0)
            {
                Console.WriteLine();
                string finalText = assistant["content"]?.GetValue<string>() ?? "";
                if (string.IsNullOrWhiteSpace(finalText))
                {
                    // The model ended the turn with no tool calls AND no text.
                    // Nudge it once to actually state the answer, then stop.
                    Console.WriteLine("── (model returned an EMPTY final message — nudging it to answer) ──");
                    messages.Add(new JsonObject
                    {
                        ["role"] = "user",
                        ["content"] = "Please state the final numeric answer in one sentence.",
                    });
                    continue;
                }
                Console.WriteLine("── (no tool_calls — the streamed text above is the final answer) ──");
                return;
            }

            // The model wants tools. Run each and append the results.
            Console.WriteLine();
            Console.WriteLine($"── AI ASKED TO CALL {toolCalls.Count} TOOL(S) ──");

            foreach (JsonNode? callNode in toolCalls)
            {
                var fn = callNode!["function"]!.AsObject();
                string name = fn["name"]!.GetValue<string>();
                JsonObject callArgs = fn["arguments"] as JsonObject ?? new JsonObject();
                string? callId = callNode["id"]?.GetValue<string>();

                Console.WriteLine($"  [TOOL CALL] {name}({callArgs.ToJsonString()})");

                string result;
                if (toolImplementations.TryGetValue(name, out var impl))
                    result = impl(callArgs);
                else
                    result = $"Error: unknown tool '{name}'.";

                Console.WriteLine($"  [EXECUTED ] {name} -> {result}");

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

    // ── Streams one assistant turn, printing text as it arrives, and returns
    //    the reassembled assistant message (content + any tool_calls). ──
    private static async Task<JsonObject> StreamOneTurn(JsonArray conversation)
    {
        var requestBody = new JsonObject
        {
            ["model"] = Model,
            ["messages"] = conversation.DeepClone(),
            ["tools"] = BuildToolSchemas(),
            ["stream"] = true,   // ← the whole point: ask Ollama to stream chunks
        };

        using var request = new HttpRequestMessage(HttpMethod.Post, OllamaUrl)
        {
            Content = JsonContent.Create(requestBody),
        };

        // ResponseHeadersRead lets us start reading the body before it's fully
        // downloaded — essential for streaming (otherwise HttpClient buffers it all).
        using HttpResponseMessage resp = await http.SendAsync(
            request, HttpCompletionOption.ResponseHeadersRead);
        resp.EnsureSuccessStatusCode();

        // Accumulators — we rebuild the full assistant message from the chunks.
        var fullContent = new System.Text.StringBuilder();
        var collectedToolCalls = new JsonArray();
        bool printedStreamHeader = false;

        await using Stream stream = await resp.Content.ReadAsStreamAsync();
        using var reader = new StreamReader(stream);

        // Ollama streams NDJSON: one JSON object per line.
        string? line;
        while ((line = await reader.ReadLineAsync()) is not null)
        {
            if (string.IsNullOrWhiteSpace(line)) continue;

            JsonObject chunk = JsonNode.Parse(line)!.AsObject();
            JsonObject? msg = chunk["message"] as JsonObject;

            if (msg is not null)
            {
                // Print any text token the moment it arrives.
                string? piece = msg["content"]?.GetValue<string>();
                if (!string.IsNullOrEmpty(piece))
                {
                    if (!printedStreamHeader)
                    {
                        Console.Write("Model : ");
                        printedStreamHeader = true;
                    }
                    Console.Write(piece);              // live token output
                    fullContent.Append(piece);         // keep for history
                }

                // Tool calls may arrive in a chunk — collect them.
                if (msg["tool_calls"] is JsonArray calls)
                {
                    foreach (JsonNode? c in calls)
                        collectedToolCalls.Add(c!.DeepClone());
                }
            }

            // The final chunk has "done": true (+ timing / done_reason).
            if (chunk["done"]?.GetValue<bool>() == true)
                break;
        }

        if (printedStreamHeader) Console.WriteLine();  // newline after streamed text

        // Rebuild the assistant message exactly as the non-streaming API would return it.
        var assistant = new JsonObject
        {
            ["role"] = "assistant",
            ["content"] = fullContent.ToString(),
        };
        if (collectedToolCalls.Count > 0)
            assistant["tool_calls"] = collectedToolCalls;

        return assistant;
    }

    // ── Tool schemas: name, description, input schema, required fields ──
    private static JsonArray BuildToolSchemas()
    {
        static JsonObject MathTool(string name, string description) => new()
        {
            ["type"] = "function",
            ["function"] = new JsonObject
            {
                ["name"] = name,
                ["description"] = description,
                ["parameters"] = new JsonObject
                {
                    ["type"] = "object",
                    ["properties"] = new JsonObject
                    {
                        ["a"] = new JsonObject { ["type"] = "number", ["description"] = "The first operand" },
                        ["b"] = new JsonObject { ["type"] = "number", ["description"] = "The second operand" },
                    },
                    ["required"] = new JsonArray("a", "b"),
                },
            },
        };

        return new JsonArray
        {
            MathTool("add",      "Add two numbers together and return the sum."),
            MathTool("subtract", "Subtract the second number from the first."),
            MathTool("multiply", "Multiply two numbers and return the product."),
            MathTool("divide",   "Divide the first number by the second."),
        };
    }

    // ── Helper: read a numeric argument the model supplied (number or string) ──
    private static double GetNum(JsonObject args, string key)
    {
        JsonNode? node = args[key];
        if (node is null) return 0;
        return node.GetValueKind() == JsonValueKind.String
            ? double.Parse(node.GetValue<string>())
            : node.GetValue<double>();
    }
}
