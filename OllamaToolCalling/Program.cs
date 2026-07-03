using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace OllamaToolCalling;

// ─────────────────────────────────────────────────────────────────────────────
//  Ollama Tool-Calling Practice  (gemma4:e2b)
//
//  Flow (mirrors the Claude/OpenAI tool-use loop):
//    1. We describe our tools to the model as JSON Schema  -> "tools"
//    2. We ask a question with the list of tools attached
//    3. The model replies with a "tool_calls" block (which tool + arguments)
//    4. WE run the tool locally and append a "tool" result message
//    5. We send the whole conversation back so the model can answer in words
//    6. Repeat 3-5 until the model stops asking for tools
// ─────────────────────────────────────────────────────────────────────────────

internal class Program
{
    private const string OllamaUrl = "http://localhost:11434/api/chat";
    private const string Model = "gemma4:e2b";

    // One shared HttpClient for the whole program.
    private static readonly HttpClient http = new() { Timeout = TimeSpan.FromMinutes(5) };

    // ── The tool registry: name -> the actual C# function that does the work ──
    // Each function takes the model-supplied arguments (as JSON) and returns a string.
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

    // ── THE ENTRY POINT ──
    private static async Task Main(string[] args)
    {
        // The running conversation history (sent back in full every call).
        var messages = new JsonArray
        {
            new JsonObject
            {
                ["role"] = "system",
                ["content"] = "You are a helpful assistant. Use the provided math tools to " +
                              "perform any calculations instead of computing them yourself.",
            },
        };

        // The question we want answered. Pass it as a command-line argument, e.g.
        //   dotnet run -- "divide 100 by 8 then add 3.5"
        string userQuestion = args.Length > 0
            ? string.Join(' ', args)
            : "What is 24 multiplied by 7, and then subtract 18 from that result?";

        messages.Add(new JsonObject { ["role"] = "user", ["content"] = userQuestion });

        Console.WriteLine($"Model : {Model}");
        Console.WriteLine($"You   : {userQuestion}");
        Console.WriteLine(new string('─', 60));

        // ── The agentic loop ──
        const int maxTurns = 8;   // safety cap so we never loop forever
        for (int turn = 1; turn <= maxTurns; turn++)
        {
            Console.WriteLine();
            Console.WriteLine($"╔══════════════════════ TURN {turn} ══════════════════════╗");

            JsonObject assistant = await CallOllama(messages);

            // Keep the assistant's turn in history exactly as returned.
            messages.Add(assistant);

            JsonArray? toolCalls = assistant["tool_calls"] as JsonArray;

            // No tool calls -> the model gave its final natural-language answer.
            if (toolCalls is null || toolCalls.Count == 0)
            {
                string content = assistant["content"]?.GetValue<string>() ?? "(empty response)";
                Console.WriteLine();
                Console.WriteLine("── FINAL ANSWER (no tool_calls in response) ──");
                Console.WriteLine($"Model : {content}");
                return;
            }

            // The model wants to call one or more tools. Run each locally.
            Console.WriteLine();
            Console.WriteLine($"── AI ASKED TO CALL {toolCalls.Count} TOOL(S) ──");

            foreach (JsonNode? callNode in toolCalls)
            {
                var fn = callNode!["function"]!.AsObject();
                string name = fn["name"]!.GetValue<string>();
                JsonObject callArgs = fn["arguments"] as JsonObject ?? new JsonObject();

                // The model may attach an id to the call (e.g. "call_7kgjnzu8").
                // We echo it back so the result binds to THIS exact call — important
                // when the model makes several calls to the same tool in one turn.
                string? callId = callNode["id"]?.GetValue<string>();

                // Show the exact tool-call block the model produced.
                Console.WriteLine();
                Console.WriteLine($"  [TOOL CALL] the AI wants: {name}");
                Console.WriteLine($"  [ARGUMENTS] {callArgs.ToJsonString()}");
                Console.WriteLine("  [RAW tool_call block returned by the model]:");
                Console.WriteLine(Indent(Pretty(callNode!), "    "));

                // Actually run the C# function that backs this tool.
                string result;
                if (toolImplementations.TryGetValue(name, out var impl))
                    result = impl(callArgs);
                else
                    result = $"Error: unknown tool '{name}'.";

                Console.WriteLine($"  [EXECUTED ] {name} -> {result}");

                // Build the "tool" role message (the tool_result block from your notes).
                // tool_name identifies which tool ran; tool_call_id ties the result to
                // the specific call the model made (only added when the model gave one).
                var toolResultMessage = new JsonObject
                {
                    ["role"] = "tool",
                    ["tool_name"] = name,
                    ["content"] = result,
                };
                if (callId is not null)
                    toolResultMessage["tool_call_id"] = callId;

                Console.WriteLine("  [SENDING BACK TO AI] tool_result message:");
                Console.WriteLine(Indent(Pretty(toolResultMessage), "    "));

                messages.Add(toolResultMessage);
            }
            // Loop again so the model can read the tool result(s) and continue.
        }

        Console.WriteLine("Reached max turns without a final answer.");
    }

    // ── The tool SCHEMAS: how we DESCRIBE those tools to the model ──
    // This is the JSON Schema "input model" from your notes: name, description,
    // input schema (type + properties), and which fields are required.
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

    // ── Sends the conversation + tool schemas to Ollama, returns the assistant msg ──
    private static async Task<JsonObject> CallOllama(JsonArray conversation)
    {
        var requestBody = new JsonObject
        {
            ["model"] = Model,
            ["messages"] = conversation.DeepClone(),   // clone: we're about to reuse it
            ["tools"] = BuildToolSchemas(),
            ["stream"] = false,   // simplest to parse; set true for token streaming
        };

        // Log the exact JSON we POST to Ollama.
        Console.WriteLine();
        Console.WriteLine($"── REQUEST -> POST {OllamaUrl} ──");
        Console.WriteLine(Pretty(requestBody));

        using HttpResponseMessage resp = await http.PostAsJsonAsync(OllamaUrl, requestBody);
        resp.EnsureSuccessStatusCode();

        string json = await resp.Content.ReadAsStringAsync();

        // Log the raw JSON response exactly as Ollama returned it.
        Console.WriteLine();
        Console.WriteLine("── RAW RESPONSE <- Ollama ──");
        Console.WriteLine(Pretty(JsonNode.Parse(json)!));

        JsonObject root = JsonNode.Parse(json)!.AsObject();
        // DeepClone detaches the node from its parent document so we can safely
        // append it to our own `messages` array.
        return (JsonObject)root["message"]!.DeepClone();
    }

    // ── Helpers for readable logging ──
    private static string Pretty(JsonNode node) =>
        node.ToJsonString(new JsonSerializerOptions { WriteIndented = true });

    private static string Indent(string text, string pad) =>
        string.Join('\n', text.Split('\n').Select(line => pad + line));

    // ── Helper: read a numeric argument the model supplied (handles number or string) ──
    private static double GetNum(JsonObject args, string key)
    {
        JsonNode? node = args[key];
        if (node is null) return 0;
        // The model sometimes sends numbers as strings; handle both.
        return node.GetValueKind() == JsonValueKind.String
            ? double.Parse(node.GetValue<string>())
            : node.GetValue<double>();
    }
}
