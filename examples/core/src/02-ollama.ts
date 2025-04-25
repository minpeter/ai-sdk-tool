import { createToolMiddleware, gemmaToolMiddleware } from "@ai-sdk-tool/parser";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { extractReasoningMiddleware, streamText, wrapLanguageModel } from "ai";
import { z } from "zod";

const ollama = createOpenAICompatible({
  name: "ollama",
  apiKey: "ollama",
  baseURL: "http://localhost:11434/v1",
});

async function main() {
  const result = streamText({
    model: wrapLanguageModel({
      model: ollama("deepseek-r1:8b"),

      middleware: [
        // For deepseek-r1 distill 8b, it seems to tend to stick to ```json output, use custom
        createToolMiddleware({
          toolSystemPromptTemplate(tools) {
            return `You are a function calling AI model.
        You are provided with function signatures within <tools></tools> XML tags.
        You may call one or more functions to assist with the user query.
        Don't make assumptions about what values to plug into functions.
        Here are the available tools: <tools>${tools}</tools>
        Use the following pydantic model json schema for each tool call you will make: {'title': 'FunctionCall', 'type': 'object', 'properties': {'arguments': {'title': 'Arguments', 'type': 'object'}, 'name': {'title': 'Name', 'type': 'string'}}, 'required': ['arguments', 'name']}
        
        Emphasize again, if you decide to call a function, print it out right away with the <tool_call> tag.

        For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
        <tool_call>
        {'arguments': <args-dict>, 'name': <function-name>}
        </tool_call>`;
          },
          toolCallTag: "<tool_call>",
          toolCallEndTag: "</tool_call>",
          toolResponseTag: "<tool_response>",
          toolResponseEndTag: "</tool_response>",
        }),
        extractReasoningMiddleware({ tagName: "think" }),
      ],
    }),
    prompt: "What is the weather in New York and Los Angeles?",
    maxSteps: 4,
    tools: {
      get_location: {
        description: "Get the User's location.",
        parameters: z.object({}),
        execute: async () => {
          // Simulate a location API call
          return {
            city: "New York",
            country: "USA",
          };
        },
      },
      get_weather: {
        description:
          "Get the weather for a given city. " +
          "Example cities: 'New York', 'Los Angeles', 'Paris'.",
        parameters: z.object({ city: z.string() }),
        execute: async ({ city }) => {
          // Simulate a weather API call
          const temperature = Math.floor(Math.random() * 100);
          return {
            city,
            temperature,
            condition: "sunny",
          };
        },
      },
    },
  });

  for await (const part of result.fullStream) {
    if (part.type === "text") {
      process.stdout.write(part.text);
    } else if (part.type === "reasoning" && part.reasoningType === "text") {
      // Print reasoning text in a different color (e.g., yellow)
      process.stdout.write(`\x1b[33m${part.text}\x1b[0m`);
    } else if (part.type === "tool-result") {
      console.log({
        name: part.toolName,
        args: part.args,
        result: part.result,
      });
    }
  }

  console.log("\n\n<Complete>");
}

main().catch(console.error);
