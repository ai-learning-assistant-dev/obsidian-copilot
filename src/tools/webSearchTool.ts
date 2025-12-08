import { Tool } from "@langchain/core/tools";
import { getSettings } from "@/settings/model";
import axios from "axios";
import * as crypto from "crypto";
// import { Notice } from "obsidian";

// Helper function to generate Zhipu AI JWT token
function generateToken(apiKey: string): string {
  const [id, secret] = apiKey.split(".");
  if (!id || !secret) {
    throw new Error("Invalid Zhipu API Key format.");
  }

  const payload = {
    api_key: id,
    exp: Math.floor(Date.now() / 1000) + 3600, // Expires in 1 hour
    timestamp: Date.now(),
  };

  const header = {
    alg: "HS256",
    sign_type: "SIGN",
  };

  const base64UrlEncode = (data: object) => {
    return Buffer.from(JSON.stringify(data))
      .toString("base64")
      .replace(/\+/g, "-")
      .replace(/\//g, "_")
      .replace(/=/g, "");
  };

  const encodedHeader = base64UrlEncode(header);
  const encodedPayload = base64UrlEncode(payload);

  const signature = crypto
    .createHmac("sha256", secret)
    .update(`${encodedHeader}.${encodedPayload}`)
    .digest("base64")
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=/g, "");

  return `${encodedHeader}.${encodedPayload}.${signature}`;
}

interface SearchResult {
  title: string;
  content: string;
  link: string;
  media: string;
  icon: string;
  refer: string;
  publish_date: string;
}

export class WebSearchTool extends Tool {
  name = "web_search";
  description =
    "A web search tool. Use this when you need to find information about current events, specific product models, or any recent information that is not in your knowledge base.";

  public async _call(query: string): Promise<string> {
    const { zhipuApiKey } = getSettings();
    if (!zhipuApiKey) {
      return "Zhipu AI API key is not set. Please set it in the settings.";
    }

    try {
      const token = generateToken(zhipuApiKey);

      const response = await axios.post(
        "https://open.bigmodel.cn/api/paas/v4/web_search",
        {
          search_query: query,
          search_engine: "search_pro", // Use the professional version
          search_intent: false, // Agent has already determined the intent to search
        },
        {
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
          },
        }
      );

      const searchResults: SearchResult[] = response.data?.search_result;

      if (!searchResults || searchResults.length === 0) {
        return "No web search results found for that query.";
      }

      // Format the results into a clean string for the AI.
      const formattedResults = searchResults
        .map(
          (result) =>
            `Title: ${result.title}\nLink: ${result.link}\nSource: ${result.media}\nPublished Date: ${result.publish_date}\nSummary: ${result.content}`
        )
        .join("\n-----------------\n");

      return formattedResults;
    } catch (error) {
      console.error("Error during Zhipu web search:", error);
      // Check for specific API errors if the error object structure is known
      if (axios.isAxiosError(error) && error.response) {
        return `An error occurred during web search. Status: ${error.response.status}, Data: ${JSON.stringify(error.response.data)}`;
      }
      return "An unknown error occurred while trying to perform a web search.";
    }
  }
}
