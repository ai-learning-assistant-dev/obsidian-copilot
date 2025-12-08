import { getCurrentProject } from "@/aiParams";
import { getStandaloneQuestion } from "@/chainUtils";
import {
  ABORT_REASON,
  AI_SENDER,
  EMPTY_INDEX_ERROR_MESSAGE,
  LOADING_MESSAGES,
  MAX_CHARS_FOR_LOCAL_SEARCH_CONTEXT,
  ModelCapability,
} from "@/constants";
import {
  ImageBatchProcessor,
  ImageContent,
  ImageProcessingResult,
  MessageContent,
} from "@/imageProcessing/imageProcessor";
import { BrevilabsClient } from "@/LLMProviders/brevilabsClient";
import { logError, logInfo, logWarn } from "@/logger";
import { HybridRetriever } from "@/search/hybridRetriever";
import { getSettings, getSystemPrompt } from "@/settings/model";
import { ChatMessage } from "@/sharedState";
import { ToolManager } from "@/tools/toolManager";
import {
  err2String,
  extractChatHistory,
  // extractUniqueTitlesFromDocs,
  extractUniqueLinksFromDocs,
  extractYoutubeUrl,
  formatDateTime,
  getApiErrorMessage,
  getMessageRole,
  withSuppressedTokenWarnings,
} from "@/utils";
import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { Notice } from "obsidian";
import ChainManager from "./chainManager";
import { COPILOT_TOOL_NAMES, IntentAnalyzer } from "./intentAnalyzer";
import ProjectManager from "./projectManager";
import { WebSearchTool } from "@/tools/webSearchTool";
import { FreeBingSearchTool } from "@/tools/freeBingSearchTool";

class ThinkBlockStreamer {
  private hasOpenThinkBlock = false;
  private fullResponse = "";
  private ttsIndex = 0; // 添加TTS索引计数器

  constructor(private updateCurrentAiMessage: (message: string) => void) {}

  private handleClaude37Chunk(content: any[]) {
    let textContent = "";
    for (const item of content) {
      switch (item.type) {
        case "text":
          textContent += item.text;
          break;
        case "thinking":
          if (!this.hasOpenThinkBlock) {
            this.fullResponse += "\n<think>";
            this.hasOpenThinkBlock = true;
          }
          this.fullResponse += item.thinking;
          this.updateCurrentAiMessage(this.fullResponse);
          return true; // Indicate we handled a thinking chunk
      }
    }
    if (textContent) {
      this.fullResponse += textContent;
    }
    return false; // No thinking chunk handled
  }

  private async handleDeepseekChunk(chunk: any) {
    // Handle standard string content
    if (typeof chunk.content === "string") {
      this.fullResponse += chunk.content;

      // 发送文本到aloud插件进行TTS播放
      await this.sendTextToTTS(chunk.content);
    }

    // Handle deepseek reasoning/thinking content
    if (chunk.additional_kwargs?.reasoning_content) {
      if (!this.hasOpenThinkBlock) {
        this.fullResponse += "\n<think>";
        this.hasOpenThinkBlock = true;
      }
      this.fullResponse += chunk.additional_kwargs.reasoning_content;
      // 根据设置判断是否播放思考内容
      await this.sendThinkingToTTS(chunk.additional_kwargs.reasoning_content);
      return true; // Indicate we handled a thinking chunk
    }
    return false; // No thinking chunk handled
  }

  // 添加发送思考内容到TTS的方法
  private async sendThinkingToTTS(thinkingContent: string) {
    // 获取设置判断是否播放思考内容
    const settings = getSettings();
    if (settings.promptEnhancements?.autoSpeech?.speakThinkContent) {
      await this.sendTextToTTS(thinkingContent);
    }
  }

  // 添加发送文本到TTS插件的方法
  private async sendTextToTTS(text: string) {
    if (!text) return;

    try {
      // 获取设置判断是否启用自动语音播放
      const settings = getSettings();
      if (!settings.promptEnhancements?.autoSpeech?.enabled) {
        return;
      }

      // 获取aloud插件实例
      const ttsPlugin = (window as any).app?.plugins?.getPlugin("aloud-tts-ai-learning-assistant");
      if (ttsPlugin?.ttsService?.playStreamText) {
        // 直接发送文本，不分块
        await ttsPlugin.ttsService.playStreamText({
          index: this.ttsIndex++,
          length: text.length,
          text: text,
        });
      }
    } catch (error) {
      console.warn("Failed to send text to TTS plugin:", error);
    }
  }

  // 添加结束TTS流的方法
  private async finishTTSStream() {
    try {
      // 获取设置判断是否启用自动语音播放
      const settings = getSettings();
      if (!settings.promptEnhancements?.autoSpeech?.enabled) {
        return;
      }

      const ttsPlugin = (window as any).app?.plugins?.getPlugin("aloud-tts-ai-learning-assistant");
      if (ttsPlugin?.ttsService?.playStreamText) {
        // 发送结束信号
        await ttsPlugin.ttsService.playStreamText({
          index: this.ttsIndex++,
          length: 0,
          text: "",
        });
      }
    } catch (error) {
      console.warn("Failed to finish TTS stream:", error);
    }
  }
  async processChunk(chunk: any) {
    let handledThinking = false;

    // Handle Claude 3.7 array-based content
    if (Array.isArray(chunk.content)) {
      handledThinking = this.handleClaude37Chunk(chunk.content);
    } else {
      // Handle deepseek format
      handledThinking = await this.handleDeepseekChunk(chunk);
    }

    // Close think block if we have one open and didn't handle thinking content
    if (this.hasOpenThinkBlock && !handledThinking) {
      this.fullResponse += "</think>";
      this.hasOpenThinkBlock = false;
    }

    this.updateCurrentAiMessage(this.fullResponse);
  }

  async close() {
    // Make sure to close any open think block at the end
    if (this.hasOpenThinkBlock) {
      this.fullResponse += "</think>";
      this.updateCurrentAiMessage(this.fullResponse);
    }

    // 结束TTS流
    await this.finishTTSStream();

    return this.fullResponse;
  }
}

export interface ChainRunner {
  run(
    userMessage: ChatMessage,
    abortController: AbortController,
    updateCurrentAiMessage: (message: string) => void,
    addMessage: (message: ChatMessage) => void,
    options: {
      debug?: boolean;
      ignoreSystemMessage?: boolean;
      updateLoading?: (loading: boolean) => void;
    }
  ): Promise<string>;
}

abstract class BaseChainRunner implements ChainRunner {
  protected chainManager: ChainManager;

  constructor(chainManager: ChainManager) {
    this.chainManager = chainManager;
  }

  abstract run(
    userMessage: ChatMessage,
    abortController: AbortController,
    updateCurrentAiMessage: (message: string) => void,
    addMessage: (message: ChatMessage) => void,
    options: {
      debug?: boolean;
      ignoreSystemMessage?: boolean;
      updateLoading?: (loading: boolean) => void;
    }
  ): Promise<string>;

  protected async handleResponse(
    fullAIResponse: string,
    userMessage: ChatMessage,
    abortController: AbortController,
    addMessage: (message: ChatMessage) => void,
    updateCurrentAiMessage: (message: string) => void,
    sources?: { title: string; score: number }[]
  ) {
    // Save to memory and add message if we have a response
    // Skip only if it's a NEW_CHAT abort (clearing everything)
    if (
      fullAIResponse &&
      !(abortController.signal.aborted && abortController.signal.reason === ABORT_REASON.NEW_CHAT)
    ) {
      await this.chainManager.memoryManager
        .getMemory()
        .saveContext({ input: userMessage.message }, { output: fullAIResponse });

      addMessage({
        message: fullAIResponse,
        sender: AI_SENDER,
        isVisible: true,
        timestamp: formatDateTime(new Date()),
        sources: sources,
      });

      // Clear the streaming message since it's now in chat history
      updateCurrentAiMessage("");
    } else if (abortController.signal.reason === ABORT_REASON.NEW_CHAT) {
      // Also clear if it's a new chat
      updateCurrentAiMessage("");
    }
    logInfo(
      "==== Chat Memory ====\n",
      (this.chainManager.memoryManager.getMemory().chatHistory as any).messages.map(
        (m: any) => m.content
      )
    );
    logInfo("==== Final AI Response ====\n", fullAIResponse);
    return fullAIResponse;
  }

  protected async handleError(
    error: any,
    addMessage?: (message: ChatMessage) => void,
    updateCurrentAiMessage?: (message: string) => void
  ) {
    const msg = err2String(error);
    logError("Error during LLM invocation:", msg);
    const errorData = error?.response?.data?.error || msg;
    const errorCode = errorData?.code || msg;
    let errorMessage = "";

    // Check for specific error messages
    if (error?.message?.includes("Invalid license key")) {
      errorMessage = "Invalid Copilot Plus license key. Please check your license key in settings.";
    } else if (errorCode === "model_not_found") {
      errorMessage =
        "You do not have access to this model or the model does not exist, please check with your API provider.";
    } else {
      errorMessage = `${errorCode}`;
    }

    logError(errorData);

    if (addMessage && updateCurrentAiMessage) {
      updateCurrentAiMessage("");

      // remove langchain troubleshooting URL from error message
      const ignoreEndIndex = errorMessage.search("Troubleshooting URL");
      errorMessage = ignoreEndIndex !== -1 ? errorMessage.slice(0, ignoreEndIndex) : errorMessage;

      // add more user guide for invalid API key
      if (msg.search(/401|invalid|not valid/gi) !== -1) {
        errorMessage =
          "Something went wrong. Please check if you have set your API key." +
          "\nPath: Settings > copilot plugin > Basic Tab > Set Keys." +
          "\nOr check model config" +
          "\nError Details: " +
          errorMessage;
      }

      addMessage({
        message: errorMessage,
        isErrorMessage: true,
        sender: AI_SENDER,
        isVisible: true,
        timestamp: formatDateTime(new Date()),
      });
    } else {
      // Fallback to Notice if message handlers aren't provided
      new Notice(errorMessage);
      logError(errorData);
    }
  }
}

class LLMChainRunner extends BaseChainRunner {
  /**
   * 快速预过滤：对于明显的场景直接返回判断结果，无需调用 LLM
   * 这可以减少 API 调用次数和成本
   */
  private quickSearchCheck(message: string): {
    confident: boolean;
    needsSearch: boolean;
  } {
    const query = message.toLowerCase();

    // 100% 确定不需要搜索的场景
    const definitelyNoSearch = [
      /^(你好|嗨|hi|hello|嘿)/, // 问候
      /^(谢谢|感谢|thanks|thank you)/, // 感谢
      /^(再见|拜拜|bye|goodbye)/, // 告别
      /(帮我|请|能否|可以)(写|创建|生成|制作)(?!.*搜索)/, // 内容创作
      /写.*代码|创建.*函数|生成.*文件|编写.*程序/, // 代码相关
      /(翻译|总结|解释|分析)(?!.*最新|.*今天)/, // 文本处理
    ];

    for (const pattern of definitelyNoSearch) {
      if (pattern.test(query)) {
        logInfo("Quick filter: Definitely NO search needed");
        return { confident: true, needsSearch: false };
      }
    }

    // 100% 确定需要搜索的场景
    const definitelyNeedSearch = [
      /@websearch|@web\b/, // 明确的搜索命令
      /^(搜索|查找|查询|search|find)(?!.*文件|.*笔记)/, // 明确搜索意图
      /今天.*天气|现在.*温度|当前.*气温/, // 实时天气
      /现在.*价格|最新.*股价|当前.*汇率/, // 实时价格
      /(今天|最新|最近).*新闻/, // 新闻资讯
    ];

    for (const pattern of definitelyNeedSearch) {
      if (pattern.test(query)) {
        logInfo("Quick filter: Definitely needs search");
        return { confident: true, needsSearch: true };
      }
    }

    // 不确定，需要 LLM 判断
    return { confident: false, needsSearch: false };
  }

  /**
   * 使用用户配置的 LLM 来判断是否需要网络搜索
   * 这样可以利用用户已有的 API key，不需要额外配置
   */
  private async analyzeSearchIntentWithLLM(message: string): Promise<{
    needsSearch: boolean;
    searchQuery: string;
  }> {
    try {
      const systemPrompt = `你是一个智能助手，负责判断用户的消息是否需要进行网络搜索。

需要网络搜索的情况：
1. 需要实时信息：天气、新闻、股价、汇率等
2. 需要最新信息：最新版本、最新产品、最新发布等
3. 需要验证的事实：具体数据、统计信息、官方信息等
4. 用户明确要求搜索

不需要网络搜索的情况：
1. 问候、感谢、道别等社交对话
2. 内容创作：写代码、写文章、生成内容等
3. 文本处理：翻译、总结、解释、分析等
4. 一般知识问答：你的知识库可以直接回答的问题
5. 在本地笔记中搜索

请分析用户消息，并以JSON格式返回：
{
  "needsSearch": true/false,
  "searchQuery": "如果需要搜索，提供优化后的搜索关键词；否则为空字符串",
  "reasoning": "简短说明判断理由（1-2句话）"
}

只返回JSON，不要有其他内容。`;

      const userPrompt = `用户消息：${message}`;

      logInfo("==== Analyzing search intent with user's LLM ====");

      // 使用当前配置的 LLM 进行判断
      const response = await this.chainManager.chatModelManager.getChatModel().invoke([
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt },
      ]);

      const content = (response.content as string).trim();
      logInfo("LLM search intent response:", content);

      // 提取 JSON（可能被包裹在 markdown 代码块中）
      let jsonStr = content;
      const jsonMatch = content.match(/```(?:json)?\s*(\{[\s\S]*?\})\s*```/);
      if (jsonMatch) {
        jsonStr = jsonMatch[1];
      } else {
        const directMatch = content.match(/\{[\s\S]*\}/);
        if (directMatch) {
          jsonStr = directMatch[0];
        }
      }

      const result = JSON.parse(jsonStr);
      logInfo("Parsed search intent:", result);

      return {
        needsSearch: result.needsSearch === true,
        searchQuery: result.searchQuery || message,
      };
    } catch (error) {
      logWarn("Failed to analyze search intent with LLM:", error);
      // 降级到简单的本地规则
      return this.fallbackSearchIntent(message);
    }
  }

  /**
   * 降级方案：当 LLM 分析失败时使用的简单本地规则
   */
  private fallbackSearchIntent(message: string): {
    needsSearch: boolean;
    searchQuery: string;
  } {
    const query = message.toLowerCase();

    // 简单的关键词匹配
    const searchKeywords = [
      "今天",
      "最新",
      "现在",
      "当前",
      "天气",
      "新闻",
      "价格",
      "股价",
      "搜索",
      "查找",
      "查询",
    ];

    const hasSearchKeyword = searchKeywords.some((keyword) => query.includes(keyword));

    // 排除创作类关键词
    const createKeywords = ["写", "创建", "生成", "翻译", "总结"];
    const isCreateRequest = createKeywords.some((keyword) => query.includes(keyword));

    const needsSearch = hasSearchKeyword && !isCreateRequest;

    logInfo(`Fallback search intent: ${needsSearch}`);
    return {
      needsSearch,
      searchQuery: message,
    };
  }

  async run(
    userMessage: ChatMessage,
    abortController: AbortController,
    updateCurrentAiMessage: (message: string) => void,
    addMessage: (message: ChatMessage) => void,
    options: {
      debug?: boolean;
      ignoreSystemMessage?: boolean;
      updateLoading?: (loading: boolean) => void;
    }
  ): Promise<string> {
    const streamer = new ThinkBlockStreamer(updateCurrentAiMessage);
    let searchSources: { title: string; score: number }[] = [];

    try {
      let enhancedMessage = userMessage.message;

      // 使用 LLM 智能判断是否需要网络搜索
      if (getSettings().useWebSearch) {
        try {
          const messageForAnalysis = userMessage.originalMessage || userMessage.message;

          // 第一步：快速过滤明显场景（节省 API 调用）
          const quickCheck = this.quickSearchCheck(messageForAnalysis);

          let needsWebSearch = false;
          let searchQuery = messageForAnalysis;

          if (quickCheck.confident) {
            // 明显场景，直接使用快速判断结果
            needsWebSearch = quickCheck.needsSearch;
            logInfo(`Quick decision: needsSearch = ${needsWebSearch}`);
          } else {
            // 不确定的场景，使用 LLM 判断
            logInfo("Uncertain case, analyzing with LLM...");
            const intentResult = await this.analyzeSearchIntentWithLLM(messageForAnalysis);
            needsWebSearch = intentResult.needsSearch;
            searchQuery = intentResult.searchQuery;
          }

          // 如果需要搜索，执行网络搜索
          if (needsWebSearch) {
            const searchEngineType = getSettings().searchEngineType || "free";
            const searchEngineName = searchEngineType === "zhipu" ? "智谱AI" : "免费搜索引擎";
            new Notice(`正在使用${searchEngineName}进行网络搜索...`);
            logInfo(
              `==== Executing web search with ${searchEngineName} (${searchEngineType}), query: ${searchQuery} ====`
            );

            // 根据设置选择搜索工具
            const searchTool =
              searchEngineType === "zhipu" ? new WebSearchTool() : new FreeBingSearchTool();
            const searchResults = await searchTool._call(searchQuery);

            if (searchResults) {
              logInfo("==== Web Search Results ====\n", searchResults);

              // 解析搜索结果，提取来源信息
              const sourceMatches = searchResults.matchAll(/Title: ([^\n]+)\nLink: ([^\n]+)/g);
              searchSources = Array.from(sourceMatches).map((match, index) => ({
                title: match[1],
                score: 1.0 - index * 0.05, // 给来源一个递减的分数
              }));

              enhancedMessage = `
基于以下网络搜索结果，请按照这个格式回答：

1. 首先列出搜索来源（用 "## 🔍 搜索来源" 作为标题）
2. 然后提供完整的回答

网络搜索结果：
---
${searchResults}
---

用户的原始问题：
"${userMessage.message}"

请在回答开头列出搜索来源，格式如下：
## 🔍 搜索来源
- [来源标题1](链接1)
- [来源标题2](链接2)
...

然后再提供详细回答。
`;
            }
          }
        } catch (error) {
          console.error("Intent analysis or web search failed:", error);
          logWarn("Falling back to direct LLM without web search");
          // 失败时回退到原始消息，不影响正常聊天流程
        }
      }

      // 使用增强后的消息（如果有搜索结果）或原始消息
      const finalMessage = enhancedMessage;

      // Get chat history from memory
      const memory = this.chainManager.memoryManager.getMemory();
      const memoryVariables = await memory.loadMemoryVariables({});
      const chatHistory = extractChatHistory(memoryVariables);

      // Create messages array starting with system message
      const messages: any[] = [];

      // Add system message if available
      const systemPrompt = getSystemPrompt();
      const chatModel = this.chainManager.chatModelManager.getChatModel();

      if (systemPrompt) {
        messages.push({
          role: getMessageRole(chatModel),
          content: systemPrompt,
        });
      }

      // Add chat history
      for (const entry of chatHistory) {
        messages.push({ role: entry.role, content: entry.content });
      }

      // Add current user message (使用增强后的消息)
      messages.push({
        role: "user",
        content: finalMessage,
      });

      logInfo("==== Final Request to AI ====\n", messages);

      // Stream with abort signal, suppress warnings for the entire streaming lifecycle
      await withSuppressedTokenWarnings(async () => {
        const chatStream = await this.chainManager.chatModelManager
          .getChatModel()
          .stream(messages, { signal: abortController.signal });

        for await (const chunk of chatStream) {
          if (abortController.signal.aborted) {
            logInfo("Stream iteration aborted", { reason: abortController.signal.reason });
            break;
          }
          streamer.processChunk(chunk);
        }
      });
    } catch (error: any) {
      // Check if the error is due to abort signal
      if (error.name === "AbortError" || abortController.signal.aborted) {
        logInfo("Stream aborted by user", { reason: abortController.signal.reason });
        // Don't show error message for user-initiated aborts
      } else {
        await this.handleError(error, addMessage, updateCurrentAiMessage);
      }
    }

    // Always return the response, even if partial
    const response = await streamer.close();

    // Only skip saving if it's a new chat (clearing everything)
    if (abortController.signal.aborted && abortController.signal.reason === ABORT_REASON.NEW_CHAT) {
      updateCurrentAiMessage("");
      return "";
    }

    return this.handleResponse(
      response,
      userMessage,
      abortController,
      addMessage,
      updateCurrentAiMessage,
      searchSources.length > 0 ? searchSources : undefined
    );
  }
}

class VaultQAChainRunner extends BaseChainRunner {
  async run(
    userMessage: ChatMessage,
    abortController: AbortController,
    updateCurrentAiMessage: (message: string) => void,
    addMessage: (message: ChatMessage) => void,
    options: {
      debug?: boolean;
      ignoreSystemMessage?: boolean;
      updateLoading?: (loading: boolean) => void;
    }
  ): Promise<string> {
    const streamer = new ThinkBlockStreamer(updateCurrentAiMessage);

    try {
      // Add check for empty index
      const indexEmpty = await this.chainManager.vectorStoreManager.isIndexEmpty();
      if (indexEmpty) {
        return this.handleResponse(
          EMPTY_INDEX_ERROR_MESSAGE,
          userMessage,
          abortController,
          addMessage,
          updateCurrentAiMessage
        );
      }

      // Get chat history from memory
      const memory = this.chainManager.memoryManager.getMemory();
      const memoryVariables = await memory.loadMemoryVariables({});
      const chatHistory = extractChatHistory(memoryVariables);

      // Generate standalone question from user message + chat history
      // This is similar to what the conversational retrieval chain does
      let standaloneQuestion = userMessage.message;
      if (chatHistory.length > 0) {
        // For simplicity, we'll use the original question directly
        // The original chain would rephrase it, but this approach should work for most cases
        standaloneQuestion = userMessage.message;
      }

      // Create retriever (similar to how it's done in chainManager)
      const retriever = new HybridRetriever({
        minSimilarityScore: 0.01,
        maxK: getSettings().maxSourceChunks,
        salientTerms: [],
      });

      // Retrieve relevant documents
      const retrievedDocs = await retriever.getRelevantDocuments(standaloneQuestion);

      // Store retrieved documents for sources
      this.chainManager.storeRetrieverDocuments(retrievedDocs);

      // Format documents as context
      const context = retrievedDocs.map((doc: any) => doc.pageContent).join("\n\n");

      // Create messages array
      const messages: any[] = [];

      // Add system message with QA instruction
      const systemPrompt = getSystemPrompt();
      const qaInstructions =
        "\n\nAnswer the question with as detailed as possible based only on the following context:\n" +
        context;
      const fullSystemMessage = systemPrompt + qaInstructions;

      const chatModel = this.chainManager.chatModelManager.getChatModel();
      if (fullSystemMessage) {
        messages.push({
          role: getMessageRole(chatModel),
          content: fullSystemMessage,
        });
      }

      // Add chat history
      for (const entry of chatHistory) {
        messages.push({ role: entry.role, content: entry.content });
      }

      // Add current user question
      messages.push({
        role: "user",
        content: userMessage.message,
      });

      logInfo("==== Final Request to AI ====\n", messages);

      // Stream with abort signal, suppress warnings for the entire streaming lifecycle
      await withSuppressedTokenWarnings(async () => {
        const chatStream = await this.chainManager.chatModelManager
          .getChatModel()
          .stream(messages, { signal: abortController.signal });

        for await (const chunk of chatStream) {
          if (abortController.signal.aborted) {
            logInfo("VaultQA stream iteration aborted", { reason: abortController.signal.reason });
            break;
          }
          // 修改为 await 调用
          await streamer.processChunk(chunk);
        }
      });
    } catch (error: any) {
      // Check if the error is due to abort signal
      if (error.name === "AbortError" || abortController.signal.aborted) {
        logInfo("VaultQA stream aborted by user", { reason: abortController.signal.reason });
        // Don't show error message for user-initiated aborts
      } else {
        await this.handleError(error, addMessage, updateCurrentAiMessage);
      }
    }

    // Always get the response, even if partial
    let fullAIResponse = await streamer.close();

    // Only skip saving if it's a new chat (clearing everything)
    if (abortController.signal.aborted && abortController.signal.reason === ABORT_REASON.NEW_CHAT) {
      updateCurrentAiMessage("");
      return "";
    }

    // Add sources to the response using only current-turn retrieved documents
    fullAIResponse = this.addSourcestoResponse(
      fullAIResponse,
      this.chainManager.getRetrievedDocuments()
    );

    return this.handleResponse(
      fullAIResponse,
      userMessage,
      abortController,
      addMessage,
      updateCurrentAiMessage
    );
  }

  private addSourcestoResponse(response: string, docsForThisTurn: any[]): string {
    // 移除响应中任何已有的 Sources 段，以避免重复
    const sourcesSectionRegex = /(?:\r?\n){0,2}#### Sources:\r?\n\r?\n(?:- .*\r?\n)+/g;
    let cleaned = response.replace(sourcesSectionRegex, "");

    const docLinks = extractUniqueLinksFromDocs(docsForThisTurn);
    if (docLinks.length > 0) {
      const links = docLinks.map((link) => `- ${link}`).join("\n");
      cleaned += "\n\n#### Sources:\n\n" + links;
    }
    return cleaned;
  }
}

class CopilotPlusChainRunner extends BaseChainRunner {
  private isYoutubeOnlyMessage(message: string): boolean {
    const trimmedMessage = message.trim();
    const hasYoutubeCommand = trimmedMessage.includes("@youtube");
    const youtubeUrl = extractYoutubeUrl(trimmedMessage);

    // Check if message only contains @youtube command and a valid URL
    const words = trimmedMessage
      .split(/\s+/)
      .filter((word) => word !== "@youtube" && word.length > 0);

    return hasYoutubeCommand && youtubeUrl !== null && words.length === 1;
  }

  private async processImageUrls(urls: string[]): Promise<ImageProcessingResult> {
    const failedImages: string[] = [];
    const processedImages = await ImageBatchProcessor.processUrlBatch(
      urls,
      failedImages,
      this.chainManager.app.vault
    );
    ImageBatchProcessor.showFailedImagesNotice(failedImages);
    return processedImages;
  }

  private async processChatInputImages(content: MessageContent[]): Promise<ImageProcessingResult> {
    const failedImages: string[] = [];
    const processedImages = await ImageBatchProcessor.processChatImageBatch(
      content,
      failedImages,
      this.chainManager.app.vault
    );
    ImageBatchProcessor.showFailedImagesNotice(failedImages);
    return processedImages;
  }

  private async extractEmbeddedImages(content: string): Promise<string[]> {
    const imageRegex = /!\[\[(.*?\.(png|jpg|jpeg|gif|webp|bmp|svg))\]\]/g;
    const matches = [...content.matchAll(imageRegex)];
    const images = matches.map((match) => match[1]);
    return images;
  }

  private async buildMessageContent(
    textContent: string,
    userMessage: ChatMessage
  ): Promise<MessageContent[]> {
    const failureMessages: string[] = [];
    const successfulImages: ImageContent[] = [];
    const settings = getSettings();

    // Collect all image sources
    const imageSources: { urls: string[]; type: string }[] = [];

    // Safely check and add context URLs
    const contextUrls = userMessage.context?.urls;
    if (contextUrls && contextUrls.length > 0) {
      imageSources.push({ urls: contextUrls, type: "context" });
    }

    // Process embedded images only if setting is enabled
    if (settings.passMarkdownImages) {
      const embeddedImages = await this.extractEmbeddedImages(textContent);
      if (embeddedImages.length > 0) {
        imageSources.push({ urls: embeddedImages, type: "embedded" });
      }
    }

    // Process all image sources
    for (const source of imageSources) {
      const result = await this.processImageUrls(source.urls);
      successfulImages.push(...result.successfulImages);
      failureMessages.push(...result.failureDescriptions);
    }

    // Process existing chat content images if present
    const existingContent = userMessage.content;
    if (existingContent && existingContent.length > 0) {
      const result = await this.processChatInputImages(existingContent);
      successfulImages.push(...result.successfulImages);
      failureMessages.push(...result.failureDescriptions);
    }

    // Let the LLM know about the image processing failures
    let finalText = textContent;
    if (failureMessages.length > 0) {
      finalText = `${textContent}\n\nNote: \n${failureMessages.join("\n")}\n`;
    }

    const messageContent: MessageContent[] = [
      {
        type: "text",
        text: finalText,
      },
    ];

    // Add successful images after the text content
    if (successfulImages.length > 0) {
      messageContent.push(...successfulImages);
    }

    return messageContent;
  }

  private hasCapability(model: BaseChatModel, capability: ModelCapability): boolean {
    const modelName = (model as any).modelName || (model as any).model || "";
    const customModel = this.chainManager.chatModelManager.findModelByName(modelName);
    return customModel?.capabilities?.includes(capability) ?? false;
  }

  private isMultimodalModel(model: BaseChatModel): boolean {
    return this.hasCapability(model, ModelCapability.VISION);
  }

  private async streamMultimodalResponse(
    textContent: string,
    userMessage: ChatMessage,
    abortController: AbortController,
    updateCurrentAiMessage: (message: string) => void
  ): Promise<string> {
    // Get chat history
    const memory = this.chainManager.memoryManager.getMemory();
    const memoryVariables = await memory.loadMemoryVariables({});
    const chatHistory = extractChatHistory(memoryVariables);

    // Create messages array starting with system message
    const messages: any[] = [];

    // Add system message if available
    let fullSystemMessage = await this.getSystemPrompt();

    // Add chat history context to system message if exists
    if (chatHistory.length > 0) {
      fullSystemMessage +=
        "\n\nThe following is the relevant conversation history. Use this context to maintain consistency in your responses:";
    }

    // Get chat model for role determination for O-series models
    const chatModel = this.chainManager.chatModelManager.getChatModel();

    // Add the combined system message with appropriate role
    if (fullSystemMessage) {
      messages.push({
        role: getMessageRole(chatModel),
        content: `${fullSystemMessage}\nIMPORTANT: Maintain consistency with previous responses in the conversation. If you've provided information about a person or topic before, use that same information in follow-up questions.`,
      });
    }

    // Add chat history
    for (const entry of chatHistory) {
      messages.push({ role: entry.role, content: entry.content });
    }

    // Get the current chat model
    const chatModelCurrent = this.chainManager.chatModelManager.getChatModel();
    const isMultimodalCurrent = this.isMultimodalModel(chatModelCurrent);

    // Build message content with text and images for multimodal models, or just text for text-only models
    const content = isMultimodalCurrent
      ? await this.buildMessageContent(textContent, userMessage)
      : textContent;

    // Add current user message
    messages.push({
      role: "user",
      content,
    });

    const enhancedUserMessage = content instanceof Array ? (content[0] as any).text : content;
    logInfo("Enhanced user message: ", enhancedUserMessage);
    logInfo("==== Final Request to AI ====\n", messages);
    const streamer = new ThinkBlockStreamer(updateCurrentAiMessage);

    // Wrap the entire streaming lifecycle with warning suppression
    await withSuppressedTokenWarnings(async () => {
      const chatStream = await this.chainManager.chatModelManager
        .getChatModel()
        .stream(messages, { signal: abortController.signal });

      for await (const chunk of chatStream) {
        if (abortController.signal.aborted) {
          logInfo("CopilotPlus multimodal stream iteration aborted", {
            reason: abortController.signal.reason,
          });
          break;
        }
        streamer.processChunk(chunk);
      }
    });

    return streamer.close();
  }

  async run(
    userMessage: ChatMessage,
    abortController: AbortController,
    updateCurrentAiMessage: (message: string) => void,
    addMessage: (message: ChatMessage) => void,
    options: {
      debug?: boolean;
      ignoreSystemMessage?: boolean;
      updateLoading?: (loading: boolean) => void;
      updateLoadingMessage?: (message: string) => void;
    }
  ): Promise<string> {
    const { updateLoadingMessage } = options;
    let fullAIResponse = "";
    let sources: { title: string; score: number }[] = [];
    let currentPartialResponse = "";

    // Wrapper to track partial response
    const trackAndUpdateAiMessage = (message: string) => {
      currentPartialResponse = message;
      updateCurrentAiMessage(message);
    };

    try {
      // Check if this is a YouTube-only message
      if (this.isYoutubeOnlyMessage(userMessage.message)) {
        const url = extractYoutubeUrl(userMessage.message);
        const failMessage =
          "Transcript not available. Only videos with the auto transcript option turned on are supported at the moment.";
        if (url) {
          try {
            const response = await BrevilabsClient.getInstance().youtube4llm(url);
            if (response.response.transcript) {
              return this.handleResponse(
                response.response.transcript,
                userMessage,
                abortController,
                addMessage,
                updateCurrentAiMessage
              );
            }
            return this.handleResponse(
              failMessage,
              userMessage,
              abortController,
              addMessage,
              updateCurrentAiMessage
            );
          } catch (error) {
            logError("Error processing YouTube video:", error);
            return this.handleResponse(
              failMessage,
              userMessage,
              abortController,
              addMessage,
              updateCurrentAiMessage
            );
          }
        }
      }

      logInfo("==== Step 1: Analyzing intent ====");
      let toolCalls;
      // Use the original message for intent analysis
      const messageForAnalysis = userMessage.originalMessage || userMessage.message;
      try {
        toolCalls = await IntentAnalyzer.analyzeIntent(messageForAnalysis);
      } catch (error: any) {
        return this.handleResponse(
          getApiErrorMessage(error),
          userMessage,
          abortController,
          addMessage,
          updateCurrentAiMessage
        );
      }

      // Use the same removeAtCommands logic as IntentAnalyzer
      const cleanedUserMessage = userMessage.message
        .split(" ")
        .filter((word) => !COPILOT_TOOL_NAMES.includes(word.toLowerCase()))
        .join(" ")
        .trim();

      const toolOutputs = await this.executeToolCalls(toolCalls, updateLoadingMessage);
      const localSearchResult = toolOutputs.find(
        (output) => output.tool === "localSearch" && output.output && output.output.length > 0
      );

      // Format chat history from memory
      const memory = this.chainManager.memoryManager.getMemory();
      const memoryVariables = await memory.loadMemoryVariables({});
      const chatHistory = extractChatHistory(memoryVariables);

      if (localSearchResult) {
        logInfo("==== Step 2: Processing local search results ====");
        const documents = JSON.parse(localSearchResult.output);

        logInfo("==== Step 3: Condensing Question ====");
        const standaloneQuestion = await getStandaloneQuestion(cleanedUserMessage, chatHistory);
        logInfo("Condensed standalone question: ", standaloneQuestion);

        logInfo("==== Step 4: Preparing context ====");
        const timeExpression = this.getTimeExpression(toolCalls);
        const context = this.prepareLocalSearchResult(documents, timeExpression);

        const currentTimeOutputs = toolOutputs.filter((output) => output.tool === "getCurrentTime");
        const enhancedQuestion = this.prepareEnhancedUserMessage(
          standaloneQuestion,
          currentTimeOutputs
        );

        logInfo(context);
        logInfo("==== Step 5: Invoking QA Chain ====");
        const qaPrompt = await this.chainManager.promptManager.getQAPrompt({
          question: enhancedQuestion,
          context,
          systemMessage: "", // System prompt is added separately in streamMultimodalResponse
        });

        fullAIResponse = await this.streamMultimodalResponse(
          qaPrompt,
          userMessage,
          abortController,
          trackAndUpdateAiMessage
        );

        // Append sources to the response
        sources = this.getSources(documents);
      } else {
        // Enhance with tool outputs.
        const enhancedUserMessage = this.prepareEnhancedUserMessage(
          cleanedUserMessage,
          toolOutputs
        );
        // If no results, default to LLM Chain
        logInfo("No local search results. Using standard LLM Chain.");

        fullAIResponse = await this.streamMultimodalResponse(
          enhancedUserMessage,
          userMessage,
          abortController,
          trackAndUpdateAiMessage
        );
      }
    } catch (error: any) {
      // Reset loading message to default
      updateLoadingMessage?.(LOADING_MESSAGES.DEFAULT);

      // Check if the error is due to abort signal
      if (error.name === "AbortError" || abortController.signal.aborted) {
        logInfo("CopilotPlus stream aborted by user", { reason: abortController.signal.reason });
        // Don't show error message for user-initiated aborts
      } else {
        await this.handleError(error, addMessage, updateCurrentAiMessage);
      }
    }

    // Only skip saving if it's a new chat (clearing everything)
    if (abortController.signal.aborted && abortController.signal.reason === ABORT_REASON.NEW_CHAT) {
      updateCurrentAiMessage("");
      return "";
    }

    // If aborted but not a new chat, use the partial response
    if (abortController.signal.aborted && currentPartialResponse) {
      fullAIResponse = currentPartialResponse;
    }

    return this.handleResponse(
      fullAIResponse,
      userMessage,
      abortController,
      addMessage,
      updateCurrentAiMessage,
      sources
    );
  }

  private getSources(documents: any): { title: string; score: number }[] {
    if (!documents || !Array.isArray(documents)) {
      logWarn("No valid documents provided to getSources");
      return [];
    }
    return this.sortUniqueDocsByScore(documents);
  }

  private sortUniqueDocsByScore(documents: any[]): any[] {
    const uniqueDocs = new Map<string, any>();

    // Iterate through all documents
    for (const doc of documents) {
      if (!doc.title || (!doc?.score && !doc?.rerank_score)) {
        logWarn("Invalid document structure:", doc);
        continue;
      }

      const currentDoc = uniqueDocs.get(doc.title);
      const isReranked = doc && "rerank_score" in doc;
      const docScore = isReranked ? doc.rerank_score : doc.score;

      // If the title doesn't exist in the map, or if the new doc has a higher score, update the map
      if (!currentDoc || docScore > (currentDoc.score ?? 0)) {
        uniqueDocs.set(doc.title, {
          title: doc.title,
          score: docScore,
          isReranked: isReranked,
        });
      }
    }

    // Convert the map values back to an array and sort by score in descending order
    return Array.from(uniqueDocs.values()).sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
  }

  private async executeToolCalls(
    toolCalls: any[],
    updateLoadingMessage?: (message: string) => void
  ) {
    const toolOutputs = [];
    for (const toolCall of toolCalls) {
      logInfo(`==== Step 2: Calling tool: ${toolCall.tool.name} ====`);
      if (toolCall.tool.name === "localSearch") {
        updateLoadingMessage?.(LOADING_MESSAGES.READING_FILES);
      } else if (toolCall.tool.name === "webSearch") {
        updateLoadingMessage?.(LOADING_MESSAGES.SEARCHING_WEB);
      } else if (toolCall.tool.name === "getFileTree") {
        updateLoadingMessage?.(LOADING_MESSAGES.READING_FILE_TREE);
      }
      const output = await ToolManager.callTool(toolCall.tool, toolCall.args);
      toolOutputs.push({ tool: toolCall.tool.name, output });
    }
    return toolOutputs;
  }

  private prepareEnhancedUserMessage(userMessage: string, toolOutputs: any[]) {
    let context = "";
    if (toolOutputs.length > 0) {
      const validOutputs = toolOutputs.filter((output) => output.output != null);
      if (validOutputs.length > 0) {
        context =
          "\n\n# Additional context:\n\n" +
          validOutputs
            .map(
              (output) =>
                `<${output.tool}>\n${typeof output.output !== "string" ? JSON.stringify(output.output) : output.output}\n</${output.tool}>`
            )
            .join("\n\n");
      }
    }
    return `${userMessage}${context}`;
  }

  private getTimeExpression(toolCalls: any[]): string {
    const timeRangeCall = toolCalls.find((call) => call.tool.name === "getTimeRangeMs");
    return timeRangeCall ? timeRangeCall.args.timeExpression : "";
  }

  private prepareLocalSearchResult(documents: any[], timeExpression: string): string {
    // First filter documents with includeInContext
    const includedDocs = documents.filter((doc) => doc.includeInContext);

    // Calculate total content length
    const totalLength = includedDocs.reduce((sum, doc) => sum + doc.content.length, 0);

    // If total length exceeds threshold, calculate truncation ratio
    let truncatedDocs = includedDocs;
    if (totalLength > MAX_CHARS_FOR_LOCAL_SEARCH_CONTEXT) {
      const truncationRatio = MAX_CHARS_FOR_LOCAL_SEARCH_CONTEXT / totalLength;
      logInfo("Truncating documents to fit context length. Truncation ratio:", truncationRatio);
      truncatedDocs = includedDocs.map((doc) => ({
        ...doc,
        content: doc.content.slice(0, Math.floor(doc.content.length * truncationRatio)),
      }));
    }

    const formattedDocs = truncatedDocs
      .map((doc: any) => `Note in Vault: ${doc.content}`)
      .join("\n\n");

    return timeExpression
      ? `Local Search Result for ${timeExpression}:\n${formattedDocs}`
      : `Local Search Result:\n${formattedDocs}`;
  }

  protected async getSystemPrompt(): Promise<string> {
    return getSystemPrompt();
  }
}

class ProjectChainRunner extends CopilotPlusChainRunner {
  protected async getSystemPrompt(): Promise<string> {
    let finalPrompt = getSystemPrompt();
    const projectConfig = getCurrentProject();
    if (!projectConfig) {
      return finalPrompt;
    }

    // Get context asynchronously
    const context = await ProjectManager.instance.getProjectContext(projectConfig.id);
    finalPrompt = `${finalPrompt}\n\n<project_system_prompt>\n${projectConfig.systemPrompt}\n</project_system_prompt>`;

    // TODO: Move project context out of the system prompt and into the user prompt.
    if (context) {
      finalPrompt = `${finalPrompt}\n\n <project_context>\n${context}\n</project_context>`;
    }

    return finalPrompt;
  }
}

export { CopilotPlusChainRunner, LLMChainRunner, ProjectChainRunner, VaultQAChainRunner };
