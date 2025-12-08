import { Tool } from "@langchain/core/tools";
import { requestUrl } from "obsidian";
import * as cheerio from "cheerio";

interface SearchApiResult {
  title: string;
  link: string;
  snippet?: string;
}

/**
 * 免费搜索工具
 * 使用多个免费搜索API，无需API Key
 * 在中国可用
 */
export class FreeBingSearchTool extends Tool {
  name = "free_search";
  description =
    "A free search tool. Use this when you need to find information about current events, specific product models, or any recent information that is not in your knowledge base.";

  public async _call(query: string): Promise<string> {
    try {
      console.log(`[Free Search] Starting search for query: "${query}"`);

      // 方法1: 优先使用百度（在中国速度快，中文搜索效果好）
      let results = await this.searchBaidu(query);
      let source = "百度搜索";

      if (!results || results.length === 0) {
        console.log("[Free Search] Baidu failed, trying Bing...");
        // 方法2: 备用必应国际版
        results = await this.searchBing(query);
        source = "必应搜索";
      }

      if (!results || results.length === 0) {
        console.log("[Free Search] Bing failed, trying DuckDuckGo...");
        // 方法3: 最后尝试DuckDuckGo
        results = await this.searchDuckDuckGoLite(query);
        source = "DuckDuckGo";
      }

      if (!results || results.length === 0) {
        console.log("[Free Search] All methods failed");
        return this.getFallbackMessage(query);
      }

      console.log(`[Free Search] Found ${results.length} results from ${source}`);

      // 格式化结果，与智谱API格式保持一致
      const formattedResults = results
        .slice(0, 5)
        .map(
          (result) =>
            `Title: ${result.title}\nLink: ${result.link}\nSource: ${source}\nPublished Date: Recent\nSummary: ${result.snippet || "相关搜索结果"}`
        )
        .join("\n-----------------\n");

      return formattedResults;
    } catch (error) {
      console.error("[Free Search] Error during search:", error);
      return this.getFallbackMessage(query);
    }
  }

  /**
   * 使用百度搜索
   */
  private async searchBaidu(query: string): Promise<SearchApiResult[]> {
    try {
      console.log("[Baidu] Searching...");
      const searchUrl = `https://www.baidu.com/s?wd=${encodeURIComponent(query)}&rn=10`;

      const response = await requestUrl({
        url: searchUrl,
        method: "GET",
      });

      // 调试：输出HTML片段
      const htmlPreview = response.text.substring(0, 500);
      console.log("[Baidu] HTML preview:", htmlPreview);

      const results = this.parseBaiduResults(response.text);
      console.log(`[Baidu] Found ${results.length} results`);

      // 如果没找到结果，输出更多调试信息
      if (results.length === 0) {
        console.log("[Baidu] Failed to parse results. Checking HTML structure...");
        const $ = cheerio.load(response.text);
        console.log("[Baidu] .result elements:", $(".result").length);
        console.log("[Baidu] [data-tools] elements:", $("[data-tools]").length);
        console.log("[Baidu] .c-container elements:", $(".c-container").length);
      }

      return results;
    } catch (error) {
      console.error("[Baidu] Search failed:", error);
      return [];
    }
  }

  /**
   * 使用cheerio解析百度搜索结果
   */
  private parseBaiduResults(html: string): SearchApiResult[] {
    const results: SearchApiResult[] = [];

    try {
      // 使用cheerio加载HTML
      const $ = cheerio.load(html);

      // 尝试多种百度结果容器选择器
      const selectors = [
        ".result", // 标准结果
        ".c-container", // 新版容器
        "[tpl]", // 带模板属性
        "#content_left > div", // 左侧内容区
      ];

      for (const selector of selectors) {
        if (results.length >= 5) break;

        $(selector).each((index, element) => {
          if (results.length >= 5) return false;

          const $element = $(element);

          // 跳过广告
          if ($element.hasClass("ec_wise_ad") || $element.attr("data-is-ad")) {
            return;
          }

          // 尝试多种标题选择器
          let titleElement = $element.find('h3 a, .t a, [class*="title"] a').first();
          let title = titleElement.text().trim();
          let link = titleElement.attr("href");

          // 如果没找到，尝试直接查找链接
          if (!link) {
            const allLinks = $element.find("a[href]");
            for (let i = 0; i < allLinks.length; i++) {
              const $link = $(allLinks[i]);
              const href = $link.attr("href");
              if (href && !href.includes("baidu.com/link") && !href.startsWith("javascript:")) {
                link = href;
                if (!title) {
                  title = $link.text().trim();
                }
                break;
              }
            }
          }

          // 提取摘要 - 尝试多个选择器
          const snippet = $element
            .find('.c-abstract, .content-right_8Zs40, [class*="abstract"], .c-span9, .c-span-last')
            .first()
            .text()
            .trim()
            .substring(0, 200); // 限制长度

          // 验证并添加结果
          if (
            title &&
            link &&
            !link.includes("baidu.com/link") &&
            !link.startsWith("javascript:")
          ) {
            results.push({
              title,
              link,
              snippet: snippet || "百度搜索结果",
            });
          }
        });

        // 如果已经找到结果，不需要尝试其他选择器
        if (results.length > 0) break;
      }

      return results;
    } catch (error) {
      console.error("[Baidu] Error parsing results with cheerio:", error);
      return [];
    }
  }

  /**
   * 使用必应搜索（参考用户提供的代码，使用cheerio解析）
   */
  private async searchBing(query: string): Promise<SearchApiResult[]> {
    try {
      console.log("[Bing] Searching...");
      // 使用cn.bing.com（国内版必应，速度更快）
      const encodedQuery = encodeURIComponent(query);
      const searchUrl = `https://cn.bing.com/search?q=${encodedQuery}`;

      const response = await requestUrl({
        url: searchUrl,
        method: "GET",
      });

      // 调试：输出HTML片段
      const htmlPreview = response.text.substring(0, 500);
      console.log("[Bing] HTML preview:", htmlPreview);

      const results = this.parseBingResultsWithCheerio(response.text);
      console.log(`[Bing] Found ${results.length} results`);

      // 输出调试信息
      if (results.length > 0) {
        console.log("[Bing] Sample result:", results[0]);
      } else {
        const $ = cheerio.load(response.text);
        console.log("[Bing] li.b_algo elements:", $("li.b_algo").length);
        console.log("[Bing] .b_algo elements:", $(".b_algo").length);
      }

      return results;
    } catch (error) {
      console.error("[Bing] Search failed:", error);
      return [];
    }
  }

  /**
   * 使用cheerio解析必应搜索结果（更可靠的方法）
   */
  private parseBingResultsWithCheerio(html: string): SearchApiResult[] {
    const results: SearchApiResult[] = [];

    try {
      // 使用cheerio加载HTML
      const $ = cheerio.load(html);

      // 必应的搜索结果项在 class="b_algo" 的 <li> 标签里
      $("li.b_algo").each((index, element) => {
        if (results.length >= 5) return false; // 只取前5条

        const $element = $(element);
        const titleElement = $element.find("h2 a");

        const title = titleElement.text().trim();
        const link = titleElement.attr("href");

        // 提取摘要信息 - 尝试多个选择器以获得更完整的摘要
        let snippet = "";

        // 方法1: 标准摘要
        snippet = $element.find(".b_caption p, .b_caption, p").first().text().trim();

        // 方法2: 如果没找到，尝试其他位置
        if (!snippet) {
          snippet = $element
            .find('.b_algoSlug, .b_snippetBigText, [class*="snippet"]')
            .first()
            .text()
            .trim();
        }

        // 方法3: 获取所有文本内容（排除标题）
        if (!snippet) {
          const allText = $element.text().trim();
          // 从全部文本中移除标题，剩下的作为摘要
          snippet = allText.replace(title, "").trim();
        }

        // 限制摘要长度并清理
        snippet = snippet.substring(0, 300).replace(/\s+/g, " ").trim();

        if (title && link) {
          results.push({
            title: title,
            link: link,
            snippet: snippet || "相关搜索结果",
          });
        }
      });

      return results;
    } catch (error) {
      console.error("[Bing] Error parsing results with cheerio:", error);
      return [];
    }
  }

  /**
   * 使用DuckDuckGo Lite搜索（更简单的HTML结构）
   */
  private async searchDuckDuckGoLite(query: string): Promise<SearchApiResult[]> {
    try {
      console.log("[DuckDuckGo] Searching...");
      // 使用DuckDuckGo Lite版本，HTML结构更简单
      const searchUrl = `https://lite.duckduckgo.com/lite/?q=${encodeURIComponent(query)}`;

      const response = await requestUrl({
        url: searchUrl,
        method: "GET",
      });

      const results = this.parseDuckDuckGoLite(response.text);
      console.log(`[DuckDuckGo] Found ${results.length} results`);
      return results;
    } catch (error) {
      console.error("[DuckDuckGo] Search failed:", error);
      return [];
    }
  }

  /**
   * 解析DuckDuckGo Lite结果
   */
  private parseDuckDuckGoLite(html: string): SearchApiResult[] {
    const results: SearchApiResult[] = [];

    try {
      // DuckDuckGo Lite 使用简单的表格结构
      // 查找所有包含结果的行
      const lines = html.split("\n");
      let currentTitle = "";
      let currentLink = "";
      let currentSnippet = "";

      for (let i = 0; i < lines.length && results.length < 5; i++) {
        const line = lines[i].trim();

        // 查找结果链接
        const linkMatch = line.match(
          /<a[^>]*class="[^"]*result-link[^"]*"[^>]*href="([^"]*)"[^>]*>(.*?)<\/a>/i
        );
        if (linkMatch) {
          currentLink = linkMatch[1];
          currentTitle = this.cleanText(linkMatch[2]);
        }

        // 查找摘要
        const snippetMatch = line.match(
          /<td[^>]*class="[^"]*result-snippet[^"]*"[^>]*>(.*?)<\/td>/i
        );
        if (snippetMatch) {
          currentSnippet = this.cleanText(snippetMatch[1]);
        }

        // 如果有标题和链接，添加结果
        if (currentTitle && currentLink && !currentLink.includes("duckduckgo.com")) {
          results.push({
            title: currentTitle,
            link: currentLink,
            snippet: currentSnippet || "搜索结果",
          });
          currentTitle = "";
          currentLink = "";
          currentSnippet = "";
        }
      }

      return results;
    } catch (error) {
      console.error("[DuckDuckGo] Error parsing results:", error);
      return [];
    }
  }

  /**
   * 获取降级提示信息
   */
  private getFallbackMessage(query: string): string {
    return `暂时无法完成网络搜索。

建议：
1. 尝试切换到智谱AI搜索（在设置中配置API Key）
2. 检查网络连接是否正常
3. 稍后重试

您的搜索词："${query}"

如果这是技术问题，我可以基于我的知识库尝试回答，但可能不包含最新信息。`;
  }

  /**
   * 清理文本中的HTML标签和特殊字符
   */
  private cleanText(text: string): string {
    return text
      .replace(/<[^>]*>/g, "") // 移除HTML标签
      .replace(/&amp;/g, "&")
      .replace(/&lt;/g, "<")
      .replace(/&gt;/g, ">")
      .replace(/&quot;/g, '"')
      .replace(/&#39;/g, "'")
      .replace(/&nbsp;/g, " ")
      .replace(/&middot;/g, "·")
      .replace(/\s+/g, " ") // 合并多个空格
      .trim();
  }
}
