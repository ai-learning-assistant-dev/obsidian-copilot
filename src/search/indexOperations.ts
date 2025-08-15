import EmbeddingsManager from "@/LLMProviders/embeddingManager";
import { logError, logInfo } from "@/logger";
import { RateLimiter } from "@/rateLimiter";
import { getSettings, subscribeToSettingsChange } from "@/settings/model";
import { formatDateTime } from "@/utils";
import {
  findAllWorkspaces,
  WorkspaceInfo,
  getWorkspaceConfigByInfo,
  DEFAULT_WORKSPACE_CHUNK_SIZE,
} from "@/utils/workspaceUtils";
import { MD5 } from "crypto-js";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { App, Notice, TFile } from "obsidian";
import { DBOperations } from "./dbOperations";
import { preprocessMarkdownDocument } from "./markdownPreprocessor";
import {
  extractAppIgnoreSettings,
  getDecodedPatterns,
  getMatchingPatterns,
  shouldIndexFile,
} from "./searchUtils";

export interface IndexingState {
  isIndexingPaused: boolean;
  isIndexingCancelled: boolean;
  indexedCount: number;
  totalFilesToIndex: number;
  processedFiles: Set<string>;
  currentIndexingNotice: Notice | null;
  indexNoticeMessage: HTMLSpanElement | null;
}

export class IndexOperations {
  private rateLimiter: RateLimiter;
  private checkpointInterval: number;
  private embeddingBatchSize: number;
  private state: IndexingState = {
    isIndexingPaused: false,
    isIndexingCancelled: false,
    indexedCount: 0,
    totalFilesToIndex: 0,
    processedFiles: new Set(),
    currentIndexingNotice: null,
    indexNoticeMessage: null,
  };

  constructor(
    private app: App,
    private dbOps: DBOperations,
    private embeddingsManager: EmbeddingsManager
  ) {
    const settings = getSettings();
    this.rateLimiter = new RateLimiter(settings.embeddingRequestsPerMin);
    this.embeddingBatchSize = settings.embeddingBatchSize;
    this.checkpointInterval = 8 * this.embeddingBatchSize;

    // Subscribe to settings changes
    subscribeToSettingsChange(async () => {
      const settings = getSettings();
      this.rateLimiter = new RateLimiter(settings.embeddingRequestsPerMin);
      this.embeddingBatchSize = settings.embeddingBatchSize;
      this.checkpointInterval = 8 * this.embeddingBatchSize;
    });
  }

  public async indexVaultToVectorStore(overwrite?: boolean): Promise<number> {
    const errors: string[] = [];

    try {
      const embeddingInstance = await this.embeddingsManager.getEmbeddingsAPI();
      if (!embeddingInstance) {
        console.error("Embedding instance not found.");
        return 0;
      }

      // Check for model change first
      const modelChanged = await this.dbOps.checkAndHandleEmbeddingModelChange(embeddingInstance);
      if (modelChanged) {
        // If model changed, force a full reindex by setting overwrite to true
        overwrite = true;
      }

      // Clear index and tracking if overwrite is true
      if (overwrite) {
        await this.dbOps.clearIndex(embeddingInstance);
        this.dbOps.clearFilesMissingEmbeddings();
      } else {
        // Run garbage collection first to clean up stale documents
        await this.dbOps.garbageCollect();
      }

      const files = await this.getFilesToIndex(overwrite);
      if (files.length === 0) {
        new Notice("Copilot vault index is up-to-date.");
        return 0;
      }

      // 先准备chunks来获得实际要处理的文件数量
      logInfo("Preparing chunks and filtering by workspace...");
      const allChunks = await this.prepareAllChunks(files);
      if (allChunks.length === 0) {
        new Notice("No valid content to index (no files in workspaces).");
        return 0;
      }

      // 统计实际要处理的文件数量
      const actualFilesToProcess = new Set(allChunks.map((chunk) => chunk.fileInfo.path)).size;
      this.initializeIndexingState(actualFilesToProcess);
      this.createIndexingNotice();

      // Clear the missing embeddings list before starting new indexing
      this.dbOps.clearFilesMissingEmbeddings();

      // 为了避免旧索引残留：在写入新的chunk之前，先删除这些文件的所有旧文档
      try {
        const uniquePathsToReindex = Array.from(
          new Set(allChunks.map((chunk) => chunk.fileInfo.path))
        );
        for (const path of uniquePathsToReindex) {
          await this.dbOps.removeDocs(path);
        }
      } catch (err) {
        this.handleError(err, { errors });
      }

      // Process chunks in batches
      for (let i = 0; i < allChunks.length; i += this.embeddingBatchSize) {
        if (this.state.isIndexingCancelled) break;
        await this.handlePause();

        const batch = allChunks.slice(i, i + this.embeddingBatchSize);
        try {
          await this.rateLimiter.wait();
          const embeddings = await embeddingInstance.embedDocuments(
            batch.map((chunk) => chunk.content)
          );

          // Validate embeddings
          if (!embeddings || embeddings.length !== batch.length) {
            throw new Error(
              `Embedding model returned ${embeddings?.length ?? 0} embeddings for ${batch.length} documents`
            );
          }

          // Save batch to database
          for (let j = 0; j < batch.length; j++) {
            const chunk = batch[j];
            const embedding = embeddings[j];

            // Skip documents with invalid embeddings
            if (!embedding || !Array.isArray(embedding) || embedding.length === 0) {
              logError(`Invalid embedding for document ${chunk.fileInfo.path}: ${embedding}`);
              this.dbOps.markFileMissingEmbeddings(chunk.fileInfo.path);
              continue;
            }

            try {
              await this.dbOps.upsert({
                ...chunk.fileInfo,
                id: this.getDocHash(chunk.content, chunk.fileInfo.path, chunk.subtitle),
                content: chunk.content,
                embedding,
                created_at: Date.now(),
                nchars: chunk.content.length,
                subtitle: chunk.subtitle || "/", // 确保subtitle字段始终有值
              });
              // Mark success for this file
              this.state.processedFiles.add(chunk.fileInfo.path);
            } catch (err) {
              // Log error but continue processing other documents in batch
              this.handleError(err, {
                filePath: chunk.fileInfo.path,
                errors,
              });
              this.dbOps.markFileMissingEmbeddings(chunk.fileInfo.path);
              continue;
            }
          }

          // Update progress after the batch
          this.state.indexedCount = this.state.processedFiles.size;
          this.updateIndexingNoticeMessage();

          // Calculate if we've crossed a checkpoint threshold
          const previousCheckpoint = Math.floor(
            (this.state.indexedCount - batch.length) / this.checkpointInterval
          );
          const currentCheckpoint = Math.floor(this.state.indexedCount / this.checkpointInterval);

          if (currentCheckpoint > previousCheckpoint) {
            await this.dbOps.saveDB();
            console.log("Copilot index checkpoint save completed.");
          }
        } catch (err) {
          this.handleError(err, {
            filePath: batch?.[0]?.fileInfo?.path,
            errors,
            batch,
          });
          if (this.isRateLimitError(err)) {
            break;
          }
        }
      }

      // Show completion notice before running integrity check
      this.finalizeIndexing(errors);

      // Run save and integrity check with setTimeout to ensure it's non-blocking
      setTimeout(() => {
        this.dbOps
          .saveDB()
          .then(() => {
            logInfo("Copilot index final save completed.");
            this.dbOps.checkIndexIntegrity().catch((err) => {
              logError("Background integrity check failed:", err);
            });

            // 自动展示数据库内容
            setTimeout(() => {
              this.displayDatabaseContents();
            }, 500);
          })
          .catch((err) => {
            logError("Background save failed:", err);
          });
      }, 100); // 100ms delay

      return this.state.indexedCount;
    } catch (error) {
      this.handleError(error);
      return 0;
    }
  }

  private async prepareAllChunks(files: TFile[]): Promise<
    Array<{
      content: string;
      fileInfo: any;
      subtitle?: string;
    }>
  > {
    const embeddingInstance = await this.embeddingsManager.getEmbeddingsAPI();
    if (!embeddingInstance) {
      console.error("Embedding instance not found.");
      return [];
    }
    const embeddingModel = EmbeddingsManager.getModelName(embeddingInstance);

    // 获取所有工作区信息
    const allWorkspaces: WorkspaceInfo[] = await findAllWorkspaces(this.app);
    logInfo(`Found ${allWorkspaces.length} workspaces for indexing`);

    // 辅助函数：确定文件属于哪个工作区
    const getWorkspaceForFile = (filePath: string): WorkspaceInfo | null => {
      // 按路径长度排序，最长的路径优先匹配（更具体的工作区优先）
      const sortedWorkspaces = [...allWorkspaces].sort(
        (a, b) => b.relativePath.length - a.relativePath.length
      );

      for (const workspace of sortedWorkspaces) {
        if (workspace.relativePath === "/") {
          // 根目录工作区：只有当文件直接在根目录且没有其他更具体的匹配时才匹配
          // 检查文件是否在根目录下（不包含子目录）
          if (!filePath.includes("/")) {
            return workspace;
          }
        } else {
          // 非根目录工作区：检查文件是否在此工作区路径下
          if (filePath.startsWith(workspace.relativePath + "/")) {
            return workspace;
          }
          // 检查是否是工作区目录本身的文件（如果有的话）
          if (filePath === workspace.relativePath) {
            return workspace;
          }
        }
      }
      return null;
    };

    // 缓存工作区配置和对应的文本分割器
    const workspaceConfigCache = new Map<
      string,
      {
        chunkSize: number;
        textSplitter: RecursiveCharacterTextSplitter;
        excludedPaths: string[];
      }
    >();

    // 获取工作区配置的辅助函数
    const getWorkspaceConfigData = async (workspace: WorkspaceInfo) => {
      const cacheKey = workspace.relativePath;

      if (workspaceConfigCache.has(cacheKey)) {
        return workspaceConfigCache.get(cacheKey)!;
      }

      // 获取工作区配置
      let chunkSize = DEFAULT_WORKSPACE_CHUNK_SIZE; // 默认chunk size
      let excludedPaths: string[] = [];

      try {
        const workspaceConfig = await getWorkspaceConfigByInfo(this.app, workspace);
        if (workspaceConfig?.chunk_size && workspaceConfig.chunk_size > 0) {
          chunkSize = workspaceConfig.chunk_size;
          logInfo(`Using custom chunk size ${chunkSize} for workspace ${workspace.name}`);
        } else {
          logInfo(`Using default chunk size ${chunkSize} for workspace ${workspace.name}`);
        }

        // 获取排除路径
        if (workspaceConfig?.excludedPaths && Array.isArray(workspaceConfig.excludedPaths)) {
          excludedPaths = workspaceConfig.excludedPaths;
          logInfo(
            `Found ${excludedPaths.length} excluded paths for workspace ${workspace.name}: ${excludedPaths.join(", ")}`
          );
        }
      } catch (error) {
        console.warn(
          `Failed to load config for workspace ${workspace.name}, using default settings:`,
          error
        );
      }

      // 创建文本分割器
      const textSplitter = RecursiveCharacterTextSplitter.fromLanguage("markdown", {
        chunkSize: chunkSize,
      });

      // 缓存配置和分割器
      const configData = { chunkSize, textSplitter, excludedPaths };
      workspaceConfigCache.set(cacheKey, configData);

      return configData;
    };

    const allChunks: Array<{ content: string; fileInfo: any; subtitle?: string }> = [];

    for (const file of files) {
      // 跳过 data.md 配置文件，不进行向量化
      if (file.name === "data.md") {
        logInfo(`Skipping configuration file ${file.path} - not indexing data.md files`);
        continue;
      }

      // 检查文件是否属于某个工作区
      const fileWorkspace = getWorkspaceForFile(file.path);
      if (!fileWorkspace) {
        // 不在任何工作区中的文件跳过索引
        logInfo(`Skipping file ${file.path} - not in any workspace`);
        continue;
      }

      // 获取该工作区的配置数据
      const workspaceConfigData = await getWorkspaceConfigData(fileWorkspace);

      // 检查文件是否在排除列表中
      if (workspaceConfigData.excludedPaths.includes(file.name)) {
        logInfo(
          `Skipping file ${file.path} - excluded by workspace ${fileWorkspace.name} (excludedPaths)`
        );
        continue;
      }

      const content = await this.app.vault.cachedRead(file);
      if (!content?.trim()) continue;

      // 获取该工作区对应的文本分割器
      const textSplitter = workspaceConfigData.textSplitter;

      const fileCache = this.app.metadataCache.getFileCache(file);
      const fileInfo = {
        title: file.basename,
        path: file.path,
        embeddingModel,
        ctime: file.stat.ctime,
        mtime: file.stat.mtime,
        tags: fileCache?.tags?.map((tag) => tag.tag) ?? [],
        extension: file.extension,
        workspace_name: fileWorkspace.name,
        workspace_path: fileWorkspace.relativePath,
        metadata: {
          ...(fileCache?.frontmatter ?? {}),
          created: formatDateTime(new Date(file.stat.ctime)).display,
          modified: formatDateTime(new Date(file.stat.mtime)).display,
        },
      };

      // 首先进行markdown预分段处理
      const markdownSections = preprocessMarkdownDocument(content);

      // 对每个section进行chunk分割
      for (const section of markdownSections) {
        // Add note title as contextual chunk headers
        // https://js.langchain.com/docs/modules/data_connection/document_transformers/contextual_chunk_headers
        const chunkHeader = `\n\nNOTE TITLE: [[${fileInfo.title}]]\n\nSUBTITLE: ${section.subtitle}\n\nMETADATA:${JSON.stringify(
          fileInfo.metadata
        )}\n\nNOTE BLOCK CONTENT:\n\n`;

        const chunks = await textSplitter.createDocuments([section.content], [], {
          chunkHeader,
          appendChunkOverlapHeader: true,
        });

        chunks.forEach((chunk) => {
          if (chunk.pageContent.trim()) {
            allChunks.push({
              content: chunk.pageContent,
              fileInfo,
              subtitle: section.subtitle,
            });
          }
        });
      }
    }

    // 输出每个工作区使用的chunk size信息
    logInfo(`Workspace chunk size summary:`);
    workspaceConfigCache.forEach((config, workspacePath) => {
      logInfo(
        `  - ${workspacePath}: ${config.chunkSize} (excluded: ${config.excludedPaths.length} files)`
      );
    });

    logInfo(`Prepared ${allChunks.length} chunks from workspace files`);

    // Debug模式下输出前两个分块的文本内容作为示例
    if (getSettings().debug) {
      console.log("=== DEBUG: Sample Chunks Content (showing first 2) ===");
      allChunks.slice(0, 2).forEach((chunk, index) => {
        console.log(`\n--- Chunk ${index + 1}/${allChunks.length} ---`);
        console.log(`File: ${chunk.fileInfo.path}`);
        console.log(
          `Workspace: ${chunk.fileInfo.workspace_name} (${chunk.fileInfo.workspace_path})`
        );
        console.log(`Subtitle: ${chunk.subtitle || "/"}`);
        console.log(`Length: ${chunk.content.length} characters`);
        console.log("Content:");
        console.log(chunk.content);
        console.log("--- End of Chunk ---");
      });
      if (allChunks.length > 2) {
        console.log(`\n... 省略剩余 ${allChunks.length - 2} 个chunk ...`);
      }
      console.log("=== END DEBUG: Sample Chunks Content ===\n");
    }

    return allChunks;
  }

  private getDocHash(sourceDocument: string, path: string, subtitle?: string): string {
    const sub = subtitle || "/";
    return MD5(`${path}|${sub}|${sourceDocument}`).toString();
  }

  /**
   * 展示数据库中所有文档的详细信息，包括workspace属性
   */
  public async displayDatabaseContents(): Promise<void> {
    try {
      const db = await this.dbOps.getDb();
      if (!db) {
        console.log("❌ 数据库未初始化");
        new Notice("数据库未初始化，请先创建索引");
        return;
      }

      const allDocuments = await DBOperations.getAllDocuments(db);

      if (allDocuments.length === 0) {
        console.log("📭 数据库为空，没有已索引的文档");
        new Notice("数据库为空，没有已索引的文档");
        return;
      }

      console.log(`\n🗃️  数据库内容展示 - 共 ${allDocuments.length} 个文档片段\n`);
      console.log("=".repeat(80));

      // 按工作区分组统计
      const workspaceStats = new Map<string, number>();
      const workspaceDetails = new Map<string, any[]>();

      allDocuments.forEach((doc, index) => {
        const workspaceName = doc.workspace_name || "未分配工作区";
        const workspacePath = doc.workspace_path || "无路径";
        const workspaceKey = `${workspaceName} (${workspacePath})`;

        // 统计数量
        workspaceStats.set(workspaceKey, (workspaceStats.get(workspaceKey) || 0) + 1);

        // 收集详细信息
        if (!workspaceDetails.has(workspaceKey)) {
          workspaceDetails.set(workspaceKey, []);
        }

        workspaceDetails.get(workspaceKey)!.push({
          index: index + 1,
          id: doc.id,
          title: doc.title,
          path: doc.path,
          subtitle: doc.subtitle || "/",
          extension: doc.extension,
          tags: doc.tags,
          contentLength: doc.content?.length || 0,
          embeddingModel: doc.embeddingModel,
          created: new Date(doc.created_at).toLocaleString(),
          modified: new Date(doc.mtime).toLocaleString(),
        });
      });

      // 展示工作区统计
      console.log("📊 工作区统计:");
      workspaceStats.forEach((count, workspace) => {
        console.log(`   ${workspace}: ${count} 个文档片段`);
      });

      console.log("\n" + "=".repeat(80));

      // 展示每个工作区的详细信息（仅显示前2个文档作为示例）
      workspaceDetails.forEach((docs, workspaceKey) => {
        console.log(`\n📁 工作区: ${workspaceKey}`);
        console.log("-".repeat(60));

        const docsToShow = docs.slice(0, 2);
        docsToShow.forEach((doc) => {
          console.log(`
  📄 [${doc.index}] ${doc.title}
     📍 路径: ${doc.path}
     📑 标题路径: ${doc.subtitle}
     🔗 ID: ${doc.id.substring(0, 16)}...
     📝 内容长度: ${doc.contentLength} 字符
     🏷️  标签: ${doc.tags.join(", ") || "无"}
     🤖 嵌入模型: ${doc.embeddingModel}
     📅 创建时间: ${doc.created}
     ✏️  修改时间: ${doc.modified}
     📐 扩展名: ${doc.extension}`);
        });

        if (docs.length > 2) {
          console.log(`\n  ... 省略剩余 ${docs.length - 2} 个文档 ...`);
        }
      });

      console.log("\n" + "=".repeat(80));
      console.log(`✅ 数据库内容展示完成 - 总计 ${allDocuments.length} 个文档片段`);

      new Notice(
        `数据库包含 ${allDocuments.length} 个文档片段，分布在 ${workspaceStats.size} 个工作区中`
      );
    } catch (error) {
      console.error("❌ 展示数据库内容时出错:", error);
      new Notice("展示数据库内容时出错，请查看控制台");
    }
  }

  private async getFilesToIndex(overwrite?: boolean): Promise<TFile[]> {
    const { inclusions, exclusions } = getMatchingPatterns();
    const allMarkdownFiles = this.app.vault.getMarkdownFiles();

    // If overwrite is true, return all markdown files that match current filters
    if (overwrite) {
      return allMarkdownFiles.filter((file) => {
        return shouldIndexFile(file, inclusions, exclusions);
      });
    }

    // Get currently indexed files and latest mtime
    const indexedFilePaths = new Set(await this.dbOps.getIndexedFiles());
    const latestMtime = await this.dbOps.getLatestFileMtime();
    const filesMissingEmbeddings = new Set(this.dbOps.getFilesMissingEmbeddings());

    // Get all markdown files that should be indexed under current rules
    const filesToIndex = new Set<TFile>();
    const emptyFiles = new Set<string>();

    for (const file of allMarkdownFiles) {
      if (!shouldIndexFile(file, inclusions, exclusions)) {
        continue;
      }

      // Check actual content
      const content = await this.app.vault.cachedRead(file);
      if (!content || content.trim().length === 0) {
        emptyFiles.add(file.path);
        continue;
      }

      const isIndexed = indexedFilePaths.has(file.path);
      const needsEmbeddings = filesMissingEmbeddings.has(file.path);

      if (!isIndexed || needsEmbeddings || file.stat.mtime > latestMtime) {
        filesToIndex.add(file);
      }
    }

    logInfo(
      [
        `Files to index: ${filesToIndex.size}`,
        `Previously indexed: ${indexedFilePaths.size}`,
        `Empty files skipped: ${emptyFiles.size}`,
        `Files missing embeddings: ${filesMissingEmbeddings.size}`,
      ].join("\n")
    );

    return Array.from(filesToIndex);
  }

  private initializeIndexingState(totalFiles: number) {
    this.state = {
      isIndexingPaused: false,
      isIndexingCancelled: false,
      indexedCount: 0,
      totalFilesToIndex: totalFiles,
      processedFiles: new Set(),
      currentIndexingNotice: null,
      indexNoticeMessage: null,
    };
  }

  private createIndexingNotice(): Notice {
    const frag = document.createDocumentFragment();
    const container = frag.createEl("div", { cls: "copilot-notice-container" });

    this.state.indexNoticeMessage = container.createEl("div", { cls: "copilot-notice-message" });
    this.updateIndexingNoticeMessage();

    // Create button container for better layout
    const buttonContainer = container.createEl("div", { cls: "copilot-notice-buttons" });

    // Pause/Resume button
    const pauseButton = buttonContainer.createEl("button");
    pauseButton.textContent = "Pause";
    pauseButton.addEventListener("click", (event) => {
      event.stopPropagation();
      event.preventDefault();
      if (this.state.isIndexingPaused) {
        this.resumeIndexing();
        pauseButton.textContent = "Pause";
      } else {
        this.pauseIndexing();
        pauseButton.textContent = "Resume";
      }
    });

    // Stop button
    const stopButton = buttonContainer.createEl("button");
    stopButton.textContent = "Stop";
    stopButton.style.marginLeft = "8px";
    stopButton.addEventListener("click", (event) => {
      event.stopPropagation();
      event.preventDefault();
      this.cancelIndexing();
    });

    frag.appendChild(this.state.indexNoticeMessage);
    frag.appendChild(buttonContainer);

    this.state.currentIndexingNotice = new Notice(frag, 0);
    return this.state.currentIndexingNotice;
  }

  private async handlePause(): Promise<void> {
    if (this.state.isIndexingPaused) {
      while (this.state.isIndexingPaused && !this.state.isIndexingCancelled) {
        await new Promise((resolve) => setTimeout(resolve, 100));
      }

      // After we exit the pause loop (meaning we've resumed), re-evaluate files
      if (!this.state.isIndexingCancelled) {
        const files = await this.getFilesToIndex();
        if (files.length === 0) {
          // If no files to index after filter change, cancel the indexing
          console.log("No files to index after filter change, stopping indexing");
          this.cancelIndexing();
          new Notice("No files to index with current filters");
          return;
        }
        this.state.totalFilesToIndex = files.length;
        console.log("Total files to index:", this.state.totalFilesToIndex);
        console.log("Files to index:", files);
        this.updateIndexingNoticeMessage();
      }
    }
  }

  private pauseIndexing(): void {
    this.state.isIndexingPaused = true;
  }

  private resumeIndexing(): void {
    this.state.isIndexingPaused = false;
  }

  private updateIndexingNoticeMessage(): void {
    if (this.state.indexNoticeMessage) {
      const status = this.state.isIndexingPaused ? " (Paused)" : "";
      const messages = [
        `Copilot is indexing your vault...`,
        `${this.state.indexedCount}/${this.state.totalFilesToIndex} files processed${status}`,
      ];

      const settings = getSettings();

      const inclusions = getDecodedPatterns(settings.qaInclusions);
      if (inclusions.length > 0) {
        messages.push(`Inclusions: ${inclusions.join(", ")}`);
      }

      const obsidianIgnoreFolders = extractAppIgnoreSettings(this.app);
      const exclusions = [...obsidianIgnoreFolders, ...getDecodedPatterns(settings.qaExclusions)];
      if (exclusions.length > 0) {
        messages.push(`Exclusions: ${exclusions.join(", ")}`);
      }

      this.state.indexNoticeMessage.textContent = messages.join("\n");
    }
  }

  private isStringLengthError(error: any): boolean {
    if (!error) return false;

    // Check if it's a direct RangeError
    if (error instanceof RangeError && error.message.toLowerCase().includes("string length")) {
      return true;
    }

    // Check the error message at any depth
    const message = error.message || error.toString();
    const lowerMessage = message.toLowerCase();
    return lowerMessage.includes("string length") || lowerMessage.includes("rangeerror");
  }

  private handleError(
    error: any,
    context?: {
      filePath?: string;
      errors?: string[];
      batch?: Array<{ content: string; fileInfo: any }>;
    }
  ): void {
    const filePath = context?.filePath;

    // Log the error with appropriate context
    if (filePath) {
      if (context.batch) {
        // Detailed batch processing error logging
        console.error("Batch processing error:", {
          error,
          batchSize: context.batch.length || 0,
          firstChunk: context.batch[0]
            ? {
                path: context.batch[0].fileInfo?.path,
                contentLength: context.batch[0].content?.length,
                hasFileInfo: !!context.batch[0].fileInfo,
              }
            : "No chunks in batch",
          errorType: error?.constructor?.name,
          errorMessage: error?.message,
        });
      } else {
        console.error(`Error indexing file ${filePath}:`, error);
      }
      context.errors?.push(filePath);
    } else {
      console.error("Fatal error during indexing:", error);
    }

    // Hide any existing indexing notice
    if (this.state.currentIndexingNotice) {
      this.state.currentIndexingNotice.hide();
    }

    // Handle json stringify string length error consistently
    if (this.isStringLengthError(error)) {
      new Notice(
        "Vault is too large for 1 partition, please increase the number of partitions in your Copilot QA settings!",
        10000 // Show for 10 seconds
      );
      return;
    }

    // Show appropriate error notice
    if (this.isRateLimitError(error)) {
      return; // Don't show duplicate notices for rate limit errors
    }

    const message = filePath
      ? `Error indexing file ${filePath}. Check console for details.`
      : "Fatal error during indexing. Check console for details.";
    new Notice(message);
  }

  private isRateLimitError(err: any): boolean {
    return err?.message?.includes?.("rate limit") || false;
  }

  private finalizeIndexing(errors: string[]): void {
    if (this.state.currentIndexingNotice) {
      this.state.currentIndexingNotice.hide();
    }

    if (this.state.isIndexingCancelled) {
      new Notice(`Indexing cancelled`);
      return;
    }

    if (errors.length > 0) {
      new Notice(`Indexing completed with ${errors.length} errors. Check console for details.`);
    } else {
      new Notice("Indexing completed successfully!");
    }
  }

  public async reindexFile(file: TFile): Promise<void> {
    try {
      const embeddingInstance = await this.embeddingsManager.getEmbeddingsAPI();
      if (!embeddingInstance) {
        return;
      }

      await this.dbOps.removeDocs(file.path);

      // Check for model change
      const modelChanged = await this.dbOps.checkAndHandleEmbeddingModelChange(embeddingInstance);
      if (modelChanged) {
        await this.indexVaultToVectorStore(true);
        return;
      }

      // Reuse prepareAllChunks with a single file
      const chunks = await this.prepareAllChunks([file]);
      if (chunks.length === 0) return;

      // Process chunks
      const embeddings = await embeddingInstance.embedDocuments(
        chunks.map((chunk) => chunk.content)
      );

      // Save to database
      for (let i = 0; i < chunks.length; i++) {
        const chunk = chunks[i];
        await this.dbOps.upsert({
          ...chunk.fileInfo,
          id: this.getDocHash(chunk.content, chunk.fileInfo.path, chunk.subtitle),
          content: chunk.content,
          embedding: embeddings[i],
          created_at: Date.now(),
          nchars: chunk.content.length,
          subtitle: chunk.subtitle || "/", // 确保subtitle字段始终有值
        });
      }

      // Mark that we have unsaved changes instead of saving immediately
      this.dbOps.markUnsavedChanges();

      if (getSettings().debug) {
        console.log(`Reindexed file: ${file.path}`);
      }
    } catch (error) {
      this.handleError(error, { filePath: file.path });
    }
  }

  public async cancelIndexing(): Promise<void> {
    console.log("Indexing cancelled by user");
    this.state.isIndexingCancelled = true;

    // Add a small delay to ensure all state updates are processed
    await new Promise((resolve) => setTimeout(resolve, 100));

    if (this.state.currentIndexingNotice) {
      this.state.currentIndexingNotice.hide();
    }
  }
}
