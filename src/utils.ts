import { ChainType, Document } from "@/chainFactory";
import {
  ChatModelProviders,
  EmbeddingModelProviders,
  NOMIC_EMBED_TEXT,
  Provider,
  ProviderInfo,
  ProviderMetadata,
  ProviderSettingsKeyMap,
  SettingKeyProviders,
  USER_SENDER,
} from "@/constants";
import { logInfo } from "@/logger";
import { CopilotSettings } from "@/settings/model";
import { ChatMessage } from "@/sharedState";
import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { MemoryVariables } from "@langchain/core/memory";
import { RunnableSequence } from "@langchain/core/runnables";
import { BaseChain, RetrievalQAChain } from "langchain/chains";
import moment from "moment";
import { MarkdownView, Notice, TFile, Vault, requestUrl } from "obsidian";
import { CustomModel } from "./aiParams";

export class Paragraph extends TFile {
  reference: string;
  lineRange?: { start: number; end: number };
  constructor(reference: string, file: TFile | Paragraph) {
    // because TFile is having a private constructor, we need to use @ts-ignore
    // @ts-ignore
    super(file.vault, file.path);
    this.reference = reference;
    this.path = file.path;
    this.name = file.name;
    this.extension = file.extension;
    this.basename = file.basename;
    this.parent = file.parent;
    this.vault = file.vault;
    this.stat = file.stat;
    if (file instanceof Paragraph) {
      if (file.lineRange) {
        this.lineRange = {
          start: file.lineRange.start,
          end: file.lineRange.end,
        };
      }
    }
  }
}

// Add custom error type at the top of the file
interface APIError extends Error {
  json?: any;
}

// Error message constants
export const ERROR_MESSAGES = {
  INVALID_LICENSE_KEY_USER:
    "Invalid Copilot Plus license key. Please check your license key in settings.",
  UNKNOWN_ERROR: "An unknown error occurred",
  REQUEST_FAILED: (status: number) => `Request failed, status ${status}`,
} as const;

// Error handling utilities
export interface ErrorDetail {
  status?: number;
  message?: string;
  reason?: string;
}

export function extractErrorDetail(error: any): ErrorDetail {
  const errorDetail = error?.detail || {};
  return {
    status: errorDetail.status,
    message: errorDetail.message || error?.message,
    reason: errorDetail.reason,
  };
}

export function isLicenseKeyError(error: any): boolean {
  const errorDetail = extractErrorDetail(error);
  return (
    errorDetail.reason === "Invalid license key" ||
    error?.message === "Invalid license key" ||
    error?.message?.includes("status 403") ||
    errorDetail.status === 403
  );
}

export function getApiErrorMessage(error: any): string {
  const errorDetail = extractErrorDetail(error);
  if (isLicenseKeyError(error)) {
    return ERROR_MESSAGES.INVALID_LICENSE_KEY_USER;
  }
  return (
    errorDetail.message ||
    (errorDetail.reason ? `Error: ${errorDetail.reason}` : ERROR_MESSAGES.UNKNOWN_ERROR)
  );
}

export const getModelNameFromKey = (modelKey: string): string => {
  return modelKey.split("|")[0];
};

export const isFolderMatch = (fileFullpath: string, inputPath: string): boolean => {
  const fileSegments = fileFullpath.split("/").map((segment) => segment.toLowerCase());
  return fileSegments.includes(inputPath.toLowerCase());
};

/** TODO: Rewrite with app.vault.getAbstractFileByPath() */
export const getNotesFromPath = (vault: Vault, path: string): TFile[] => {
  const files = vault.getMarkdownFiles();

  // Special handling for the root path '/'
  if (path === "/") {
    return files;
  }

  // Normalize the input path
  const normalizedPath = path.toLowerCase().replace(/^\/|\/$/g, "");

  return files.filter((file) => {
    // Normalize the file path
    const normalizedFilePath = file.path.toLowerCase();
    const filePathParts = normalizedFilePath.split("/");
    const pathParts = normalizedPath.split("/");

    // Check if the file path contains all parts of the input path in order
    let filePathIndex = 0;
    for (const pathPart of pathParts) {
      while (filePathIndex < filePathParts.length) {
        if (filePathParts[filePathIndex] === pathPart) {
          break;
        }
        filePathIndex++;
      }
      if (filePathIndex >= filePathParts.length) {
        return false;
      }
    }

    return true;
  });
};

/**
 * @param tag - The tag to strip the hash symbol from.
 * @returns The tag without the hash symbol in lowercase.
 */
export function stripHash(tag: string): string {
  return tag.replace(/^#/, "").trim().toLowerCase();
}

/**
 * @param file - The note file to get tags from.
 * @param frontmatterOnly - Whether to only get tags from frontmatter.
 * @returns An array of lowercase tags without the hash symbol.
 */
export function getTagsFromNote(file: TFile, frontmatterOnly = true): string[] {
  const metadata = app.metadataCache.getFileCache(file);
  const frontmatterTags = metadata?.frontmatter?.tags;
  const allTags = new Set<string>();

  if (!frontmatterOnly) {
    const inlineTags = metadata?.tags?.map((tag) => tag.tag);
    if (inlineTags) {
      inlineTags.forEach((tag) => allTags.add(stripHash(tag)));
    }
  }

  // Add frontmatter tags
  if (frontmatterTags) {
    if (Array.isArray(frontmatterTags)) {
      frontmatterTags.forEach((tag) => {
        if (typeof tag === "string") {
          allTags.add(stripHash(tag));
        }
      });
    } else if (typeof frontmatterTags === "string") {
      allTags.add(stripHash(frontmatterTags));
    }
  }

  return Array.from(allTags);
}

/**
 * Get notes from tags.
 * @param vault - The vault to get notes from.
 * @param tags - The tags to get notes from. Tags should be with the hash symbol.
 * @param noteFiles - The notes to get notes from.
 * @returns An array of note files.
 */
export function getNotesFromTags(vault: Vault, tags: string[], noteFiles?: TFile[]): TFile[] {
  if (tags.length === 0) {
    return [];
  }

  tags = tags.map((tag) => stripHash(tag));

  const files = noteFiles && noteFiles.length > 0 ? noteFiles : getNotesFromPath(vault, "/");
  const filesWithTag = [];

  for (const file of files) {
    const noteTags = getTagsFromNote(file);
    if (tags.some((tag) => noteTags.includes(tag))) {
      filesWithTag.push(file);
    }
  }

  return filesWithTag;
}

export const stringToChainType = (chain: string): ChainType => {
  switch (chain) {
    case "llm_chain":
      return ChainType.LLM_CHAIN;
    case "vault_qa":
      return ChainType.VAULT_QA_CHAIN;
    case "copilot_plus":
      return ChainType.COPILOT_PLUS_CHAIN;
    default:
      throw new Error(`Unknown chain type: ${chain}`);
  }
};

export const isLLMChain = (chain: RunnableSequence): chain is RunnableSequence => {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return (chain as any).last.bound.modelName || (chain as any).last.bound.model;
};

export const isRetrievalQAChain = (chain: BaseChain): chain is RetrievalQAChain => {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return (chain as any).last.bound.retriever !== undefined;
};

export const isSupportedChain = (chain: RunnableSequence): chain is RunnableSequence => {
  return isLLMChain(chain) || isRetrievalQAChain(chain);
};

// Returns the last N messages from the chat history,
// last one being the newest ai message
export const getChatContext = (chatHistory: ChatMessage[], contextSize: number) => {
  if (chatHistory.length === 0) {
    return [];
  }
  const lastAiMessageIndex = chatHistory
    .slice()
    .reverse()
    .findIndex((msg) => msg.sender !== USER_SENDER);
  if (lastAiMessageIndex === -1) {
    // No ai messages found, return an empty array
    return [];
  }

  const lastIndex = chatHistory.length - 1 - lastAiMessageIndex;
  const startIndex = Math.max(0, lastIndex - contextSize + 1);
  return chatHistory.slice(startIndex, lastIndex + 1);
};

export interface FormattedDateTime {
  fileName: string;
  display: string;
  epoch: number;
}

export const formatDateTime = (
  now: Date,
  timezone: "local" | "utc" = "local"
): FormattedDateTime => {
  const formattedDateTime = moment(now);

  if (timezone === "utc") {
    formattedDateTime.utc();
  }

  return {
    fileName: formattedDateTime.format("YYYYMMDD_HHmmss"),
    display: formattedDateTime.format("YYYY/MM/DD HH:mm:ss"),
    epoch: formattedDateTime.valueOf(),
  };
};

export function stringToFormattedDateTime(timestamp: string): FormattedDateTime {
  const date = moment(timestamp, "YYYY/MM/DD HH:mm:ss");
  if (!date.isValid()) {
    // If the string is not in the expected format, return current date/time
    return formatDateTime(new Date());
  }
  return {
    fileName: date.format("YYYYMMDD_HHmmss"),
    display: date.format("YYYY/MM/DD HH:mm:ss"),
    epoch: date.valueOf(),
  };
}

export async function getFileContent(file: TFile, vault: Vault): Promise<string | null> {
  if (file.extension != "md" && file.extension != "canvas") return null;
  return await vault.cachedRead(file);
}

export function sliceFileParagraphs(file: Paragraph, content: string): string {
  let slicedContent = content.trim();
  if (file.lineRange) {
    slicedContent = content
      .split("\n")
      .slice(
        isFinite(file.lineRange.start) ? file.lineRange.start - 1 : 0,
        isFinite(file.lineRange.end) ? file.lineRange.end : undefined
      )
      .join("\n");
  }
  return slicedContent;
}

export async function getFileParagraphs(file: Paragraph, vault: Vault): Promise<string | null> {
  if (file.extension != "md" && file.extension != "canvas") return null;
  const fileContent = await vault.cachedRead(file);
  return sliceFileParagraphs(file, fileContent);
}

export function getFileName(file: TFile): string {
  return file.basename;
}

export async function getAllNotesContent(vault: Vault): Promise<string> {
  let allContent = "";

  const markdownFiles = vault.getMarkdownFiles();

  for (const file of markdownFiles) {
    const fileContent = await vault.cachedRead(file);
    allContent += fileContent + " ";
  }

  return allContent;
}

export function areEmbeddingModelsSame(
  model1: string | undefined,
  model2: string | undefined
): boolean {
  if (!model1 || !model2) return false;
  // TODO: Hacks to handle different embedding model names for the same model. Need better handling.
  if (model1.includes(NOMIC_EMBED_TEXT) && model2.includes(NOMIC_EMBED_TEXT)) {
    return true;
  }
  if (
    (model1 === "small" && model2 === "cohereai") ||
    (model1 === "cohereai" && model2 === "small")
  ) {
    return true;
  }
  return model1 === model2;
}

// Basic prompts
export function sendNotesContentPrompt(notes: { name: string; content: string }[]): string {
  const formattedNotes = notes.map((note) => `## ${note.name}\n\n${note.content}`).join("\n\n");

  return (
    `Please read the notes below and be ready to answer questions about them. ` +
    `If there's no information about a certain topic, just say the note ` +
    `does not mention it. ` +
    `The content of the notes is between "/***/":\n\n/***/\n\n${formattedNotes}\n\n/***/\n\n` +
    `Please reply with the following word for word:` +
    `"OK I've read these notes. ` +
    `Feel free to ask related questions, such as 'give me a summary of these notes in bullet points', 'what key questions do these notes answer', etc. "\n`
  );
}

function getNoteTitleAndTags(noteWithTag: {
  name: string;
  content: string;
  tags?: string[];
}): string {
  return (
    `[[${noteWithTag.name}]]` +
    (noteWithTag.tags && noteWithTag.tags.length > 0 ? `\ntags: ${noteWithTag.tags.join(",")}` : "")
  );
}

function getChatContextStr(chatNoteContextPath: string, chatNoteContextTags: string[]): string {
  const pathStr = chatNoteContextPath ? `\nChat context by path: ${chatNoteContextPath}` : "";
  const tagsStr =
    chatNoteContextTags?.length > 0 ? `\nChat context by tags: ${chatNoteContextTags}` : "";
  return pathStr + tagsStr;
}

export function getSendChatContextNotesPrompt(
  notes: { name: string; content: string }[],
  chatNoteContextPath: string,
  chatNoteContextTags: string[]
): string {
  const noteTitles = notes.map((note) => getNoteTitleAndTags(note)).join("\n\n");
  return (
    `Please read the notes below and be ready to answer questions about them. ` +
    getChatContextStr(chatNoteContextPath, chatNoteContextTags) +
    `\n\n${noteTitles}`
  );
}

export function extractChatHistory(memoryVariables: MemoryVariables): [string, string][] {
  const chatHistory: [string, string][] = [];
  const { history } = memoryVariables;

  for (let i = 0; i < history.length; i += 2) {
    const userMessage = history[i]?.content || "";
    const aiMessage = history[i + 1]?.content || "";
    chatHistory.push([userMessage, aiMessage]);
  }

  return chatHistory;
}

export function extractNoteFiles(query: string, vault: Vault): TFile[] {
  // Use a regular expression to extract note titles and paths wrapped in [[]]
  const regex = /\[\[(.*?)\]\]/g;
  const matches = query.match(regex);
  const uniqueFiles = new Map<string, TFile>();

  if (matches) {
    matches.forEach((match) => {
      const inner = match.slice(2, -2);

      // First try to get file by full path
      const file = vault.getAbstractFileByPath(inner);

      if (file instanceof TFile) {
        // Found by path, use it directly
        uniqueFiles.set(file.path, file);
      } else {
        // Try to find by title
        const files = vault.getMarkdownFiles();
        const matchingFiles = files.filter((f) => f.basename === inner);

        if (matchingFiles.length > 0) {
          if (isNoteTitleUnique(inner, vault)) {
            // Only one file with this title, use it
            uniqueFiles.set(matchingFiles[0].path, matchingFiles[0]);
          } else {
            // Multiple files with same title - this shouldn't happen
            // as we should be using full paths for duplicate titles
            console.warn(
              `Found multiple files with title "${inner}". Expected a full path for duplicate titles.`
            );
          }
        }
      }
    });
  }

  return Array.from(uniqueFiles.values());
}

export function extractNoteParagraphs(query: string, vault: Vault): Paragraph[] {
  // Use a regular expression to extract note titles and paths wrapped in [[]]
  const regex = /\[\[(.*?)(#([0-9])*){1,2}\]\]/g;
  const matches = query.match(regex);
  const uniqueFiles = new Map<string, Paragraph>();

  if (matches) {
    matches.forEach((match) => {
      const referName = match.slice(2, -2);

      // support format [[filename#0#100]]
      const inner = referName.split("#")[0];

      // First try to get file by full path
      const file = vault.getAbstractFileByPath(referName);

      if (file instanceof TFile) {
        // Found by path, use it directly
        uniqueFiles.set(referName, new Paragraph(referName, file));
      } else {
        // Try to find by title
        const files = vault.getMarkdownFiles();
        const matchingFiles = files
          .filter((f) => f.basename === inner)
          .map((f) => new Paragraph(referName, f));

        // support format [[filename#0#100]]
        const rangeStart = referName.split("#")[1];
        const rangeEnd = referName.split("#")[2];
        if (rangeStart != "" || rangeEnd != "") {
          matchingFiles.forEach((f) => {
            f.lineRange = {
              start: parseInt(rangeStart),
              end: parseInt(rangeEnd),
            };
          });
        }

        if (matchingFiles.length > 0) {
          if (isNoteTitleUnique(inner, vault)) {
            // Only one file with this title, use it
            uniqueFiles.set(referName, new Paragraph(referName, matchingFiles[0]));
          } else {
            // Multiple files with same title - this shouldn't happen
            // as we should be using full paths for duplicate titles
            console.warn(
              `Found multiple files with title "${inner}". Expected a full path for duplicate titles.`
            );
          }
        }
      }
    });
  }

  return Array.from(uniqueFiles.values());
}

// Helper function to check if a note title is unique in the vault
export function isNoteTitleUnique(title: string, vault: Vault): boolean {
  const files = vault.getMarkdownFiles();
  return files.filter((f) => f.basename === title).length === 1;
}

// Helper function to determine if we should show the full path for a file
export function shouldShowPath(file: TFile): boolean {
  return (file as any).needsPathDisplay === true;
}

/**
 * Process the variable name to generate a note path if it's enclosed in double brackets,
 * otherwise return the variable name as is.
 *
 * @param {string} variableName - The name of the variable to process
 * @return {string} The processed note path or the variable name itself
 */
export function processVariableNameForNotePath(variableName: string): string {
  variableName = variableName.trim();
  // Check if the variable name is enclosed in double brackets indicating it's a note
  if (variableName.startsWith("[[") && variableName.endsWith("]]")) {
    // It's a note, so we remove the brackets and append '.md'
    return `${variableName.slice(2, -2).trim()}.md`;
  }
  // It's a path, so we just return it as is
  return variableName;
}

export function extractUniqueTitlesFromDocs(docs: Document[]): string[] {
  const titlesSet = new Set<string>();
  docs.forEach((doc) => {
    if (doc.metadata?.title) {
      titlesSet.add(doc.metadata?.title);
    }
  });

  return Array.from(titlesSet);
}

export function extractJsonFromCodeBlock(content: string): any {
  const codeBlockMatch = content.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
  const jsonContent = codeBlockMatch ? codeBlockMatch[1].trim() : content.trim();
  return JSON.parse(jsonContent);
}

const YOUTUBE_URL_REGEX =
  /(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:watch\?v=|shorts\/)|youtu\.be\/)([^\s&]+)/;

export function isYoutubeUrl(url: string): boolean {
  return YOUTUBE_URL_REGEX.test(url);
}

export function extractYoutubeUrl(text: string): string | null {
  const match = text.match(YOUTUBE_URL_REGEX);
  return match ? match[0] : null;
}

/** Proxy function to use in place of fetch() to bypass CORS restrictions.
 * It currently doesn't support streaming until this is implemented
 * https://forum.obsidian.md/t/support-streaming-the-request-and-requesturl-response-body/87381 */
export async function safeFetch(url: string, options: RequestInit = {}): Promise<Response> {
  // Initialize headers if not provided
  const headers = options.headers ? { ...options.headers } : {};

  // Remove content-length if it exists
  delete (headers as Record<string, string>)["content-length"];

  if (typeof options.body === "string") {
    const newBody = JSON.parse(options.body ?? {});
    delete newBody["frequency_penalty"];
    options.body = JSON.stringify(newBody);
  }

  logInfo("==== safeFetch method request ====");

  const method = options.method?.toUpperCase() || "POST";
  const methodsWithBody = ["POST", "PUT", "PATCH"];

  const response = await requestUrl({
    url,
    contentType: "application/json",
    headers: headers as Record<string, string>,
    method: method,
    ...(methodsWithBody.includes(method) && { body: options.body?.toString() }),
    throw: false, // Don't throw so we can get the response body
  });

  // Check if response is error status
  if (response.status >= 400) {
    let errorJson;
    try {
      errorJson = typeof response.json === "string" ? JSON.parse(response.json) : response.json;
    } catch {
      try {
        errorJson = typeof response.text === "string" ? JSON.parse(response.text) : response.text;
      } catch {
        errorJson = null;
      }
    }

    // Create error with proper structure
    const error = new Error(ERROR_MESSAGES.REQUEST_FAILED(response.status)) as APIError;
    error.json = errorJson;

    // Handle nested error structure
    if (
      errorJson?.detail?.reason === "Invalid license key" ||
      errorJson?.reason === "Invalid license key"
    ) {
      error.message = "Invalid license key";
    } else if (errorJson?.detail?.message || errorJson?.message) {
      const message = errorJson?.detail?.message || errorJson?.message;
      const reason = errorJson?.detail?.reason || errorJson?.reason;
      error.message = reason ? `${message}: ${reason}` : message;
    } else if (errorJson?.detail) {
      error.message = JSON.stringify(errorJson.detail);
    }

    throw error;
  }

  return {
    ok: response.status >= 200 && response.status < 300,
    status: response.status,
    statusText: response.status.toString(),
    headers: new Headers(response.headers),
    url: url,
    type: "basic" as ResponseType,
    redirected: false,
    bytes: () => Promise.resolve(new Uint8Array(0)),
    body: createReadableStreamFromString(response.text),
    bodyUsed: true,
    json: () => response.json,
    text: async () => response.text,
    arrayBuffer: async () => {
      if (response.arrayBuffer) {
        return response.arrayBuffer;
      }
      const base64 = response.text.replace(/^data:.*;base64,/, "");
      const binaryString = atob(base64);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      return bytes.buffer;
    },
    blob: () => {
      throw new Error("not implemented");
    },
    formData: () => {
      throw new Error("not implemented");
    },
    clone: () => {
      throw new Error("not implemented");
    },
  };
}

function createReadableStreamFromString(input: string) {
  return new ReadableStream({
    start(controller) {
      // Convert the input string to a Uint8Array
      const encoder = new TextEncoder();
      const uint8Array = encoder.encode(input);

      // Push the data to the stream
      controller.enqueue(uint8Array);

      // Close the stream
      controller.close();
    },
  });
}

export function err2String(err: any, stack = false) {
  // maybe to be improved
  return err instanceof Error
    ? err.message +
        "\n" +
        `${err?.cause ? "more message: " + (err.cause as Error).message : ""}` +
        "\n" +
        `${stack ? err.stack : ""}`
    : JSON.stringify(err);
}

export function omit<T extends Record<string, any>, K extends keyof T>(
  obj: T,
  keys: K[]
): Omit<T, K> {
  const result = { ...obj };
  keys.forEach((key) => {
    delete result[key];
  });
  return result;
}

export function findCustomModel(modelKey: string, activeModels: CustomModel[]): CustomModel {
  const [modelName, provider] = modelKey.split("|");
  const model = activeModels.find((m) => m.name === modelName && m.provider === provider);
  if (!model) {
    throw new Error(`No model configuration found for: ${modelKey}`);
  }
  return model;
}

export function getProviderInfo(provider: string): ProviderMetadata {
  const info = ProviderInfo[provider as Provider];
  return {
    ...info,
    label: info.label || provider,
  };
}

export function getProviderLabel(provider: string, model?: CustomModel): string {
  const baseLabel = ProviderInfo[provider as Provider]?.label || provider;
  return baseLabel + (model?.believerExclusive && baseLabel === "Copilot Plus" ? "(Believer)" : "");
}

export function getProviderHost(provider: string): string {
  return ProviderInfo[provider as Provider]?.host || "";
}

export function getProviderKeyManagementURL(provider: string): string {
  return ProviderInfo[provider as Provider]?.keyManagementURL || "";
}

export async function insertIntoEditor(message: string, replace: boolean = false) {
  let leaf = app.workspace.getMostRecentLeaf();
  if (!leaf) {
    new Notice("No active leaf found.");
    return;
  }

  if (!(leaf.view instanceof MarkdownView)) {
    leaf = app.workspace.getLeaf(false);
    await leaf.setViewState({ type: "markdown", state: leaf.view.getState() });
  }

  if (!(leaf.view instanceof MarkdownView)) {
    new Notice("Failed to open a markdown view.");
    return;
  }

  const editor = leaf.view.editor;
  const cursorFrom = editor.getCursor("from");
  const cursorTo = editor.getCursor("to");
  if (replace) {
    editor.replaceRange(message, cursorFrom, cursorTo);
  } else {
    editor.replaceRange(message, cursorTo);
  }
  new Notice("Message inserted into the active note.");
}

export function debounce<T extends (...args: any[]) => void>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout;
  return (...args: Parameters<T>) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
}

/**
 * Compare two semantic version strings.
 * @returns true if latest version is newer than current version
 */
export function isNewerVersion(latest: string, current: string): boolean {
  const latestParts = latest.split(".").map(Number);
  const currentParts = current.split(".").map(Number);

  for (let i = 0; i < 3; i++) {
    if (latestParts[i] > currentParts[i]) return true;
    if (latestParts[i] < currentParts[i]) return false;
  }
  return false;
}

/**
 * Check for latest version from GitHub releases.
 * @returns latest version string or error message
 */
export async function checkLatestVersion(): Promise<{
  version: string | null;
  error: string | null;
}> {
  try {
    const response = await requestUrl({
      url: "https://api.github.com/repos/ai-learning-assistant-dev/obsidian-copilot/releases/latest",
      // url: "https://api.github.com/repos/logancyang/obsidian-copilot/releases/latest",
      method: "GET",
    });
    const version = response.json.tag_name.replace("v", "");
    return { version, error: null };
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : "Failed to check for updates";
    return { version: null, error: errorMessage };
  }
}

export function isOSeriesModel(model: BaseChatModel | string): boolean {
  if (typeof model === "string") {
    return model.startsWith("o1") || model.startsWith("o3") || model.startsWith("o4");
  }

  // For BaseChatModel instances
  const modelName = (model as any).modelName || (model as any).model || "";
  return modelName.startsWith("o1") || modelName.startsWith("o3") || modelName.startsWith("o4");
}

export function getMessageRole(
  model: BaseChatModel | string,
  defaultRole: "system" | "human" = "system"
): "system" | "human" {
  return isOSeriesModel(model) ? "human" : defaultRole;
}

export function getNeedSetKeyProvider() {
  // List of providers to exclude
  const excludeProviders: Provider[] = [
    ChatModelProviders.OPENAI_FORMAT,
    ChatModelProviders.OLLAMA,
    ChatModelProviders.LM_STUDIO,
    ChatModelProviders.AZURE_OPENAI,
    EmbeddingModelProviders.COPILOT_PLUS,
    EmbeddingModelProviders.COPILOT_PLUS_JINA,
  ];

  return Object.entries(ProviderInfo)
    .filter(([key]) => !excludeProviders.includes(key as Provider))
    .map(([key]) => key as Provider);
}

export function checkModelApiKey(
  model: CustomModel,
  settings: Readonly<CopilotSettings>
): {
  hasApiKey: boolean;
  errorNotice?: string;
} {
  const needSetKeyPath = !!getNeedSetKeyProvider().find((provider) => provider === model.provider);
  const providerKeyName = ProviderSettingsKeyMap[model.provider as SettingKeyProviders];
  const hasNoApiKey = !model.apiKey && !settings[providerKeyName];

  if (needSetKeyPath && hasNoApiKey) {
    const notice =
      `Please configure API Key for ${model.name} in settings first.` +
      "\nPath: Settings > copilot plugin > Basic Tab > Set Keys";
    return {
      hasApiKey: false,
      errorNotice: notice,
    };
  }

  return {
    hasApiKey: true,
  };
}

/**
 * Removes any <think> tags and their content from the text.
 * This is used to clean model outputs before using them for RAG.
 * @param text - The text to remove think tags from
 * @returns The text with think tags removed
 */
export function removeThinkTags(text: string): string {
  return text.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
}
/* Utility functions for Obsidian Transcript */
import { App, FileSystemAdapter, getBlobArrayBuffer } from "obsidian";
import {
  WhisperASRResponse,
  WhisperASRSegment,
  WhisperASRWordTimestamp,
} from "./asr/types/whisper-asr";

export const randomString = (length: number) =>
  Array(length + 1)
    .join((Math.random().toString(36) + "00000000000000000").slice(2, 18))
    .slice(0, length);
export const getAllLinesFromFile = (cache: string) => cache.split(/\r?\n/);
export const combineFileLines = (lines: string[]) => lines.join("\n");

export function getVaultAbsolutePath(app: App) {
  // Original code was copied 2021-08-22 from https://github.com/phibr0/obsidian-open-with/blob/84f0e25ba8e8355ff83b22f4050adde4cc6763ea/main.ts#L66-L67
  // But the code has been rewritten 2021-08-27 as per https://github.com/obsidianmd/obsidian-releases/pull/433#issuecomment-906087095
  const adapter = app.vault.adapter;
  if (adapter instanceof FileSystemAdapter) {
    return adapter.getBasePath();
  }
  return null;
}

type PayloadAndBoundary = [ArrayBuffer, string];

export type PayloadData = { [key: string]: string | Blob | ArrayBuffer };

export async function payloadGenerator(payload_data: PayloadData): Promise<PayloadAndBoundary> {
  // This function is a workaround to current Obsidian API limitations: requestURL only supports string data or an unnamed blob, not key-value formdata
  // Essentially what we're doing here is constructing a multipart/form-data payload manually as a string and then passing it to requestURL

  const boundary_string = `Boundary${randomString(16)}`;
  const boundary = `------${boundary_string}`;
  const chunks: Uint8Array | ArrayBuffer[] = [];
  // NOTE Could this cause corrupt files via synchronous operations? So far no issues, but it's worth keeping an eye on
  for (const [key, value] of Object.entries(payload_data)) {
    // Start of a new part
    chunks.push(new TextEncoder().encode(`${boundary}\r\n`));

    // If the value is a string, then it's a key-value pair
    if (typeof value === "string") {
      chunks.push(
        new TextEncoder().encode(`Content-Disposition: form-data; name="${key}"\r\n\r\n`)
      );
      chunks.push(new TextEncoder().encode(`${value}\r\n`));
    } else if (value instanceof Blob) {
      chunks.push(
        new TextEncoder().encode(
          `Content-Disposition: form-data; name="${key}"; filename="blob"\r\nContent-Type: "application/octet-stream"\r\n\r\n`
        )
      );
      chunks.push(await getBlobArrayBuffer(value));
      chunks.push(new TextEncoder().encode("\r\n"));
    } else {
      chunks.push(new Uint8Array(await new Response(value).arrayBuffer()));
      chunks.push(new TextEncoder().encode("\r\n"));
    }
  }
  await Promise.all(chunks);
  chunks.push(new TextEncoder().encode(`${boundary}--\r\n`));
  return [await new Blob(chunks).arrayBuffer(), boundary_string];
}

/**
 * Preprocesses the raw response from Whisper ASR into a more structured and typed format.
 * @param rawResponse - The raw response object from Whisper ASR.
 */
export function preprocessWhisperASRResponse(rawResponse: any): WhisperASRResponse {
  return {
    language: rawResponse.language,
    text: rawResponse.text,
    segments: rawResponse.segments.map((segment: any) => {
      const baseSegment = {
        segmentIndex: segment[0],
        seek: segment[1],
        start: segment[2],
        end: segment[3],
        text: segment[4],
        tokens: segment[5],
        temperature: segment[6],
        avg_logprob: segment[7],
        compression_ratio: segment[8],
        no_speech_prob: segment[9],
        words: null,
      } as WhisperASRSegment;
      if (segment[10] !== null) {
        // easier to read than a ternary-destructured assignment
        baseSegment.words = segment[10].map(
          (wordTimestamp: unknown[]) =>
            ({
              start: wordTimestamp[0],
              end: wordTimestamp[1],
              word: wordTimestamp[2],
              probability: wordTimestamp[3],
            }) as WhisperASRWordTimestamp
        );
      }
      return baseSegment;
    }),
  };
}

export function getBaseFileName(filePath: string) {
  // Extract the file name including extension
  const fileName = filePath.substring(filePath.lastIndexOf("/") + 1);

  // Remove the extension from the file name
  const baseFileName = fileName.substring(0, fileName.lastIndexOf("."));

  return baseFileName;
}
