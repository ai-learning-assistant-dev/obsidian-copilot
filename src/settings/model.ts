import { CustomModel } from "@/aiParams";
import { atom, createStore, useAtomValue } from "jotai";
import { v4 as uuidv4 } from "uuid";

import { type ChainType } from "@/chainFactory";
import {
  BUILTIN_CHAT_MODELS,
  BUILTIN_EMBEDDING_MODELS,
  DEFAULT_OPEN_AREA,
  DEFAULT_SETTINGS,
  DEFAULT_SYSTEM_PROMPT,
  EmbeddingModelProviders,
} from "@/constants";

export interface InlineEditCommandSettings {
  /**
   * The name of the command. The name will be turned into id by replacing
   * spaces with underscores.
   */
  name: string;

  /**
   * The model key of the command. If not provided, the current chat model will
   * be used.
   */
  modelKey?: string;

  /**
   * The prompt of the command.
   */
  prompt: string;

  /**
   * Whether to show the command in the context menu.
   */
  showInContextMenu: boolean;
}

export interface CopilotSettings {
  Asr_apiKey: string;
  Asr_apiUrl: string;
  Asr_debugMode: boolean;
  Asr_encode: boolean;
  Asr_prompt: string;
  Asr_language: string;
  Asr_lineSpacing: string;
  Asr_createNewFileAfterRecording: boolean;
  Asr_createNewFileAfterRecordingPath: string;
  Asr_saveAudioFile: boolean;
  Asr_saveAudioFilePath: string;
  Asr_useLocalService: boolean;
  Asr_localServiceUrl: string;
  Asr_translate: boolean;
  Asr_transcriptionEngine: string;
  Asr_timestamps: boolean;
  Asr_timestampFormat: string;
  Asr_timestampInterval: string; // easier to store as a string and convert to number when needed
  Asr_wordTimestamps: boolean;
  Asr_vadFilter: boolean;
  userId: string;
  plusLicenseKey: string;
  openAIApiKey: string;
  openAIOrgId: string;
  huggingfaceApiKey: string;
  cohereApiKey: string;
  anthropicApiKey: string;
  azureOpenAIApiKey: string;
  azureOpenAIApiInstanceName: string;
  azureOpenAIApiDeploymentName: string;
  azureOpenAIApiVersion: string;
  azureOpenAIApiEmbeddingDeploymentName: string;
  googleApiKey: string;
  openRouterAiApiKey: string;
  xaiApiKey: string;
  mistralApiKey: string;
  deepseekApiKey: string;
  defaultChainType: ChainType;
  defaultModelKey: string;
  embeddingModelKey: string;
  temperature: number;
  maxTokens: number;
  contextTurns: number;
  lastDismissedVersion: string | null;
  // Do not use this directly, use getSystemPrompt() instead
  userSystemPrompt: string;
  openAIProxyBaseUrl: string;
  openAIEmbeddingProxyBaseUrl: string;
  stream: boolean;
  defaultSaveFolder: string;
  defaultConversationTag: string;
  autosaveChat: boolean;
  customPromptsFolder: string;
  indexVaultToVectorStore: string;
  chatNoteContextPath: string;
  chatNoteContextTags: string[];
  enableIndexSync: boolean;
  debug: boolean;
  enableEncryption: boolean;
  maxSourceChunks: number;
  qaExclusions: string;
  qaInclusions: string;
  groqApiKey: string;
  activeModels: Array<CustomModel>;
  activeEmbeddingModels: Array<CustomModel>;
  promptUsageTimestamps: Record<string, number>;
  embeddingRequestsPerMin: number;
  embeddingBatchSize: number;
  defaultOpenArea: DEFAULT_OPEN_AREA;
  disableIndexOnMobile: boolean;
  showSuggestedPrompts: boolean;
  showRelevantNotes: boolean;
  numPartitions: number;
  defaultConversationNoteName: string;
  // undefined means never checked
  isPlusUser: boolean | undefined;
  inlineEditCommands: InlineEditCommandSettings[] | undefined;
  passMarkdownImages: boolean;
  enableCustomPromptTemplating: boolean;
  systemPrompts?: {
    default: string;
    custom?: Record<string, string>;
    characterPresets?: CharacterPreset[];
    presets?: CharacterPresetItem[]; // 新增presets数组
    activePresetId?: string | null;
    activeTraits?: CharacterTrait;
    checkedItems?: CheckedItems;
    selectedValues?: SelectedValues;
    traitOrder?: string[]; // 新增特征顺序数组
  };
  promptEnhancements?: {
    autoFollowUp?: {
      enabled: boolean;
      prompt: string;
    };
    autoSpeech?: {
      enabled?: boolean;
      prompt?: string;
      useOralPrompt?: boolean;
    };
    appendDefaultPrompt?: boolean; // 新增是否拼接默认系统提示词
  };
}

// 用于人设列表
interface CharacterPresetItem {
  id: string;
  name: string;
  prompt: string;
  isActive: boolean;
}

interface CheckedItems {
  [key: string]: boolean | undefined; // key为选项ID，value表示是否被勾选
}

interface SelectedValues {
  // 新增类型定义
  [key: string]: string | undefined;
}

// 新增类型定义
interface CharacterTrait {
  [key: string]: string;
}

interface CharacterPreset {
  id: string;
  name: string;
  prompt: string;
  traits: CharacterTrait;
}

export const settingsStore = createStore();
export const settingsAtom = atom<CopilotSettings>(DEFAULT_SETTINGS);

/**
 * Sets the settings in the atom.
 */
export function setSettings(settings: Partial<CopilotSettings>) {
  const newSettings = mergeAllActiveModelsWithCoreModels({ ...getSettings(), ...settings });
  settingsStore.set(settingsAtom, newSettings);
}

/**
 * Sets a single setting in the atom.
 */
export function updateSetting<K extends keyof CopilotSettings>(key: K, value: CopilotSettings[K]) {
  const settings = getSettings();
  setSettings({ ...settings, [key]: value });
}

/**
 * Gets the settings from the atom. Use this if you don't need to subscribe to
 * changes.
 */
export function getSettings(): Readonly<CopilotSettings> {
  return settingsStore.get(settingsAtom);
}

/**
 * Resets the settings to the default values.
 */
export function resetSettings(): void {
  const defaultSettingsWithBuiltIns = {
    ...DEFAULT_SETTINGS,
    activeModels: BUILTIN_CHAT_MODELS.map((model) => ({ ...model, enabled: true })),
    activeEmbeddingModels: BUILTIN_EMBEDDING_MODELS.map((model) => ({ ...model, enabled: true })),
  };
  setSettings(defaultSettingsWithBuiltIns);
}

/**
 * Subscribes to changes in the settings atom.
 */
export function subscribeToSettingsChange(
  callback: (prev: CopilotSettings, next: CopilotSettings) => void
): () => void {
  let previousValue = getSettings();

  return settingsStore.sub(settingsAtom, () => {
    const currentValue = getSettings();
    callback(previousValue, currentValue);
    previousValue = currentValue;
  });
}

/**
 * Hook to get the settings value from the atom.
 */
export function useSettingsValue(): Readonly<CopilotSettings> {
  return useAtomValue(settingsAtom, {
    store: settingsStore,
  });
}

/**
 * Sanitizes the settings to ensure they are valid.
 * Note: This will be better handled by Zod in the future.
 */
export function sanitizeSettings(settings: CopilotSettings): CopilotSettings {
  // If settings is null/undefined, use DEFAULT_SETTINGS
  const settingsToSanitize = settings || DEFAULT_SETTINGS;

  if (!settingsToSanitize.userId) {
    settingsToSanitize.userId = uuidv4();
  }

  // fix: Maintain consistency between EmbeddingModelProviders.AZURE_OPENAI and ChatModelProviders.AZURE_OPENAI,
  // where it was 'azure_openai' before EmbeddingModelProviders.AZURE_OPENAI.
  if (!settingsToSanitize.activeEmbeddingModels) {
    settingsToSanitize.activeEmbeddingModels = BUILTIN_EMBEDDING_MODELS.map((model) => ({
      ...model,
      enabled: true,
    }));
  } else {
    settingsToSanitize.activeEmbeddingModels = settingsToSanitize.activeEmbeddingModels.map((m) => {
      return {
        ...m,
        provider: m.provider === "azure_openai" ? EmbeddingModelProviders.AZURE_OPENAI : m.provider,
      };
    });
  }

  const sanitizedSettings: CopilotSettings = { ...settingsToSanitize };

  // Stuff in settings are string even when the interface has number type!
  const temperature = Number(settingsToSanitize.temperature);
  sanitizedSettings.temperature = isNaN(temperature) ? DEFAULT_SETTINGS.temperature : temperature;

  const maxTokens = Number(settingsToSanitize.maxTokens);
  sanitizedSettings.maxTokens = isNaN(maxTokens) ? DEFAULT_SETTINGS.maxTokens : maxTokens;

  const contextTurns = Number(settingsToSanitize.contextTurns);
  sanitizedSettings.contextTurns = isNaN(contextTurns)
    ? DEFAULT_SETTINGS.contextTurns
    : contextTurns;

  const embeddingRequestsPerMin = Number(settingsToSanitize.embeddingRequestsPerMin);
  sanitizedSettings.embeddingRequestsPerMin = isNaN(embeddingRequestsPerMin)
    ? DEFAULT_SETTINGS.embeddingRequestsPerMin
    : embeddingRequestsPerMin;

  const embeddingBatchSize = Number(settingsToSanitize.embeddingBatchSize);
  sanitizedSettings.embeddingBatchSize = isNaN(embeddingBatchSize)
    ? DEFAULT_SETTINGS.embeddingBatchSize
    : embeddingBatchSize;

  // Ensure passMarkdownImages has a default value
  if (typeof sanitizedSettings.passMarkdownImages !== "boolean") {
    sanitizedSettings.passMarkdownImages = DEFAULT_SETTINGS.passMarkdownImages;
  }

  // Ensure enableCustomPromptTemplating has a default value
  if (typeof sanitizedSettings.enableCustomPromptTemplating !== "boolean") {
    sanitizedSettings.enableCustomPromptTemplating = DEFAULT_SETTINGS.enableCustomPromptTemplating;
  }

  return sanitizedSettings;
}

export function getSystemPrompt(): string {
  // const userPrompt = getSettings().userSystemPrompt;
  // return userPrompt ? `${DEFAULT_SYSTEM_PROMPT}\n\n${userPrompt}` : DEFAULT_SYSTEM_PROMPT;

  const settings = getSettings();
  const userPrompt = settings.userSystemPrompt;

  let basePrompt: string;

  // 根据是否拼接默认提示词进行判断
  if (settings.promptEnhancements?.appendDefaultPrompt === false) {
    // 不拼接默认提示词
    basePrompt = userPrompt || "";
  } else {
    // 拼接默认提示词
    basePrompt = userPrompt ? `${DEFAULT_SYSTEM_PROMPT}\n\n${userPrompt}` : DEFAULT_SYSTEM_PROMPT;
  }

  // 如果自动衍生问题功能开启且设置了提示词
  if (
    settings.promptEnhancements?.autoFollowUp?.enabled &&
    settings.promptEnhancements.autoFollowUp.prompt
  ) {
    basePrompt += `\n\n${settings.promptEnhancements.autoFollowUp.prompt}`;
  }

  // 如果自动语音播放功能开启且设置了提示词
  if (
    settings.promptEnhancements?.autoSpeech?.useOralPrompt &&
    settings.promptEnhancements.autoSpeech.prompt
  ) {
    basePrompt += `\n\n${settings.promptEnhancements.autoSpeech.prompt}`;
  }

  return basePrompt;
}

function mergeAllActiveModelsWithCoreModels(settings: CopilotSettings): CopilotSettings {
  settings.activeModels = mergeActiveModels(settings.activeModels, BUILTIN_CHAT_MODELS);
  settings.activeEmbeddingModels = mergeActiveModels(
    settings.activeEmbeddingModels,
    BUILTIN_EMBEDDING_MODELS
  );
  return settings;
}

/**
 * Get a unique model key from a CustomModel instance
 * Format: modelName|provider
 */
export function getModelKeyFromModel(model: CustomModel): string {
  return `${model.name}|${model.provider}`;
}

function mergeActiveModels(
  existingActiveModels: CustomModel[],
  builtInModels: CustomModel[]
): CustomModel[] {
  const modelMap = new Map<string, CustomModel>();

  // Add core models to the map first
  builtInModels
    .filter((model) => model.core)
    .forEach((model) => {
      modelMap.set(getModelKeyFromModel(model), { ...model });
    });

  // Add or update existing models in the map
  existingActiveModels.forEach((model) => {
    const key = getModelKeyFromModel(model);
    const existingModel = modelMap.get(key);
    if (existingModel) {
      // If it's a built-in model, preserve all built-in properties
      const builtInModel = builtInModels.find(
        (m) => m.name === model.name && m.provider === model.provider
      );
      if (builtInModel) {
        modelMap.set(key, {
          ...builtInModel,
          ...model,
          isBuiltIn: true,
          believerExclusive: builtInModel.believerExclusive,
        });
      } else {
        modelMap.set(key, {
          ...model,
          isBuiltIn: existingModel.isBuiltIn,
        });
      }
    } else {
      modelMap.set(key, model);
    }
  });

  return Array.from(modelMap.values());
}
