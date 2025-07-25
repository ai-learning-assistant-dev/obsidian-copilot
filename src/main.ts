import { BrevilabsClient } from "@/LLMProviders/brevilabsClient";
import ChainManager from "@/LLMProviders/chainManager";
import { CustomModel } from "@/aiParams";
import { parseChatContent, updateChatMemory } from "@/chatUtils";
import { registerCommands } from "@/commands";
import CopilotView from "@/components/CopilotView";
import { LoadChatHistoryModal } from "@/components/modals/LoadChatHistoryModal";
import { CHAT_VIEWTYPE, DEFAULT_OPEN_AREA, EVENT_NAMES } from "@/constants";
import { registerContextMenu } from "@/contextMenu";
import { encryptAllKeys } from "@/encryptionService";
import { checkIsPlusUser } from "@/plusUtils";
import { HybridRetriever } from "@/search/hybridRetriever";
import VectorStoreManager from "@/search/vectorStoreManager";
import { CopilotSettingTab } from "@/settings/SettingsPage";
import {
  getModelKeyFromModel,
  getSettings,
  sanitizeSettings,
  setSettings,
  subscribeToSettingsChange,
} from "@/settings/model";
import SharedState from "@/sharedState";
import { FileParserManager } from "@/tools/FileParserManager";
import {
  App,
  Editor,
  FuzzySuggestModal,
  MarkdownView,
  Menu,
  Modal,
  Notice,
  Platform,
  Plugin,
  PluginManifest,
  TFile,
  TFolder,
  WorkspaceLeaf,
} from "obsidian";
import { IntentAnalyzer } from "./LLMProviders/intentAnalyzer";

import { ChildProcess } from "child_process";
import { TranscriptionEngine } from "./asr/transcribe";
import { StatusBarReadwise } from "./asr/status";
import { FileLink } from "./asr/fileLink";
import { Timer } from "./asr/Timer";
import { Controls } from "./asr/Controls";
import { AudioHandler } from "./asr/AudioHandler";
import { AsrSettings } from "./asr/AsrSettingsTab";
import { NativeAudioRecorder } from "./asr/AudioRecorder";
import { RecordingStatus, StatusBarRecord } from "./asr/StatusBar";

export default class CopilotPlugin extends Plugin {
  // A chat history that stores the messages sent and received
  // Only reset when the user explicitly clicks "New Chat"
  sharedState: SharedState;
  chainManager: ChainManager;
  brevilabsClient: BrevilabsClient;
  userMessageHistory: string[] = [];
  vectorStoreManager: VectorStoreManager;
  fileParserManager: FileParserManager;
  settingsUnsubscriber?: () => void;

  asrSettings: AsrSettings;
  timer: Timer;
  recorder: NativeAudioRecorder;
  audioHandler: AudioHandler;
  controls: Controls | null = null;
  statusBarRecord: StatusBarRecord;
  statusBarReadwise: StatusBarReadwise;

  public static plugin: Plugin;
  public static children: Array<ChildProcess> = [];
  public transcriptionEngine: TranscriptionEngine;

  private ongoingTranscriptionTasks: Array<{
    task: Promise<void>;
    abortController: AbortController;
  }> = [];
  public static transcribeFileExtensions: string[] = [
    "mp3",
    "wav",
    "webm",
    "ogg",
    "flac",
    "m4a",
    "aac",
    "amr",
    "opus",
    "aiff",
    "m3gp",
    "mp4",
    "m4v",
    "mov",
    "avi",
    "wmv",
    "flv",
    "mpeg",
    "mpg",
    "mkv",
  ];

  public setChatInput: ((input: string) => void) | undefined;

  constructor(app: App, manifest: PluginManifest) {
    super(app, manifest);
    // Additional initialization if needed
  }

  public getTranscribeableFiles = async (file: TFile) => {
    // Get all linked files in the markdown file
    const filesLinked = Object.keys(this.app.metadataCache.resolvedLinks[file.path]);

    // Now that we have all the files linked in the markdown file, we need to filter them by the file extensions we want to transcribe
    const filesToTranscribe: TFile[] = [];
    for (const linkedFilePath of filesLinked) {
      const linkedFileExtension = linkedFilePath.split(".").pop();
      if (
        linkedFileExtension === undefined ||
        !CopilotPlugin.transcribeFileExtensions.includes(linkedFileExtension.toLowerCase())
      ) {
        if (this.asrSettings.Asr_debugMode)
          console.log(
            "Skipping " +
              linkedFilePath +
              " because the file extension is not in the list of transcribeable file extensions"
          );
        continue;
      }

      // We now know that the file extension is in the list of transcribeable file extensions
      const linkedFile = this.app.vault.getAbstractFileByPath(linkedFilePath);

      // Validate that we are dealing with a file and add it to the list of verified files to transcribe
      if (linkedFile instanceof TFile) filesToTranscribe.push(linkedFile);
      else {
        if (this.asrSettings.Asr_debugMode) console.log("Could not find file " + linkedFilePath);
        continue;
      }
    }
    return filesToTranscribe;
  };

  public async transcribeAndWrite(
    parent_file: TFile,
    file: TFile,
    abortController: AbortController | null
  ) {
    try {
      if (this.asrSettings.Asr_debugMode) console.log("Transcribing " + file.path);

      const transcription = await this.transcriptionEngine.getTranscription(file);

      let fileText = await this.app.vault.read(parent_file);
      const fileLinkString = this.app.metadataCache.fileToLinktext(file, parent_file.path);
      const fileLinkStringTagged = `[[${fileLinkString}]]`;

      const startReplacementIndex =
        fileText.indexOf(fileLinkStringTagged) + fileLinkStringTagged.length;

      if (this.asrSettings.Asr_lineSpacing === "single") {
        fileText = [
          fileText.slice(0, startReplacementIndex),
          `${transcription}`,
          fileText.slice(startReplacementIndex),
        ].join(" ");
      } else {
        fileText = [
          fileText.slice(0, startReplacementIndex),
          `\n${transcription}`,
          fileText.slice(startReplacementIndex),
        ].join("");
      }

      //check if abortion signal is aborted

      if (abortController?.signal?.aborted) {
        new Notice(`Transcription of ${file.name} cancelled!`, 5 * 1000);
        return;
      }

      await this.app.vault.modify(parent_file, fileText);
    } catch (error) {
      // First check if 402 is in the error message, if so alert the user that they need to pay

      if (error?.message?.includes("402")) {
        new Notice(
          "You have exceeded the free tier.\nPlease upgrade to a paid plan at swiftink.io/pricing to continue transcribing files.\nThanks for using Swiftink!",
          10 * 1000
        );
      } else {
        if (this.asrSettings.Asr_debugMode) console.log(error);
        new Notice(`Error transcribing file: ${error}`, 10 * 1000);
      }
    } finally {
      // Clear the AbortController after completion or cancellation
      abortController = null;
    }
  }

  async onload(): Promise<void> {
    await this.loadSettings();
    this.settingsUnsubscriber = subscribeToSettingsChange(async (prev, next) => {
      if (next.enableEncryption) {
        await this.saveData(await encryptAllKeys(next));
      } else {
        await this.saveData(next);
      }
      registerCommands(this, prev, next);
    });
    this.addSettingTab(new CopilotSettingTab(this.app, this));
    // Always have one instance of sharedState and chainManager in the plugin
    this.sharedState = new SharedState();

    this.vectorStoreManager = VectorStoreManager.getInstance();

    // Initialize BrevilabsClient
    this.brevilabsClient = BrevilabsClient.getInstance();
    this.brevilabsClient.setPluginVersion(this.manifest.version);
    checkIsPlusUser();

    this.chainManager = new ChainManager(this.app, this.vectorStoreManager);

    // Initialize FileParserManager early with other core services
    this.fileParserManager = new FileParserManager(this.brevilabsClient, this.app.vault);

    this.registerView(CHAT_VIEWTYPE, (leaf: WorkspaceLeaf) => new CopilotView(leaf, this));

    this.initActiveLeafChangeHandler();

    this.addRibbonIcon("message-square", "Open Copilot Chat", (evt: MouseEvent) => {
      this.activateView();
    });

    registerCommands(this, undefined, getSettings());

    IntentAnalyzer.initTools(this.app.vault);

    this.registerEvent(
      this.app.workspace.on("editor-menu", (menu: Menu, editor: Editor) => {
        const selectedText = editor.getSelection().trim();
        if (selectedText) {
          this.handleContextMenu(menu, editor);
        }
      })
    );

    this.registerEvent(
      this.app.workspace.on("active-leaf-change", (leaf) => {
        if (leaf && leaf.view instanceof MarkdownView) {
          const file = leaf.view.file;
          if (file) {
            const activeCopilotView = this.app.workspace
              .getLeavesOfType(CHAT_VIEWTYPE)
              .find((leaf) => leaf.view instanceof CopilotView)?.view as CopilotView;

            if (activeCopilotView) {
              const event = new CustomEvent(EVENT_NAMES.ACTIVE_LEAF_CHANGE);
              activeCopilotView.eventTarget.dispatchEvent(event);
            }
          }
        }
      })
    );

    console.log("Loading Obsidian Transcription");
    if (this.asrSettings.Asr_debugMode) console.log("Debug mode enabled");

    this.transcriptionEngine = new TranscriptionEngine(
      this,
      this.app.vault,
      this.statusBarReadwise,
      this.app
    );

    if (!Platform.isMobileApp) {
      this.statusBarReadwise = new StatusBarReadwise(this.addStatusBarItem());
      this.registerInterval(window.setInterval(() => this.statusBarReadwise.display(), 1000));
    }

    // Register the file-menu event
    this.registerEvent(this.app.workspace.on("file-menu", this.onFileMenu.bind(this)));

    this.addCommand({
      id: "obsidian-transcription-add-file",
      name: "Add File to Transcription",
      editorCallback: async () => {
        class FileSelectionModal extends Modal {
          onOpen() {
            const { contentEl } = this;
            contentEl.createEl("h2", { text: "Select files:" });
            const input = contentEl.createEl("input", {
              type: "file",
              attr: { multiple: "" },
            });
            contentEl.createEl("br");
            contentEl.createEl("br");
            const button = contentEl.createEl("button", { text: "Add file link" });
            button.addEventListener("click", () => {
              const fileList = input.files;
              if (fileList) {
                const files = Array.from(fileList);
                let path = "";
                for (const file of files) {
                  //     console.log(file)
                  //@ts-ignore
                  path = this.app.vault.getResourcePath(file).toString();
                  //console.log(path.toString())
                }
                // this.app.vault.copy

                // //@ts-ignore
                // let attachementFolder = this.app.vault.config.attachmentFolderPath;
                //@ts-ignore
                const basePath = this.app.vault.adapter.basePath;
                // console.log(attachementFolder);
                // console.log(basePath);

                const fe = new FileLink(path, basePath);

                files.forEach((file: File) => {
                  fe.embedFile(file);
                });
              }
            });
          }
        }
        new FileSelectionModal(this.app).open();
      },
    });

    this.addCommand({
      id: "obsidian-transcription-stop",
      name: "Stop Transcription",
      editorCallback: async () => {
        try {
          // Check if there is an ongoing transcription task
          if (this.ongoingTranscriptionTasks.length > 0) {
            console.log("Stopping ongoing transcription...");

            // Loop through each ongoing task and signal abort
            for (const { abortController, task } of this.ongoingTranscriptionTasks) {
              abortController.abort();
              await task.catch(() => {}); // Catch any errors during abortion
            }

            // Clear the ongoing transcription tasks after completion or cancellation
            this.ongoingTranscriptionTasks = [];
          } else {
            new Notice("No ongoing transcription to stop", 5 * 1000);
          }
        } catch (error) {
          console.error("Error stopping transcription:", error);
        }
      },
    });

    this.addCommand({
      id: "obsidian-transcription-transcribe-all-in-view",
      name: "Transcribe all files in view",
      editorCallback: async (editor: Editor, view: MarkdownView) => {
        if (view.file === null) return;

        const filesToTranscribe = await this.getTranscribeableFiles(view.file);
        const fileNames = filesToTranscribe.map((file) => file.name).join(", ");
        new Notice(`Files Selected: ${fileNames}`, 5 * 1000);
        for (const fileToTranscribe of filesToTranscribe) {
          const abortController = new AbortController();
          const task = this.transcribeAndWrite(view.file, fileToTranscribe, abortController);
          this.ongoingTranscriptionTasks.push({ task, abortController });
          await task;
        }
      },
    });

    this.addCommand({
      id: "obsidian-transcription-transcribe-specific-file-in-view",
      name: "Transcribe file in view",
      editorCallback: async (editor: Editor, view: MarkdownView) => {
        // Get the current filepath
        if (view.file === null) return;

        const filesToTranscribe = await this.getTranscribeableFiles(view.file);

        // Now that we have all the files to transcribe, we can prompt the user to choose which one they want to transcribe

        class FileSelectionModal extends FuzzySuggestModal<TFile> {
          public transcriptionInstance: CopilotPlugin; // Reference to Transcription instance

          constructor(app: App, transcriptionInstance: CopilotPlugin) {
            super(app);
            this.transcriptionInstance = transcriptionInstance;
          }

          getItems(): TFile[] {
            return filesToTranscribe;
          }

          getItemText(file: TFile): string {
            return file.name;
          }

          async onChooseItem(file: TFile) {
            if (view.file === null) return;

            new Notice(`File Selected: ${file.name}`, 5 * 1000);
            const abortController = new AbortController();
            const task = this.transcriptionInstance.transcribeAndWrite(
              view.file,
              file,
              abortController
            );
            this.transcriptionInstance.ongoingTranscriptionTasks.push({
              task,
              abortController,
            });
            await task;
          }
        }

        new FileSelectionModal(this.app, this).open();
      },
    });

    // Kill child processes when the plugin is unloaded
    this.app.workspace.on("quit", () => {
      CopilotPlugin.children.forEach((child) => {
        child.kill();
      });
    });

    // This adds a settings tab so the user can configure various aspects of the plugin
    //this.addSettingTab(new TranscriptionSettingTab(this.app, this));
    console.log("Loading Obsidian Whisper");

    this.addRibbonIcon("activity", "Open recording controls", (evt) => {
      this.openRecordingControls();
    });

    //this.addSettingTab(new WhisperSettingsTab(this.app, this));

    this.timer = new Timer();
    this.audioHandler = new AudioHandler(this);
    this.recorder = new NativeAudioRecorder();

    this.statusBarRecord = new StatusBarRecord(this);
    this.addCommand({
      id: "start-stop-recording",
      name: "Start/stop recording",
      callback: async () => {
        if (this.statusBarRecord.status !== RecordingStatus.Recording) {
          this.statusBarRecord.updateStatus(RecordingStatus.Recording);
          await this.recorder.startRecording();
        } else {
          this.statusBarRecord.updateStatus(RecordingStatus.Processing);
          const audioBlob = await this.recorder.stopRecording();
          const extension = this.recorder.getMimeType()?.split("/")[1];
          const fileName = `${new Date().toISOString().replace(/[:.]/g, "-")}.${extension}`;
          // Use audioBlob to send or save the recorded audio as needed
          await this.audioHandler.sendAudioData(audioBlob, fileName);
          this.statusBarRecord.updateStatus(RecordingStatus.Idle);
        }
      },
      hotkeys: [
        {
          modifiers: ["Alt"],
          key: "Q",
        },
      ],
    });

    this.addCommand({
      id: "upload-audio-file",
      name: "Upload audio file",
      callback: () => {
        // Create an input element for file selection
        const fileInput = document.createElement("input");
        fileInput.type = "file";
        fileInput.accept = "audio/*"; // Accept only audio files

        // Handle file selection
        fileInput.onchange = async (event) => {
          const files = (event.target as HTMLInputElement).files;
          if (files && files.length > 0) {
            const file = files[0];
            const fileName = file.name;
            const audioBlob = file.slice(0, file.size, file.type);
            // Use audioBlob to send or save the uploaded audio as needed
            await this.audioHandler.sendAudioData(audioBlob, fileName);
          }
        };

        // Programmatically open the file dialog
        fileInput.click();
      },
    });

    this.addCommand({
      id: "open-recording-controls",
      name: "打开录音控制面板",
      callback: () => {
        this.openRecordingControls();
      },
    });
    this.registerEditorMenu();
  }

  onFileMenu(menu: Menu, file: TFile) {
    const parentFile = this.app.workspace.getActiveFile();

    // Check if the parent file is not null and the file is of a type you want to handle
    if (parentFile instanceof TFile && file instanceof TFile) {
      // Get the file extension
      const fileExtension = file.extension?.toLowerCase();

      // Check if the file extension is in the allowed list
      if (fileExtension && CopilotPlugin.transcribeFileExtensions.includes(fileExtension)) {
        // Add a new item to the right-click menu
        menu.addItem((item) => {
          item
            .setTitle("Transcribe")
            .setIcon("headphones")
            .onClick(async () => {
              // Handle the click event
              const abortController = new AbortController();
              const task = this.transcribeAndWrite(parentFile, file, abortController);
              this.ongoingTranscriptionTasks.push({
                task,
                abortController,
              });
              await task;
            });
        });
      }
    }
  }

  async onunload() {
    // Clean up VectorStoreManager
    if (this.vectorStoreManager) {
      this.vectorStoreManager.onunload();
    }
    this.settingsUnsubscriber?.();

    console.log("Copilot plugin unloaded");

    if (this.asrSettings.Asr_debugMode) console.log("Unloading Obsidian Transcription");
  }

  updateUserMessageHistory(newMessage: string) {
    this.userMessageHistory = [...this.userMessageHistory, newMessage];
  }

  async autosaveCurrentChat() {
    if (getSettings().autosaveChat) {
      const chatView = this.app.workspace.getLeavesOfType(CHAT_VIEWTYPE)[0]?.view as CopilotView;
      if (chatView && chatView.sharedState.chatHistory.length > 0) {
        await chatView.saveChat();
      }
    }
  }

  async processText(
    editor: Editor,
    eventType: string,
    eventSubtype?: string,
    checkSelectedText = true
  ) {
    const selectedText = await editor.getSelection();

    const isChatWindowActive = this.app.workspace.getLeavesOfType(CHAT_VIEWTYPE).length > 0;

    if (!isChatWindowActive) {
      await this.activateView();
    }

    // Without the timeout, the view is not yet active
    setTimeout(() => {
      const activeCopilotView = this.app.workspace
        .getLeavesOfType(CHAT_VIEWTYPE)
        .find((leaf) => leaf.view instanceof CopilotView)?.view as CopilotView;
      if (activeCopilotView && (!checkSelectedText || selectedText)) {
        const event = new CustomEvent(eventType, { detail: { selectedText, eventSubtype } });
        activeCopilotView.eventTarget.dispatchEvent(event);
      }
    }, 0);
  }

  processSelection(editor: Editor, eventType: string, eventSubtype?: string) {
    this.processText(editor, eventType, eventSubtype);
  }

  emitChatIsVisible() {
    const activeCopilotView = this.app.workspace
      .getLeavesOfType(CHAT_VIEWTYPE)
      .find((leaf) => leaf.view instanceof CopilotView)?.view as CopilotView;

    if (activeCopilotView) {
      const event = new CustomEvent(EVENT_NAMES.CHAT_IS_VISIBLE);
      activeCopilotView.eventTarget.dispatchEvent(event);
    }
  }

  initActiveLeafChangeHandler() {
    this.registerEvent(
      this.app.workspace.on("active-leaf-change", (leaf) => {
        if (!leaf) {
          return;
        }
        if (leaf.getViewState().type === CHAT_VIEWTYPE) {
          this.emitChatIsVisible();
        }
      })
    );
  }

  private getCurrentEditorOrDummy(): Editor {
    const activeView = this.app.workspace.getActiveViewOfType(MarkdownView);
    return {
      getSelection: () => {
        const selection = activeView?.editor?.getSelection();
        if (selection) return selection;
        // Default to the entire active file if no selection
        const activeFile = this.app.workspace.getActiveFile();
        return activeFile ? this.app.vault.cachedRead(activeFile) : "";
      },
      replaceSelection: activeView?.editor?.replaceSelection.bind(activeView.editor) || (() => {}),
    } as Partial<Editor> as Editor;
  }

  processCustomPrompt(eventType: string, customPrompt: string) {
    const editor = this.getCurrentEditorOrDummy();
    this.processText(editor, eventType, customPrompt, false);
  }

  toggleView() {
    const leaves = this.app.workspace.getLeavesOfType(CHAT_VIEWTYPE);
    if (leaves.length > 0) {
      this.deactivateView();
    } else {
      this.activateView();
    }
  }

  async activateView(): Promise<void> {
    const leaves = this.app.workspace.getLeavesOfType(CHAT_VIEWTYPE);
    if (leaves.length === 0) {
      if (getSettings().defaultOpenArea === DEFAULT_OPEN_AREA.VIEW) {
        await this.app.workspace.getRightLeaf(false).setViewState({
          type: CHAT_VIEWTYPE,
          active: true,
        });
      } else {
        await this.app.workspace.getLeaf(true).setViewState({
          type: CHAT_VIEWTYPE,
          active: true,
        });
      }
    } else {
      this.app.workspace.revealLeaf(leaves[0]);
    }
    this.emitChatIsVisible();
  }

  async deactivateView() {
    this.app.workspace.detachLeavesOfType(CHAT_VIEWTYPE);
  }

  async loadSettings() {
    const savedSettings = await this.loadData();
    const sanitizedSettings = sanitizeSettings(savedSettings);
    setSettings(sanitizedSettings);
    this.asrSettings = Object.assign({}, sanitizedSettings, await this.loadData());
  }

  registerEditorMenu() {
    this.registerEvent(
      this.app.workspace.on("editor-menu", (menu, editor, view) => {
        menu.addItem((item) => {
          item
            .setTitle("语音输入文字")
            .setIcon("microphone")
            .onClick(() => {
              this.openRecordingControls();
            });
        });
      })
    );
  }

  openRecordingControls() {
    const CustomParam = { isCopilot: false };
    if (!this.controls) {
      this.controls = new Controls(this, CustomParam);
    }
    this.controls.open();
  }
  mergeActiveModels(
    existingActiveModels: CustomModel[],
    builtInModels: CustomModel[]
  ): CustomModel[] {
    const modelMap = new Map<string, CustomModel>();

    // Create a unique key for each model, it's model (name + provider)

    // Add or update existing models in the map
    existingActiveModels.forEach((model) => {
      const key = getModelKeyFromModel(model);
      const existingModel = modelMap.get(key);
      if (existingModel) {
        // If it's a built-in model, preserve the built-in status
        modelMap.set(key, {
          ...model,
          isBuiltIn: existingModel.isBuiltIn || model.isBuiltIn,
        });
      } else {
        modelMap.set(key, model);
      }
    });

    return Array.from(modelMap.values());
  }

  handleContextMenu = (menu: Menu, editor: Editor): void => {
    registerContextMenu(menu, editor, this);
  };

  async loadCopilotChatHistory() {
    const chatFiles = await this.getChatHistoryFiles();
    if (chatFiles.length === 0) {
      new Notice("No chat history found.");
      return;
    }
    new LoadChatHistoryModal(this.app, chatFiles, this.loadChatHistory.bind(this)).open();
  }

  async getChatHistoryFiles(): Promise<TFile[]> {
    const folder = this.app.vault.getAbstractFileByPath(getSettings().defaultSaveFolder);
    if (!(folder instanceof TFolder)) {
      return [];
    }
    const files = await this.app.vault.getMarkdownFiles();
    return files.filter((file) => file.path.startsWith(folder.path));
  }

  async loadChatHistory(file: TFile) {
    const content = await this.app.vault.read(file);
    const messages = parseChatContent(content);
    this.sharedState.clearChatHistory();
    messages.forEach((message) => this.sharedState.addMessage(message));

    // Update the chain's memory with the loaded messages
    await updateChatMemory(messages, this.chainManager.memoryManager);

    // Check if the Copilot view is already active
    const existingView = this.app.workspace.getLeavesOfType(CHAT_VIEWTYPE)[0];
    if (!existingView) {
      // Only activate the view if it's not already open
      this.activateView();
    } else {
      // If the view is already open, just update its content
      const copilotView = existingView.view as CopilotView;
      copilotView.updateView();
    }
  }

  async customSearchDB(query: string, salientTerms: string[], textWeight: number): Promise<any[]> {
    const hybridRetriever = new HybridRetriever({
      minSimilarityScore: 0.3,
      maxK: 20,
      salientTerms: salientTerms,
      textWeight: textWeight,
    });

    const results = await hybridRetriever.getOramaChunks(query, salientTerms);
    return results.map((doc) => ({
      content: doc.pageContent,
      metadata: doc.metadata,
    }));
  }
}
