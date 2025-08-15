import axios from "axios";
import Whisper from "@/main";
import { Notice, MarkdownView, requestUrl } from "obsidian";
import { getBaseFileName, payloadGenerator } from "../utils";
import { DEFAULT_SETTINGS } from "@/constants";

export class AudioHandler {
  private plugin: Whisper;

  constructor(plugin: Whisper) {
    this.plugin = plugin;
  }

  async sendAudioData(blob: Blob, fileName: string): Promise<void> {
    // Get the base file name without extension
    const baseFileName = getBaseFileName(fileName);

    const audioFilePath = `${
      this.plugin.asrSettings.Asr_saveAudioFilePath
        ? `${this.plugin.asrSettings.Asr_saveAudioFilePath}/`
        : ""
    }${fileName}`;

    const noteFilePath = `${
      this.plugin.asrSettings.Asr_createNewFileAfterRecordingPath
        ? `${this.plugin.asrSettings.Asr_createNewFileAfterRecordingPath}/`
        : ""
    }${baseFileName}.md`;

    if (this.plugin.asrSettings.Asr_debugMode) {
      new Notice(`Sending audio data size: ${blob.size / 1000} KB`);
    }

    if (!this.plugin.asrSettings.Asr_useLocalService && !this.plugin.asrSettings.Asr_apiKey) {
      new Notice("API key is missing. Please add your API key in the settings.");
      return;
    }

    const formData = new FormData();
    formData.append("file", blob, fileName);
    formData.append("model", this.plugin.asrSettings.Asr_transcriptionEngine);
    formData.append("language", this.plugin.asrSettings.Asr_language);
    if (this.plugin.asrSettings.Asr_prompt)
      formData.append("prompt", this.plugin.asrSettings.Asr_prompt);

    try {
      // If the saveAudioFile setting is true, save the audio file
      if (this.plugin.asrSettings.Asr_saveAudioFile) {
        const arrayBuffer = await blob.arrayBuffer();
        await this.plugin.app.vault.adapter.writeBinary(audioFilePath, new Uint8Array(arrayBuffer));
        new Notice("Audio saved successfully.");
      }
    } catch (err) {
      console.error("Error saving audio file:", err);
      new Notice("Error saving audio file: " + err.message);
    }

    try {
      if (this.plugin.asrSettings.Asr_debugMode) {
        new Notice("Parsing audio data:" + fileName);
      }

      let response;
      if (this.plugin.asrSettings.Asr_useLocalService) {
        // 使用本地 whisper ASR 服务
        response = await this.sendToLocalService(blob, fileName);
      } else {
        // 使用远程 OpenAI API
        response = await axios.post(this.plugin.asrSettings.Asr_apiUrl, formData, {
          headers: {
            "Content-Type": "multipart/form-data",
            Authorization: `Bearer ${this.plugin.asrSettings.Asr_apiKey}`,
          },
        });
      }

      // Determine if a new file should be created
      const activeView = this.plugin.app.workspace.getActiveViewOfType(MarkdownView);
      const shouldCreateNewFile =
        this.plugin.asrSettings.Asr_createNewFileAfterRecording || !activeView;

      if (shouldCreateNewFile) {
        await this.plugin.app.vault.create(
          noteFilePath,
          `![[${audioFilePath}]]\n${response.data.text}`
        );
        await this.plugin.app.workspace.openLinkText(noteFilePath, "", true);
      } else {
        // Insert the transcription at the cursor position
        const editor = this.plugin.app.workspace.getActiveViewOfType(MarkdownView)?.editor;
        if (editor) {
          const cursorPosition = editor.getCursor();
          editor.replaceRange(response.data.text, cursorPosition);

          // Move the cursor to the end of the inserted text
          const newPosition = {
            line: cursorPosition.line,
            ch: cursorPosition.ch + response.data.text.length,
          };
          editor.setCursor(newPosition);
        }
      }

      new Notice("Audio parsed successfully.");
    } catch (err) {
      console.error("Error parsing audio:", err);
      new Notice("Error parsing audio: " + err.message);
    }
  }

  async sendAudioData2copilot(blob: Blob, fileName: string): Promise<void> {
    // Get the base file name without extension

    const audioFilePath = `${
      this.plugin.asrSettings.Asr_saveAudioFilePath
        ? `${this.plugin.asrSettings.Asr_saveAudioFilePath}/`
        : ""
    }${fileName}`;

    if (this.plugin.asrSettings.Asr_debugMode) {
      new Notice(`Sending audio data size: ${blob.size / 1000} KB`);
    }

    if (!this.plugin.asrSettings.Asr_useLocalService && !this.plugin.asrSettings.Asr_apiKey) {
      new Notice("API key is missing. Please add your API key in the settings.");
      return;
    }

    const formData = new FormData();
    formData.append("file", blob, fileName);
    formData.append("model", this.plugin.asrSettings.Asr_transcriptionEngine);
    formData.append("language", this.plugin.asrSettings.Asr_language);
    if (this.plugin.asrSettings.Asr_prompt)
      formData.append("prompt", this.plugin.asrSettings.Asr_prompt);

    try {
      // If the saveAudioFile setting is true, save the audio file
      if (this.plugin.asrSettings.Asr_saveAudioFile) {
        const arrayBuffer = await blob.arrayBuffer();
        await this.plugin.app.vault.adapter.writeBinary(audioFilePath, new Uint8Array(arrayBuffer));
        new Notice("Audio saved successfully.");
      }
    } catch (err) {
      console.error("Error saving audio file:", err);
      new Notice("Error saving audio file: " + err.message);
    }

    try {
      if (this.plugin.asrSettings.Asr_debugMode) {
        new Notice("Parsing audio data:" + fileName);
      }

      let response;
      if (this.plugin.asrSettings.Asr_useLocalService) {
        // 使用本地 whisper ASR 服务
        response = await this.sendToLocalService(blob, fileName);
      } else {
        // 使用远程 OpenAI API
        response = await axios.post(this.plugin.asrSettings.Asr_apiUrl, formData, {
          headers: {
            "Content-Type": "multipart/form-data",
            Authorization: `Bearer ${this.plugin.asrSettings.Asr_apiKey}`,
          },
        });
      }
      if (this.plugin.setChatInput) {
        this.plugin.setChatInput(response.data.text);
      }
      new Notice("Audio parsed successfully.");
    } catch (err) {
      console.error("Error parsing audio:", err);
      new Notice("Error parsing audio: " + err.message);
    }
  }

  async sendToLocalService(blob: Blob, fileName: string): Promise<{ data: { text: string } }> {
    const payload_data: Record<string, any> = {};
    payload_data["audio_file"] = blob;
    const [request_body, boundary_string] = await payloadGenerator(payload_data);

    let args = "output=json";
    args += `&word_timestamps=true`;

    const { Asr_translate, Asr_encode, Asr_vadFilter, Asr_language, Asr_prompt } =
      this.plugin.asrSettings;
    if (Asr_translate) args += `&task=translate`;
    if (Asr_encode !== true) args += `&encode=${Asr_encode}`;
    if (Asr_vadFilter !== false) args += `&vad_filter=${Asr_vadFilter}`;
    if (Asr_language !== "en") args += `&language=${Asr_language}`;
    if (Asr_prompt) args += `&initial_prompt=${Asr_prompt}`;


    // 修复：确保 Asr_localServiceUrl 存在且为字符串，如果不存在则使用默认值
    const localServiceUrl = this.plugin.asrSettings.Asr_localServiceUrl || DEFAULT_SETTINGS.Asr_localServiceUrl;
    const urls = localServiceUrl.split(";").filter(Boolean);

    // const urls = this.plugin.asrSettings.Asr_localServiceUrl.split(";").filter(Boolean);

    for (const baseUrl of urls) {
      const url = `${baseUrl}/asr?${args}`;
      console.log("Trying URL:", url);

      const options = {
        method: "POST",
        url,
        contentType: `multipart/form-data; boundary=----${boundary_string}`,
        body: request_body,
      };

      console.log("Options:", options);

      try {
        const response = await requestUrl(options);
        if (this.plugin.asrSettings.Asr_debugMode) {
          console.log("Raw response:", response);
        }

        // 处理响应数据，确保格式与 OpenAI API 兼容
        let transcriptionText = "";
        if (response.json && response.json.text) {
          transcriptionText = response.json.text;
        } else if (response.json && response.json.segments) {
          transcriptionText = response.json.segments.map((segment: any) => segment.text).join("");
        }

        return {
          data: {
            text: transcriptionText,
          },
        };
      } catch (error) {
        if (this.plugin.asrSettings.Asr_debugMode) {
          console.error("Error with URL:", url, error);
        }
        // 如果是最后一个 URL，抛出错误
        if (baseUrl === urls[urls.length - 1]) {
          throw error;
        }
      }
    }

    throw new Error("All local service URLs failed");
  }
}
