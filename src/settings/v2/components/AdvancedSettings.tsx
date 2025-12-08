import { SettingItem } from "@/components/ui/setting-item";
import { updateSetting, useSettingsValue } from "@/settings/model";
import React from "react";

export const AdvancedSettings: React.FC = () => {
  const settings = useSettingsValue();

  return (
    <div className="tw-space-y-4">
      {/* Privacy Settings Section */}
      <section>
        <SettingItem
          type="textarea"
          title="User System Prompt"
          description="Customize the system prompt for all messages, may result in unexpected behavior!"
          value={settings.userSystemPrompt}
          onChange={(value) => updateSetting("userSystemPrompt", value)}
          placeholder="Enter your system prompt here..."
        />

        <div className="tw-space-y-4">
          <SettingItem
            type="switch"
            title="Enable Encryption"
            description="Enable encryption for the API keys."
            checked={settings.enableEncryption}
            onCheckedChange={(checked) => {
              updateSetting("enableEncryption", checked);
            }}
          />

          <SettingItem
            type="switch"
            title="Enable Web Search"
            description="Allow Copilot to search the web to answer questions."
            checked={settings.useWebSearch}
            onCheckedChange={(checked) => {
              updateSetting("useWebSearch", checked);
            }}
          />

          {settings.useWebSearch && (
            <>
              <SettingItem
                type="select"
                title="Search Engine Type"
                description="Choose between free search (Baidu first, Bing fallback, no API key, works in China) or Zhipu AI (requires API key, better results)."
                value={settings.searchEngineType || "free"}
                onChange={(value) => updateSetting("searchEngineType", value as "zhipu" | "free")}
                options={[
                  { label: "Free Search (免费搜索 - 百度优先)", value: "free" },
                  { label: "Zhipu AI (智谱AI - 需要API Key)", value: "zhipu" },
                ]}
              />

              {settings.searchEngineType === "zhipu" && (
                <SettingItem
                  type="text"
                  title="Zhipu API Key"
                  description="API Key for Zhipu AI web search."
                  value={settings.zhipuApiKey}
                  onChange={(value) => updateSetting("zhipuApiKey", value)}
                  placeholder="Enter your Zhipu API Key"
                />
              )}
            </>
          )}
          <SettingItem
            type="switch"
            title="Debug Mode"
            description="Debug mode will log some debug message to the console."
            checked={settings.debug}
            onCheckedChange={(checked) => {
              updateSetting("debug", checked);
            }}
          />
        </div>
      </section>
    </div>
  );
};
