import { SettingItem } from "@/components/ui/setting-item";
import { updateSetting, useSettingsValue } from "@/settings/model";
import React from "react";
import { t } from "@/lang/helper";

export const AdvancedSettings: React.FC = () => {
  const settings = useSettingsValue();

  return (
    <div className="space-y-4">
      {/* Privacy Settings Section */}
      <section>
        <SettingItem
          type="textarea"
          title={t("User System Prompt")}
          description={t(
            "Customize the system prompt for all messages, may result in unexpected behavior!"
          )}
          value={settings.userSystemPrompt}
          onChange={(value) => updateSetting("userSystemPrompt", value)}
          placeholder={t("Enter your system prompt here...")}
        />

        <div className="space-y-4">
          <SettingItem
            type="switch"
            title={t("Custom Prompt Templating")}
            description={t(
              "Enable templating to process variables like {activenote}, {foldername} or {#tag} in prompts. Disable to use raw prompts without any processing."
            )}
            checked={settings.enableCustomPromptTemplating}
            onCheckedChange={(checked) => {
              updateSetting("enableCustomPromptTemplating", checked);
            }}
          />

          <SettingItem
            type="switch"
            title={t("Images in Markdown (Plus)")}
            description={t(
              "Pass embedded images in markdown to the AI along with the text. Only works with multimodal models (plus only)."
            )}
            checked={settings.passMarkdownImages}
            onCheckedChange={(checked) => {
              updateSetting("passMarkdownImages", checked);
            }}
          />

          <SettingItem
            type="switch"
            title={t("Enable Encryption")}
            description={t("Enable encryption for the API keys.")}
            checked={settings.enableEncryption}
            onCheckedChange={(checked) => {
              updateSetting("enableEncryption", checked);
            }}
          />

          <SettingItem
            type="switch"
            title={t("Debug Mode")}
            description={t("Debug mode will log some debug message to the console.")}
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
