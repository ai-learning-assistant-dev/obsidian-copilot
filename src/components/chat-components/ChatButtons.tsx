import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { USER_SENDER } from "@/constants";
import { cn } from "@/lib/utils";
import { ChatMessage } from "@/sharedState";
import {
  Check,
  Copy,
  LibraryBig,
  PenSquare,
  RotateCw,
  TextCursorInput,
  Trash2,
  Volume2, // 新增TTS图标
} from "lucide-react";
import { Platform } from "obsidian";
import React from "react";

interface ChatButtonsProps {
  message: ChatMessage;
  onCopy: () => void;
  isCopied: boolean;
  onInsertIntoEditor?: () => void;
  onRegenerate?: () => void;
  onEdit?: () => void;
  onDelete: () => void;
  onShowSources?: () => void;
  hasSources: boolean;
  onSpeak?: (textToSpeak?: string) => void; // 新增TTS回调
}

export const ChatButtons: React.FC<ChatButtonsProps> = ({
  message,
  onCopy,
  isCopied,
  onInsertIntoEditor,
  onRegenerate,
  onEdit,
  onDelete,
  onShowSources,
  hasSources,
  onSpeak, // 新增
}) => {
  return (
    <div
      className={cn("flex gap-1", {
        "group-hover:opacity-100 opacity-0": !Platform.isMobile,
      })}
    >
      {message.sender === USER_SENDER ? (
        <>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button onClick={onEdit} variant="ghost2" size="fit" title="Edit">
                <PenSquare className="size-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Edit</TooltipContent>
          </Tooltip>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button onClick={onDelete} variant="ghost2" size="fit" title="Delete">
                <Trash2 className="size-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Delete</TooltipContent>
          </Tooltip>
        </>
      ) : (
        <>
          {hasSources && (
            <Tooltip>
              <TooltipTrigger asChild>
                <Button onClick={onShowSources} variant="ghost2" size="fit" title="Show Sources">
                  <LibraryBig className="size-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Show Sources</TooltipContent>
            </Tooltip>
          )}
          {/* 新增TTS播放按钮 */}
          <Tooltip>
            <TooltipTrigger asChild>
              <Button 
                onClick={() => onSpeak?.(message.message)} 
                variant="ghost2" 
                size="fit" 
                title="Text-to-Speech"
              >
                <Volume2 className="size-4" /> {/* 需要从lucide-react导入Volume2图标 */}
              </Button>
            </TooltipTrigger>
            <TooltipContent>Read Aloud</TooltipContent>
          </Tooltip>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                onClick={onInsertIntoEditor}
                variant="ghost2"
                size="fit"
                title="Insert / Replace at cursor"
              >
                <TextCursorInput className="size-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Insert / Replace at cursor</TooltipContent>
          </Tooltip>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="ghost2" size="fit" onClick={onCopy} title="Copy">
                {isCopied ? <Check className="size-4" /> : <Copy className="size-4" />}
              </Button>
            </TooltipTrigger>
            <TooltipContent>Copy</TooltipContent>
          </Tooltip>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button onClick={onRegenerate} variant="ghost2" size="fit" title="Regenerate">
                <RotateCw className="size-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Regenerate</TooltipContent>
          </Tooltip>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button onClick={onDelete} variant="ghost2" size="fit" title="Delete">
                <Trash2 className="size-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Delete</TooltipContent>
          </Tooltip>
        </>
      )}
    </div>
  );
};
