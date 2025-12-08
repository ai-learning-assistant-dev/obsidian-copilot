/**
 * 上下文图片收集器
 * 用于收集笔记和文件夹中的图片信息，提供给 AI 作为上下文
 */

import { App, TFile, TFolder } from "obsidian";
import { SUPPORTED_IMAGE_EXTENSIONS } from "./imageLoader";

/** 图片信息 */
export interface ImageInfo {
  /** 图片文件名 */
  name: string;
  /** 图片完整路径 */
  path: string;
  /** 文件大小（字节） */
  size: number;
  /** 所在文件夹 */
  folder: string;
}

/**
 * 从笔记内容中提取所有图片引用
 * 
 * @param content - 笔记内容
 * @returns 图片路径列表
 * 
 * @example
 * extractImageReferencesFromContent("文本 ![[image.png]] 更多文本 ![alt](photo.jpg)")
 * // 返回: ["image.png", "photo.jpg"]
 */
export function extractImageReferencesFromContent(content: string): string[] {
  const imagePaths: string[] = [];
  
  // 匹配 Obsidian 语法: ![[image.png]]
  const obsidianRegex = /!\[\[([^\]]+?\.(png|jpg|jpeg|gif|webp|svg|bmp|tiff))\]\]/gi;
  let match: RegExpExecArray | null;
  
  while ((match = obsidianRegex.exec(content)) !== null) {
    imagePaths.push(match[1].trim());
  }
  
  // 匹配 Markdown 语法: ![alt](image.png)
  const markdownRegex = /!\[([^\]]*)\]\(([^)]+?\.(png|jpg|jpeg|gif|webp|svg|bmp|tiff))\)/gi;
  
  while ((match = markdownRegex.exec(content)) !== null) {
    imagePaths.push(match[2].trim());
  }
  
  // 去重
  return [...new Set(imagePaths)];
}

/**
 * 从笔记文件中获取所有图片引用
 * 
 * @param app - Obsidian App 实例
 * @param file - 笔记文件
 * @returns 图片路径列表
 */
export async function getImagesFromNote(app: App, file: TFile): Promise<string[]> {
  try {
    const content = await app.vault.read(file);
    return extractImageReferencesFromContent(content);
  } catch (error) {
    console.error(`Error reading note ${file.path}:`, error);
    return [];
  }
}

/**
 * 获取文件夹中的所有图片文件
 * 
 * @param app - Obsidian App 实例
 * @param folderPath - 文件夹路径
 * @param recursive - 是否递归搜索子文件夹（默认 true）
 * @returns 图片信息列表
 */
export function getImagesFromFolder(
  app: App,
  folderPath: string,
  recursive: boolean = true
): ImageInfo[] {
  const images: ImageInfo[] = [];
  
  const folder = app.vault.getAbstractFileByPath(folderPath);
  if (!(folder instanceof TFolder)) {
    return images;
  }
  
  const processFolder = (currentFolder: TFolder) => {
    for (const child of currentFolder.children) {
      if (child instanceof TFile) {
        // 检查是否为图片文件
        const ext = child.extension.toLowerCase();
        if (SUPPORTED_IMAGE_EXTENSIONS.includes(ext)) {
          images.push({
            name: child.name,
            path: child.path,
            size: child.stat.size,
            folder: currentFolder.path,
          });
        }
      } else if (recursive && child instanceof TFolder) {
        // 递归处理子文件夹
        processFolder(child);
      }
    }
  };
  
  processFolder(folder);
  return images;
}

/**
 * 获取整个 vault 中的所有图片文件
 * 
 * @param app - Obsidian App 实例
 * @param excludeFolders - 要排除的文件夹列表（可选）
 * @returns 图片信息列表
 */
export function getAllImagesInVault(
  app: App,
  excludeFolders: string[] = []
): ImageInfo[] {
  const images: ImageInfo[] = [];
  const allFiles = app.vault.getFiles();
  
  for (const file of allFiles) {
    // 检查文件是否在排除文件夹中
    const isExcluded = excludeFolders.some(excludePath => 
      file.path.startsWith(excludePath + '/')
    );
    
    if (isExcluded) continue;
    
    // 检查是否为图片文件
    const ext = file.extension.toLowerCase();
    if (SUPPORTED_IMAGE_EXTENSIONS.includes(ext)) {
      const folderPath = file.path.substring(0, file.path.lastIndexOf('/'));
      images.push({
        name: file.name,
        path: file.path,
        size: file.stat.size,
        folder: folderPath || '/',
      });
    }
  }
  
  return images;
}

/**
 * 根据活动笔记收集相关图片
 * 策略：
 * 1. 笔记中引用的图片
 * 2. 笔记所在文件夹的图片
 * 3. 附件文件夹的图片（如果配置了）
 * 
 * @param app - Obsidian App 实例
 * @param activeFile - 当前活动文件
 * @param includeFolder - 是否包含文件夹中的其他图片（默认 true）
 * @returns 图片路径列表（去重）
 */
export async function getRelevantImagesForNote(
  app: App,
  activeFile: TFile,
  includeFolder: boolean = true
): Promise<string[]> {
  const imagePaths = new Set<string>();
  
  // 1. 收集笔记中引用的图片
  const referencedImages = await getImagesFromNote(app, activeFile);
  referencedImages.forEach(path => imagePaths.add(path));
  
  // 2. 收集笔记所在文件夹的图片
  if (includeFolder) {
    const folderPath = activeFile.path.substring(0, activeFile.path.lastIndexOf('/'));
    if (folderPath) {
      const folderImages = getImagesFromFolder(app, folderPath, false); // 不递归
      folderImages.forEach(img => imagePaths.add(img.name));
    }
  }
  
  // 3. 检查是否配置了附件文件夹
  // @ts-ignore - Obsidian 内部 API
  const attachmentFolder = app.vault.config?.attachmentFolderPath;
  if (attachmentFolder && includeFolder) {
    const attachmentImages = getImagesFromFolder(app, attachmentFolder, true);
    attachmentImages.forEach(img => imagePaths.add(img.name));
  }
  
  return Array.from(imagePaths);
}

/**
 * 格式化图片列表为 AI 友好的文本
 * 
 * @param imagePaths - 图片路径列表
 * @param maxCount - 最大显示数量（默认 20）
 * @returns 格式化的文本
 */
export function formatImagesForAI(imagePaths: string[], maxCount: number = 20): string {
  if (imagePaths.length === 0) {
    return '';
  }
  
  const displayPaths = imagePaths.slice(0, maxCount);
  const hasMore = imagePaths.length > maxCount;
  
  let text = '\n\n---\n**Available Images in Context:**\n\n';
  text += 'You can reference these images in your response using the syntax: `![[image-name.png]]`\n\n';
  
  displayPaths.forEach(path => {
    text += `- ![[${path}]]\n`;
  });
  
  if (hasMore) {
    text += `\n...and ${imagePaths.length - maxCount} more images.\n`;
  }
  
  text += '\n---\n';
  
  return text;
}

/**
 * 生成图片上下文提示（用于添加到用户消息或系统提示中）
 * 
 * @param app - Obsidian App 实例
 * @param activeFile - 当前活动文件（可选）
 * @param includeAll - 是否包含所有相关图片（默认 false，只包含引用的）
 * @returns 图片上下文文本
 */
export async function generateImageContext(
  app: App,
  activeFile?: TFile,
  includeAll: boolean = false
): Promise<string> {
  if (!activeFile) {
    return '';
  }
  
  let imagePaths: string[];
  
  if (includeAll) {
    // 收集所有相关图片
    imagePaths = await getRelevantImagesForNote(app, activeFile, true);
  } else {
    // 只收集笔记中引用的图片
    imagePaths = await getImagesFromNote(app, activeFile);
  }
  
  return formatImagesForAI(imagePaths);
}

