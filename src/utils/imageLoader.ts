/**
 * 图片加载工具
 * 从 Obsidian vault 中读取图片并转换为可显示的格式
 */

import { App, TFile, TAbstractFile } from "obsidian";

/** 支持的图片文件扩展名 */
export const SUPPORTED_IMAGE_EXTENSIONS = [
  'png', 'jpg', 'jpeg', 'gif', 'webp', 'svg', 'bmp', 'tiff'
];

/** 图片加载结果 */
export interface ImageLoadResult {
  /** 是否成功加载 */
  success: boolean;
  /** Data URL 格式的图片数据 (data:image/png;base64,...) */
  dataUrl?: string;
  /** 错误信息（如果加载失败） */
  error?: string;
  /** 原始文件路径 */
  originalPath: string;
  /** 解析后的文件路径（实际找到的文件） */
  resolvedPath?: string;
}

/**
 * 从路径中提取文件扩展名
 */
function getFileExtension(path: string): string {
  const parts = path.split('.');
  return parts.length > 1 ? parts[parts.length - 1].toLowerCase() : '';
}

/**
 * 根据文件扩展名获取 MIME 类型
 */
function getMimeType(extension: string): string {
  const mimeTypes: Record<string, string> = {
    'png': 'image/png',
    'jpg': 'image/jpeg',
    'jpeg': 'image/jpeg',
    'gif': 'image/gif',
    'webp': 'image/webp',
    'svg': 'image/svg+xml',
    'bmp': 'image/bmp',
    'tiff': 'image/tiff',
  };
  
  return mimeTypes[extension.toLowerCase()] || 'image/png';
}

/**
 * 检查文件扩展名是否为支持的图片格式
 * @param extension - 文件扩展名（可以是 "png" 或 "image.png" 格式）
 */
export function isSupportedImageFile(extension: string): boolean {
  // 如果包含点号，说明是完整路径，需要提取扩展名
  // 如果不包含点号，说明已经是扩展名了
  const ext = extension.includes('.') 
    ? getFileExtension(extension) 
    : extension.toLowerCase();
  return SUPPORTED_IMAGE_EXTENSIONS.includes(ext);
}

/**
 * 在 vault 中查找图片文件
 * 支持多种查找策略：
 * 1. 直接路径匹配
 * 2. 文件名匹配（在整个 vault 中搜索）
 * 3. 相对路径匹配
 * 
 * @param app - Obsidian App 实例
 * @param imagePath - 图片路径（可能是完整路径或仅文件名）
 * @param contextFilePath - 上下文文件路径（用于解析相对路径）
 * @returns 找到的文件或 null
 */
export function findImageFile(
  app: App,
  imagePath: string,
  contextFilePath?: string
): TFile | null {
  // 清理路径（去除多余空格）
  const cleanPath = imagePath.trim();
  
  // 策略 1: 直接通过完整路径查找
  let file = app.vault.getAbstractFileByPath(cleanPath);
  if (file instanceof TFile && isSupportedImageFile(file.extension)) {
    return file;
  }
  
  // 策略 2: 如果只是文件名，在整个 vault 中搜索
  if (!cleanPath.includes('/')) {
    const allFiles = app.vault.getFiles();
    const matchingFiles = allFiles.filter(f => 
      f.name === cleanPath && isSupportedImageFile(f.extension)
    );
    
    if (matchingFiles.length > 0) {
      // 如果有多个同名文件，优先返回与上下文文件在同一目录的
      if (contextFilePath && matchingFiles.length > 1) {
        const contextDir = contextFilePath.substring(0, contextFilePath.lastIndexOf('/'));
        const sameDir = matchingFiles.find(f => f.path.startsWith(contextDir));
        if (sameDir) return sameDir;
      }
      return matchingFiles[0];
    }
  }
  
  // 策略 3: 尝试解析为相对于上下文文件的路径
  if (contextFilePath) {
    const contextDir = contextFilePath.substring(0, contextFilePath.lastIndexOf('/'));
    const relativePath = contextDir ? `${contextDir}/${cleanPath}` : cleanPath;
    file = app.vault.getAbstractFileByPath(relativePath);
    if (file instanceof TFile && isSupportedImageFile(file.extension)) {
      return file;
    }
  }
  
  // 策略 4: 尝试通过 Obsidian 的链接解析器（处理 wiki-link 风格）
  const metadataCache = app.metadataCache;
  if (metadataCache) {
    const fileName = cleanPath.split('/').pop() || cleanPath;
    const resolved = metadataCache.getFirstLinkpathDest(fileName, contextFilePath || '');
    if (resolved instanceof TFile && isSupportedImageFile(resolved.extension)) {
      return resolved;
    }
  }
  
  return null;
}

/**
 * 从 vault 加载图片并转换为 Data URL
 * 
 * @param app - Obsidian App 实例
 * @param imagePath - 图片路径
 * @param contextFilePath - 上下文文件路径（可选，用于解析相对路径）
 * @returns 图片加载结果
 * 
 * @example
 * const result = await loadVaultImage(app, "screenshots/demo.png");
 * if (result.success) {
 *   // 使用 result.dataUrl 显示图片
 * }
 */
export async function loadVaultImage(
  app: App,
  imagePath: string,
  contextFilePath?: string
): Promise<ImageLoadResult> {
  try {
    // 验证路径不为空
    if (!imagePath || imagePath.trim() === '') {
      return {
        success: false,
        error: 'Image path is empty',
        originalPath: imagePath,
      };
    }
    
    // 查找图片文件
    const imageFile = findImageFile(app, imagePath, contextFilePath);
    
    if (!imageFile) {
      return {
        success: false,
        error: `Image file not found: ${imagePath}`,
        originalPath: imagePath,
      };
    }
    
    // 检查文件扩展名
    if (!isSupportedImageFile(imageFile.extension)) {
      return {
        success: false,
        error: `Unsupported image format: ${imageFile.extension}`,
        originalPath: imagePath,
        resolvedPath: imageFile.path,
      };
    }
    
    // 读取图片二进制数据
    const arrayBuffer = await app.vault.readBinary(imageFile);
    
    // 转换为 base64
    const bytes = new Uint8Array(arrayBuffer);
    let binary = '';
    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    const base64 = btoa(binary);
    
    // 获取 MIME 类型并构造 Data URL
    const mimeType = getMimeType(imageFile.extension);
    const dataUrl = `data:${mimeType};base64,${base64}`;
    
    return {
      success: true,
      dataUrl: dataUrl,
      originalPath: imagePath,
      resolvedPath: imageFile.path,
    };
    
  } catch (error) {
    console.error(`Error loading image ${imagePath}:`, error);
    return {
      success: false,
      error: error instanceof Error ? error.message : String(error),
      originalPath: imagePath,
    };
  }
}

/**
 * 批量加载多个图片
 * 
 * @param app - Obsidian App 实例
 * @param imagePaths - 图片路径数组
 * @param contextFilePath - 上下文文件路径（可选）
 * @returns 图片加载结果数组
 */
export async function loadMultipleVaultImages(
  app: App,
  imagePaths: string[],
  contextFilePath?: string
): Promise<ImageLoadResult[]> {
  const loadPromises = imagePaths.map(path => 
    loadVaultImage(app, path, contextFilePath)
  );
  
  return Promise.all(loadPromises);
}

/**
 * 图片缓存管理器（简单实现）
 * 避免重复加载同一图片
 */
export class ImageCache {
  private cache: Map<string, ImageLoadResult> = new Map();
  private maxSize: number;
  
  constructor(maxSize: number = 50) {
    this.maxSize = maxSize;
  }
  
  /**
   * 获取缓存的图片
   */
  get(path: string): ImageLoadResult | undefined {
    return this.cache.get(path);
  }
  
  /**
   * 设置缓存
   */
  set(path: string, result: ImageLoadResult): void {
    // 如果缓存已满，删除最早的条目
    if (this.cache.size >= this.maxSize) {
      const firstKey = this.cache.keys().next().value;
      if (firstKey) {
        this.cache.delete(firstKey);
      }
    }
    
    this.cache.set(path, result);
  }
  
  /**
   * 清空缓存
   */
  clear(): void {
    this.cache.clear();
  }
  
  /**
   * 获取缓存大小
   */
  get size(): number {
    return this.cache.size;
  }
}

/** 全局图片缓存实例 */
export const globalImageCache = new ImageCache(100);

/**
 * 加载图片（带缓存）
 * 
 * @param app - Obsidian App 实例
 * @param imagePath - 图片路径
 * @param contextFilePath - 上下文文件路径（可选）
 * @param useCache - 是否使用缓存（默认 true）
 * @returns 图片加载结果
 */
export async function loadVaultImageWithCache(
  app: App,
  imagePath: string,
  contextFilePath?: string,
  useCache: boolean = true
): Promise<ImageLoadResult> {
  // 构造缓存键（包含上下文路径以处理相对路径）
  const cacheKey = contextFilePath ? `${contextFilePath}:${imagePath}` : imagePath;
  
  // 检查缓存
  if (useCache) {
    const cached = globalImageCache.get(cacheKey);
    if (cached) {
      return cached;
    }
  }
  
  // 加载图片
  const result = await loadVaultImage(app, imagePath, contextFilePath);
  
  // 只缓存成功的结果
  if (useCache && result.success) {
    globalImageCache.set(cacheKey, result);
  }
  
  return result;
}

