/**
 * 图片引用解析器
 * 用于识别和解析消息中的图片引用语法
 */

export interface ImageReference {
  /** 图片引用类型 */
  type: 'obsidian' | 'markdown';
  /** 图片路径（可能包含子文件夹） */
  path: string;
  /** 图片的 alt 文本（仅 markdown 格式） */
  alt?: string;
  /** 在原始文本中的起始位置 */
  startIndex: number;
  /** 在原始文本中的结束位置 */
  endIndex: number;
  /** 原始匹配文本（用于替换） */
  rawText: string;
}

/**
 * 解析 Obsidian 图片语法：![[image.png]] 或 ![[folder/image.png]]
 * 
 * @param text - 要解析的文本
 * @returns 找到的所有图片引用
 * 
 * @example
 * parseObsidianImageSyntax("看这张图 ![[test.png]] 和这张 ![[folder/pic.jpg]]")
 * // 返回两个 ImageReference 对象
 */
export function parseObsidianImageSyntax(text: string): ImageReference[] {
  const references: ImageReference[] = [];
  
  // 匹配 ![[xxx]] 格式，支持路径中的斜杠和各种字符
  // 使用非贪婪匹配，避免跨多个图片引用
  const obsidianRegex = /!\[\[([^\]]+?\.(png|jpg|jpeg|gif|webp|svg|bmp|tiff))\]\]/gi;
  
  let match: RegExpExecArray | null;
  
  while ((match = obsidianRegex.exec(text)) !== null) {
    const rawText = match[0]; // 完整的 ![[...]]
    const path = match[1].trim(); // 提取路径部分
    
    references.push({
      type: 'obsidian',
      path: path,
      startIndex: match.index,
      endIndex: match.index + rawText.length,
      rawText: rawText,
    });
  }
  
  return references;
}

/**
 * 解析 Markdown 标准图片语法：![alt](path/to/image.png)
 * 
 * @param text - 要解析的文本
 * @returns 找到的所有图片引用
 * 
 * @example
 * parseMarkdownImageSyntax("图片：![示例](images/test.png)")
 * // 返回一个 ImageReference 对象
 */
export function parseMarkdownImageSyntax(text: string): ImageReference[] {
  const references: ImageReference[] = [];
  
  // 匹配 ![alt](path) 格式
  const markdownRegex = /!\[([^\]]*)\]\(([^)]+?\.(png|jpg|jpeg|gif|webp|svg|bmp|tiff))\)/gi;
  
  let match: RegExpExecArray | null;
  
  while ((match = markdownRegex.exec(text)) !== null) {
    const rawText = match[0]; // 完整的 ![...](...) 
    const alt = match[1].trim(); // alt 文本
    const path = match[2].trim(); // 图片路径
    
    references.push({
      type: 'markdown',
      path: path,
      alt: alt || undefined,
      startIndex: match.index,
      endIndex: match.index + rawText.length,
      rawText: rawText,
    });
  }
  
  return references;
}

/**
 * 解析所有支持的图片语法（Obsidian + Markdown）
 * 
 * @param text - 要解析的文本
 * @returns 找到的所有图片引用，按出现顺序排序
 * 
 * @example
 * parseAllImageSyntax("有 ![[obs.png]] 和 ![md](test.jpg)")
 * // 返回两个 ImageReference 对象，按位置排序
 */
export function parseAllImageSyntax(text: string): ImageReference[] {
  const obsidianRefs = parseObsidianImageSyntax(text);
  const markdownRefs = parseMarkdownImageSyntax(text);
  
  // 合并并按位置排序
  const allRefs = [...obsidianRefs, ...markdownRefs];
  allRefs.sort((a, b) => a.startIndex - b.startIndex);
  
  return allRefs;
}

/**
 * 检查文本是否包含图片引用
 * 
 * @param text - 要检查的文本
 * @returns 是否包含图片引用
 */
export function hasImageReferences(text: string): boolean {
  return parseAllImageSyntax(text).length > 0;
}

/**
 * 从路径中提取文件名（不含路径）
 * 
 * @param path - 文件路径
 * @returns 文件名
 * 
 * @example
 * getFileNameFromPath("folder/subfolder/image.png") // "image.png"
 * getFileNameFromPath("image.png") // "image.png"
 */
export function getFileNameFromPath(path: string): string {
  const parts = path.split('/');
  return parts[parts.length - 1];
}

