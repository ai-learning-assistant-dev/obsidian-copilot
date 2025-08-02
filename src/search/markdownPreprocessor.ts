export interface MarkdownSection {
  content: string;
  subtitle: string; // 标题路径，如 "/" 或 "/一级标题" 或 "/一级标题/二级标题"
}

/**
 * 根据markdown标题结构对文档进行预分段
 * @param content markdown文档内容
 * @returns 分段结果数组
 */
export function preprocessMarkdownDocument(content: string): MarkdownSection[] {
  const lines = content.split("\n");
  const sections: MarkdownSection[] = [];

  let currentSection: string[] = [];
  let currentH1: string | null = null;
  let currentH2: string | null = null;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmedLine = line.trim();

    // 检测一级标题
    if (trimmedLine.startsWith("# ") && !trimmedLine.startsWith("## ")) {
      // 保存之前的section
      if (currentSection.length > 0) {
        const subtitle = buildSubtitle(currentH1, currentH2);
        sections.push({
          content: currentSection.join("\n").trim(),
          subtitle,
        });
        currentSection = [];
      }

      // 更新当前标题状态
      currentH1 = trimmedLine.substring(2).trim();
      currentH2 = null;
      currentSection.push(line);
    }
    // 检测二级标题
    else if (trimmedLine.startsWith("## ") && !trimmedLine.startsWith("### ")) {
      // 保存之前的section
      if (currentSection.length > 0) {
        const subtitle = buildSubtitle(currentH1, currentH2);
        sections.push({
          content: currentSection.join("\n").trim(),
          subtitle,
        });
        currentSection = [];
      }

      // 更新当前标题状态
      currentH2 = trimmedLine.substring(3).trim();
      currentSection.push(line);
    }
    // 普通内容
    else {
      currentSection.push(line);
    }
  }

  // 处理最后一个section
  if (currentSection.length > 0) {
    const subtitle = buildSubtitle(currentH1, currentH2);
    sections.push({
      content: currentSection.join("\n").trim(),
      subtitle,
    });
  }

  // 如果没有任何标题，整个文档作为一个section
  if (sections.length === 0 && content.trim()) {
    sections.push({
      content: content.trim(),
      subtitle: "/",
    });
  }

  // 过滤掉空的sections
  return sections.filter((section) => section.content.trim().length > 0);
}

/**
 * 构建标题路径
 * @param h1 一级标题
 * @param h2 二级标题
 * @returns 标题路径字符串
 */
function buildSubtitle(h1: string | null, h2: string | null): string {
  if (!h1) {
    return "/";
  }

  if (!h2) {
    return `/${h1}`;
  }

  return `/${h1}/${h2}`;
}
