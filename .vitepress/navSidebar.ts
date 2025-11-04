import fs from 'node:fs'
import path from 'node:path'
import type { DefaultTheme } from 'vitepress'

// 类型别名，保持代码清晰
type SidebarItem = DefaultTheme.SidebarItem
type NavItem = DefaultTheme.NavItem
type Sidebar = DefaultTheme.Sidebar

const DOC_EXTENSIONS = ['.md']
// 需要排除的目录
const EXCLUDED_DIRS = new Set(['.vitepress', 'node_modules', 'public'])

// --- 辅助函数 ---

/**
 * 检查路径是否为目录
 */
function isDirectory(p: string): boolean {
  try {
    return fs.statSync(p).isDirectory()
  }
  catch {
    return false
  }
}

/**
 * 检查路径是否为 Markdown 文件
 */
function isMarkdownFile(p: string): boolean {
  try {
    const stat = fs.statSync(p)
    return stat.isFile() && DOC_EXTENSIONS.includes(path.extname(p).toLowerCase())
  }
  catch {
    return false
  }
}

/**
 * 从文件名或目录名中提取干净的标题
 * (例如："1-计算机网络的基本概念" -> "计算机网络的基本概念")
 */
function getCleanTitle(name: string): string {
  return name.replace(/^\d+-/, '').replace(/\.md$/i, '')
}

/**
 * 将常见中文数字字符串转换为阿拉伯数字
 */
function chineseToArabic(str: string): number {
    if (!str) return -1;
    const map: { [key: string]: number } = {
        '零': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
    };
    const unitMap = { '十': 10, '百': 100 };

    // 直接匹配 "十"
    if (str === '十') return 10;
    
    let total = 0;
    let currentNum = 0;

    // 处理 "十三", "二十三" 等情况
    if (str.startsWith('十')) {
        total = 10;
        str = str.substring(1);
    }

    for (const char of str) {
        if (map[char] !== undefined) {
            currentNum = map[char];
        } else if (unitMap[char as keyof typeof unitMap]) {
            total += (currentNum === 0 ? 1 : currentNum) * unitMap[char as keyof typeof unitMap];
            currentNum = 0;
        }
    }
    total += currentNum;
    return total > 0 ? total : -1;
}


/**
 * 【已修正】自定义排序函数，优先级：阿拉伯数字 > 中文数字 > 拼音
 */
function customSort(a: string, b: string): number {
  // 1. 尝试按阿拉伯数字前缀排序
  const numA = parseInt(a.match(/^(\d+)/)?.[1] || '', 10);
  const numB = parseInt(b.match(/^(\d+)/)?.[1] || '', 10);

  if (!isNaN(numA) && !isNaN(numB)) return numA - numB;
  if (!isNaN(numA) && isNaN(numB)) return -1;
  if (isNaN(numA) && !isNaN(numB)) return 1;

  // 2. 尝试按中文数字前缀排序 (例如 "第一章", "第二节")
  //    【关键修正】: 正则表达式现在会忽略 "第" 并捕获后面的数字部分
  const chineseRegex = /^第?([一二三四五六七八九十百]+)/;
  const chineseA = a.match(chineseRegex)?.[1]; // 使用捕获组 1
  const chineseB = b.match(chineseRegex)?.[1];

  const cNumA = chineseToArabic(chineseA || '');
  const cNumB = chineseToArabic(chineseB || '');
  
  if (cNumA !== -1 && cNumB !== -1) return cNumA - cNumB;
  if (cNumA !== -1 && cNumB === -1) return -1;
  if (cNumA === -1 && cNumB !== -1) return 1;

  // 3. 回退到拼音排序
  return a.localeCompare(b, 'zh-Hans-CN-u-co-pinyin');
}

// --- 核心递归函数 ---

/**
 * 递归地从指定目录创建侧边栏项目数组
 * @param rootDocsPath - 文档根目录的绝对路径
 * @param currentDir - 当前正在处理的目录的绝对路径
 * @returns SidebarItem 数组
 */
function createSidebarItems(rootDocsPath: string, currentDir: string): SidebarItem[] {
  const items: SidebarItem[] = []
  const entries = fs.readdirSync(currentDir).sort(customSort)

  for (const entry of entries) {
    // 忽略 index 文件和隐藏文件
    const lowerEntry = entry.toLowerCase()
    if (entry.startsWith('.') || lowerEntry === 'index.md' || lowerEntry === 'readme.md') {
      continue
    }

    const fullPath = path.join(currentDir, entry)
    // 生成标准的 web 路径 (使用 / 并进行 URI 编码)
    const relativePath = `/${path.relative(rootDocsPath, fullPath).replace(/\\/g, '/')}`

    if (isDirectory(fullPath)) {
      // 关键：只有包含 index.md 的子目录才被视为一个可折叠的分组
      const indexFile = ['index.md', 'README.md'].find(f => fs.existsSync(path.join(fullPath, f)))
      if (indexFile) {
        items.push({
          text: getCleanTitle(entry),
          // VitePress 会自动将 /folder/ 解析为 /folder/index.md
          link: `${relativePath}/`,
          items: createSidebarItems(rootDocsPath, fullPath), // <-- 递归调用
          collapsed: true, // 默认折叠，体验更好
        })
      }
    }
    else if (isMarkdownFile(fullPath)) {
      items.push({
        text: getCleanTitle(entry),
        link: relativePath,
      })
    }
  }

  return items
}


// --- 主函数 ---

/**
 * 通过扫描文档目录，生成 VitePress 的 Nav 和 Sidebar 配置
 * @param docsDir - 文档目录相对于项目根目录的路径 (例如: 'docs')
 */
export function generateNavAndSidebar(docsDir: string) {
  const rootDocsPath = path.resolve(process.cwd(), docsDir)
  const nav: NavItem[] = []
  const sidebar: Sidebar = {}

  // 读取并筛选顶层目录
  const topLevelDirs = fs
    .readdirSync(rootDocsPath)
    .filter(dir => {
      const fullPath = path.join(rootDocsPath, dir)
      return isDirectory(fullPath) && !EXCLUDED_DIRS.has(dir) && !dir.startsWith('.')
    })
    .sort(customSort)

  for (const dir of topLevelDirs) {
    const topLevelDirPath = path.join(rootDocsPath, dir)
    const link = `/${dir}/`

    // 创建导航项
    nav.push({
      text: getCleanTitle(dir),
      link: link,
    })

    // 创建侧边栏
    // 这是关键的修正：将生成的 items 数组包装在顶层对象中
    sidebar[link] = [
      {
        text: getCleanTitle(dir), // 侧边栏分组的大标题
        items: createSidebarItems(rootDocsPath, topLevelDirPath),
      },
    ]
  }

  return { nav, sidebar }
}