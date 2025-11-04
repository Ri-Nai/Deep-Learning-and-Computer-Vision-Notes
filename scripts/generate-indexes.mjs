import fs from 'node:fs'
import path from 'node:path'

const ROOT = path.join(process.cwd(), 'docs')
const EXCLUDED_DIRS = new Set()

function isDirectory(p) {
  return fs.existsSync(p) && fs.statSync(p).isDirectory()
}

function isMarkdown(p) {
  return fs.existsSync(p) && fs.statSync(p).isFile() && path.extname(p).toLowerCase() === '.md'
}

function sortByPinyinOrName(a, b) {
  return a.localeCompare(b, 'zh-Hans-CN-u-co-pinyin')
}

function titleFromName(name) {
  return name.replace(/\.md$/i, '')
}

function buildIndexContent(dirName, items) {
  const header = `# ${dirName}\n\n<!-- AUTO-GENERATED: index for ${dirName}. Edit source files instead. -->\n\n`
  if (items.length === 0) return header + '（暂无条目）\n'
  const list = items
    .sort((a, b) => sortByPinyinOrName(a.name, b.name))
    .map((item) => {
      if (item.isDir) {
        // 文件夹链接到其 index.md
        return `- [${item.name}](./${encodeURI(item.name)}/)`
      } else {
        // Markdown 文件
        return `- [${titleFromName(item.name)}](./${encodeURI(item.name)})`
      }
    })
    .join('\n')
  return header + list + '\n'
}

function shouldOverwriteExisting(indexPath) {
  if (!fs.existsSync(indexPath)) return true
  const content = fs.readFileSync(indexPath, 'utf8')
  // 仅覆盖带有标记的自动生成文件，避免覆盖人工维护的索引
  return content.includes('AUTO-GENERATED: index')
}

function main() {
  const entries = fs.readdirSync(ROOT)
  const dirs = entries
    .filter((e) => isDirectory(path.join(ROOT, e)))
    .filter((e) => !EXCLUDED_DIRS.has(e) && !e.startsWith('.'))
    .sort(sortByPinyinOrName)

  let changed = 0
  for (const dir of dirs) {
    const abs = path.join(ROOT, dir)
    const entries = fs.readdirSync(abs)
    
    // 收集 Markdown 文件和子文件夹
    const items = []
    
    for (const entry of entries) {
      const entryPath = path.join(abs, entry)
      const lowerEntry = entry.toLowerCase()
      
      // 跳过 index.md 和 readme.md
      if (lowerEntry === 'index.md' || lowerEntry === 'readme.md') {
        continue
      }
      
      if (isDirectory(entryPath)) {
        // 只包含包含 index.md 的子文件夹
        if (fs.existsSync(path.join(entryPath, 'index.md'))) {
          items.push({ name: entry, isDir: true })
        }
      } else if (isMarkdown(entryPath)) {
        items.push({ name: entry, isDir: false })
      }
    }

    const indexPath = path.join(abs, 'index.md')
    if (!shouldOverwriteExisting(indexPath)) continue

    const content = buildIndexContent(dir, items)
    fs.writeFileSync(indexPath, content, 'utf8')
    changed++
  }
  console.log(`[generate-indexes] updated ${changed} index file(s).`)
}

main()
