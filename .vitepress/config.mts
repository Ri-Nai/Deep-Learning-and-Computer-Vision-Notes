// @ts-nocheck

import { defineConfig } from 'vitepress'
import { MermaidMarkdown, MermaidPlugin } from 'vitepress-plugin-mermaid';
import { generateNavAndSidebar } from './navSidebar'
import { projectConfig } from './projectConfig.mjs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const docsDir = path.join(__dirname, '../docs')
const { nav, sidebar } = generateNavAndSidebar(docsDir)

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: projectConfig.title,
  description: projectConfig.title,
  srcDir: './docs',
  base: `/${projectConfig.projectName}/`,
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    editLink: {
      pattern: `https://github.com/Ri-Nai-BIT-SE/${projectConfig.projectName}/edit/main/docs/:path`,
      text: 'Edit this page on GitHub'
    },
    nav: [
      { text: '首页', link: '/' },
      ...nav,
    ],
    outline: [1, 5],
    sidebar,
    socialLinks: [
      { icon: 'github', link: `https://github.com/Ri-Nai-BIT-SE/${projectConfig.projectName}` }
    ],
  },
  markdown: {
    math: true,
    config(md: any) {
      md.use(MermaidMarkdown);
    },
  },
  vite: {
    plugins: [MermaidPlugin()],
    optimizeDeps: {
      include: ['mermaid'],
    },
    ssr: {
      noExternal: ['mermaid'],
    },
  },
})
