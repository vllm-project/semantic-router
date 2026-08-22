import { readdirSync, readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const readSource = (name: string) => readFileSync(new URL(name, import.meta.url), 'utf8')

const collectComponentSources = (directory: URL): string[] =>
  readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const child = new URL(entry.name + (entry.isDirectory() ? '/' : ''), directory)
    if (entry.isDirectory()) return collectComponentSources(child)
    if (!entry.name.endsWith('.tsx') || entry.name.endsWith('.test.tsx')) return []
    return [readFileSync(child, 'utf8')]
  })

describe('dashboard product dialog system', () => {
  it('applies one material contract to dialog and alertdialog surfaces', () => {
    const main = readSource('./main.tsx')
    const styles = readSource('./productDialog.css')

    expect(main).toContain("import './productDialog.css'")
    expect(styles).toContain("[role='dialog'], [role='alertdialog']")
    expect(styles).toContain('border: 2px solid rgba(255, 255, 255, 0.72) !important;')
    expect(styles).toContain('backdrop-filter: blur(28px) saturate(140%);')
    expect(styles).toContain(':has(> :where(')
  })

  it('keeps generic edit and detail experiences centered and branded', () => {
    const edit = readSource('./components/EditModal.tsx')
    const editStyles = readSource('./components/EditModal.module.css')
    const view = readSource('./components/ViewPanel.tsx')
    const viewStyles = readSource('./components/ViewModal.module.css')

    expect(edit).toContain('<img src="/vllm.png" alt="" />')
    expect(view).toContain('<img src="/vllm.png" alt="" />')
    expect(editStyles).toContain('justify-content: center;')
    expect(viewStyles).toContain('justify-content: center;')
    expect(editStyles).not.toContain('drawer-in')
    expect(viewStyles).not.toContain('drawer-in')
  })

  it('keeps every dashboard dialog in the shared modal contract', () => {
    const sources = collectComponentSources(new URL('./', import.meta.url))
    const dialogSources = sources.filter((source) =>
      /role=["'](?:dialog|alertdialog)["']/.test(source),
    )

    expect(dialogSources.length).toBeGreaterThan(20)
    dialogSources.forEach((source) => expect(source).toContain('aria-modal="true"'))
  })
})
