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
    expect(styles).toContain('--product-dialog-content-width: 940px;')
    expect(styles).toContain('--product-dialog-border: rgba(255, 255, 255, 0.72);')
    expect(styles).toContain('border: 2px solid var(--product-dialog-border) !important;')
    expect(styles).toContain('backdrop-filter: blur(28px) saturate(140%);')
    expect(styles).toContain(':has(> :where(')
    expect(styles).toContain('max-height: calc(100dvh - 0.75rem);')
    expect(styles).toContain('font-size: 1rem !important;')
  })

  it('keeps generic edit and detail experiences centered and branded', () => {
    const edit = readSource('./components/EditModal.tsx')
    const editStyles = readSource('./components/EditModal.module.css')
    const view = readSource('./components/ViewPanel.tsx')
    const viewStyles = readSource('./components/ViewModal.module.css')

    expect(edit).toContain('<img src="/vllm.png" alt="" />')
    expect(view).toContain('<img src="/vllm.png" alt="" />')
    expect(edit).toContain('<ProductIcon name="close" />')
    expect(view).toContain('<ProductIcon name="edit" />')
    expect(editStyles).toContain('justify-content: center;')
    expect(viewStyles).toContain('justify-content: center;')
    expect(editStyles).toContain('border: 2px solid var(--product-dialog-border);')
    expect(viewStyles).toContain('border: 2px solid var(--product-dialog-border);')
    expect(editStyles).not.toContain('drawer-in')
    expect(viewStyles).not.toContain('drawer-in')
    expect(viewStyles).not.toContain('drawerShell')
  })

  it('keeps model and Access management dialogs on one responsive content measure', () => {
    const addModelStyles = readSource('./pages/ConfigPageAddModelsDialog.module.css')
    const editStyles = readSource('./components/EditModal.module.css')
    const viewStyles = readSource('./components/ViewModal.module.css')
    const accessStyles = readSource('./pages/AccessControlPage.module.css')
    const modelSection = readSource('./pages/ConfigPageModelsSection.tsx')
    const styles = [addModelStyles, editStyles, viewStyles, accessStyles]

    styles.forEach((source) =>
      expect(source).toContain('width: min(var(--product-dialog-content-width), 100%);'),
    )
    expect(
      accessStyles.match(/width: min\(var\(--product-dialog-content-width\), 100%\);/g),
    ).toHaveLength(3)
    expect(modelSection).toContain('openEditModal<ModelFormState>')
    expect(modelSection).toContain('openViewModal(')
    expect(addModelStyles).toMatch(
      /@media \(max-width: 760px\)[\s\S]*?\.dialog\s*{[\s\S]*?width: 100%;/,
    )
    expect(editStyles).toMatch(/@media \(max-width: 768px\)[\s\S]*?\.modal\s*{[\s\S]*?width: 100%;/)
    expect(viewStyles).toMatch(
      /@media \(max-width: 768px\)[\s\S]*?\.dialogShell\s*{[\s\S]*?width: 100%;/,
    )
  })

  it('uses the same glass material for Builder and DSL import dialogs', () => {
    const buildStyles = readSource('./pages/BuilderPage.module.css')
    const dslStyles = readSource('./pages/DslEditorPage.module.css')

    for (const styles of [buildStyles, dslStyles]) {
      expect(styles).toContain('border: 2px solid var(--product-dialog-border);')
      expect(styles).toContain('background: var(--product-dialog-surface);')
      expect(styles).toContain('box-shadow: var(--product-dialog-shadow);')
      expect(styles).toMatch(
        /@media \(max-width: 640px\)[\s\S]*?\.modal\s*{[\s\S]*?width: 100%;[\s\S]*?max-height: calc\(100dvh - 0\.75rem\);/,
      )
    }
  })

  it('keeps core Build dialogs centered instead of turning them into mobile sheets', () => {
    const addModelsStyles = readSource('./pages/ConfigPageAddModelsDialog.module.css')
    const mixtureStyles = readSource('./pages/ConfigPageMixtureDialog.module.css')
    const buildDialogStyles = [addModelsStyles, mixtureStyles]

    buildDialogStyles.forEach((styles) => {
      const mobileStyles = styles.slice(styles.lastIndexOf('@media (max-width:'))
      expect(mobileStyles).toContain('.backdrop {\n    align-items: center;')
      expect(mobileStyles).not.toContain('.backdrop {\n    align-items: end;')
      expect(mobileStyles).toContain('border-radius: 14px;')
    })
  })

  it('keeps every dashboard dialog in the shared modal contract', () => {
    const sources = collectComponentSources(new URL('./', import.meta.url))
    const dialogSources = sources.filter((source) =>
      /role=["'](?:dialog|alertdialog)["']/.test(source),
    )
    expect(dialogSources.length).toBeGreaterThan(20)
    dialogSources.forEach((source) => expect(source).toContain('aria-modal="true"'))
  })

  it('uses product dialogs and the resilient clipboard path instead of browser-native UI', () => {
    const sources = collectComponentSources(new URL('./', import.meta.url)).join('\n')
    const clipboard = readSource('./utils/clipboard.ts')

    expect(sources).not.toContain('window.confirm(')
    expect(sources).not.toContain('window.alert(')
    expect(sources).not.toContain('window.prompt(')
    expect(sources).not.toContain('navigator.clipboard.writeText(')
    expect(clipboard).toContain('navigator.clipboard?.writeText')
    expect(clipboard).toContain("document.execCommand('copy')")
  })
})
