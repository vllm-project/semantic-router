import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const readSource = (name: string) => readFileSync(new URL(name, import.meta.url), 'utf8')

describe('access-control modal experience', () => {
  it('uses the same glass composition for welcome and generated-key dialogs', () => {
    const inviteStyles = readSource('./InviteCompletionDialog.module.css')
    const accessStyles = readSource('./AccessControlPage.module.css')

    expect(inviteStyles).toContain('width: min(560px, 100%);')
    expect(inviteStyles).toContain('border: 2px solid rgba(255, 255, 255, 0.72);')
    expect(inviteStyles).toContain('backdrop-filter: blur(28px) saturate(140%);')
    expect(accessStyles).toContain('.secretModal {')
    expect(accessStyles).toContain('height: 442px;')
    expect(accessStyles).toContain('.secretActions {')
  })

  it('keeps the generated key connected to its detail view', () => {
    const dialog = readSource('./AccessControlDialog.tsx')
    const page = readSource('./AccessControlPage.tsx')

    expect(dialog).toContain('onViewDetails: () => void')
    expect(dialog).toContain('View details')
    expect(page).toContain("openDetail('key', keyID)")
  })

  it('uses a centered vLLM entity dialog with an explicit Team member roster', () => {
    const detail = readSource('./AccessEntityDetail.tsx')
    const styles = readSource('./AccessControlPage.module.css')

    expect(detail).toContain('className={styles.entityDetailDialog}')
    expect(detail).not.toContain('className={styles.detailDrawer}')
    expect(detail).toContain('<img src="/vllm.png" alt="" />')
    expect(detail).toContain('className={styles.teamMemberList}')
    expect(detail).toContain('setConfirmingDelete(true)')
    expect(styles).toContain('.entityDetailBackdrop {')
    expect(styles).toContain('justify-content: center;')
  })

  it('returns from entity editing to the same detail dialog', () => {
    const page = readSource('./AccessControlPage.tsx')

    expect(page).toContain('setEntityEditorReturn({ kind, id: item.id })')
    expect(page).toContain('?item=${encodeURIComponent(returnTarget.id)}')
  })
})
