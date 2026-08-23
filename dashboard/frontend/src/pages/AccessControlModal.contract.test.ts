import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const readSource = (name: string) => readFileSync(new URL(name, import.meta.url), 'utf8')

describe('access-control modal experience', () => {
  it('uses the glass composition for generated-key dialogs', () => {
    const dialog = readSource('./AccessControlDialog.tsx')
    const welcome = readSource('../components/InvitationWelcomeDialog.tsx')
    const momentStyles = readSource('../components/ProductMomentDialog.module.css')

    expect(dialog).toContain('<ProductMomentDialog')
    expect(welcome).toContain('<ProductMomentDialog')
    expect(momentStyles).toContain('height: min(442px, calc(100dvh - 2.8rem));')
    expect(momentStyles).toContain('border: 2px solid rgba(255, 255, 255, 0.76);')
  })

  it('keeps the generated key connected to its detail view', () => {
    const dialog = readSource('./AccessControlDialog.tsx')
    const page = readSource('./AccessControlPage.tsx')

    expect(dialog).toContain('onViewDetails: () => void')
    expect(dialog).toContain('View details')
    expect(page).toContain("openDetail('key', keyID)")
    expect(page).toContain('onboardingKey')
    expect(page).toContain('state: null')
  })

  it('keeps the generated invitation result focused on the vLLM mark', () => {
    const dialog = readSource('./DashboardMemberInviteDialog.tsx')

    expect(dialog).toContain('Invitation ready')
    expect(dialog).toContain('<img src="/vllm.png" alt="" />')
    expect(dialog).toContain("copyStatus === 'copied' ? 'check' : 'copy'")
    expect(dialog).toContain('<ProductIcon name="inbox" />')
    expect(dialog).not.toContain('styles.inviteAvatar')
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

  it('uses plain-language key lifecycle actions and shows live quota capacity', () => {
    const detail = readSource('./APIKeyDetail.tsx')

    expect(detail).toContain("'Disable'")
    expect(detail).toContain("'Enable'")
    expect(detail).toContain('Renew key')
    expect(detail).toContain('Delete key')
    expect(detail).toContain('key.quota.meters.map')
    expect(detail).toContain('formatQuotaValue')
    expect(detail).toContain('formatCosts(usage.costs)')
    expect(detail).toContain('Usage unavailable')
    expect(detail).toContain('Edit access & quota')
    expect(detail).not.toContain('Rotate key')
  })

  it('refreshes an open key quota without replacing the dialog with a loading state', () => {
    const detail = readSource('./APIKeyDetail.tsx')

    expect(detail).toContain('const KEY_QUOTA_REFRESH_MS = 5000')
    expect(detail).toContain('window.setInterval(() => void refreshQuota(), KEY_QUOTA_REFRESH_MS)')
    expect(detail).toContain("document.addEventListener('visibilitychange', refreshWhenVisible)")
    expect(detail).toContain('if (document.hidden || inFlight) return')
    expect(detail).toContain('if (!cancelled) setKey(next)')
  })

  it('keeps the key quota override visible without opening advanced settings', () => {
    const fields = readSource('./AccessControlEditorFields.tsx')

    const quotaIndex = fields.indexOf('inheritBudgetLabel={inheritBudgetLabel}')
    const advancedIndex = fields.indexOf('<Advanced label="Advanced settings">', quotaIndex)

    expect(quotaIndex).toBeGreaterThan(-1)
    expect(quotaIndex).toBeLessThan(advancedIndex)
    expect(fields).toContain("'Inherit the owner’s effective quota'")
    expect(fields).toContain('User')
    expect(fields).toContain('Team')
  })

  it('keeps table pagination independent from bounded form selectors', () => {
    const page = readSource('./AccessControlPage.tsx')
    const fields = readSource('./AccessControlEditorFields.tsx')
    const invitation = readSource('./DashboardMemberInviteDialog.tsx')
    const picker = readSource('./AccessAsyncResourcePicker.tsx')
    const usage = readSource('./AccessControlUsageView.tsx')

    expect(page).toContain('accessPageQuery(pageState, pageCursors[activeView]?.[pageState.page])')
    expect(page).toContain('selectors={accessControlSelectorSources}')
    expect(page).not.toContain('inferenceAccessApi.users({ limit: 100 })')
    expect(page).not.toContain('inferenceAccessApi.teams({ limit: 100 })')
    expect(fields).toContain('source={selectors.users}')
    expect(fields).toContain('source={selectors.teams}')
    expect(fields).toContain('source={selectors.groups}')
    expect(fields).toContain('source={selectors.budgets}')
    expect(fields).toContain('source={selectors.entrypoints}')
    expect(fields).toContain('source={selectors.models}')
    expect(page).not.toContain('routingManagementApi.listModels()')
    expect(page).not.toContain('routingManagementApi.listEntrypoints()')
    expect(invitation).toContain('source={teamSource}')
    expect(picker).toContain('source.detail(id)')
    expect(picker).toContain('load(nextCursor)')
    expect(usage).toContain('source={props.selectors.keys}')
    expect(usage).toContain('selectors.users.detail(id)')
    expect(usage).toContain('selectors.teams.detail(id)')
    expect(usage).toContain('selectors.keys.detail(id)')
    expect(usage).toContain('.slice(0, 100)')
    expect(usage).not.toContain('props.users.find')
  })

  it('exposes pending state on long-running access dialogs', () => {
    const editor = readSource('./AccessControlDialog.tsx')
    const keyDetail = readSource('./APIKeyDetail.tsx')
    const logDetail = readSource('./RequestLogDetail.tsx')

    expect(editor).toContain('aria-busy={saving}')
    expect(keyDetail).toContain('aria-busy={loading || pending}')
    expect(logDetail).toContain('aria-busy={loading}')
  })
})
