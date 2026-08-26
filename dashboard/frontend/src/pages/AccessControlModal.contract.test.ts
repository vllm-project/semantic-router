import { readFileSync } from 'node:fs'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { AccessGroupResourceTags } from './AccessEntityDetailSupport'

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
    const overlays = readSource('./AccessControlPageOverlays.tsx')

    expect(dialog).toContain('onViewDetails: () => void')
    expect(dialog).toContain('View details')
    expect(page).toContain("openDetail('key', keyId)")
    expect(overlays).toContain('onCreatedKeyDetails(createdKey.id)')
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

  it('uses the centered detail dialog for every Access detail surface', () => {
    const details = [
      readSource('./APIKeyDetail.tsx'),
      readSource('./DashboardMemberDetail.tsx'),
      readSource('./RequestLogDetail.tsx'),
    ].join('\n')
    const styles = readSource('./AccessControlPage.module.css')

    expect(details.match(/className=\{styles\.detailDialog\}/g)).toHaveLength(3)
    expect(details).not.toContain('detailDrawer')
    expect(styles).toContain('.detailDialog {')
    expect(styles).not.toContain('.detailDrawer {')
  })

  it('gives every Access editor and detail the shared wide dialog measure', () => {
    const styles = readSource('./AccessControlPage.module.css')

    expect(
      styles.match(/width: min\(var\(--product-dialog-content-width\), 100%\);/g),
    ).toHaveLength(3)
    expect(styles).toMatch(/@media \(max-width: 760px\)[\s\S]*?\.modal\s*{[\s\S]*?width: 100%;/)
    expect(styles).toMatch(
      /@media \(max-width: 760px\)[\s\S]*?\.detailDialog\s*{[\s\S]*?width: 100%;/,
    )
    expect(styles).toMatch(
      /@media \(max-width: 760px\)[\s\S]*?\.entityDetailDialog\s*{[\s\S]*?width: 100%;/,
    )
  })

  it('presents invitation Team search as one integrated picker surface', () => {
    const dialog = readSource('./DashboardMemberInviteDialog.tsx')
    const picker = readSource('./AccessAsyncResourcePicker.tsx')
    const styles = readSource('./AccessControlPage.module.css')

    expect(dialog).toContain('inlineCompactMenu')
    expect(dialog).toContain('placeholder="Search Team name"')
    expect(picker).toContain('styles.asyncPickerInlineExpanded')
    expect(picker.match(/type="search"/g)).toHaveLength(1)
    expect(styles).toMatch(/\.asyncPickerInlineExpanded \.asyncPickerSearch\s*{[\s\S]*?border: 0;/)
    expect(styles).toMatch(/\.asyncPickerInlineExpanded \.asyncPickerMenu\s*{[\s\S]*?border: 0;/)
  })

  it('returns from entity editing to the same detail dialog', () => {
    const page = readSource('./AccessControlPage.tsx')

    expect(page).toContain('setEntityEditorReturn({ kind, id: item.id })')
    expect(page).toContain('?item=${encodeURIComponent(returnTarget.id)}')
  })

  it('keeps entity deletion pending and failure state inside the open dialog', () => {
    const detail = readSource('./AccessEntityDetail.tsx')
    const page = readSource('./AccessControlPage.tsx')
    const directory = readSource('./useAccessControlDirectory.ts')
    const viewData = readSource('./useAccessControlViewData.ts')
    const tombstones = readSource('./accessEntityDeletionState.ts')

    expect(detail).toContain('const [deletePending, setDeletePending]')
    expect(detail).toContain('const [deleteError, setDeleteError]')
    expect(detail).toContain('await onDelete(kind, item.id)')
    expect(detail).toContain('aria-busy={deletePending}')
    expect(detail).toContain("deletePending ? 'Deleting…' : 'Delete'")
    expect(page).toContain('removeLocalEntity(kind, id)')
    expect(page).toContain('refreshAfterDelete()')
    expect(page).toContain('rememberDeletedAccessEntity(deletionTombstonesRef.current, kind, id)')
    expect(directory).toContain('omitDeletedAccessEntities(')
    expect(viewData).toContain('generation === viewLoadGenerationRef.current')
    expect(tombstones).toContain('dashboard-member')
    expect(tombstones).toContain("| 'key'")
    expect(page).toContain(
      "rememberDeletedAccessEntity(deletionTombstonesRef.current, 'key', keyId)",
    )
  })

  it('separates Dashboard login removal from coordinated user deletion', () => {
    const detail = readSource('./DashboardMemberDetail.tsx')
    const overlays = readSource('./AccessControlDetailOverlays.tsx')
    const page = readSource('./AccessControlPage.tsx')

    expect(detail).toContain('Remove login')
    expect(detail).toContain('Delete user')
    expect(detail).toContain('canManageModelUser')
    expect(detail).toContain('canDeleteUnifiedUser')
    expect(detail).toContain('UnifiedUserDeletionError')
    expect(detail).toContain('findLinkedModelUser(member, users)')
    expect(page).toContain('error instanceof ManagementApiError && error.status === 404')
    expect(overlays).toContain('canManage && canManageDashboardMembers')
    expect(page).toContain('deleteUnifiedUser(progress')
    expect(page).toContain("removeLocalEntity('user', modelUserId)")
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
    expect(detail).toContain('quotaCapacityLabel(meter)')
    expect(detail).toContain("'No requests'")
    expect(detail).toContain("'Finalizing recent requests'")
    expect(detail).not.toContain('Usage unavailable')
    expect(detail).toContain('Edit access & quota')
    expect(detail).toContain('disabled={!snippets}')
    expect(detail).toContain('No request-ready model is available for this key.')
    expect(detail).toContain('apiKeyVisibleResourceNames(resources, resourceResolutions)')
    expect(detail).toContain('apiKeyQuickstartModel(resources, resourceResolutions)')
    expect(detail).toContain("? 'Names unavailable'")
    expect(detail).not.toContain("key.accessGroupIds.join(', ')")
    expect(detail).not.toContain('YOUR_MODEL')
    expect(detail).not.toContain('Rotate key')
    expect(detail.match(/<dt>Owner<\/dt>/g)).toHaveLength(1)
  })

  it('aligns invitation Team controls to one shared 40px measure', () => {
    const styles = readSource('./AccessControlPage.module.css')

    expect(styles).toMatch(/\.asyncPickerSearch\s*{[\s\S]*?height: 40px;[\s\S]*?min-height: 40px;/)
    expect(styles).toMatch(
      /\.asyncPickerCompactValue\s*{[\s\S]*?height: 40px;[\s\S]*?min-height: 40px;/,
    )
    expect(styles).toMatch(
      /\.asyncPickerInlineExpanded \.asyncPickerSearch\s*{[\s\S]*?height: 40px;[\s\S]*?min-height: 40px;/,
    )
  })

  it('loads a key snapshot once without background polling or auth-object churn', () => {
    const detail = readSource('./APIKeyDetail.tsx')

    expect(detail).toContain(
      'const includeInternalUsageDimensions = canReadInternalUsageDimensions(currentUser)',
    )
    expect(detail).toContain('[includeInternalUsageDimensions, keyId, selfService]')
    expect(detail).not.toContain('KEY_QUOTA_REFRESH_MS')
    expect(detail).not.toContain('window.setInterval')
    expect(detail).not.toContain("document.addEventListener('visibilitychange'")
    expect(detail).not.toContain('[currentUser, keyId, selfService]')
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

  it('keeps identity, model access, and quota in the visible editing hierarchy', () => {
    const fields = readSource('./AccessControlEditorFields.tsx')
    const primitives = readSource('./AccessControlEditorPrimitives.tsx')
    const userFields = fields.slice(
      fields.indexOf('function UserFields'),
      fields.indexOf('function TeamFields'),
    )
    const teamFields = fields.slice(
      fields.indexOf('function TeamFields'),
      fields.indexOf('function KeyFields'),
    )
    const keyFields = fields.slice(
      fields.indexOf('function KeyFields'),
      fields.indexOf('function GroupFields'),
    )
    const groupFields = fields.slice(
      fields.indexOf('function GroupFields'),
      fields.indexOf('function BudgetFields'),
    )
    const budgetFields = fields.slice(
      fields.indexOf('function BudgetFields'),
      fields.indexOf('function PolicyFields'),
    )
    const policyFields = fields.slice(fields.indexOf('function PolicyFields'))

    expect(primitives).toContain('data-access-section="core"')
    expect(policyFields).toContain('title="Model access"')
    expect(policyFields).toContain('title="Quota"')
    expect(policyFields).not.toContain('compact')
    expect(userFields.indexOf('<PolicyFields')).toBeLessThan(userFields.indexOf('<Advanced>'))
    expect(teamFields).toContain('<CoreSection title="Members"')
    expect(teamFields.indexOf('<CoreSection title="Members"')).toBeLessThan(
      teamFields.indexOf('<PolicyFields'),
    )
    expect(teamFields).not.toContain('compact')
    expect(keyFields.indexOf('label="Key override"')).toBeLessThan(
      keyFields.indexOf('<Advanced label="Advanced settings">'),
    )
    expect(groupFields).toContain('<CoreSection title="Mixture-of-Models"')
    expect(groupFields).toContain('<CoreSection title="Single Model"')
    expect(groupFields).not.toContain('<Advanced')
    expect(budgetFields).toContain('<AccessBudgetRuleEditor')
    expect(budgetFields).not.toContain('<Advanced')
  })

  it('keeps Team members, model access, and quota expanded in the detail dialog', () => {
    const detail = readSource('./AccessEntityDetail.tsx')
    const teamOverview = detail.slice(detail.indexOf('{team ? ('), detail.indexOf('{group ? ('))

    expect(teamOverview).toContain('<dt>Model access</dt>')
    expect(teamOverview).toContain('<dt>Quota</dt>')
    expect(detail).toContain('<span>Members</span>')
    expect(detail).not.toContain('<details')
  })

  it('keeps table pagination independent from bounded form selectors', () => {
    const page = readSource('./AccessControlPage.tsx')
    const directory = readSource('./useAccessControlDirectory.ts')
    const viewData = readSource('./useAccessControlViewData.ts')
    const fields = readSource('./AccessControlEditorFields.tsx')
    const invitation = readSource('./DashboardMemberInviteDialog.tsx')
    const picker = readSource('./AccessAsyncResourcePicker.tsx')
    const usage = readSource('./AccessControlUsageView.tsx')

    expect(viewData).toContain(
      'accessPageQuery(pageState, pageCursors[activeView]?.[pageState.page])',
    )
    expect(viewData).toContain('loadAllAccessUsers(inferenceAccessApi.users)')
    expect(directory).toContain('loadAllDashboardMembers()')
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

  it('renders access grants with product names instead of internal resource ids', () => {
    const labels = readSource('./useAccessControlViewData.ts')
    const policies = readSource('./AccessControlPolicyViews.tsx')
    const detail = renderToStaticMarkup(
      createElement(AccessGroupResourceTags, {
        resources: [
          { resourceType: 'model', resourceId: 'model-internal-id' },
          { resourceType: 'entrypoint', resourceId: 'entrypoint-internal-id' },
        ],
        resourceName: (resourceType, resourceId) =>
          resourceType === 'model' && resourceId === 'model-internal-id'
            ? 'Customer support model'
            : 'Customer support endpoint',
      }),
    )

    expect(labels).toContain('accessControlSelectorSources.models.detail(resource.resourceId)')
    expect(labels).toContain('accessControlSelectorSources.entrypoints.detail(resource.resourceId)')
    expect(labels).toContain("'Model name unavailable'")
    expect(labels).toContain("'Mixture-of-Model name unavailable'")
    expect(policies).toContain('props.resourceName(resource.resourceType, resource.resourceId)')
    expect(detail).toContain('Customer support model')
    expect(detail).toContain('Customer support endpoint')
    expect(detail).not.toContain('model-internal-id')
    expect(detail).not.toContain('entrypoint-internal-id')
  })

  it('lets viewers read Dashboard identities without requesting invitation authority', () => {
    const directory = readSource('./useAccessControlDirectory.ts')
    const identities = readSource('./AccessControlIdentityViews.tsx')

    expect(directory).toContain('if (!canReadDashboardMembers) return')
    expect(directory).toContain('canManageDashboardMembers')
    expect(directory).toContain('? dashboardMemberInvitationApi.list()')
    expect(directory).toContain(': Promise.resolve({ items: [] as DashboardMemberInvitation[] })')
    expect(identities).toContain(
      "props.canManageDashboardMembers && props.identityTab === 'invitations'",
    )
    expect(identities).toContain('mergeAccessIdentityRows(')
    const dashboardRoleBranch = identities.slice(
      identities.indexOf('{row.member ? ('),
      identities.indexOf(') : row.invitation ?', identities.indexOf('{row.member ? (')),
    )
    expect(dashboardRoleBranch.match(/\{row\.member\.role\}/g)).toHaveLength(1)
  })

  it('keeps dense access tables navigable on narrow screens', () => {
    const styles = readSource('./AccessControlPage.module.css')

    expect(styles).toMatch(/\.dataTable\s*{[^}]*overscroll-behavior-inline: contain;/s)
    expect(styles).toMatch(
      /@media \(max-width: 760px\)[\s\S]*\.dataRow > :first-child\s*{[^}]*position: sticky;/,
    )
  })

  it('exposes pending state on long-running access dialogs', () => {
    const editor = readSource('./AccessControlDialog.tsx')
    const keyDetail = readSource('./APIKeyDetail.tsx')
    const logDetail = readSource('./RequestLogDetail.tsx')

    expect(editor).toContain('aria-busy={saving}')
    expect(keyDetail).toContain('aria-busy={loading || pending}')
    expect(logDetail).toContain('aria-busy={loading}')
  })

  it('submits access editors as a form so required fields are enforced before saving', () => {
    const editor = readSource('./AccessControlDialog.tsx')
    const dashboardAccess = readSource('./DashboardAccessDialog.tsx')

    expect(editor).toContain('useAccessibleDialog<HTMLFormElement>')
    expect(editor).toContain('<form')
    expect(editor).toContain('onSubmit={(event) => {')
    expect(editor).toContain('<button type="submit"')
    expect(dashboardAccess).toContain('useAccessibleDialog<HTMLFormElement>')
    expect(dashboardAccess).toContain('type="submit"')
    expect(dashboardAccess).toContain('minLength={9}')
  })

  it('keeps resource pickers clear of decorative fieldset borders', () => {
    const primitives = readSource('./AccessControlEditorPrimitives.tsx')
    const styles = readSource('./AccessControlPage.module.css')

    expect(primitives).toContain('role="group"')
    expect(primitives).toContain('aria-labelledby={titleId}')
    expect(primitives).not.toContain('<fieldset className={styles.selectionSection}>')
    expect(styles).toContain('.selectionSectionHeader {')
  })
})
