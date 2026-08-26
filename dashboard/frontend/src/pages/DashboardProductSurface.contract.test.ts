import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const readSource = (name: string) => readFileSync(new URL(name, import.meta.url), 'utf8')

describe('dashboard product surfaces', () => {
  it('puts a calm system overview in the first dashboard surface', () => {
    const page = readSource('./DashboardPage.tsx')
    const hero = readSource('./DashboardRoutingHero.tsx')

    expect(page.indexOf('<DashboardRoutingHero')).toBeLessThan(page.indexOf('mainGrid'))
    expect(hero).toContain('Your model system, at a glance.')
    expect(hero).toContain('Every capability path visible, governed, and ready.')
    expect(hero).not.toContain('activeRoute')
    expect(hero).not.toContain('routeDots')
  })

  it('keeps model density and routing metrics readable at phone widths', () => {
    const pageStyles = readSource('./DashboardPage.module.css')
    const heroStyles = readSource('./DashboardRoutingHero.module.css')

    expect(pageStyles).toMatch(
      /\.modelList\s*{[^}]*grid-template-columns: repeat\(2, minmax\(0, 1fr\)\);/s,
    )
    expect(pageStyles).toMatch(
      /@media \(max-width: 640px\)[\s\S]*\.modelList\s*{[^}]*grid-template-columns: minmax\(0, 1fr\);/,
    )
    expect(heroStyles).toMatch(
      /@media \(max-width: 420px\)[\s\S]*\.metricStrip\s*{[^}]*grid-template-columns: minmax\(0, 1fr\);/,
    )
  })

  it('keeps the home model inventory passive instead of deep-linking to a backend model', () => {
    const page = readSource('./DashboardPage.tsx')
    const pageStyles = readSource('./DashboardPage.module.css')

    expect(page).not.toContain('styles.modelIcon')
    expect(pageStyles).not.toContain('.modelIcon')
    expect(page).not.toContain('canChatWithSingleModel')
    expect(page).not.toContain('/playground?model=')
    expect(page).not.toContain('styles.modelAction')
    expect(pageStyles).not.toContain('.modelAction')
  })

  it('shows every key-authorized single model from the delegated data plane', () => {
    const playground = readSource('../components/AgentPlayground.tsx')
    const routingHook = readSource('../components/usePlaygroundRoutingModel.ts')

    expect(playground).not.toContain('includeSingleModels')
    expect(routingHook).toContain('includeIndividualModels: true')
    expect(routingHook).toContain('Dashboard role must not hide a Router-authorized model')
    expect(routingHook).toContain('getRouterModelsEndpoint(endpoint)')
    expect(routingHook).toContain('fetchPlaygroundModelPayload(')
    expect(routingHook).toContain('getAccessToken')
    expect(playground).not.toContain('routingManagementApi.listModelCards')
    expect(routingHook).not.toContain('routingManagementApi.listModelCards')
  })

  it('does not fetch Router Management routing data without routing.read', () => {
    const page = readSource('./DashboardPage.tsx')

    expect(page).toContain(
      'canReadConfig ? configRequest.run({ allowHidden: true }) : Promise.resolve()',
    )
    expect(page).toContain('if (!canReadConfig) return')
  })

  it('keeps the consumer home useful through its key-scoped routing projection', () => {
    const page = readSource('./DashboardPage.tsx')

    expect(page).toContain('useInferenceRoutingAccess()')
    expect(page).toContain(
      'const overviewConfig = canReadConfig ? config : usesKeyScopedCatalog ? catalogSnapshot : null',
    )
    expect(page).toContain('{canReadIntelligence ? (')
    expect(page).toContain('{showSystemHealth ? (')
    expect(page).toContain('<h2 className={styles.cardTitle}>Access</h2>')
    expect(page).toContain('{canReadStatus ? (')
    expect(page).toContain('No accessible routing paths')
  })

  it('does not present dashboard actions that the current identity cannot open', () => {
    const page = readSource('./DashboardPage.tsx')
    const hero = readSource('./DashboardRoutingHero.tsx')

    expect(page).toContain("canAccessDashboardPath(user, '/status')")
    expect(page).toContain('useSystemStatus()')
    expect(page).toContain("routingAccess !== 'operational'")
    expect(page).not.toContain('fetchSystemStatus')
    expect(page).not.toContain('statusRequest')
    expect(page).toContain('{canReadStatus ? (')
    expect(page).toContain('<h2 className={styles.cardTitle}>System Health</h2>')
    expect(page).toContain("canAccessDashboardPath(user, '/playground')")
    expect(page).toContain('showRoutingMetrics={canReadConfig}')
    expect(page).toContain('showAPIKeyMetric={canReadAccess}')
    expect(page).toContain('showPlaygroundAction={canUsePlayground}')
    expect(page).toContain('showStatus={canReadStatus}')
    expect(hero).toContain('{showPlaygroundAction ? (')
    expect(hero).toContain('{showRoutingMetrics ? (')
    expect(hero).toContain('{showAPIKeyMetric ? (')
    expect(hero).toContain('{showStatus ? (')
  })

  it('uses one shared authentication composition for login and invitations', () => {
    const login = readSource('./LoginPage.tsx')
    const invite = readSource('./InviteAcceptPage.tsx')

    expect(login).toContain('<AuthExperienceShell')
    expect(invite).toContain('<AuthExperienceShell')
    expect(invite).toContain('Your invitation is here')
    expect(invite).toContain('Build what one model can’t.')
    expect(invite).toContain('Valid until')
    expect(invite).not.toContain('markFirstAPIKeyOnboardingPending')
  })

  it('welcomes invited users before delivering the one-time key experience', () => {
    const invite = readSource('./InviteAcceptPage.tsx')
    const shell = readSource('../app/AuthenticatedShell.tsx')
    const access = readSource('./AccessControlPage.tsx')
    const welcome = readSource('../components/InvitationWelcomeDialog.tsx')
    const moment = readSource('../components/ProductMomentDialog.tsx')

    expect(invite).toContain("navigate('/dashboard'")
    expect(invite).toContain(
      'stageInvitationOnboarding({ displayName: invitation.name, onboardingKey })',
    )
    expect(invite).not.toContain('state: { invitationOnboarding')
    expect(invite).not.toContain('markFirstAPIKeyOnboardingPending')
    expect(shell).toContain('<InvitationWelcomeDialog')
    expect(shell).toContain("navigate('/access/api-keys?onboarding=invitation'")
    expect(access).toContain('claimInvitationOnboarding(selfUserId)')
    expect(shell).not.toContain('state: { onboardingKey')
    expect(shell).not.toContain('inferenceAccessApi.createSelfKey')
    expect(welcome).toContain('Reveal my API key')
    expect(welcome).toContain('One key. Your team’s models. Ready to build.')
    expect(moment).toContain('src="/vllm.png"')
  })

  it('uses invitation as the only dashboard user creation path', () => {
    const access = readSource('./AccessControlPage.tsx')
    const workspace = readSource('./AccessControlWorkspace.tsx')
    const invitation = readSource('./DashboardMemberInviteDialog.tsx')

    expect(workspace).toContain('<ProductIcon name="plus" /> Invite user')
    expect(access).not.toContain("target === 'user'")
    expect(access).not.toContain("? 'New user'")
    expect(invitation).toContain('Dashboard role')
    expect(invitation).toContain('Team <small>Optional</small>')
    expect(invitation).toContain('Team role')
  })

  it('keeps every Build manager on the shared banner composition', () => {
    const manager = readSource('./ConfigPageManagerLayout.tsx')

    expect(manager).toContain('className={styles.headerGrid}')
    expect(manager).toContain('className={styles.surfacePulse}')
    expect(manager).toContain('Semantic Router')
  })

  it('uses the shared semantic icon language on topology and access choices', () => {
    const topologyFiles = [
      './topology/TopologyPageEnhanced.tsx',
      './topology/components/CustomNodes/AlgorithmNode.tsx',
      './topology/components/CustomNodes/ClientNode.tsx',
      './topology/components/CustomNodes/DecisionNode.tsx',
      './topology/components/CustomNodes/DefaultRouteNode.tsx',
      './topology/components/CustomNodes/FallbackDecisionNode.tsx',
      './topology/components/CustomNodes/GlobalPluginNode.tsx',
      './topology/components/CustomNodes/ModelNode.tsx',
      './topology/components/CustomNodes/PluginChainNode.tsx',
      './topology/components/CustomNodes/SignalGroupNode.tsx',
      './topology/components/ResultCard/ResultCard.tsx',
    ].map(readSource)
    const accessFields =
      readSource('./AccessControlEditorFields.tsx') +
      readSource('./AccessControlEditorPrimitives.tsx')

    for (const source of topologyFiles) {
      expect(source).toContain('ProductIcon')
      expect(source).not.toMatch(/[🤖🧠🔌📊⚠✓✗]/u)
    }
    expect(accessFields).toContain('className={styles.choiceCheck}')
    expect(accessFields).not.toContain('<i>✓</i>')
  })
})
