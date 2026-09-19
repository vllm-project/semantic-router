import { expect, test, type Page } from '@playwright/test'
import { mockAuthenticatedAppShell } from '../support/auth'

const experimentID = 'exp-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
const experiment = {
  id: experimentID,
  name: 'Scoped recipe study',
  created_at: '2026-09-18T00:00:00Z',
  updated_at: '2026-09-18T00:00:00Z',
  run_count: 3,
  active_run_count: 0,
}
const single = {
  id: 'single',
  kind: 'single',
  model: 'provider/single',
  base_url: 'http://localhost:8000/v1',
}
const mom = { ...single, id: 'balance', kind: 'mom', model: 'router/balance' }
const frozen = {
  version: 'sr-bench-1.0',
  name: 'Study baseline',
  mode: 'live',
  profile: 'quick',
  seed: 42,
  targets: [single],
  sampling: { temperature: 0, top_p: 1, max_tokens: 512 },
  limits: {
    concurrency: 1,
    max_output_tokens: 512,
    max_run_seconds: 120,
    max_cost_usd: 1,
    max_calls_per_case: 1,
    total_timeout_s: 60,
    idle_timeout_s: 30,
  },
  cases: [
    { id: 'a', benchmark: 'mmlu-pro', messages: [{ role: 'user', content: '1+1?' }] },
    { id: 'b', benchmark: 'mmlu-pro', messages: [{ role: 'user', content: '2+2?' }] },
  ],
}
function savedRun(id: string, name: string, baseline = false) {
  return {
    id,
    status: 'completed',
    created_at: experiment.created_at,
    updated_at: experiment.updated_at,
    experiment_roles: [baseline ? 'baseline' : 'candidate'],
    progress: { total: 2, completed: 2, failed: 0 },
    manifest: { ...frozen, name, targets: [baseline ? single : mom] },
  }
}
const baseline = savedRun('run-baseline', 'Study baseline', true)
const candidate = savedRun('run-candidate', 'Study candidate')
const outsider = savedRun('run-outside', 'Outside candidate')
const outsideBaseline = savedRun('run-outside-baseline', 'Outside baseline', true)
const memberWithoutCandidate = savedRun('run-alone', 'Baseline without a study candidate', true)
const choice = (run: ReturnType<typeof savedRun>) => ({
  run_id: run.id,
  name: run.manifest.name,
  profile: run.manifest.profile,
  case_count: 2,
})

async function setup(page: Page, profile: 'quick' | 'standard' = 'quick') {
  await mockAuthenticatedAppShell(page)
  const comparisonReads: unknown[] = []
  const submissions: Array<{ manifest: typeof frozen }> = []
  const membershipPages: string[] = []
  const runs = [baseline, candidate, outsider, outsideBaseline, memberWithoutCandidate].map(
    (run) => ({ ...run, manifest: { ...run.manifest, profile } }),
  )
  await page.route('**/api/sr-bench/v1/**', async (route) => {
    const url = new URL(route.request().url())
    const path = url.pathname.replace('/api/sr-bench/v1', '')
    let body: unknown
    if (path === '/catalog') body = { profiles: [], benchmarks: [] }
    else if (path === '/datasets') body = { datasets: [] }
    else if (path === '/targets') body = { targets: [single, mom] }
    else if (path === '/runs' && route.request().method() === 'POST') {
      const request = route.request().postDataJSON()
      submissions.push(request)
      body = { ...candidate, id: 'run-started', manifest: request.manifest }
    } else if (path === '/runs') body = { runs }
    else if (path === `/experiments/${experimentID}/runs`) {
      const after = url.searchParams.get('after') ?? '0'
      membershipPages.push(after)
      body = {
        experiment,
        members: (after === '0' ? [memberWithoutCandidate] : [baseline, candidate]).map((run) => ({
          run_id: run.id,
          role: run === candidate ? 'candidate' : 'baseline',
          hypothesis: '',
          linked_at: experiment.created_at,
        })),
        next_cursor: after === '0' ? 20 : null,
        has_more: after === '0',
      }
    } else if (path === '/experiments')
      body = { experiments: [experiment], next_cursor: null, has_more: false }
    else if (path === '/comparison-options') {
      const selected = url.searchParams.get('baseline_run_id')
      const selectedRun = runs.find((run) => run.id === selected)
      body = {
        baseline: selectedRun ? choice(selectedRun) : null,
        baselines: selected ? [] : [outsideBaseline, memberWithoutCandidate, baseline].map(choice),
        options: selected
          ? (selected === baseline.id ? [candidate, outsider] : [outsider]).map(choice)
          : [],
        next_cursor: null,
        has_more: false,
        scanned_pairs: 3,
        scan_limited: false,
        unverified_pairs: 0,
        unverified_baselines: 0,
        model_requests: 0,
      }
    } else if (path === '/replay-options')
      body = { baseline: null, baselines: [], options: [], next_cursor: null, has_more: false }
    else if (path === '/comparisons') {
      comparisonReads.push(route.request().postDataJSON())
      body = { comparisons: [] }
    } else if (path === `/runs/${baseline.id}/candidate-plan`) {
      const request = route.request().postDataJSON()
      body = {
        status: 'validated',
        total: 2,
        plan_sha256: 'a'.repeat(64),
        manifest: {
          ...frozen,
          profile,
          name: request.name,
          mode: request.mode,
          targets: [mom],
          baseline_run_id: baseline.id,
          experiment: request.experiment,
        },
      }
    } else if (path === '/runs/run-started')
      body = { ...candidate, id: 'run-started', manifest: submissions[0]?.manifest ?? frozen }
    else if (path.startsWith('/runs/') && path.endsWith('/report'))
      body = {
        version: 'sr-bench-1.0',
        status: 'completed',
        summary: { targets: [], wall_time_s: 1 },
        benchmarks: [],
        provenance: {},
        limitations: [],
      }
    else if (/\/runs\/[^/]+\/(results|calls|events)$/.test(path)) {
      const field = path.split('/').slice(-1)[0]
      body = { [field]: [], total: 0, next_cursor: null }
    } else if (path.startsWith('/runs/')) body = runs.find((run) => path === `/runs/${run.id}`)
    if (body === undefined)
      return route.fulfill({ status: 404, json: { error: 'Missing scoped study fixture' } })
    return route.fulfill({ json: body })
  })
  return { comparisonReads, submissions, membershipPages }
}

async function choose(page: Page, label: string, value: string) {
  await page.getByRole('combobox', { name: label, exact: true }).click()
  await page.getByRole('listbox').locator(`[data-value="${value}"]`).click()
}

test('uses all membership pages and excludes baselines with only outside compatible candidates', async ({
  page,
}) => {
  const state = await setup(page)
  await page.goto(`/evaluation?view=compare&experiment=${experimentID}`)
  await expect(page.getByRole('combobox', { name: 'Reference run', exact: true })).toBeVisible()
  await page.getByRole('combobox', { name: 'Reference run', exact: true }).click()
  await expect(page.getByRole('option')).toHaveCount(1)
  await expect(page.getByRole('option', { name: /Study baseline/ })).toBeVisible()
  await page.getByRole('option', { name: /Study baseline/ }).click()
  await expect(page.getByRole('checkbox', { name: /Study candidate/ })).toBeVisible()
  await expect(page.getByRole('checkbox', { name: /Outside candidate/ })).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Compare runs', exact: true })).toBeDisabled()
  expect(state.membershipPages).toContain('20')
  expect(state.comparisonReads).toEqual([])
})

test('blocks outside deep-linked comparison until an explicit change of scope', async ({
  page,
}) => {
  const state = await setup(page)
  await page.goto(
    `/evaluation?view=compare&experiment=${experimentID}&baseline=${outsideBaseline.id}&candidate=${outsider.id}`,
  )
  const scope = page.getByRole('checkbox', {
    name: 'Include runs outside this experiment',
    exact: true,
  })
  await expect(scope).toBeVisible()
  await expect(
    page.getByText('The selected reference is outside this experiment.', { exact: false }),
  ).toBeVisible()
  await expect(page.getByRole('button', { name: 'Compare runs', exact: true })).toHaveCount(0)
  expect(state.comparisonReads).toEqual([])
  await scope.click()
  await expect(scope).toBeChecked()
  await expect(page).toHaveURL(/comparison_scope=all/)
  await expect.poll(() => new URL(page.url()).searchParams.has('candidate')).toBe(false)
  await choose(page, 'Reference run', outsideBaseline.id)
  await page.getByRole('checkbox', { name: /Outside candidate/ }).check()
  await page.getByRole('button', { name: 'Compare runs', exact: true }).click()
  await expect.poll(() => state.comparisonReads.length).toBe(1)
  expect(state.comparisonReads[0]).toEqual({
    baseline_run_id: outsideBaseline.id,
    candidate_run_id: outsider.id,
  })
})

for (const { profile, inputRole, expectedRole } of [
  { profile: 'quick', inputRole: 'initial', expectedRole: 'initial' },
  { profile: 'standard', inputRole: 'candidate', expectedRole: 'validation' },
] as const) {
  test(`keeps ${profile} recipe context through submission with role ${expectedRole}`, async ({
    page,
  }) => {
    const state = await setup(page, profile)
    await page.goto(
      `/evaluation?view=new&baseline=${baseline.id}&experiment=${experimentID}&role=${inputRole}`,
    )
    await expect(page.getByRole('region', { name: 'Frozen baseline protocol' })).toBeVisible()
    await page.getByRole('combobox', { name: 'Add configured target', exact: true }).click()
    await expect(page.getByRole('listbox').locator(`[data-value="${single.id}"]`)).toHaveCount(0)
    await page.getByRole('listbox').locator(`[data-value="${mom.id}"]`).click()
    await page.getByLabel('Run name', { exact: true }).fill('Starting recipe result')
    await page.getByRole('button', { name: 'Review plan', exact: true }).click()
    await page.getByRole('button', { name: 'Start evaluation', exact: true }).click()
    await expect(page).toHaveURL(/run=run-started/)
    expect(new URL(page.url()).searchParams.get('experiment')).toBe(experimentID)
    expect(state.submissions).toHaveLength(1)
    expect(state.submissions[0].manifest).toMatchObject({
      experiment: { id: experimentID, role: expectedRole },
      profile,
    })
    await page.getByRole('button', { name: 'Back to experiment', exact: true }).click()
    await expect(page).toHaveURL(new RegExp(`view=experiments&experiment=${experimentID}$`))
  })
}

test('global create and preview actions preserve the study with legal explicit roles', async ({
  page,
}) => {
  const state = await setup(page)
  await page.goto(`/evaluation?view=compare&experiment=${experimentID}&role=candidate`)
  await page.getByRole('button', { name: 'Create evaluation', exact: true }).click()
  await expect
    .poll(() => {
      const search = new URL(page.url()).searchParams
      return [search.get('view'), search.get('experiment'), search.get('role')]
    })
    .toEqual(['new', experimentID, 'baseline'])
  await page.getByRole('combobox', { name: 'Add configured target', exact: true }).click()
  await expect(page.getByRole('listbox').locator(`[data-value="${mom.id}"]`)).toHaveCount(0)
  await expect(page.getByRole('listbox').locator(`[data-value="${single.id}"]`)).toBeVisible()
  await page.keyboard.press('Escape')
  await page.getByRole('button', { name: 'Preview routing', exact: true }).click()
  await expect
    .poll(() => {
      const search = new URL(page.url()).searchParams
      return [search.get('view'), search.get('experiment'), search.get('role')]
    })
    .toEqual(['preview', experimentID, 'preview'])
  expect(state.submissions).toEqual([])
})
