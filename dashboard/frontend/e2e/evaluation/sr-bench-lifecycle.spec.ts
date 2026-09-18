import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { expect, test, type Page, type Response, type TestInfo } from '@playwright/test'
import type {
  CallRecord,
  Manifest,
  RecoveryPlan,
  Report,
  Run,
} from '../../src/components/sr-bench/types'

interface LifecyclePlan {
  base_url: string
  execution_authorized: boolean
  expected_deployment_sha: string
  purpose: 'synthetic-dashboard-lifecycle-only'
  run_name: string
  profile: 'smoke' | 'quick' | 'standard'
  dataset_id: string
  dataset_sha256: string
  case_ids: string[]
  target_id: string
  target_model: string
  limits: {
    max_cost_usd: number
    max_run_seconds: number
    total_timeout_s: number
    idle_timeout_s: number
    max_output_tokens: number
  }
  attempt_receipt_path: string
}

const planPath = process.env.SR_BENCH_LIFECYCLE_PLAN
const plan: LifecyclePlan | null = planPath
  ? (JSON.parse(readFileSync(planPath, 'utf8')) as LifecyclePlan)
  : null
const purpose = 'synthetic-dashboard-lifecycle-only'
if (plan) {
  const origin = new URL(plan.base_url)
  if (
    origin.protocol !== 'http:' ||
    !['127.0.0.1', 'localhost', '[::1]'].includes(origin.hostname) ||
    origin.username ||
    origin.password ||
    origin.pathname !== '/' ||
    origin.search ||
    origin.hash
  )
    throw new Error('Lifecycle acceptance requires an explicit credential-free loopback origin.')
  plan.base_url = origin.origin
  if (
    plan.execution_authorized !== true ||
    plan.purpose !== purpose ||
    !/^[a-f0-9]{40}$/.test(plan.expected_deployment_sha) ||
    !/^[a-f0-9]{64}$/.test(plan.dataset_sha256) ||
    !plan.run_name.toLowerCase().includes('synthetic') ||
    !['smoke', 'quick', 'standard'].includes(plan.profile) ||
    plan.case_ids.length < 3 ||
    plan.case_ids.length > 16 ||
    new Set(plan.case_ids).size !== plan.case_ids.length ||
    !plan.case_ids.every((id) => id.includes('synthetic-dashboard-lifecycle')) ||
    !plan.attempt_receipt_path.startsWith('/') ||
    !plan.target_id ||
    !plan.target_model ||
    Object.values(plan.limits).some((value) => !Number.isFinite(value) || value <= 0) ||
    plan.limits.max_cost_usd > 0.25 ||
    plan.limits.max_run_seconds > 180 ||
    plan.limits.total_timeout_s > 90 ||
    plan.limits.idle_timeout_s > plan.limits.total_timeout_s ||
    plan.limits.max_output_tokens > 512
  )
    throw new Error(
      'Lifecycle plan lacks authorization, a fresh synthetic scope, or bounded limits.',
    )
}

test.skip(!plan, 'Opt in with an explicitly authorized SR_BENCH_LIFECYCLE_PLAN.')

const api = '/api/sr-bench/v1'
const key = (cell: { case_id: string; target_id: string }) => `${cell.target_id}:${cell.case_id}`
const terminal = (run: Run) => !['queued', 'running'].includes(run.status)

async function read<T>(page: Page, path: string): Promise<T> {
  const response = await page.request.get(`${plan!.base_url}${api}${path}`)
  expect(response.ok(), `Read-only evidence ${path} must be available`).toBe(true)
  return response.json() as Promise<T>
}

function responseFor(path: string) {
  return (response: Response) =>
    response.request().method() === 'POST' && new URL(response.url()).pathname === `${api}${path}`
}

async function capture(page: Page, info: TestInfo, name: string) {
  await page.screenshot({ path: info.outputPath(name), fullPage: true })
  await info.attach(name, { path: info.outputPath(name), contentType: 'image/png' })
}

async function waitForTerminal(page: Page, id: string, timeout = 30000) {
  let latest = await read<Run>(page, `/runs/${id}`)
  await expect
    .poll(
      async () => {
        latest = await read<Run>(page, `/runs/${id}`)
        return terminal(latest)
      },
      { timeout },
    )
    .toBe(true)
  return latest
}

test('authorized synthetic UI launch, cancellation and undispatched recovery remain durable', async ({
  page,
}, info) => {
  const approved = plan!
  expect(
    existsSync(approved.attempt_receipt_path),
    'An existing one-shot receipt requires manual reconciliation; never rerun this attempt.',
  ).toBe(false)
  const evidence: Record<string, unknown> = {
    purpose,
    capability_claim: false,
    expected_deployment_sha: approved.expected_deployment_sha,
    dataset_sha256: approved.dataset_sha256,
    source_cell_count: approved.case_ids.length,
    target_id: approved.target_id,
    limits_per_attempt: approved.limits,
    combined_attempt_budget_ceiling_usd: approved.limits.max_cost_usd * 2,
    recovery_scope: 'Exactly one cell with no prior dispatched call',
    started_at: new Date().toISOString(),
  }
  let claimed = false
  let parentID = ''
  let selected: { case_id: string; target_id: string } | null = null
  let recoveryHash = ''
  const mutations: string[] = []
  const blocked: string[] = []
  const save = () => {
    if (claimed)
      writeFileSync(approved.attempt_receipt_path, JSON.stringify(evidence, null, 2) + '\n')
  }
  const assertScope = (manifest: Manifest) => {
    expect(manifest.name === approved.run_name && manifest.mode === 'live').toBe(true)
    expect(manifest.profile).toBe(approved.profile)
    expect(manifest.cost_policy).toBe('require_priced')
    expect(manifest.dataset?.sha256).toBe(approved.dataset_sha256)
    expect(manifest.targets).toHaveLength(1)
    const target = manifest.targets[0]
    expect(
      target.id === approved.target_id &&
        target.kind === 'single' &&
        target.model === approved.target_model,
      'Only the approved single Flash target may execute',
    ).toBe(true)
    expect(manifest.limits.concurrency).toBe(1)
    expect(manifest.limits.max_calls_per_case).toBe(1)
    for (const [name, value] of Object.entries(approved.limits))
      expect(manifest.limits[name as keyof Manifest['limits']]).toBe(value)
    expect(manifest.sampling.max_tokens).toBe(approved.limits.max_output_tokens)
    expect(manifest.sampling.temperature).toBe(0)
    expect(manifest.sampling.top_p).toBe(1)
  }

  await page.route(`**${api}/**`, async (route) => {
    const request = route.request()
    if (['GET', 'HEAD'].includes(request.method())) return route.continue()
    const path = new URL(request.url()).pathname.slice(api.length)
    let allowed = false
    try {
      const body = request.postDataJSON()
      if (request.method() === 'POST' && ['/plans', '/runs'].includes(path)) {
        assertScope(body.manifest)
        allowed = !mutations.includes(path)
        if (path === '/runs' && allowed) {
          // Claim before dispatch, never delete on failure. An interrupted browser
          // attempt must be reconciled from this receipt instead of started again.
          writeFileSync(approved.attempt_receipt_path, JSON.stringify(evidence, null, 2) + '\n', {
            flag: 'wx',
          })
          claimed = true
          evidence.launch_idempotency_key = body.idempotency_key
        }
      } else if (request.method() === 'POST' && parentID) {
        if (path === `/runs/${parentID}/cancel`) allowed = !mutations.includes(path)
        if (path === `/runs/${parentID}/recover-plan`)
          allowed = body.mode === 'undispatched' && !mutations.includes(path)
        if (path === `/runs/${parentID}/recover`)
          allowed =
            !mutations.includes(path) &&
            body.mode === 'undispatched' &&
            body.plan_sha256 === recoveryHash &&
            body.cells.length === 1 &&
            selected !== null &&
            key(body.cells[0]) === key(selected)
      }
    } catch {
      allowed = false
    }
    if (!allowed) {
      blocked.push(`${request.method()} ${path}`)
      evidence.blocked_mutations = blocked
      save()
      return route.abort('blockedbyclient')
    }
    mutations.push(path)
    evidence.mutations = mutations
    save()
    return route.continue()
  })

  try {
    await page.goto(`${approved.base_url}/__acceptance/login`)
    await page.getByRole('button', { name: 'Start acceptance session', exact: true }).click()
    await page.waitForURL((url) => url.pathname !== '/__acceptance/login')
    await page.goto(`${approved.base_url}/evaluation?view=new`)
    await expect(page.getByRole('heading', { name: 'Create evaluation' })).toBeVisible()
    const existing = await read<{ runs: Run[] }>(page, '/runs')
    expect(
      existing.runs.some(
        (run) =>
          run.manifest.name === approved.run_name ||
          run.manifest.dataset?.sha256 === approved.dataset_sha256,
      ),
      'This dataset must have no previous evaluation attempt',
    ).toBe(false)
    await page.getByLabel('Run name', { exact: true }).fill(approved.run_name)
    await page.getByLabel('Profile', { exact: true }).selectOption(approved.profile)
    await page.getByLabel('Prepared dataset', { exact: true }).selectOption(approved.dataset_id)
    await page.getByLabel('Add configured target', { exact: true }).selectOption(approved.target_id)
    for (const [label, value] of [
      ['Budget (USD)', approved.limits.max_cost_usd],
      ['Run deadline (seconds)', approved.limits.max_run_seconds],
      ['Request deadline (seconds)', approved.limits.total_timeout_s],
      ['Idle timeout (seconds)', approved.limits.idle_timeout_s],
      ['Max output tokens', approved.limits.max_output_tokens],
      ['Concurrency', 1],
    ] as const)
      await page.getByLabel(label, { exact: true }).fill(String(value))
    await page.getByText('Advanced manifest', { exact: true }).click()
    await page.getByLabel('Use edited manifest', { exact: true }).check()
    const manifest = JSON.parse(await page.getByLabel('Manifest JSON').inputValue()) as Manifest
    manifest.limits.max_calls_per_case = 1
    await page.getByLabel('Manifest JSON').fill(JSON.stringify(manifest, null, 2))
    const plannedResponse = page.waitForResponse(responseFor('/plans'))
    await page.getByRole('button', { name: 'Review plan', exact: true }).click()
    const plannedHTTP = await plannedResponse
    expect(plannedHTTP.ok()).toBe(true)
    const planned = await plannedHTTP.json()
    assertScope(planned.manifest)
    expect(planned.total).toBe(approved.case_ids.length)
    expect(planned.manifest.cases.map((item: { id: string }) => item.id).sort()).toEqual(
      [...approved.case_ids].sort(),
    )
    expect(
      planned.manifest.cases.every(
        (item: { metadata?: { purpose?: string } }) => item.metadata?.purpose === purpose,
      ),
      'Synthetic cases must remain labeled non-capability lifecycle evidence',
    ).toBe(true)
    evidence.plan_sha256 = planned.plan_sha256
    await expect(page.getByRole('heading', { name: 'Plan ready for review' })).toBeVisible()
    await capture(page, info, 'synthetic-reviewed-plan.png')
    const startResponse = page.waitForResponse(responseFor('/runs'))
    await page.getByRole('button', { name: 'Start evaluation', exact: true }).click()
    const startHTTP = await startResponse
    expect(startHTTP.ok()).toBe(true)
    parentID = ((await startHTTP.json()) as Run).id
    evidence.parent_run_id = parentID
    save()
    let parentCalls: CallRecord[] = []
    // The synthetic scope can finish quickly. Observe dispatch and the actual UI
    // concurrently, then cancel immediately; never prolong the model workload.
    const cancelButton = page.getByRole('button', { name: 'Cancel evaluation', exact: true })
    await Promise.all([
      expect(page).toHaveURL(new RegExp(`run=${parentID}`)),
      expect(cancelButton).toBeVisible(),
      expect
        .poll(async () => {
          parentCalls = (await read<{ calls: CallRecord[] }>(page, `/runs/${parentID}/calls`)).calls
          return parentCalls.length
        })
        .toBeGreaterThan(0),
    ])
    const cancelResponse = page.waitForResponse(responseFor(`/runs/${parentID}/cancel`))
    await cancelButton.click()
    expect((await cancelResponse).ok()).toBe(true)
    evidence.cancel_requested_after_durable_call = true
    await capture(page, info, 'synthetic-cancellation-requested.png')
    const parent = await waitForTerminal(page, parentID)
    expect(parent.status).toBe('cancelled')
    await page.reload()
    await expect(page.getByRole('progressbar', { name: 'Evaluation progress' })).toBeVisible()
    await expect(page).toHaveURL(new RegExp(`run=${parentID}`))
    await expect(
      page.getByText('This run is cancelled. Partial results are not a completed evaluation.', {
        exact: true,
      }),
    ).toBeVisible()
    evidence.persisted_after_reload = true
    evidence.persisted_after_reload_scope =
      'Cancelled parent identity and terminal status; active-run reload is verified separately.'
    await capture(page, info, 'synthetic-cancelled-parent-after-reload.png')
    evidence.parent_progress = parent.progress
    parentCalls = (await read<{ calls: CallRecord[] }>(page, `/runs/${parentID}/calls`)).calls
    const dispatched = new Set(parentCalls.map(key))
    const parentReport = await read<Report>(page, `/runs/${parentID}/report`)
    expect(parentReport.summary.targets).toHaveLength(1)
    expect(parentReport.summary.targets[0].total).toBe(approved.case_ids.length)
    const knownParentSpend = parentCalls.reduce(
      (sum, call) => sum + (typeof call.cost_usd === 'number' ? call.cost_usd : 0),
      0,
    )
    if (parentCalls.every((call) => typeof call.cost_usd === 'number'))
      expect(parentReport.summary.total_spend_usd).toBeCloseTo(knownParentSpend, 10)
    else expect(parentReport.summary.total_spend_usd).toBeNull()
    evidence.parent_dispatched_cells = [...dispatched]
    save()
    await page.getByRole('button', { name: 'Refresh evidence', exact: true }).click()
    await expect(page.getByRole('heading', { name: 'Recover unfinished work' })).toBeVisible()
    await page.getByLabel('Recovery scope', { exact: true }).selectOption('undispatched')
    const recoveryPlanResponse = page.waitForResponse(responseFor(`/runs/${parentID}/recover-plan`))
    await page.getByRole('button', { name: 'Review recovery plan', exact: true }).click()
    const recoveryPlanHTTP = await recoveryPlanResponse
    expect(recoveryPlanHTTP.ok()).toBe(true)
    const recovery = (await recoveryPlanHTTP.json()) as RecoveryPlan
    expect(recovery.parent.progress).toEqual(parent.progress)
    expect(recovery.parent.known_spend_usd).toBeCloseTo(knownParentSpend, 10)
    expect(recovery.parent.spend_complete).toBe(
      parentCalls.every((call) => typeof call.cost_usd === 'number'),
    )
    expect(recovery.eligible_cells.every((cell) => !dispatched.has(key(cell)))).toBe(true)
    selected =
      recovery.eligible_cells.find(
        (cell) => cell.target_id === approved.target_id && approved.case_ids.includes(cell.case_id),
      ) ?? null
    expect(
      selected,
      'Cancellation must leave an undispatched cell; never retry to manufacture one',
    ).not.toBeNull()
    recoveryHash = recovery.plan_sha256
    evidence.recovery_plan_sha256 = recoveryHash
    evidence.recovery_selected_cell = selected
    save()
    await page
      .getByRole('checkbox', {
        name: `Recover ${selected!.target_id} ${selected!.case_id}`,
        exact: true,
      })
      .check()
    await capture(page, info, 'synthetic-undispatched-recovery-review.png')
    const recoverResponse = page.waitForResponse(responseFor(`/runs/${parentID}/recover`))
    await page.getByRole('button', { name: 'Create recovery run (1 cases)', exact: true }).click()
    const recoverHTTP = await recoverResponse
    expect(recoverHTTP.ok()).toBe(true)
    const childID = ((await recoverHTTP.json()) as Run).id
    evidence.child_run_id = childID
    save()
    await expect(page).toHaveURL(new RegExp(`run=${childID}`))
    const child = await waitForTerminal(
      page,
      childID,
      (approved.limits.max_run_seconds + 10) * 1000,
    )
    expect(child.status).toBe('completed')
    expect(child.progress).toEqual({ total: 1, completed: 1, failed: 0 })
    const childCalls = (await read<{ calls: CallRecord[] }>(page, `/runs/${childID}/calls`)).calls
    expect(childCalls).toHaveLength(1)
    expect(key(childCalls[0])).toBe(key(selected!))
    expect(childCalls.some((call) => dispatched.has(key(call)))).toBe(false)
    const preservedParent = await read<Run>(page, `/runs/${parentID}`)
    expect(preservedParent.status).toBe('cancelled')
    expect(preservedParent.progress).toEqual(parent.progress)
    evidence.child_progress = child.progress
    evidence.no_repeated_dispatched_cells = true
    const preservedParentReport = await read<Report>(page, `/runs/${parentID}/report`)
    expect(preservedParentReport.summary).toEqual(parentReport.summary)
    const childReport = await read<Report>(page, `/runs/${childID}/report`)
    expect(childReport.summary.targets).toHaveLength(1)
    expect(childReport.summary.targets[0]).toMatchObject({ total: 1, completed: 1, failed: 0 })
    expect(childCalls.every((call) => typeof call.cost_usd === 'number')).toBe(true)
    const childSpend = childCalls.reduce((sum, call) => sum + (call.cost_usd as number), 0)
    expect(childReport.summary.total_spend_usd).toBeCloseTo(childSpend, 10)
    expect(childReport.summary.targets[0].total_spend_usd).toBeCloseTo(childSpend, 10)
    expect(childReport.recovery).toMatchObject({
      parent_run_id: parentID,
      parent_snapshot: recovery.parent,
      selected_cells: [selected],
    })
    evidence.parent_metrics = preservedParentReport.summary
    evidence.child_metrics = childReport.summary
    evidence.accounting_checks = {
      parent_summary_unchanged: true,
      child_denominator: 1,
      child_spend_equals_own_receipts: true,
      parent_snapshot_spend_complete: recovery.parent.spend_complete,
    }
    await page.getByRole('button', { name: 'Refresh evidence', exact: true }).click()
    await expect(page.getByRole('heading', { name: 'Recovery lineage', exact: true })).toBeVisible()
    await expect(page.getByRole('link', { name: 'Open parent run', exact: true })).toHaveAttribute(
      'href',
      `?view=runs&run=${parentID}`,
    )
    await capture(page, info, 'synthetic-recovered-child-desktop.png')
    await page.setViewportSize({ width: 390, height: 844 })
    await expect
      .poll(() =>
        page
          .getByRole('region', { name: 'sr-bench workspace' })
          .evaluate((element) => element.getBoundingClientRect().right <= window.innerWidth),
      )
      .toBe(true)
    await capture(page, info, 'synthetic-recovered-child-mobile.png')
    evidence.finished_at = new Date().toISOString()
    evidence.result = 'passed'
    expect(blocked).toEqual([])
  } finally {
    if (!evidence.result) {
      evidence.result = 'incomplete; reconcile saved run identities without resubmission'
      await capture(page, info, 'synthetic-lifecycle-incomplete.png').catch(() => undefined)
    }
    save()
    await info.attach('synthetic-lifecycle-receipt.json', {
      body: JSON.stringify(evidence, null, 2),
      contentType: 'application/json',
    })
  }
})
