import { expect, test } from '@playwright/test'

import { evaluationCatalog } from './support/catalog'
import { EVALUATION_RUN_IDS } from './support/mixtureFixture'
import { mockEvaluationPlane } from './support/mockEvaluationPlane'
import { defaultEvaluationRuns } from './support/runFixtures'
import {
  captureEvaluationSurface,
  expectKeyboardScrollable,
  expectNoHorizontalOverflow,
  expectProductEvaluationLanguage,
} from './support/pageAssertions'
import { mockEvaluationUserSession } from './support/session'

test.describe('Evaluation Plane · Overview', () => {
  test.beforeEach(async ({ page }) => {
    await mockEvaluationUserSession(page)
  })

  test('shows complete evaluation coverage and benchmark readiness in product language', async ({
    page,
  }) => {
    await mockEvaluationPlane(page)
    await page.goto('/evaluation')

    await expect(page.getByRole('heading', { name: 'Evaluation', exact: true })).toBeVisible()
    for (const tab of ['Overview', 'New experiment', 'Runs', 'Reports', 'Compare']) {
      await expect(page.getByRole('tab', { name: tab, exact: true })).toBeVisible()
    }

    await expect(page.getByText('Decision quality', { exact: true })).toBeVisible()
    await expect(
      page.getByText(
        'This run is useful for exploration, but it is not ready to support a release decision. Run a qualified benchmark or live evaluation before changing production.',
        { exact: true },
      ),
    ).toBeVisible()
    await expectProductEvaluationLanguage(page)

    const readiness = page.getByRole('table', {
      name: 'Available measurements and latest results by evaluation area',
    })
    await expect(readiness.getByRole('row')).toHaveCount(evaluationCatalog.tracks.length + 1)
    for (const track of evaluationCatalog.tracks) {
      await expect(
        readiness.getByRole('row').filter({
          has: page.getByText(track.name, { exact: true }),
        }),
      ).toBeVisible()
    }
    await expect(
      readiness.getByRole('row').filter({ has: page.getByText('Routing', { exact: true }) }),
    ).toContainText('Diagnostic · Routing validation')
    await expectKeyboardScrollable(
      page.getByRole('region', { name: 'Scrollable evaluation area readiness' }),
      'vertical',
    )
    const declaredMethodCount = evaluationCatalog.suites.reduce(
      (count, suite) => count + suite.methods.length,
      0,
    )
    const methods = page.locator('section[aria-labelledby="evaluation-methods-title"]')
    const methodSummary = methods.locator('details > summary').filter({
      has: page.getByText('Browse benchmark methods', { exact: true }),
    })
    const methodDisclosure = methodSummary.locator('..')
    await expect(methodDisclosure).not.toHaveAttribute('open', '')
    await methodSummary.click()
    const methodTable = methods.getByRole('table', {
      name: 'Available evaluation methods and setup readiness',
    })
    await expect(methodTable.getByRole('row')).toHaveCount(declaredMethodCount + 1)
    await expectKeyboardScrollable(
      methods.getByRole('region', { name: 'Scrollable evaluation method readiness' }),
      'vertical',
    )
    const methodSearch = methods.getByLabel('Search evaluation methods')
    const hardPolicySuite = evaluationCatalog.suites.find((suite) =>
      suite.methods.some((method) => method.id === 'policy.hard-enforcement.v1'),
    )!
    await methodSearch.fill('hard-policy')
    await expect(
      methodTable.getByRole('row').filter({
        has: page.getByText(hardPolicySuite.name, { exact: true }),
      }),
    ).toBeVisible()
    await expect(methods.getByRole('status')).toHaveText(
      `Showing 1 of ${declaredMethodCount} methods`,
    )
    await methodSearch.clear()
    await methods.getByLabel('Method evaluation area filter').selectOption('safety')
    await methods.getByLabel('Method readiness filter').selectOption('setup_required')
    await expect(
      methodTable.getByRole('row').filter({
        has: page.getByText(hardPolicySuite.name, { exact: true }),
      }),
    ).toContainText('Setup required')
    await expect(methods.getByRole('status')).toHaveText(
      `Showing 1 of ${declaredMethodCount} methods`,
    )
    await captureEvaluationSurface(page, 'overview-desktop')
  })

  test('contains long readiness evidence inside its own scroll region at 320px', async ({
    page,
  }) => {
    const longDescription =
      'Routes exact production request cohorts across a frozen Mixture-of-Models pool while preserving abstention, fallback, selector latency, and per-arm outcome provenance for release review.'
    const longCatalog = {
      ...evaluationCatalog,
      tracks: evaluationCatalog.tracks.map((track) =>
        track.id === 'routing'
          ? {
              ...track,
              description: longDescription,
            }
          : track,
      ),
    }
    await page.setViewportSize({ width: 320, height: 568 })
    await mockEvaluationPlane(page, defaultEvaluationRuns, { catalog: longCatalog })
    await page.goto('/evaluation')

    const readiness = page.getByRole('region', {
      name: 'Scrollable evaluation area readiness',
    })
    await expect(readiness).toBeVisible()
    await expect(readiness).toHaveAttribute('tabindex', '0')
    await expect(readiness.getByText(longDescription, { exact: true })).toBeVisible()
    await expect
      .poll(() => readiness.evaluate((element) => element.scrollWidth - element.clientWidth))
      .toBeGreaterThan(0)
    await expectKeyboardScrollable(readiness, 'horizontal')
    await expectNoHorizontalOverflow(page)
  })

  test('keeps the initial loading boundary until catalog and durable ledger both settle', async ({
    page,
  }) => {
    await mockEvaluationPlane(page, defaultEvaluationRuns, { ledgerDelayMs: 750 })
    const catalogResponse = page.waitForResponse('**/api/evaluation/v1/catalog')
    await page.goto('/evaluation')
    await catalogResponse

    await expect(page.getByText('Loading evaluation', { exact: true })).toBeVisible()
    await expect(page.getByText('Latest decision', { exact: true })).toHaveCount(0)

    await expect(page.getByText('Loading evaluation', { exact: true })).toHaveCount(0)
    await expect(page.getByText('Latest decision', { exact: true })).toBeVisible()
  })

  test('keeps evidence navigation available while suppressing run mutations in read-only mode', async ({
    page,
  }) => {
    await mockEvaluationUserSession(page, {
      readonlyMode: true,
      serverReadonly: true,
    })
    await mockEvaluationPlane(page)
    await page.goto(`/evaluation?view=runs&run=${EVALUATION_RUN_IDS.candidate}`)

    await expect(page.getByText(/server is in read-only mode/i)).toBeVisible()
    await expect(
      page.getByRole('button', { name: `Open report for Candidate recipe` }),
    ).toBeVisible()
    await expect(page.getByRole('button', { name: 'Delete Candidate recipe' })).toHaveCount(0)
    await page.getByRole('button', { name: `Open report for Candidate recipe` }).click()
    await expect(page.getByRole('heading', { name: 'Candidate recipe' })).toBeVisible()
  })

  test('keeps release decision inputs progressive and touch discoverable at 320px', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 320, height: 568 })
    await mockEvaluationPlane(page)
    await page.goto('/evaluation?view=compare')

    const summary = page.locator('details > summary').filter({
      has: page.getByText('Prepare a release decision', { exact: true }),
    })
    const disclosure = summary.locator('..')
    const primarySelects = page.locator('#evaluation-panel select:visible')
    await expect(primarySelects).toHaveCount(1)
    const primarySelectStyles = await primarySelects.evaluateAll((elements) =>
      elements.map((element) => {
        const style = getComputedStyle(element)
        return {
          backgroundColor: style.backgroundColor,
          borderRadius: style.borderRadius,
          height: element.getBoundingClientRect().height,
        }
      }),
    )
    expect(new Set(primarySelectStyles.map((style) => style.backgroundColor)).size).toBe(1)
    expect(new Set(primarySelectStyles.map((style) => style.borderRadius)).size).toBe(1)
    expect(new Set(primarySelectStyles.map((style) => style.height)).size).toBe(1)
    await expect(disclosure).not.toHaveAttribute('open', '')
    await expect(disclosure.locator('select:visible')).toHaveCount(0)
    await expect
      .poll(() => summary.evaluate((element) => getComputedStyle(element, '::after').content))
      .not.toBe('none')
    await summary.click()
    await expect(disclosure).toHaveAttribute('open', '')
    await expect(page.getByLabel('Release decision change type')).toBeVisible()
    const inputSummary = disclosure.locator('details > summary').filter({
      has: page.getByText('Review evaluation inputs', { exact: true }),
    })
    const inputs = inputSummary.locator('..')
    await expect(inputs).not.toHaveAttribute('open', '')
    await inputSummary.click()
    await expect(page.getByRole('region', { name: 'Release decision inputs' })).toBeVisible()
    await expectNoHorizontalOverflow(page)
    await inputSummary.click()
    await expect(inputs).not.toHaveAttribute('open', '')
  })

  test('keeps native radio and checkbox inline width outside the shared field skin', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 320, height: 568 })
    await mockEvaluationPlane(page)
    await page.goto('/evaluation?view=new')

    for (const control of [page.getByRole('radio').first(), page.getByRole('checkbox').first()]) {
      await expect(control).toBeVisible()
      await control.hover()
      await control.focus()
      const width = await control.evaluate((element) => element.getBoundingClientRect().width)
      expect(width).toBeLessThanOrEqual(24)
    }
    await expectNoHorizontalOverflow(page)
  })
})
