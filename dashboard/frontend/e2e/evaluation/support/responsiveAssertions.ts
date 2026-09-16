import { expect, type Page } from '@playwright/test'

import { EVALUATION_RUN_IDS } from './mixtureFixture'
import {
  captureEvaluationFullPage,
  captureEvaluationSurface,
  expectDefaultCompareWorkspace,
  expectEvaluationBottomGutter,
  expectNoHorizontalOverflow,
  expectOverviewActionParity,
  expectPageBottomReachable,
  expectProductEvaluationLanguage,
  expectRunsWorkspaceLayout,
  expectScrollRegionsKeyboardReachable,
} from './pageAssertions'
import { expectEvaluationControlSystem } from './visualAssertions'

export const responsiveEvaluationSurfaces = [
  { tab: 'Overview', route: '/evaluation', visibleText: 'Latest decision', capture: 'overview' },
  {
    tab: 'New experiment',
    route: '/evaluation?view=new',
    visibleText: 'New evaluation experiment',
    capture: 'new-experiment',
  },
  { tab: 'Runs', route: '/evaluation?view=runs', visibleText: 'Evaluation runs', capture: 'runs' },
  {
    tab: 'Reports',
    route: `/evaluation?view=reports&report=${EVALUATION_RUN_IDS.candidate}`,
    visibleText: 'Reports',
    capture: 'reports',
  },
  {
    tab: 'Compare',
    route: '/evaluation?view=compare',
    visibleText: 'Compare a candidate with its baseline',
    capture: 'compare',
  },
] as const

export async function expectResponsiveEvaluationSurface(
  page: Page,
  surface: (typeof responsiveEvaluationSurfaces)[number],
  viewportName: string,
) {
  const mobileViewport = viewportName.startsWith('mobile')
  await page.goto(surface.route)
  await expect(page.getByText(surface.visibleText, { exact: true }).first()).toBeVisible()
  const brand = page.getByRole('link', { name: 'vLLM Semantic Router home' })
  await expect(brand).toBeVisible()
  await expect
    .poll(async () => (await brand.boundingBox())?.y ?? -Infinity)
    .toBeGreaterThanOrEqual(0)
  const hero = page
    .getByRole('heading', { name: 'Evaluation', exact: true })
    .locator('xpath=ancestor::header[1]')
  await expect(hero).toBeVisible()
  const pageShell = hero.locator('xpath=ancestor::section[1]/..')
  await expect
    .poll(async () => {
      const [heroBox, shellBox, paddingTop] = await Promise.all([
        hero.boundingBox(),
        pageShell.boundingBox(),
        pageShell.evaluate((element) => Number.parseFloat(getComputedStyle(element).paddingTop)),
      ])
      return heroBox && shellBox ? Math.abs(heroBox.y - shellBox.y - paddingTop) : Infinity
    })
    .toBeLessThanOrEqual(1)
  if (mobileViewport) {
    await expect.poll(async () => (await hero.boundingBox())?.height ?? Infinity).toBeLessThan(190)
  }
  await expect(page.getByRole('tablist', { name: 'Evaluation views' })).toBeVisible()
  await expect
    .poll(async () => {
      const [tab, tablist] = await Promise.all([
        page.getByRole('tab', { name: surface.tab, exact: true }).boundingBox(),
        page.getByRole('tablist', { name: 'Evaluation views' }).boundingBox(),
      ])
      return Boolean(
        tab &&
          tablist &&
          tab.x >= tablist.x - 1 &&
          tab.x + tab.width <= tablist.x + tablist.width + 1,
      )
    })
    .toBe(true)
  await expect(page.getByRole('button', { name: /product guide/i })).toHaveCount(0)
  await expect
    .poll(() =>
      page.evaluate(() =>
        Math.max(window.scrollY, document.documentElement.scrollTop, document.body.scrollTop),
      ),
    )
    .toBeLessThanOrEqual(1)
  await expectNoHorizontalOverflow(page)
  await expectProductEvaluationLanguage(page)
  await expectEvaluationControlSystem(page)
  if (surface.capture === 'overview') await expectOverviewActionParity(page)
  if (surface.capture === 'runs') await expectRunsWorkspaceLayout(page)
  if (surface.capture === 'compare') await expectDefaultCompareWorkspace(page)
  await captureEvaluationSurface(page, `${surface.capture}-${viewportName}`)
  if (viewportName === 'desktop') {
    await captureEvaluationFullPage(page, `${surface.capture}-${viewportName}-full`)
  }
  await expectScrollRegionsKeyboardReachable(page)
  await expectPageBottomReachable(page)
  await expectEvaluationBottomGutter(page)
  await captureEvaluationSurface(page, `${surface.capture}-${viewportName}-bottom`)
}
