import { expect, type Locator, type Page } from '@playwright/test'

export async function captureEvaluationSurface(page: Page, name: string) {
  const directory = process.env.EVALUATION_VISUAL_CAPTURE_DIR
  if (!directory) return
  await page.screenshot({ path: `${directory}/${name}.png` })
}

export async function captureEvaluationFullPage(page: Page, name: string) {
  const directory = process.env.EVALUATION_VISUAL_CAPTURE_DIR
  if (!directory) return
  await page.screenshot({ path: `${directory}/${name}.png`, fullPage: true })
}

export async function captureEvaluationElement(element: Locator, name: string) {
  const directory = process.env.EVALUATION_VISUAL_CAPTURE_DIR
  if (!directory) return
  await element.screenshot({ path: `${directory}/${name}.png` })
}

export async function expectNoHorizontalOverflow(page: Page) {
  await expect
    .poll(() =>
      page.evaluate(
        () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
      ),
    )
    .toBeLessThanOrEqual(1)
}

export async function expectPageBottomReachable(page: Page) {
  await expect
    .poll(() =>
      page.evaluate(() => {
        const root = document.scrollingElement
        if (!root) return false
        return (
          root.scrollHeight + 1 >= document.body.scrollHeight &&
          root.scrollHeight + 1 >= (document.getElementById('root')?.scrollHeight || 0)
        )
      }),
    )
    .toBe(true)
  await page.evaluate(() => {
    const root = document.scrollingElement
    if (!root) throw new Error('Document has no scrolling element.')
    root.scrollTop = root.scrollHeight
  })
  await expect
    .poll(() =>
      page.evaluate(() => {
        const root = document.scrollingElement
        if (!root) return false
        return Math.ceil(root.scrollTop + root.clientHeight) >= root.scrollHeight - 1
      }),
    )
    .toBe(true)
}

export async function expectEvaluationBottomGutter(page: Page) {
  const panel = page.getByRole('tabpanel')
  await expect(panel).toBeVisible()
  const geometry = await panel.evaluate((element) => {
    const panelRect = element.getBoundingClientRect()
    const lastChild = element.lastElementChild
    const lastRect = lastChild?.getBoundingClientRect()
    return {
      paddingBottom: Number.parseFloat(getComputedStyle(element).paddingBottom),
      contentGap: lastRect ? panelRect.bottom - lastRect.bottom : 0,
    }
  })
  expect(geometry.paddingBottom).toBeGreaterThanOrEqual(47)
  expect(geometry.contentGap).toBeGreaterThanOrEqual(geometry.paddingBottom - 1)
}

const INTERNAL_EVALUATION_UI_PATTERN =
  /\b(?:E[0-5]|G[0-9])\b|E0\s*[–-]\s*E5|(?:schema|contract)_version|evaluation-release-gates|Schema evaluation|Contract range|Evidence needed|\b(?:evaluation-smoke|live-mom-core|normalized-promotion-cohort)\b|\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b/i

export async function expectProductEvaluationLanguage(page: Page) {
  const visibleMainText = await page.locator('main').innerText()
  expect(visibleMainText).not.toMatch(INTERNAL_EVALUATION_UI_PATTERN)
}

export async function expectOverviewActionParity(page: Page) {
  const readiness = page.locator('section[aria-labelledby="evaluation-readiness-title"]')
  const geometry = await Promise.all(
    ['New experiment', 'Inspect runs'].map(async (name) => {
      const box = await readiness.getByRole('button', { name, exact: true }).boundingBox()
      return box?.height || 0
    }),
  )
  expect(geometry[0]).toBeGreaterThan(0)
  expect(geometry[0]).toBe(geometry[1])
}

export async function expectRunsWorkspaceLayout(page: Page) {
  const viewportWidth = await page.evaluate(() => window.innerWidth)
  const inspector = page.getByRole('complementary', { name: 'Selected evaluation run' })
  const workspace = inspector.locator('..')
  const history = workspace.locator(':scope > div').first()
  const [historyBox, inspectorBox] = await Promise.all([
    history.boundingBox(),
    inspector.boundingBox(),
  ])
  expect(historyBox).not.toBeNull()
  expect(inspectorBox).not.toBeNull()
  if (!historyBox || !inspectorBox) return

  if (viewportWidth > 1160) {
    expect(inspectorBox.x - (historyBox.x + historyBox.width)).toBeGreaterThanOrEqual(24)
    expect(Math.abs(inspectorBox.y - historyBox.y)).toBeLessThanOrEqual(2)
  } else {
    expect(inspectorBox.y - (historyBox.y + historyBox.height)).toBeGreaterThanOrEqual(24)
    expect(Math.abs(inspectorBox.x - historyBox.x)).toBeLessThanOrEqual(2)
  }
}

export async function expectDefaultCompareWorkspace(page: Page) {
  const panel = page.getByRole('tabpanel')
  const candidate = page.getByLabel('Comparison candidate', { exact: true })
  await expect(candidate).toBeVisible()
  await expect(candidate).toBeEnabled()
  await expect(panel.locator('select:visible:not(:disabled)')).toHaveCount(1)

  const releaseDecisionSummary = page.locator('details > summary').filter({
    has: page.getByText('Prepare a release decision', { exact: true }),
  })
  const releaseDecision = releaseDecisionSummary.locator('..')
  await expect(releaseDecision).not.toHaveAttribute('open', '')
  await expect(releaseDecision.locator('select:visible')).toHaveCount(0)
}

export async function expectDialogBottomReachable(page: Page, dialog: Locator) {
  await expect(dialog).toBeVisible()
  await expect
    .poll(async () => {
      const [box, viewportHeight] = await Promise.all([
        dialog.boundingBox(),
        page.evaluate(() => window.innerHeight),
      ])
      return Boolean(box && box.y >= -1 && box.y + box.height <= viewportHeight + 1)
    })
    .toBe(true)
  const controls = await dialog.locator('button:visible').evaluateAll((elements) =>
    elements.map((element) => ({
      height: Math.round(element.getBoundingClientRect().height),
      borderRadius: getComputedStyle(element).borderRadius,
      whiteSpace: getComputedStyle(element).whiteSpace,
    })),
  )
  for (const control of controls) {
    expect(control.height).toBe(40)
    expect(control.borderRadius).toBe('6px')
    expect(control.whiteSpace).toBe('nowrap')
  }
  const confirmation = dialog.locator('input:visible')
  if ((await confirmation.count()) > 0) {
    await expect
      .poll(() =>
        confirmation.first().evaluate((element) => ({
          height: Math.round(element.getBoundingClientRect().height),
          borderRadius: getComputedStyle(element).borderRadius,
        })),
      )
      .toEqual({ height: 40, borderRadius: '6px' })
  }
  await dialog.evaluate((element) => {
    element.scrollTop = element.scrollHeight
  })
  await expect
    .poll(() =>
      dialog.evaluate(
        (element) =>
          Math.ceil(element.scrollTop + element.clientHeight) >= element.scrollHeight - 1,
      ),
    )
    .toBe(true)
}

export async function expectKeyboardScrollable(region: Locator, axis: 'vertical' | 'horizontal') {
  const scrollProperty = axis === 'vertical' ? 'scrollTop' : 'scrollLeft'
  const sizeProperty = axis === 'vertical' ? 'scrollHeight' : 'scrollWidth'
  const clientProperty = axis === 'vertical' ? 'clientHeight' : 'clientWidth'
  await expect
    .poll(() =>
      region.evaluate(
        (element, properties) =>
          element[properties.sizeProperty as 'scrollHeight'] >
          element[properties.clientProperty as 'clientHeight'],
        { sizeProperty, clientProperty },
      ),
    )
    .toBe(true)
  await region.evaluate((element, property) => {
    element[property as 'scrollTop'] = 0
  }, scrollProperty)
  await region.focus()
  await region.press(axis === 'vertical' ? 'ArrowDown' : 'ArrowRight')
  await expect
    .poll(() =>
      region.evaluate((element, property) => element[property as 'scrollTop'], scrollProperty),
    )
    .toBeGreaterThan(0)
  await region.evaluate(async (element, property) => {
    if (element instanceof HTMLElement) element.blur()
    await new Promise((resolve) => window.setTimeout(resolve, 250))
    element[property as 'scrollTop'] = 0
  }, scrollProperty)
  await expect
    .poll(() =>
      region.evaluate((element, property) => element[property as 'scrollTop'], scrollProperty),
    )
    .toBe(0)
}

export async function expectScrollRegionsKeyboardReachable(page: Page) {
  const regions = page.locator('main [role="region"][tabindex="0"]:visible')
  for (let index = 0; index < (await regions.count()); index += 1) {
    const region = regions.nth(index)
    const overflow = await region.evaluate((element) => ({
      horizontal: element.scrollWidth > element.clientWidth,
      vertical: element.scrollHeight > element.clientHeight,
    }))
    if (overflow.horizontal) await expectKeyboardScrollable(region, 'horizontal')
    if (overflow.vertical) await expectKeyboardScrollable(region, 'vertical')
  }
}
