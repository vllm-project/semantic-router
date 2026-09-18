import { expect, type Locator, type Page } from '@playwright/test'

export async function expectEvaluationContrastContract(page: Page) {
  const ratios = await page.getByTestId('evaluation-scope').evaluate((element) => {
    const style = getComputedStyle(element)
    const parseHex = (value: string) => {
      const match = value.trim().match(/^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i)
      if (!match) throw new Error(`Evaluation contrast token must be an opaque hex color: ${value}`)
      return match.slice(1).map((channel) => Number.parseInt(channel, 16) / 255)
    }
    const luminance = (value: string) =>
      parseHex(value)
        .map((channel) =>
          channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4,
        )
        .reduce((total, channel, index) => total + channel * [0.2126, 0.7152, 0.0722][index], 0)
    const contrast = (foreground: string, background: string) => {
      const values = [luminance(foreground), luminance(background)].sort(
        (left, right) => right - left,
      )
      return (values[0] + 0.05) / (values[1] + 0.05)
    }
    const foregrounds = {
      muted: style.getPropertyValue('--text-muted'),
      secondary: style.getPropertyValue('--text-secondary'),
      accent: style.getPropertyValue('--evaluation-accent-text'),
    }
    const backgrounds = {
      canvas: style.getPropertyValue('--surface-canvas'),
      shell: style.getPropertyValue('--surface-shell'),
      panel: style.getPropertyValue('--surface-panel'),
      raised: style.getPropertyValue('--surface-raised'),
    }
    return Object.entries(foregrounds).flatMap(([foregroundName, foreground]) =>
      Object.entries(backgrounds).map(([backgroundName, background]) => ({
        pair: `${foregroundName}/${backgroundName}`,
        ratio: contrast(foreground, background),
      })),
    )
  })
  for (const result of ratios) {
    expect(result.ratio, `${result.pair} contrast`).toBeGreaterThanOrEqual(4.5)
  }

  const unavailableReasons = await page
    .locator('[data-evaluation-unavailable-reason="true"]:visible')
    .evaluateAll((elements) =>
      elements.map((element) => {
        let cumulativeOpacity = 1
        let current: Element | null = element
        while (current) {
          cumulativeOpacity *= Number.parseFloat(getComputedStyle(current).opacity)
          if (current.hasAttribute('data-testid')) break
          current = current.parentElement
        }
        return {
          cumulativeOpacity,
          color: getComputedStyle(element).color,
          expectedColor: getComputedStyle(element).getPropertyValue('--text-secondary').trim(),
        }
      }),
    )
  for (const reason of unavailableReasons) {
    expect(reason.cumulativeOpacity).toBe(1)
    expect(reason.color).toBe('rgb(178, 178, 184)')
    expect(reason.expectedColor).toBe('#b2b2b8')
  }
}

export async function expectEvaluationControlSystem(page: Page) {
  const panel = page.getByRole('tabpanel')
  const selects = panel.locator('select:visible')
  const selectGeometry = await selects.evaluateAll((elements) =>
    elements.map((element) => {
      const style = getComputedStyle(element)
      return {
        backgroundColor: style.backgroundColor,
        borderRadius: style.borderRadius,
        height: Math.round(element.getBoundingClientRect().height),
      }
    }),
  )
  for (const geometry of selectGeometry) {
    expect(geometry.height).toBe(40)
  }
  if (selectGeometry.length > 1) {
    expect(new Set(selectGeometry.map((geometry) => geometry.backgroundColor)).size).toBe(1)
    expect(new Set(selectGeometry.map((geometry) => geometry.borderRadius)).size).toBe(1)
  }

  const fieldGeometry = await panel
    .locator(
      'input:visible:not([type="checkbox"]):not([type="radio"]):not([type="range"]):not([type="hidden"])',
    )
    .evaluateAll((elements) =>
      elements.map((element) => ({
        label:
          element.getAttribute('aria-label') ||
          element.getAttribute('name') ||
          element.getAttribute('placeholder') ||
          element.tagName.toLowerCase(),
        height: Math.round(element.getBoundingClientRect().height),
        borderRadius: getComputedStyle(element).borderRadius,
      })),
    )
  for (const geometry of fieldGeometry) {
    expect(geometry.height, `${geometry.label} field height`).toBe(40)
    expect(geometry.borderRadius, `${geometry.label} field radius`).toBe('6px')
  }

  const actions = panel.locator('[data-evaluation-action="true"]:visible')
  const actionGeometry = await actions.evaluateAll((elements) =>
    elements.map((element) => ({
      density: element.getAttribute('data-density'),
      height: Math.round(element.getBoundingClientRect().height),
      borderRadius: getComputedStyle(element).borderRadius,
      whiteSpace: getComputedStyle(element).whiteSpace,
    })),
  )
  for (const geometry of actionGeometry) {
    expect(geometry.height).toBe(geometry.density === 'compact' ? 34 : 40)
    expect(geometry.borderRadius).toBe('6px')
    expect(geometry.whiteSpace).toBe('nowrap')
  }

  const tagGeometry = await panel
    .locator('[data-evaluation-tag="true"]:visible')
    .evaluateAll((elements) =>
      elements.map((element) => ({
        height: Math.round(element.getBoundingClientRect().height),
        borderRadius: getComputedStyle(element).borderRadius,
        whiteSpace: getComputedStyle(element).whiteSpace,
      })),
    )
  for (const geometry of tagGeometry) {
    expect(geometry.height).toBe(22)
    expect(geometry.borderRadius).toBe('999px')
    expect(geometry.whiteSpace).toBe('nowrap')
  }

  const navigationGeometry = await page
    .locator('[data-evaluation-navigation-tab="true"]:visible')
    .evaluateAll((elements) =>
      elements.map((element) => ({
        height: Math.round(element.getBoundingClientRect().height),
        borderRadius: getComputedStyle(element).borderRadius,
      })),
    )
  expect(navigationGeometry.length).toBeGreaterThan(0)
  for (const geometry of navigationGeometry) {
    expect(geometry.height).toBeGreaterThanOrEqual(40)
    expect(geometry.height).toBeLessThanOrEqual(44)
    expect(geometry.borderRadius).toBe('0px')
  }
  expect(new Set(navigationGeometry.map((geometry) => geometry.height)).size).toBe(1)

  const ledgerGeometry = await panel
    .locator('[data-evaluation-ledger-row="true"]:visible')
    .evaluateAll((elements) =>
      elements.map((element) => ({
        height: Math.round(element.getBoundingClientRect().height),
        borderRadius: getComputedStyle(element).borderRadius,
        parentRuleWidth: getComputedStyle(element.parentElement as HTMLElement).borderBottomWidth,
      })),
    )
  for (const geometry of ledgerGeometry) {
    expect(geometry.height).toBeGreaterThanOrEqual(82)
    expect(geometry.borderRadius).toBe('0px')
    expect(geometry.parentRuleWidth).toBe('1px')
  }
  if (ledgerGeometry.length > 1) {
    expect(new Set(ledgerGeometry.map((geometry) => geometry.height)).size).toBe(1)
  }

  const ledgerRows = panel.locator('[data-evaluation-ledger-row="true"]:visible')
  if ((await ledgerRows.count()) > 0) {
    const row = ledgerRows.first()
    const before = await row.boundingBox()
    await row.focus()
    const focused = await row.evaluate((element) => {
      const style = getComputedStyle(element)
      const rect = element.getBoundingClientRect()
      const scrollport = element.closest('ol')?.getBoundingClientRect()
      const width = Number.parseFloat(style.outlineWidth)
      const offset = Number.parseFloat(style.outlineOffset)
      const expansion = Math.max(0, width + offset)
      return {
        focusVisible: element.matches(':focus-visible'),
        outlineStyle: style.outlineStyle,
        outlineWidth: width,
        outlineOffset: style.outlineOffset,
        paintContained:
          !scrollport ||
          (rect.left - expansion >= scrollport.left - 1 &&
            rect.right + expansion <= scrollport.right + 1),
      }
    })
    const after = await row.boundingBox()
    expect(focused.focusVisible).toBe(true)
    expect(focused.outlineStyle).not.toBe('none')
    expect(focused.outlineWidth).toBeGreaterThanOrEqual(1)
    expect(focused.outlineOffset).toBe('-2px')
    expect(focused.paintContained).toBe(true)
    expect(after?.width).toBe(before?.width)
    expect(after?.height).toBe(before?.height)
  }

  const actionGroups = panel.getByTestId('evaluation-run-actions')
  for (let index = 0; index < (await actionGroups.count()); index += 1) {
    const buttons = actionGroups.nth(index).locator('button:visible')
    const geometry = await buttons.evaluateAll((elements) =>
      elements.map((element) => {
        const rect = element.getBoundingClientRect()
        return {
          left: rect.left,
          right: rect.right,
          top: rect.top,
          bottom: rect.bottom,
          marginLeft: getComputedStyle(element).marginLeft,
        }
      }),
    )
    for (const button of geometry) expect(button.marginLeft).toBe('0px')
    for (let buttonIndex = 1; buttonIndex < geometry.length; buttonIndex += 1) {
      const previous = geometry[buttonIndex - 1]
      const current = geometry[buttonIndex]
      const sameRow = Math.abs(previous.top - current.top) <= 1
      const gap = sameRow ? current.left - previous.right : current.top - previous.bottom
      expect(gap, 'run inspector actions remain one visual group').toBeGreaterThanOrEqual(0)
      expect(gap, 'run inspector actions remain one visual group').toBeLessThanOrEqual(12)
    }
  }

  const disclosures = panel.locator(
    '[data-evaluation-report-disclosure="true"]:visible > summary:visible',
  )
  if ((await disclosures.count()) > 0) {
    const summary = disclosures.first()
    await summary.focus()
    await page.keyboard.press('Tab')
    await page.keyboard.press('Shift+Tab')
    await expect(summary).toBeFocused()
    const focus = await summary.evaluate((element) => {
      const style = getComputedStyle(element)
      return {
        focusVisible: element.matches(':focus-visible'),
        outlineStyle: style.outlineStyle,
        outlineWidth: Number.parseFloat(style.outlineWidth),
        outlineOffset: style.outlineOffset,
      }
    })
    expect(focus.focusVisible).toBe(true)
    expect(focus.outlineStyle).not.toBe('none')
    expect(focus.outlineWidth).toBeGreaterThanOrEqual(1)
    expect(focus.outlineOffset).toBe('-3px')
  }

  await expectEvaluationContrastContract(page)
}

export async function expectCompactVerticalFlow(container: Locator) {
  const geometry = await container.evaluate((element) => {
    const containerRect = element.getBoundingClientRect()
    const childRects = Array.from(element.children)
      .filter((child) => getComputedStyle(child).display !== 'none')
      .map((child) => child.getBoundingClientRect())
    const gaps = childRects.slice(1).map((rect, index) => rect.top - childRects[index].bottom)
    return {
      childCount: childRects.length,
      topInset: childRects.length ? childRects[0].top - containerRect.top : Infinity,
      bottomInset: childRects.length
        ? containerRect.bottom - childRects[childRects.length - 1].bottom
        : Infinity,
      maximumGap: gaps.length ? Math.max(...gaps) : 0,
    }
  })

  expect(geometry.childCount).toBeGreaterThan(1)
  expect(geometry.topInset).toBeLessThanOrEqual(32)
  expect(geometry.bottomInset).toBeLessThanOrEqual(32)
  expect(geometry.maximumGap).toBeLessThanOrEqual(40)
}
