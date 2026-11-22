/**
 * Helpers for AsciiOcean's quietRects: the boxes the hero copy covers, in viewport coordinates.
 */

/** The line boxes of an element's text, one per line: tighter than its box when centred lines are ragged. */
export function textRects(el: Element): DOMRect[] {
  const lines: DOMRect[] = []
  const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT)
  const range = document.createRange()
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    if (!node.textContent?.trim()) continue
    range.selectNodeContents(node)
    for (const r of Array.from(range.getClientRects())) {
      if (r.width < 1 || r.height < 1) continue
      const line = lines.find(l => Math.min(l.bottom, r.bottom) - Math.max(l.top, r.top) > Math.min(l.height, r.height) / 2)
      if (!line) {
        lines.push(DOMRect.fromRect(r))
        continue
      }
      const left = Math.min(line.left, r.left)
      const top = Math.min(line.top, r.top)
      line.width = Math.max(line.right, r.right) - left
      line.height = Math.max(line.bottom, r.bottom) - top
      line.x = left
      line.y = top
    }
  }
  return lines
}

/**
 * The rects of the elements under root marked data-ocean-quiet: the line boxes of their text for
 * data-ocean-quiet="text" (a headline, a line of copy), else their border boxes (a button, a row of them).
 */
export function quietRectsIn(root: Element | null | undefined, selector = '[data-ocean-quiet]'): DOMRect[] {
  if (!root) return []
  const rects: DOMRect[] = []
  for (const el of Array.from(root.querySelectorAll(selector))) {
    if (el.getAttribute('data-ocean-quiet') === 'text') rects.push(...textRects(el))
    else rects.push(el.getBoundingClientRect())
  }
  return rects
}
