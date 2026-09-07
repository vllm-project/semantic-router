import { readdirSync, readFileSync } from 'node:fs'

import { describe, expect, it } from 'vitest'

function readProductionSources(root: URL, include: (name: string) => boolean): string[] {
  return readdirSync(root, { withFileTypes: true }).flatMap((entry) => {
    const resource = new URL(entry.name + (entry.isDirectory() ? '/' : ''), root)
    if (entry.isDirectory()) return readProductionSources(resource, include)
    if (
      !/\.(?:ts|tsx)$/.test(entry.name) ||
      entry.name.includes('.test.') ||
      !include(entry.name)
    ) {
      return []
    }
    return readFileSync(resource, 'utf8')
  })
}

const sourceRoot = new URL('../', import.meta.url)
const evaluationBrowserSources = [
  ...readProductionSources(new URL('components/evaluation-plane/', sourceRoot), () => true),
  ...readProductionSources(new URL('pages/', sourceRoot), (name) =>
    /^(?:Evaluation|useEvaluation)/.test(name),
  ),
  ...readProductionSources(new URL('types/', sourceRoot), (name) => name.startsWith('evaluation')),
  ...readProductionSources(new URL('utils/', sourceRoot), (name) => name.startsWith('evaluation')),
]
const evaluationBrowserSource = evaluationBrowserSources.join('\n')

describe('Evaluation browser security boundary', () => {
  it('keeps hidden grading material outside the browser bundle', () => {
    expect(evaluationBrowserSource).not.toMatch(
      /casegrading|hidden[_ ]?(?:label|grading)|answer[_ ]?key|reference[_ ]?answer/i,
    )
  })

  it('keeps technical disclosure behavior in one typed boundary', () => {
    expect(evaluationBrowserSource.match(/data-evaluation-technical-details=/g)).toHaveLength(1)
    evaluationBrowserSources.forEach((source) => {
      expect(source).not.toMatch(
        /<EvaluationDisclosure\b[^>]*summary=(?:"Technical details"|\{`Technical details)/s,
      )
    })
  })
})
