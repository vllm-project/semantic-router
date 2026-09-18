import { mkdirSync, rmSync, writeFileSync } from 'node:fs'
import { dirname } from 'node:path'

// Display names are not unique for it.each cases. Vitest's public TestCase.id
// identifies the collected task independently of its eventual test result.
export default class EvidenceReporter {
  expected = []
  cases = []

  onInit() {
    this.output = process.env.VITEST_EVIDENCE_PATH
    if (!this.output) throw new Error('VITEST_EVIDENCE_PATH is required')
    rmSync(this.output, { force: true })
  }

  onTestModuleCollected(module) {
    for (const test of module.children.allTests()) {
      this.expected.push(`vitest:${test.id}`)
    }
  }

  onTestCaseResult(test) {
    this.cases.push({
      id: `vitest:${test.id}`,
      name: test.fullName,
      status: test.result().state,
    })
  }

  onTestRunEnd(_modules, errors, reason) {
    const evidence = { expected_cases: this.expected, cases: this.cases }
    mkdirSync(dirname(this.output), { recursive: true })
    writeFileSync(this.output, `${JSON.stringify(evidence, null, 2)}\n`)

    const expected = new Set(this.expected)
    const actual = new Set(this.cases.map(test => test.id))
    if (!this.expected.length) throw new Error('empty Vitest collection')
    if (expected.size !== this.expected.length || actual.size !== this.cases.length) {
      throw new Error('duplicate Vitest task identity')
    }
    if (expected.size !== actual.size || [...expected].some(id => !actual.has(id))) {
      throw new Error('Vitest execution differs from collected inventory')
    }
    if (this.cases.some(test => test.status !== 'passed')) {
      throw new Error('required Vitest case did not pass')
    }
    if (reason !== 'passed' || errors.length) {
      throw new Error('Vitest run did not pass')
    }
  }
}
