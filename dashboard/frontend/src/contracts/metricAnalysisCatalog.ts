// The imported JSON mirror is generated from Python package data by
// tools/ci/sync_evaluation_catalogs.py; this module owns browser-side validation.
import catalogDocument from './metric_analysis_catalog.v1.json' with { type: 'json' }
import {
  decodeSubjectID,
  encodeSubjectID,
  type IdentifierEncoding,
  validateEncoding,
} from './metricAnalysisIdentifierEncoding'

export const METRIC_ANALYSIS_CATALOG_SCHEMA_VERSION = 'metric-analysis-catalog.v1'
export const METRIC_ANALYSIS_CONTRACT_VERSION = 'metric-analysis.v1'

const TRACK_IDS = new Set([
  'agentic',
  'capacity',
  'joint',
  'model_pool',
  'multimodal',
  'preference',
  'routing',
  'safety',
])
const PROJECTION_SOURCES = new Set([
  'capacity_load_plan',
  'compound_budget_plan',
  'evaluation_case_plan',
  'frozen_model_pool_matrix',
  'method_ledger',
  'routing_recipe_plan',
])
const ANALYSIS_IDENTIFIER_PATTERN = /^[a-z0-9][a-z0-9.-]{0,159}$/

export interface MetricAnalysisPlannedUnitFilter {
  readonly field: string
  readonly capture: string
}

export interface MetricAnalysisPlannedUnitProjection {
  readonly source: string
  readonly track_id: string
  readonly coordinates: readonly string[]
  readonly required_dimensions?: readonly string[]
  readonly filters?: readonly MetricAnalysisPlannedUnitFilter[]
}

export interface MetricAnalysisCatalogSpecification {
  readonly id: string
  readonly track_id: string
  readonly estimator_id: string
  readonly estimator_version: string
  readonly analysis_unit: string
  readonly cluster_unit: string
  readonly weighting: string
  readonly missingness: 'fail_closed'
  readonly exclusion_policy: 'exclude_unavailable_evidence'
  readonly planned_unit_projection: MetricAnalysisPlannedUnitProjection
}

export interface MetricAnalysisCatalogMatch {
  readonly metric_id: string
  readonly family_id?: string
  readonly captures: Readonly<Record<string, string>>
  readonly specification: MetricAnalysisCatalogSpecification
}

interface StaticMetric {
  readonly id: string
  readonly analysis_ref: string
}

interface DynamicCapture {
  readonly name: string
  readonly group: number
  readonly type: 'encoded_portable_id' | 'positive_int' | 'enum'
  readonly values?: readonly string[]
  readonly minimum?: number
  readonly maximum?: number
}

interface DynamicVariant {
  readonly value: string
  readonly analysis_ref: string
}

interface DynamicExample {
  readonly metric_id: string
  readonly captures: Readonly<Record<string, string>>
  readonly analysis_ref: string
}

interface DynamicFamily {
  readonly id: string
  readonly literal_prefix: string
  readonly pattern: string
  readonly captures: readonly DynamicCapture[]
  readonly selector_capture: string
  readonly variants: readonly DynamicVariant[]
  readonly examples: readonly DynamicExample[]
}

interface MetricAnalysisCatalogDocument {
  readonly schema_version: string
  readonly provenance_contract_version: string
  readonly identifier_encoding: IdentifierEncoding
  readonly analysis_templates: readonly MetricAnalysisCatalogSpecification[]
  readonly static_metrics: readonly StaticMetric[]
  readonly dynamic_families: readonly DynamicFamily[]
}

interface CompiledFamily extends DynamicFamily {
  readonly compiled: RegExp
}

interface CatalogIndex {
  readonly document: MetricAnalysisCatalogDocument
  readonly templates: ReadonlyMap<string, MetricAnalysisCatalogSpecification>
  readonly staticMetrics: ReadonlyMap<string, StaticMetric>
  readonly families: readonly CompiledFamily[]
}

export class MetricAnalysisCatalogResolutionError extends Error {
  readonly kind: 'unknown' | 'ambiguous' | 'invalid'

  constructor(kind: 'unknown' | 'ambiguous' | 'invalid', metricID: string) {
    super(`${kind} evaluation metric id: ${metricID}`)
    this.name = 'MetricAnalysisCatalogResolutionError'
    this.kind = kind
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function assertCondition(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(`metric analysis catalog: ${message}`)
}

function assertExactKeys(
  value: Record<string, unknown>,
  expected: readonly string[],
  context: string,
) {
  const actual = Object.keys(value).sort()
  const wanted = [...expected].sort()
  assertCondition(
    actual.length === wanted.length && actual.every((key, index) => key === wanted[index]),
    `${context} fields are invalid`,
  )
}

function isSortedUnique(values: readonly string[]): boolean {
  return values.every((value, index) => index === 0 || values[index - 1] < value)
}

function sameStringRecord(
  left: Readonly<Record<string, string>>,
  right: Readonly<Record<string, string>>,
): boolean {
  const keys = Object.keys(left)
  return keys.length === Object.keys(right).length && keys.every((key) => left[key] === right[key])
}

function assertTrimmedText(value: unknown, context: string): asserts value is string {
  assertCondition(
    typeof value === 'string' && value.length > 0 && value.trim() === value,
    `${context} is invalid`,
  )
}

function assertStringArray(value: unknown, context: string): asserts value is string[] {
  assertCondition(Array.isArray(value), `${context} must be an array`)
  assertCondition(
    value.every((item) => typeof item === 'string' && item.length > 0),
    `${context} is invalid`,
  )
  assertCondition(new Set(value).size === value.length, `${context} must be unique`)
}

function validateProjection(value: unknown, captures?: ReadonlySet<string>) {
  assertCondition(isRecord(value), 'planned-unit projection must be an object')
  const allowed = ['coordinates', 'filters', 'required_dimensions', 'source', 'track_id']
  const required = ['coordinates', 'source', 'track_id']
  assertCondition(
    Object.keys(value).every((key) => allowed.includes(key)),
    'planned-unit projection fields are invalid',
  )
  assertCondition(
    required.every((key) => key in value),
    'planned-unit projection fields are incomplete',
  )
  assertCondition(
    PROJECTION_SOURCES.has(String(value.source)),
    'planned-unit projection source is invalid',
  )
  assertCondition(TRACK_IDS.has(String(value.track_id)), 'planned-unit projection track is invalid')
  assertStringArray(value.coordinates, 'planned-unit projection coordinates')
  assertCondition(value.coordinates.length > 0, 'planned-unit projection coordinates are empty')
  if (value.required_dimensions !== undefined) {
    assertStringArray(value.required_dimensions, 'planned-unit projection required dimensions')
  }
  if (value.filters === undefined) return
  assertCondition(Array.isArray(value.filters), 'planned-unit projection filters must be an array')
  const fields: string[] = []
  for (const filter of value.filters) {
    assertCondition(isRecord(filter), 'planned-unit projection filter must be an object')
    assertExactKeys(filter, ['capture', 'field'], 'planned-unit projection filter')
    assertTrimmedText(filter.field, 'planned-unit projection filter field')
    assertTrimmedText(filter.capture, 'planned-unit projection filter capture')
    assertCondition(
      captures === undefined || captures.has(filter.capture),
      'planned-unit projection filter capture is unknown',
    )
    fields.push(filter.field)
  }
  assertCondition(
    new Set(fields).size === fields.length,
    'planned-unit projection filter fields must be unique',
  )
}

function validateTemplate(
  value: unknown,
  captures?: ReadonlySet<string>,
): asserts value is MetricAnalysisCatalogSpecification {
  assertCondition(isRecord(value), 'analysis template must be an object')
  assertExactKeys(
    value,
    [
      'analysis_unit',
      'cluster_unit',
      'estimator_id',
      'estimator_version',
      'exclusion_policy',
      'id',
      'missingness',
      'planned_unit_projection',
      'track_id',
      'weighting',
    ],
    'analysis template',
  )
  assertCondition(
    typeof value.id === 'string' && ANALYSIS_IDENTIFIER_PATTERN.test(value.id),
    'analysis template id is invalid',
  )
  for (const field of [
    'estimator_id',
    'estimator_version',
    'analysis_unit',
    'cluster_unit',
    'weighting',
  ] as const) {
    assertTrimmedText(value[field], `analysis template ${field}`)
  }
  assertCondition(TRACK_IDS.has(String(value.track_id)), 'analysis template track is invalid')
  assertCondition(value.missingness === 'fail_closed', 'analysis template missingness is invalid')
  assertCondition(
    value.exclusion_policy === 'exclude_unavailable_evidence',
    'analysis template exclusion policy is invalid',
  )
  validateProjection(value.planned_unit_projection, captures)
}

function captureGroupCount(pattern: string): number {
  let count = 0
  let escaped = false
  let inCharacterClass = false
  for (let index = 0; index < pattern.length; index += 1) {
    const character = pattern[index]
    if (escaped) {
      escaped = false
      continue
    }
    if (character === '\\') {
      escaped = true
      continue
    }
    if (character === '[') inCharacterClass = true
    if (character === ']') inCharacterClass = false
    if (!inCharacterClass && character === '(' && pattern[index + 1] !== '?') count += 1
  }
  return count
}

function captureValues(
  family: CompiledFamily,
  match: RegExpExecArray,
  encoding: IdentifierEncoding,
): Record<string, string> {
  const result: Record<string, string> = {}
  for (const capture of family.captures) {
    const raw = match[capture.group]
    if (capture.type === 'encoded_portable_id') {
      decodeSubjectID(raw, encoding)
    } else if (capture.type === 'positive_int') {
      const number = Number(raw)
      if (
        !Number.isSafeInteger(number) ||
        String(number) !== raw ||
        number < (capture.minimum ?? 1) ||
        number > (capture.maximum ?? 0)
      ) {
        throw new MetricAnalysisCatalogResolutionError('invalid', family.id)
      }
    } else if (!capture.values?.includes(raw)) {
      throw new MetricAnalysisCatalogResolutionError('invalid', family.id)
    }
    result[capture.name] = raw
  }
  return result
}

function resolveFromIndex(metricID: string, index: CatalogIndex): MetricAnalysisCatalogMatch {
  if (typeof metricID !== 'string' || !metricID || metricID.trim() !== metricID) {
    throw new MetricAnalysisCatalogResolutionError('invalid', String(metricID))
  }
  const staticMetric = index.staticMetrics.get(metricID)
  if (staticMetric) {
    return {
      metric_id: metricID,
      captures: Object.freeze({}),
      specification: index.templates.get(staticMetric.analysis_ref)!,
    }
  }
  const matches: Array<{ family: CompiledFamily; captures: Record<string, string> }> = []
  for (const family of index.families) {
    const match = family.compiled.exec(metricID)
    if (!match) continue
    try {
      matches.push({
        family,
        captures: captureValues(family, match, index.document.identifier_encoding),
      })
    } catch {
      throw new MetricAnalysisCatalogResolutionError('invalid', metricID)
    }
  }
  if (matches.length === 0) throw new MetricAnalysisCatalogResolutionError('unknown', metricID)
  if (matches.length !== 1) throw new MetricAnalysisCatalogResolutionError('ambiguous', metricID)
  const { family, captures } = matches[0]
  const selector = captures[family.selector_capture]
  const variant =
    family.variants.find((item) => item.value === selector) ??
    family.variants.find((item) => item.value === '*')
  if (!variant) throw new MetricAnalysisCatalogResolutionError('invalid', metricID)
  return {
    metric_id: metricID,
    family_id: family.id,
    captures: Object.freeze(captures),
    specification: index.templates.get(variant.analysis_ref)!,
  }
}

function validateCatalogRoot(value: unknown): asserts value is Record<string, unknown> {
  assertCondition(isRecord(value), 'root must be an object')
  assertExactKeys(
    value,
    [
      'analysis_templates',
      'dynamic_families',
      'identifier_encoding',
      'provenance_contract_version',
      'schema_version',
      'static_metrics',
    ],
    'root',
  )
  assertCondition(
    value.schema_version === METRIC_ANALYSIS_CATALOG_SCHEMA_VERSION,
    'schema version is invalid',
  )
  assertCondition(
    value.provenance_contract_version === METRIC_ANALYSIS_CONTRACT_VERSION,
    'provenance version is invalid',
  )
}

function buildTemplateIndex(value: unknown): Map<string, MetricAnalysisCatalogSpecification> {
  assertCondition(
    Array.isArray(value) && value.length > 0,
    'analysis template inventory is invalid',
  )
  const templates = new Map<string, MetricAnalysisCatalogSpecification>()
  for (const template of value) {
    validateTemplate(template)
    assertCondition(!templates.has(template.id), `analysis template ${template.id} is duplicated`)
    templates.set(template.id, template)
  }
  assertCondition(
    isSortedUnique([...templates.keys()]),
    'analysis templates are not sorted and unique',
  )
  return templates
}

function validateFamilyCapture(
  value: unknown,
  index: number,
  familyID: string,
  captureNames: Set<string>,
) {
  assertCondition(isRecord(value), `dynamic family ${familyID} capture must be an object`)
  assertTrimmedText(value.name, `dynamic family ${familyID} capture name`)
  assertCondition(value.group === index + 1, `dynamic family ${familyID} capture group is invalid`)
  assertCondition(
    ['encoded_portable_id', 'positive_int', 'enum'].includes(String(value.type)),
    `dynamic family ${familyID} capture type is invalid`,
  )
  if (value.type === 'enum') {
    assertExactKeys(value, ['group', 'name', 'type', 'values'], 'enum capture')
    assertStringArray(value.values, 'enum capture values')
    assertCondition(isSortedUnique(value.values), 'enum capture values are not sorted')
  } else if (value.type === 'positive_int') {
    assertExactKeys(value, ['group', 'maximum', 'minimum', 'name', 'type'], 'integer capture')
    assertCondition(
      Number.isSafeInteger(value.minimum) &&
        Number.isSafeInteger(value.maximum) &&
        Number(value.minimum) >= 1 &&
        Number(value.maximum) >= Number(value.minimum),
      'integer capture bounds are invalid',
    )
  } else {
    assertExactKeys(value, ['group', 'name', 'type'], 'encoded-id capture')
  }
  assertCondition(!captureNames.has(value.name), `dynamic family ${familyID} capture is duplicated`)
  captureNames.add(value.name)
}

function validateFamilyCaptures(value: Record<string, unknown>): Set<string> {
  assertCondition(
    Array.isArray(value.captures) &&
      value.captures.length === captureGroupCount(String(value.pattern)),
    `dynamic family ${value.id} capture cardinality is invalid`,
  )
  const captureNames = new Set<string>()
  for (const [index, capture] of value.captures.entries()) {
    validateFamilyCapture(capture, index, String(value.id), captureNames)
  }
  return captureNames
}

function validateFamilyVariants(
  value: Record<string, unknown>,
  selector: Record<string, unknown>,
  templates: ReadonlyMap<string, MetricAnalysisCatalogSpecification>,
  captureNames: ReadonlySet<string>,
): DynamicVariant[] {
  assertCondition(
    Array.isArray(value.variants) && value.variants.length > 0,
    `dynamic family ${value.id} variants are missing`,
  )
  const variants: DynamicVariant[] = []
  for (const rawVariant of value.variants) {
    assertCondition(isRecord(rawVariant), `dynamic family ${value.id} variant must be an object`)
    assertExactKeys(rawVariant, ['analysis_ref', 'value'], 'dynamic variant')
    assertTrimmedText(rawVariant.value, 'dynamic variant value')
    assertTrimmedText(rawVariant.analysis_ref, 'dynamic variant analysis ref')
    const template = templates.get(rawVariant.analysis_ref)
    assertCondition(
      template !== undefined,
      `dynamic family ${value.id} references an unknown template`,
    )
    validateTemplate(template, captureNames)
    variants.push(rawVariant as unknown as DynamicVariant)
  }
  const variantValues = variants.map((variant) => variant.value)
  assertCondition(
    isSortedUnique(variantValues),
    `dynamic family ${value.id} variants are not sorted and unique`,
  )
  const expectedVariants = selector.type === 'enum' ? selector.values : ['*']
  assertCondition(
    Array.isArray(expectedVariants) &&
      variantValues.length === expectedVariants.length &&
      variantValues.every((item, index) => item === expectedVariants[index]),
    `dynamic family ${value.id} variants do not cover the selector`,
  )
  return variants
}

function validateFamilyExamples(
  value: Record<string, unknown>,
  templates: ReadonlyMap<string, MetricAnalysisCatalogSpecification>,
  captureNames: ReadonlySet<string>,
) {
  assertCondition(
    Array.isArray(value.examples) && value.examples.length > 0,
    `dynamic family ${value.id} examples are missing`,
  )
  for (const rawExample of value.examples) {
    assertCondition(isRecord(rawExample), `dynamic family ${value.id} example must be an object`)
    assertExactKeys(rawExample, ['analysis_ref', 'captures', 'metric_id'], 'dynamic example')
    assertTrimmedText(rawExample.metric_id, 'dynamic example metric id')
    assertTrimmedText(rawExample.analysis_ref, 'dynamic example analysis ref')
    assertCondition(
      templates.has(rawExample.analysis_ref),
      `dynamic family ${value.id} example references an unknown template`,
    )
    assertCondition(
      isRecord(rawExample.captures),
      `dynamic family ${value.id} example captures must be an object`,
    )
    assertCondition(
      Object.keys(rawExample.captures).length === captureNames.size &&
        Object.entries(rawExample.captures).every(
          ([name, capture]) => captureNames.has(name) && typeof capture === 'string',
        ),
      `dynamic family ${value.id} example captures are invalid`,
    )
  }
}

function compileFamily(
  value: unknown,
  templates: ReadonlyMap<string, MetricAnalysisCatalogSpecification>,
): CompiledFamily {
  assertCondition(isRecord(value), 'dynamic family must be an object')
  assertExactKeys(
    value,
    ['captures', 'examples', 'id', 'literal_prefix', 'pattern', 'selector_capture', 'variants'],
    'dynamic family',
  )
  assertTrimmedText(value.id, 'dynamic family id')
  assertTrimmedText(value.literal_prefix, 'dynamic family literal prefix')
  assertTrimmedText(value.pattern, 'dynamic family pattern')
  assertTrimmedText(value.selector_capture, 'dynamic family selector')
  assertCondition(
    value.pattern.startsWith('^') && value.pattern.endsWith('$'),
    `dynamic family ${value.id} pattern is not anchored`,
  )
  const compiled = new RegExp(value.pattern)
  const captureNames = validateFamilyCaptures(value)
  const captures = value.captures as unknown[]
  const selector = captures.find(
    (capture) => isRecord(capture) && capture.name === value.selector_capture,
  )
  assertCondition(isRecord(selector), `dynamic family ${value.id} selector capture is unknown`)
  const variants = validateFamilyVariants(value, selector, templates, captureNames)
  validateFamilyExamples(value, templates, captureNames)
  return {
    ...(value as unknown as DynamicFamily),
    captures: captures as DynamicCapture[],
    variants,
    compiled,
  }
}

function assertFamilyPrefixesDoNotOverlap(families: readonly CompiledFamily[]) {
  for (let left = 0; left < families.length; left += 1) {
    for (let right = left + 1; right < families.length; right += 1) {
      assertCondition(
        !families[left].literal_prefix.startsWith(families[right].literal_prefix) &&
          !families[right].literal_prefix.startsWith(families[left].literal_prefix),
        'dynamic family literal prefixes overlap',
      )
    }
  }
}

function buildFamilyIndex(
  value: unknown,
  templates: ReadonlyMap<string, MetricAnalysisCatalogSpecification>,
): CompiledFamily[] {
  assertCondition(
    Array.isArray(value) && value.length > 0,
    'dynamic family inventory is invalid',
  )
  const families = value.map((family) => compileFamily(family, templates))
  assertCondition(
    isSortedUnique(families.map((family) => family.id)),
    'dynamic family ids are not sorted and unique',
  )
  assertFamilyPrefixesDoNotOverlap(families)
  return families
}

function buildStaticMetricIndex(
  value: unknown,
  templates: ReadonlyMap<string, MetricAnalysisCatalogSpecification>,
  families: readonly CompiledFamily[],
): Map<string, StaticMetric> {
  assertCondition(
    Array.isArray(value) && value.length > 0,
    'static metric inventory is invalid',
  )
  const staticMetrics = new Map<string, StaticMetric>()
  for (const rawMetric of value) {
    assertCondition(isRecord(rawMetric), 'static metric must be an object')
    assertExactKeys(rawMetric, ['analysis_ref', 'id'], 'static metric')
    assertTrimmedText(rawMetric.id, 'static metric id')
    assertTrimmedText(rawMetric.analysis_ref, 'static metric analysis ref')
    const metricID = rawMetric.id
    const analysisRef = rawMetric.analysis_ref
    assertCondition(
      templates.has(analysisRef),
      `static metric ${metricID} references an unknown template`,
    )
    assertCondition(!staticMetrics.has(metricID), `static metric ${metricID} is duplicated`)
    assertCondition(
      !families.some((family) => family.compiled.test(metricID)),
      `static metric ${metricID} overlaps a dynamic family`,
    )
    const template = templates.get(analysisRef)!
    validateTemplate(template, new Set())
    staticMetrics.set(metricID, rawMetric as unknown as StaticMetric)
  }
  assertCondition(
    isSortedUnique([...staticMetrics.keys()]),
    'static metric ids are not sorted and unique',
  )
  return staticMetrics
}

function validateTemplateReferences(
  templates: ReadonlyMap<string, MetricAnalysisCatalogSpecification>,
  staticMetrics: ReadonlyMap<string, StaticMetric>,
  families: readonly CompiledFamily[],
) {
  const referenced = new Set<string>()
  for (const metric of staticMetrics.values()) referenced.add(metric.analysis_ref)
  for (const family of families) {
    for (const variant of family.variants) referenced.add(variant.analysis_ref)
  }
  assertCondition(
    referenced.size === templates.size && [...templates.keys()].every((id) => referenced.has(id)),
    'analysis templates must be referenced exhaustively',
  )
}

function validateFamilyGoldenExamples(index: CatalogIndex) {
  for (const family of index.families) {
    for (const example of family.examples) {
      assertCondition(
        example.metric_id.startsWith(family.literal_prefix),
        `dynamic family ${family.id} example has the wrong prefix`,
      )
      const resolved = resolveFromIndex(example.metric_id, index)
      assertCondition(
        resolved.family_id === family.id && resolved.specification.id === example.analysis_ref,
        `dynamic family ${family.id} golden example drifted`,
      )
      assertCondition(
        sameStringRecord(resolved.captures, example.captures),
        `dynamic family ${family.id} golden captures drifted`,
      )
    }
  }
}

function validateIdentifierEncodingVectors(document: MetricAnalysisCatalogDocument) {
  for (const vector of document.identifier_encoding.vectors) {
    assertCondition(
      encodeSubjectID(vector.raw, document.identifier_encoding) === vector.encoded,
      `identifier encoding vector ${vector.raw} drifted`,
    )
    assertCondition(
      decodeSubjectID(vector.encoded, document.identifier_encoding) === vector.raw,
      `identifier decoding vector ${vector.raw} drifted`,
    )
  }
}

function validateCatalogSource(source: string): CatalogIndex {
  const value: unknown = JSON.parse(source)
  validateCatalogRoot(value)
  validateEncoding(value.identifier_encoding)
  const templates = buildTemplateIndex(value.analysis_templates)
  const families = buildFamilyIndex(value.dynamic_families, templates)
  const staticMetrics = buildStaticMetricIndex(value.static_metrics, templates, families)
  validateTemplateReferences(templates, staticMetrics, families)
  const document = value as unknown as MetricAnalysisCatalogDocument
  const index: CatalogIndex = { document, templates, staticMetrics, families }
  validateFamilyGoldenExamples(index)
  validateIdentifierEncodingVectors(document)
  return index
}

// The imported JSON asset is the only browser/Node runtime source. The source
// tree parity test gates its bytes against the canonical Python package data
// and the Go embed; JSON import attributes work in Node, Vite, and Playwright.
export const METRIC_ANALYSIS_CATALOG_SOURCE = JSON.stringify(catalogDocument)
const CATALOG = validateCatalogSource(METRIC_ANALYSIS_CATALOG_SOURCE)

export const STATIC_METRIC_ANALYSIS_IDS = Object.freeze([...CATALOG.staticMetrics.keys()])
export const DYNAMIC_METRIC_ANALYSIS_FAMILY_IDS = Object.freeze(
  CATALOG.families.map((family) => family.id),
)

export function encodeMetricAnalysisSubjectID(rawID: string): string {
  return encodeSubjectID(rawID, CATALOG.document.identifier_encoding)
}

export function decodeMetricAnalysisSubjectID(encodedID: string): string {
  return decodeSubjectID(encodedID, CATALOG.document.identifier_encoding)
}

export function resolveMetricAnalysisCatalog(metricID: string): MetricAnalysisCatalogMatch {
  return resolveFromIndex(metricID, CATALOG)
}

export function tryResolveMetricAnalysisCatalog(
  metricID: string,
): MetricAnalysisCatalogMatch | undefined {
  try {
    return resolveMetricAnalysisCatalog(metricID)
  } catch (error) {
    if (error instanceof MetricAnalysisCatalogResolutionError) return undefined
    throw error
  }
}

/** Runtime validation hook used by parity and ambiguity contract tests. */
export function validateMetricAnalysisCatalogSource(source: string): void {
  validateCatalogSource(source)
}
