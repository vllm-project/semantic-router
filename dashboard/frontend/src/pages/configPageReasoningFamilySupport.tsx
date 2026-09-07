import type { Column } from '../components/DataTable'
import type { ViewSection } from '../components/ViewModal'
import styles from './ConfigPageModelsSection.module.css'
import type { ReasoningFamily } from './configPageSupport'

export interface ReasoningFamilyRow {
  name: string
  type: string
  parameter: string
  levels: string
  defaultLevel: string
}

const reasoningFamilyMonogram = (name: string): string =>
  name
    .split(/[-_\s]+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0]?.toLocaleUpperCase())
    .join('') || 'R'

export const reasoningFamilyColumns: Column<ReasoningFamilyRow>[] = [
  {
    key: 'name',
    header: 'Family Name',
    sortable: true,
    render: (row) => (
      <span className={styles.reasoningFamilyIdentity}>
        <span className={styles.reasoningFamilyMark} aria-hidden="true">
          {reasoningFamilyMonogram(row.name)}
        </span>
        <span>
          <strong className={styles.reasoningFamilyName}>{row.name}</strong>
          <small>
            {row.defaultLevel ? `Default · ${row.defaultLevel}` : 'Model-selected default'}
          </small>
        </span>
      </span>
    ),
  },
  {
    key: 'type',
    header: 'Type',
    width: '200px',
    sortable: true,
    render: (row) => <span className={styles.reasoningFamilyType}>{row.type}</span>,
  },
  {
    key: 'parameter',
    header: 'Parameter',
    sortable: true,
    render: (row) => <code className={styles.reasoningFamilyParameter}>{row.parameter}</code>,
  },
  { key: 'levels', header: 'Controls', render: (row) => row.levels || 'N/A' },
  {
    key: 'defaultLevel',
    header: 'Default',
    width: '120px',
    render: (row) => row.defaultLevel || 'N/A',
  },
]

export const reasoningFamilyRows = (
  families: Record<string, ReasoningFamily>,
): ReasoningFamilyRow[] =>
  Object.entries(families).map(([name, config]) => ({
    name,
    type: config.type,
    parameter: config.parameter,
    levels: [...(config.levels || []), ...(config.modes || [])].join(', '),
    defaultLevel: config.default || config.default_mode || '',
  }))

export const filterReasoningFamilyRows = (
  rows: ReasoningFamilyRow[],
  search: string,
): ReasoningFamilyRow[] => {
  const query = search.trim().toLocaleLowerCase()
  if (!query) return rows
  return rows.filter(
    (family) =>
      family.name.toLocaleLowerCase().includes(query) ||
      family.type.toLocaleLowerCase().includes(query) ||
      family.parameter.toLocaleLowerCase().includes(query) ||
      family.levels.toLocaleLowerCase().includes(query),
  )
}

export const reasoningFamilyViewSections = (
  familyName: string,
  family: ReasoningFamily,
): ViewSection[] => [
  {
    title: 'Configuration',
    fields: [
      { label: 'Family Name', value: familyName },
      { label: 'Type', value: family.type },
      { label: 'Parameter', value: family.parameter },
      { label: 'Levels', value: family.levels?.join(', ') || 'N/A' },
      { label: 'Modes', value: family.modes?.join(', ') || 'N/A' },
      { label: 'Default', value: family.default || family.default_mode || 'N/A' },
    ],
  },
]
