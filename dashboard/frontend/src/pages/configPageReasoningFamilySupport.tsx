import type { Column } from '../components/DataTable'
import type { ViewSection } from '../components/ViewModal'
import styles from './ConfigPage.module.css'
import type { ReasoningFamily } from './configPageSupport'

export interface ReasoningFamilyRow {
  name: string
  type: string
  parameter: string
  levels: string
  defaultLevel: string
}

export const reasoningFamilyColumns: Column<ReasoningFamilyRow>[] = [
  {
    key: 'name',
    header: 'Family Name',
    sortable: true,
    render: (row) => <span className={styles.reasoningFamilyName}>{row.name}</span>,
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
  { key: 'levels', header: 'Levels', render: (row) => row.levels || 'N/A' },
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
    levels: config.levels?.join(', ') || '',
    defaultLevel: config.default || '',
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
      { label: 'Default', value: family.default || 'N/A' },
    ],
  },
]
