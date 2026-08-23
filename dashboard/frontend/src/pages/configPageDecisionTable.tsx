import type { Column } from '../components/DataTable'
import { formatRoutingMetadataValue } from '../components/routingMetadataDisplay'
import styles from './ConfigPage.module.css'
import type { DecisionConfig } from './configPageSupport'
import { TABLE_COLUMN_WIDTH } from './configPageSupport'

export const decisionColumns: Column<DecisionConfig>[] = [
  {
    key: 'name',
    header: 'Name',
    sortable: true,
    render: (row) => (
      <span style={{ fontWeight: 600 }}>
        {formatRoutingMetadataValue('x-vsr-selected-decision', row.name)}
      </span>
    ),
  },
  {
    key: 'priority',
    header: 'Priority',
    width: TABLE_COLUMN_WIDTH.compact,
    align: 'center',
    sortable: true,
    render: (row) => (
      <span className={`${styles.tableMetaBadge} ${styles.tableMetaBadgeMono}`}>
        P{row.priority}
      </span>
    ),
  },
  {
    key: 'conditions',
    header: 'Conditions',
    width: TABLE_COLUMN_WIDTH.medium,
    render: (row) => {
      const count = row.rules?.conditions?.length || 0
      return (
        <span>
          {count} {count === 1 ? 'condition' : 'conditions'}
        </span>
      )
    },
  },
]
