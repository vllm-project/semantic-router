import { formatPermissionAction, groupAccountPermissions } from './LayoutAccountControlSupport'
import styles from './PermissionList.module.css'
import ProductIcon from './ProductIcon'

interface PermissionListProps {
  permissions: readonly string[]
  emptyMessage?: string
  compact?: boolean
}

export default function PermissionList({
  permissions,
  emptyMessage = 'No explicit permissions',
  compact = false,
}: PermissionListProps) {
  const groups = groupAccountPermissions(permissions)

  if (groups.length === 0) {
    return <p className={styles.empty}>{emptyMessage}</p>
  }

  return (
    <div className={`${styles.groups} ${compact ? styles.compact : ''}`.trim()}>
      {groups.map((group) => (
        <section key={group.key} className={styles.group}>
          <header>
            <span className={styles.groupMark} aria-hidden="true">
              {group.label.slice(0, 1)}
            </span>
            <strong>{group.label}</strong>
            <small>{group.permissions.length}</small>
          </header>
          <ul>
            {group.permissions.map((permission) => (
              <li key={permission} title={permission}>
                <ProductIcon name="check" aria-hidden="true" />
                {formatPermissionAction(permission)}
              </li>
            ))}
          </ul>
        </section>
      ))}
    </div>
  )
}
