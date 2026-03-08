import styles from './ExpressionBuilder.module.css'
import { DRAG_MIME, encodeDrag, type SignalDescriptor } from './ExpressionBuilderSupport'
import { OPERATOR_META, OPERATOR_ORDER } from './ExpressionBuilderNodes'

interface ExpressionBuilderToolboxProps {
  collapsedGroups: Set<string>
  filteredGroups: Array<[string, SignalDescriptor[]]>
  signalCount: number
  signalSearch: string
  toolboxCollapsed: boolean
  onClear: () => void
  onSignalSearchChange: (value: string) => void
  onToggleCollapsed: () => void
  onToggleGroup: (group: string) => void
}

export default function ExpressionBuilderToolbox({
  collapsedGroups,
  filteredGroups,
  signalCount,
  signalSearch,
  toolboxCollapsed,
  onClear,
  onSignalSearchChange,
  onToggleCollapsed,
  onToggleGroup,
}: ExpressionBuilderToolboxProps) {
  return (
    <div className={`${styles.toolbox} ${toolboxCollapsed ? styles.toolboxCollapsed : ''}`}>
      <div className={styles.toolboxHeader} onClick={onToggleCollapsed}>
        <span className={styles.toolboxHeaderTitle}>{toolboxCollapsed ? '▶' : '▼'} Toolbox</span>
        <span className={styles.toolboxHeaderCount}>{signalCount} signals</span>
      </div>

      {!toolboxCollapsed ? (
        <div className={styles.toolboxContent}>
          <div className={styles.toolboxOperators}>
            {OPERATOR_ORDER.map(operator => {
              const meta = OPERATOR_META[operator]
              return (
                <div
                  key={operator}
                  className={`${styles.toolboxOp} ${styles[`toolboxOp${operator}`]}`}
                  draggable
                  onDragStart={event => {
                    event.dataTransfer.setData(
                      DRAG_MIME,
                      encodeDrag({ kind: 'operator', operator })
                    )
                    event.dataTransfer.effectAllowed = 'copyMove'
                  }}
                  onClick={event => event.stopPropagation()}
                  title={`Drag ${operator} gate to canvas`}
                >
                  <span className={styles.toolboxOpIcon} style={{ color: meta.color }}>
                    {meta.icon}
                  </span>
                  {operator}
                </div>
              )
            })}
            <button
              className={styles.clearBtn}
              onClick={event => {
                event.stopPropagation()
                onClear()
              }}
            >
              Clear
            </button>
          </div>

          <div className={styles.toolboxSearch}>
            <input
              className={styles.toolboxSearchInput}
              value={signalSearch}
              onChange={event => onSignalSearchChange(event.target.value)}
              placeholder="Search signals..."
              onClick={event => event.stopPropagation()}
            />
            {signalSearch ? (
              <button
                className={styles.toolboxSearchClear}
                onClick={() => onSignalSearchChange('')}
              >
                ×
              </button>
            ) : null}
          </div>

          <div className={styles.toolboxSignals}>
            {filteredGroups.map(([type, signals]) => {
              const collapsed = collapsedGroups.has(type)

              return (
                <div key={type} className={styles.signalGroup}>
                  <div
                    className={styles.signalGroupHeader}
                    onClick={event => {
                      event.stopPropagation()
                      onToggleGroup(type)
                    }}
                  >
                    <span className={styles.signalGroupToggle}>{collapsed ? '▶' : '▼'}</span>
                    <span className={styles.signalGroupName}>{type}</span>
                    <span className={styles.signalGroupCount}>{signals.length}</span>
                  </div>
                  {!collapsed ? (
                    <div className={styles.signalGroupItems}>
                      {signals.map(signal => (
                        <div
                          key={`${signal.signalType}-${signal.name}`}
                          className={styles.toolboxChip}
                          draggable
                          onDragStart={event => {
                            event.dataTransfer.setData(
                              DRAG_MIME,
                              encodeDrag({
                                kind: 'signal',
                                signalType: signal.signalType,
                                signalName: signal.name,
                              })
                            )
                            event.dataTransfer.effectAllowed = 'copyMove'
                          }}
                          onClick={event => event.stopPropagation()}
                          title={`Drag to canvas to add ${signal.signalType}("${signal.name}")`}
                        >
                          {signal.name}
                        </div>
                      ))}
                    </div>
                  ) : null}
                </div>
              )
            })}
            {filteredGroups.length === 0 ? (
              <span className={styles.toolboxEmpty}>
                {signalSearch ? 'No matching signals' : 'No signals defined'}
              </span>
            ) : null}
          </div>
        </div>
      ) : null}
    </div>
  )
}
