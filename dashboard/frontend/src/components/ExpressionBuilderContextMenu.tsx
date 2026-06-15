import styles from './ExpressionBuilder.module.css'
import {
  getNodeAtPath,
  isLeaf,
  isOperator,
  type NodePath,
  type RuleNode,
} from './ExpressionBuilderSupport'
import type { OperatorKind } from './ExpressionBuilderNodes'

interface ExpressionBuilderContextMenuProps {
  contextMenu: { x: number; y: number; path: NodePath }
  tree: RuleNode
  onAddChild: (path: NodePath) => void
  onChangeOp: (path: NodePath, newOp: 'AND' | 'OR') => void
  onDeleteNode: (path: NodePath) => void
  onEditSignal: (path: NodePath, signalType: string, signalName: string) => void
  onInsertSibling: (target: { parentPath: NodePath; index: number }) => void
  onUnwrap: (path: NodePath) => void
  onWrap: (path: NodePath, operator: OperatorKind) => void
}

export default function ExpressionBuilderContextMenu({
  contextMenu,
  tree,
  onAddChild,
  onChangeOp,
  onDeleteNode,
  onEditSignal,
  onInsertSibling,
  onUnwrap,
  onWrap,
}: ExpressionBuilderContextMenuProps) {
  const node = getNodeAtPath(tree, contextMenu.path)
  if (!node) return null

  return (
    <div
      className={styles.ctxMenu}
      style={{ left: contextMenu.x, top: contextMenu.y }}
      onClick={event => event.stopPropagation()}
    >
      {isLeaf(node) ? (
        <div
          className={styles.ctxMenuItem}
          onClick={() => onEditSignal(contextMenu.path, node.signalType, node.signalName)}
        >
          Edit Signal
        </div>
      ) : null}
      {isOperator(node) && node.operator !== 'NOT' ? (
        <div
          className={styles.ctxMenuItem}
          onClick={() => onChangeOp(contextMenu.path, node.operator === 'AND' ? 'OR' : 'AND')}
        >
          Toggle to {node.operator === 'AND' ? 'OR' : 'AND'}
        </div>
      ) : null}
      {isOperator(node) && (node.operator !== 'NOT' || (node.conditions as RuleNode[]).length === 0) ? (
        <div className={styles.ctxMenuItem} onClick={() => onAddChild(contextMenu.path)}>
          Add child...
        </div>
      ) : null}
      <div className={styles.ctxMenuItem} onClick={() => onWrap(contextMenu.path, 'AND')}>
        Wrap with AND
      </div>
      <div className={styles.ctxMenuItem} onClick={() => onWrap(contextMenu.path, 'OR')}>
        Wrap with OR
      </div>
      <div className={styles.ctxMenuItem} onClick={() => onWrap(contextMenu.path, 'NOT')}>
        Wrap with NOT
      </div>
      {contextMenu.path.length > 0 ? (
        <>
          <div className={styles.ctxMenuDivider} />
          <div
            className={styles.ctxMenuItem}
            onClick={() =>
              onInsertSibling({
                parentPath: contextMenu.path.slice(0, -1),
                index: contextMenu.path[contextMenu.path.length - 1],
              })
            }
          >
            Insert before...
          </div>
          <div
            className={styles.ctxMenuItem}
            onClick={() =>
              onInsertSibling({
                parentPath: contextMenu.path.slice(0, -1),
                index: contextMenu.path[contextMenu.path.length - 1] + 1,
              })
            }
          >
            Insert after...
          </div>
        </>
      ) : null}
      {isOperator(node) && node.conditions.length > 0 ? (
        <div className={styles.ctxMenuItem} onClick={() => onUnwrap(contextMenu.path)}>
          Unwrap (replace with first child)
        </div>
      ) : null}
      {isOperator(node) && node.operator !== 'NOT' ? (
        <>
          {node.operator !== 'AND' ? (
            <div className={styles.ctxMenuItem} onClick={() => onChangeOp(contextMenu.path, 'AND')}>
              Change to AND
            </div>
          ) : null}
          {node.operator !== 'OR' ? (
            <div className={styles.ctxMenuItem} onClick={() => onChangeOp(contextMenu.path, 'OR')}>
              Change to OR
            </div>
          ) : null}
        </>
      ) : null}
      <div className={styles.ctxMenuDivider} />
      <div
        className={`${styles.ctxMenuItem} ${styles.ctxMenuDanger}`}
        onClick={() => onDeleteNode(contextMenu.path)}
      >
        Delete
      </div>
    </div>
  )
}
