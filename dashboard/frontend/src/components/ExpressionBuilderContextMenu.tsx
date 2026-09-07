import { useEffect, useRef, type KeyboardEvent } from 'react'

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
  onClose: () => void
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
  onClose,
  onUnwrap,
  onWrap,
}: ExpressionBuilderContextMenuProps) {
  const menuRef = useRef<HTMLDivElement>(null)
  const restoreFocusRef = useRef<HTMLElement | null>(null)
  const node = getNodeAtPath(tree, contextMenu.path)

  useEffect(() => {
    restoreFocusRef.current =
      document.activeElement instanceof HTMLElement ? document.activeElement : null
    menuRef.current?.querySelector<HTMLButtonElement>('[role="menuitem"]')?.focus()
  }, [])

  if (!node) return null

  const closeAndRestoreFocus = () => {
    onClose()
    restoreFocusRef.current?.focus()
  }

  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Escape') {
      event.preventDefault()
      event.stopPropagation()
      closeAndRestoreFocus()
      return
    }

    if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return

    const menuItems = Array.from(
      menuRef.current?.querySelectorAll<HTMLButtonElement>('[role="menuitem"]') ?? [],
    )
    if (menuItems.length === 0) return

    event.preventDefault()
    const currentIndex = menuItems.indexOf(document.activeElement as HTMLButtonElement)
    if (event.key === 'Home') {
      menuItems[0]?.focus()
      return
    }
    if (event.key === 'End') {
      menuItems[menuItems.length - 1]?.focus()
      return
    }

    const direction = event.key === 'ArrowDown' ? 1 : -1
    const fallbackIndex = direction === 1 ? -1 : 0
    const nextIndex = (currentIndex === -1 ? fallbackIndex : currentIndex) + direction
    menuItems[(nextIndex + menuItems.length) % menuItems.length]?.focus()
  }

  return (
    <div
      ref={menuRef}
      className={styles.ctxMenu}
      style={{ left: contextMenu.x, top: contextMenu.y }}
      role="menu"
      aria-label="Expression actions"
      onClick={(event) => event.stopPropagation()}
      onKeyDown={handleKeyDown}
    >
      {isLeaf(node) ? (
        <button
          type="button"
          role="menuitem"
          className={styles.ctxMenuItem}
          onClick={() => onEditSignal(contextMenu.path, node.signalType, node.signalName)}
        >
          Edit Signal
        </button>
      ) : null}
      {isOperator(node) && node.operator !== 'NOT' ? (
        <button
          type="button"
          role="menuitem"
          className={styles.ctxMenuItem}
          onClick={() => onChangeOp(contextMenu.path, node.operator === 'AND' ? 'OR' : 'AND')}
        >
          Toggle to {node.operator === 'AND' ? 'OR' : 'AND'}
        </button>
      ) : null}
      {isOperator(node) &&
      (node.operator !== 'NOT' || (node.conditions as RuleNode[]).length === 0) ? (
        <button
          type="button"
          role="menuitem"
          className={styles.ctxMenuItem}
          onClick={() => onAddChild(contextMenu.path)}
        >
          Add child...
        </button>
      ) : null}
      <button
        type="button"
        role="menuitem"
        className={styles.ctxMenuItem}
        onClick={() => onWrap(contextMenu.path, 'AND')}
      >
        Wrap with AND
      </button>
      <button
        type="button"
        role="menuitem"
        className={styles.ctxMenuItem}
        onClick={() => onWrap(contextMenu.path, 'OR')}
      >
        Wrap with OR
      </button>
      <button
        type="button"
        role="menuitem"
        className={styles.ctxMenuItem}
        onClick={() => onWrap(contextMenu.path, 'NOT')}
      >
        Wrap with NOT
      </button>
      {contextMenu.path.length > 0 ? (
        <>
          <div className={styles.ctxMenuDivider} role="separator" />
          <button
            type="button"
            role="menuitem"
            className={styles.ctxMenuItem}
            onClick={() =>
              onInsertSibling({
                parentPath: contextMenu.path.slice(0, -1),
                index: contextMenu.path[contextMenu.path.length - 1],
              })
            }
          >
            Insert before...
          </button>
          <button
            type="button"
            role="menuitem"
            className={styles.ctxMenuItem}
            onClick={() =>
              onInsertSibling({
                parentPath: contextMenu.path.slice(0, -1),
                index: contextMenu.path[contextMenu.path.length - 1] + 1,
              })
            }
          >
            Insert after...
          </button>
        </>
      ) : null}
      {isOperator(node) && node.conditions.length > 0 ? (
        <button
          type="button"
          role="menuitem"
          className={styles.ctxMenuItem}
          onClick={() => onUnwrap(contextMenu.path)}
        >
          Unwrap (replace with first child)
        </button>
      ) : null}
      {isOperator(node) && node.operator !== 'NOT' ? (
        <>
          {node.operator !== 'AND' ? (
            <button
              type="button"
              role="menuitem"
              className={styles.ctxMenuItem}
              onClick={() => onChangeOp(contextMenu.path, 'AND')}
            >
              Change to AND
            </button>
          ) : null}
          {node.operator !== 'OR' ? (
            <button
              type="button"
              role="menuitem"
              className={styles.ctxMenuItem}
              onClick={() => onChangeOp(contextMenu.path, 'OR')}
            >
              Change to OR
            </button>
          ) : null}
        </>
      ) : null}
      <div className={styles.ctxMenuDivider} role="separator" />
      <button
        type="button"
        role="menuitem"
        className={`${styles.ctxMenuItem} ${styles.ctxMenuDanger}`}
        onClick={() => onDeleteNode(contextMenu.path)}
      >
        Delete
      </button>
    </div>
  )
}
