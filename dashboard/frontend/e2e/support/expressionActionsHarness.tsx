import React, { useState } from 'react'
import { createRoot } from 'react-dom/client'
import ExpressionBuilder from '../../src/components/ExpressionBuilder'
import ExpressionBuilderContextMenu from '../../src/components/ExpressionBuilderContextMenu'

export function ExpressionActionsHarness() {
  const [value, setValue] = useState('keyword("example")')
  const [menuOpen, setMenuOpen] = useState(false)
  const [action, setAction] = useState('')
  const record = (name: string) => {
    setAction(name)
    setMenuOpen(false)
  }
  return (
    <>
      <button onClick={() => setMenuOpen(true)}>Expression actions</button>
      {menuOpen && (
        <ExpressionBuilderContextMenu
          contextMenu={{ x: 10, y: 40, path: [] }}
          tree={{ signalType: 'keyword', signalName: 'example' }}
          onAddChild={() => record('add')}
          onChangeOp={() => record('change')}
          onClose={() => setMenuOpen(false)}
          onDeleteNode={() => record('delete')}
          onEditSignal={() => record('edit')}
          onInsertSibling={() => record('insert')}
          onUnwrap={() => record('unwrap')}
          onWrap={(_, operator) => record(`wrap ${operator}`)}
        />
      )}
      <button>After actions</button>
      <output aria-label="Last action">{action}</output>
      <div style={{ height: 650, width: 1000 }}>
        <ExpressionBuilder
          value={value}
          onChange={setValue}
          availableSignals={[{ signalType: 'keyword', signalName: 'example' }]}
        />
      </div>
      <output aria-label="Expression value">{value}</output>
    </>
  )
}

createRoot(document.getElementById('root')!).render(<ExpressionActionsHarness />)
