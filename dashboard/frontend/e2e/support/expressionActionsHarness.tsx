import React, { useState } from 'react'
import { createRoot } from 'react-dom/client'
import ExpressionBuilder from '../../src/components/ExpressionBuilder'
import ExpressionBuilderContextMenu from '../../src/components/ExpressionBuilderContextMenu'
import { incompleteRawExpression } from '../../src/lib/decisionRuleIssues'

export function ExpressionActionsHarness() {
  const [value, setValue] = useState('keyword("example")')
  const [second, setSecond] = useState('classifier("intent", label: "billing", predicate: { gte: 0.6 })')
  const [rawStatus, setRawStatus] = useState('complete')
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
          availableSignals={[{ signalType: 'keyword', name: 'example' }]}
          onIssuesChange={(issues) => setRawStatus(incompleteRawExpression(issues) ? 'incomplete' : 'complete')}
        />
      </div>
      <output aria-label="Expression value">{value}</output>
      <output aria-label="Raw status">{rawStatus}</output>
      <section aria-label="Second editor" style={{ height: 650, width: 1000 }}>
        <ExpressionBuilder
          value={second}
          onChange={setSecond}
          availableSignals={[
            { signalType: 'classifier', name: 'intent', labels: ['billing', 'support'] },
            { signalType: 'keyword', name: 'example' },
          ]}
        />
      </section>
      <output aria-label="Other editor value">{second}</output>
    </>
  )
}

createRoot(document.getElementById('root')!).render(<ExpressionActionsHarness />)
