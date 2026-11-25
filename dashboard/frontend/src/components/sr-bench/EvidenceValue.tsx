import { useState } from 'react'
import BenchPagination from './BenchPagination'
import { Disclosure } from './benchUi'
import { codeBlockClass, evidenceFieldsClass } from './benchStyles'

/** Recorded settings and tool arguments are data, never HTML or executable links. */
export default function EvidenceValue({ value, depth = 0 }: { value: unknown; depth?: number }) {
  const [page, setPage] = useState(0)
  if (value === null) return <>Not set</>
  if (typeof value === 'boolean') return <>{value ? 'Enabled' : 'Disabled'}</>
  if (typeof value !== 'object') return <>{String(value)}</>
  if (depth >= 8)
    return (
      <Disclosure summary="Nested value" bare className="my-1">
        <pre className={codeBlockClass}>{JSON.stringify(value, null, 2)}</pre>
      </Disclosure>
    )
  const entries = Object.entries(value)
  if (!entries.length) return <>{Array.isArray(value) ? 'Empty list' : 'No fields'}</>
  return (
    <>
      <dl className={evidenceFieldsClass}>
        {entries.slice(page * 10, page * 10 + 10).map(([key, item]) => (
          <div key={key}>
            <dt>{Array.isArray(value) ? `Item ${Number(key) + 1}` : key}</dt>
            <dd>
              <EvidenceValue value={item} depth={depth + 1} />
            </dd>
          </div>
        ))}
      </dl>
      <BenchPagination
        label="Fields"
        total={entries.length}
        page={page}
        pageSize={10}
        onChange={setPage}
      />
    </>
  )
}
