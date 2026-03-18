import { useEffect, useState } from 'react'
import TableHeader from '../components/TableHeader'
import { DataTable, type Column } from '../components/DataTable'
import FleetSimSurfaceLayout from './FleetSimSurfaceLayout'
import styles from './FleetSimPage.module.css'
import {
  createFleet,
  deleteFleet,
  listFleets,
  listGpuProfiles,
  type FleetConfig,
  type GpuProfile,
  type PoolConfig,
} from '../utils/fleetSimApi'
import { formatDateTime, formatMoneyKusd, formatNumber } from './fleetSimPageSupport'

const GPU_OPTIONS = ['a100', 'h100', 'a10g']

const DEFAULT_POOLS: PoolConfig[] = [
  { pool_id: 'short', gpu: 'a100', n_gpus: 4, max_ctx: 4096 },
  { pool_id: 'long', gpu: 'h100', n_gpus: 2, max_ctx: 32768 },
]

export default function FleetSimFleetsPage() {
  const [gpuProfiles, setGpuProfiles] = useState<GpuProfile[]>([])
  const [fleets, setFleets] = useState<FleetConfig[]>([])
  const [expandedFleetIDs, setExpandedFleetIDs] = useState<Set<string>>(new Set())
  const [fleetName, setFleetName] = useState('Balanced production')
  const [routerType, setRouterType] = useState('length')
  const [compressGamma, setCompressGamma] = useState('1.25')
  const [pools, setPools] = useState<PoolConfig[]>(DEFAULT_POOLS)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')

  const load = async () => {
    const [profilesData, fleetsData] = await Promise.all([listGpuProfiles(), listFleets()])
    setGpuProfiles(profilesData)
    setFleets(fleetsData)
  }

  useEffect(() => {
    void load().catch((loadError) => {
      setError(loadError instanceof Error ? loadError.message : 'Failed to load fleet catalog')
    })
  }, [])

  const updatePool = (index: number, patch: Partial<PoolConfig>) => {
    setPools((current) => current.map((pool, poolIndex) => (poolIndex === index ? { ...pool, ...patch } : pool)))
  }

  const addPool = () => {
    setPools((current) => [
      ...current,
      {
        pool_id: `pool-${current.length + 1}`,
        gpu: 'a100',
        n_gpus: 1,
        max_ctx: 8192,
      },
    ])
  }

  const removePool = (index: number) => {
    setPools((current) => current.filter((_, poolIndex) => poolIndex !== index))
  }

  const handleCreateFleet = async () => {
    try {
      await createFleet({
        name: fleetName,
        pools,
        router: routerType,
        compress_gamma: routerType === 'compress_route' ? Number(compressGamma) : null,
      })
      setMessage(`Saved fleet ${fleetName}`)
      setError('')
      await load()
    } catch (createError) {
      setError(createError instanceof Error ? createError.message : 'Failed to save fleet')
    }
  }

  const handleDeleteFleet = async (fleet: FleetConfig) => {
    try {
      await deleteFleet(fleet.id)
      setMessage(`Deleted ${fleet.name}`)
      setError('')
      await load()
    } catch (deleteError) {
      setError(deleteError instanceof Error ? deleteError.message : 'Failed to delete fleet')
    }
  }

  const fleetColumns: Column<FleetConfig>[] = [
    {
      key: 'name',
      header: 'Fleet',
      sortable: true,
      render: (row) => (
        <div className={styles.compactListMeta}>
          <span className={styles.compactListTitle}>{row.name}</span>
          <span className={styles.compactListText}>{row.router} router · created {formatDateTime(row.created_at)}</span>
        </div>
      ),
    },
    {
      key: 'total_gpus',
      header: 'GPUs',
      sortable: true,
      render: (row) => formatNumber(row.total_gpus),
    },
    {
      key: 'estimated_annual_cost_kusd',
      header: 'Annual Cost',
      sortable: true,
      render: (row) => formatMoneyKusd(row.estimated_annual_cost_kusd),
    },
  ]

  return (
    <FleetSimSurfaceLayout
      title="Fleets"
      description="Define reusable GPU pool layouts for the simulator. These saved fleets become selectable inputs for simulation and what-if runs, so the jobs page stays lightweight."
      currentPath="/fleet-sim/fleets"
      meta={[
        { label: 'GPU profiles', value: formatNumber(gpuProfiles.length) },
        { label: 'Saved fleets', value: formatNumber(fleets.length) },
        { label: 'Pools in editor', value: formatNumber(pools.length) },
      ]}
    >
      <div className={styles.splitGrid}>
        <section className={styles.sectionCard}>
          <div className={styles.sectionHeader}>
            <div>
              <h2 className={styles.sectionTitle}>Fleet Composer</h2>
              <p className={styles.sectionDescription}>Compose one or more GPU pools, choose a router strategy, and save the layout for later runs.</p>
            </div>
          </div>
          <div className={styles.formGrid}>
            <label className={styles.field}>
              <span className={styles.fieldLabel}>Fleet Name</span>
              <input className={styles.input} value={fleetName} onChange={(event) => setFleetName(event.target.value)} />
            </label>
            <label className={styles.field}>
              <span className={styles.fieldLabel}>Router</span>
              <select className={styles.select} value={routerType} onChange={(event) => setRouterType(event.target.value)}>
                <option value="length">length</option>
                <option value="least_loaded">least_loaded</option>
                <option value="random">random</option>
                <option value="compress_route">compress_route</option>
              </select>
            </label>
            {routerType === 'compress_route' ? (
              <label className={styles.field}>
                <span className={styles.fieldLabel}>Compress Gamma</span>
                <input className={styles.input} value={compressGamma} onChange={(event) => setCompressGamma(event.target.value)} />
              </label>
            ) : null}
          </div>

          <div className={styles.compactList} style={{ marginTop: '1rem' }}>
            {pools.map((pool, index) => (
              <div key={`${pool.pool_id}-${index}`} className={styles.compactListItem}>
                <div className={styles.formGrid} style={{ flex: 1 }}>
                  <label className={styles.field}>
                    <span className={styles.fieldLabel}>Pool ID</span>
                    <input className={styles.input} value={pool.pool_id} onChange={(event) => updatePool(index, { pool_id: event.target.value })} />
                  </label>
                  <label className={styles.field}>
                    <span className={styles.fieldLabel}>GPU</span>
                    <select className={styles.select} value={pool.gpu} onChange={(event) => updatePool(index, { gpu: event.target.value })}>
                      {GPU_OPTIONS.map((gpu) => (
                        <option key={gpu} value={gpu}>
                          {gpu}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className={styles.field}>
                    <span className={styles.fieldLabel}>GPU Count</span>
                    <input
                      className={styles.input}
                      type="number"
                      min={1}
                      value={pool.n_gpus}
                      onChange={(event) => updatePool(index, { n_gpus: Number(event.target.value) })}
                    />
                  </label>
                  <label className={styles.field}>
                    <span className={styles.fieldLabel}>Max Context</span>
                    <input
                      className={styles.input}
                      type="number"
                      min={512}
                      value={pool.max_ctx}
                      onChange={(event) => updatePool(index, { max_ctx: Number(event.target.value) })}
                    />
                  </label>
                </div>
                <button type="button" className={styles.dangerButton} onClick={() => removePool(index)} disabled={pools.length === 1}>
                  Remove
                </button>
              </div>
            ))}
          </div>

          <div className={styles.buttonRow} style={{ marginTop: '1rem' }}>
            <button type="button" className={styles.secondaryButton} onClick={addPool}>
              Add Pool
            </button>
            <button type="button" className={styles.primaryButton} onClick={() => void handleCreateFleet()}>
              Save Fleet
            </button>
          </div>
          {message ? <p className={`${styles.message} ${styles.messageSuccess}`}>{message}</p> : null}
          {error ? <p className={`${styles.message} ${styles.messageError}`}>{error}</p> : null}
        </section>

        <section className={styles.sectionCard}>
          <div className={styles.sectionHeader}>
            <div>
              <h2 className={styles.sectionTitle}>Profile Reference</h2>
              <p className={styles.sectionDescription}>The simulator exposes a compact hardware catalog with calibrated latency and cost assumptions per GPU SKU.</p>
            </div>
          </div>
          <ul className={styles.compactList}>
            {gpuProfiles.map((profile) => (
              <li key={profile.name} className={styles.compactListItem}>
                <div className={styles.compactListMeta}>
                  <span className={styles.compactListTitle}>{profile.name}</span>
                  <span className={styles.compactListText}>
                    {formatNumber(profile.max_slots)} slots · {formatNumber(profile.total_kv_blks)} KV blocks
                  </span>
                </div>
                <span className={styles.compactListText}>${profile.cost_per_hr.toFixed(2)}/hr</span>
              </li>
            ))}
          </ul>
        </section>
      </div>

      <section className={styles.sectionCard}>
        <TableHeader title="Saved Fleets" count={fleets.length} variant="embedded" />
        <DataTable
          columns={fleetColumns}
          data={fleets}
          keyExtractor={(row) => row.id}
          onDelete={(row) => void handleDeleteFleet(row)}
          expandable
          isRowExpanded={(row) => expandedFleetIDs.has(row.id)}
          onToggleExpand={(row) => {
            setExpandedFleetIDs((current) => {
              const next = new Set(current)
              if (next.has(row.id)) next.delete(row.id)
              else next.add(row.id)
              return next
            })
          }}
          renderExpandedRow={(row) => (
            <div className={styles.resultGrid}>
              {row.pools.map((pool) => (
                <div key={pool.pool_id} className={styles.resultMetric}>
                  <span className={styles.resultMetricLabel}>{pool.pool_id}</span>
                  <span className={styles.resultMetricValue}>{pool.gpu}</span>
                  <div className={styles.inlineHint}>
                    {pool.n_gpus} GPUs · max ctx {formatNumber(pool.max_ctx)}
                  </div>
                </div>
              ))}
            </div>
          )}
          emptyMessage="No fleets saved yet."
        />
      </section>
    </FleetSimSurfaceLayout>
  )
}
