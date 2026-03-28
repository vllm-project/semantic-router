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
import {
  describeRouterType,
  formatDateTime,
  formatGpuLabel,
  formatMoneyKusd,
  formatNumber,
  formatRouterType,
} from './fleetSimPageSupport'

const GPU_OPTIONS = ['a100', 'h100', 'a10g']
const ROUTER_OPTIONS = [
  { value: 'length', title: 'Length split' },
  { value: 'least_loaded', title: 'Least loaded' },
  { value: 'random', title: 'Random baseline' },
  { value: 'compress_route', title: 'Compression-aware' },
] as const

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
      if (!fleetName.trim()) {
        throw new Error('Give the fleet a name before saving it.')
      }
      await createFleet({
        name: fleetName.trim(),
        pools,
        router: routerType,
        compress_gamma: routerType === 'compress_route' ? Number(compressGamma) : null,
      })
      setMessage(`Saved fleet ${fleetName.trim()}`)
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

  const gpuChoices = Array.from(new Set([...gpuProfiles.map((profile) => profile.name), ...GPU_OPTIONS]))
  const draftTotalGpus = pools.reduce((sum, pool) => sum + pool.n_gpus, 0)
  const draftCostPerHour = pools.reduce((sum, pool) => {
    const profile = gpuProfiles.find((candidate) => candidate.name === pool.gpu)
    return sum + (profile?.cost_per_hr || 0) * pool.n_gpus
  }, 0)
  const draftAnnualCostKusd = draftCostPerHour * 24 * 365 / 1000
  const largestContext = pools.reduce((current, pool) => Math.max(current, pool.max_ctx), 0)

  const fleetColumns: Column<FleetConfig>[] = [
    {
      key: 'name',
      header: 'Fleet',
      sortable: true,
      render: (row) => (
        <div className={styles.compactListMeta}>
          <span className={styles.compactListTitle}>{row.name}</span>
          <span className={styles.compactListText}>
            {formatRouterType(row.router)} · created {formatDateTime(row.created_at)}
          </span>
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
      description="Build reusable GPU fleet plans that teams can replay, compare, and stress-test from the dashboard."
      currentPath="/fleet-sim/fleets"
      meta={[
        { label: 'GPU profiles', value: formatNumber(gpuProfiles.length) },
        { label: 'Saved fleets', value: formatNumber(fleets.length) },
        { label: 'Pools in editor', value: formatNumber(pools.length) },
      ]}
    >
      <div className={styles.composerSplit}>
        <section className={styles.sectionCard}>
          <div className={styles.sectionHeader}>
            <div>
              <span className={styles.sectionKicker}>Composer</span>
              <h2 className={styles.sectionTitle}>Fleet Composer</h2>
              <p className={styles.sectionDescription}>Compose the pools, choose a routing strategy, and save a fleet definition your team can reuse in runs.</p>
            </div>
          </div>
          <div className={styles.formStack}>
            <div className={styles.fieldSection}>
              <div className={styles.formGrid}>
                <label className={styles.field}>
                  <span className={styles.fieldLabel}>Fleet name</span>
                  <input className={styles.input} value={fleetName} onChange={(event) => setFleetName(event.target.value)} />
                </label>
                {routerType === 'compress_route' ? (
                  <label className={styles.field}>
                    <span className={styles.fieldLabel}>Compress gamma</span>
                    <input className={styles.input} value={compressGamma} onChange={(event) => setCompressGamma(event.target.value)} />
                  </label>
                ) : null}
              </div>
            </div>

            <div className={styles.fieldSection}>
              <div>
                <span className={styles.sectionKicker}>Routing</span>
                <p className={styles.sectionDescription}>Choose the behavior operators should assume when this fleet is replayed in a planning run.</p>
              </div>
              <div className={styles.choiceGrid}>
                {ROUTER_OPTIONS.map((option) => (
                  <button
                    key={option.value}
                    type="button"
                    className={`${styles.choiceButton} ${routerType === option.value ? styles.choiceButtonActive : ''}`}
                    onClick={() => setRouterType(option.value)}
                  >
                    <span className={styles.choiceButtonTitle}>{option.title}</span>
                    <span className={styles.choiceButtonDescription}>{describeRouterType(option.value)}</span>
                  </button>
                ))}
              </div>
            </div>

            <div className={styles.fieldSection}>
              <div className={styles.sectionHeader}>
                <div>
                  <span className={styles.sectionKicker}>Pools</span>
                  <h3 className={styles.sectionTitle}>GPU pools</h3>
                  <p className={styles.sectionDescription}>Keep each pool focused: one hardware tier, one capacity target, one max context envelope.</p>
                </div>
              </div>
              <div className={styles.poolEditorList}>
                {pools.map((pool, index) => (
                  <article key={`${pool.pool_id}-${index}`} className={styles.poolEditorCard}>
                    <div className={styles.poolEditorHeader}>
                      <div>
                        <span className={styles.poolEditorEyebrow}>Pool {index + 1}</span>
                        <h4 className={styles.poolEditorTitle}>{pool.pool_id || `Pool ${index + 1}`}</h4>
                      </div>
                      <button type="button" className={styles.dangerButton} onClick={() => removePool(index)} disabled={pools.length === 1}>
                        Remove
                      </button>
                    </div>
                    <div className={styles.poolEditorGrid}>
                      <label className={styles.field}>
                        <span className={styles.fieldLabel}>Pool label</span>
                        <input className={styles.input} value={pool.pool_id} onChange={(event) => updatePool(index, { pool_id: event.target.value })} />
                      </label>
                      <label className={styles.field}>
                        <span className={styles.fieldLabel}>GPU</span>
                        <select className={styles.select} value={pool.gpu} onChange={(event) => updatePool(index, { gpu: event.target.value })}>
                          {gpuChoices.map((gpu) => (
                            <option key={gpu} value={gpu}>
                              {formatGpuLabel(gpu)}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label className={styles.field}>
                        <span className={styles.fieldLabel}>GPU count</span>
                        <input
                          className={styles.input}
                          type="number"
                          min={1}
                          value={pool.n_gpus}
                          onChange={(event) => updatePool(index, { n_gpus: Number(event.target.value) })}
                        />
                      </label>
                      <label className={styles.field}>
                        <span className={styles.fieldLabel}>Max context</span>
                        <input
                          className={styles.input}
                          type="number"
                          min={512}
                          value={pool.max_ctx}
                          onChange={(event) => updatePool(index, { max_ctx: Number(event.target.value) })}
                        />
                      </label>
                    </div>
                  </article>
                ))}
              </div>
            </div>
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

        <aside className={styles.summaryCard}>
          <div className={styles.summaryHeader}>
            <span className={styles.summaryEyebrow}>Draft summary</span>
            <h2 className={styles.summaryTitle}>{fleetName}</h2>
            <p className={styles.summaryText}>{describeRouterType(routerType)}</p>
          </div>
          <div className={styles.summaryPillRow}>
            <span className={styles.summaryPill}>{formatRouterType(routerType)}</span>
            {routerType === 'compress_route' ? <span className={styles.summaryPill}>Gamma {compressGamma}</span> : null}
          </div>
          <dl className={styles.summaryList}>
            <div className={styles.summaryItem}>
              <dt className={styles.summaryLabel}>Pools</dt>
              <dd className={styles.summaryValue}>{formatNumber(pools.length)}</dd>
            </div>
            <div className={styles.summaryItem}>
              <dt className={styles.summaryLabel}>Total GPUs</dt>
              <dd className={styles.summaryValue}>{formatNumber(draftTotalGpus)}</dd>
            </div>
            <div className={styles.summaryItem}>
              <dt className={styles.summaryLabel}>Largest context</dt>
              <dd className={styles.summaryValue}>{formatNumber(largestContext)}</dd>
            </div>
            <div className={styles.summaryItem}>
              <dt className={styles.summaryLabel}>Est. annual cost</dt>
              <dd className={styles.summaryValue}>{formatMoneyKusd(draftAnnualCostKusd)}</dd>
            </div>
          </dl>

          <div className={styles.fieldSection}>
            <div>
              <span className={styles.sectionKicker}>Reference</span>
              <p className={styles.sectionDescription}>Calibrated hardware assumptions available for this workspace.</p>
            </div>
            <div className={styles.profileGrid}>
              {gpuProfiles.map((profile) => (
                <article key={profile.name} className={styles.profileCard}>
                  <div className={styles.libraryCardHeader}>
                    <div>
                      <span className={styles.libraryCardKey}>GPU profile</span>
                      <h3 className={styles.libraryCardTitle}>{formatGpuLabel(profile.name)}</h3>
                    </div>
                    <span className={styles.metricPill}>${profile.cost_per_hr.toFixed(2)}/hr</span>
                  </div>
                  <div className={styles.profileMetricRow}>
                    <span className={styles.profileMetricLabel}>Slots</span>
                    <span className={styles.profileMetricValue}>{formatNumber(profile.max_slots)}</span>
                  </div>
                  <div className={styles.profileMetricRow}>
                    <span className={styles.profileMetricLabel}>KV blocks</span>
                    <span className={styles.profileMetricValue}>{formatNumber(profile.total_kv_blks)}</span>
                  </div>
                  <div className={styles.profileMetricRow}>
                    <span className={styles.profileMetricLabel}>Chunk</span>
                    <span className={styles.profileMetricValue}>{formatNumber(profile.chunk)}</span>
                  </div>
                </article>
              ))}
            </div>
          </div>
        </aside>
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
            <div className={styles.expandedPanel}>
              <div className={styles.inlineDetailGrid}>
                <div className={styles.inlineDetailCell}>
                  <span className={styles.inlineDetailLabel}>Router</span>
                  <span className={styles.inlineDetailValue}>{formatRouterType(row.router)}</span>
                </div>
                <div className={styles.inlineDetailCell}>
                  <span className={styles.inlineDetailLabel}>Total GPUs</span>
                  <span className={styles.inlineDetailValue}>{formatNumber(row.total_gpus)}</span>
                </div>
                <div className={styles.inlineDetailCell}>
                  <span className={styles.inlineDetailLabel}>Annual cost</span>
                  <span className={styles.inlineDetailValue}>{formatMoneyKusd(row.estimated_annual_cost_kusd)}</span>
                </div>
              </div>
              <div className={styles.inlineDetailRows}>
                {row.pools.map((pool) => (
                  <div key={pool.pool_id} className={styles.inlineDetailRow}>
                    <span className={styles.inlineDetailValue}>{pool.pool_id}</span>
                    <span className={styles.inlineDetailText}>{formatGpuLabel(pool.gpu)}</span>
                    <span className={styles.inlineDetailText}>{formatNumber(pool.n_gpus)} GPUs</span>
                    <span className={styles.inlineDetailText}>Max context {formatNumber(pool.max_ctx)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
          emptyMessage="No fleets saved yet."
        />
      </section>
    </FleetSimSurfaceLayout>
  )
}
