import { useEffect, useId, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import ProductIcon from '../ProductIcon'
import ProductLoadingState from '../ProductLoadingState'
import { benchApi } from './api'
import { benchmarkTitle, friendlyDatasetName, profileTitle } from './datasetPresentation'
import { number } from './model'
import type { Dataset, DatasetCase, DatasetCasePage, DatasetDetail } from './types'
import shared from './SrBench.module.css'
import styles from './DatasetInventory.module.css'

const PAGE_SIZE = 25

export default function DatasetDetails({
  id,
  dataset,
  canRun,
  onUse,
  onBack,
}: {
  id: string
  dataset?: Dataset
  canRun: boolean
  onUse: (dataset: Dataset) => void
  onBack: () => void
}) {
  const [detail, setDetail] = useState<DatasetDetail | null>(null)
  const [error, setError] = useState('')
  const [retry, setRetry] = useState(0)
  const [tab, setTab] = useState<'questions' | 'coverage' | 'about'>('questions')
  useEffect(() => {
    const controller = new AbortController()
    setError('')
    benchApi
      .dataset(id, controller.signal)
      .then(setDetail)
      .catch((cause) => {
        if (!controller.signal.aborted)
          setError(cause instanceof Error ? cause.message : 'Dataset could not be loaded.')
      })
    return () => controller.abort()
  }, [id, retry])
  const name = friendlyDatasetName(
    dataset ?? { name: detail?.name, benchmarks: detail?.benchmarks.map((item) => item.id) },
  )
  return (
    <section className={styles.detail} aria-label="Dataset details">
      <button className={styles.back} onClick={onBack}>
        <ProductIcon name="arrow-left" width={16} height={16} />
        Back to datasets
      </button>
      <div className={styles.detailHero}>
        <div>
          <span className={styles.eyebrow}>
            Dataset · {profileTitle(detail?.profile ?? dataset?.profile)}
          </span>
          <h2>{name}</h2>
          <p>
            {number(detail?.case_count ?? dataset?.case_count)} questions
            {detail && ` across ${number(detail.benchmarks.length)} benchmarks`}
            {detail?.split === 'holdout' ? ' · Holdout set' : ''}
          </p>
        </div>
        <button
          className={shared.primary}
          disabled={!canRun || !dataset}
          onClick={() => dataset && onUse(dataset)}
        >
          <ProductIcon name="play" width={16} height={16} />
          Evaluate dataset
        </button>
      </div>
      {error ? (
        <div className={shared.error} role="alert">
          <p>{error}</p>
          <button onClick={() => setRetry((value) => value + 1)}>Retry loading dataset</button>
        </div>
      ) : !detail ? (
        <ProductLoadingState label="Loading dataset" compact />
      ) : (
        <>
          <nav className={styles.detailTabs} aria-label="Dataset sections">
            {(
              [
                { id: 'questions', title: 'Questions', icon: 'list' },
                { id: 'coverage', title: 'Coverage', icon: 'chart' },
                { id: 'about', title: 'About', icon: 'audit' },
              ] as const
            ).map((item) => (
              <button
                key={item.id}
                aria-current={tab === item.id ? 'page' : undefined}
                onClick={() => setTab(item.id)}
              >
                <ProductIcon name={item.icon} width={16} height={16} />
                {item.title}
              </button>
            ))}
          </nav>
          {tab === 'questions' && <DatasetQuestions id={id} detail={detail} />}
          {tab === 'coverage' && <DatasetCoverage detail={detail} />}
          {tab === 'about' && (
            <div className={styles.about}>
              <h3>Source and reproducibility</h3>
              <p>
                Every run using this dataset evaluates the same frozen questions. Reference answers
                and hidden tests stay out of the question browser.
              </p>
              {detail.provenance.sources.length > 0 && (
                <ul className={styles.sources}>
                  {detail.provenance.sources.map((source) => (
                    <li key={source.benchmark}>
                      <strong>{benchmarkTitle(source.benchmark)}</strong>
                      {safeLink(source.url) && (
                        <a href={source.url} target="_blank" rel="noreferrer">
                          Dataset source <ProductIcon name="link" width={14} height={14} />
                        </a>
                      )}
                      {source.access_note && <p>{source.access_note}</p>}
                    </li>
                  ))}
                </ul>
              )}
              <details className={styles.technical}>
                <summary>Reproducibility details</summary>
                <dl>
                  <dt>Dataset ID</dt>
                  <dd>
                    <code>{id}</code>
                  </dd>
                  <dt>Cases SHA-256</dt>
                  <dd>
                    <code>{detail.provenance.sha256 ?? 'Unavailable'}</code>
                  </dd>
                  <dt>Seed</dt>
                  <dd>{detail.provenance.seed ?? 'Unspecified'}</dd>
                  <dt>Split</dt>
                  <dd>{detail.split ?? 'Unspecified'}</dd>
                </dl>
              </details>
            </div>
          )}
        </>
      )}
    </section>
  )
}

function safeLink(value?: string) {
  if (!value) return false
  try {
    return ['https:', 'http:'].includes(new URL(value).protocol)
  } catch {
    return false
  }
}

function DatasetCoverage({ detail }: { detail: DatasetDetail }) {
  const [selected, setSelected] = useState('')
  const maximum = Math.max(1, ...detail.benchmarks.map((item) => item.count))
  const categories = detail.categories.filter((item) => !selected || item.benchmark === selected)
  const [page, setPage] = useState(0)
  return (
    <div className={styles.coverage}>
      <div>
        <h3>Benchmark coverage</h3>
        <p>Question counts in this case set. Select a benchmark to inspect its subject groups.</p>
        <div className={styles.bars}>
          {detail.benchmarks.map((item, index) => (
            <button
              key={item.id}
              aria-pressed={selected === item.id}
              onClick={() => {
                setSelected(selected === item.id ? '' : item.id)
                setPage(0)
              }}
              className={styles.barRow}
            >
              <span>{item.title || benchmarkTitle(item.id)}</span>
              <span className={styles.barTrack}>
                <span
                  style={{
                    width: `${(100 * item.count) / maximum}%`,
                    background: ['#6b8afd', '#54b8ad', '#be88d8', '#d1a65b'][index % 4],
                  }}
                />
              </span>
              <strong>{number(item.count)}</strong>
            </button>
          ))}
        </div>
      </div>
      <section className={styles.strata}>
        <div className={shared.sectionHeading}>
          <h3>{selected ? benchmarkTitle(selected) : 'All benchmarks'} · Subject groups</h3>
          <span className={shared.badge}>{number(categories.length)} groups</span>
        </div>
        <div className={styles.categoryGrid}>
          {categories.slice(page * 12, (page + 1) * 12).map((item) => (
            <div key={`${item.benchmark}:${item.name}`}>
              <div>
                <strong>{item.name || 'General'}</strong>
                {!selected && <span>{benchmarkTitle(item.benchmark)}</span>}
              </div>
              <b>{number(item.count)}</b>
            </div>
          ))}
        </div>
        {!categories.length && <p>No subject grouping is provided by this source.</p>}
        {categories.length > 12 && (
          <nav className={styles.pagination} aria-label="Subject group pages">
            <span>
              {page * 12 + 1}–{Math.min((page + 1) * 12, categories.length)} of {categories.length}
            </span>
            <div>
              <button disabled={!page} onClick={() => setPage(page - 1)}>
                Previous groups
              </button>
              <button
                disabled={(page + 1) * 12 >= categories.length}
                onClick={() => setPage(page + 1)}
              >
                Next groups
              </button>
            </div>
          </nav>
        )}
      </section>
    </div>
  )
}

function DatasetQuestions({ id, detail }: { id: string; detail: DatasetDetail }) {
  const searchID = useId()
  const [benchmark, setBenchmark] = useState('')
  const [category, setCategory] = useState('')
  const [query, setQuery] = useState('')
  const [search, setSearch] = useState('')
  const [cursor, setCursor] = useState('0')
  const [previous, setPrevious] = useState<string[]>([])
  const [data, setData] = useState<DatasetCasePage | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [retry, setRetry] = useState(0)
  const [question, setQuestion] = useState<DatasetCase | null>(null)
  const categories = [
    ...new Set(
      detail.categories
        .filter((item) => !benchmark || item.benchmark === benchmark)
        .map((item) => item.name),
    ),
  ].sort()
  const reset = () => {
    setCursor('0')
    setPrevious([])
    setQuestion(null)
  }
  useEffect(() => {
    const controller = new AbortController()
    setLoading(true)
    setError('')
    setData(null)
    benchApi
      .datasetCases(
        id,
        { cursor, limit: PAGE_SIZE, benchmark, category, q: search },
        controller.signal,
      )
      .then((value) => {
        if (!controller.signal.aborted) setData(value)
      })
      .catch((cause) => {
        if (!controller.signal.aborted)
          setError(cause instanceof Error ? cause.message : 'Questions could not be loaded.')
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })
    return () => controller.abort()
  }, [id, cursor, benchmark, category, search, retry])
  if (question)
    return (
      <article className={styles.questionDetail}>
        <button className={styles.back} onClick={() => setQuestion(null)}>
          <ProductIcon name="arrow-left" width={16} height={16} />
          Back to questions
        </button>
        <div className={styles.questionMeta}>
          <span>{benchmarkTitle(question.benchmark)}</span>
          {question.category && <span>{question.category}</span>}
        </div>
        <h3>Question</h3>
        {question.input_status === 'unavailable' ? (
          <p className={shared.notice}>
            {question.input_notice || 'The source question is not available on this service.'}
          </p>
        ) : (
          <>
            <div className={styles.questionBody}>
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {question.question ||
                  question.messages.map((message) => message.content).join('\n\n')}
              </ReactMarkdown>
            </div>
            {!!question.choices?.length && (
              <ol className={styles.choices} type="A">
                {question.choices.map((choice, index) => (
                  <li key={index}>
                    <ReactMarkdown>{choice}</ReactMarkdown>
                  </li>
                ))}
              </ol>
            )}
          </>
        )}
        {question.input_notice && question.input_status === 'available' && (
          <p className={shared.muted}>{question.input_notice}</p>
        )}
        <details className={styles.technical}>
          <summary>Question reference</summary>
          <code>{question.id}</code>
        </details>
      </article>
    )
  return (
    <section aria-label="Dataset questions">
      <form
        className={styles.questionFilters}
        onSubmit={(event) => {
          event.preventDefault()
          setSearch(query.trim())
          reset()
        }}
      >
        <div className={styles.searchField}>
          <label htmlFor={searchID}>Search questions</label>
          <div className={styles.searchInput}>
            <input
              id={searchID}
              type="search"
              maxLength={200}
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Find a question"
            />
            <button type="submit" aria-label="Submit question search">
              <ProductIcon name="search" width={18} height={18} />
            </button>
          </div>
        </div>
        <label>
          Benchmark
          <select
            aria-label="Benchmark"
            value={benchmark}
            onChange={(event) => {
              setBenchmark(event.target.value)
              setCategory('')
              reset()
            }}
          >
            <option value="">All benchmarks</option>
            {detail.benchmarks.map((item) => (
              <option key={item.id} value={item.id}>
                {item.title || benchmarkTitle(item.id)} ({item.count})
              </option>
            ))}
          </select>
        </label>
        <label>
          Subject group
          <select
            aria-label="Subject group"
            value={category}
            onChange={(event) => {
              setCategory(event.target.value)
              reset()
            }}
          >
            <option value="">All subjects</option>
            {categories.map((item) => (
              <option key={item} value={item}>
                {item || 'General'}
              </option>
            ))}
          </select>
        </label>
      </form>
      {loading ? (
        <ProductLoadingState label="Loading questions" compact />
      ) : error ? (
        <div className={shared.error} role="alert">
          <p>{error}</p>
          <button onClick={() => setRetry((value) => value + 1)}>Retry loading questions</button>
        </div>
      ) : (
        data && (
          <>
            <p className={styles.resultCount}>
              {number(data.total)} {data.total === 1 ? 'question' : 'questions'}
              {search ? ` matching “${search}”` : ''}
            </p>
            <div className={styles.questionList}>
              {data.cases.map((item, index) => (
                <button
                  key={item.id}
                  className={styles.questionRow}
                  onClick={() => setQuestion(item)}
                  aria-label={`Open question ${Number(cursor) + index + 1}`}
                >
                  <span className={styles.questionNumber}>{Number(cursor) + index + 1}</span>
                  <div>
                    <span className={styles.questionPreview}>
                      {item.input_status === 'available'
                        ? item.question || item.messages.map((message) => message.content).join(' ')
                        : item.input_notice || 'Source question unavailable'}
                    </span>
                    <span className={styles.questionTags}>
                      {benchmarkTitle(item.benchmark)}
                      {item.category ? ` · ${item.category}` : ''}
                    </span>
                  </div>
                  <ProductIcon name="chevron-right" width={18} height={18} />
                </button>
              ))}
            </div>
            {!data.cases.length && (
              <div className={styles.empty}>
                <ProductIcon name="search" width={28} height={28} />
                <h3>No matching questions</h3>
                <p>Try a different search or subject group.</p>
              </div>
            )}
            <nav className={styles.pagination} aria-label="Question pages">
              <span>
                {data.total
                  ? `${Number(cursor) + 1}–${Number(cursor) + data.cases.length} of ${number(data.total)}`
                  : '0 questions'}
              </span>
              <div>
                <button
                  disabled={!previous.length}
                  onClick={() => {
                    setCursor(previous[previous.length - 1])
                    setPrevious(previous.slice(0, -1))
                  }}
                >
                  <ProductIcon name="chevron-left" width={16} height={16} />
                  Previous questions
                </button>
                <button
                  disabled={data.next_cursor === null}
                  onClick={() => {
                    setPrevious([...previous, cursor])
                    setCursor(String(data.next_cursor))
                  }}
                >
                  Next questions
                  <ProductIcon name="chevron-right" width={16} height={16} />
                </button>
              </div>
            </nav>
          </>
        )
      )}
    </section>
  )
}
