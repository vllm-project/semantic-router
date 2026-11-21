import { useEffect, useId, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  ArrowLeft,
  ArrowUpRight,
  ChartColumn,
  ChevronLeft,
  ChevronRight,
  FileSearch,
  ListChecks,
  Play,
  Search,
  SearchX,
} from 'lucide-react'
import {
  Badge,
  Button,
  DetailSheet,
  DetailSheetBody,
  DetailSheetFooter,
  DetailSheetHeader,
  IconTile,
  Input,
  cn,
  useRetainedValue,
} from '../../ui'
import BenchSelect from './BenchSelect'
import { benchApi } from './api'
import {
  benchmarkTitle,
  datasetEvaluationTitle,
  friendlyDatasetName,
  profileTitle,
} from './datasetPresentation'
import DatasetExposure from './DatasetExposure'
import { number } from './model'
import type { Dataset, DatasetCase, DatasetCasePage, DatasetDetail } from './types'
import { BenchBlock, BenchLoading, Disclosure, EmptyNote, Notice } from './benchUi'
import { benchText, identityClass, lineTab, lineTabList } from './benchStyles'

const PAGE_SIZE = 25

const barColors = ['var(--chart-1)', 'var(--chart-4)', 'var(--chart-3)', 'var(--chart-2)']

const markdownClass =
  'min-w-0 break-words text-[14px] leading-relaxed text-foreground [&_a]:text-primary [&_a]:underline [&_blockquote]:border-l-2 [&_blockquote]:border-border [&_blockquote]:pl-3 [&_blockquote]:text-muted-foreground [&_code]:rounded [&_code]:bg-surface-sunken [&_code]:px-1 [&_code]:font-mono [&_code]:text-[12.5px] [&_li]:my-1 [&_ol]:list-decimal [&_ol]:pl-5 [&_p]:my-2 [&_pre]:my-2 [&_pre]:overflow-auto [&_pre]:rounded-lg [&_pre]:bg-surface-sunken [&_pre]:p-3 [&_pre_code]:bg-transparent [&_pre_code]:p-0 [&_table]:my-2 [&_table]:w-full [&_table]:text-xs [&_td]:border [&_td]:border-border [&_td]:px-2 [&_td]:py-1 [&_th]:border [&_th]:border-border [&_th]:px-2 [&_th]:py-1 [&_ul]:list-disc [&_ul]:pl-5'

const pagerClass =
  'mt-4 flex flex-wrap items-center justify-between gap-3 text-xs text-muted-foreground'

export function DatasetSheet({
  id,
  datasets,
  canRun,
  onUse,
  onClose,
}: {
  id: string | null
  datasets: Dataset[]
  canRun: boolean
  onUse: (dataset: Dataset) => void
  onClose: () => void
}) {
  const retained = useRetainedValue(id)
  return (
    <DetailSheet open={!!id} onOpenChange={(open) => !open && onClose()} size="xl">
      {retained ? (
        <DatasetDetails
          key={retained}
          id={retained}
          dataset={datasets.find((item) => item.id === retained)}
          canRun={canRun}
          onUse={onUse}
          onBack={onClose}
        />
      ) : null}
    </DetailSheet>
  )
}

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
  const preparation = detail ? detail.provenance.preparation : dataset?.preparation
  const benchmarks = detail ? detail.benchmarks.map((item) => item.id) : dataset?.benchmarks
  const evaluationTitle = datasetEvaluationTitle(preparation, benchmarks)
  return (
    <>
      <DetailSheetHeader
        icon={<IconTile icon={ListChecks} tone="orange" size="lg" />}
        title={name}
        badges={<Badge tone="outline">{profileTitle(detail?.profile ?? dataset?.profile)}</Badge>}
        description={
          <>
            <span>{number(detail?.case_count ?? dataset?.case_count)} questions</span>
            {detail && (
              <>
                <span className="text-subtle-foreground">·</span>
                <span>{number(detail.benchmarks.length)} benchmarks</span>
              </>
            )}
            {evaluationTitle && (
              <>
                <span className="text-subtle-foreground">·</span>
                <span>{evaluationTitle}</span>
              </>
            )}
          </>
        }
      />
      <DetailSheetBody>
        <section aria-label="Dataset details" className="min-w-0 px-5 pb-6 pt-4">
          <DatasetExposure preparation={preparation} benchmarks={benchmarks} />
          {error ? (
            <Notice tone="danger" role="alert">
              <p>{error}</p>
              <div>
                <Button variant="outline" size="xs" onClick={() => setRetry((value) => value + 1)}>
                  Retry loading dataset
                </Button>
              </div>
            </Notice>
          ) : !detail ? (
            <BenchLoading label="Loading dataset" lines={4} />
          ) : (
            <>
              <nav className={`${lineTabList} mb-4 mt-2`} aria-label="Dataset sections">
                {(
                  [
                    { id: 'questions', title: 'Questions', icon: ListChecks },
                    { id: 'coverage', title: 'Coverage', icon: ChartColumn },
                    { id: 'about', title: 'About', icon: FileSearch },
                  ] as const
                ).map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    className={lineTab}
                    aria-current={tab === item.id ? 'page' : undefined}
                    onClick={() => setTab(item.id)}
                  >
                    <item.icon aria-hidden="true" />
                    {item.title}
                  </button>
                ))}
              </nav>
              {tab === 'questions' && <DatasetQuestions id={id} detail={detail} />}
              {tab === 'coverage' && <DatasetCoverage detail={detail} />}
              {tab === 'about' && (
                <div className="max-w-3xl">
                  <h3 className={benchText.h3}>Source and reproducibility</h3>
                  <p className={`${benchText.muted} mt-1 text-[13px]`}>
                    Every run using this dataset evaluates the same frozen questions. Reference
                    answers and hidden tests stay out of the question browser.
                  </p>
                  {detail.provenance.sources.length > 0 && (
                    <ul className="mt-3 grid gap-2">
                      {detail.provenance.sources.map((source) => (
                        <li
                          key={source.benchmark}
                          className="rounded-lg border border-border bg-surface px-3.5 py-3"
                        >
                          <div className="flex flex-wrap items-center justify-between gap-2">
                            <strong className="text-[13px] font-medium text-foreground">
                              {benchmarkTitle(source.benchmark)}
                            </strong>
                            {safeLink(source.url) && (
                              <Button variant="ghost" size="xs" asChild>
                                <a href={source.url} target="_blank" rel="noreferrer">
                                  Dataset source <ArrowUpRight aria-hidden="true" />
                                </a>
                              </Button>
                            )}
                          </div>
                          {source.access_note && (
                            <p className={`${benchText.muted} mt-1`}>{source.access_note}</p>
                          )}
                        </li>
                      ))}
                    </ul>
                  )}
                  <Disclosure summary="Reproducibility details">
                    <dl className={`${identityClass} my-0`}>
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
                  </Disclosure>
                </div>
              )}
            </>
          )}
        </section>
      </DetailSheetBody>
      <DetailSheetFooter>
        <span className="text-xs text-muted-foreground">
          Frozen questions · reused by every run
        </span>
        <div className="ml-auto flex items-center gap-2">
          <Button variant="ghost" size="sm" onClick={onBack}>
            Close
          </Button>
          <Button
            size="sm"
            disabled={!canRun || !dataset}
            onClick={() => dataset && onUse(dataset)}
          >
            <Play aria-hidden="true" />
            Evaluate dataset
          </Button>
        </div>
      </DetailSheetFooter>
    </>
  )
}

/** One-line preview of a markdown question: emphasis and code markers are dropped. */
function previewText(markdown: string) {
  return markdown.replace(/(\*\*|__|`)/g, '').replace(/\s+/g, ' ')
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
    <div className="grid min-w-0 gap-5 lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
      <div className="min-w-0">
        <h3 className={benchText.h3}>Benchmark coverage</h3>
        <p className={`${benchText.muted} mt-0.5`}>
          Question counts in this case set. Select a benchmark to inspect its subject groups.
        </p>
        <div className="mt-3 grid gap-1">
          {detail.benchmarks.map((item, index) => (
            <button
              key={item.id}
              type="button"
              aria-pressed={selected === item.id}
              onClick={() => {
                setSelected(selected === item.id ? '' : item.id)
                setPage(0)
              }}
              className="grid min-w-0 grid-cols-[minmax(0,9rem)_minmax(0,1fr)_auto] items-center gap-3 rounded-md px-2 py-1.5 text-left text-xs outline-none transition-colors hover:bg-accent focus-visible:ring-[3px] focus-visible:ring-ring/40 aria-pressed:bg-primary-soft/70"
            >
              <span className="truncate text-foreground/90">
                {item.title || benchmarkTitle(item.id)}
              </span>
              <span className="h-1.5 overflow-hidden rounded-full bg-muted">
                <span
                  className="block h-full rounded-full"
                  style={{
                    width: `${(100 * item.count) / maximum}%`,
                    background: barColors[index % barColors.length],
                  }}
                />
              </span>
              <strong className="font-medium text-foreground tabular-nums">
                {number(item.count)}
              </strong>
            </button>
          ))}
        </div>
      </div>
      <BenchBlock className="min-w-0">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <h3 className={benchText.h3}>
            {selected ? benchmarkTitle(selected) : 'All benchmarks'} · Subject groups
          </h3>
          <Badge tone="outline">{number(categories.length)} groups</Badge>
        </div>
        <div className="grid grid-cols-[repeat(auto-fill,minmax(min(200px,100%),1fr))] gap-2">
          {categories.slice(page * 12, (page + 1) * 12).map((item) => (
            <div
              key={`${item.benchmark}:${item.name}`}
              className="flex min-w-0 items-center justify-between gap-3 rounded-md bg-surface-subtle px-3 py-2"
            >
              <div className="min-w-0">
                <strong className="block truncate text-[13px] font-medium text-foreground">
                  {item.name || 'General'}
                </strong>
                {!selected && (
                  <span className="block truncate text-[11px] text-muted-foreground">
                    {benchmarkTitle(item.benchmark)}
                  </span>
                )}
              </div>
              <b className="text-[13px] font-semibold text-foreground tabular-nums">
                {number(item.count)}
              </b>
            </div>
          ))}
        </div>
        {!categories.length && (
          <p className={benchText.muted}>No subject grouping is provided by this source.</p>
        )}
        {categories.length > 12 && (
          <nav className={pagerClass} aria-label="Subject group pages">
            <span className="tabular-nums">
              {page * 12 + 1}–{Math.min((page + 1) * 12, categories.length)} of {categories.length}
            </span>
            <div className="flex items-center gap-1.5">
              <Button
                variant="outline"
                size="xs"
                disabled={!page}
                onClick={() => setPage(page - 1)}
              >
                Previous groups
              </Button>
              <Button
                variant="outline"
                size="xs"
                disabled={(page + 1) * 12 >= categories.length}
                onClick={() => setPage(page + 1)}
              >
                Next groups
              </Button>
            </div>
          </nav>
        )}
      </BenchBlock>
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
      <article className="min-w-0 max-w-4xl">
        <Button variant="ghost" size="sm" className="-ml-2 mb-3" onClick={() => setQuestion(null)}>
          <ArrowLeft aria-hidden="true" />
          Back to questions
        </Button>
        <div className="flex flex-wrap gap-1.5">
          <Badge tone="outline">{benchmarkTitle(question.benchmark)}</Badge>
          {question.category && <Badge tone="neutral">{question.category}</Badge>}
        </div>
        <h3 className={`${benchText.h3} mt-4`}>Question</h3>
        {question.input_status === 'unavailable' ? (
          <Notice tone="warning" as="p">
            {question.input_notice || 'The source question is not available on this service.'}
          </Notice>
        ) : (
          <>
            <div
              className={cn(
                markdownClass,
                'mt-2 rounded-lg border border-border bg-surface px-4 py-3',
              )}
            >
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {question.question ||
                  question.messages.map((message) => message.content).join('\n\n')}
              </ReactMarkdown>
            </div>
            {!!question.choices?.length && (
              <ol
                className="mt-3 grid list-[upper-alpha] gap-2 pl-6 marker:font-medium marker:text-muted-foreground"
                type="A"
              >
                {question.choices.map((choice, index) => (
                  <li
                    key={index}
                    className={cn(
                      markdownClass,
                      'rounded-md border border-border bg-surface px-3 py-1.5 text-[13px] [&_p]:my-0.5',
                    )}
                  >
                    <ReactMarkdown>{choice}</ReactMarkdown>
                  </li>
                ))}
              </ol>
            )}
          </>
        )}
        {question.input_notice && question.input_status === 'available' && (
          <p className={`${benchText.muted} mt-2`}>{question.input_notice}</p>
        )}
        <Disclosure summary="Question reference">
          <code className="font-mono text-xs text-foreground">{question.id}</code>
        </Disclosure>
      </article>
    )
  return (
    <section aria-label="Dataset questions" className="min-w-0">
      <form
        className="grid min-w-0 grid-cols-1 items-end gap-3 md:grid-cols-[minmax(0,2fr)_minmax(0,1fr)_minmax(0,1fr)]"
        onSubmit={(event) => {
          event.preventDefault()
          setSearch(query.trim())
          reset()
        }}
      >
        <div className="flex min-w-0 flex-col gap-1.5">
          <label htmlFor={searchID} className="text-xs font-medium text-muted-foreground">
            Search questions
          </label>
          <div className="relative flex items-center">
            <Input
              id={searchID}
              type="search"
              maxLength={200}
              value={query}
              className="pr-10"
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Find a question"
            />
            <Button
              type="submit"
              variant="ghost"
              size="icon-xs"
              className="absolute right-1"
              aria-label="Submit question search"
            >
              <Search aria-hidden="true" />
            </Button>
          </div>
        </div>
        <BenchSelect
          label="Benchmark"
          value={benchmark}
          onChange={(value) => {
            setBenchmark(value)
            setCategory('')
            reset()
          }}
          options={[
            { value: '', label: 'All benchmarks' },
            ...detail.benchmarks.map((item) => ({
              value: item.id,
              label: `${item.title || benchmarkTitle(item.id)} (${item.count})`,
            })),
          ]}
        />
        <BenchSelect
          label="Subject group"
          value={category}
          onChange={(value) => {
            setCategory(value)
            reset()
          }}
          searchable
          options={[
            { value: '', label: 'All subjects' },
            ...categories.map((item) => ({ value: item, label: item || 'General' })),
          ]}
        />
      </form>
      {loading ? (
        <BenchLoading label="Loading questions" lines={5} />
      ) : error ? (
        <Notice tone="danger" role="alert">
          <p>{error}</p>
          <div>
            <Button variant="outline" size="xs" onClick={() => setRetry((value) => value + 1)}>
              Retry loading questions
            </Button>
          </div>
        </Notice>
      ) : (
        data && (
          <>
            <p className="mb-2 mt-4 text-xs text-muted-foreground tabular-nums">
              {number(data.total)} {data.total === 1 ? 'question' : 'questions'}
              {search ? ` matching “${search}”` : ''}
            </p>
            <div className="divide-y divide-border overflow-hidden rounded-lg border border-border bg-card empty:hidden">
              {data.cases.map((item, index) => (
                <button
                  key={item.id}
                  type="button"
                  className="group grid w-full min-w-0 grid-cols-[2.25rem_minmax(0,1fr)_auto] items-start gap-3 px-3 py-3 text-left outline-none transition-colors hover:bg-surface-subtle focus-visible:bg-surface-subtle focus-visible:ring-[3px] focus-visible:ring-inset focus-visible:ring-ring/40"
                  onClick={() => setQuestion(item)}
                  aria-label={`Open question ${Number(cursor) + index + 1}`}
                >
                  <span className="mt-0.5 grid h-6 place-items-center rounded-md bg-surface-subtle font-mono text-[11px] text-muted-foreground tabular-nums">
                    {Number(cursor) + index + 1}
                  </span>
                  <div className="min-w-0">
                    <span className="line-clamp-2 text-[13px] leading-relaxed text-foreground/90">
                      {item.input_status === 'available'
                        ? previewText(
                            item.question ||
                              item.messages.map((message) => message.content).join(' '),
                          )
                        : item.input_notice || 'Source question unavailable'}
                    </span>
                    <span className="mt-1 block text-[11.5px] text-muted-foreground">
                      {benchmarkTitle(item.benchmark)}
                      {item.category ? ` · ${item.category}` : ''}
                    </span>
                  </div>
                  <ChevronRight
                    aria-hidden="true"
                    className="mt-1 size-4 text-subtle-foreground transition-transform group-hover:translate-x-0.5"
                  />
                </button>
              ))}
            </div>
            {!data.cases.length && (
              <EmptyNote icon={SearchX} title="No matching questions">
                Try a different search or subject group.
              </EmptyNote>
            )}
            <nav className={pagerClass} aria-label="Question pages">
              <span className="tabular-nums">
                {data.total
                  ? `${Number(cursor) + 1}–${Number(cursor) + data.cases.length} of ${number(data.total)}`
                  : '0 questions'}
              </span>
              <div className="flex items-center gap-1.5">
                <Button
                  variant="outline"
                  size="xs"
                  disabled={!previous.length}
                  onClick={() => {
                    setCursor(previous[previous.length - 1])
                    setPrevious(previous.slice(0, -1))
                  }}
                >
                  <ChevronLeft aria-hidden="true" />
                  Previous questions
                </Button>
                <Button
                  variant="outline"
                  size="xs"
                  disabled={data.next_cursor === null}
                  onClick={() => {
                    setPrevious([...previous, cursor])
                    setCursor(String(data.next_cursor))
                  }}
                >
                  Next questions
                  <ChevronRight aria-hidden="true" />
                </Button>
              </div>
            </nav>
          </>
        )
      )}
    </section>
  )
}
