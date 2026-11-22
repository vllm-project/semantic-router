import * as React from 'react'
import { Link } from 'react-router-dom'
import { ArrowUpRight, ChevronRight, OctagonX, RotateCw, SlidersHorizontal } from 'lucide-react'
import {
  Alert,
  Badge,
  Button,
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
  IconTile,
  Skeleton,
  cn,
} from '../../ui'
import { PathFields } from '../ConfigPageRouterConfigSectionFields'
import { projectCanonicalConfigForManager } from '../configPageCanonicalization'
import { responseErrorMessage } from '../configPageRequestErrors'
import { getAt, isRecord, leafPaths, type JsonRecord } from '../configPageRouterSettingsDraft'
import {
  RouterSettingsEditorContext,
  type RouterSettingsEditor,
} from '../configPageRouterSettingsEditor'
import {
  routerSettingsSections,
  type RouterSettingsSection,
} from '../configPageRouterSettingsModel'
import { validateRouterSettings } from '../configPageRouterSettingsValidation'
import type { ConfigData } from '../configPageSupport'

// The editor shows router-wide settings with the Router settings page's own
// schema, field renderer and validator, read-only: recipes deploy from here,
// router settings save from their page, so each has one write path.

type Load =
  | { state: 'loading' }
  | { state: 'error'; message: string }
  | { state: 'ready'; global: JsonRecord; effective: JsonRecord | null }

const noop = () => undefined

function sectionEntries(section: RouterSettingsSection) {
  return section.master ? [section.master, ...section.fields] : section.fields
}

function explicitCount(section: RouterSettingsSection, global: JsonRecord): number {
  return sectionEntries(section).reduce(
    (total, field) => total + leafPaths(getAt(global, field.path)).length,
    0,
  )
}

function useRouterSettingsSnapshot() {
  const [load, setLoad] = React.useState<Load>({ state: 'loading' })
  const fetchSnapshot = React.useCallback(async (signal?: AbortSignal) => {
    setLoad({ state: 'loading' })
    try {
      const [configResponse, effectiveResponse] = await Promise.all([
        fetch('/api/router/config/all', { signal }),
        fetch('/api/router/config/global', { signal }).catch(() => null),
      ])
      if (!configResponse.ok) throw new Error(await responseErrorMessage(configResponse))
      const config = projectCanonicalConfigForManager((await configResponse.json()) as ConfigData)
      const effective =
        effectiveResponse && effectiveResponse.ok
          ? ((await effectiveResponse.json()) as unknown)
          : null
      setLoad({
        state: 'ready',
        global: isRecord(config.global) ? (config.global as JsonRecord) : {},
        effective: isRecord(effective) ? effective : null,
      })
    } catch (error) {
      if (signal?.aborted) return
      setLoad({
        state: 'error',
        message: error instanceof Error ? error.message : 'Router settings could not be loaded.',
      })
    }
  }, [])
  React.useEffect(() => {
    const controller = new AbortController()
    void fetchSnapshot(controller.signal)
    return () => controller.abort()
  }, [fetchSnapshot])
  return { load, reload: () => void fetchSnapshot() }
}

function SectionCard({
  section,
  count,
  errors,
}: {
  section: RouterSettingsSection
  count: number
  errors: number
}) {
  return (
    <section
      aria-labelledby={`${section.id}-editor-title`}
      className="overflow-hidden rounded-xl border border-border bg-card shadow-xs"
    >
      <header className="flex items-start gap-3 border-b border-border px-4 py-3">
        <IconTile icon={section.icon} tone={section.tone} size="sm" />
        <div className="flex min-w-0 flex-1 flex-col gap-0.5">
          <div className="flex items-center gap-2">
            <h3 id={`${section.id}-editor-title`} className="text-[13px] font-semibold">
              {section.title}
            </h3>
            <span className="text-xs tabular text-muted-foreground">{count} set</span>
            {errors ? (
              <Badge tone="danger" size="sm">
                {errors} invalid
              </Badge>
            ) : null}
          </div>
          <p className="text-xs text-muted-foreground">{section.description}</p>
        </div>
        <Button asChild variant="ghost" size="xs">
          <Link to={`/settings/router#${section.id}`}>
            Edit
            <ArrowUpRight />
          </Link>
        </Button>
      </header>
      <div className="@container px-4 py-4">
        {section.custom ? (
          <p className="text-[13px] text-muted-foreground">
            This section has its own editor in Router settings.
          </p>
        ) : (
          <PathFields entries={sectionEntries(section)} disabled />
        )}
      </div>
    </section>
  )
}

/** Router-wide settings, read-only, rendered by the Router settings schema and fields. */
export function EditorRouterSettings() {
  const { load, reload } = useRouterSettingsSnapshot()
  const sections = React.useMemo(() => routerSettingsSections(), [])
  const global = React.useMemo<JsonRecord>(
    () => (load.state === 'ready' ? load.global : {}),
    [load],
  )
  const errors = React.useMemo(
    () => (load.state === 'ready' ? validateRouterSettings(sections, global, {}) : new Map()),
    [global, load.state, sections],
  )
  const editor = React.useMemo<RouterSettingsEditor>(
    () => ({
      draft: global,
      saved: global,
      effective: load.state === 'ready' ? load.effective : null,
      readOnly: true,
      errors,
      query: '',
      show: 'set',
      set: noop,
      unset: noop,
      restore: noop,
      applyPatch: noop,
      reportInvalid: noop,
    }),
    [errors, global, load],
  )
  const configured = sections
    .map((section) => ({ section, count: explicitCount(section, global) }))
    .filter((entry) => entry.count > 0 && !entry.section.custom)
  const defaults = sections.filter(
    (section) => section.custom || explicitCount(section, global) === 0,
  )
  const sectionErrors = (section: RouterSettingsSection) =>
    sectionEntries(section).filter((field) =>
      [...errors.keys()].some((key) => key.startsWith(field.path.join('.'))),
    ).length

  return (
    <div className="mx-auto flex w-full max-w-[1040px] flex-col gap-4">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div className="flex min-w-0 items-start gap-3">
          <IconTile icon={SlidersHorizontal} tone="slate" size="md" />
          <div className="flex min-w-0 flex-col gap-1">
            <h2 className="text-[15px] font-semibold tracking-[-0.01em]">Router settings</h2>
            <p className="max-w-2xl text-[13px] text-muted-foreground">
              Router-wide defaults every recipe inherits. They are not part of the DSL and do not
              deploy from the editor; this view uses the same schema and checks as Router settings.
            </p>
          </div>
        </div>
        <Button asChild variant="outline" size="sm">
          <Link to="/settings/router">
            Open Router settings
            <ArrowUpRight />
          </Link>
        </Button>
      </header>

      {load.state === 'loading' ? (
        <div className="flex flex-col gap-3" aria-label="Loading router settings">
          {[0, 1].map((index) => (
            <div key={index} className="rounded-xl border border-border bg-card p-4">
              <Skeleton className="mb-4 h-4 w-48" />
              <div className="grid grid-cols-2 gap-4">
                <Skeleton className="h-9" />
                <Skeleton className="h-9" />
              </div>
            </div>
          ))}
        </div>
      ) : load.state === 'error' ? (
        <Alert
          tone="danger"
          title="Router settings could not be loaded"
          actions={
            <Button variant="outline" size="xs" onClick={reload}>
              <RotateCw />
              Retry
            </Button>
          }
        >
          {load.message}
        </Alert>
      ) : (
        <RouterSettingsEditorContext.Provider value={editor}>
          {errors.size ? (
            <Alert
              tone="danger"
              icon={<OctagonX />}
              title={`${errors.size} ${errors.size === 1 ? 'setting does' : 'settings do'} not match the contract`}
            >
              The Router settings page shows the same problems and lets you fix them.
            </Alert>
          ) : null}
          {configured.length === 0 ? (
            <Alert tone="neutral" title="Every router setting uses its default">
              Nothing is overridden in config.yaml.
            </Alert>
          ) : (
            configured.map(({ section, count }) => (
              <SectionCard
                key={section.id}
                section={section}
                count={count}
                errors={sectionErrors(section)}
              />
            ))
          )}
          {defaults.length ? (
            <Collapsible>
              <CollapsibleTrigger className="group flex w-full items-center gap-2 rounded-lg px-1 py-1.5 text-left text-xs font-medium text-muted-foreground outline-none transition-colors hover:text-foreground focus-visible:ring-[3px] focus-visible:ring-ring/40">
                <ChevronRight className="size-3.5 transition-transform group-data-[state=open]:rotate-90" />
                {defaults.length} {defaults.length === 1 ? 'section uses' : 'sections use'} the
                Router defaults
              </CollapsibleTrigger>
              <CollapsibleContent>
                <ul className="mt-1 grid gap-1.5 sm:grid-cols-2">
                  {defaults.map((section) => (
                    <li key={section.id}>
                      <Link
                        to={`/settings/router#${section.id}`}
                        className={cn(
                          'flex items-center gap-2.5 rounded-lg border border-border bg-card px-3 py-2 text-[13px] outline-none transition-colors hover:bg-surface-subtle focus-visible:ring-[3px] focus-visible:ring-ring/40',
                        )}
                      >
                        <section.icon className="size-3.5 text-subtle-foreground" />
                        <span className="min-w-0 flex-1 truncate">{section.title}</span>
                        <ArrowUpRight className="size-3.5 text-subtle-foreground" />
                      </Link>
                    </li>
                  ))}
                </ul>
              </CollapsibleContent>
            </Collapsible>
          ) : null}
        </RouterSettingsEditorContext.Provider>
      )}
    </div>
  )
}
