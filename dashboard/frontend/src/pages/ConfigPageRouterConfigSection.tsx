import * as React from 'react'
import { useLocation, useNavigate, useSearchParams } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import {
  Braces,
  FileCode2,
  ListChecks,
  Lock,
  RotateCw,
  SearchX,
  SlidersHorizontal,
} from 'lucide-react'
import {
  Alert,
  Badge,
  Button,
  Card,
  CardDescription,
  CardTitle,
  DataToolbar,
  EmptyState,
  IconTile,
  Kbd,
  Segmented,
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
  Switch,
  Tooltip,
  cn,
  toast,
} from '../ui'
import ConfigSaveWarnings from '../components/ConfigSaveWarnings'
import { useAuth } from '../contexts/AuthContext'
import { useReadonly } from '../contexts/ReadonlyContext'
import { canWriteConfig } from '../utils/accessControl'
import { configSaveWarnings, type RouterConfigWarning } from '../utils/configSaveWarnings'
import ConfigPageListenerVSRPanel from './ConfigPageListenerVSRPanel'
import ConfigPageManagerLayout from './ConfigPageManagerLayout'
import { SettingsAside } from './SettingsLayout'
import { DecisionModelSettings, EmbeddingSettings } from './ConfigPageRouterConfigSectionCustom'
import { PathFields } from './ConfigPageRouterConfigSectionFields'
import {
  RouterSettingsEditorContext,
  fieldVisible,
  type RouterSettingsEditor,
  type SettingsShow,
} from './configPageRouterSettingsEditor'
import {
  RouterSettingsReviewDialog,
  type RestartNotice,
} from './ConfigPageRouterConfigSectionReview'
import { RouterSettingsYamlPanel } from './ConfigPageRouterConfigSectionYaml'
import { responseErrorMessage } from './configPageRequestErrors'
import {
  buildEffectiveRouterConfig,
  buildRouterSectionCards,
} from './configPageRouterDefaultsSupport'
import type { OpenEditModal } from './configPageRouterSectionSupport'
import {
  applyMergePatch,
  changedLeafPaths,
  cloneJson,
  deepEqual,
  getAt,
  isRecord,
  leafPaths,
  mergePatchDiff,
  rebaseDraft,
  setAt,
  startsWithPath,
  unsetAt,
  type JsonRecord,
  type SettingPath,
} from './configPageRouterSettingsDraft'
import {
  ROUTER_CONTRACT_VERSION,
  ROUTER_SETTINGS_LAYERS,
  restartReason,
  routerSettingsSections,
  type RouterSettingsSection,
} from './configPageRouterSettingsModel'
import { validateRouterSettings } from './configPageRouterSettingsValidation'
import type { CanonicalGlobalConfig, ConfigData } from './configPageSupport'

interface ConfigPageRouterConfigSectionProps {
  config: ConfigData | null
  isReadonly: boolean
  openEditModal?: OpenEditModal
  refreshConfig?: () => Promise<void>
  saveConfig: (config: ConfigData) => Promise<void>
}

const VSR_ID = 'global-layer-vsr'
const LEGACY_ANCHORS: Record<string, string> = {
  clear_route_cache: 'router_core',
  model_selection: 'router_core',
}

function sectionChangeCount(section: RouterSettingsSection, changes: string[][]): number {
  return changes.filter((path) =>
    section.roots.some((root) => {
      if (!startsWithPath(path, root.path)) return false
      return !(root.omit ?? []).some((child) => startsWithPath(path, [...root.path, child]))
    }),
  ).length
}

function sectionExplicitCount(section: RouterSettingsSection, draft: JsonRecord): number {
  return [...section.fields, ...(section.master ? [section.master] : [])].reduce(
    (total, field) => total + leafPaths(getAt(draft, field.path)).length,
    0,
  )
}

function useScrollSpy(ids: string[]) {
  const [active, setActive] = React.useState(ids[0] ?? '')
  const key = ids.join('|')
  React.useEffect(() => {
    const root = document.getElementById('main')
    const list = key.split('|').filter(Boolean)
    if (!root || list.length === 0) return
    let frame = 0
    const update = () => {
      frame = 0
      const threshold = root.getBoundingClientRect().top + 140
      let current = list[0]
      for (const id of list) {
        const element = document.getElementById(id)
        if (element && element.getBoundingClientRect().top <= threshold) current = id
      }
      if (root.scrollTop + root.clientHeight >= root.scrollHeight - 8)
        current = list[list.length - 1]
      setActive(current)
    }
    const onScroll = () => {
      if (!frame) frame = window.requestAnimationFrame(update)
    }
    update()
    root.addEventListener('scroll', onScroll, { passive: true })
    return () => {
      root.removeEventListener('scroll', onScroll)
      if (frame) window.cancelAnimationFrame(frame)
    }
  }, [key])
  return [active, setActive] as const
}

function SectionIndex({
  groups,
  active,
  edits,
  onSelect,
}: {
  groups: Array<{ title: string; items: Array<{ id: string; title: string }> }>
  active: string
  edits: Record<string, number>
  onSelect: (id: string) => void
}) {
  return (
    <nav aria-label="Router settings sections" className="flex flex-col gap-4">
      <p className="px-2.5 text-[11px] font-medium uppercase tracking-[0.06em] text-subtle-foreground">
        On this page
      </p>
      {groups.map((group) => (
        <div key={group.title} className="flex flex-col gap-1">
          <p className="px-2.5 text-xs font-medium text-muted-foreground">{group.title}</p>
          <ul className="ml-2.5 flex flex-col border-l border-border">
            {group.items.map((item) => {
              const isActive = item.id === active
              return (
                <li key={item.id} className="relative">
                  {isActive ? (
                    <motion.span
                      layoutId="router-settings-index-active"
                      className="absolute -left-px bottom-1 top-1 w-0.5 rounded-full bg-primary"
                      transition={{ type: 'spring', bounce: 0.2, duration: 0.4 }}
                    />
                  ) : null}
                  <button
                    type="button"
                    onClick={() => onSelect(item.id)}
                    aria-current={isActive ? 'location' : undefined}
                    className={cn(
                      'flex w-full items-center gap-2 rounded-r-md py-1 pl-3 pr-1.5 text-left text-[12.5px] outline-none transition-colors focus-visible:ring-[3px] focus-visible:ring-ring/40',
                      isActive
                        ? 'font-medium text-foreground'
                        : 'text-muted-foreground hover:text-foreground',
                    )}
                  >
                    <span className="truncate">{item.title}</span>
                    {edits[item.id] ? (
                      <span className="ml-auto rounded-full bg-primary-soft px-1.5 font-mono text-[10.5px] tabular text-primary-soft-foreground">
                        {edits[item.id]}
                      </span>
                    ) : null}
                  </button>
                </li>
              )
            })}
          </ul>
        </div>
      ))}
    </nav>
  )
}

function SectionCard({
  section,
  editor,
  changeCount,
  highlight,
  children,
}: {
  section: RouterSettingsSection
  editor: RouterSettingsEditor
  changeCount: number
  highlight: boolean
  children: (disabled: boolean) => React.ReactNode
}) {
  const master = section.master
  const masterExplicit = master ? getAt(editor.draft, master.path) : undefined
  const masterInherited =
    master && getAt(editor.saved, master.path) === undefined
      ? getAt(editor.effective, master.path)
      : undefined
  const masterOn = master ? (masterExplicit ?? masterInherited) === true : true
  const masterEdited = master ? !deepEqual(masterExplicit, getAt(editor.saved, master.path)) : false
  const body = children(!masterOn)

  return (
    <Card
      id={section.id}
      className={cn(
        'scroll-mt-6 transition-shadow duration-700',
        highlight && 'shadow-[0_0_0_3px_var(--ring)]',
      )}
    >
      <div className="flex items-start gap-3 px-5 pb-4 pt-4">
        <IconTile icon={section.icon} tone={section.tone} size="sm" className="mt-0.5" />
        <div className="flex min-w-0 flex-1 flex-col gap-1">
          <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
            <CardTitle>{section.title}</CardTitle>
            {section.paths.map((path) => (
              <code key={path} className="font-mono text-[11px] text-subtle-foreground">
                {path}
              </code>
            ))}
            {changeCount ? (
              <Badge size="sm" tone="brand" className="self-center">
                {changeCount} edited
              </Badge>
            ) : null}
          </div>
          <CardDescription className="leading-relaxed">{section.description}</CardDescription>
        </div>
        {master ? (
          <span className="flex shrink-0 items-center gap-2 pt-0.5">
            {masterEdited ? (
              <span className="size-1.5 rounded-full bg-primary" role="img" aria-label="Edited" />
            ) : null}
            {masterExplicit === undefined ? (
              <Badge size="sm" tone="outline" className="font-normal">
                inherited
              </Badge>
            ) : null}
            <span className="text-xs text-muted-foreground">{masterOn ? 'On' : 'Off'}</span>
            <Switch
              checked={masterOn}
              disabled={editor.readOnly}
              onCheckedChange={(checked) => editor.set(master.path, checked)}
              aria-label={`${section.title} enabled`}
            />
          </span>
        ) : null}
      </div>
      {body ? (
        <div
          className={cn(
            'flex flex-col gap-5 border-t border-border px-5 pb-5 pt-5 @container transition-opacity',
            !masterOn && 'opacity-60',
          )}
        >
          {!masterOn ? (
            <p className="text-xs text-muted-foreground">
              Off. Turn it on to change these settings.
            </p>
          ) : null}
          {body}
        </div>
      ) : null}
    </Card>
  )
}

export default function ConfigPageRouterConfigSection({
  config,
  isReadonly,
  saveConfig,
  refreshConfig,
}: ConfigPageRouterConfigSectionProps) {
  const navigate = useNavigate()
  const { hash } = useLocation()
  const [params, setParams] = useSearchParams()
  const { isReadonly: dashboardReadonly } = useReadonly()
  const { user } = useAuth()
  const view = params.get('view') === 'yaml' ? 'yaml' : 'form'
  const sections = React.useMemo(() => routerSettingsSections(), [])

  const configGlobal = React.useMemo<JsonRecord>(
    () => (isRecord(config?.global) ? cloneJson(config.global as JsonRecord) : {}),
    [config],
  )
  const savedRef = React.useRef<JsonRecord>(configGlobal)
  const [saved, setSaved] = React.useState<JsonRecord>(configGlobal)
  const [draft, setDraft] = React.useState<JsonRecord>(configGlobal)
  const adoptSaved = React.useCallback((next: JsonRecord) => {
    const previous = savedRef.current
    savedRef.current = next
    setSaved(next)
    setDraft((current) => rebaseDraft(previous, current, next))
  }, [])
  React.useEffect(() => {
    if (!deepEqual(savedRef.current, configGlobal)) adoptSaved(configGlobal)
  }, [adoptSaved, configGlobal])

  const [effective, setEffective] = React.useState<JsonRecord | null>(null)
  const [effectiveState, setEffectiveState] = React.useState<'loading' | 'ready' | 'error'>(
    'loading',
  )
  const loadEffective = React.useCallback(async () => {
    try {
      const response = await fetch('/api/router/config/global')
      if (!response.ok) throw new Error(await responseErrorMessage(response))
      const body = (await response.json()) as CanonicalGlobalConfig
      setEffective(isRecord(body) ? (body as JsonRecord) : null)
      setEffectiveState('ready')
    } catch {
      setEffective(null)
      setEffectiveState('error')
    }
  }, [])

  const [rawYaml, setRawYaml] = React.useState('')
  const [rawState, setRawState] = React.useState<{ loading: boolean; error: string | null }>({
    loading: true,
    error: null,
  })
  const loadRaw = React.useCallback(async () => {
    setRawState({ loading: true, error: null })
    try {
      const response = await fetch('/api/router/config/global/raw')
      if (!response.ok)
        throw new Error(
          (await response.text()) || `HTTP ${response.status}: ${response.statusText}`,
        )
      setRawYaml((await response.text()) || '{}\n')
      setRawState({ loading: false, error: null })
    } catch (error) {
      setRawState({
        loading: false,
        error: error instanceof Error ? error.message : 'Failed to load the effective global YAML.',
      })
    }
  }, [])

  React.useEffect(() => {
    void loadEffective()
    void loadRaw()
  }, [loadEffective, loadRaw])

  const refreshSaved = React.useCallback(async () => {
    await refreshConfig?.()
    void loadEffective()
    void loadRaw()
  }, [loadEffective, loadRaw, refreshConfig])

  const readOnlyReason = dashboardReadonly
    ? 'The dashboard runs in read-only mode, so router settings can be viewed but not changed.'
    : !canWriteConfig(user)
      ? 'Your role can view router settings. Changing them needs the config.write permission.'
      : isReadonly
        ? 'Router settings are read-only here.'
        : null
  const readOnly = isReadonly || readOnlyReason !== null

  const [search, setSearch] = React.useState('')
  const [show, setShow] = React.useState<SettingsShow>('all')
  const query = search.trim().toLowerCase()
  const [invalid, setInvalid] = React.useState<Record<string, string>>({})
  const reportInvalid = React.useCallback((key: string, message: string | null) => {
    setInvalid((current) => {
      if (message === null) {
        if (!(key in current)) return current
        const next = { ...current }
        delete next[key]
        return next
      }
      return current[key] === message ? current : { ...current, [key]: message }
    })
  }, [])

  const changes = React.useMemo(() => changedLeafPaths(saved, draft), [draft, saved])
  const dirty = changes.length > 0
  const errors = React.useMemo(() => {
    const found = validateRouterSettings(sections, draft, saved)
    for (const [key, message] of Object.entries(invalid)) found.set(key, message)
    return found
  }, [draft, invalid, saved, sections])
  const restartNotices = React.useMemo<RestartNotice[]>(() => {
    const notices = new Map<string, RestartNotice>()
    for (const path of changes) {
      const reason = restartReason(path)
      if (reason) notices.set(path.join('.'), { path: `global.${path.join('.')}`, reason })
    }
    return [...notices.values()]
  }, [changes])

  const editor = React.useMemo<RouterSettingsEditor>(
    () => ({
      draft,
      saved,
      effective,
      readOnly,
      errors,
      query,
      show,
      set: (path: SettingPath, value: unknown) =>
        setDraft((current) => setAt(current, path, value)),
      unset: (path: SettingPath) => setDraft((current) => unsetAt(current, path)),
      restore: (path: SettingPath) =>
        setDraft((current) => {
          const previous = getAt(savedRef.current, path)
          return previous === undefined
            ? unsetAt(current, path)
            : setAt(current, path, cloneJson(previous))
        }),
      applyPatch: (patch: unknown) =>
        setDraft((current) => applyMergePatch(current, patch) as JsonRecord),
      reportInvalid,
    }),
    [draft, effective, errors, query, readOnly, reportInvalid, saved, show],
  )

  // The two support-module cards read the draft as if it were saved.
  const legacyCards = React.useMemo(() => {
    const draftConfig = { ...(config ?? {}), global: draft } as ConfigData
    const draftEffective = effective
      ? (applyMergePatch(effective, mergePatchDiff(saved, draft) ?? {}) as CanonicalGlobalConfig)
      : null
    return buildRouterSectionCards({
      config: draftConfig,
      routerConfig: buildEffectiveRouterConfig(draftEffective, draftConfig),
      routerDefaults: draftEffective,
      only: ['system_models', 'embedding_models'],
    })
  }, [config, draft, effective, saved])
  const systemCard = legacyCards.find((card) => card.key === 'system_models')
  const embeddingCard = legacyCards.find((card) => card.key === 'embedding_models')

  const sectionEdits = React.useMemo(
    () =>
      Object.fromEntries(
        sections.map((section) => [section.id, sectionChangeCount(section, changes)]),
      ),
    [changes, sections],
  )

  const visibleSections = sections.filter((section) => {
    const titleMatch =
      query !== '' &&
      `${section.title} ${section.description} ${section.paths.join(' ')}`
        .toLowerCase()
        .includes(query)
    if (show === 'edited' && sectionEdits[section.id] === 0) return false
    if (show === 'set' && sectionExplicitCount(section, draft) === 0) return false
    if (section.custom) return query === '' || titleMatch
    if (titleMatch) return true
    const scoped = { ...editor }
    return (
      section.fields.some((field) => fieldVisible(scoped, field.schema, field.key, field.path)) ||
      Boolean(
        section.master &&
          fieldVisible(scoped, section.master.schema, section.master.key, section.master.path),
      )
    )
  })

  const listenerCount = (config?.listeners ?? []).filter((listener) =>
    Boolean(listener?.apis?.systemone),
  ).length
  const showVsr = view === 'form' && listenerCount > 0 && query === '' && show === 'all'
  const indexGroups = [
    ...ROUTER_SETTINGS_LAYERS.map((layer) => ({
      title: layer.title,
      items: visibleSections
        .filter((section) => section.layer === layer.key)
        .map((section) => ({ id: section.id, title: section.title })),
    })).filter((group) => group.items.length > 0),
    ...(showVsr ? [{ title: 'Agents', items: [{ id: VSR_ID, title: 'vSR agent plugins' }] }] : []),
  ]
  const indexIds = indexGroups.flatMap((group) => group.items.map((item) => item.id))
  const [activeSection, setActiveSection] = useScrollSpy(view === 'form' ? indexIds : [])
  const [flash, setFlash] = React.useState<string | null>(null)

  const scrollTo = React.useCallback(
    (id: string) => {
      setActiveSection(id)
      setFlash(id)
      document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
      window.setTimeout(() => setFlash((current) => (current === id ? null : current)), 1200)
    },
    [setActiveSection],
  )

  React.useEffect(() => {
    if (view !== 'form' || !hash.startsWith('#global-')) return
    const raw = hash.slice(1)
    const legacy = raw.replace(/^global-section-/, '')
    const id = LEGACY_ANCHORS[legacy] ? `global-section-${LEGACY_ANCHORS[legacy]}` : raw
    const frame = window.requestAnimationFrame(() =>
      document.getElementById(id)?.scrollIntoView({ block: 'start' }),
    )
    return () => window.cancelAnimationFrame(frame)
  }, [hash, view])

  const [reviewOpen, setReviewOpen] = React.useState(false)
  const [saving, setSaving] = React.useState(false)
  const [saveError, setSaveError] = React.useState<string | null>(null)
  const [saveWarnings, setSaveWarnings] = React.useState<RouterConfigWarning[]>([])
  const canReview = dirty && errors.size === 0 && !readOnly

  const openReview = React.useCallback(() => {
    if (!canReview) return
    setSaveError(null)
    setReviewOpen(true)
  }, [canReview])

  React.useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (
        (event.metaKey || event.ctrlKey) &&
        event.key.toLowerCase() === 's' &&
        view === 'form' &&
        dirty
      ) {
        event.preventDefault()
        openReview()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [dirty, openReview, view])

  React.useEffect(() => {
    if (!dirty) return
    const onBeforeUnload = (event: BeforeUnloadEvent) => {
      event.preventDefault()
      event.returnValue = ''
    }
    window.addEventListener('beforeunload', onBeforeUnload)
    return () => window.removeEventListener('beforeunload', onBeforeUnload)
  }, [dirty])

  const discard = () => {
    const previous = draft
    setDraft(saved)
    setInvalid({})
    toast('Changes discarded', { action: { label: 'Undo', onClick: () => setDraft(previous) } })
  }

  const announceSave = (body: unknown, count: number) => {
    const status = isRecord(body) ? body.status : undefined
    const message = isRecord(body) && typeof body.message === 'string' ? body.message : undefined
    if (status === 'restart_required') {
      toast.warning('Saved. The router needs a restart', {
        description: message ?? 'Run vllm-sr serve to apply the saved settings.',
      })
    } else if (status === 'persisted') {
      toast('Saved to the ConfigMap', {
        description: message ?? 'Roll out the Router and Envoy deployments to apply it.',
      })
    } else {
      toast.success('Router settings saved', {
        description: `${count} ${count === 1 ? 'change' : 'changes'} applied with hot reload.`,
      })
    }
  }

  const save = async () => {
    const patch = mergePatchDiff(saved, draft)
    if (patch === undefined) {
      setReviewOpen(false)
      return
    }
    setSaving(true)
    setSaveError(null)
    try {
      const response = await fetch('/api/router/config/global/update', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(patch),
      })
      if (!response.ok) throw new Error(await responseErrorMessage(response))
      const body: unknown = await response.json().catch(() => ({}))
      setSaveWarnings(configSaveWarnings(body))
      const count = changes.length
      adoptSaved(cloneJson(draft))
      setReviewOpen(false)
      announceSave(body, count)
      void refreshSaved()
    } catch (error) {
      setSaveError(error instanceof Error ? error.message : 'Failed to save router settings.')
    } finally {
      setSaving(false)
    }
  }

  const saveRawYaml = async (yaml: string) => {
    setSaveWarnings([])
    const response = await fetch('/api/router/config/global/raw/update', {
      method: 'POST',
      headers: { 'Content-Type': 'text/yaml; charset=utf-8' },
      body: yaml,
    })
    if (!response.ok)
      throw new Error((await response.text()) || `HTTP ${response.status}: ${response.statusText}`)
    const body: unknown = await response.json().catch(() => ({}))
    setSaveWarnings(configSaveWarnings(body))
    announceSave(body, 1)
    await refreshSaved()
  }

  const explicitTotal = leafPaths(draft).length
  const inheritedTotal = React.useMemo(
    () =>
      effective
        ? leafPaths(effective).filter((path) => getAt(draft, path) === undefined).length
        : null,
    [draft, effective],
  )
  const switchView = (next: 'form' | 'yaml') => {
    const nextParams = new URLSearchParams(params)
    if (next === 'yaml') nextParams.set('view', 'yaml')
    else nextParams.delete('view')
    setParams(nextParams, { replace: true })
  }

  const renderSectionBody = (section: RouterSettingsSection, disabled: boolean) => {
    if (section.custom === 'decision-model')
      return systemCard ? <DecisionModelSettings card={systemCard} disabled={disabled} /> : null
    if (section.custom === 'embeddings')
      return embeddingCard ? <EmbeddingSettings card={embeddingCard} disabled={disabled} /> : null
    if (section.fields.length === 0) return null
    const titleMatch =
      query !== '' &&
      `${section.title} ${section.description} ${section.paths.join(' ')}`
        .toLowerCase()
        .includes(query)
    const bare = section.roots.length === 1 && section.roots[0].mode === 'field'
    const content = <PathFields entries={section.fields} disabled={disabled} bare={bare} />
    if (!titleMatch) return content
    return (
      <RouterSettingsEditorContext.Provider value={{ ...editor, query: '' }}>
        {content}
      </RouterSettingsEditorContext.Provider>
    )
  }

  const firstError = errors.size ? [...errors.values()][0] : null

  return (
    <ConfigPageManagerLayout
      title="Router settings"
      description="Router-wide defaults that every recipe inherits. Recipes and decisions override them field by field."
      icon={<IconTile icon={SlidersHorizontal} tone="slate" size="lg" />}
      actions={
        <>
          <Segmented
            aria-label="View"
            value={view}
            onValueChange={switchView}
            options={[
              { value: 'form', label: 'Form', icon: <ListChecks /> },
              { value: 'yaml', label: 'YAML', icon: <Braces /> },
            ]}
          />
          <Button variant="outline" onClick={() => navigate('/routing/editor')}>
            <FileCode2 />
            Edit as code
          </Button>
        </>
      }
    >
      <RouterSettingsEditorContext.Provider value={editor}>
        <div className="flex flex-col gap-4">
          {readOnlyReason ? (
            <Alert tone="neutral" title="View only" icon={<Lock />}>
              {readOnlyReason}
            </Alert>
          ) : null}
          {effectiveState === 'error' ? (
            <Alert
              tone="warning"
              title="Router defaults are unavailable"
              actions={
                <Button size="sm" variant="outline" onClick={() => void loadEffective()}>
                  Try again
                </Button>
              }
            >
              The form shows what config.yaml sets. Inherited values appear once the defaults load
              again.
            </Alert>
          ) : null}
          <ConfigSaveWarnings warnings={saveWarnings} onDismiss={() => setSaveWarnings([])} />

          <div>
            <div className="min-w-0">
              {view === 'form' ? (
                <>
                  <DataToolbar
                    search={search}
                    onSearchChange={setSearch}
                    placeholder="Find a setting…"
                    filters={
                      <Segmented
                        size="sm"
                        aria-label="Show"
                        value={show}
                        onValueChange={setShow}
                        options={[
                          { value: 'all', label: 'All' },
                          { value: 'set', label: 'Set in config' },
                          {
                            value: 'edited',
                            label: changes.length ? `Edited · ${changes.length}` : 'Edited',
                          },
                        ]}
                      />
                    }
                    actions={
                      <>
                        <Tooltip
                          content={`Writes the global section of config.yaml. Empty fields inherit the router default. Checked against the ${ROUTER_CONTRACT_VERSION} contract.`}
                        >
                          <span className="hidden cursor-default items-center gap-1.5 text-xs text-muted-foreground lg:inline-flex">
                            <FileCode2 className="size-3.5 text-subtle-foreground" />
                            <span>
                              <span className="tabular">{explicitTotal}</span> set in config.yaml
                              {inheritedTotal !== null ? (
                                <>
                                  {' '}
                                  · <span className="tabular">{inheritedTotal}</span> inherited
                                </>
                              ) : null}
                            </span>
                          </span>
                        </Tooltip>
                        {indexIds.length > 1 ? (
                          <div className="xl:hidden">
                            <Select value={activeSection} onValueChange={scrollTo}>
                              <SelectTrigger
                                size="sm"
                                aria-label="Jump to a section"
                                className="min-w-44"
                              >
                                <SelectValue placeholder="Jump to…" />
                              </SelectTrigger>
                              <SelectContent align="end">
                                {indexGroups.map((group) => (
                                  <SelectGroup key={group.title}>
                                    <SelectLabel>{group.title}</SelectLabel>
                                    {group.items.map((item) => (
                                      <SelectItem key={item.id} value={item.id}>
                                        {item.title}
                                        {sectionEdits[item.id]
                                          ? ` · ${sectionEdits[item.id]} edited`
                                          : ''}
                                      </SelectItem>
                                    ))}
                                  </SelectGroup>
                                ))}
                              </SelectContent>
                            </Select>
                          </div>
                        ) : null}
                      </>
                    }
                    className="mb-4"
                  />
                  {visibleSections.length ? (
                    <div className="flex flex-col gap-8">
                      {ROUTER_SETTINGS_LAYERS.map((layer) => {
                        const layerSections = visibleSections.filter(
                          (section) => section.layer === layer.key,
                        )
                        if (layerSections.length === 0) return null
                        return (
                          <section
                            key={layer.key}
                            aria-label={layer.title}
                            className="flex flex-col gap-3"
                          >
                            <h3 className="px-1 text-[11px] font-medium uppercase tracking-[0.06em] text-subtle-foreground">
                              {layer.title}
                            </h3>
                            {layerSections.map((section) => (
                              <SectionCard
                                key={section.key}
                                section={section}
                                editor={editor}
                                changeCount={sectionEdits[section.id] ?? 0}
                                highlight={flash === section.id}
                              >
                                {(disabled) => renderSectionBody(section, disabled)}
                              </SectionCard>
                            ))}
                          </section>
                        )
                      })}
                      {showVsr ? (
                        <section aria-label="Agents" className="flex flex-col gap-3">
                          <h3 className="px-1 text-[11px] font-medium uppercase tracking-[0.06em] text-subtle-foreground">
                            Agents
                          </h3>
                          <ConfigPageListenerVSRPanel
                            config={
                              config ? { ...config, global: saved as ConfigData['global'] } : null
                            }
                            isReadonly={readOnly || dirty}
                            saveConfig={saveConfig}
                            notice={
                              dirty
                                ? 'Save or discard the router settings above before you change agent plugins.'
                                : undefined
                            }
                            highlight={flash === VSR_ID}
                          />
                        </section>
                      ) : null}
                    </div>
                  ) : (
                    <Card>
                      <EmptyState
                        compact
                        icon={SearchX}
                        tone="slate"
                        title="No settings match"
                        description={
                          show === 'edited'
                            ? 'Nothing is edited yet. Change a field and it shows up here.'
                            : show === 'set'
                              ? 'config.yaml sets nothing that matches. Every matching field uses its default.'
                              : 'Try another word, or search by config key such as drain_timeout.'
                        }
                        actions={
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => {
                              setSearch('')
                              setShow('all')
                            }}
                          >
                            Clear filters
                          </Button>
                        }
                      />
                    </Card>
                  )}
                </>
              ) : (
                <RouterSettingsYamlPanel
                  saved={saved}
                  draft={draft}
                  dirty={dirty}
                  readOnly={readOnly}
                  rawYaml={rawYaml}
                  rawLoading={rawState.loading}
                  rawError={rawState.error}
                  onReloadRaw={() => void loadRaw()}
                  onSaveRaw={saveRawYaml}
                  onEditForm={() => switchView('form')}
                />
              )}

              <AnimatePresence>
                {dirty && !readOnly ? (
                  <motion.div
                    initial={{ y: 18, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    exit={{ y: 18, opacity: 0 }}
                    transition={{ type: 'spring', bounce: 0.18, duration: 0.45 }}
                    className="sticky bottom-5 z-20 mt-6 flex items-center gap-3 rounded-xl border border-border bg-popover py-2 pl-4 pr-2 shadow-xl"
                    role="region"
                    aria-label="Unsaved changes"
                  >
                    <span className="size-2 shrink-0 rounded-full bg-primary shadow-[0_0_0_4px_var(--primary-soft)]" />
                    <span className="text-[13px] font-medium">
                      <span className="tabular">{changes.length}</span> unsaved{' '}
                      {changes.length === 1 ? 'change' : 'changes'}
                    </span>
                    {errors.size ? (
                      <Tooltip content={firstError ?? ''}>
                        <span className="truncate text-xs text-destructive-foreground">
                          Fix {errors.size === 1 ? 'one field' : `${errors.size} fields`} to save
                        </span>
                      </Tooltip>
                    ) : restartNotices.length ? (
                      <Badge tone="warning" size="sm">
                        <RotateCw />
                        Restart required
                      </Badge>
                    ) : (
                      <Badge tone="success" size="sm">
                        Hot reload
                      </Badge>
                    )}
                    <span
                      className="ml-auto hidden items-center gap-0.5 text-xs text-subtle-foreground sm:flex"
                      aria-hidden
                    >
                      <Kbd>⌘</Kbd>
                      <Kbd>S</Kbd>
                    </span>
                    <Button variant="ghost" size="sm" onClick={discard}>
                      Discard
                    </Button>
                    <Button size="sm" onClick={openReview} disabled={!canReview}>
                      Review &amp; save
                    </Button>
                  </motion.div>
                ) : null}
              </AnimatePresence>
            </div>

            {view === 'form' ? (
              <SettingsAside>
                <div className="-mr-2 max-h-[calc(100dvh-17rem)] overflow-y-auto pb-2 pr-2">
                  <SectionIndex
                    groups={indexGroups}
                    active={activeSection}
                    edits={sectionEdits}
                    onSelect={scrollTo}
                  />
                </div>
              </SettingsAside>
            ) : null}
          </div>
        </div>

        <RouterSettingsReviewDialog
          open={reviewOpen}
          onOpenChange={setReviewOpen}
          saved={saved}
          draft={draft}
          changeCount={changes.length}
          restart={restartNotices}
          saving={saving}
          error={saveError}
          onSave={() => void save()}
        />
      </RouterSettingsEditorContext.Provider>
    </ConfigPageManagerLayout>
  )
}
