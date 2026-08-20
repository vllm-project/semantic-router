import React, { useState } from 'react'
import useAccessibleDialog from '../hooks/useAccessibleDialog'
import type {
  AccessAPIKey,
  AccessBinding,
  AccessBudget,
  AccessGroup,
  AccessTeam,
  AccessUser,
  CreatedAccessAPIKey,
} from '../utils/inferenceAccessApi'
import type { AccessEditor } from './AccessControlPageSupport'
import styles from './AccessControlPage.module.css'

type EditorProps = {
  editor: AccessEditor
  users: AccessUser[]
  teams: AccessTeam[]
  keys: AccessAPIKey[]
  groups: AccessGroup[]
  budgets: AccessBudget[]
  error: string
  saving: boolean
  onChange: (value: AccessEditor) => void
  onClose: () => void
  onSave: () => void
  selfService?: boolean
  secret?: never
}

type SecretProps = {
  secret: CreatedAccessAPIKey
  onClose: () => void
  editor?: never
  users?: never
  teams?: never
  keys?: never
  groups?: never
  budgets?: never
  error?: never
  saving?: never
  onChange?: never
  onSave?: never
  selfService?: never
}

type Props = EditorProps | SecretProps

const TITLES: Record<
  AccessEditor['kind'],
  { eyebrow: string; create: string; edit: string; description: string }
> = {
  user: {
    eyebrow: 'Identity',
    create: 'Add user',
    edit: 'Edit user',
    description: 'A user can own API keys, join teams, and optionally receive Dashboard access.',
  },
  team: {
    eyebrow: 'Identity',
    create: 'Create team',
    edit: 'Edit team',
    description: 'Group users under shared model grants and quota.',
  },
  key: {
    eyebrow: 'Credential',
    create: 'Create API key',
    edit: 'API key',
    description: 'Choose an owner, model visibility, and an optional key-specific limit.',
  },
  group: {
    eyebrow: 'Model policy',
    create: 'Create access group',
    edit: 'Edit access group',
    description: 'Compose reusable model grants and assign them to identities or keys.',
  },
  budget: {
    eyebrow: 'Rate limit',
    create: 'Create budget',
    edit: 'Edit budget',
    description: 'Enforce RPM, TPM, and daily tokens at any scope.',
  },
}

export default function AccessControlDialog(props: Props) {
  const [copied, setCopied] = useState(false)
  const dialogRef = useAccessibleDialog<HTMLDivElement>({
    isOpen: true,
    onClose: props.onClose,
    dismissible: !('saving' in props && props.saving),
  })

  if (props.secret) {
    return (
      <div
        className={styles.modalBackdrop}
        onMouseDown={(event) => {
          if (event.target === event.currentTarget) props.onClose()
        }}
      >
        <section
          ref={dialogRef}
          className={`${styles.modal} ${styles.secretModal}`}
          role="dialog"
          aria-modal="true"
          aria-labelledby="api-secret-title"
          tabIndex={-1}
        >
          <button
            type="button"
            className={styles.modalClose}
            onClick={props.onClose}
            aria-label="Close"
          >
            ×
          </button>
          <div className={styles.secretIcon}>✓</div>
          <span className={styles.modalEyebrow}>API key created</span>
          <h2 id="api-secret-title">Save it somewhere safe.</h2>
          <p>Copy it now. You can securely reveal or rotate it later from the key details.</p>
          <div className={styles.secretValue}>
            <code>{props.secret.secret}</code>
            <button
              type="button"
              onClick={() => {
                void navigator.clipboard.writeText(props.secret.secret)
                setCopied(true)
              }}
            >
              {copied ? 'Copied' : 'Copy'}
            </button>
          </div>
          <div className={styles.secretMeta}>
            <span>
              <small>Key</small>
              {props.secret.name}
            </span>
            <span>
              <small>Prefix</small>
              {props.secret.prefix}
            </span>
          </div>
          <button type="button" className={styles.primaryButton} onClick={props.onClose}>
            I saved the key
          </button>
        </section>
      </div>
    )
  }

  const {
    editor,
    users,
    teams,
    keys,
    groups,
    budgets,
    error,
    saving,
    onChange,
    onClose,
    onSave,
    selfService = false,
  } = props
  const meta = TITLES[editor.kind]
  const update = (patch: Record<string, unknown>) =>
    onChange({ ...editor, value: { ...editor.value, ...patch } } as AccessEditor)
  const subjectOptions =
    editor.kind === 'budget'
      ? editor.value.scopeType === 'user'
        ? users
        : editor.value.scopeType === 'team'
          ? teams
          : editor.value.scopeType === 'key'
            ? keys
            : []
      : []
  return (
    <div
      className={styles.modalBackdrop}
      onMouseDown={(event) => {
        if (event.target === event.currentTarget && !saving) onClose()
      }}
    >
      <section
        ref={dialogRef}
        className={styles.modal}
        role="dialog"
        aria-modal="true"
        aria-labelledby="access-dialog-title"
        tabIndex={-1}
      >
        <header className={styles.modalHeader}>
          <div>
            <span className={styles.modalEyebrow}>{meta.eyebrow}</span>
            <h2 id="access-dialog-title">{editor.value.id ? meta.edit : meta.create}</h2>
            <p>{meta.description}</p>
          </div>
          <button
            type="button"
            className={styles.modalClose}
            onClick={onClose}
            disabled={saving}
            aria-label="Close"
          >
            ×
          </button>
        </header>
        <div className={styles.modalBody}>
          {error ? (
            <div className={styles.modalError} role="alert">
              <span>!</span>
              <div>
                <strong>Couldn’t save</strong>
                <p>{error}</p>
              </div>
            </div>
          ) : null}
          <div className={styles.formGrid}>
            <Field label="Name" wide>
              <input
                value={editor.value.name || ''}
                onChange={(event) => update({ name: event.target.value })}
                placeholder={editor.kind === 'key' ? 'Production gateway' : 'Descriptive name'}
                data-dialog-initial-focus
                required
              />
            </Field>
            {editor.kind === 'user' ? (
              <>
                <Field label="Email" wide>
                  <input
                    type="email"
                    value={editor.value.email || ''}
                    onChange={(event) => update({ email: event.target.value })}
                    placeholder="name@company.com"
                    required
                  />
                </Field>
                <StatusField
                  value={editor.value.status || 'active'}
                  onChange={(status) => update({ status })}
                />
              </>
            ) : null}
            {editor.kind === 'team' ? (
              <>
                <Field label="Description" wide>
                  <textarea
                    value={editor.value.description || ''}
                    onChange={(event) => update({ description: event.target.value })}
                    placeholder="What this team owns"
                    rows={3}
                  />
                </Field>
                <StatusField
                  value={editor.value.status || 'active'}
                  onChange={(status) => update({ status })}
                />
                <SelectionSection
                  title="Members"
                  detail={`${editor.value.userIds?.length || 0} selected`}
                >
                  {users.length ? (
                    users.map((user) => (
                      <CheckCard
                        key={user.id}
                        checked={(editor.value.userIds || []).includes(user.id)}
                        title={user.name}
                        detail={user.email}
                        onChange={(checked) =>
                          update({
                            userIds: checked
                              ? [...(editor.value.userIds || []), user.id]
                              : (editor.value.userIds || []).filter((id) => id !== user.id),
                          })
                        }
                      />
                    ))
                  ) : (
                    <EmptyChoice text="Add a user first." />
                  )}
                </SelectionSection>
              </>
            ) : null}
            {editor.kind === 'key' ? (
              <>
                {selfService ? (
                  <div className={`${styles.formField} ${styles.formFieldWide}`}>
                    <span>Your credential</span>
                    <small>Model grants and quota are assigned by your administrator.</small>
                  </div>
                ) : (
                  <>
                    <Field label="Owner type">
                      <select
                        value={editor.ownerType}
                        onChange={(event) =>
                          onChange({
                            ...editor,
                            ownerType: event.target.value as 'user' | 'team',
                            value: { ...editor.value, userId: undefined, teamId: undefined },
                          })
                        }
                      >
                        <option value="user">User</option>
                        <option value="team">Team</option>
                      </select>
                    </Field>
                    <Field label="Owner">
                      <select
                        value={
                          (editor.ownerType === 'user'
                            ? editor.value.userId
                            : editor.value.teamId) || ''
                        }
                        onChange={(event) =>
                          update(
                            editor.ownerType === 'user'
                              ? { userId: event.target.value }
                              : { teamId: event.target.value },
                          )
                        }
                        required
                      >
                        <option value="">Select owner</option>
                        {(editor.ownerType === 'user' ? users : teams).map((owner) => (
                          <option value={owner.id} key={owner.id}>
                            {owner.name}
                          </option>
                        ))}
                      </select>
                    </Field>
                  </>
                )}
                <Field label="Expiration" wide hint="Leave blank for a non-expiring key.">
                  <input
                    type="datetime-local"
                    value={editor.value.expiresAt ? toLocalDateTime(editor.value.expiresAt) : ''}
                    onChange={(event) =>
                      update({
                        expiresAt: event.target.value
                          ? new Date(event.target.value).toISOString()
                          : undefined,
                      })
                    }
                  />
                </Field>
                {!selfService ? (
                  <>
                    <SelectionSection
                      title="Model visibility"
                      detail={
                        (editor.value.accessGroupIds || []).length
                          ? 'Key-specific override'
                          : 'Inherits from owner'
                      }
                    >
                      {groups.length ? (
                        groups.map((group) => (
                          <CheckCard
                            key={group.id}
                            checked={(editor.value.accessGroupIds || []).includes(group.id)}
                            title={group.name}
                            detail={group.modelPatterns.join(', ')}
                            onChange={(checked) =>
                              update({
                                accessGroupIds: checked
                                  ? [...(editor.value.accessGroupIds || []), group.id]
                                  : (editor.value.accessGroupIds || []).filter(
                                      (id) => id !== group.id,
                                    ),
                              })
                            }
                          />
                        ))
                      ) : (
                        <EmptyChoice text="Create an access group first." />
                      )}
                    </SelectionSection>
                    <Field
                      label="Budget"
                      wide
                      hint="Optional. Shared usage is counted against the selected budget."
                    >
                      <select
                        value={editor.value.budgetId || ''}
                        onChange={(event) => update({ budgetId: event.target.value || undefined })}
                      >
                        <option value="">No linked budget</option>
                        {budgets
                          .filter((budget) => budget.enabled)
                          .map((budget) => (
                            <option value={budget.id} key={budget.id}>
                              {budget.name} · {budget.rpm || '∞'} RPM · {budget.tpm || '∞'} TPM
                            </option>
                          ))}
                      </select>
                    </Field>
                    <SelectionSection
                      title="Key limits"
                      detail="Optional · enforced with the linked budget and inherited limits"
                    >
                      <div className={styles.quotaInputs}>
                        <NumberField
                          label="RPM"
                          value={editor.value.budget?.rpm || 0}
                          onChange={(rpm) =>
                            update({
                              budget: {
                                ...(editor.value.budget || { tpm: 0, dailyTokens: 0 }),
                                rpm,
                              },
                            })
                          }
                        />
                        <NumberField
                          label="TPM"
                          value={editor.value.budget?.tpm || 0}
                          onChange={(tpm) =>
                            update({
                              budget: {
                                ...(editor.value.budget || { rpm: 0, dailyTokens: 0 }),
                                tpm,
                              },
                            })
                          }
                        />
                        <NumberField
                          label="Daily tokens"
                          value={editor.value.budget?.dailyTokens || 0}
                          onChange={(dailyTokens) =>
                            update({
                              budget: {
                                ...(editor.value.budget || { rpm: 0, tpm: 0 }),
                                dailyTokens,
                              },
                            })
                          }
                        />
                      </div>
                    </SelectionSection>
                  </>
                ) : null}
              </>
            ) : null}
            {editor.kind === 'group' ? (
              <>
                <Field label="Description" wide>
                  <textarea
                    value={editor.value.description || ''}
                    onChange={(event) => update({ description: event.target.value })}
                    placeholder="Who should use these models"
                    rows={3}
                  />
                </Field>
                <Field
                  label="Model IDs or patterns"
                  wide
                  hint="One exact model ID or trailing wildcard per line."
                >
                  <textarea
                    value={(editor.value.modelPatterns || []).join('\n')}
                    onChange={(event) =>
                      update({
                        modelPatterns: event.target.value
                          .split(/\n|,/)
                          .map((value) => value.trim())
                          .filter(Boolean),
                      })
                    }
                    placeholder={'vllm-sr/mom-v1-lite\nvllm-sr/mom-*'}
                    rows={5}
                  />
                </Field>
                <BindingChoices
                  value={editor.value.bindings || []}
                  users={users}
                  teams={teams}
                  keys={keys}
                  onChange={(bindings) => update({ bindings })}
                />
              </>
            ) : null}
            {editor.kind === 'budget' ? (
              <>
                <Field label="Scope">
                  <select
                    value={editor.value.scopeType || 'global'}
                    onChange={(event) => update({ scopeType: event.target.value, scopeId: '' })}
                  >
                    <option value="global">Global</option>
                    <option value="user">User</option>
                    <option value="team">Team</option>
                    <option value="key">API key</option>
                  </select>
                </Field>
                {editor.value.scopeType !== 'global' ? (
                  <Field label="Target">
                    <select
                      value={editor.value.scopeId || ''}
                      onChange={(event) => update({ scopeId: event.target.value })}
                      required
                    >
                      <option value="">Select target</option>
                      {subjectOptions.map((item) => (
                        <option value={item.id} key={item.id}>
                          {'prefix' in item ? `${item.name} · ${item.prefix}` : item.name}
                        </option>
                      ))}
                    </select>
                  </Field>
                ) : (
                  <div />
                )}
                <NumberField
                  label="RPM"
                  value={editor.value.rpm || 0}
                  onChange={(rpm) => update({ rpm })}
                />
                <NumberField
                  label="TPM"
                  value={editor.value.tpm || 0}
                  onChange={(tpm) => update({ tpm })}
                />
                <NumberField
                  label="Daily tokens"
                  value={editor.value.dailyTokens || 0}
                  onChange={(dailyTokens) => update({ dailyTokens })}
                />
                <label className={styles.toggleField}>
                  <input
                    type="checkbox"
                    checked={editor.value.enabled !== false}
                    onChange={(event) => update({ enabled: event.target.checked })}
                  />
                  <span>
                    <i />
                    <strong>Enforce this budget</strong>
                    <small>Disabled budgets remain saved but do not reserve quota.</small>
                  </span>
                </label>
              </>
            ) : null}
          </div>
        </div>
        <footer className={styles.modalFooter}>
          <button
            type="button"
            className={styles.secondaryButton}
            onClick={onClose}
            disabled={saving}
          >
            Cancel
          </button>
          <button type="button" className={styles.primaryButton} onClick={onSave} disabled={saving}>
            {saving ? 'Saving…' : editor.value.id ? 'Save changes' : 'Create'}
          </button>
        </footer>
      </section>
    </div>
  )
}

function Field({
  label,
  hint,
  wide = false,
  children,
}: React.PropsWithChildren<{ label: string; hint?: string; wide?: boolean }>) {
  return (
    <label className={`${styles.formField} ${wide ? styles.formFieldWide : ''}`}>
      <span>{label}</span>
      {children}
      {hint ? <small>{hint}</small> : null}
    </label>
  )
}
function StatusField({
  value,
  onChange,
}: {
  value: string
  onChange: (value: 'active' | 'disabled') => void
}) {
  return (
    <Field label="Status">
      <select
        value={value}
        onChange={(event) => onChange(event.target.value as 'active' | 'disabled')}
      >
        <option value="active">Active</option>
        <option value="disabled">Disabled</option>
      </select>
    </Field>
  )
}
function NumberField({
  label,
  value,
  onChange,
}: {
  label: string
  value: number
  onChange: (value: number) => void
}) {
  return (
    <Field label={label}>
      <input
        type="number"
        min="0"
        step="1"
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </Field>
  )
}
function SelectionSection({
  title,
  detail,
  children,
}: React.PropsWithChildren<{ title: string; detail: string }>) {
  return (
    <fieldset className={styles.selectionSection}>
      <legend>
        <span>{title}</span>
        <small>{detail}</small>
      </legend>
      <div className={styles.choiceGrid}>{children}</div>
    </fieldset>
  )
}
function CheckCard({
  checked,
  title,
  detail,
  onChange,
}: {
  checked: boolean
  title: string
  detail: string
  onChange: (value: boolean) => void
}) {
  return (
    <label className={`${styles.checkCard} ${checked ? styles.checkCardSelected : ''}`}>
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
      />
      <span>
        <strong>{title}</strong>
        <small>{detail}</small>
      </span>
      <i>✓</i>
    </label>
  )
}
function EmptyChoice({ text }: { text: string }) {
  return <p className={styles.emptyChoice}>{text}</p>
}

function BindingChoices({
  value,
  users,
  teams,
  keys,
  onChange,
}: {
  value: AccessBinding[]
  users: AccessUser[]
  teams: AccessTeam[]
  keys: AccessAPIKey[]
  onChange: (value: AccessBinding[]) => void
}) {
  const toggle = (binding: AccessBinding, selected: boolean) =>
    onChange(
      selected
        ? [...value, binding]
        : value.filter(
            (item) =>
              item.subjectType !== binding.subjectType || item.subjectId !== binding.subjectId,
          ),
    )
  return (
    <SelectionSection title="Assignments" detail={`${value.length} selected`}>
      {(
        [
          ['user', users],
          ['team', teams],
          ['key', keys],
        ] as const
      ).flatMap(([subjectType, items]) =>
        items.map((item) => {
          const binding: AccessBinding = { subjectType, subjectId: item.id }
          return (
            <CheckCard
              key={`${subjectType}:${item.id}`}
              checked={value.some(
                (current) => current.subjectType === subjectType && current.subjectId === item.id,
              )}
              title={item.name}
              detail={subjectType === 'key' ? (item as AccessAPIKey).prefix : subjectType}
              onChange={(selected) => toggle(binding, selected)}
            />
          )
        }),
      )}
    </SelectionSection>
  )
}

function toLocalDateTime(value: string) {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ''
  const local = new Date(date.getTime() - date.getTimezoneOffset() * 60_000)
  return local.toISOString().slice(0, 16)
}
