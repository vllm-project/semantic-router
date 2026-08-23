import type { PropsWithChildren } from 'react'
import type { AccessAPIKey, AccessTeam, TeamMembership } from '../utils/inferenceAccessApi'
import { toLocalDateTime } from './AccessControlFormSupport'
import AccessBudgetRuleEditor from './AccessBudgetRuleEditor'
import AccessAsyncResourcePicker from './AccessAsyncResourcePicker'
import ProductIcon from '../components/ProductIcon'
import type { AccessEditor } from './AccessControlPageSupport'
import type { AccessControlSelectorSources } from './accessControlSelectorSources'
import styles from './AccessControlPage.module.css'

type Props = {
  editor: AccessEditor
  teams: AccessTeam[]
  keys: AccessAPIKey[]
  selectors: AccessControlSelectorSources
  selfService: boolean
  selfUserId: string
  onChange: (value: AccessEditor) => void
}

export default function AccessControlEditorFields({
  editor,
  teams,
  keys,
  selectors,
  selfService,
  selfUserId,
  onChange,
}: Props) {
  const update = (patch: Record<string, unknown>) =>
    onChange({ ...editor, value: { ...editor.value, ...patch } } as AccessEditor)
  return (
    <div className={styles.formGrid}>
      <Field label="Name" wide>
        <input
          value={editor.value.name || ''}
          onChange={(event) => update({ name: event.target.value })}
          placeholder={editor.kind === 'key' ? 'Production key' : 'Descriptive name'}
          data-dialog-initial-focus
          required
        />
      </Field>
      {editor.kind === 'user' ? (
        <UserFields editor={editor} selectors={selectors} update={update} />
      ) : null}
      {editor.kind === 'team' ? (
        <TeamFields
          editor={editor}
          selectors={selectors}
          selfService={selfService}
          update={update}
        />
      ) : null}
      {editor.kind === 'key' ? (
        <KeyFields
          editor={editor}
          teams={teams}
          selectors={selectors}
          keys={keys}
          selfService={selfService}
          selfUserId={selfUserId}
          onChange={onChange}
          update={update}
        />
      ) : null}
      {editor.kind === 'group' ? (
        <GroupFields editor={editor} selectors={selectors} update={update} />
      ) : null}
      {editor.kind === 'budget' ? <BudgetFields editor={editor} update={update} /> : null}
    </div>
  )
}

function UserFields({
  editor,
  selectors,
  update,
}: {
  editor: Extract<AccessEditor, { kind: 'user' }>
  selectors: AccessControlSelectorSources
  update: (patch: Record<string, unknown>) => void
}) {
  return (
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
      <PolicyFields
        accessGroupIds={editor.value.accessGroupIds || []}
        budgetId={editor.value.budgetId}
        selectors={selectors}
        onGroups={(accessGroupIds) => update({ accessGroupIds })}
        onBudget={(budgetId) => update({ budgetId })}
        label="User override"
      />
      <Advanced>
        <StatusField
          value={editor.value.status || 'active'}
          onChange={(status) => update({ status })}
        />
      </Advanced>
    </>
  )
}

function TeamFields({
  editor,
  selectors,
  selfService,
  update,
}: {
  editor: Extract<AccessEditor, { kind: 'team' }>
  selectors: AccessControlSelectorSources
  selfService: boolean
  update: (patch: Record<string, unknown>) => void
}) {
  const members = editor.value.members || []
  const setRole = (userId: string, role: TeamMembership['role']) =>
    update({
      members: members.map((member) => (member.userId === userId ? { ...member, role } : member)),
    })
  return (
    <>
      <Field label="Description" wide hint="Optional">
        <textarea
          value={editor.value.description || ''}
          onChange={(event) => update({ description: event.target.value })}
          placeholder="What this team is building"
          rows={2}
        />
      </Field>
      {!selfService ? (
        <PolicyFields
          accessGroupIds={editor.value.accessGroupIds || []}
          budgetId={editor.value.budgetId}
          selectors={selectors}
          onGroups={(accessGroupIds) => update({ accessGroupIds })}
          onBudget={(budgetId) => update({ budgetId })}
          required
          label="Team defaults"
        />
      ) : null}
      <SelectionSection title="Members" detail={`${members.length} selected · optional`}>
        <AccessAsyncResourcePicker
          ariaLabel="Search users"
          source={selectors.users}
          selectedIds={members.map((member) => member.userId)}
          multiple
          placeholder="Search by name or email"
          emptyText="No users found"
          onChange={(selectedIds) => {
            const existing = new Map(members.map((member) => [member.userId, member]))
            update({
              members: selectedIds.map(
                (userId, index): TeamMembership =>
                  existing.get(userId) || {
                    userId,
                    teamId: editor.value.id || '',
                    role: members.length === 0 && index === 0 ? 'admin' : 'member',
                  },
              ),
            })
          }}
          renderSelectedDetail={(user) => {
            const member = members.find((item) => item.userId === user.id)
            if (!member) return null
            return (
              <label className={styles.asyncPickerMemberRole}>
                <span>
                  <strong>{user.name}</strong>
                  <small>{user.email}</small>
                </span>
                <select
                  aria-label={`${user.name} Team role`}
                  value={member.role}
                  onChange={(event) =>
                    setRole(user.id, event.target.value as TeamMembership['role'])
                  }
                >
                  <option value="member">Member</option>
                  <option value="admin">Admin</option>
                </select>
              </label>
            )
          }}
        />
      </SelectionSection>
      {!selfService ? (
        <Advanced>
          <StatusField
            value={editor.value.status || 'active'}
            onChange={(status) => update({ status })}
          />
        </Advanced>
      ) : null}
    </>
  )
}

function KeyFields({
  editor,
  teams,
  selectors,
  keys,
  selfService,
  selfUserId,
  onChange,
  update,
}: {
  editor: Extract<AccessEditor, { kind: 'key' }>
  teams: AccessTeam[]
  selectors: AccessControlSelectorSources
  keys: AccessAPIKey[]
  selfService: boolean
  selfUserId: string
  onChange: (value: AccessEditor) => void
  update: (patch: Record<string, unknown>) => void
}) {
  const ownerId = editor.value.ownerId || ''
  const manageableTeams = selfService
    ? teams.filter((team) =>
        team.members.some(
          (membership) => membership.userId === selfUserId && membership.role === 'admin',
        ),
      )
    : teams
  const personalKeyExists = keys.some(
    (key) => key.ownerType === 'user' && key.ownerId === selfUserId && key.id !== editor.value.id,
  )
  const inheritBudgetLabel = 'Inherit the owner’s effective quota'
  const chooseRateLimitMode = (mode: 'inherit' | 'budget' | 'custom') => {
    const inlineRateLimit = editor.inlineRateLimit ?? {
      name: `${editor.value.name?.trim() || 'API key'} quota`,
      description: '',
      rules: [
        {
          metric: 'requests' as const,
          algorithm: 'sliding_log' as const,
          limit: '60',
          window: 'PT1M',
          accounting: 'request' as const,
          enforcement: 'enforce' as const,
        },
      ],
    }
    onChange({
      ...editor,
      rateLimitMode: mode,
      inlineRateLimit,
      value: {
        ...editor.value,
        budgetId: mode === 'budget' ? editor.value.budgetId : undefined,
      },
    })
  }
  return (
    <>
      {selfService && manageableTeams.length ? (
        <fieldset className={styles.ownerSection}>
          <legend>
            Owned by <small>Required · choose one</small>
          </legend>
          <div className={styles.ownerChoices} role="radiogroup" aria-label="Owned by">
            <OwnerChoice
              active={editor.ownerType === 'user'}
              disabled={personalKeyExists}
              title="Personal"
              detail={personalKeyExists ? 'One personal key already exists' : 'Only you can use it'}
              onSelect={() =>
                onChange({
                  ...editor,
                  ownerType: 'user',
                  value: {
                    ...editor.value,
                    ownerType: 'user',
                    ownerId: selfUserId,
                    contextTeamId: undefined,
                  },
                })
              }
            />
            <OwnerChoice
              active={editor.ownerType === 'team'}
              title="Team"
              detail="Shared with a Team you administer"
              onSelect={() =>
                onChange({
                  ...editor,
                  ownerType: 'team',
                  value: {
                    ...editor.value,
                    ownerType: 'team',
                    ownerId: manageableTeams[0]?.id || '',
                    contextTeamId: manageableTeams[0]?.id,
                  },
                })
              }
            />
          </div>
        </fieldset>
      ) : null}
      {!selfService ? (
        <>
          <fieldset className={styles.ownerSection}>
            <legend>
              Owned by <small>Required · choose one</small>
            </legend>
            <div className={styles.ownerChoices} role="radiogroup" aria-label="Owned by">
              <OwnerChoice
                active={editor.ownerType === 'user'}
                title="Personal"
                detail="A key for one user"
                onSelect={() =>
                  onChange({
                    ...editor,
                    ownerType: 'user',
                    value: {
                      ...editor.value,
                      ownerType: 'user',
                      ownerId: '',
                      contextTeamId: undefined,
                    },
                  })
                }
              />
              <OwnerChoice
                active={editor.ownerType === 'team'}
                title="Team"
                detail="A shared service key"
                onSelect={() =>
                  onChange({
                    ...editor,
                    ownerType: 'team',
                    value: {
                      ...editor.value,
                      ownerType: 'team',
                      ownerId: '',
                      contextTeamId: undefined,
                    },
                  })
                }
              />
            </div>
          </fieldset>
          <PickerField label={editor.ownerType === 'user' ? 'User' : 'Team'}>
            {editor.ownerType === 'user' ? (
              <AccessAsyncResourcePicker
                key="key-owner-user"
                ariaLabel="Search users"
                source={selectors.users}
                selectedIds={ownerId ? [ownerId] : []}
                placeholder="Search name or email"
                emptyText="No users found"
                onChange={(selectedIds) =>
                  update({ ownerType: 'user', ownerId: selectedIds[0] || '' })
                }
              />
            ) : (
              <AccessAsyncResourcePicker
                key="key-owner-team"
                ariaLabel="Search Teams"
                source={selectors.teams}
                selectedIds={ownerId ? [ownerId] : []}
                placeholder="Search Team name"
                emptyText="No Teams found"
                onChange={(selectedIds) =>
                  update({
                    ownerType: 'team',
                    ownerId: selectedIds[0] || '',
                    contextTeamId: selectedIds[0] || undefined,
                  })
                }
              />
            )}
          </PickerField>
        </>
      ) : editor.ownerType === 'team' ? (
        <Field label="Team" wide>
          <select
            value={ownerId}
            onChange={(event) =>
              update({
                ownerType: 'team',
                ownerId: event.target.value,
                contextTeamId: event.target.value,
              })
            }
            required
          >
            {manageableTeams.map((team) => (
              <option value={team.id} key={team.id}>
                {team.name}
              </option>
            ))}
          </select>
        </Field>
      ) : (
        <div className={`${styles.formField} ${styles.formFieldWide}`}>
          <span>Personal key</span>
          <small>Your administrator controls model access and quota.</small>
        </div>
      )}
      {editor.ownerType === 'user' && !selfService ? (
        <PickerField
          label="Team context"
          hint="Optional. The key inherits this Team when the user has no override."
        >
          <AccessAsyncResourcePicker
            ariaLabel="Search Team context"
            source={selectors.teams}
            selectedIds={editor.value.contextTeamId ? [editor.value.contextTeamId] : []}
            optional
            optionalTitle="Personal policy only"
            optionalDescription="Do not inherit a Team policy"
            placeholder="Search Team name"
            emptyText="No Teams found"
            onChange={(selectedIds) => update({ contextTeamId: selectedIds[0] || undefined })}
          />
        </PickerField>
      ) : null}
      {!selfService ? (
        <>
          {editor.value.id ? (
            <PolicyFields
              accessGroupIds={editor.value.accessGroupIds || []}
              budgetId={editor.value.budgetId}
              selectors={selectors}
              onGroups={(accessGroupIds) => update({ accessGroupIds })}
              onBudget={(budgetId) => update({ budgetId })}
              label="Key limit"
              inheritBudgetLabel={inheritBudgetLabel}
              showModels={false}
            />
          ) : (
            <>
              <fieldset className={styles.ownerSection}>
                <legend>
                  Quota <small>Optional · choose one</small>
                </legend>
                <div
                  className={`${styles.ownerChoices} ${styles.quotaChoices}`}
                  role="radiogroup"
                  aria-label="Quota"
                >
                  <OwnerChoice
                    active={editor.rateLimitMode === 'inherit'}
                    title="Inherit"
                    detail={inheritBudgetLabel}
                    onSelect={() => chooseRateLimitMode('inherit')}
                  />
                  <OwnerChoice
                    active={editor.rateLimitMode === 'budget'}
                    title="Budget"
                    detail="Use an existing budget"
                    onSelect={() => chooseRateLimitMode('budget')}
                  />
                  <OwnerChoice
                    active={editor.rateLimitMode === 'custom'}
                    title="Custom"
                    detail="Set limits for this key"
                    onSelect={() => chooseRateLimitMode('custom')}
                  />
                </div>
              </fieldset>
              {editor.rateLimitMode === 'budget' ? (
                <PickerField label="Budget">
                  <AccessAsyncResourcePicker
                    ariaLabel="Search budgets"
                    source={selectors.budgets}
                    selectedIds={editor.value.budgetId ? [editor.value.budgetId] : []}
                    placeholder="Search budget name"
                    emptyText="No budgets found"
                    onChange={(selectedIds) => update({ budgetId: selectedIds[0] || undefined })}
                  />
                </PickerField>
              ) : null}
              {editor.rateLimitMode === 'custom' && editor.inlineRateLimit ? (
                <div className={styles.inlineQuotaEditor}>
                  <Field label="Quota name" wide>
                    <input
                      value={editor.inlineRateLimit.name}
                      onChange={(event) =>
                        onChange({
                          ...editor,
                          inlineRateLimit: {
                            ...editor.inlineRateLimit!,
                            name: event.target.value,
                          },
                        })
                      }
                      placeholder="Production key quota"
                      required
                    />
                  </Field>
                  <AccessBudgetRuleEditor
                    rules={editor.inlineRateLimit.rules}
                    onChange={(rules) =>
                      onChange({
                        ...editor,
                        inlineRateLimit: { ...editor.inlineRateLimit!, rules },
                      })
                    }
                  />
                </div>
              ) : null}
            </>
          )}
          <Advanced label="Advanced settings">
            <PolicyFields
              accessGroupIds={editor.value.accessGroupIds || []}
              budgetId={editor.value.budgetId}
              selectors={selectors}
              onGroups={(accessGroupIds) => update({ accessGroupIds })}
              onBudget={(budgetId) => update({ budgetId })}
              label="Key override"
              showBudget={false}
            />
            <Field label="Expiration" wide hint="Optional">
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
            <StatusField
              value={editor.value.status || 'active'}
              onChange={(status) => update({ status })}
            />
          </Advanced>
        </>
      ) : null}
    </>
  )
}

function GroupFields({
  editor,
  selectors,
  update,
}: {
  editor: Extract<AccessEditor, { kind: 'group' }>
  selectors: AccessControlSelectorSources
  update: (patch: Record<string, unknown>) => void
}) {
  const selected = editor.value.resources || []
  const selectedIds = (resourceType: 'entrypoint' | 'model') =>
    selected
      .filter((resource) => resource.resourceType === resourceType)
      .map((resource) => resource.resourceId)
  const setResources = (resourceType: 'entrypoint' | 'model', resourceIds: string[]) => {
    update({
      resources: [
        ...selected.filter((resource) => resource.resourceType !== resourceType),
        ...resourceIds.map((resourceId) => ({ resourceType, resourceId })),
      ],
    })
  }
  return (
    <>
      <SelectionSection title="Mixture-of-Models" detail="Published API models">
        <AccessAsyncResourcePicker
          ariaLabel="Search Mixture-of-Models"
          selectedIds={selectedIds('entrypoint')}
          source={selectors.entrypoints}
          onChange={(resourceIds) => setResources('entrypoint', resourceIds)}
          multiple
          placeholder="Search Mixture-of-Models"
          emptyText="No Mixture-of-Models found"
        />
      </SelectionSection>
      <Advanced label="Single model access">
        <SelectionSection title="Single models" detail="Direct model access">
          <AccessAsyncResourcePicker
            ariaLabel="Search single models"
            selectedIds={selectedIds('model')}
            source={selectors.models}
            onChange={(resourceIds) => setResources('model', resourceIds)}
            multiple
            placeholder="Search single models"
            emptyText="No models found"
          />
        </SelectionSection>
      </Advanced>
      <Field label="Description" wide hint="Optional">
        <textarea
          value={editor.value.description || ''}
          onChange={(event) => update({ description: event.target.value })}
          placeholder="Who this model collection is for"
          rows={2}
        />
      </Field>
    </>
  )
}

function BudgetFields({
  editor,
  update,
}: {
  editor: Extract<AccessEditor, { kind: 'budget' }>
  update: (patch: Record<string, unknown>) => void
}) {
  return (
    <>
      <Field label="Description" wide hint="Optional">
        <textarea
          value={editor.value.description || ''}
          onChange={(event) => update({ description: event.target.value })}
          placeholder="When this quota should be used"
          rows={2}
        />
      </Field>
      <AccessBudgetRuleEditor
        rules={editor.value.rules || []}
        onChange={(rules) => update({ rules })}
      />
      <label className={styles.toggleField}>
        <input
          type="checkbox"
          checked={editor.value.enabled !== false}
          onChange={(event) => update({ enabled: event.target.checked })}
        />
        <span>
          <i />
          <strong>Active</strong>
          <small>Available for new assignments.</small>
        </span>
      </label>
    </>
  )
}

function PolicyFields({
  accessGroupIds,
  budgetId,
  selectors,
  onGroups,
  onBudget,
  required = false,
  label,
  inheritBudgetLabel,
  showModels = true,
  showBudget = true,
}: {
  accessGroupIds: string[]
  budgetId?: string
  selectors: AccessControlSelectorSources
  onGroups: (ids: string[]) => void
  onBudget: (id?: string) => void
  required?: boolean
  label: string
  inheritBudgetLabel?: string
  showModels?: boolean
  showBudget?: boolean
}) {
  return (
    <>
      {showModels ? (
        <SelectionSection
          title="Model access"
          detail={`${label} · ${required ? 'required' : 'optional'}`}
        >
          <AccessAsyncResourcePicker
            ariaLabel="Search access groups"
            source={selectors.groups}
            selectedIds={accessGroupIds}
            multiple
            placeholder="Search access group name"
            emptyText="No access groups found"
            onChange={onGroups}
          />
        </SelectionSection>
      ) : null}
      {showBudget ? (
        <PickerField label="Quota" hint={`${label} · ${required ? 'required' : 'optional'}`}>
          <AccessAsyncResourcePicker
            ariaLabel="Search budgets"
            source={selectors.budgets}
            selectedIds={budgetId ? [budgetId] : []}
            optional={!required}
            optionalTitle={inheritBudgetLabel || 'Inherit'}
            optionalDescription="Use the next policy in the ownership chain"
            placeholder="Search budget name"
            emptyText="No budgets found"
            onChange={(selectedIds) => onBudget(selectedIds[0])}
          />
        </PickerField>
      ) : null}
    </>
  )
}

function Advanced({ children, label = 'Advanced' }: PropsWithChildren<{ label?: string }>) {
  return (
    <details className={styles.advancedSection}>
      <summary>
        <span>{label}</span>
        <small>Optional settings</small>
      </summary>
      <div className={styles.advancedGrid}>{children}</div>
    </details>
  )
}
function Field({
  label,
  hint,
  wide = false,
  children,
}: PropsWithChildren<{ label: string; hint?: string; wide?: boolean }>) {
  return (
    <label className={`${styles.formField} ${wide ? styles.formFieldWide : ''}`}>
      <span>{label}</span>
      {children}
      {hint ? <small>{hint}</small> : null}
    </label>
  )
}
function PickerField({
  label,
  hint,
  children,
}: PropsWithChildren<{ label: string; hint?: string }>) {
  return (
    <div className={`${styles.formField} ${styles.formFieldWide}`}>
      <span>{label}</span>
      {children}
      {hint ? <small>{hint}</small> : null}
    </div>
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
function SelectionSection({
  title,
  detail,
  children,
}: PropsWithChildren<{ title: string; detail: string }>) {
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
function OwnerChoice({
  active,
  disabled = false,
  title,
  detail,
  onSelect,
}: {
  active: boolean
  disabled?: boolean
  title: string
  detail: string
  onSelect: () => void
}) {
  return (
    <button
      type="button"
      className={`${styles.ownerChoice} ${active ? styles.ownerChoiceActive : ''}`}
      role="radio"
      aria-checked={active}
      disabled={disabled}
      onClick={onSelect}
    >
      <span>{title}</span>
      <small>{detail}</small>
      {active ? (
        <ProductIcon className={styles.choiceCheck} name="check" aria-hidden="true" />
      ) : null}
    </button>
  )
}
