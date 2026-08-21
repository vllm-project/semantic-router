import type { PropsWithChildren } from 'react'
import type {
  AccessBudget,
  AccessAPIKey,
  AccessGroup,
  AccessTeam,
  AccessUser,
  TeamMembership,
} from '../utils/inferenceAccessApi'
import { toLocalDateTime } from './AccessControlFormSupport'
import type { AccessEditor } from './AccessControlPageSupport'
import styles from './AccessControlPage.module.css'

type Props = {
  editor: AccessEditor
  users: AccessUser[]
  teams: AccessTeam[]
  groups: AccessGroup[]
  budgets: AccessBudget[]
  keys: AccessAPIKey[]
  selfService: boolean
  selfUserId: string
  onChange: (value: AccessEditor) => void
}

export default function AccessControlEditorFields({
  editor,
  users,
  teams,
  groups,
  budgets,
  keys,
  selfService,
  selfUserId,
  onChange,
}: Props) {
  const update = (patch: Record<string, unknown>) =>
    onChange({ ...editor, value: { ...editor.value, ...patch } } as AccessEditor)
  const selectedBudgetId = 'budgetId' in editor.value ? editor.value.budgetId : undefined
  const activeBudgets = budgets.filter((budget) => budget.enabled || budget.id === selectedBudgetId)
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
        <UserFields editor={editor} groups={groups} budgets={activeBudgets} update={update} />
      ) : null}
      {editor.kind === 'team' ? (
        <TeamFields
          editor={editor}
          users={users}
          groups={groups}
          budgets={activeBudgets}
          selfService={selfService}
          update={update}
        />
      ) : null}
      {editor.kind === 'key' ? (
        <KeyFields
          editor={editor}
          users={users}
          teams={teams}
          groups={groups}
          budgets={activeBudgets}
          keys={keys}
          selfService={selfService}
          selfUserId={selfUserId}
          onChange={onChange}
          update={update}
        />
      ) : null}
      {editor.kind === 'group' ? <GroupFields editor={editor} update={update} /> : null}
      {editor.kind === 'budget' ? <BudgetFields editor={editor} update={update} /> : null}
    </div>
  )
}

function UserFields({
  editor,
  groups,
  budgets,
  update,
}: {
  editor: Extract<AccessEditor, { kind: 'user' }>
  groups: AccessGroup[]
  budgets: AccessBudget[]
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
        groups={groups}
        budgets={budgets}
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
  users,
  groups,
  budgets,
  selfService,
  update,
}: {
  editor: Extract<AccessEditor, { kind: 'team' }>
  users: AccessUser[]
  groups: AccessGroup[]
  budgets: AccessBudget[]
  selfService: boolean
  update: (patch: Record<string, unknown>) => void
}) {
  const members = editor.value.members || []
  const setMember = (userId: string, selected: boolean) =>
    update({
      members: selected
        ? [
            ...members,
            {
              userId,
              teamId: editor.value.id || '',
              role: members.length === 0 ? ('admin' as const) : ('member' as const),
            },
          ]
        : members.filter((member) => member.userId !== userId),
    })
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
          groups={groups}
          budgets={budgets}
          onGroups={(accessGroupIds) => update({ accessGroupIds })}
          onBudget={(budgetId) => update({ budgetId })}
          required
          label="Team defaults"
        />
      ) : null}
      <SelectionSection title="Members" detail={`${members.length} selected · optional`}>
        {users.map((user) => {
          const member = members.find((item) => item.userId === user.id)
          return (
            <div className={styles.memberChoice} key={user.id}>
              <CheckCard
                checked={Boolean(member)}
                title={user.name}
                detail={user.email}
                onChange={(selected) => setMember(user.id, selected)}
              />
              {member ? (
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
              ) : null}
            </div>
          )
        })}
        {!users.length ? <EmptyChoice text="Invite a user first." /> : null}
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
  users,
  teams,
  groups,
  budgets,
  keys,
  selfService,
  selfUserId,
  onChange,
  update,
}: {
  editor: Extract<AccessEditor, { kind: 'key' }>
  users: AccessUser[]
  teams: AccessTeam[]
  groups: AccessGroup[]
  budgets: AccessBudget[]
  keys: AccessAPIKey[]
  selfService: boolean
  selfUserId: string
  onChange: (value: AccessEditor) => void
  update: (patch: Record<string, unknown>) => void
}) {
  const ownerId = editor.value.ownerId || ''
  const ownerUser = users.find((user) => user.id === ownerId)
  const eligibleTeams = ownerUser
    ? teams.filter((team) =>
        ownerUser.memberships.some((membership) => membership.teamId === team.id),
      )
    : teams
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
          <Field label={editor.ownerType === 'user' ? 'User' : 'Team'} wide>
            <select
              value={ownerId}
              onChange={(event) =>
                update({
                  ownerType: editor.ownerType,
                  ownerId: event.target.value,
                  contextTeamId: editor.ownerType === 'team' ? event.target.value : undefined,
                })
              }
              required
            >
              <option value="">Select {editor.ownerType === 'user' ? 'a user' : 'a Team'}</option>
              {(editor.ownerType === 'user' ? users : teams).map((owner) => (
                <option value={owner.id} key={owner.id}>
                  {owner.name}
                </option>
              ))}
            </select>
          </Field>
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
        <Field
          label="Team context"
          wide
          hint="Optional. The key inherits this Team when the user has no override."
        >
          <select
            value={editor.value.contextTeamId || ''}
            onChange={(event) => update({ contextTeamId: event.target.value || undefined })}
          >
            <option value="">Personal policy only</option>
            {eligibleTeams.map((team) => (
              <option value={team.id} key={team.id}>
                {team.name}
              </option>
            ))}
          </select>
        </Field>
      ) : null}
      {!selfService ? (
        <Advanced label="Advanced settings">
          <PolicyFields
            accessGroupIds={editor.value.accessGroupIds || []}
            budgetId={editor.value.budgetId}
            groups={groups}
            budgets={budgets}
            onGroups={(accessGroupIds) => update({ accessGroupIds })}
            onBudget={(budgetId) => update({ budgetId })}
            label="Key override"
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
      ) : null}
    </>
  )
}

function GroupFields({
  editor,
  update,
}: {
  editor: Extract<AccessEditor, { kind: 'group' }>
  update: (patch: Record<string, unknown>) => void
}) {
  return (
    <>
      <Field label="Models" wide hint="Required · one model ID or trailing wildcard per line">
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
          required
        />
      </Field>
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
      <div className={styles.quotaInputs}>
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
      </div>
      <p className={styles.formNote}>
        Set at least one limit. Zero means unlimited for that dimension.
      </p>
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
  groups,
  budgets,
  onGroups,
  onBudget,
  required = false,
  label,
}: {
  accessGroupIds: string[]
  budgetId?: string
  groups: AccessGroup[]
  budgets: AccessBudget[]
  onGroups: (ids: string[]) => void
  onBudget: (id?: string) => void
  required?: boolean
  label: string
}) {
  return (
    <>
      <SelectionSection
        title="Model access"
        detail={`${label} · ${required ? 'required' : 'optional'}`}
      >
        {groups.map((group) => (
          <CheckCard
            key={group.id}
            checked={accessGroupIds.includes(group.id)}
            title={group.name}
            detail={group.modelPatterns.join(', ')}
            onChange={(checked) =>
              onGroups(
                checked
                  ? [...accessGroupIds, group.id]
                  : accessGroupIds.filter((id) => id !== group.id),
              )
            }
          />
        ))}
        {!groups.length ? <EmptyChoice text="Create an access group first." /> : null}
      </SelectionSection>
      <Field label="Budget" wide hint={`${label} · ${required ? 'required' : 'optional'}`}>
        <select
          value={budgetId || ''}
          onChange={(event) => onBudget(event.target.value || undefined)}
          required={required}
        >
          <option value="">{required ? 'Select a budget' : 'Inherit'}</option>
          {budgets.map((budget) => (
            <option value={budget.id} key={budget.id}>
              {budget.name} · {budget.rpm || '∞'} RPM · {budget.tpm || '∞'} TPM
            </option>
          ))}
        </select>
      </Field>
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
      <i>{active ? '✓' : ''}</i>
    </button>
  )
}
function EmptyChoice({ text }: { text: string }) {
  return <p className={styles.emptyChoice}>{text}</p>
}
