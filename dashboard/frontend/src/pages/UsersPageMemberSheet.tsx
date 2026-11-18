import { useEffect, useState } from 'react'
import { Check, Minus, MoreHorizontal, Trash2 } from 'lucide-react'

import PasswordInput from '../components/PasswordInput'
import {
  Alert,
  Avatar,
  Badge,
  Button,
  CopyButton,
  DetailSheet,
  DetailSheetBody,
  DetailSheetFooter,
  DetailSheetHeader,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  KeyValueGrid,
  SectionLabel,
  Skeleton,
  StatusLabel,
  Switch,
  cn,
} from '../ui'
import {
  DASHBOARD_ROLES,
  ROLE_META,
  absoluteTime,
  relativeTime,
  roleLabel,
  validateUserPassword,
  type AdminUser,
  type UsersPageRolePermissions,
} from './usersPageSupport'

export interface MemberDraft {
  role: string
  status: string
  password: string
}

interface UsersPageMemberSheetProps {
  user: AdminUser | null
  open: boolean
  isSelf: boolean
  canManage: boolean
  rolePermissions: UsersPageRolePermissions
  loadingPermissions: boolean
  submitting: boolean
  error: string | null
  onOpenChange: (open: boolean) => void
  onSubmit: (draft: MemberDraft) => void
  onDelete: (user: AdminUser) => void
}

export default function UsersPageMemberSheet({
  user,
  open,
  isSelf,
  canManage,
  rolePermissions,
  loadingPermissions,
  submitting,
  error,
  onOpenChange,
  onSubmit,
  onDelete,
}: UsersPageMemberSheetProps) {
  const [draft, setDraft] = useState<MemberDraft>({ role: 'read', status: 'active', password: '' })
  useEffect(() => {
    if (user) setDraft({ role: user.role, status: user.status, password: '' })
  }, [user])

  if (!user) return null
  const dirty = draft.role !== user.role || draft.status !== user.status || draft.password !== ''
  const passwordError = draft.password ? validateUserPassword(draft.password, 'edit') : null
  const locked = !canManage || isSelf
  const allPermissions = [...new Set(Object.values(rolePermissions).flat())].sort()
  const granted = new Set(rolePermissions[draft.role] ?? [])
  const firstName = user.name?.split(' ')[0] || user.email

  return (
    <DetailSheet open={open} onOpenChange={onOpenChange} size="md">
      <DetailSheetHeader
        icon={
          <Avatar
            name={user.name || user.email}
            size={44}
            className={cn(user.status !== 'active' && 'opacity-60 grayscale')}
          />
        }
        title={user.name || user.email}
        badges={
          <>
            <Badge tone={ROLE_META[user.role]?.tone ?? 'neutral'}>{roleLabel(user.role)}</Badge>
            {isSelf ? (
              <Badge tone="outline" size="sm">
                You
              </Badge>
            ) : null}
          </>
        }
        description={
          <>
            <span className="min-w-0 truncate" title={user.email}>
              {user.email}
            </span>
            <span className="text-subtle-foreground">·</span>
            <StatusLabel tone={user.status === 'active' ? 'success' : 'neutral'}>
              {user.status === 'active' ? 'Active' : 'Inactive'}
            </StatusLabel>
          </>
        }
        actions={
          canManage ? (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="icon-sm" aria-label="More actions">
                  <MoreHorizontal />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-44">
                <DropdownMenuItem
                  variant="destructive"
                  disabled={isSelf}
                  onSelect={() => onDelete(user)}
                >
                  <Trash2 />
                  Delete user
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          ) : undefined
        }
      />
      <DetailSheetBody className="flex flex-col gap-6 px-5 py-5">
        {error ? (
          <Alert tone="danger" title="Not everything was saved">
            {error}
          </Alert>
        ) : null}
        <KeyValueGrid
          items={[
            {
              label: 'Last sign-in',
              value: (
                <span title={absoluteTime(user.lastLoginAt)}>{relativeTime(user.lastLoginAt)}</span>
              ),
            },
            { label: 'Joined', value: absoluteTime(user.createdAt) },
            { label: 'Updated', value: absoluteTime(user.updatedAt) },
            {
              label: 'User ID',
              value: (
                <span className="flex min-w-0 items-center gap-1">
                  <span className="truncate font-mono text-[12px]">{user.id}</span>
                  <CopyButton value={user.id} label="Copy user ID" />
                </span>
              ),
            },
          ]}
        />

        <section className="flex flex-col gap-2">
          <SectionLabel id="member-role-label">Role</SectionLabel>
          <div
            role="radiogroup"
            aria-labelledby="member-role-label"
            className="grid grid-cols-3 gap-2"
          >
            {DASHBOARD_ROLES.map((value) => {
              const active = value === draft.role
              return (
                <button
                  key={value}
                  type="button"
                  role="radio"
                  aria-checked={active}
                  disabled={locked || submitting}
                  onClick={() => setDraft((current) => ({ ...current, role: value }))}
                  className={cn(
                    'flex items-center justify-between gap-2 rounded-lg border bg-card px-3 py-2.5 text-left text-[13px] font-medium outline-none transition-[border-color,box-shadow] duration-200 focus-visible:ring-[3px] focus-visible:ring-ring/40 disabled:cursor-not-allowed disabled:opacity-60',
                    active
                      ? 'border-primary shadow-[0_0_0_3px_var(--ring)]'
                      : 'border-border enabled:hover:border-border-strong',
                  )}
                >
                  {ROLE_META[value].label}
                  <span
                    className={cn(
                      'grid size-4 place-items-center rounded-full border',
                      active ? 'border-primary' : 'border-border-strong',
                    )}
                  >
                    {active ? <span className="size-2 rounded-full bg-primary" /> : null}
                  </span>
                </button>
              )
            })}
          </div>
          <p className="text-xs text-muted-foreground">
            {isSelf
              ? 'You can’t change your own role or access. Ask another admin.'
              : ROLE_META[draft.role]?.description}
          </p>
        </section>

        <section className="flex items-center justify-between gap-4 rounded-lg border border-border bg-surface-subtle/60 px-3.5 py-3">
          <span className="flex flex-col">
            <span className="text-[13px] font-medium">Can sign in</span>
            <span className="text-xs text-muted-foreground">
              {draft.status === 'active'
                ? 'The account is active.'
                : 'Inactive accounts keep their history but cannot sign in.'}
            </span>
          </span>
          <Switch
            checked={draft.status === 'active'}
            disabled={locked || submitting}
            onCheckedChange={(checked) =>
              setDraft((current) => ({ ...current, status: checked ? 'active' : 'inactive' }))
            }
            aria-label="Can sign in"
          />
        </section>

        <section className="flex flex-col gap-2">
          <SectionLabel>What {isSelf ? 'you' : firstName} can do</SectionLabel>
          {loadingPermissions ? (
            <div className="flex flex-col gap-2">
              {Array.from({ length: 4 }).map((_, index) => (
                <Skeleton key={index} className="h-8" />
              ))}
            </div>
          ) : allPermissions.length ? (
            <div className="grid grid-cols-1 gap-x-4 gap-y-2 rounded-lg border border-border px-3 py-2.5 sm:grid-cols-2">
              {allPermissions.map((permission) => {
                const allowed = granted.has(permission)
                return (
                  <div key={permission} className="flex min-w-0 items-center gap-2 text-[13px]">
                    {allowed ? (
                      <Check
                        className="size-4 shrink-0 text-success"
                        strokeWidth={2.4}
                        aria-label="Allowed"
                      />
                    ) : (
                      <Minus
                        className="size-4 shrink-0 text-subtle-foreground/60"
                        aria-label="Not allowed"
                      />
                    )}
                    <code
                      className={cn(
                        'truncate font-mono text-[12px]',
                        !allowed && 'text-subtle-foreground',
                      )}
                    >
                      {permission}
                    </code>
                  </div>
                )
              })}
            </div>
          ) : (
            <p className="text-[13px] text-muted-foreground">Permissions are not available.</p>
          )}
        </section>

        {canManage ? (
          <section className="flex flex-col gap-2">
            <SectionLabel>
              <label htmlFor="member-password">New password</label>
            </SectionLabel>
            <PasswordInput
              id="member-password"
              name="new-password"
              autoComplete="new-password"
              value={draft.password}
              disabled={submitting}
              placeholder="Leave blank to keep the current password"
              aria-invalid={Boolean(passwordError) || undefined}
              onChange={(event) =>
                setDraft((current) => ({ ...current, password: event.target.value }))
              }
              className="h-9"
            />
            <p
              className={cn(
                'text-xs',
                passwordError ? 'text-destructive-foreground' : 'text-muted-foreground',
              )}
            >
              {passwordError ??
                'Rotates their password after the role and access change. At most 72 bytes.'}
            </p>
          </section>
        ) : null}
      </DetailSheetBody>
      {canManage ? (
        <DetailSheetFooter highlight={dirty}>
          <span className="text-xs text-muted-foreground">
            {dirty
              ? `${firstName}'s changes apply on their next request`
              : 'Role changes apply on the next request'}
          </span>
          <div className="ml-auto flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              disabled={submitting}
              onClick={() =>
                dirty
                  ? setDraft({ role: user.role, status: user.status, password: '' })
                  : onOpenChange(false)
              }
            >
              {dirty ? 'Discard' : 'Close'}
            </Button>
            <Button
              size="sm"
              disabled={!dirty || Boolean(passwordError)}
              loading={submitting}
              onClick={() => onSubmit(draft)}
            >
              Save changes
            </Button>
          </div>
        </DetailSheetFooter>
      ) : null}
    </DetailSheet>
  )
}
