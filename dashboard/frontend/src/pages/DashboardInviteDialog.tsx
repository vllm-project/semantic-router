import { useEffect, useMemo, useState, type FormEvent } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { CircleCheck, Link2, Minus, Plus, UserPlus, UsersRound } from 'lucide-react'

import {
  Alert,
  Badge,
  Button,
  CopyField,
  Dialog,
  DialogBody,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  Field,
  Input,
  Label,
  Segmented,
  cn,
} from '../ui'
import type { DashboardInvitation, DashboardRole, InvitationKind } from './dashboardInvitationTypes'
import { ROLE_META, absoluteTime, invitationUrl, responseText } from './usersPageSupport'

interface Props {
  isOpen: boolean
  onClose: () => void
  onCreated?: (invitation: DashboardInvitation) => void
}

const ROLE_ORDER: DashboardRole[] = ['read', 'write', 'admin']
const EMAIL = /^[^\s@]+@[^\s@]+\.[^\s@]+$/

export default function DashboardInviteDialog({ isOpen, onClose, onCreated }: Props) {
  const [email, setEmail] = useState('')
  const [name, setName] = useState('')
  const [role, setRole] = useState<DashboardRole>('read')
  const [kind, setKind] = useState<InvitationKind>('personal')
  const [capacity, setCapacity] = useState(10)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState('')
  const [submitted, setSubmitted] = useState(false)
  const [created, setCreated] = useState<{ invitation: DashboardInvitation; url: string } | null>(
    null,
  )

  useEffect(() => {
    if (!isOpen) return
    setEmail('')
    setName('')
    setRole('read')
    setKind('personal')
    setCapacity(10)
    setPending(false)
    setError('')
    setSubmitted(false)
    setCreated(null)
  }, [isOpen])

  const expiry = useMemo(
    () => (created ? absoluteTime(created.invitation.expiresAt) : ''),
    [created],
  )
  const nameError =
    kind === 'personal' && submitted && !name.trim() ? 'Enter their name.' : undefined
  const emailError =
    kind === 'personal' && submitted && !EMAIL.test(email.trim())
      ? 'Enter a valid email address.'
      : undefined

  const submit = async (event: FormEvent) => {
    event.preventDefault()
    setSubmitted(true)
    if (kind === 'personal' && (!name.trim() || !EMAIL.test(email.trim()))) return
    setPending(true)
    setError('')
    try {
      const response = await fetch('/api/admin/invitations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          kind,
          role,
          ...(kind === 'personal' ? { email: email.trim(), name: name.trim() } : { capacity }),
        }),
      })
      if (!response.ok) throw new Error(await responseText(response))
      const payload = (await response.json()) as { invitation: DashboardInvitation; token: string }
      setCreated({ invitation: payload.invitation, url: invitationUrl(payload.token) })
      onCreated?.(payload.invitation)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not create the invitation.')
    } finally {
      setPending(false)
    }
  }

  const title = created
    ? created.invitation.kind === 'shared'
      ? 'Shared invitation ready'
      : `Welcome ${created.invitation.name}`
    : 'Invite user'

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && !pending && onClose()}>
      <DialogContent size="lg" aria-describedby="invite-dialog-description">
        <AnimatePresence mode="wait" initial={false}>
          {created ? (
            <motion.div
              key="ready"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
            >
              <DialogHeader className="items-start pt-6">
                <span className="mb-2 grid size-10 place-items-center rounded-full bg-success-soft text-success-foreground">
                  <CircleCheck className="size-5" />
                </span>
                <DialogTitle>{title}</DialogTitle>
                <DialogDescription id="invite-dialog-description">
                  {created.invitation.kind === 'shared'
                    ? `${created.invitation.maxUses} tickets are ready. Each person creates their own account; the link expires ${expiry}.`
                    : `A workspace ticket is reserved for ${created.invitation.email}. The link works once and expires ${expiry}.`}
                </DialogDescription>
              </DialogHeader>
              <DialogBody className="flex flex-col gap-3 pb-5">
                <div className="flex flex-col gap-1.5">
                  <Label>
                    {created.invitation.kind === 'shared'
                      ? 'Shared invitation link'
                      : 'One-time invitation link'}
                  </Label>
                  <CopyField value={created.url} />
                </div>
                <Alert tone="info">
                  The dashboard does not send email. Share the link yourself, for example in your
                  team chat. It joins as{' '}
                  {ROLE_META[created.invitation.role]?.label ?? created.invitation.role}.
                </Alert>
              </DialogBody>
              <DialogFooter>
                <Button variant="ghost" className="mr-auto" onClick={() => setCreated(null)}>
                  Invite someone else
                </Button>
                <Button onClick={onClose}>Done</Button>
              </DialogFooter>
            </motion.div>
          ) : (
            <motion.form
              key="form"
              onSubmit={submit}
              noValidate
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
            >
              <DialogHeader>
                <DialogTitle>{title}</DialogTitle>
                <DialogDescription id="invite-dialog-description">
                  {kind === 'personal'
                    ? 'Reserve their identity. They only choose a password.'
                    : 'One link, one role, and a clear number of tickets.'}
                </DialogDescription>
              </DialogHeader>
              <DialogBody className="flex flex-col gap-5 pb-5">
                <Segmented
                  className="self-start"
                  aria-label="Invitation type"
                  value={kind}
                  onValueChange={(next) => {
                    setKind(next)
                    setSubmitted(false)
                  }}
                  options={[
                    { value: 'personal', label: 'One person', icon: <UserPlus /> },
                    { value: 'shared', label: 'Shared link', icon: <Link2 /> },
                  ]}
                />
                {error ? (
                  <Alert tone="danger" title="The invitation was not created">
                    {error}
                  </Alert>
                ) : null}
                {kind === 'personal' ? (
                  <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                    <Field label="Name" htmlFor="invite-name" error={nameError}>
                      <Input
                        id="invite-name"
                        value={name}
                        onChange={(event) => setName(event.target.value)}
                        placeholder="Ada Lovelace"
                        autoFocus
                        autoComplete="off"
                        aria-invalid={Boolean(nameError) || undefined}
                      />
                    </Field>
                    <Field label="Email" htmlFor="invite-email" error={emailError}>
                      <Input
                        id="invite-email"
                        type="email"
                        value={email}
                        onChange={(event) => setEmail(event.target.value)}
                        placeholder="ada@example.com"
                        autoComplete="off"
                        aria-invalid={Boolean(emailError) || undefined}
                      />
                    </Field>
                  </div>
                ) : (
                  <div className="flex flex-wrap items-center justify-between gap-4 rounded-lg border border-border bg-surface-subtle/60 px-3.5 py-3">
                    <span className="flex items-start gap-3">
                      <UsersRound className="mt-0.5 size-4 text-subtle-foreground" />
                      <span className="flex flex-col">
                        <span className="text-[13px] font-medium">Available tickets</span>
                        <span className="text-xs text-muted-foreground">
                          Each completed registration uses one ticket.
                        </span>
                      </span>
                    </span>
                    <span className="flex items-center gap-1.5">
                      <Button
                        type="button"
                        variant="outline"
                        size="icon-sm"
                        aria-label="Remove one ticket"
                        onClick={() => setCapacity((value) => Math.max(2, value - 1))}
                        disabled={capacity <= 2}
                      >
                        <Minus />
                      </Button>
                      <Input
                        type="number"
                        min={2}
                        max={100}
                        value={capacity}
                        onChange={(event) =>
                          setCapacity(Math.min(100, Math.max(2, Number(event.target.value) || 2)))
                        }
                        aria-label="Invitation capacity"
                        autoFocus
                        className="h-8 w-16 text-center font-mono tabular"
                      />
                      <Button
                        type="button"
                        variant="outline"
                        size="icon-sm"
                        aria-label="Add one ticket"
                        onClick={() => setCapacity((value) => Math.min(100, value + 1))}
                        disabled={capacity >= 100}
                      >
                        <Plus />
                      </Button>
                    </span>
                  </div>
                )}
                <div className="grid gap-1.5">
                  <Label id="invite-role-label">Dashboard role</Label>
                  <div
                    role="radiogroup"
                    aria-labelledby="invite-role-label"
                    className="grid grid-cols-1 gap-2 sm:grid-cols-3"
                  >
                    {ROLE_ORDER.map((value) => {
                      const active = value === role
                      const meta = ROLE_META[value]
                      return (
                        <button
                          key={value}
                          type="button"
                          role="radio"
                          aria-checked={active}
                          aria-label={`${meta.label}: ${meta.description}`}
                          onClick={() => setRole(value)}
                          className={cn(
                            'flex flex-col gap-1.5 rounded-lg border bg-card p-3 text-left outline-none transition-[border-color,box-shadow] duration-200 focus-visible:ring-[3px] focus-visible:ring-ring/40',
                            active
                              ? 'border-primary shadow-[0_0_0_3px_var(--ring)]'
                              : 'border-border hover:border-border-strong',
                          )}
                        >
                          <span className="flex items-center justify-between">
                            <Badge tone={meta.tone}>{meta.label}</Badge>
                            <span
                              className={cn(
                                'grid size-4 place-items-center rounded-full border',
                                active ? 'border-primary' : 'border-border-strong',
                              )}
                            >
                              {active ? <span className="size-2 rounded-full bg-primary" /> : null}
                            </span>
                          </span>
                          <span className="text-xs leading-relaxed text-muted-foreground">
                            {meta.description}
                          </span>
                        </button>
                      )
                    })}
                  </div>
                </div>
              </DialogBody>
              <DialogFooter>
                <Button type="button" variant="ghost" onClick={onClose} disabled={pending}>
                  Cancel
                </Button>
                <Button type="submit" loading={pending}>
                  {pending ? null : kind === 'shared' ? <Link2 /> : <UserPlus />}
                  {kind === 'shared' ? 'Create shared link' : 'Create invitation'}
                </Button>
              </DialogFooter>
            </motion.form>
          )}
        </AnimatePresence>
      </DialogContent>
    </Dialog>
  )
}
