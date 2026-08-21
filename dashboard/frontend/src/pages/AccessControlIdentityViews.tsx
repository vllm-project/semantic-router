import { useMemo, useState } from 'react'
import {
  absoluteInvitationURL,
  dashboardMemberInvitationApi,
  type DashboardMemberInvitation,
} from '../utils/dashboardMemberInvitations'
import { copyText } from '../utils/clipboard'
import type { AccessUser } from '../utils/inferenceAccessApi'
import type { AccessControlViewProps as Props, DashboardMember } from './AccessControlViewTypes'
import {
  Avatar,
  Empty,
  EntityTable,
  ListToolbar,
  Pagination,
  Status,
} from './AccessControlViewPrimitives'
import { date, initials, slicePage } from './AccessControlViewSupport'
import styles from './AccessControlPage.module.css'

interface IdentityRow {
  key: string
  access?: AccessUser
  member?: DashboardMember
  invitation?: DashboardMemberInvitation
  name: string
  email: string
}

function identityRows(
  users: AccessUser[],
  members: DashboardMember[],
  invitations: DashboardMemberInvitation[],
) {
  const rows = new Map<string, IdentityRow>()
  users.forEach((user) =>
    rows.set(user.email.toLowerCase(), {
      key: user.id,
      access: user,
      name: user.name,
      email: user.email,
    }),
  )
  members.forEach((member) => {
    const key = member.email.toLowerCase()
    const current = rows.get(key)
    rows.set(key, {
      key: current?.key || member.id,
      ...current,
      member,
      name: current?.name || member.name,
      email: member.email,
    })
  })
  invitations
    .filter((item) => item.status === 'pending')
    .forEach((invitation) => {
      const key = invitation.email.toLowerCase()
      const current = rows.get(key)
      rows.set(key, {
        key: current?.key || invitation.id,
        ...current,
        invitation,
        name: current?.name || invitation.name,
        email: invitation.email,
      })
    })
  return [...rows.values()].sort((a, b) => a.name.localeCompare(b.name))
}

export function UsersView(props: Props) {
  const rows = useMemo(
    () => identityRows(props.users, props.dashboardMembers, props.invitations),
    [props.dashboardMembers, props.invitations, props.users],
  )
  const filtered = rows.filter((row) =>
    `${row.name} ${row.email}`.toLowerCase().includes(props.pageState.query.toLowerCase()),
  )
  const paged = slicePage(filtered, props.pageState)
  const activeInvites = props.invitations.filter((item) => item.status === 'pending').length
  return (
    <div className={styles.viewStack}>
      <div className={styles.tabBar}>
        <button
          type="button"
          className={props.identityTab === 'users' ? styles.tabActive : ''}
          onClick={() => props.onIdentityTabChange('users')}
        >
          Users <span>{rows.length}</span>
        </button>
        <button
          type="button"
          className={props.identityTab === 'invitations' ? styles.tabActive : ''}
          onClick={() => props.onIdentityTabChange('invitations')}
        >
          Invitations <span>{activeInvites}</span>
        </button>
      </div>
      {props.identityTab === 'invitations' ? (
        <InvitationsView {...props} />
      ) : (
        <>
          <ListToolbar
            state={props.pageState}
            onChange={props.onPageStateChange}
            placeholder="Search users"
          />
          <div className={styles.dataTable}>
            <div className={`${styles.dataRow} ${styles.userColumns} ${styles.dataHeader}`}>
              <span>User</span>
              <span>Model access</span>
              <span>Dashboard access</span>
              <span>Activity</span>
              <span />
            </div>
            {paged.map((row) => {
              const teamCount = row.access
                ? props.teams.filter((team) => team.userIds.includes(row.access!.id)).length
                : 0
              const keyCount = row.access
                ? props.keys.filter((key) => key.userId === row.access!.id).length
                : 0
              return (
                <div
                  className={`${styles.dataRow} ${styles.userColumns} ${row.access || row.member ? styles.dataRowInteractive : ''}`}
                  key={row.key}
                  role={row.access || row.member ? 'link' : undefined}
                  tabIndex={row.access || row.member ? 0 : undefined}
                  onClick={() => {
                    if (row.access) props.onOpenEntity(row.access.id)
                    else if (row.member) props.onOpenDashboardMember(row.member.id)
                  }}
                  onKeyDown={(event) => {
                    if (event.key !== 'Enter') return
                    if (row.access) props.onOpenEntity(row.access.id)
                    else if (row.member) props.onOpenDashboardMember(row.member.id)
                  }}
                >
                  <div className={styles.identityCell}>
                    <Avatar name={row.name} />
                    <div>
                      <strong>{row.name}</strong>
                      <span>{row.email}</span>
                    </div>
                  </div>
                  <div className={styles.stackCell}>
                    {row.access ? (
                      <>
                        <Status value={row.access.status} />
                        <small>
                          {keyCount} keys · {teamCount} teams
                        </small>
                      </>
                    ) : (
                      <>
                        <span className={styles.mutedBadge}>Not enabled</span>
                        <small>No API identity</small>
                      </>
                    )}
                  </div>
                  <div className={styles.stackCell}>
                    {row.member ? (
                      <>
                        <span className={styles.roleBadge}>{row.member.role}</span>
                        <small>{row.member.status}</small>
                      </>
                    ) : row.invitation ? (
                      <>
                        <span className={styles.pendingBadge}>Invited</span>
                        <small>Expires {date(row.invitation.expiresAt)}</small>
                      </>
                    ) : (
                      <>
                        <span className={styles.mutedBadge}>No login</span>
                        <small>API access only</small>
                      </>
                    )}
                  </div>
                  <div className={styles.stackCell}>
                    <span>
                      {row.member?.lastLoginAt ? date(row.member.lastLoginAt) : 'No sign-in yet'}
                    </span>
                    <small>
                      {row.access?.createdAt
                        ? `Added ${date(row.access.createdAt)}`
                        : 'Dashboard member'}
                    </small>
                  </div>
                  <span className={styles.rowChevron} aria-hidden="true">
                    ›
                  </span>
                </div>
              )
            })}
            {paged.length === 0 ? (
              <Empty
                title="No users found"
                detail="Try a different search or add the first user."
              />
            ) : null}
          </div>
          <Pagination
            total={filtered.length}
            state={props.pageState}
            onChange={props.onPageStateChange}
          />
        </>
      )}
    </div>
  )
}

function InvitationsView(props: Props) {
  const [pending, setPending] = useState('')
  const [revealed, setRevealed] = useState('')
  const [copied, setCopied] = useState('')
  const items = props.invitations.filter((item) =>
    `${item.name} ${item.email} ${item.status}`
      .toLowerCase()
      .includes(props.pageState.query.toLowerCase()),
  )
  const act = async (action: 'resend' | 'revoke', item: DashboardMemberInvitation) => {
    setPending(item.id)
    try {
      if (action === 'resend') {
        const next = await dashboardMemberInvitationApi.resend(item.id, true)
        setRevealed(absoluteInvitationURL(next))
      } else if (window.confirm(`Revoke ${item.name}'s invitation?`)) {
        await dashboardMemberInvitationApi.revoke(item.id)
      }
      props.onInvitationsChanged()
    } finally {
      setPending('')
    }
  }
  return (
    <>
      <ListToolbar
        state={props.pageState}
        onChange={props.onPageStateChange}
        placeholder="Search invitations"
      />
      {revealed ? (
        <div className={styles.linkReveal}>
          <div>
            <span>Fresh invitation link</span>
            <code>{revealed}</code>
          </div>
          <button
            type="button"
            onClick={() => {
              void copyText(revealed).then((success) => setCopied(success ? revealed : ''))
            }}
          >
            {copied === revealed ? 'Copied' : 'Copy link'}
          </button>
          <button type="button" aria-label="Close link" onClick={() => setRevealed('')}>
            ×
          </button>
        </div>
      ) : null}
      <div className={styles.dataTable}>
        <div className={`${styles.dataRow} ${styles.inviteColumns} ${styles.dataHeader}`}>
          <span>Invitee</span>
          <span>Role</span>
          <span>Delivery</span>
          <span>Expires</span>
          <span />
        </div>
        {slicePage(items, props.pageState).map((item) => (
          <div className={`${styles.dataRow} ${styles.inviteColumns}`} key={item.id}>
            <div className={styles.identityCell}>
              <Avatar name={item.name} />
              <div>
                <strong>{item.name}</strong>
                <span>{item.email}</span>
              </div>
            </div>
            <span className={styles.roleBadge}>{item.role}</span>
            <div className={styles.stackCell}>
              <Status
                value={item.status === 'pending' ? 'active' : 'disabled'}
                label={item.status}
              />
              <small>{item.deliveryStatus.replace(/_/g, ' ')}</small>
            </div>
            <div className={styles.stackCell}>
              <span>{date(item.expiresAt)}</span>
              <small>Sent {date(item.lastSentAt || item.createdAt)}</small>
            </div>
            <div className={styles.rowMenu}>
              {item.status === 'pending' ? (
                <>
                  <button
                    type="button"
                    disabled={pending === item.id}
                    onClick={() => void act('resend', item)}
                  >
                    New link
                  </button>
                  <button
                    type="button"
                    className={styles.dangerText}
                    disabled={pending === item.id}
                    onClick={() => void act('revoke', item)}
                  >
                    Revoke
                  </button>
                </>
              ) : (
                <span className={styles.closedLabel}>Closed</span>
              )}
            </div>
          </div>
        ))}
        {items.length === 0 ? (
          <Empty
            title="No invitations"
            detail="Invite a user to give them a personal Dashboard sign-up link."
          />
        ) : null}
      </div>
      <Pagination total={items.length} state={props.pageState} onChange={props.onPageStateChange} />
    </>
  )
}

export function TeamsView(props: Props) {
  const filtered = props.teams
  return (
    <EntityTable
      toolbar={
        <ListToolbar
          state={props.pageState}
          onChange={props.onPageStateChange}
          placeholder="Search teams"
        />
      }
      pagination={
        <Pagination
          total={props.entityTotals.teams}
          state={props.pageState}
          onChange={props.onPageStateChange}
        />
      }
    >
      <div className={`${styles.dataRow} ${styles.teamColumns} ${styles.dataHeader}`}>
        <span>Team</span>
        <span>Members</span>
        <span>Model access</span>
        <span>Budget</span>
        <span>Status</span>
        <span />
      </div>
      {filtered.map((team) => (
        <div
          className={`${styles.dataRow} ${styles.teamColumns} ${styles.dataRowInteractive}`}
          key={team.id}
          role="link"
          tabIndex={0}
          onClick={() => props.onOpenEntity(team.id)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') props.onOpenEntity(team.id)
          }}
        >
          <div className={styles.identityCell}>
            <Avatar name={team.name} square />
            <div>
              <strong>{team.name}</strong>
              <span>{team.description || 'No description'}</span>
            </div>
          </div>
          <div className={styles.avatarPile}>
            {team.userIds.slice(0, 4).map((id) => (
              <span key={id} title={props.users.find((user) => user.id === id)?.name}>
                {initials(props.users.find((user) => user.id === id)?.name || id)}
              </span>
            ))}
            <small>{team.userIds.length} users</small>
          </div>
          <div className={styles.stackCell}>
            <span>{team.accessGroupIds.length} groups</span>
            <small>Team default</small>
          </div>
          <div className={styles.stackCell}>
            <span>{team.budget ? `${team.budget.rpm || '∞'} RPM` : 'Not set'}</span>
            <small>{team.budget ? `${team.budget.tpm || '∞'} TPM` : 'Required'}</small>
          </div>
          <Status value={team.status} />
          <span className={styles.rowChevron} aria-hidden="true">
            ›
          </span>
        </div>
      ))}
      {filtered.length === 0 ? (
        <Empty
          title="No teams found"
          detail="Teams share model access and quota across a group of users."
        />
      ) : null}
    </EntityTable>
  )
}
