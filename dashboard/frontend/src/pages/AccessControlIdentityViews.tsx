import { useMemo, useState } from 'react'
import ConfirmDialog from '../components/ConfirmDialog'
import ProductIcon from '../components/ProductIcon'
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
import { date, slicePage } from './AccessControlViewSupport'
import styles from './AccessControlPage.module.css'

function identityRows(
  users: AccessUser[],
  members: DashboardMember[],
  invitations: DashboardMemberInvitation[],
) {
  const memberByEmail = new Map(members.map((member) => [member.email.toLowerCase(), member]))
  const invitationByEmail = new Map(
    invitations
      .filter((item) => item.status === 'pending')
      .map((invitation) => [invitation.email.toLowerCase(), invitation]),
  )
  return users.map((user) => {
    const email = user.email.toLowerCase()
    return {
      key: user.id,
      access: user,
      member: memberByEmail.get(email),
      invitation: invitationByEmail.get(email),
      name: user.name,
      email: user.email,
    }
  })
}

export function UsersView(props: Props) {
  const rows = useMemo(
    () => identityRows(props.users, props.dashboardMembers, props.invitations),
    [props.dashboardMembers, props.invitations, props.users],
  )
  const activeInvites = props.invitations.filter((item) => item.status === 'pending').length
  return (
    <div className={styles.viewStack}>
      <div className={styles.tabBar}>
        <button
          type="button"
          className={props.identityTab === 'users' ? styles.tabActive : ''}
          onClick={() => props.onIdentityTabChange('users')}
        >
          <ProductIcon name="user" /> Users <span>{props.entityTotals.users}</span>
        </button>
        <button
          type="button"
          className={props.identityTab === 'invitations' ? styles.tabActive : ''}
          onClick={() => props.onIdentityTabChange('invitations')}
        >
          <ProductIcon name="inbox" /> Invitations <span>{activeInvites}</span>
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
            {rows.map((row) => (
              <div
                className={`${styles.dataRow} ${styles.userColumns} ${row.access || row.member ? styles.dataRowInteractive : ''}`}
                key={row.key}
                role={row.access || row.member ? 'link' : undefined}
                tabIndex={row.access || row.member ? 0 : undefined}
                onClick={() => {
                  if (row.member) props.onOpenDashboardMember(row.member.id)
                  else if (row.access) props.onOpenEntity(row.access.id)
                }}
                onKeyDown={(event) => {
                  if (event.key !== 'Enter') return
                  if (row.member) props.onOpenDashboardMember(row.member.id)
                  else if (row.access) props.onOpenEntity(row.access.id)
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
                      <small>Open for Teams, keys, and policy</small>
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
                  <ProductIcon name="chevron-right" />
                </span>
              </div>
            ))}
            {rows.length === 0 ? (
              <Empty
                title="No users found"
                detail="Try a different search or add the first user."
              />
            ) : null}
          </div>
          <Pagination
            total={props.entityTotals.users}
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
  const [revokeTarget, setRevokeTarget] = useState<DashboardMemberInvitation | null>(null)
  const items = props.invitations.filter((item) =>
    `${item.name} ${item.email} ${item.status}`
      .toLowerCase()
      .includes(props.pageState.query.toLowerCase()),
  )
  const act = async (action: 'resend' | 'revoke', item: DashboardMemberInvitation) => {
    setPending(item.id)
    try {
      if (action === 'resend') {
        const next = await dashboardMemberInvitationApi.resend(item.id, item.revision, true)
        setRevealed(absoluteInvitationURL(next))
      } else {
        await dashboardMemberInvitationApi.revoke(item.id, item.revision)
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
            <ProductIcon name={copied === revealed ? 'check' : 'copy'} />
            {copied === revealed ? 'Copied' : 'Copy link'}
          </button>
          <button type="button" aria-label="Close link" onClick={() => setRevealed('')}>
            <ProductIcon name="close" />
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
                    <ProductIcon name="refresh" />
                    New link
                  </button>
                  <button
                    type="button"
                    className={styles.dangerText}
                    disabled={pending === item.id}
                    onClick={() => setRevokeTarget(item)}
                  >
                    <ProductIcon name="trash" />
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
      <ConfirmDialog
        isOpen={Boolean(revokeTarget)}
        title="Revoke this invitation?"
        description={
          revokeTarget ? `${revokeTarget.name}'s current sign-up link will stop working.` : ''
        }
        eyebrow="Invitation"
        confirmLabel="Revoke invitation"
        pending={Boolean(revokeTarget && pending === revokeTarget.id)}
        onCancel={() => setRevokeTarget(null)}
        onConfirm={async () => {
          if (!revokeTarget) return
          await act('revoke', revokeTarget)
          setRevokeTarget(null)
        }}
      />
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
          <div className={styles.stackCell}>
            <span>View roster</span>
            <small>Loaded on demand</small>
          </div>
          <div className={styles.stackCell}>
            <span>View policy</span>
            <small>Resolved in details</small>
          </div>
          <div className={styles.stackCell}>
            <span>View quota</span>
            <small>Resolved in details</small>
          </div>
          <Status value={team.status} />
          <span className={styles.rowChevron} aria-hidden="true">
            <ProductIcon name="chevron-right" />
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
