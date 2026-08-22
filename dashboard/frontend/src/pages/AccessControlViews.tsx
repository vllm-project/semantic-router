import AccessControlUsageView from './AccessControlUsageView'
import { AuditView, RequestLogsView } from './AccessControlLogViews'
import { TeamsView, UsersView } from './AccessControlIdentityViews'
import { BudgetsView, GroupsView, KeysView } from './AccessControlPolicyViews'
import type { AccessControlViewProps } from './AccessControlViewTypes'

export type { DashboardMember, IdentityTab } from './AccessControlViewTypes'
export { default as DashboardAccessDialog } from './DashboardAccessDialog'

export default function AccessControlViews(props: AccessControlViewProps) {
  if (props.view === 'users') return <UsersView {...props} />
  if (props.view === 'teams') return <TeamsView {...props} />
  if (props.view === 'api-keys') return <KeysView {...props} />
  if (props.view === 'access-groups') return <GroupsView {...props} />
  if (props.view === 'budgets') return <BudgetsView {...props} />
  if (props.view === 'usage')
    return (
      <AccessControlUsageView
        overview={props.overview}
        usage={props.usage}
        users={props.users}
        teams={props.teams}
        keys={props.keys}
        groups={props.groups}
        usageScope={props.usageScope}
        onUsageScopeChange={props.onUsageScopeChange}
        loading={props.loading}
      />
    )
  if (props.view === 'request-logs') return <RequestLogsView {...props} />
  return <AuditView {...props} />
}
