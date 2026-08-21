import {
  type AccessAPIKey,
  type AccessBudget,
  type AccessGroup,
  type AccessTeam,
  type AccessUser,
} from '../utils/inferenceAccessApi'
import {
  AccessEntityDetail,
  APIKeyDetail,
  DashboardMemberDetail,
  RequestLogDetail,
} from './AccessControlDetails'
import type { EntityDetailKind, EntityDetailValue } from './AccessEntityDetail'

interface AccessControlDetailOverlaysProps {
  detailKeyId: string
  detailLogId: string
  detailEntityId: string
  detailMemberId: string
  entityKind: EntityDetailKind | null
  users: AccessUser[]
  teams: AccessTeam[]
  keys: AccessAPIKey[]
  groups: AccessGroup[]
  budgets: AccessBudget[]
  canManage: boolean
  canManageDashboardMembers: boolean
  selfService: boolean
  selfUserId: string
  onClose: () => void
  onCatalogChanged: () => void
  onDashboardMembersChanged: () => void
  onEditKey: (key: AccessAPIKey) => void
  onEditEntity: (kind: EntityDetailKind, item: EntityDetailValue) => void
  onDeleteEntity: (kind: EntityDetailKind, id: string) => void
  onEditModelAccess: (user: AccessUser) => void
}

export default function AccessControlDetailOverlays(props: AccessControlDetailOverlaysProps) {
  const {
    detailKeyId,
    detailLogId,
    detailEntityId,
    detailMemberId,
    entityKind,
    users,
    teams,
    keys,
    groups,
    budgets,
    canManage,
    canManageDashboardMembers,
    selfService,
    selfUserId,
    onClose,
    onCatalogChanged,
    onDashboardMembersChanged,
    onEditKey,
    onEditEntity,
    onDeleteEntity,
    onEditModelAccess,
  } = props
  const canEditEntity = Boolean(
    canManage ||
      (selfService &&
        entityKind === 'team' &&
        teams.some(
          (team) =>
            team.id === detailEntityId &&
            team.members.some(
              (membership) => membership.userId === selfUserId && membership.role === 'admin',
            ),
        )),
  )

  return (
    <>
      {detailKeyId ? (
        <APIKeyDetail
          keyId={detailKeyId}
          users={users}
          teams={teams}
          groups={groups}
          budgets={budgets}
          canManage={canManage || selfService}
          canEditPolicy={canManage}
          selfService={selfService}
          onEdit={onEditKey}
          onClose={onClose}
          onChanged={onCatalogChanged}
        />
      ) : null}
      {detailLogId ? (
        <RequestLogDetail
          logId={detailLogId}
          users={users}
          teams={teams}
          keys={keys}
          selfService={selfService}
          onClose={onClose}
        />
      ) : null}
      {detailEntityId && entityKind ? (
        <AccessEntityDetail
          kind={entityKind}
          id={detailEntityId}
          users={users}
          teams={teams}
          keys={keys}
          groups={groups}
          budgets={budgets}
          canEdit={canEditEntity}
          canDelete={canManage}
          selfService={selfService}
          onEdit={onEditEntity}
          onDelete={onDeleteEntity}
          onClose={onClose}
        />
      ) : null}
      {detailMemberId ? (
        <DashboardMemberDetail
          memberId={detailMemberId}
          canManage={canManageDashboardMembers}
          onChanged={onDashboardMembersChanged}
          onEditModelAccess={onEditModelAccess}
          onClose={onClose}
        />
      ) : null}
    </>
  )
}
