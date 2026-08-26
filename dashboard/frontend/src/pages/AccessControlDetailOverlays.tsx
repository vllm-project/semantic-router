import { type AccessAPIKey, type AccessTeam, type AccessUser } from '../utils/inferenceAccessApi'
import {
  AccessEntityDetail,
  APIKeyDetail,
  DashboardMemberDetail,
  RequestLogDetail,
} from './AccessControlDetails'
import type { EntityDetailKind, EntityDetailValue } from './AccessEntityDetail'
import type { UnifiedUserDeletionProgress } from './unifiedUserDeletion'

interface AccessControlDetailOverlaysProps {
  detailKeyId: string
  detailLogId: string
  detailEntityId: string
  detailMemberId: string
  entityKind: EntityDetailKind | null
  users: AccessUser[]
  teams: AccessTeam[]
  keys: AccessAPIKey[]
  canManage: boolean
  canRevealKeys: boolean
  canManageDashboardMembers: boolean
  selfService: boolean
  selfUserId: string
  resourceName: (resourceType: 'model' | 'entrypoint', resourceId: string) => string
  onClose: () => void
  onCatalogChanged: () => void
  onDashboardMembersChanged: () => void
  onEditKey: (key: AccessAPIKey) => void
  onEditEntity: (kind: EntityDetailKind, item: EntityDetailValue) => void
  onDeleteEntity: (kind: EntityDetailKind, id: string) => Promise<void>
  onRemoveDashboardLogin: (memberId: string) => Promise<void>
  onDeleteUnifiedUser: (
    memberId: string,
    modelUserId: string,
    progress: UnifiedUserDeletionProgress,
  ) => Promise<UnifiedUserDeletionProgress>
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
    canManage,
    canRevealKeys,
    canManageDashboardMembers,
    selfService,
    selfUserId,
    resourceName,
    onClose,
    onCatalogChanged,
    onDashboardMembersChanged,
    onEditKey,
    onEditEntity,
    onDeleteEntity,
    onRemoveDashboardLogin,
    onDeleteUnifiedUser,
    onEditModelAccess,
  } = props
  return (
    <>
      {detailKeyId ? (
        <APIKeyDetail
          keyId={detailKeyId}
          canManage={canManage}
          canReveal={canRevealKeys}
          canEditPolicy={canManage}
          selfService={selfService}
          selfUserId={selfUserId}
          onEdit={onEditKey}
          onClose={onClose}
          onChanged={onCatalogChanged}
          onDeleted={() => {
            onClose()
            onCatalogChanged()
          }}
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
          canEdit={canManage}
          canDelete={canManage}
          selfService={selfService}
          selfUserId={selfUserId}
          resourceName={resourceName}
          onEdit={onEditEntity}
          onDelete={onDeleteEntity}
          onClose={onClose}
        />
      ) : null}
      {detailMemberId ? (
        <DashboardMemberDetail
          memberId={detailMemberId}
          users={users}
          canManage={canManageDashboardMembers}
          canManageModelUser={canManage}
          canDeleteUnifiedUser={canManage && canManageDashboardMembers}
          onChanged={onDashboardMembersChanged}
          onEditModelAccess={onEditModelAccess}
          onRemoveLogin={onRemoveDashboardLogin}
          onDeleteUser={onDeleteUnifiedUser}
          onClose={onClose}
        />
      ) : null}
    </>
  )
}
