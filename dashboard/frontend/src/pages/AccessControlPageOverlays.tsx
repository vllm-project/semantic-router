import type {
  AccessAPIKey,
  AccessBudget,
  AccessGroup,
  AccessTeam,
  AccessUser,
  CreatedAccessAPIKey,
} from '../utils/inferenceAccessApi'
import type { EntityDetailKind, EntityDetailValue } from './AccessEntityDetail'
import AccessControlDialog from './AccessControlDialog'
import AccessControlDetailOverlays from './AccessControlDetailOverlays'
import type { AccessEditor } from './AccessControlPageSupport'
import DashboardMemberInviteDialog from './DashboardMemberInviteDialog'
import { accessControlSelectorSources } from './accessControlSelectorSources'

interface AccessControlPageOverlaysProps {
  editor: AccessEditor | null
  createdKey: CreatedAccessAPIKey | null
  inviteOpen: boolean
  detail: {
    keyId: string
    logId: string
    entityId: string
    memberId: string
    entityKind: EntityDetailKind | null
  }
  catalog: {
    users: AccessUser[]
    teams: AccessTeam[]
    keys: AccessAPIKey[]
    groups: AccessGroup[]
    budgets: AccessBudget[]
  }
  permissions: {
    canManage: boolean
    canRevealKeys: boolean
    canManageDashboardMembers: boolean
    selfService: boolean
    selfUserId: string
  }
  editorError: string
  saving: boolean
  onEditorChange: (editor: AccessEditor) => void
  onEditorClose: () => void
  onEditorSave: () => void
  onCreatedKeyClose: () => void
  onCreatedKeyDetails: (keyId: string) => void
  onInviteClose: () => void
  onInviteCreated: () => void
  onDetailClose: () => void
  onCatalogChanged: () => void
  onDashboardMembersChanged: () => void
  onEditKey: (key: AccessAPIKey) => void
  onEditEntity: (kind: EntityDetailKind, item: EntityDetailValue) => void
  onDeleteEntity: (kind: EntityDetailKind, id: string) => void
  onEditModelAccess: (user: AccessUser) => void
}

const AccessControlPageOverlays = ({
  editor,
  createdKey,
  inviteOpen,
  detail,
  catalog,
  permissions,
  editorError,
  saving,
  onEditorChange,
  onEditorClose,
  onEditorSave,
  onCreatedKeyClose,
  onCreatedKeyDetails,
  onInviteClose,
  onInviteCreated,
  onDetailClose,
  onCatalogChanged,
  onDashboardMembersChanged,
  onEditKey,
  onEditEntity,
  onDeleteEntity,
  onEditModelAccess,
}: AccessControlPageOverlaysProps) => (
  <>
    {editor ? (
      <AccessControlDialog
        editor={editor}
        teams={catalog.teams}
        keys={catalog.keys}
        selectors={accessControlSelectorSources}
        selfService={permissions.selfService}
        selfUserId={permissions.selfUserId}
        error={editorError}
        saving={saving}
        onChange={onEditorChange}
        onClose={onEditorClose}
        onSave={onEditorSave}
      />
    ) : null}
    {createdKey ? (
      <AccessControlDialog
        secret={createdKey}
        onClose={onCreatedKeyClose}
        onViewDetails={() => onCreatedKeyDetails(createdKey.id)}
      />
    ) : null}
    <DashboardMemberInviteDialog
      isOpen={inviteOpen}
      roleOptions={['admin', 'write', 'read']}
      teamSource={accessControlSelectorSources.teams}
      onClose={onInviteClose}
      onCreated={onInviteCreated}
    />
    <AccessControlDetailOverlays
      detailKeyId={detail.keyId}
      detailLogId={detail.logId}
      detailEntityId={detail.entityId}
      detailMemberId={detail.memberId}
      entityKind={detail.entityKind}
      users={catalog.users}
      teams={catalog.teams}
      keys={catalog.keys}
      canManage={permissions.canManage}
      canRevealKeys={permissions.canRevealKeys}
      canManageDashboardMembers={permissions.canManageDashboardMembers}
      selfService={permissions.selfService}
      selfUserId={permissions.selfUserId}
      onClose={onDetailClose}
      onCatalogChanged={onCatalogChanged}
      onDashboardMembersChanged={onDashboardMembersChanged}
      onEditKey={onEditKey}
      onEditEntity={onEditEntity}
      onDeleteEntity={onDeleteEntity}
      onEditModelAccess={onEditModelAccess}
    />
  </>
)

export default AccessControlPageOverlays
