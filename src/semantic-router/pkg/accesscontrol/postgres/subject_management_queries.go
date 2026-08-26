package postgres

const (
	subjectUserColumns = `id, namespace_id, email, display_name,
       CASE WHEN deleted_at IS NULL THEN status ELSE 'deleted' END,
       revision, created_at, updated_at, deleted_at`
	subjectTeamColumns = `id, namespace_id, name, description, status,
       revision, created_at, updated_at, deleted_at`
	subjectMembershipColumns = `namespace_id, team_id, user_id, role, status,
       revision, created_at, updated_at`

	subjectGetUserQuery = `SELECT ` + subjectUserColumns + ` FROM access_users
WHERE namespace_id = $1 AND id = $2`
	subjectListUsersQuery = `SELECT ` + subjectUserColumns + ` FROM access_users
WHERE namespace_id = $1
  AND (($2 = '' AND deleted_at IS NULL)
       OR ($2 <> '' AND CASE WHEN deleted_at IS NULL THEN status ELSE 'deleted' END = $2))
  AND ($3 OR id = ANY($4::uuid[]))
  AND ($5::timestamptz IS NULL OR created_at < $5 OR (created_at = $5 AND id > $6::uuid))
ORDER BY created_at DESC, id ASC
LIMIT $7`
	subjectSearchUsersQuery = `SELECT ` + subjectUserColumns + ` FROM access_users
WHERE namespace_id = $1
  AND (($2 = '' AND deleted_at IS NULL)
       OR ($2 <> '' AND CASE WHEN deleted_at IS NULL THEN status ELSE 'deleted' END = $2))
  AND ($3 OR id = ANY($4::uuid[]))
  AND (lower(email) LIKE $5 ESCAPE E'\\' OR lower(display_name) LIKE $5 ESCAPE E'\\'
       OR id::text LIKE $5 ESCAPE E'\\')
  AND ($6::timestamptz IS NULL OR created_at < $6 OR (created_at = $6 AND id > $7::uuid))
ORDER BY created_at DESC, id ASC
LIMIT $8`
	subjectInsertUserQuery = `INSERT INTO access_users
  (id, namespace_id, email, display_name, status, revision, created_at, updated_at)
VALUES ($1,$2,$3,$4,'active',1,$5,$5)
RETURNING ` + subjectUserColumns
	subjectUpdateUserQuery = `UPDATE access_users
SET email = $4, display_name = $5, status = $6,
    revision = revision + 1, updated_at = clock_timestamp()
WHERE namespace_id = $1 AND id = $2 AND revision = $3 AND deleted_at IS NULL
RETURNING ` + subjectUserColumns
	subjectDeleteUserQuery = `UPDATE access_users
SET status = 'disabled', deleted_at = clock_timestamp(),
    revision = revision + 1, updated_at = clock_timestamp()
WHERE namespace_id = $1 AND id = $2 AND revision = $3 AND deleted_at IS NULL
RETURNING ` + subjectUserColumns
	subjectDisableUserMembershipsQuery = `UPDATE access_team_memberships
SET status = 'disabled', revision = revision + 1, updated_at = clock_timestamp()
WHERE namespace_id = $1 AND user_id = $2 AND status = 'active'`

	subjectGetTeamQuery = `SELECT ` + subjectTeamColumns + ` FROM access_teams
WHERE namespace_id = $1 AND id = $2`
	subjectListTeamsQuery = `SELECT ` + subjectTeamColumns + ` FROM access_teams
WHERE namespace_id = $1 AND ($2 = '' OR status = $2)
  AND ($3 OR id = ANY($4::uuid[]))
  AND ($5::timestamptz IS NULL OR created_at < $5 OR (created_at = $5 AND id > $6::uuid))
ORDER BY created_at DESC, id ASC
LIMIT $7`
	subjectSearchTeamsQuery = `SELECT ` + subjectTeamColumns + ` FROM access_teams
WHERE namespace_id = $1 AND ($2 = '' OR status = $2)
  AND ($3 OR id = ANY($4::uuid[]))
  AND (lower(name) LIKE $5 ESCAPE E'\\' OR id::text LIKE $5 ESCAPE E'\\')
  AND ($6::timestamptz IS NULL OR created_at < $6 OR (created_at = $6 AND id > $7::uuid))
ORDER BY created_at DESC, id ASC
LIMIT $8`
	subjectResolveTeamDefaultsQuery = `SELECT sp.revision,
       ap.id, ap.revision, rp.id, rp.revision
FROM self_service_policies sp
JOIN access_policies ap ON ap.namespace_id = sp.namespace_id
  AND ap.id = sp.default_access_policy_id AND ap.status = 'active'
JOIN rate_limit_policies rp ON rp.namespace_id = sp.namespace_id
  AND rp.id = sp.default_rate_limit_policy_id AND rp.status = 'active'
WHERE sp.namespace_id = $1`
	subjectLockTeamDefaultsQuery = subjectResolveTeamDefaultsQuery + ` FOR UPDATE OF sp, ap, rp`
	subjectInsertTeamQuery       = `INSERT INTO access_teams
  (id, namespace_id, name, description, status, revision, created_at, updated_at)
VALUES ($1,$2,$3,$4,'active',1,$5,$5)
RETURNING ` + subjectTeamColumns
	subjectInsertAccessBindingQuery = `INSERT INTO access_policy_bindings
  (id, namespace_id, policy_id, subject_id, status, revision, created_at, updated_at)
VALUES ($1,$2,$3,$4,'active',1,$5,$5)`
	subjectLockSelectedAccessPoliciesQuery = `SELECT id FROM access_policies
WHERE namespace_id = $1 AND id = ANY($2::uuid[]) AND status = 'active'
ORDER BY id
FOR SHARE`
	subjectLockSelectedRatePolicyQuery = `SELECT id FROM rate_limit_policies
WHERE namespace_id = $1 AND id = $2 AND status = 'active'
FOR SHARE`
	subjectInsertRateBindingQuery = `INSERT INTO rate_limit_bindings
  (id, namespace_id, policy_id, subject_id, binding_mode, quota_partition_id,
   status, revision, created_at, updated_at)
SELECT $1,$2,$3,$4,'allocation',quota_partition_id,'active',1,$5,$5
FROM access_namespaces WHERE id = $2 AND status = 'active'`
	subjectUpdateTeamQuery = `UPDATE access_teams
SET name = $4, description = $5, status = $6,
    revision = revision + 1, updated_at = clock_timestamp()
WHERE namespace_id = $1 AND id = $2 AND revision = $3 AND deleted_at IS NULL
RETURNING ` + subjectTeamColumns
	subjectDeleteTeamQuery = `UPDATE access_teams
SET status = 'disabled', deleted_at = clock_timestamp(),
    revision = revision + 1, updated_at = clock_timestamp()
WHERE namespace_id = $1 AND id = $2 AND revision = $3 AND deleted_at IS NULL
RETURNING ` + subjectTeamColumns
	subjectDisableTeamMembershipsQuery = `UPDATE access_team_memberships
SET status = 'disabled', revision = revision + 1, updated_at = clock_timestamp()
WHERE namespace_id = $1 AND team_id = $2 AND status = 'active'`
	subjectDisableTeamAccessBindingsQuery = `UPDATE access_policy_bindings
SET status = 'disabled', revision = revision + 1, updated_at = clock_timestamp()
WHERE namespace_id = $1 AND subject_id = $2 AND status = 'active'`
	subjectDisableTeamRateBindingsQuery = `UPDATE rate_limit_bindings
SET status = 'disabled', revision = revision + 1, updated_at = clock_timestamp()
WHERE namespace_id = $1 AND subject_id = $2 AND status = 'active'`

	subjectGetMembershipQuery = `SELECT ` + subjectMembershipColumns + ` FROM access_team_memberships
WHERE namespace_id = $1 AND team_id = $2 AND user_id = $3`
	subjectListUserMembershipsQuery = `SELECT m.namespace_id, m.team_id, m.user_id, m.role, m.status,
       m.revision, m.created_at, m.updated_at, t.name, t.status
FROM access_team_memberships m
JOIN access_teams t ON t.namespace_id = m.namespace_id AND t.id = m.team_id
WHERE m.namespace_id = $1 AND m.user_id = $2
  AND ($3 = '' OR m.status = $3)
  AND ($4 OR m.team_id = ANY($5::uuid[]))
  AND ($6::timestamptz IS NULL OR m.created_at < $6 OR (m.created_at = $6 AND m.team_id > $7::uuid))
ORDER BY m.created_at DESC, m.team_id ASC
LIMIT $8`
	subjectCountUserMembershipsQuery = `SELECT count(*)
FROM access_team_memberships m
WHERE m.namespace_id = $1 AND m.user_id = $2
  AND ($3 = '' OR m.status = $3)
  AND ($4 OR m.team_id = ANY($5::uuid[]))`
	subjectListTeamMembersQuery = `SELECT m.namespace_id, m.team_id, m.user_id, m.role, m.status,
	       m.revision, m.created_at, m.updated_at, u.display_name, u.email,
	       CASE WHEN u.deleted_at IS NULL THEN u.status ELSE 'disabled' END
FROM access_team_memberships m
JOIN access_users u ON u.namespace_id = m.namespace_id AND u.id = m.user_id
WHERE m.namespace_id = $1 AND m.team_id = $2
  AND ($3 = '' OR m.status = $3)
  AND ($4 OR m.user_id = ANY($5::uuid[]))
  AND ($6::timestamptz IS NULL OR m.created_at < $6 OR (m.created_at = $6 AND m.user_id > $7::uuid))
ORDER BY m.created_at DESC, m.user_id ASC
LIMIT $8`
	subjectCountTeamMembersQuery = `SELECT count(*)
FROM access_team_memberships m
WHERE m.namespace_id = $1 AND m.team_id = $2
  AND ($3 = '' OR m.status = $3)
  AND ($4 OR m.user_id = ANY($5::uuid[]))`
	subjectCheckMembershipParentsQuery = `SELECT EXISTS (
  SELECT 1 FROM access_teams WHERE namespace_id = $1 AND id = $2
    AND status = 'active' AND deleted_at IS NULL
), EXISTS (
  SELECT 1 FROM access_users WHERE namespace_id = $1 AND id = $3
    AND status = 'active' AND deleted_at IS NULL
)`
	subjectInsertMembershipQuery = `INSERT INTO access_team_memberships
  (namespace_id, team_id, user_id, role, status, revision, created_at, updated_at)
VALUES ($1,$2,$3,$4,'active',1,$5,$5)
RETURNING ` + subjectMembershipColumns
	subjectUpdateMembershipQuery = `UPDATE access_team_memberships
SET role = $5, status = $6, revision = revision + 1,
    updated_at = clock_timestamp()
WHERE namespace_id = $1 AND team_id = $2 AND user_id = $3 AND revision = $4
RETURNING ` + subjectMembershipColumns
	subjectDeleteMembershipQuery = `UPDATE access_team_memberships
SET status = 'disabled', revision = revision + 1, updated_at = clock_timestamp()
WHERE namespace_id = $1 AND team_id = $2 AND user_id = $3 AND revision = $4
  AND status = 'active'
RETURNING ` + subjectMembershipColumns
)
