package postgres

const liveSessionQuery = `SELECT
  s.id::text,
  s.principal_id::text,
  s.issuer_session_id,
  s.token_id,
  s.audience,
  s.auth_source_kind,
  s.auth_source_id::text,
  s.evidence_kind,
  s.assurance,
  s.authenticated_at,
  s.source_assured_at,
  s.expires_at,
  s.status,
  s.revoked_at,
  s.created_at,
  p.status,
  CASE
    WHEN s.auth_source_kind = 'issuer'
      AND i.id IS NOT NULL AND i.status = 'active' AND p.issuer = i.issuer THEN 'active'
    WHEN s.auth_source_kind = 'service_credential'
      AND c.id IS NOT NULL AND c.status IN ('active', 'retiring')
      AND a.status = 'active' AND a.principal_id = s.principal_id THEN 'active'
    WHEN s.auth_source_kind = 'mtls'
      AND m.id IS NOT NULL AND m.status = 'active' AND m.principal_id = s.principal_id THEN 'active'
    ELSE NULL
  END AS auth_source_status,
	CASE WHEN s.auth_source_kind = 'service_credential' THEN c.not_before END,
  CASE WHEN s.auth_source_kind = 'service_credential' THEN c.expires_at END,
  CASE
    WHEN s.auth_source_kind = 'service_credential' THEN c.source_assured_at
    WHEN s.auth_source_kind = 'mtls' THEN m.source_assured_at
  END
FROM management_sessions s
JOIN management_principals p ON p.id = s.principal_id
LEFT JOIN trusted_identity_issuers i
  ON s.auth_source_kind = 'issuer' AND i.id = s.auth_source_id
LEFT JOIN management_service_account_credentials c
  ON s.auth_source_kind = 'service_credential' AND c.id = s.auth_source_id
LEFT JOIN management_service_accounts a ON a.id = c.service_account_id
LEFT JOIN management_mtls_mappings m
  ON s.auth_source_kind = 'mtls' AND m.id = s.auth_source_id
WHERE s.id = $1`

const (
	lockPrincipalQuery = `SELECT status
FROM management_principals
WHERE id = $1
FOR UPDATE`

	loadSessionPolicyQuery = `SELECT
  clock_timestamp(),
  session_ttl_seconds,
  max_active_sessions
FROM management_session_policy
WHERE singleton = TRUE
FOR SHARE`

	expireSessionsQuery = `UPDATE management_sessions
SET status = 'expired'
WHERE principal_id = $1
  AND status = 'active'
  AND expires_at <= $2`

	countActiveSessionsQuery = `SELECT count(*)
FROM management_sessions
WHERE principal_id = $1
  AND status = 'active'
  AND expires_at > $2`

	insertSessionQuery = `INSERT INTO management_sessions (
  id,
  principal_id,
  issuer_session_id,
  token_id,
  audience,
  auth_source_kind,
  auth_source_id,
  evidence_kind,
  assurance,
  authenticated_at,
  source_assured_at,
  expires_at,
  status,
  revoked_at,
  created_at
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, 'active', NULL, $13)`

	lockSessionQuery = `SELECT token_id, status, expires_at, revoked_at, clock_timestamp()
FROM management_sessions
WHERE id = $1
FOR UPDATE`

	rotateSessionTokenIDQuery = `UPDATE management_sessions
SET token_id = $3
WHERE id = $1
  AND token_id = $2
  AND status = 'active'`

	lockIssuerSessionQuery = `SELECT id::text, clock_timestamp()
FROM management_sessions
WHERE principal_id = $1
  AND auth_source_kind = 'issuer'
  AND auth_source_id = $2
  AND issuer_session_id = $3
  AND audience = $4
  AND evidence_kind = 'human'
  AND authenticated_at = $5
  AND assurance = $6::jsonb
ORDER BY created_at DESC
LIMIT 1
FOR UPDATE`

	revokeSessionQuery = `UPDATE management_sessions
SET status = 'revoked', revoked_at = clock_timestamp()
WHERE id = $1
  AND token_id = $2
  AND status = 'active'
RETURNING revoked_at`
)
