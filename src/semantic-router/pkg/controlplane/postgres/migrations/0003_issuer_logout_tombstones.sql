-- Durable issuer logout selectors close the interval in which a valid
-- back-channel logout can arrive before the corresponding Management session.
-- Selectors are domain-separated SHA-256 digests; the upstream SID or subject
-- is never retained in this table.
CREATE TABLE management_issuer_logout_tombstones (
  issuer_id UUID NOT NULL REFERENCES trusted_identity_issuers(id) ON DELETE CASCADE,
  selector_kind TEXT NOT NULL CHECK (selector_kind IN ('sid','subject')),
  selector_digest BYTEA NOT NULL CHECK (octet_length(selector_digest) = 32),
  logout_issued_at TIMESTAMPTZ NOT NULL,
  logout_expires_at TIMESTAMPTZ NOT NULL,
  installed_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY (issuer_id, selector_kind, selector_digest),
  CHECK (logout_expires_at > logout_issued_at)
);
