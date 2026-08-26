-- Audit actor chains are append-only JSON arrays. Keep the shape invariant in
-- PostgreSQL so every writer shares the same durable contract.

ALTER TABLE access_audit_events
  ADD CONSTRAINT access_audit_events_actor_chain_array_ck
  CHECK (jsonb_typeof(actor_chain) = 'array');
