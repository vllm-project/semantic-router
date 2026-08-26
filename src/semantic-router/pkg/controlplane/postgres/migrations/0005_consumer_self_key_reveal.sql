-- Consumer authority is bound to one user scope. Granting key.reveal here lets
-- a consumer recover only credentials for that user's own keys; API-key targets
-- owned by another user or by a Team are outside the binding scope.
DO $$
DECLARE
  updated_rows INTEGER;
BEGIN
  UPDATE management_roles
  SET permissions = '["access_policy.read","agent.read","agent.use","delegation.use","key.read","key.reveal","operation.read","quota.read","rate_policy.read","routing_context.read","team.read","tool.invoke","tool.read","usage.read","user.read"]'::jsonb,
      permissions_digest = decode('50e9622a3481c88a7177cf7288023193b43691cb26a923ceee09422344fa64a3', 'hex'),
      updated_at = clock_timestamp()
  WHERE id = '10000000-0000-5000-8000-000000000008'
    AND namespace_id IS NULL
    AND name = 'consumer'
    AND builtin = TRUE
    AND status = 'active'
    AND revision = 1
    AND permissions = '["access_policy.read","agent.read","agent.use","delegation.use","key.read","operation.read","quota.read","rate_policy.read","routing_context.read","team.read","tool.invoke","tool.read","usage.read","user.read"]'::jsonb
    AND permissions_digest = decode('42f87d9c0231abac6d6f5256f4c40d5d5789f9c9d8739c264785d0bf58560fd6', 'hex');

  GET DIAGNOSTICS updated_rows = ROW_COUNT;
  IF updated_rows = 0 AND NOT EXISTS (
    SELECT 1
    FROM management_roles
    WHERE id = '10000000-0000-5000-8000-000000000008'
      AND namespace_id IS NULL
      AND name = 'consumer'
      AND builtin = TRUE
      AND status = 'active'
      AND revision = 1
      AND permissions = '["access_policy.read","agent.read","agent.use","delegation.use","key.read","key.reveal","operation.read","quota.read","rate_policy.read","routing_context.read","team.read","tool.invoke","tool.read","usage.read","user.read"]'::jsonb
      AND permissions_digest = decode('50e9622a3481c88a7177cf7288023193b43691cb26a923ceee09422344fa64a3', 'hex')
  ) THEN
    RAISE EXCEPTION 'consumer built-in role cannot be advanced to the self-key reveal contract';
  END IF;
END
$$;
