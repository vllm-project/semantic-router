package accesspublisher

import "github.com/redis/go-redis/v9"

var registerNamespaceScript = redis.NewScript(`
local existing = redis.call('HGET', KEYS[1], ARGV[1])
if existing then
  if existing ~= ARGV[2] then
    return redis.error_reply('NAMESPACE_PARTITION_CONFLICT')
  end
  return 0
end
if redis.call('HLEN', KEYS[1]) >= tonumber(ARGV[3]) then
  return redis.error_reply('NAMESPACE_DIRECTORY_FULL')
end
redis.call('HSET', KEYS[1], ARGV[1], ARGV[2])
return 1
`)

var preparePublicationScript = redis.NewScript(`
local epoch = redis.call('GET', KEYS[1])
if not epoch then
  redis.call('SET', KEYS[1], ARGV[1])
elseif epoch ~= ARGV[1] then
  return redis.error_reply('EPOCH_MISMATCH')
end
local access_gate = redis.call('HGET', KEYS[2], 'publication_id') or ''
local routing_gate = redis.call('HGET', KEYS[3], 'publication_id') or ''
if access_gate ~= ARGV[2] or routing_gate ~= ARGV[3] then
  return redis.error_reply('GATE_CONFLICT')
end
if access_gate ~= '' and access_gate ~= ARGV[4] then
  local prior_state = redis.call('HGET', KEYS[6], 'state') or ''
  if prior_state ~= 'compacted' and prior_state ~= 'applied' and prior_state ~= 'finalized' then
    return redis.error_reply('PRIOR_NOT_COMPACTED')
  end
end
local pending_id = redis.call('HGET', KEYS[7], 'publication_id') or ''
local pending_revision = tonumber(redis.call('HGET', KEYS[7], 'revision') or '0')
local desired_revision = tonumber(ARGV[8])
if pending_id ~= '' and pending_id ~= ARGV[4] then
  if pending_revision > desired_revision then
    return redis.error_reply('HEAD_SUPERSEDED')
  end
  if pending_revision == desired_revision then
    return redis.error_reply('PUBLICATION_CONFLICT')
  end
end
local existing = redis.call('HGET', KEYS[4], 'publication_digest')
if existing then
  if existing ~= ARGV[5]
     or (redis.call('HGET', KEYS[4], 'plan') or '') ~= ARGV[11]
     or (redis.call('HGET', KEYS[4], 'namespace_id') or '') ~= ARGV[6]
     or (redis.call('HGET', KEYS[4], 'quota_partition') or '') ~= ARGV[7]
     or (redis.call('HGET', KEYS[4], 'desired_revision') or '') ~= ARGV[8]
     or (redis.call('HGET', KEYS[4], 'runtime_epoch') or '') ~= ARGV[1] then
    return redis.error_reply('PUBLICATION_CONFLICT')
  end
  redis.call('HSET', KEYS[7], 'publication_id', ARGV[4], 'revision', ARGV[8], 'digest', ARGV[5])
  return 0
end
redis.call('HSET', KEYS[4],
  'publication_id', ARGV[4],
  'publication_digest', ARGV[5],
  'namespace_id', ARGV[6],
  'quota_partition', ARGV[7],
  'desired_revision', ARGV[8],
  'runtime_epoch', ARGV[1],
  'manifest_digest', ARGV[9],
  'routing_digest', ARGV[10],
  'prior_access_gate', ARGV[2],
  'prior_routing_gate', ARGV[3],
  'plan', ARGV[11],
  'state', 'prepared',
  'compact_cursor', '0')
redis.call('ZADD', KEYS[5], ARGV[8], ARGV[4])
redis.call('HSET', KEYS[7], 'publication_id', ARGV[4], 'revision', ARGV[8], 'digest', ARGV[5])
return 1
`)

var putImmutableStringScript = redis.NewScript(`
local existing = redis.call('GET', KEYS[1])
if existing then
  if existing ~= ARGV[1] then
    return redis.error_reply('IMMUTABLE_CONFLICT')
  end
  return 0
end
redis.call('SET', KEYS[1], ARGV[1])
return 1
`)

var putImmutableHashScript = redis.NewScript(`
local existing_digest = redis.call('HGET', KEYS[1], 'digest')
if existing_digest then
  if existing_digest ~= ARGV[1] or (redis.call('HGET', KEYS[1], 'document') or '') ~= ARGV[2] then
    return redis.error_reply('IMMUTABLE_CONFLICT')
  end
  return 0
end
if redis.call('EXISTS', KEYS[1]) ~= 0 then
  return redis.error_reply('IMMUTABLE_CONFLICT')
end
redis.call('HSET', KEYS[1],
  'digest', ARGV[1], 'document', ARGV[2],
  'namespace_id', ARGV[3], 'publication_id', ARGV[4], 'revision', ARGV[5])
return 1
`)

var installOneBarrierScript = redis.NewScript(`
redis.call('SADD', KEYS[1], ARGV[1])
redis.call('SADD', KEYS[2], KEYS[1])
return 1
`)

// stagePointerScript accepts field/value pairs whose names all start with
// pending_. A strictly newer revision may replace an older staged pointer so a
// coalesced full publication can supersede abandoned work without rewriting
// the active fields. Once a publication has already been promoted, a
// byte-identical retry is a no-op.
var stagePointerScript = redis.NewScript(`
local pending = redis.call('HGET', KEYS[1], 'pending_publication_id')
if pending and pending ~= ARGV[1] then
  local next_revision = 0
  for index = 2, #ARGV, 2 do
    if ARGV[index] == 'pending_revision' then
      next_revision = tonumber(ARGV[index + 1]) or 0
    end
  end
  local pending_revision = tonumber(redis.call('HGET', KEYS[1], 'pending_revision') or '0')
  if pending_revision == 0 or next_revision == 0 or pending_revision >= next_revision then
    return redis.error_reply('POINTER_CONFLICT')
  end
  local values = redis.call('HKEYS', KEYS[1])
  for _, field in ipairs(values) do
    if string.sub(field, 1, 8) == 'pending_' then
      redis.call('HDEL', KEYS[1], field)
    end
  end
  pending = nil
end
if not pending and (redis.call('HGET', KEYS[1], 'publication_id') or '') == ARGV[1] then
  return 2
end
for index = 2, #ARGV, 2 do
  local field = ARGV[index]
  local value = ARGV[index + 1]
  if string.sub(field, 1, 8) ~= 'pending_' then
    return redis.error_reply('INVALID_PENDING_FIELD')
  end
  local existing = redis.call('HGET', KEYS[1], field)
  if existing and existing ~= value then
    return redis.error_reply('POINTER_CONFLICT')
  end
end
for index = 2, #ARGV, 2 do
  redis.call('HSET', KEYS[1], ARGV[index], ARGV[index + 1])
end
return 1
`)

var installBarriersScript = redis.NewScript(`
local now = redis.call('TIME')
local now_ms = tonumber(now[1]) * 1000 + math.floor(tonumber(now[2]) / 1000)
redis.call('ZREMRANGEBYSCORE', KEYS[2], '-inf', now_ms)
local replicas = redis.call('ZRANGE', KEYS[2], 0, -1)
for _, replica in ipairs(replicas) do
  redis.call('SADD', KEYS[3], replica)
end
local state = redis.call('HGET', KEYS[1], 'state') or ''
if state == 'prepared' or state == 'barriers_installed' then
  redis.call('HSET', KEYS[1], 'state', 'barriers_installed', 'barrier_count', ARGV[1])
elseif state ~= 'staged' and state ~= 'validated' and state ~= 'active'
   and state ~= 'compacted' and state ~= 'applied' and state ~= 'finalized' then
  return redis.error_reply('PUBLICATION_STATE_CONFLICT')
end
return #replicas
`)

var finalizeStageScript = redis.NewScript(`
local now = redis.call('TIME')
local now_ms = tonumber(now[1]) * 1000 + math.floor(tonumber(now[2]) / 1000)
redis.call('ZREMRANGEBYSCORE', KEYS[2], '-inf', now_ms)
local replicas = redis.call('ZRANGE', KEYS[2], 0, -1)
for _, replica in ipairs(replicas) do
  redis.call('SADD', KEYS[3], replica)
end
for index = 3, #ARGV do
  redis.call('SADD', KEYS[3], ARGV[index])
end
local state = redis.call('HGET', KEYS[1], 'state') or ''
if ARGV[2] == '1' and state == 'prepared' then
  return redis.error_reply('BARRIERS_REQUIRED')
end
if state == 'prepared' or state == 'barriers_installed' or state == 'staged' then
  redis.call('HSET', KEYS[1], 'state', 'staged', 'pointer_count', ARGV[1])
elseif state ~= 'validated' and state ~= 'active' and state ~= 'compacted'
   and state ~= 'applied' and state ~= 'finalized' then
  return redis.error_reply('PUBLICATION_STATE_CONFLICT')
end
return #replicas
`)

var registerFleetReplicaScript = redis.NewScript(`
local now = redis.call('TIME')
local now_ms = tonumber(now[1]) * 1000 + math.floor(tonumber(now[2]) / 1000)
local expires = now_ms + tonumber(ARGV[2])
redis.call('ZADD', KEYS[1], expires, ARGV[1])
return tostring(expires)
`)

var liveFleetReplicasScript = redis.NewScript(`
local now = redis.call('TIME')
local now_ms = tonumber(now[1]) * 1000 + math.floor(tonumber(now[2]) / 1000)
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now_ms)
return redis.call('ZRANGE', KEYS[1], 0, -1)
`)

var replaceRequiredReplicasScript = redis.NewScript(`
redis.call('DEL', KEYS[1])
for index = 1, #ARGV do
  redis.call('SADD', KEYS[1], ARGV[index])
end
return #ARGV
`)

var markValidatedScript = redis.NewScript(`
local state = redis.call('HGET', KEYS[1], 'state') or ''
if state == 'staged' then
  redis.call('HSET', KEYS[1], 'state', 'validated', 'validated_digest', ARGV[1])
  return 1
end
if state == 'validated' or state == 'active' or state == 'compacted' or state == 'applied' or state == 'finalized' then
  if (redis.call('HGET', KEYS[1], 'validated_digest') or '') ~= ARGV[1] then
    return redis.error_reply('VALIDATION_CONFLICT')
  end
  return 0
end
return redis.error_reply('PUBLICATION_STATE_CONFLICT')
`)

var registerReplicaScript = redis.NewScript(`
local epoch = redis.call('GET', KEYS[1]) or ''
if epoch ~= ARGV[2] then
  return redis.error_reply('EPOCH_MISMATCH')
end
local access_gate = redis.call('HGET', KEYS[2], 'publication_id') or ''
local routing_gate = redis.call('HGET', KEYS[3], 'publication_id') or ''
if access_gate ~= ARGV[3] or routing_gate ~= ARGV[4] then
  return redis.error_reply('GATE_CONFLICT')
end
local now = redis.call('TIME')
local now_ms = tonumber(now[1]) * 1000 + math.floor(tonumber(now[2]) / 1000)
local expires = now_ms + tonumber(ARGV[5])
redis.call('HSET', KEYS[4],
  'replica_id', ARGV[1], 'runtime_epoch', ARGV[2],
  'access_publication', ARGV[3], 'routing_publication', ARGV[4],
  'lease_expires_at_ms', tostring(expires))
redis.call('ZADD', KEYS[5], expires, ARGV[1])
return tostring(expires)
`)

var acknowledgeScript = redis.NewScript(`
local now = redis.call('TIME')
local now_ms = tonumber(now[1]) * 1000 + math.floor(tonumber(now[2]) / 1000)
local score = redis.call('ZSCORE', KEYS[1], ARGV[1])
if not score or tonumber(score) <= now_ms then
  return redis.error_reply('REPLICA_LEASE_EXPIRED')
end
if (redis.call('HGET', KEYS[2], 'publication_digest') or '') ~= ARGV[3] then
  return redis.error_reply('PUBLICATION_CONFLICT')
end
local state = redis.call('HGET', KEYS[2], 'state') or ''
if state ~= 'barriers_installed' and state ~= 'staged' and state ~= 'validated'
   and state ~= 'active' and state ~= 'compacted' and state ~= 'applied' and state ~= 'finalized' then
  return redis.error_reply('PUBLICATION_STATE_CONFLICT')
end
redis.call('SADD', KEYS[3], ARGV[1])
redis.call('SADD', KEYS[4], ARGV[1])
return 1
`)

var activatePublicationScript = redis.NewScript(`
local epoch = redis.call('GET', KEYS[1]) or ''
if epoch ~= ARGV[1] then
  return redis.error_reply('EPOCH_MISMATCH')
end
local current_access = redis.call('HGET', KEYS[2], 'publication_id') or ''
local current_routing = redis.call('HGET', KEYS[3], 'publication_id') or ''
if current_access == ARGV[4] and current_routing == ARGV[4] then
  return 0
end
if current_access ~= ARGV[2] or current_routing ~= ARGV[3] then
  return redis.error_reply('GATE_CONFLICT')
end
if (redis.call('HGET', KEYS[9], 'publication_id') or '') ~= ARGV[4]
   or (redis.call('HGET', KEYS[9], 'revision') or '') ~= ARGV[6] then
  return redis.error_reply('HEAD_SUPERSEDED')
end
if (redis.call('HGET', KEYS[4], 'state') or '') ~= 'validated'
   or (redis.call('HGET', KEYS[4], 'validated_digest') or '') ~= ARGV[5] then
  return redis.error_reply('PUBLICATION_NOT_VALIDATED')
end
local now = redis.call('TIME')
local now_ms = tonumber(now[1]) * 1000 + math.floor(tonumber(now[2]) / 1000)
redis.call('ZREMRANGEBYSCORE', KEYS[5], '-inf', now_ms)
local replicas = redis.call('ZRANGE', KEYS[5], 0, -1)
for _, replica in ipairs(replicas) do
  redis.call('SADD', KEYS[6], replica)
end
local required = redis.call('SMEMBERS', KEYS[6])
for _, replica in ipairs(required) do
  if ARGV[10] == '1' and redis.call('SISMEMBER', KEYS[7], replica) == 0 then
    return redis.error_reply('BARRIER_ACK_INCOMPLETE')
  end
  if redis.call('SISMEMBER', KEYS[8], replica) == 0 then
    return redis.error_reply('ROUTING_ACK_INCOMPLETE')
  end
end
redis.call('HSET', KEYS[2],
  'publication_id', ARGV[4], 'revision', ARGV[6], 'runtime_epoch', ARGV[1],
  'publication_digest', ARGV[5], 'manifest_digest', ARGV[7])
redis.call('HSET', KEYS[3],
  'publication_id', ARGV[4], 'revision', ARGV[6], 'runtime_epoch', ARGV[1],
  'publication_digest', ARGV[5], 'snapshot_digest', ARGV[8], 'snapshot_key', ARGV[9])
redis.call('HSET', KEYS[4], 'state', 'active', 'activated_at_ms', tostring(now_ms))
return #required
`)

// Promotion is intentionally per pointer. The namespace publication gate is
// already active, so runtime readers select pending fields until every pointer
// has been promoted. A second publication cannot activate while this one is
// not compacted.
var promotePointerScript = redis.NewScript(`
local pending = redis.call('HGET', KEYS[1], 'pending_publication_id')
if not pending then
  if (redis.call('HGET', KEYS[1], 'publication_id') or '') == ARGV[1] or redis.call('EXISTS', KEYS[1]) == 0 then
    return 0
  end
  return redis.error_reply('POINTER_CONFLICT')
end
if pending ~= ARGV[1] then
  return redis.error_reply('POINTER_CONFLICT')
end
local state = redis.call('HGET', KEYS[1], 'pending_state') or ''
if state == 'tombstone' then
  redis.call('DEL', KEYS[1])
  return 1
end
if state ~= 'active' then
  return redis.error_reply('POINTER_STATE_INVALID')
end
local values = redis.call('HGETALL', KEYS[1])
local promoted = {}
for index = 1, #values, 2 do
  local field = values[index]
  if string.sub(field, 1, 8) == 'pending_' then
    promoted[#promoted + 1] = string.sub(field, 9)
    promoted[#promoted + 1] = values[index + 1]
  end
end
redis.call('DEL', KEYS[1])
if #promoted > 0 then
  redis.call('HSET', KEYS[1], unpack(promoted))
end
return 1
`)

var finishCompactionScript = redis.NewScript(`
if (redis.call('HGET', KEYS[2], 'publication_id') or '') ~= ARGV[1]
   or (redis.call('HGET', KEYS[3], 'publication_id') or '') ~= ARGV[1] then
  return redis.error_reply('GATE_CONFLICT')
end
local state = redis.call('HGET', KEYS[1], 'state') or ''
if state == 'active' then
  redis.call('HSET', KEYS[1], 'state', 'compacted')
  return 1
end
if state == 'compacted' or state == 'applied' or state == 'finalized' then
  return 0
end
return redis.error_reply('PUBLICATION_STATE_CONFLICT')
`)

var markAppliedScript = redis.NewScript(`
if (redis.call('GET', KEYS[1]) or '') ~= ARGV[1] then
  return redis.error_reply('EPOCH_MISMATCH')
end
if (redis.call('HGET', KEYS[2], 'publication_id') or '') ~= ARGV[2]
   or (redis.call('HGET', KEYS[3], 'publication_id') or '') ~= ARGV[2] then
  return redis.error_reply('GATE_CONFLICT')
end
local state = redis.call('HGET', KEYS[4], 'state') or ''
if state ~= 'compacted' and state ~= 'applied' and state ~= 'finalized' then
  return redis.error_reply('PUBLICATION_STATE_CONFLICT')
end
local previous_revision = tonumber(redis.call('HGET', KEYS[5], 'desired_revision') or '0')
if previous_revision > tonumber(ARGV[3]) then
  return redis.error_reply('APPLIED_REVISION_REGRESSION')
end
if previous_revision == tonumber(ARGV[3]) then
  local previous_publication = redis.call('HGET', KEYS[5], 'publication_id') or ''
  if previous_publication ~= '' and previous_publication ~= ARGV[2] then
    return redis.error_reply('APPLIED_REVISION_CONFLICT')
  end
end
local now = redis.call('TIME')
local now_ms = tonumber(now[1]) * 1000 + math.floor(tonumber(now[2]) / 1000)
redis.call('HSET', KEYS[5],
  'namespace_id', ARGV[4], 'runtime_epoch', ARGV[1],
  'desired_revision', ARGV[3], 'publication_id', ARGV[2],
  'access_digest', ARGV[5], 'routing_digest', ARGV[6],
  'applied_at_ms', tostring(now_ms))
redis.call('HSET', KEYS[4], 'state', 'applied', 'applied_at_ms', tostring(now_ms))
return 1
`)

var clearPendingPublicationScript = redis.NewScript(`
if (redis.call('HGET', KEYS[1], 'publication_id') or '') == ARGV[1] then
  redis.call('DEL', KEYS[1])
  return 1
end
return 0
`)

var clearOneBarrierScript = redis.NewScript(`
redis.call('SREM', KEYS[1], ARGV[1])
if redis.call('SCARD', KEYS[1]) == 0 then
  redis.call('DEL', KEYS[1])
end
return 1
`)
