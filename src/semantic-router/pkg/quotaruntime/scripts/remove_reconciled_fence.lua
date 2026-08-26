local fence_id = ARGV[1]
local reconciliation_id = ARGV[2]
local plan_digest = ARGV[3]
local marker_ttl_ms = tonumber(ARGV[4])
local counter_count = tonumber(ARGV[5])
local now = quota_time_milliseconds()

local function quota_key_has_type(key, expected)
  local observed = redis.call("TYPE", key)
  if type(observed) == "table" then observed = observed.ok end
  return observed == "none" or observed == expected
end

if counter_count == nil or counter_count < 0 or counter_count > 4096
    or counter_count ~= math.floor(counter_count)
    or #KEYS ~= 1 + counter_count * 2 or #ARGV ~= 5 then
  return redis.error_reply("QUOTA_INVALID fence release shape")
end
if not quota_key_has_type(KEYS[1], "hash") then
  return redis.error_reply("QUOTA_CORRUPT fence release marker key type")
end

local state = redis.call("HGET", KEYS[1], "state")
if state == "released" then
  if redis.call("HGET", KEYS[1], "reconciliation_id") == reconciliation_id
      and redis.call("HGET", KEYS[1], "reconciliation_digest") == plan_digest then
    return {"released", "1", string.format("%.0f", now)}
  end
  return redis.error_reply("QUOTA_CONFLICT released fence reconciliation differs")
end
if state ~= "corrected"
    or redis.call("HGET", KEYS[1], "reconciliation_id") ~= reconciliation_id
    or redis.call("HGET", KEYS[1], "reconciliation_digest") ~= plan_digest then
  return redis.error_reply("QUOTA_CONFLICT fence correction is not durable")
end
if marker_ttl_ms == nil or marker_ttl_ms <= 0 then
  return redis.error_reply("QUOTA_INVALID reconciliation marker TTL")
end

local binding_sets = {}
for index = 1, counter_count do
  local key_offset = 1 + (index - 1) * 2
  local binding_fences_key = KEYS[key_offset + 1]
  local meta_key = KEYS[key_offset + 2]
  if not quota_key_has_type(binding_fences_key, "set")
      or not quota_key_has_type(meta_key, "hash") then
    return redis.error_reply("QUOTA_CORRUPT fence release counter key type")
  end
  if redis.call("HEXISTS", KEYS[1], "binding:" .. binding_fences_key) ~= 1
      or redis.call("HEXISTS", KEYS[1], "rule:" .. meta_key) ~= 1
      or redis.call("SISMEMBER", binding_fences_key, fence_id) ~= 1 then
    return redis.error_reply("QUOTA_CONFLICT fence release counter differs")
  end
  local incomplete = redis.call("HGET", meta_key, "incomplete") or "0"
  if quota_parse(incomplete) == nil then
    return redis.error_reply("QUOTA_CORRUPT invalid incomplete counter")
  end
  -- Validate before any SREM. Redis scripts do not roll back writes after an
  -- error, so this preflight keeps the fence and counter projection atomic.
  if redis.call("SCARD", binding_fences_key) == 1
      and quota_compare(incomplete, "0") ~= 0 then
    return redis.error_reply("QUOTA_CORRUPT last fence has incomplete usage")
  end
  binding_sets[binding_fences_key] = true
end
for binding_fences_key, _ in pairs(binding_sets) do
  redis.call("SREM", binding_fences_key, fence_id)
end
redis.call("HSET", KEYS[1], "state", "released", "released_at", string.format("%.0f", now))
redis.call("PEXPIRE", KEYS[1], marker_ttl_ms)
return {"released", "0", string.format("%.0f", now)}
