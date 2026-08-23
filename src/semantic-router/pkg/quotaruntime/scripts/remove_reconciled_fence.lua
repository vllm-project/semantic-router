local fence_id = ARGV[1]
local reconciliation_id = ARGV[2]
local plan_digest = ARGV[3]
local marker_ttl_ms = tonumber(ARGV[4])
local now = quota_time_milliseconds()

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
for index = 2, #KEYS do
  redis.call("SREM", KEYS[index], fence_id)
end
redis.call("HSET", KEYS[1], "state", "released", "released_at", string.format("%.0f", now))
redis.call("PEXPIRE", KEYS[1], marker_ttl_ms)
return {"released", "0", string.format("%.0f", now)}
