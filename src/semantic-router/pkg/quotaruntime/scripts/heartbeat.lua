local admission_id = ARGV[1]
local admission_digest = ARGV[2]
local plan_digest = ARGV[3]
local lease_milliseconds = tonumber(ARGV[4])
local concurrency_count = tonumber(ARGV[5])
local now = quota_time_milliseconds()

local function quota_key_has_type(key, expected)
  local observed = redis.call("TYPE", key)
  if type(observed) == "table" then
    observed = observed.ok
  end
  return observed == "none" or observed == expected
end

if concurrency_count == nil or concurrency_count < 0
    or #KEYS ~= 3 + concurrency_count or #ARGV ~= 5 + concurrency_count then
  return redis.error_reply("QUOTA_INVALID admission heartbeat shape")
end
if lease_milliseconds == nil or lease_milliseconds < 1 then
  return redis.error_reply("QUOTA_INVALID admission heartbeat lease")
end
if not quota_key_has_type(KEYS[1], "zset")
    or not quota_key_has_type(KEYS[2], "hash")
    or not quota_key_has_type(KEYS[3], "hash") then
  return redis.error_reply("QUOTA_CORRUPT admission heartbeat base key type")
end

local terminal_state = redis.call("HGET", KEYS[3], "state")
if terminal_state ~= false then
  if terminal_state ~= "finalized"
      or redis.call("HGET", KEYS[3], "admission_digest") ~= admission_digest
      or redis.call("HGET", KEYS[3], "admission_plan_digest") ~= plan_digest then
    return redis.error_reply("QUOTA_CONFLICT terminal admission identity differs")
  end
  return {"stopped", string.format("%.0f", now), ""}
end

if redis.call("HGET", KEYS[2], "state") ~= "admitted" then
  return redis.error_reply("QUOTA_NOT_FOUND admission is not pending")
end
if redis.call("HGET", KEYS[2], "digest") ~= admission_digest
    or redis.call("HGET", KEYS[2], "plan_digest") ~= plan_digest then
  return redis.error_reply("QUOTA_CONFLICT admission heartbeat identity differs")
end
if tonumber(redis.call("HGET", KEYS[2], "lease_ms") or "-1") ~= lease_milliseconds then
  return redis.error_reply("QUOTA_CONFLICT admission heartbeat lease differs")
end
if tonumber(redis.call("HGET", KEYS[2], "concurrency_count") or "-1") ~= concurrency_count then
  return redis.error_reply("QUOTA_CONFLICT admission heartbeat concurrency rules differ")
end

local old_deadline = tonumber(redis.call("HGET", KEYS[2], "deadline") or "")
local index_deadline = tonumber(redis.call("ZSCORE", KEYS[1], admission_id) or "")
if old_deadline == nil or index_deadline == nil or old_deadline ~= index_deadline then
  return redis.error_reply("QUOTA_CORRUPT admission heartbeat deadline differs")
end
if now >= old_deadline then
  return redis.error_reply("QUOTA_CONFLICT expired admission cannot be renewed")
end

for index = 1, concurrency_count do
  local event_key = KEYS[3 + index]
  if not quota_key_has_type(event_key, "zset") then
    return redis.error_reply("QUOTA_CORRUPT admission heartbeat concurrency key type")
  end
  if redis.call("HGET", KEYS[2], "concurrency:" .. event_key) ~= ARGV[5 + index] then
    return redis.error_reply("QUOTA_CONFLICT admission heartbeat concurrency fingerprint differs")
  end
  local concurrency_deadline = tonumber(redis.call("ZSCORE", event_key, admission_id) or "")
  if concurrency_deadline == nil or concurrency_deadline ~= old_deadline then
    return redis.error_reply("QUOTA_CORRUPT admission heartbeat concurrency lease differs")
  end
end

local new_deadline = now + lease_milliseconds
if new_deadline < old_deadline then
  new_deadline = old_deadline
end
redis.call("HSET", KEYS[2],
  "heartbeat_at", string.format("%.0f", now),
  "deadline", string.format("%.0f", new_deadline))
redis.call("ZADD", KEYS[1], new_deadline, admission_id)
for index = 1, concurrency_count do
  redis.call("ZADD", KEYS[3 + index], new_deadline, admission_id)
end
return {"renewed", string.format("%.0f", now), string.format("%.0f", new_deadline)}
