local admission_id = ARGV[1]
local admission_digest = ARGV[2]
local concurrency_count = tonumber(ARGV[3])
local now = quota_time_milliseconds()

if #KEYS ~= 2 + concurrency_count or #ARGV ~= 3 + concurrency_count then
  return redis.error_reply("QUOTA_INVALID concurrency release shape")
end
local terminal_state = redis.call("HGET", KEYS[2], "state")
if terminal_state ~= false then
  if redis.call("HGET", KEYS[2], "admission_digest") ~= admission_digest then
    return redis.error_reply("QUOTA_CONFLICT admission digest differs")
  end
  return {"released", "1", string.format("%.0f", now)}
end
if redis.call("HGET", KEYS[1], "state") ~= "admitted" then
  return redis.error_reply("QUOTA_NOT_FOUND admission is not pending")
end
if redis.call("HGET", KEYS[1], "digest") ~= admission_digest then
  return redis.error_reply("QUOTA_CONFLICT admission digest differs")
end
if tonumber(redis.call("HGET", KEYS[1], "concurrency_count") or "-1") ~= concurrency_count then
  return redis.error_reply("QUOTA_CONFLICT concurrency rule set differs")
end
for index = 1, concurrency_count do
  local event_key = KEYS[2 + index]
  if redis.call("HGET", KEYS[1], "concurrency:" .. event_key) ~= ARGV[3 + index] then
    return redis.error_reply("QUOTA_CONFLICT concurrency rule fingerprint differs")
  end
end
if redis.call("HGET", KEYS[1], "concurrency_released") == "1" then
  return {"released", "1", string.format("%.0f", now)}
end
for index = 1, concurrency_count do
  redis.call("ZREM", KEYS[2 + index], admission_id)
end
redis.call("HSET", KEYS[1], "concurrency_released", "1")
return {"released", "0", string.format("%.0f", now)}
