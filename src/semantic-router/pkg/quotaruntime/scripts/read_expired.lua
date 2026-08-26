local admission_id = ARGV[1]
local expected_deadline = ARGV[2]
local max_dispatches = tonumber(ARGV[3])
local max_attempt_fields = tonumber(ARGV[4])
local now = quota_time_milliseconds()

local function key_has_type(key, expected)
  local observed = redis.call("TYPE", key)
  if type(observed) == "table" then
    observed = observed.ok
  end
  return observed == "none" or observed == expected
end

if #KEYS ~= 5 or #ARGV ~= 4 then
  return redis.error_reply("QUOTA_INVALID read expired admission shape")
end
if not key_has_type(KEYS[1], "zset") or not key_has_type(KEYS[2], "hash")
    or not key_has_type(KEYS[3], "hash") or not key_has_type(KEYS[4], "hash")
    or not key_has_type(KEYS[5], "hash") then
  return redis.error_reply("QUOTA_CORRUPT expired admission key type")
end
if max_dispatches == nil or max_dispatches < 1 or max_dispatches ~= math.floor(max_dispatches)
    or max_attempt_fields == nil or max_attempt_fields < 1
    or max_attempt_fields ~= math.floor(max_attempt_fields) then
  return redis.error_reply("QUOTA_INVALID expired admission recovery bounds")
end

if redis.call("EXISTS", KEYS[5]) == 1 then
  return {"expired_admission_terminal", string.format("%.0f", now)}
end
local indexed_deadline = redis.call("ZSCORE", KEYS[1], admission_id)
local pending_state = redis.call("HGET", KEYS[2], "state")
if indexed_deadline == false and pending_state == false then
  return {"expired_admission_gone", string.format("%.0f", now)}
end
if indexed_deadline == false or pending_state ~= "admitted"
    or redis.call("HGET", KEYS[2], "deadline") ~= indexed_deadline
    or tonumber(indexed_deadline) == nil then
  return redis.error_reply("QUOTA_CORRUPT expired admission index differs")
end
if indexed_deadline ~= expected_deadline or tonumber(indexed_deadline) > now then
  return {"expired_admission_renewed", string.format("%.0f", now)}
end
if redis.call("HLEN", KEYS[3]) > max_dispatches then
  return redis.error_reply("QUOTA_CORRUPT expired admission dispatch journal exceeds recovery bound")
end
if redis.call("HLEN", KEYS[4]) > max_attempt_fields then
  return redis.error_reply("QUOTA_CORRUPT expired admission attempt evidence exceeds recovery bound")
end

return {
  "expired_admission", string.format("%.0f", now), expected_deadline,
  redis.call("HGETALL", KEYS[2]),
  redis.call("HGETALL", KEYS[3]),
  redis.call("HGETALL", KEYS[4])
}
