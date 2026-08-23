local admission_digest = ARGV[1]
local dispatch_id = ARGV[2]
local dispatch_key = ARGV[3]
local plan_digest = ARGV[4]
local model_id = ARGV[5]
local model_revision = ARGV[6]
local request_digest = ARGV[7]
local attempt_id = ARGV[8]
local attempt_number = ARGV[9]
local backend_id = ARGV[10]
local provider_id = ARGV[11]
local evidence_state = ARGV[12]
local status_code = ARGV[13]
local error_code = ARGV[14]
local now = quota_time_milliseconds()

local function key_has_type(key, expected)
  local observed = redis.call("TYPE", key)
  if type(observed) == "table" then
    observed = observed.ok
  end
  return observed == "none" or observed == expected
end

if #KEYS ~= 4 or #ARGV ~= 14 then
  return redis.error_reply("QUOTA_INVALID finish attempt shape")
end
if not key_has_type(KEYS[1], "hash") or not key_has_type(KEYS[2], "hash")
    or not key_has_type(KEYS[3], "hash") or not key_has_type(KEYS[4], "hash") then
  return redis.error_reply("QUOTA_CORRUPT finish attempt key type")
end
if redis.call("EXISTS", KEYS[2]) == 1 then
  return redis.error_reply("QUOTA_CONFLICT admission is already terminal")
end
if redis.call("HGET", KEYS[1], "state") ~= "admitted" then
  return redis.error_reply("QUOTA_NOT_FOUND admission is not pending")
end
if redis.call("HGET", KEYS[1], "digest") ~= admission_digest then
  return redis.error_reply("QUOTA_CONFLICT admission digest differs")
end

local prefix = "dispatch:" .. dispatch_key .. ":"
local dispatch_ordinal = redis.call("HGET", KEYS[4], prefix .. "ordinal")
if dispatch_ordinal == false or redis.call("HGET", KEYS[4], prefix .. "dispatch_id") ~= dispatch_id
    or redis.call("HGET", KEYS[4], prefix .. "plan_digest") ~= plan_digest
    or redis.call("HGET", KEYS[4], prefix .. "model_id") ~= model_id
    or redis.call("HGET", KEYS[4], prefix .. "model_revision") ~= model_revision
    or redis.call("HGET", KEYS[4], prefix .. "request_digest") ~= request_digest
    or redis.call("HGET", KEYS[3], dispatch_id) ~= dispatch_ordinal .. "|" .. plan_digest then
  return redis.error_reply("QUOTA_CONFLICT dispatch attempt identity differs")
end

local attempt_prefix = prefix .. "attempt:" .. attempt_number .. ":"
if redis.call("HGET", KEYS[4], attempt_prefix .. "attempt_id") ~= attempt_id
    or redis.call("HGET", KEYS[4], attempt_prefix .. "number") ~= attempt_number
    or redis.call("HGET", KEYS[4], attempt_prefix .. "backend_id") ~= backend_id
    or redis.call("HGET", KEYS[4], attempt_prefix .. "provider_id") ~= provider_id then
  return redis.error_reply("QUOTA_CONFLICT attempt identity differs")
end
local current_state = redis.call("HGET", KEYS[4], attempt_prefix .. "state")
if current_state == false then
  return redis.error_reply("QUOTA_NOT_FOUND attempt was not started")
end
if current_state ~= "started" then
  if current_state == evidence_state
      and redis.call("HGET", KEYS[4], attempt_prefix .. "status_code") == status_code
      and redis.call("HGET", KEYS[4], attempt_prefix .. "error_code") == error_code then
    return {"attempt_finished", "1", string.format("%.0f", now),
      redis.call("HGET", KEYS[4], attempt_prefix .. "completed_at") or ""}
  end
  return redis.error_reply("QUOTA_CONFLICT attempt evidence differs")
end

local completed_at = string.format("%.0f", now)
local revision = tonumber(redis.call("HGET", KEYS[4], "revision") or "0")
if revision == nil or revision < 0 or revision >= 4294967295
    or revision ~= math.floor(revision) then
  return redis.error_reply("QUOTA_CORRUPT attempt evidence revision")
end
redis.call("HSET", KEYS[4],
  attempt_prefix .. "state", evidence_state,
  attempt_prefix .. "status_code", status_code,
  attempt_prefix .. "error_code", error_code,
  attempt_prefix .. "completed_at", completed_at)
redis.call("HINCRBY", KEYS[4], "revision", 1)
return {"attempt_finished", "0", string.format("%.0f", now), completed_at}
