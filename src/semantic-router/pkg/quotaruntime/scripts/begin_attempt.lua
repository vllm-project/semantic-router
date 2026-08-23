local admission_digest = ARGV[1]
local dispatch_id = ARGV[2]
local dispatch_key = ARGV[3]
local plan_digest = ARGV[4]
local model_id = ARGV[5]
local model_revision = ARGV[6]
local request_digest = ARGV[7]
local attempt_id = ARGV[8]
local attempt_number = tonumber(ARGV[9])
local backend_id = ARGV[10]
local provider_id = ARGV[11]
local now = quota_time_milliseconds()

local function key_has_type(key, expected)
  local observed = redis.call("TYPE", key)
  if type(observed) == "table" then
    observed = observed.ok
  end
  return observed == "none" or observed == expected
end

if #KEYS ~= 4 or #ARGV ~= 11 then
  return redis.error_reply("QUOTA_INVALID begin attempt shape")
end
if not key_has_type(KEYS[1], "hash") or not key_has_type(KEYS[2], "hash")
    or not key_has_type(KEYS[3], "hash") or not key_has_type(KEYS[4], "hash") then
  return redis.error_reply("QUOTA_CORRUPT begin attempt key type")
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
local deadline = tonumber(redis.call("HGET", KEYS[4], prefix .. "deadline") or "")
local max_attempts = tonumber(redis.call("HGET", KEYS[4], prefix .. "max_attempts") or "")
local attempt_count = tonumber(redis.call("HGET", KEYS[4], prefix .. "attempt_count") or "")
if deadline == nil or max_attempts == nil or attempt_count == nil then
  return redis.error_reply("QUOTA_CORRUPT dispatch attempt metadata")
end
if now >= deadline then
  return redis.error_reply("QUOTA_CONFLICT dispatch deadline has passed")
end
if attempt_number == nil or attempt_number < 1 or attempt_number > max_attempts then
  return redis.error_reply("QUOTA_INVALID attempt number is outside dispatch bound")
end
if attempt_number ~= attempt_count + 1 then
  return redis.error_reply("QUOTA_CONFLICT attempt number is not the next contiguous attempt")
end

local attempt_prefix = prefix .. "attempt:" .. ARGV[9] .. ":"
if redis.call("HEXISTS", KEYS[4], attempt_prefix .. "state") == 1 then
  return redis.error_reply("QUOTA_CONFLICT attempt was already started")
end
if attempt_number > 1 then
  local previous_prefix = prefix .. "attempt:" .. tostring(attempt_number - 1) .. ":"
  if redis.call("HGET", KEYS[4], previous_prefix .. "state") ~= "known_zero" then
    return redis.error_reply("QUOTA_CONFLICT only known-zero evidence permits retry")
  end
end

local started_at = string.format("%.0f", now)
local revision = tonumber(redis.call("HGET", KEYS[4], "revision") or "0")
if revision == nil or revision < 0 or revision >= 4294967295
    or revision ~= math.floor(revision) then
  return redis.error_reply("QUOTA_CORRUPT attempt evidence revision")
end
redis.call("HSET", KEYS[4],
  attempt_prefix .. "attempt_id", attempt_id,
  attempt_prefix .. "number", ARGV[9],
  attempt_prefix .. "backend_id", backend_id,
  attempt_prefix .. "provider_id", provider_id,
  attempt_prefix .. "state", "started",
  attempt_prefix .. "status_code", "0",
  attempt_prefix .. "error_code", "",
  attempt_prefix .. "started_at", started_at,
  attempt_prefix .. "completed_at", "",
  prefix .. "attempt_count", ARGV[9])
redis.call("HINCRBY", KEYS[4], "revision", 1)
return {"attempt_started", "0", string.format("%.0f", now), started_at}
