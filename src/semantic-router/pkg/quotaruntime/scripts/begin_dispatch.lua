local admission_digest = ARGV[1]
local dispatch_id = ARGV[2]
local dispatch_key = ARGV[3]
local dispatch_type = ARGV[4]
local dispatch_ordinal = ARGV[5]
local plan_digest = ARGV[6]
local model_id = ARGV[7]
local model_revision = ARGV[8]
local request_digest = ARGV[9]
local deadline = tonumber(ARGV[10])
local max_attempts = ARGV[11]
local now = quota_time_milliseconds()

local function key_has_type(key, expected)
  local observed = redis.call("TYPE", key)
  if type(observed) == "table" then
    observed = observed.ok
  end
  return observed == "none" or observed == expected
end

if #KEYS ~= 4 or #ARGV ~= 11 then
  return redis.error_reply("QUOTA_INVALID begin dispatch shape")
end
if not key_has_type(KEYS[1], "hash") or not key_has_type(KEYS[2], "hash")
    or not key_has_type(KEYS[3], "hash") or not key_has_type(KEYS[4], "hash") then
  return redis.error_reply("QUOTA_CORRUPT begin dispatch key type")
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
if redis.call("HGET", KEYS[3], dispatch_id) ~= dispatch_ordinal .. "|" .. plan_digest then
  return redis.error_reply("QUOTA_CONFLICT dispatch journal differs")
end

local admission_deadline = tonumber(redis.call("HGET", KEYS[1], "deadline") or "")
if deadline == nil or admission_deadline == nil or deadline <= now or deadline > admission_deadline then
  return redis.error_reply("QUOTA_CONFLICT dispatch deadline is outside admission lease")
end

local prefix = "dispatch:" .. dispatch_key .. ":"
local existing_digest = redis.call("HGET", KEYS[4], prefix .. "plan_digest")
if existing_digest ~= false then
  if existing_digest ~= plan_digest
      or redis.call("HGET", KEYS[4], prefix .. "dispatch_id") ~= dispatch_id
      or redis.call("HGET", KEYS[4], prefix .. "dispatch_type") ~= dispatch_type
      or redis.call("HGET", KEYS[4], prefix .. "ordinal") ~= dispatch_ordinal
      or redis.call("HGET", KEYS[4], prefix .. "model_id") ~= model_id
      or redis.call("HGET", KEYS[4], prefix .. "model_revision") ~= model_revision
      or redis.call("HGET", KEYS[4], prefix .. "request_digest") ~= request_digest
      or redis.call("HGET", KEYS[4], prefix .. "deadline") ~= ARGV[10]
      or redis.call("HGET", KEYS[4], prefix .. "max_attempts") ~= max_attempts then
    return redis.error_reply("QUOTA_CONFLICT dispatch execution identity was reused")
  end
  return {"dispatch_started", "1", string.format("%.0f", now),
    redis.call("HGET", KEYS[4], prefix .. "started_at") or "", ARGV[10]}
end
if redis.call("HEXISTS", KEYS[4], prefix .. "ordinal") == 1 then
  return redis.error_reply("QUOTA_CORRUPT partial dispatch attempt state")
end

local started_at = string.format("%.0f", now)
local revision = tonumber(redis.call("HGET", KEYS[4], "revision") or "0")
if revision == nil or revision < 0 or revision >= 4294967295
    or revision ~= math.floor(revision) then
  return redis.error_reply("QUOTA_CORRUPT attempt evidence revision")
end
redis.call("HSET", KEYS[4],
  prefix .. "dispatch_id", dispatch_id,
  prefix .. "dispatch_type", dispatch_type,
  prefix .. "ordinal", dispatch_ordinal,
  prefix .. "plan_digest", plan_digest,
  prefix .. "model_id", model_id,
  prefix .. "model_revision", model_revision,
  prefix .. "request_digest", request_digest,
  prefix .. "deadline", ARGV[10],
  prefix .. "max_attempts", max_attempts,
  prefix .. "attempt_count", "0",
  prefix .. "started_at", started_at)
redis.call("HINCRBY", KEYS[4], "revision", 1)
return {"dispatch_started", "0", string.format("%.0f", now), started_at, ARGV[10]}
