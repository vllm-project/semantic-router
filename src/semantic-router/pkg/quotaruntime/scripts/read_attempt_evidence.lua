local admission_digest = ARGV[1]
local dispatch_id = ARGV[2]
local dispatch_key = ARGV[3]
local dispatch_ordinal = ARGV[4]
local plan_digest = ARGV[5]
local model_id = ARGV[6]
local model_revision = ARGV[7]
local now = quota_time_milliseconds()

local function key_has_type(key, expected)
  local observed = redis.call("TYPE", key)
  if type(observed) == "table" then
    observed = observed.ok
  end
  return observed == "none" or observed == expected
end

if #KEYS ~= 4 or #ARGV ~= 7 then
  return redis.error_reply("QUOTA_INVALID read attempt evidence shape")
end
if not key_has_type(KEYS[1], "hash") or not key_has_type(KEYS[2], "hash")
    or not key_has_type(KEYS[3], "hash") or not key_has_type(KEYS[4], "hash") then
  return redis.error_reply("QUOTA_CORRUPT read attempt evidence key type")
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

local revision = tonumber(redis.call("HGET", KEYS[4], "revision") or "0")
if revision == nil or revision < 0 or revision > 4294967295
    or revision ~= math.floor(revision) then
  return redis.error_reply("QUOTA_CORRUPT attempt evidence revision")
end
local revision_text = string.format("%.0f", revision)
local prefix = "dispatch:" .. dispatch_key .. ":"
local existing_digest = redis.call("HGET", KEYS[4], prefix .. "plan_digest")
if existing_digest == false then
  if redis.call("HEXISTS", KEYS[4], prefix .. "dispatch_id") == 1
      or redis.call("HEXISTS", KEYS[4], prefix .. "ordinal") == 1
      or redis.call("HEXISTS", KEYS[4], prefix .. "request_digest") == 1
      or redis.call("HEXISTS", KEYS[4], prefix .. "attempt_count") == 1 then
    return redis.error_reply("QUOTA_CORRUPT partial dispatch attempt state")
  end
  return {"attempt_evidence", string.format("%.0f", now), revision_text, "0"}
end
if existing_digest ~= plan_digest
    or redis.call("HGET", KEYS[4], prefix .. "dispatch_id") ~= dispatch_id
    or redis.call("HGET", KEYS[4], prefix .. "ordinal") ~= dispatch_ordinal
    or redis.call("HGET", KEYS[4], prefix .. "model_id") ~= model_id
    or redis.call("HGET", KEYS[4], prefix .. "model_revision") ~= model_revision then
  return redis.error_reply("QUOTA_CONFLICT dispatch attempt identity differs")
end
local attempt_count = tonumber(redis.call("HGET", KEYS[4], prefix .. "attempt_count") or "")
if attempt_count == nil or attempt_count < 0 or attempt_count ~= math.floor(attempt_count) then
  return redis.error_reply("QUOTA_CORRUPT dispatch attempt count")
end

local result = {
  "attempt_evidence", string.format("%.0f", now), revision_text, "1",
  redis.call("HGET", KEYS[4], prefix .. "dispatch_type") or "",
  dispatch_ordinal,
  plan_digest,
  redis.call("HGET", KEYS[4], prefix .. "model_id") or "",
  redis.call("HGET", KEYS[4], prefix .. "model_revision") or "",
  redis.call("HGET", KEYS[4], prefix .. "request_digest") or "",
  redis.call("HGET", KEYS[4], prefix .. "started_at") or "",
  redis.call("HGET", KEYS[4], prefix .. "deadline") or "",
  redis.call("HGET", KEYS[4], prefix .. "max_attempts") or "",
  tostring(attempt_count)
}
for number = 1, attempt_count do
  local attempt_prefix = prefix .. "attempt:" .. tostring(number) .. ":"
  local state = redis.call("HGET", KEYS[4], attempt_prefix .. "state")
  if state == false then
    return redis.error_reply("QUOTA_CORRUPT missing contiguous attempt evidence")
  end
  table.insert(result, redis.call("HGET", KEYS[4], attempt_prefix .. "attempt_id") or "")
  table.insert(result, redis.call("HGET", KEYS[4], attempt_prefix .. "number") or "")
  table.insert(result, redis.call("HGET", KEYS[4], attempt_prefix .. "backend_id") or "")
  table.insert(result, redis.call("HGET", KEYS[4], attempt_prefix .. "provider_id") or "")
  table.insert(result, state)
  table.insert(result, redis.call("HGET", KEYS[4], attempt_prefix .. "status_code") or "")
  table.insert(result, redis.call("HGET", KEYS[4], attempt_prefix .. "error_code") or "")
  table.insert(result, redis.call("HGET", KEYS[4], attempt_prefix .. "started_at") or "")
  table.insert(result, redis.call("HGET", KEYS[4], attempt_prefix .. "completed_at") or "")
end
return result
