local admission_id = ARGV[1]
local admission_digest = ARGV[2]
local finalization_digest = ARGV[3]
local plan_digest = ARGV[4]
local event_payload = ARGV[5]
local marker_ttl_milliseconds = tonumber(ARGV[6])
local fence_id = ARGV[7]
local expected_dispatch_count = tonumber(ARGV[8])
local expected_evidence_revision = tonumber(ARGV[9])
local actual_count = tonumber(ARGV[10])
local now = quota_time_milliseconds()
local evidence_argument_count = 8

local function quota_key_has_type(key, expected)
  local observed = redis.call("TYPE", key)
  if type(observed) == "table" then
    observed = observed.ok
  end
  return observed == "none" or observed == expected
end

if not quota_key_has_type(KEYS[1], "zset")
    or not quota_key_has_type(KEYS[2], "hash")
    or not quota_key_has_type(KEYS[3], "hash")
    or not quota_key_has_type(KEYS[4], "hash")
    or not quota_key_has_type(KEYS[5], "stream")
    or not quota_key_has_type(KEYS[6], "hash") then
  return redis.error_reply("QUOTA_CORRUPT finalization base key type")
end

local terminal_state = redis.call("HGET", KEYS[4], "state")
if terminal_state ~= false then
  if terminal_state == "finalized"
      and redis.call("HGET", KEYS[4], "admission_digest") == admission_digest
      and redis.call("HGET", KEYS[4], "finalization_digest") == finalization_digest
      and redis.call("HGET", KEYS[4], "plan_digest") == plan_digest then
    return {"finalized", "1", string.format("%.0f", now),
      redis.call("HGET", KEYS[4], "evidence_state") or "",
      redis.call("HGET", KEYS[4], "stream_id") or ""}
  end
  return redis.error_reply("QUOTA_CONFLICT terminal finalization differs")
end
if redis.call("HGET", KEYS[2], "state") ~= "admitted" then
  return redis.error_reply("QUOTA_NOT_FOUND admission is not pending")
end
if redis.call("HGET", KEYS[2], "digest") ~= admission_digest then
  return redis.error_reply("QUOTA_CONFLICT admission digest differs")
end
local admission_plan_digest = redis.call("HGET", KEYS[2], "plan_digest")
if admission_plan_digest == false or admission_plan_digest == "" then
  return redis.error_reply("QUOTA_CORRUPT admission plan digest is missing")
end
if tonumber(redis.call("HGET", KEYS[2], "actual_rule_count") or "-1") ~= actual_count then
  return redis.error_reply("QUOTA_CONFLICT actual rule set differs")
end

local dispatch_count = quota_from_count(redis.call("HLEN", KEYS[3]))
if dispatch_count == nil then
  return redis.error_reply("QUOTA_CORRUPT dispatch count exceeds exact Lua range")
end
if expected_dispatch_count == nil or expected_dispatch_count < 1
    or expected_dispatch_count ~= math.floor(expected_dispatch_count)
    or dispatch_count ~= string.format("%.0f", expected_dispatch_count) then
  return redis.error_reply("QUOTA_CONFLICT settlement dispatch journal differs")
end

local argument_offset = 10
local key_offset = 6
local updates = {}
local known_count = 0
local unknown_count = 0

for index = 1, actual_count do
  local meta_key = KEYS[key_offset + 1]
  local event_key = KEYS[key_offset + 2]
  local value_key = KEYS[key_offset + 3]
  if not quota_key_has_type(meta_key, "hash")
      or not quota_key_has_type(event_key, "zset")
      or not quota_key_has_type(value_key, "hash") then
    return redis.error_reply("QUOTA_CORRUPT finalization counter key type")
  end

  local fingerprint = ARGV[argument_offset + 1]
  local state = ARGV[argument_offset + 2]
  local amount = ARGV[argument_offset + 3]
  local reason = ARGV[argument_offset + 4]
  local algorithm = ARGV[argument_offset + 5]
  local window_milliseconds = tonumber(ARGV[argument_offset + 6])
  local calendar_schedule = ARGV[argument_offset + 7]
  local fence_ordinal = tonumber(ARGV[argument_offset + 8])
  if redis.call("HGET", KEYS[2], "actual:" .. meta_key) ~= fingerprint then
    return redis.error_reply("QUOTA_CONFLICT actual rule fingerprint differs")
  end
  if quota_parse(amount) == nil then
    return redis.error_reply("QUOTA_INVALID finalization amount")
  end

  local update = {
    meta = meta_key, events = event_key, values = value_key, fingerprint = fingerprint,
    state = state, amount = amount, reason = reason, algorithm = algorithm,
    fence_ordinal = fence_ordinal, interval_start = "", interval_finish = "",
    interval_changed = false, used = "0", known = "0", incomplete = "0"
  }
  if state == "known" then
    known_count = known_count + 1
    if reason ~= "" or fence_ordinal ~= 0 then
      return redis.error_reply("QUOTA_INVALID known evidence shape")
    end
    if dispatch_count == "0" and quota_compare(amount, "0") > 0 then
      return redis.error_reply("QUOTA_CONFLICT nonzero usage has no dispatch journal")
    end
    local observe_error = nil
    if algorithm == "sliding_log" then
      update.used, update.known, observe_error = quota_prune_actual(
        meta_key, event_key, value_key, now - window_milliseconds)
    elseif algorithm == "calendar_window" then
      update.interval_start, update.interval_finish, observe_error =
        quota_calendar_interval(calendar_schedule, now)
      if observe_error == nil then
        local ignored_incomplete = "0"
        update.used, update.known, ignored_incomplete, update.interval_changed =
          quota_calendar_observe(meta_key, update.interval_start, update.interval_finish)
      end
    else
      return redis.error_reply("QUOTA_INVALID finalization algorithm")
    end
    if observe_error ~= nil then
      return redis.error_reply("QUOTA_CORRUPT " .. observe_error)
    end
    local add_error = nil
    update.used, add_error = quota_add(update.used, amount)
    if update.used ~= nil then
      update.known, add_error = quota_add(update.known, dispatch_count)
    end
    if update.used == nil or update.known == nil then
      return redis.error_reply("QUOTA_CORRUPT " .. add_error)
    end
  elseif state == "unknown" then
    unknown_count = unknown_count + 1
    if amount ~= "0" or reason == "" or dispatch_count == "0" then
      return redis.error_reply("QUOTA_INVALID unknown evidence shape")
    end
    if algorithm == "calendar_window" then
      local observe_error = nil
      update.interval_start, update.interval_finish, observe_error =
        quota_calendar_interval(calendar_schedule, now)
      if observe_error == nil then
        update.used, update.known, update.incomplete, update.interval_changed =
          quota_calendar_observe(meta_key, update.interval_start, update.interval_finish)
      end
      if observe_error ~= nil then
        return redis.error_reply("QUOTA_CORRUPT " .. observe_error)
      end
    elseif algorithm == "sliding_log" then
      update.incomplete = redis.call("HGET", meta_key, "incomplete") or "0"
    else
      return redis.error_reply("QUOTA_INVALID finalization algorithm")
    end
    local incomplete_error = nil
    update.incomplete, incomplete_error = quota_add(update.incomplete, dispatch_count)
    if update.incomplete == nil then
      return redis.error_reply("QUOTA_CORRUPT " .. incomplete_error)
    end
  else
    return redis.error_reply("QUOTA_INVALID actual evidence state")
  end
  updates[index] = update
  argument_offset = argument_offset + evidence_argument_count
  key_offset = key_offset + 3
end

local fence_count = tonumber(ARGV[argument_offset + 1])
argument_offset = argument_offset + 1
local fence_key_offset = key_offset
local referenced_fences = {}
for index = 1, fence_count do
  local fence_set_key = KEYS[fence_key_offset + index]
  if not quota_key_has_type(fence_set_key, "set") then
    return redis.error_reply("QUOTA_CORRUPT finalization fence key type")
  end
end
for _, update in ipairs(updates) do
  if update.fence_ordinal < 0 or update.fence_ordinal > fence_count then
    return redis.error_reply("QUOTA_INVALID finalization fence ordinal")
  end
  if update.fence_ordinal > 0 then
    if update.state ~= "unknown" then
      return redis.error_reply("QUOTA_INVALID known evidence cannot fence")
    end
    local fence_set_key = KEYS[fence_key_offset + update.fence_ordinal]
    if redis.call("HEXISTS", KEYS[2], "fence:" .. fence_set_key) ~= 1 then
      return redis.error_reply("QUOTA_CONFLICT enforce binding differs")
    end
    referenced_fences[update.fence_ordinal] = true
  end
end
for index = 1, fence_count do
  if referenced_fences[index] == nil then
    return redis.error_reply("QUOTA_INVALID unreferenced finalization fence")
  end
end
key_offset = key_offset + fence_count

local concurrency_count = tonumber(ARGV[argument_offset + 1])
argument_offset = argument_offset + 1
if tonumber(redis.call("HGET", KEYS[2], "concurrency_count") or "-1") ~= concurrency_count then
  return redis.error_reply("QUOTA_CONFLICT concurrency rule set differs")
end
if #KEYS ~= 7 + actual_count * 3 + fence_count + concurrency_count then
  return redis.error_reply("QUOTA_INVALID finalization key count")
end
if #ARGV ~= argument_offset + concurrency_count then
  return redis.error_reply("QUOTA_INVALID finalization argument count")
end
if not quota_key_has_type(KEYS[#KEYS], "hash") then
  return redis.error_reply("QUOTA_CORRUPT finalization attempt key type")
end
local evidence_revision = tonumber(redis.call("HGET", KEYS[#KEYS], "revision") or "0")
if expected_evidence_revision == nil or expected_evidence_revision < 0
    or expected_evidence_revision > 4294967295
    or expected_evidence_revision ~= math.floor(expected_evidence_revision)
    or evidence_revision == nil or evidence_revision < 0
    or evidence_revision > 4294967295
    or evidence_revision ~= math.floor(evidence_revision) then
  return redis.error_reply("QUOTA_CORRUPT finalization attempt evidence revision")
end
if evidence_revision ~= expected_evidence_revision then
  return redis.error_reply("QUOTA_EVIDENCE_CHANGED attempt evidence changed during settlement")
end
for index = 1, concurrency_count do
  local event_key = KEYS[key_offset + index]
  local fingerprint = ARGV[argument_offset + index]
  if not quota_key_has_type(event_key, "zset") then
    return redis.error_reply("QUOTA_CORRUPT finalization concurrency key type")
  end
  if redis.call("HGET", KEYS[2], "concurrency:" .. event_key) ~= fingerprint then
    return redis.error_reply("QUOTA_CONFLICT concurrency rule fingerprint differs")
  end
end

if unknown_count > 0 and fence_id == "" then
  return redis.error_reply("QUOTA_INVALID unknown finalization requires fence ID")
end
if unknown_count == 0 and fence_id ~= "" then
  return redis.error_reply("QUOTA_INVALID known finalization cannot carry fence ID")
end
if unknown_count > 0 and redis.call("EXISTS", KEYS[6]) == 1 then
  return redis.error_reply("QUOTA_CONFLICT fence identity was reused")
end

for _, update in ipairs(updates) do
  if update.state == "known" then
    if update.algorithm == "sliding_log" and dispatch_count ~= "0" then
      redis.call("ZADD", update.events, now, admission_id)
      redis.call("HSET", update.values, admission_id, update.amount .. "|" .. dispatch_count)
    elseif update.algorithm == "calendar_window" then
      redis.call("HSET", update.meta,
        "calendar_start", update.interval_start,
        "calendar_end", update.interval_finish)
      if update.interval_changed then
        redis.call("HSET", update.meta, "incomplete", "0")
      end
    end
    redis.call("HSET", update.meta, "used", update.used, "known", update.known)
  else
    if update.algorithm == "calendar_window" then
      redis.call("HSET", update.meta,
        "calendar_start", update.interval_start,
        "calendar_end", update.interval_finish,
        "used", update.used,
        "known", update.known)
    end
    redis.call("HSET", update.meta, "incomplete", update.incomplete)
    redis.call("HSET", KEYS[6],
      "rule:" .. update.meta, update.fingerprint,
      "reason:" .. update.meta, update.reason)
  end
end
for index = 1, fence_count do
  local fence_set_key = KEYS[fence_key_offset + index]
  redis.call("SADD", fence_set_key, fence_id)
  redis.call("HSET", KEYS[6], "binding:" .. fence_set_key, "1")
end
for index = 1, concurrency_count do
  redis.call("ZREM", KEYS[key_offset + index], admission_id)
end

local evidence_state = "known"
if known_count == 0 and unknown_count > 0 then
  evidence_state = "unknown"
elseif known_count > 0 and unknown_count > 0 then
  evidence_state = "mixed"
end
if unknown_count > 0 then
  redis.call("HSET", KEYS[6],
    "state", "open",
    "admission_id", admission_id,
    "admission_digest", admission_digest,
    "finalization_digest", finalization_digest,
    "created_at", string.format("%.0f", now),
    "dispatch_count", dispatch_count,
    "evidence_state", evidence_state)
end

local stream_id = redis.call("XADD", KEYS[5], "*",
  "admission_id", admission_id,
  "admission_digest", admission_digest,
  "finalization_digest", finalization_digest,
  "evidence_state", evidence_state,
  "event", event_payload)
redis.call("ZREM", KEYS[1], admission_id)
redis.call("HSET", KEYS[2],
  "state", "finalized",
  "finalization_digest", finalization_digest,
  "admission_plan_digest", admission_plan_digest,
  "plan_digest", plan_digest,
  "evidence_state", evidence_state,
  "stream_id", stream_id,
  "finalized_at", string.format("%.0f", now))
redis.call("PEXPIRE", KEYS[2], marker_ttl_milliseconds)
redis.call("PEXPIRE", KEYS[3], marker_ttl_milliseconds)
redis.call("HSET", KEYS[4],
  "state", "finalized",
  "admission_digest", admission_digest,
  "finalization_digest", finalization_digest,
  "admission_plan_digest", admission_plan_digest,
  "plan_digest", plan_digest,
  "evidence_state", evidence_state,
  "stream_id", stream_id,
  "finalized_at", string.format("%.0f", now))
redis.call("PEXPIRE", KEYS[4], marker_ttl_milliseconds)
redis.call("DEL", KEYS[#KEYS])
return {"finalized", "0", string.format("%.0f", now), evidence_state, stream_id}
