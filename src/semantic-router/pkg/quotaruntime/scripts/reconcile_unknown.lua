local fence_id = ARGV[1]
local reconciliation_id = ARGV[2]
local plan_digest = ARGV[3]
local admission_id = ARGV[4]
local event_payload = ARGV[5]
local correction_count = tonumber(ARGV[6])
local now = quota_time_milliseconds()
local correction_argument_count = 11

local function quota_key_has_type(key, expected)
  local observed = redis.call("TYPE", key)
  if type(observed) == "table" then observed = observed.ok end
  return observed == "none" or observed == expected
end

if correction_count == nil or correction_count < 1
    or #KEYS ~= 2 + correction_count * 4
    or #ARGV ~= 6 + correction_count * correction_argument_count then
  return redis.error_reply("QUOTA_INVALID reconciliation shape")
end
if not quota_key_has_type(KEYS[1], "hash")
    or not quota_key_has_type(KEYS[2], "stream") then
  return redis.error_reply("QUOTA_CORRUPT reconciliation base key type")
end

local fence_state = redis.call("HGET", KEYS[1], "state")
if fence_state == "corrected" or fence_state == "released" then
  if redis.call("HGET", KEYS[1], "reconciliation_id") == reconciliation_id
      and redis.call("HGET", KEYS[1], "reconciliation_digest") == plan_digest then
    return {"corrected", "1", redis.call("HGET", KEYS[1], "reconciliation_stream_id") or "",
      string.format("%.0f", now)}
  end
  return redis.error_reply("QUOTA_CONFLICT fence reconciliation differs")
end
if fence_state ~= "open" or redis.call("HGET", KEYS[1], "admission_id") ~= admission_id then
  return redis.error_reply("QUOTA_CONFLICT fence is not open for this admission")
end

local updates = {}
for index = 1, correction_count do
  local key_offset = 2 + (index - 1) * 4
  local argument_offset = 6 + (index - 1) * correction_argument_count
  local meta_key = KEYS[key_offset + 1]
  local events_key = KEYS[key_offset + 2]
  local values_key = KEYS[key_offset + 3]
  local binding_fences_key = KEYS[key_offset + 4]
  if not quota_key_has_type(meta_key, "hash")
      or not quota_key_has_type(events_key, "zset")
      or not quota_key_has_type(values_key, "hash")
      or not quota_key_has_type(binding_fences_key, "set") then
    return redis.error_reply("QUOTA_CORRUPT reconciliation counter key type")
  end
  local enforcement = ARGV[argument_offset + 11]
  if redis.call("HEXISTS", KEYS[1], "rule:" .. meta_key) ~= 1
      or (enforcement == "enforce" and
        (redis.call("HEXISTS", KEYS[1], "binding:" .. binding_fences_key) ~= 1
          or redis.call("SISMEMBER", binding_fences_key, fence_id) ~= 1))
      or (enforcement ~= "enforce" and enforcement ~= "shadow") then
    return redis.error_reply("QUOTA_CONFLICT reconciliation counter is not fenced")
  end
  local amount = ARGV[argument_offset + 1]
  local incomplete = ARGV[argument_offset + 2]
  local algorithm = ARGV[argument_offset + 3]
  local window_ms = tonumber(ARGV[argument_offset + 4])
  local charge_at = tonumber(ARGV[argument_offset + 5])
  local calendar_start = ARGV[argument_offset + 6]
  local calendar_end = ARGV[argument_offset + 7]
  local member = ARGV[argument_offset + 8]
  local should_charge = ARGV[argument_offset + 9]
  local known_usage = ARGV[argument_offset + 10]
  if quota_parse(amount) == nil or quota_parse(incomplete) == nil
      or quota_compare(incomplete, "0") <= 0 or charge_at == nil
      or (should_charge ~= "0" and should_charge ~= "1")
      or (known_usage ~= "0" and known_usage ~= "1")
      or (known_usage == "1" and should_charge ~= "1") then
    return redis.error_reply("QUOTA_INVALID reconciliation quantity")
  end

  local update = {meta=meta_key, events=events_key, values=values_key,
    amount=amount, incomplete=incomplete, algorithm=algorithm, charge_at=charge_at,
    calendar_start=calendar_start, calendar_end=calendar_end, member=member,
    should_charge=should_charge, known_usage=known_usage,
    used="0", known="0", next_incomplete="0", active=false}
  local observe_error = nil
  if algorithm == "sliding_log" then
    if window_ms == nil or window_ms <= 0 or calendar_start ~= "" or calendar_end ~= "" then
      return redis.error_reply("QUOTA_INVALID sliding reconciliation window")
    end
    update.used, update.known, observe_error = quota_prune_actual(meta_key, events_key, values_key, now - window_ms)
    update.active = charge_at >= now - window_ms
    local current_incomplete = redis.call("HGET", meta_key, "incomplete") or "0"
    update.next_incomplete, observe_error = quota_subtract(current_incomplete, incomplete)
  elseif algorithm == "calendar_window" then
    if window_ms ~= 0 or tonumber(calendar_start) == nil or tonumber(calendar_end) == nil
        or tonumber(calendar_start) >= tonumber(calendar_end) then
      return redis.error_reply("QUOTA_INVALID calendar reconciliation interval")
    end
    local stored_start = redis.call("HGET", meta_key, "calendar_start")
    local stored_end = redis.call("HGET", meta_key, "calendar_end")
    update.active = stored_start == calendar_start and stored_end == calendar_end
    if update.active then
      update.used = redis.call("HGET", meta_key, "used") or "0"
      update.known = redis.call("HGET", meta_key, "known") or "0"
      local current_incomplete = redis.call("HGET", meta_key, "incomplete") or "0"
      update.next_incomplete, observe_error = quota_subtract(current_incomplete, incomplete)
    end
  else
    return redis.error_reply("QUOTA_INVALID reconciliation algorithm")
  end
  if observe_error ~= nil or update.used == nil or update.known == nil
      or (update.active and update.next_incomplete == nil) then
    return redis.error_reply("QUOTA_CORRUPT " .. (observe_error or "invalid reconciliation state"))
  end
  if update.active and should_charge == "1" then
    update.used, observe_error = quota_add(update.used, amount)
    if update.used ~= nil and update.known_usage == "1" then
      update.known, observe_error = quota_add(update.known, incomplete)
    end
    if update.used == nil or update.known == nil then
      return redis.error_reply("QUOTA_CORRUPT " .. (observe_error or "reconciliation overflow"))
    end
  end
  updates[index] = update
end

for _, update in ipairs(updates) do
  if update.algorithm == "sliding_log" then
    if update.active then
      if update.should_charge == "1" then
        redis.call("ZADD", update.events, update.charge_at, update.member)
        redis.call("HSET", update.values, update.member, update.amount .. "|" .. update.incomplete)
      end
      redis.call("HSET", update.meta, "used", update.used, "known", update.known)
    end
    -- Sliding-log incomplete usage is a scalar because an unresolved dispatch
    -- has no authoritative event timestamp to index. It therefore cannot age
    -- out through event pruning. Always remove this fence's exact contribution,
    -- even when the eventual correction charge belongs to an expired window.
    redis.call("HSET", update.meta, "incomplete", update.next_incomplete)
  elseif update.active then
    redis.call("HSET", update.meta, "used", update.used, "known", update.known,
      "incomplete", update.next_incomplete)
  end
end
local stream_id = redis.call("XADD", KEYS[2], "*",
  "reconciliation_id", reconciliation_id,
  "fence_id", fence_id,
  "plan_digest", plan_digest,
  "event", event_payload)
redis.call("HSET", KEYS[1],
  "state", "corrected",
  "reconciliation_id", reconciliation_id,
  "reconciliation_digest", plan_digest,
  "reconciliation_stream_id", stream_id,
  "corrected_at", string.format("%.0f", now))
return {"corrected", "0", stream_id, string.format("%.0f", now)}
