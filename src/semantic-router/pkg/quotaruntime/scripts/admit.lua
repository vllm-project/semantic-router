local admission_id = ARGV[1]
local admission_digest = ARGV[2]
local lease_milliseconds = tonumber(ARGV[3])
local plan_digest = ARGV[4]
local precondition_count = tonumber(ARGV[5])
local rule_count = tonumber(ARGV[6])
local recovery_record = ARGV[7]
local recovery_digest = ARGV[8]
local usage_consumer_group = ARGV[#ARGV - 1]
local max_usage_backlog = tonumber(ARGV[#ARGV])
local now, now_microseconds = quota_server_time()
local deadline = now + lease_milliseconds
local precondition_argument_count = 6
local rule_argument_count = 15

if #KEYS ~= 5 + precondition_count + rule_count * 4 then
  return redis.error_reply("QUOTA_INVALID admit key count")
end
if #ARGV ~= 10 + precondition_count * precondition_argument_count + rule_count * rule_argument_count then
  return redis.error_reply("QUOTA_INVALID admit argument count")
end
if max_usage_backlog == nil or max_usage_backlog < 1 then
  return redis.error_reply("QUOTA_INVALID maximum usage backlog")
end
if usage_consumer_group == nil or usage_consumer_group == "" then
  return redis.error_reply("QUOTA_INVALID usage consumer group")
end
if (recovery_record == "" and recovery_digest ~= "") or
    (recovery_record ~= "" and recovery_digest == "") then
  return redis.error_reply("QUOTA_INVALID incomplete admission recovery record")
end

if redis.call("EXISTS", KEYS[4]) == 1 then
  return redis.error_reply("QUOTA_CONFLICT admission is already terminal")
end

local access_disposition, access_reason = quota_check_access(precondition_count, 4, 8, now)
if access_disposition == nil then
  return redis.error_reply("QUOTA_INVALID unsupported admission precondition")
end
if access_disposition ~= "allowed" then
  return {access_disposition, "0", "0", string.format("%.0f", now), "", "", access_reason}
end

local pending_state = redis.call("HGET", KEYS[2], "state")
if pending_state ~= false then
  if pending_state ~= "admitted" or redis.call("HGET", KEYS[2], "digest") ~= admission_digest
      or redis.call("HGET", KEYS[2], "plan_digest") ~= plan_digest then
    return redis.error_reply("QUOTA_CONFLICT admission identity was reused")
  end
  return {"allowed", "1", "0", string.format("%.0f", now),
    redis.call("HGET", KEYS[2], "deadline") or "", "", ""}
end

local expired = redis.call("ZRANGEBYSCORE", KEYS[1], "-inf", now, "LIMIT", 0, 1)
if #expired > 0 then
  return {"unavailable", "0", "0", string.format("%.0f", now), "", "",
    "expired pending admission requires reconciliation"}
end

local usage_stream_key = KEYS[#KEYS]
local usage_stream_type = redis.call("TYPE", usage_stream_key)
if type(usage_stream_type) == "table" then
  usage_stream_type = usage_stream_type.ok
end
if usage_stream_type ~= "none" and usage_stream_type ~= "stream" then
  return redis.error_reply("QUOTA_CORRUPT usage stream key type")
end
if usage_stream_type == "stream" then
  local consumer_group_found = false
  local usage_backlog = nil
  local groups = redis.call("XINFO", "GROUPS", usage_stream_key)
  for _, group in ipairs(groups) do
    local name = nil
    local pending = nil
    local lag = nil
    for field_index = 1, #group, 2 do
      local field = group[field_index]
      if field == "name" then
        name = group[field_index + 1]
      elseif field == "pending" then
        pending = tonumber(group[field_index + 1])
      elseif field == "lag" then
        lag = tonumber(group[field_index + 1])
      end
    end
    if name == usage_consumer_group then
      consumer_group_found = true
      if pending == nil or pending < 0 then
        return redis.error_reply("QUOTA_CORRUPT usage consumer group pending count")
      end
      if lag == nil or lag < 0 then
        return {"unavailable", "0", "0", string.format("%.0f", now), "", "",
          "usage accounting backlog is indeterminate"}
      end
      usage_backlog = pending + lag
      break
    end
  end
  if not consumer_group_found then
    return {"unavailable", "0", "0", string.format("%.0f", now), "", "",
      "usage accounting consumer group is unavailable"}
  end
  if usage_backlog >= max_usage_backlog then
    return {"unavailable", "0", "0", string.format("%.0f", now), "", "",
      "usage accounting backlog is full"}
  end
end

local observations = {}
local limiting_index = 0
local limiting_retry = ""
local limiting_reason = ""

for index = 1, rule_count do
  local key_offset = 4 + precondition_count + (index - 1) * 4
  local argument_offset = 8 + precondition_count * precondition_argument_count + (index - 1) * rule_argument_count
  local meta_key = KEYS[key_offset + 1]
  local event_key = KEYS[key_offset + 2]
  local value_key = KEYS[key_offset + 3]
  local fence_key = KEYS[key_offset + 4]
  local metric = ARGV[argument_offset + 3]
  local algorithm = ARGV[argument_offset + 4]
  local accounting = ARGV[argument_offset + 5]
  local enforcement = ARGV[argument_offset + 6]
  local limit = ARGV[argument_offset + 7]
  local window_milliseconds = tonumber(ARGV[argument_offset + 8])
  local calendar_schedule = ARGV[argument_offset + 9]
  local bucket_capacity = ARGV[argument_offset + 10]
  local refill_amount = ARGV[argument_offset + 11]
  local refill_period = tonumber(ARGV[argument_offset + 12])
  local gcra_emission = ARGV[argument_offset + 13]
  local gcra_burst = ARGV[argument_offset + 14]

  if redis.call("SCARD", fence_key) > 0 then
    return {"unavailable", "0", tostring(index), string.format("%.0f", now), "", "",
      "binding has unresolved usage"}
  end

  local observation = {
    meta = meta_key, events = event_key, algorithm = algorithm, accounting = accounting,
    used = "0", known = "0", next_used = "0", next_known = "0", retry_at = ""
  }
  local observe_error = nil
  local exhausted = false

  if algorithm == "sliding_log" then
    if accounting == "request" then
      observation.used, observation.known, observe_error = quota_prune_request(
        meta_key, event_key, now - window_milliseconds)
    else
      observation.used, observation.known, observe_error = quota_prune_actual(
        meta_key, event_key, value_key, now - window_milliseconds)
    end
    if observe_error ~= nil or observation.used == nil or observation.known == nil then
      return redis.error_reply("QUOTA_CORRUPT " .. (observe_error or "invalid sliding-log state"))
    end
    exhausted = quota_compare(observation.used, limit) >= 0
    if exhausted then
      if accounting == "request" then
        observation.retry_at, observe_error = quota_request_retry_at(
          event_key, observation.used, limit, window_milliseconds)
      else
        observation.retry_at, observe_error = quota_actual_retry_at(
          event_key, value_key, observation.used, limit, window_milliseconds)
      end
      if observe_error ~= nil then
        return redis.error_reply("QUOTA_CORRUPT " .. observe_error)
      end
    end
    observation.next_used = observation.used
    observation.next_known = observation.known
    if accounting == "request" then
      observation.next_used, observe_error = quota_add(observation.used, "1")
      if observation.next_used ~= nil then
        observation.next_known, observe_error = quota_add(observation.known, "1")
      end
    end
  elseif algorithm == "calendar_window" then
    observation.interval_start, observation.interval_finish, observe_error =
      quota_calendar_interval(calendar_schedule, now)
    if observe_error ~= nil then
      return {"unavailable", "0", tostring(index), string.format("%.0f", now), "", "",
        "calendar schedule unavailable"}
    end
    observation.used, observation.known, observation.incomplete, observation.interval_changed =
      quota_calendar_observe(meta_key, observation.interval_start, observation.interval_finish)
    if quota_compare(observation.used, limit) == nil or quota_parse(observation.known) == nil then
      return redis.error_reply("QUOTA_CORRUPT invalid calendar-window state")
    end
    observation.retry_at = observation.interval_finish
    exhausted = quota_compare(observation.used, limit) >= 0
    observation.next_used = observation.used
    observation.next_known = observation.known
    if accounting == "request" then
      observation.next_used, observe_error = quota_add(observation.used, "1")
      if observation.next_used ~= nil then
        observation.next_known, observe_error = quota_add(observation.known, "1")
      end
    end
  elseif algorithm == "token_bucket" then
    observation.tokens, observation.last_refill, observe_error = quota_bucket_observe(
      meta_key, bucket_capacity, refill_amount, refill_period, now)
    if observation.tokens ~= nil then
      exhausted = quota_compare(observation.tokens, "1") < 0
      if exhausted then
        observation.next_tokens = observation.tokens
        observation.retry_at = string.format("%.0f", tonumber(observation.last_refill) + refill_period)
      else
        observation.next_tokens, observe_error = quota_subtract(observation.tokens, "1")
      end
    end
  elseif algorithm == "gcra" then
    observation.gcra_allowed, observation.allow_at, observation.next_tat, observe_error =
      quota_gcra_observe(meta_key, gcra_emission, gcra_burst, now_microseconds)
    if observation.gcra_allowed ~= nil then
      exhausted = not observation.gcra_allowed
      if exhausted then
        observation.retry_at, observe_error = quota_microseconds_to_milliseconds(observation.allow_at)
      end
    end
  elseif algorithm == "concurrency" then
    redis.call("ZREMRANGEBYSCORE", event_key, "-inf", now)
    observation.used = quota_from_count(redis.call("ZCARD", event_key))
    observation.known = observation.used
    observation.next_used = observation.used
    observation.next_known = observation.known
    observation.retry_at = quota_reset_at(event_key, 0)
    exhausted = quota_compare(observation.used, limit) >= 0
  else
    return redis.error_reply("QUOTA_INVALID unsupported runtime algorithm")
  end

  if observe_error ~= nil or observation.used == nil or observation.known == nil then
    return redis.error_reply("QUOTA_CORRUPT " .. (observe_error or "invalid live count"))
  end
  if enforcement == "enforce" and exhausted then
    if limiting_index == 0 or (observation.retry_at ~= "" and
        (limiting_retry == "" or tonumber(observation.retry_at) > tonumber(limiting_retry))) then
      limiting_index = index
      limiting_retry = observation.retry_at
      limiting_reason = metric .. " limit exhausted"
    end
  end
  observations[index] = observation
end

if limiting_index ~= 0 then
  return {"rate_limited", "0", tostring(limiting_index), string.format("%.0f", now), "",
    limiting_retry, limiting_reason}
end

local actual_rule_count = 0
local concurrency_count = 0
local enforce_binding_count = 0
local enforce_bindings = {}
redis.call("HSET", KEYS[2], "state", "admitted", "digest", admission_digest,
  "plan_digest", plan_digest, "lease_ms", string.format("%.0f", lease_milliseconds),
  "heartbeat_at", string.format("%.0f", now), "deadline", string.format("%.0f", deadline))
if recovery_record ~= "" then
  redis.call("HSET", KEYS[2], "recovery_record", recovery_record,
    "recovery_digest", recovery_digest)
end

for index = 1, rule_count do
  local key_offset = 4 + precondition_count + (index - 1) * 4
  local argument_offset = 8 + precondition_count * precondition_argument_count + (index - 1) * rule_argument_count
  local meta_key = KEYS[key_offset + 1]
  local event_key = KEYS[key_offset + 2]
  local fence_key = KEYS[key_offset + 4]
  local accounting = ARGV[argument_offset + 5]
  local enforcement = ARGV[argument_offset + 6]
  local fingerprint = ARGV[argument_offset + 15]
  local observation = observations[index]

  if observation.algorithm == "sliding_log" and observation.accounting == "request" then
    redis.call("ZADD", event_key, now, admission_id)
    redis.call("HSET", meta_key, "used", observation.next_used, "known", observation.next_known)
  elseif observation.algorithm == "calendar_window" then
    redis.call("HSET", meta_key,
      "calendar_start", observation.interval_start,
      "calendar_end", observation.interval_finish,
      "used", observation.next_used,
      "known", observation.next_known)
    if observation.interval_changed then
      redis.call("HSET", meta_key, "incomplete", "0")
    end
  elseif observation.algorithm == "token_bucket" then
    redis.call("HSET", meta_key,
      "tokens", observation.next_tokens,
      "last_refill_ms", observation.last_refill)
  elseif observation.algorithm == "gcra" then
    redis.call("HSET", meta_key, "tat_us", observation.next_tat)
  elseif observation.algorithm == "concurrency" then
    redis.call("ZADD", event_key, deadline, admission_id)
    concurrency_count = concurrency_count + 1
    redis.call("HSET", KEYS[2], "concurrency:" .. event_key, fingerprint)
  end

  if accounting == "response_actual" then
    actual_rule_count = actual_rule_count + 1
    redis.call("HSET", KEYS[2], "actual:" .. meta_key, fingerprint)
    if enforcement == "enforce" and enforce_bindings[fence_key] == nil then
      enforce_bindings[fence_key] = true
      enforce_binding_count = enforce_binding_count + 1
      redis.call("HSET", KEYS[2], "fence:" .. fence_key, "1")
    end
  end
end

redis.call("HSET", KEYS[2], "actual_rule_count", tostring(actual_rule_count),
  "concurrency_count", tostring(concurrency_count),
  "enforce_binding_count", tostring(enforce_binding_count))
redis.call("ZADD", KEYS[1], deadline, admission_id)
return {"allowed", "0", "0", string.format("%.0f", now),
  string.format("%.0f", deadline), "", ""}
