local rule_count = tonumber(ARGV[1])
local now, now_microseconds = quota_server_time()
local rule_argument_count = 12
if #KEYS ~= 1 + rule_count * 4 or #ARGV ~= 1 + rule_count * rule_argument_count then
  return redis.error_reply("QUOTA_INVALID meter read shape")
end

local result = {string.format("%.0f", now), tostring(rule_count)}
for index = 1, rule_count do
  local key_offset = 1 + (index - 1) * 4
  local argument_offset = 1 + (index - 1) * rule_argument_count
  local meta_key = KEYS[key_offset + 1]
  local event_key = KEYS[key_offset + 2]
  local value_key = KEYS[key_offset + 3]
  local fence_key = KEYS[key_offset + 4]
  local algorithm = ARGV[argument_offset + 2]
  local accounting = ARGV[argument_offset + 3]
  local enforcement = ARGV[argument_offset + 4]
  local calendar_schedule = ARGV[argument_offset + 7]
  local bucket_capacity = ARGV[argument_offset + 8]
  local refill_amount = ARGV[argument_offset + 9]
  local refill_period = tonumber(ARGV[argument_offset + 10])
  local gcra_emission = ARGV[argument_offset + 11]
  local gcra_burst = ARGV[argument_offset + 12]
  local window_milliseconds = tonumber(ARGV[argument_offset + 6])
  local used = "0"
  local known = "0"
  local incomplete = redis.call("HGET", meta_key, "incomplete") or "0"
  local reset_at = ""
  local observe_error = nil

  if algorithm == "sliding_log" then
    if accounting == "request" then
      used, known, observe_error = quota_prune_request(
        meta_key, event_key, now - window_milliseconds)
    else
      used, known, observe_error = quota_prune_actual(
        meta_key, event_key, value_key, now - window_milliseconds)
    end
    reset_at = quota_reset_at(event_key, window_milliseconds)
  elseif algorithm == "calendar_window" then
    local interval_start = ""
    local interval_finish = ""
    interval_start, interval_finish, observe_error = quota_calendar_interval(calendar_schedule, now)
    if observe_error == nil then
      local interval_changed = false
      used, known, incomplete, interval_changed = quota_calendar_observe(
        meta_key, interval_start, interval_finish)
      if interval_changed then
        redis.call("HSET", meta_key,
          "calendar_start", interval_start,
          "calendar_end", interval_finish,
          "used", "0", "known", "0", "incomplete", "0")
      end
      reset_at = interval_finish
    end
  elseif algorithm == "token_bucket" then
    local tokens = ""
    local last_refill = ""
    tokens, last_refill, observe_error = quota_bucket_observe(
      meta_key, bucket_capacity, refill_amount, refill_period, now)
    if observe_error == nil then
      used, observe_error = quota_subtract(bucket_capacity, tokens)
      known = used
      redis.call("HSET", meta_key, "tokens", tokens, "last_refill_ms", last_refill)
      if quota_compare(tokens, bucket_capacity) < 0 then
        reset_at = string.format("%.0f", tonumber(last_refill) + refill_period)
      end
    end
  elseif algorithm == "gcra" then
    local allowed = false
    local allow_at = ""
    local ignored_next_tat = ""
    allowed, allow_at, ignored_next_tat, observe_error = quota_gcra_observe(
      meta_key, gcra_emission, gcra_burst, now_microseconds)
    if observe_error == nil and not allowed then
      used = "1"
      known = "1"
      reset_at, observe_error = quota_microseconds_to_milliseconds(allow_at)
    end
  elseif algorithm == "concurrency" then
    redis.call("ZREMRANGEBYSCORE", event_key, "-inf", now)
    used = quota_from_count(redis.call("ZCARD", event_key))
    known = used
    reset_at = quota_reset_at(event_key, 0)
  else
    return redis.error_reply("QUOTA_INVALID unsupported runtime algorithm")
  end
  if observe_error ~= nil or used == nil or known == nil then
    return redis.error_reply("QUOTA_CORRUPT " .. (observe_error or "invalid live count"))
  end

  local fence_open = "0"
  local active_fence_ids = ""
  if accounting == "response_actual" and enforcement == "enforce"
      and redis.call("SCARD", fence_key) > 0 then
    fence_open = "1"
    local fence_ids = redis.call("SMEMBERS", fence_key)
    table.sort(fence_ids)
    active_fence_ids = table.concat(fence_ids, "\0")
  end
  table.insert(result, used)
  table.insert(result, known)
  table.insert(result, incomplete)
  table.insert(result, reset_at)
  table.insert(result, fence_open)
  table.insert(result, active_fence_ids)
end
return result
