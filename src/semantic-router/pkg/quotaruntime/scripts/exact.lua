local quota_base = 10000000
local quota_limb_count = 6
local quota_limb_width = 7

local function quota_parse(value)
  if type(value) ~= "string" or value == "" or #value > 42 then
    return nil
  end
  if #value > 1 and string.sub(value, 1, 1) == "0" then
    return nil
  end
  if string.find(value, "[^0-9]") ~= nil then
    return nil
  end

  local limbs = {}
  local finish = #value
  for index = 1, quota_limb_count do
    if finish <= 0 then
      limbs[index] = 0
    else
      local start = finish - quota_limb_width + 1
      if start < 1 then
        start = 1
      end
      limbs[index] = tonumber(string.sub(value, start, finish))
      finish = start - 1
    end
  end
  if finish > 0 then
    return nil
  end
  return limbs
end

local function quota_format(limbs)
  local highest = quota_limb_count
  while highest > 1 and limbs[highest] == 0 do
    highest = highest - 1
  end
  local result = string.format("%d", limbs[highest])
  for index = highest - 1, 1, -1 do
    result = result .. string.format("%07d", limbs[index])
  end
  return result
end

local function quota_compare(left, right)
  local left_limbs = quota_parse(left)
  local right_limbs = quota_parse(right)
  if left_limbs == nil or right_limbs == nil then
    return nil
  end
  for index = quota_limb_count, 1, -1 do
    if left_limbs[index] < right_limbs[index] then
      return -1
    end
    if left_limbs[index] > right_limbs[index] then
      return 1
    end
  end
  return 0
end

local function quota_add(left, right)
  local left_limbs = quota_parse(left)
  local right_limbs = quota_parse(right)
  if left_limbs == nil or right_limbs == nil then
    return nil, "invalid quantity"
  end
  local result = {}
  local carry = 0
  for index = 1, quota_limb_count do
    local total = left_limbs[index] + right_limbs[index] + carry
    result[index] = total % quota_base
    carry = math.floor(total / quota_base)
  end
  if carry ~= 0 then
    return nil, "quantity overflow"
  end
  return quota_format(result), nil
end

local function quota_multiply(left, right)
  local left_limbs = quota_parse(left)
  local right_limbs = quota_parse(right)
  if left_limbs == nil or right_limbs == nil then
    return nil, "invalid quantity"
  end
  local product = {}
  for index = 1, quota_limb_count * 2 do
    product[index] = 0
  end
  for left_index = 1, quota_limb_count do
    for right_index = 1, quota_limb_count do
      local position = left_index + right_index - 1
      product[position] = product[position] + left_limbs[left_index] * right_limbs[right_index]
    end
  end
  for position = 1, quota_limb_count * 2 - 1 do
    local carry = math.floor(product[position] / quota_base)
    product[position] = product[position] % quota_base
    product[position + 1] = product[position + 1] + carry
  end
  for position = quota_limb_count + 1, quota_limb_count * 2 do
    if product[position] ~= 0 then
      return nil, "quantity overflow"
    end
  end
  local result = {}
  for position = 1, quota_limb_count do
    result[position] = product[position]
  end
  return quota_format(result), nil
end

local function quota_subtract(left, right)
  local comparison = quota_compare(left, right)
  if comparison == nil then
    return nil, "invalid quantity"
  end
  if comparison < 0 then
    return nil, "quantity underflow"
  end
  local left_limbs = quota_parse(left)
  local right_limbs = quota_parse(right)
  local result = {}
  local borrow = 0
  for index = 1, quota_limb_count do
    local difference = left_limbs[index] - right_limbs[index] - borrow
    if difference < 0 then
      difference = difference + quota_base
      borrow = 1
    else
      borrow = 0
    end
    result[index] = difference
  end
  return quota_format(result), nil
end

local function quota_from_count(value)
  if value < 0 or value > 9007199254740991 then
    return nil
  end
  return string.format("%.0f", value)
end

local function quota_server_time()
  local value = redis.call("TIME")
  local milliseconds = tonumber(value[1]) * 1000 + math.floor(tonumber(value[2]) / 1000)
  local microseconds = value[1] .. string.format("%06d", tonumber(value[2]))
  return milliseconds, microseconds
end

local function quota_time_milliseconds()
  local milliseconds, _ = quota_server_time()
  return milliseconds
end

local function quota_microseconds_to_milliseconds(value)
  if quota_parse(value) == nil then
    return nil, "invalid microsecond timestamp"
  end
  local quotient = "0"
  local remainder = value
  if #value > 3 then
    quotient = string.sub(value, 1, #value - 3)
    remainder = string.sub(value, #value - 2)
  end
  if tonumber(remainder) ~= 0 then
    return quota_add(quotient, "1")
  end
  return quotient, nil
end

local function quota_calendar_interval(schedule, now)
  if type(schedule) ~= "string" or schedule == "" then
    return nil, nil, "calendar schedule is empty"
  end
  for encoded in string.gmatch(schedule, "([^,]+)") do
    local start_text, finish_text = string.match(encoded, "^(%d+):(%d+)$")
    local start = tonumber(start_text)
    local finish = tonumber(finish_text)
    if start == nil or finish == nil or start >= finish then
      return nil, nil, "calendar schedule is invalid"
    end
    if now >= start and now < finish then
      return string.format("%.0f", start), string.format("%.0f", finish), nil
    end
  end
  return nil, nil, "calendar schedule does not cover server time"
end

local function quota_calendar_observe(meta_key, interval_start, interval_finish)
  local stored_start = redis.call("HGET", meta_key, "calendar_start")
  local stored_finish = redis.call("HGET", meta_key, "calendar_end")
  if stored_start ~= interval_start or stored_finish ~= interval_finish then
    return "0", "0", "0", true
  end
  return redis.call("HGET", meta_key, "used") or "0",
    redis.call("HGET", meta_key, "known") or "0",
    redis.call("HGET", meta_key, "incomplete") or "0", false
end

local function quota_bucket_observe(meta_key, capacity, refill_amount, refill_period, now)
  if quota_parse(capacity) == nil or quota_parse(refill_amount) == nil
      or refill_period == nil or refill_period <= 0 then
    return nil, nil, "invalid token-bucket parameters"
  end
  local tokens = redis.call("HGET", meta_key, "tokens") or capacity
  if quota_parse(tokens) == nil or quota_compare(tokens, capacity) > 0 then
    return nil, nil, "invalid token-bucket state"
  end
  local last_refill_text = redis.call("HGET", meta_key, "last_refill_ms")
  local last_refill = now
  if last_refill_text ~= false then
    last_refill = tonumber(last_refill_text)
    if last_refill == nil or last_refill > now then
      return nil, nil, "invalid token-bucket refill time"
    end
  end
  local periods = math.floor((now - last_refill) / refill_period)
  if periods > 0 then
    local period_quantity = quota_from_count(periods)
    local refill, multiply_error = quota_multiply(refill_amount, period_quantity)
    if refill == nil then
      tokens = capacity
      last_refill = now
    else
      local replenished, add_error = quota_add(tokens, refill)
      if replenished == nil then
        tokens = capacity
        last_refill = now
      elseif quota_compare(replenished, capacity) >= 0 then
        tokens = capacity
        last_refill = now
      else
        tokens = replenished
        last_refill = last_refill + periods * refill_period
      end
    end
  end
  return tokens, string.format("%.0f", last_refill), nil
end

local function quota_gcra_observe(meta_key, emission_interval, burst_tolerance, now_microseconds)
  if quota_parse(emission_interval) == nil or quota_compare(emission_interval, "0") <= 0
      or quota_parse(burst_tolerance) == nil then
    return nil, nil, nil, "invalid GCRA parameters"
  end
  local tat = redis.call("HGET", meta_key, "tat_us") or now_microseconds
  if quota_parse(tat) == nil then
    return nil, nil, nil, "invalid GCRA state"
  end
  local allow_at = "0"
  if quota_compare(tat, burst_tolerance) > 0 then
    local subtract_error = nil
    allow_at, subtract_error = quota_subtract(tat, burst_tolerance)
    if allow_at == nil then
      return nil, nil, nil, subtract_error
    end
  end
  local allowed = quota_compare(now_microseconds, allow_at) >= 0
  local base = tat
  if quota_compare(now_microseconds, tat) > 0 then
    base = now_microseconds
  end
  local next_tat, add_error = quota_add(base, emission_interval)
  if next_tat == nil then
    return nil, nil, nil, add_error
  end
  return allowed, allow_at, next_tat, nil
end

local function quota_check_access(precondition_count, key_offset, argument_offset, now)
  for index = 1, precondition_count do
    local key = KEYS[key_offset + index]
    local current_argument = argument_offset + (index - 1) * 5
    local kind = ARGV[current_argument + 1]
    local field = ARGV[current_argument + 2]
    local expected = ARGV[current_argument + 3]
    local failure = ARGV[current_argument + 4]
    local reason = ARGV[current_argument + 5]
    local passed = false

    if kind == "hash_equal" then
      passed = redis.call("HGET", key, field) == expected
    elseif kind == "string_equal" then
      passed = redis.call("GET", key) == expected
    elseif kind == "key_absent" then
      passed = redis.call("EXISTS", key) == 0
    elseif kind == "set_member" then
      passed = redis.call("SISMEMBER", key, expected) == 1
    elseif kind == "hash_not_before" then
      local raw = redis.call("HGET", key, field)
      if raw ~= false then
        local timestamp = tonumber(raw)
        if timestamp == nil then
          return "unavailable", "invalid access time projection"
        end
        passed = timestamp <= now
      end
    elseif kind == "hash_expires_after" then
      local raw = redis.call("HGET", key, field)
      if raw == false or raw == "" then
        passed = true
      else
        local timestamp = tonumber(raw)
        if timestamp == nil then
          return "unavailable", "invalid access time projection"
        end
        passed = timestamp > now
      end
    else
      return nil, "unsupported admission precondition"
    end
    if not passed then
      return failure, reason
    end
  end
  return "allowed", ""
end

local function quota_prune_request(meta_key, event_key, cutoff)
  local expired = redis.call("ZRANGEBYSCORE", event_key, "-inf", cutoff)
  if #expired == 0 then
    return redis.call("HGET", meta_key, "used") or "0",
      redis.call("HGET", meta_key, "known") or "0", nil
  end

  local expired_count = quota_from_count(#expired)
  if expired_count == nil then
    return nil, nil, "request event count exceeds exact Lua range"
  end
  local used, used_error = quota_subtract(redis.call("HGET", meta_key, "used") or "0", expired_count)
  local known, known_error = quota_subtract(redis.call("HGET", meta_key, "known") or "0", expired_count)
  if used == nil then
    return nil, nil, used_error
  end
  if known == nil then
    return nil, nil, known_error
  end

  redis.call("ZREMRANGEBYSCORE", event_key, "-inf", cutoff)
  redis.call("HSET", meta_key, "used", used, "known", known)
  return used, known, nil
end

local function quota_prune_actual(meta_key, event_key, value_key, cutoff)
  local expired = redis.call("ZRANGEBYSCORE", event_key, "-inf", cutoff)
  if #expired == 0 then
    return redis.call("HGET", meta_key, "used") or "0",
      redis.call("HGET", meta_key, "known") or "0", nil
  end

  local amount_sum = "0"
  local dispatch_sum = "0"
  for _, member in ipairs(expired) do
    local encoded = redis.call("HGET", value_key, member)
    if encoded == false then
      return nil, nil, "missing settlement event value"
    end
    local separator = string.find(encoded, "|", 1, true)
    if separator == nil then
      return nil, nil, "invalid settlement event value"
    end
    local amount = string.sub(encoded, 1, separator - 1)
    local dispatches = string.sub(encoded, separator + 1)
    local next_amount, amount_error = quota_add(amount_sum, amount)
    local next_dispatches, dispatch_error = quota_add(dispatch_sum, dispatches)
    if next_amount == nil then
      return nil, nil, amount_error
    end
    if next_dispatches == nil then
      return nil, nil, dispatch_error
    end
    amount_sum = next_amount
    dispatch_sum = next_dispatches
  end

  local used, used_error = quota_subtract(redis.call("HGET", meta_key, "used") or "0", amount_sum)
  local known, known_error = quota_subtract(redis.call("HGET", meta_key, "known") or "0", dispatch_sum)
  if used == nil then
    return nil, nil, used_error
  end
  if known == nil then
    return nil, nil, known_error
  end

  redis.call("ZREMRANGEBYSCORE", event_key, "-inf", cutoff)
  for start = 1, #expired, 1000 do
    local finish = math.min(start + 999, #expired)
    redis.call("HDEL", value_key, unpack(expired, start, finish))
  end
  redis.call("HSET", meta_key, "used", used, "known", known)
  return used, known, nil
end

local function quota_reset_at(event_key, window_milliseconds)
  local first = redis.call("ZRANGE", event_key, 0, 0, "WITHSCORES")
  if #first == 0 then
    return ""
  end
  return string.format("%.0f", tonumber(first[2]) + window_milliseconds)
end

local function quota_request_retry_at(event_key, used, limit, window_milliseconds)
  local overage, subtract_error = quota_subtract(used, limit)
  if overage == nil then
    return nil, subtract_error
  end
  local required, add_error = quota_add(overage, "1")
  if required == nil then
    return nil, add_error
  end
  local event_count = redis.call("ZCARD", event_key)
  local event_quantity = quota_from_count(event_count)
  if event_quantity == nil or quota_compare(required, event_quantity) > 0 then
    return nil, "request counter differs from event log"
  end
  local index = tonumber(required) - 1
  local event = redis.call("ZRANGE", event_key, index, index, "WITHSCORES")
  if #event ~= 2 then
    return nil, "request retry event is missing"
  end
  return string.format("%.0f", tonumber(event[2]) + window_milliseconds), nil
end

local function quota_actual_retry_at(event_key, value_key, used, limit, window_milliseconds)
  local remaining = used
  local events = redis.call("ZRANGE", event_key, 0, -1, "WITHSCORES")
  for index = 1, #events, 2 do
    local encoded = redis.call("HGET", value_key, events[index])
    if encoded == false then
      return nil, "missing settlement event value"
    end
    local separator = string.find(encoded, "|", 1, true)
    if separator == nil then
      return nil, "invalid settlement event value"
    end
    local amount = string.sub(encoded, 1, separator - 1)
    local subtract_error = nil
    remaining, subtract_error = quota_subtract(remaining, amount)
    if remaining == nil then
      return nil, subtract_error
    end
    if quota_compare(remaining, limit) < 0 then
      return string.format("%.0f", tonumber(events[index + 1]) + window_milliseconds), nil
    end
  end
  return nil, "actual counter differs from event log"
end
