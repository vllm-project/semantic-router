local precondition_count = tonumber(ARGV[1])
local now = quota_time_milliseconds()
if #KEYS ~= precondition_count or #ARGV ~= 1 + precondition_count * 5 then
  return redis.error_reply("QUOTA_INVALID access check shape")
end
local disposition, reason = quota_check_access(precondition_count, 0, 1, now)
if disposition == nil then
  return redis.error_reply("QUOTA_INVALID " .. reason)
end
return {disposition, string.format("%.0f", now), reason}
