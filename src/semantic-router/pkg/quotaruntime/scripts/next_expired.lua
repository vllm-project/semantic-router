local now = quota_time_milliseconds()

if #KEYS ~= 1 or #ARGV ~= 0 then
  return redis.error_reply("QUOTA_INVALID next expired admission shape")
end
local key_type = redis.call("TYPE", KEYS[1])
if type(key_type) == "table" then
  key_type = key_type.ok
end
if key_type ~= "none" and key_type ~= "zset" then
  return redis.error_reply("QUOTA_CORRUPT pending admission index type")
end

local expired = redis.call("ZRANGEBYSCORE", KEYS[1], "-inf", now, "WITHSCORES", "LIMIT", 0, 1)
if #expired == 0 then
  return {"next_expired", string.format("%.0f", now), "", ""}
end
return {"next_expired", string.format("%.0f", now), expired[1], expired[2]}
