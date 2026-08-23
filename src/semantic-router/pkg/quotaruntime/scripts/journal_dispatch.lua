local admission_digest = ARGV[2]
local now = quota_time_milliseconds()

if #KEYS ~= 3 or #ARGV ~= 5 then
  return redis.error_reply("QUOTA_INVALID dispatch journal shape")
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

local encoded = ARGV[4] .. "|" .. ARGV[5]
local previous = redis.call("HGET", KEYS[3], ARGV[3])
if previous ~= false then
  if previous ~= encoded then
    return redis.error_reply("QUOTA_CONFLICT dispatch identity was reused")
  end
  return {"journaled", "1", string.format("%.0f", now)}
end
redis.call("HSET", KEYS[3], ARGV[3], encoded)
return {"journaled", "0", string.format("%.0f", now)}
