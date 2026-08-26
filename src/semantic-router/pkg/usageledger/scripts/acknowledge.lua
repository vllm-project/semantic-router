if #KEYS ~= 1 or #ARGV < 2 then
  return redis.error_reply("USAGE_INVALID acknowledgement envelope")
end

local source = KEYS[1]
local group = ARGV[1]
local ids = {}
for index = 2, #ARGV do
  ids[#ids + 1] = ARGV[index]
end

local acknowledged = redis.call("XACK", source, group, unpack(ids))
if acknowledged > 0 then
  redis.call("XDEL", source, unpack(ids))
end
return acknowledged
