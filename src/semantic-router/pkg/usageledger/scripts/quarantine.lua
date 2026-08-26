if #KEYS ~= 2 or #ARGV ~= 5 then
  return redis.error_reply("USAGE_INVALID quarantine envelope")
end

local source = KEYS[1]
local quarantine = KEYS[2]
local group = ARGV[1]
local source_id = ARGV[2]
local reason = ARGV[3]
local payload = ARGV[4]
local digest = ARGV[5]

local quarantine_type = redis.call("TYPE", quarantine)
if type(quarantine_type) == "table" then
  quarantine_type = quarantine_type.ok
end
if quarantine_type ~= "none" and quarantine_type ~= "stream" then
  return redis.error_reply("USAGE_CORRUPT quarantine key type")
end

local pending = redis.call("XPENDING", source, group, source_id, source_id, 1)
if #pending == 0 then
  return {0, ""}
end

local now = redis.call("TIME")
local quarantined_at_ms = tostring(now[1] * 1000 + math.floor(now[2] / 1000))
local quarantine_id = redis.call("XADD", quarantine, "*",
  "source_id", source_id,
  "reason", reason,
  "payload", payload,
  "payload_digest", digest,
  "quarantined_at_ms", quarantined_at_ms)
local acknowledged = redis.call("XACK", source, group, source_id)
if acknowledged ~= 1 then
  return redis.error_reply("USAGE_CONFLICT quarantine lost pending ownership")
end
redis.call("XDEL", source, source_id)
return {1, quarantine_id}
