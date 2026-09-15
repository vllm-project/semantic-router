package sessiontools

const (
	redisLoadStatusMissing   = "missing"
	redisLoadStatusCorrupted = "corrupted"
	redisLoadStatusExpired   = "expired"
	redisLoadStatusFound     = "found"

	redisCASStatusApplied   = "applied"
	redisCASStatusMismatch  = "mismatch"
	redisCASStatusCorrupted = "corrupted"

	redisLoadMetadataReplyLength = 4
	redisLoadFoundReplyLength    = 6
)

// Every script keeps the state hash and its LRU/expiry indexes consistent in
// one Redis linearization point. Session and quota index keys are opaque hashes
// assembled by store_redis.go; payload is the identity-only State JSON.
const redisLoadScript = `
local state_key = KEYS[1]
local global_lru = KEYS[2]
local global_expiry = KEYS[3]
local ttl_ms = tonumber(ARGV[1])
local key_prefix = ARGV[2]

local function owns_key(key)
  return type(key) == "string" and string.sub(key, 1, string.len(key_prefix)) == key_prefix
end

local function server_time_ms()
  local value = redis.call("TIME")
  return tonumber(value[1]) * 1000 + math.floor(tonumber(value[2]) / 1000)
end

local function extend_ttl(key, duration_ms)
  local current = redis.call("PTTL", key)
  if current < duration_ms then
    redis.call("PEXPIRE", key, duration_ms)
  end
end

local function ensure_zset(key, protected_key)
  if not owns_key(key) then
    return false
  end
  local key_type = redis.call("TYPE", key).ok
  if key_type == "none" or key_type == "zset" then
    return true
  end
  -- Index keys are owned by this store. Remove an accidentally reused scalar
  -- key before recreating the index, but never delete the state hash itself
  -- when a corrupted field points back at it.
  if key ~= protected_key then
    redis.call("DEL", key)
    return true
  end
  return false
end

local function remove_index_member(index_key, member, protected_key)
  if not owns_key(index_key) then
    return
  end
  if not ensure_zset(index_key, protected_key) then
    return
  end
  redis.call("ZREM", index_key, member)
  if redis.call("ZCARD", index_key) == 0 then
    redis.call("DEL", index_key)
  end
end

local function remove_state(key)
  if not owns_key(key) then
    return
  end
  local quota_lru
  local quota_expiry
  if redis.call("TYPE", key).ok == "hash" then
    quota_lru = redis.call("HGET", key, "quota_lru")
    quota_expiry = redis.call("HGET", key, "quota_expiry")
  end
  redis.call("DEL", key)
  remove_index_member(global_lru, key, key)
  remove_index_member(global_expiry, key, key)
  if quota_lru then
    remove_index_member(quota_lru, key, key)
  end
  if quota_expiry then
    remove_index_member(quota_expiry, key, key)
  end
end

local state_type = redis.call("TYPE", state_key).ok
if state_type ~= "none" and state_type ~= "hash" then
  redis.call("DEL", state_key)
  remove_index_member(global_lru, state_key, state_key)
  remove_index_member(global_expiry, state_key, state_key)
  return {"corrupted"}
end
local revision = redis.call("HGET", state_key, "revision")
if (not revision or revision == "") and redis.call("EXISTS", state_key) == 1 then
  remove_state(state_key)
  return {"corrupted"}
end
if not revision then
  local indexed_expiry
  if ensure_zset(global_expiry, state_key) then
    indexed_expiry = redis.call("ZSCORE", global_expiry, state_key)
  end
  remove_index_member(global_lru, state_key, state_key)
  remove_index_member(global_expiry, state_key, state_key)
  if indexed_expiry and tonumber(indexed_expiry) <= server_time_ms() then
    -- Redis may have removed the state hash via PEXPIRE before this load
    -- reached the script. The expiry index still lets us preserve the
    -- manager's explicit expiry receipt without returning stale state.
    return {"expired", "", "0", "0"}
  end
  return {"missing"}
end

local generation = redis.call("HGET", state_key, "generation")
local payload = redis.call("HGET", state_key, "payload")
local quota_lru = redis.call("HGET", state_key, "quota_lru")
local quota_expiry = redis.call("HGET", state_key, "quota_expiry")
local expires_at = tonumber(redis.call("HGET", state_key, "expires_at_ms") or "")
if not generation or generation == "" or
   not payload or payload == "" or
   not quota_lru or quota_lru == "" or
   not quota_expiry or quota_expiry == "" or
   not owns_key(quota_lru) or not owns_key(quota_expiry) or
   quota_lru == state_key or quota_expiry == state_key or
   quota_lru == quota_expiry or
   not expires_at then
  remove_state(state_key)
  return {"corrupted"}
end
local now = server_time_ms()
if expires_at <= now then
  remove_state(state_key)
  return {"expired", "", revision, generation}
end

if not ensure_zset(global_lru, state_key) or
   not ensure_zset(global_expiry, state_key) then
  return {"corrupted"}
end
local next_expiry = now + ttl_ms
redis.call("HSET", state_key, "expires_at_ms", next_expiry)
redis.call("PEXPIRE", state_key, ttl_ms)
redis.call("ZADD", global_lru, now, state_key)
redis.call("ZADD", global_expiry, next_expiry, state_key)
extend_ttl(global_lru, ttl_ms)
extend_ttl(global_expiry, ttl_ms)
if quota_lru and quota_expiry and
   ensure_zset(quota_lru, state_key) and
   ensure_zset(quota_expiry, state_key) then
  redis.call("ZADD", quota_lru, now, state_key)
  redis.call("ZADD", quota_expiry, next_expiry, state_key)
  extend_ttl(quota_lru, ttl_ms)
  extend_ttl(quota_expiry, ttl_ms)
end
return {"found", payload, revision, generation, tostring(now), tostring(next_expiry)}
`

const redisCompareAndSwapScript = `
local state_key = KEYS[1]
local global_lru = KEYS[2]
local global_expiry = KEYS[3]
local requested_quota_lru = KEYS[4]
local requested_quota_expiry = KEYS[5]
local generation_key = KEYS[6]

local expected_revision = ARGV[1]
local next_revision = ARGV[2]
local payload = ARGV[3]
local ttl_ms = tonumber(ARGV[4])
local max_sessions = tonumber(ARGV[5])
local max_identity_sessions = tonumber(ARGV[6])
local key_prefix = ARGV[7]

local function owns_key(key)
  return type(key) == "string" and string.sub(key, 1, string.len(key_prefix)) == key_prefix
end

local function server_time_ms()
  local value = redis.call("TIME")
  return tonumber(value[1]) * 1000 + math.floor(tonumber(value[2]) / 1000)
end

local function extend_ttl(key, duration_ms)
  local current = redis.call("PTTL", key)
  if current < duration_ms then
    redis.call("PEXPIRE", key, duration_ms)
  end
end

local function ensure_zset(key, protected_key)
  if not owns_key(key) then
    return false
  end
  local key_type = redis.call("TYPE", key).ok
  if key_type == "none" or key_type == "zset" then
    return true
  end
  if key ~= protected_key then
    redis.call("DEL", key)
    return true
  end
  return false
end

local function remove_index_member(index_key, member, protected_key)
  if not owns_key(index_key) then
    return
  end
  if not ensure_zset(index_key, protected_key) then
    return
  end
  redis.call("ZREM", index_key, member)
  if redis.call("ZCARD", index_key) == 0 then
    redis.call("DEL", index_key)
  end
end

local function remove_state(key)
  if not owns_key(key) then
    return
  end
  local quota_lru
  local quota_expiry
  if redis.call("TYPE", key).ok == "hash" then
    quota_lru = redis.call("HGET", key, "quota_lru")
    quota_expiry = redis.call("HGET", key, "quota_expiry")
  end
  redis.call("DEL", key)
  remove_index_member(global_lru, key, key)
  remove_index_member(global_expiry, key, key)
  if quota_lru then
    remove_index_member(quota_lru, key, key)
  end
  if quota_expiry then
    remove_index_member(quota_expiry, key, key)
  end
end

-- A state hash can expire before its quota indexes do. Before evicting a
-- member from a quota index, verify that the current state still points at
-- that exact index pair. Without this check, a delete-and-recreate cycle can
-- leave an old quota member that accidentally evicts the newly-created state
-- (an ABA race across Redis' independently-expiring keys).
local function remove_slot_member(lru_key, expiry_key, member, now, require_expired, check_quota)
  if not owns_key(member) then
    -- A damaged index must never make this store inspect or delete a key it
    -- does not own. Remove only the bad references from our own indexes.
    remove_index_member(lru_key, member, state_key)
    remove_index_member(expiry_key, member, state_key)
    return
  end
  local state_type = redis.call("TYPE", member).ok
  if state_type == "none" then
    remove_index_member(lru_key, member, state_key)
    remove_index_member(expiry_key, member, state_key)
    return
  end
  if state_type ~= "hash" then
    -- The index member is stale or the state key is corrupt. Remove only the
    -- index references; the state key itself is handled by its own load path.
    remove_index_member(lru_key, member, state_key)
    remove_index_member(expiry_key, member, state_key)
    return
  end

  if check_quota then
    local member_lru = redis.call("HGET", member, "quota_lru")
    local member_expiry = redis.call("HGET", member, "quota_expiry")
    if not member_lru or not member_expiry then
      remove_state(member)
      remove_index_member(lru_key, member, state_key)
      remove_index_member(expiry_key, member, state_key)
      return
    end
    if member_lru == member or member_expiry == member or member_lru == member_expiry then
      remove_state(member)
      remove_index_member(lru_key, member, state_key)
      remove_index_member(expiry_key, member, state_key)
      return
    end
    if member_lru ~= lru_key or member_expiry ~= expiry_key then
      -- This member belongs to a different quota bucket after a recreate. Do
      -- not delete the live state; only discard the stale index references.
      remove_index_member(lru_key, member, state_key)
      remove_index_member(expiry_key, member, state_key)
      return
    end
  end

  if require_expired then
    local expires_at = tonumber(redis.call("HGET", member, "expires_at_ms") or "")
    if expires_at and expires_at > now then
      -- The expiry index score was stale. Repair it and let the bounded loop
      -- inspect the next candidate instead of deleting a live state.
      redis.call("ZADD", expiry_key, expires_at, member)
      return
    end
  end
  remove_state(member)
  remove_index_member(lru_key, member, state_key)
  remove_index_member(expiry_key, member, state_key)
end

local function ensure_slot(lru_key, expiry_key, limit, now, check_quota)
  if not ensure_zset(lru_key, state_key) or
     not ensure_zset(expiry_key, state_key) then
    return false
  end
  while redis.call("ZCARD", lru_key) >= limit do
    local expired = redis.call("ZRANGEBYSCORE", expiry_key, "-inf", now, "LIMIT", 0, 1)
    if #expired > 0 then
      remove_slot_member(lru_key, expiry_key, expired[1], now, true, check_quota)
    else
      local oldest = redis.call("ZRANGE", lru_key, 0, 0)
      if #oldest == 0 then
        return true
      end
      remove_slot_member(lru_key, expiry_key, oldest[1], now, false, check_quota)
    end
  end
  return true
end

local now = server_time_ms()
if not ensure_zset(global_lru, state_key) or
   not ensure_zset(global_expiry, state_key) or
   not ensure_zset(requested_quota_lru, state_key) or
   not ensure_zset(requested_quota_expiry, state_key) then
  return {"mismatch"}
end
local generation_type = redis.call("TYPE", generation_key).ok
if generation_type ~= "none" and generation_type ~= "string" then
  return {"corrupted"}
end
local generation_value = redis.call("GET", generation_key)
if generation_value and not string.match(generation_value, "^%d+$") then
  return {"corrupted"}
end
local state_type = redis.call("TYPE", state_key).ok
if state_type ~= "none" and state_type ~= "hash" then
  redis.call("DEL", state_key)
  remove_index_member(global_lru, state_key, state_key)
  remove_index_member(global_expiry, state_key, state_key)
  state_type = "none"
end
local revision = redis.call("HGET", state_key, "revision")
local exists = state_type ~= "none"
local expires_at = tonumber(redis.call("HGET", state_key, "expires_at_ms") or "")
if exists and (not revision or revision == "" or not expires_at or expires_at <= now) then
  remove_state(state_key)
  revision = nil
  exists = false
end

local generation
local quota_lru
local quota_expiry
if expected_revision == "0" then
  if revision or exists then
    return {"mismatch"}
  end
  if not ensure_slot(requested_quota_lru, requested_quota_expiry, max_identity_sessions, now, true) or
     not ensure_slot(global_lru, global_expiry, max_sessions, now, false) then
    return {"mismatch"}
  end
  redis.call("INCR", generation_key)
  -- GET returns the decimal bulk string, preserving all uint64 generation
  -- bits even when the value exceeds Lua's exact integer range.
  generation = redis.call("GET", generation_key)
  quota_lru = requested_quota_lru
  quota_expiry = requested_quota_expiry
else
  if not revision or revision ~= expected_revision then
    return {"mismatch"}
  end
  generation = redis.call("HGET", state_key, "generation")
  quota_lru = redis.call("HGET", state_key, "quota_lru")
  quota_expiry = redis.call("HGET", state_key, "quota_expiry")
  if not generation or generation == "" or
     not quota_lru or quota_lru == "" or
     not quota_expiry or quota_expiry == "" or
     not owns_key(quota_lru) or not owns_key(quota_expiry) or
     quota_lru == state_key or quota_expiry == state_key or
     quota_lru == quota_expiry then
    remove_state(state_key)
    return {"corrupted"}
  end
end

if not ensure_zset(global_lru, state_key) or
   not ensure_zset(global_expiry, state_key) or
   not ensure_zset(quota_lru, state_key) or
   not ensure_zset(quota_expiry, state_key) then
  return {"mismatch"}
end
local next_expiry = now + ttl_ms
redis.call("HSET", state_key,
  "payload", payload,
  "revision", next_revision,
  "generation", generation,
  "expires_at_ms", next_expiry,
  "quota_lru", quota_lru,
  "quota_expiry", quota_expiry)
redis.call("PEXPIRE", state_key, ttl_ms)
redis.call("ZADD", global_lru, now, state_key)
redis.call("ZADD", global_expiry, next_expiry, state_key)
extend_ttl(global_lru, ttl_ms)
extend_ttl(global_expiry, ttl_ms)
redis.call("ZADD", quota_lru, now, state_key)
redis.call("ZADD", quota_expiry, next_expiry, state_key)
extend_ttl(quota_lru, ttl_ms)
extend_ttl(quota_expiry, ttl_ms)
return {"applied", next_revision, tostring(generation), tostring(now), tostring(next_expiry)}
`

const redisDeleteScript = `
local state_key = KEYS[1]
local global_lru = KEYS[2]
local global_expiry = KEYS[3]
local key_prefix = ARGV[1]

local function owns_key(key)
  return type(key) == "string" and string.sub(key, 1, string.len(key_prefix)) == key_prefix
end

local function ensure_zset(key, protected_key)
  if not owns_key(key) then
    return false
  end
  local key_type = redis.call("TYPE", key).ok
  if key_type == "none" or key_type == "zset" then
    return true
  end
  if key ~= protected_key then
    redis.call("DEL", key)
    return true
  end
  return false
end

local function remove_index_member(index_key, member, protected_key)
  if not owns_key(index_key) then
    return
  end
  if not ensure_zset(index_key, protected_key) then
    return
  end
  redis.call("ZREM", index_key, member)
  if redis.call("ZCARD", index_key) == 0 then
    redis.call("DEL", index_key)
  end
end

ensure_zset(global_lru, state_key)
ensure_zset(global_expiry, state_key)
local state_type = redis.call("TYPE", state_key).ok
local quota_lru
local quota_expiry
if state_type == "hash" then
  quota_lru = redis.call("HGET", state_key, "quota_lru")
  quota_expiry = redis.call("HGET", state_key, "quota_expiry")
end
redis.call("DEL", state_key)
remove_index_member(global_lru, state_key, state_key)
remove_index_member(global_expiry, state_key, state_key)
if quota_lru then
  remove_index_member(quota_lru, state_key, state_key)
end
if quota_expiry then
  remove_index_member(quota_expiry, state_key, state_key)
end
return 1
`

const redisDeleteIfTokenScript = `
local state_key = KEYS[1]
local key_prefix = ARGV[3]

local function owns_key(key)
  return type(key) == "string" and string.sub(key, 1, string.len(key_prefix)) == key_prefix
end

local function ensure_zset(key, protected_key)
  if not owns_key(key) then
    return false
  end
  local key_type = redis.call("TYPE", key).ok
  if key_type == "none" or key_type == "zset" then
    return true
  end
  if key ~= protected_key then
    redis.call("DEL", key)
    return true
  end
  return false
end

local function remove_index_member(index_key, member, protected_key)
  if not owns_key(index_key) then
    return
  end
  if not ensure_zset(index_key, protected_key) then
    return
  end
  redis.call("ZREM", index_key, member)
  if redis.call("ZCARD", index_key) == 0 then
    redis.call("DEL", index_key)
  end
end

ensure_zset(KEYS[2], state_key)
ensure_zset(KEYS[3], state_key)
if redis.call("TYPE", state_key).ok ~= "hash" then
  return 0
end
local revision = redis.call("HGET", state_key, "revision")
local generation = redis.call("HGET", state_key, "generation")
if not revision or revision ~= ARGV[1] then
  return 0
end
if ARGV[2] ~= "0" and (not generation or generation ~= ARGV[2]) then
  return 0
end
local quota_lru = redis.call("HGET", state_key, "quota_lru")
local quota_expiry = redis.call("HGET", state_key, "quota_expiry")
redis.call("DEL", state_key)
remove_index_member(KEYS[2], state_key, state_key)
remove_index_member(KEYS[3], state_key, state_key)
if quota_lru then
  remove_index_member(quota_lru, state_key, state_key)
end
if quota_expiry then
  remove_index_member(quota_expiry, state_key, state_key)
end
return 1
`

const redisDeleteRawIfCurrentScript = `
local state_key = KEYS[1]
local key_prefix = ARGV[4]

local function owns_key(key)
  return type(key) == "string" and string.sub(key, 1, string.len(key_prefix)) == key_prefix
end

local function ensure_zset(key, protected_key)
  if not owns_key(key) then
    return false
  end
  local key_type = redis.call("TYPE", key).ok
  if key_type == "none" or key_type == "zset" then
    return true
  end
  if key ~= protected_key then
    redis.call("DEL", key)
    return true
  end
  return false
end

local function remove_index_member(index_key, member, protected_key)
  if not owns_key(index_key) then
    return
  end
  if not ensure_zset(index_key, protected_key) then
    return
  end
  redis.call("ZREM", index_key, member)
  if redis.call("ZCARD", index_key) == 0 then
    redis.call("DEL", index_key)
  end
end

ensure_zset(KEYS[2], state_key)
ensure_zset(KEYS[3], state_key)
if redis.call("TYPE", state_key).ok ~= "hash" then
  return 0
end
if redis.call("HGET", state_key, "payload") ~= ARGV[1] or
   redis.call("HGET", state_key, "revision") ~= ARGV[2] or
   redis.call("HGET", state_key, "generation") ~= ARGV[3] then
  return 0
end
local quota_lru = redis.call("HGET", state_key, "quota_lru")
local quota_expiry = redis.call("HGET", state_key, "quota_expiry")
redis.call("DEL", state_key)
remove_index_member(KEYS[2], state_key, state_key)
remove_index_member(KEYS[3], state_key, state_key)
if quota_lru then
  remove_index_member(quota_lru, state_key, state_key)
end
if quota_expiry then
  remove_index_member(quota_expiry, state_key, state_key)
end
return 1
`
