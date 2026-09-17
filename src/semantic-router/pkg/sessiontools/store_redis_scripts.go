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

// Mutation scripts keep the state hash and its LRU/expiry indexes consistent
// in one Redis linearization point; the load script is deliberately read-only.
// Session and quota index keys are opaque hashes assembled by store_redis.go;
// payload is the identity-only State JSON.
const redisLoadScript = `
local state_key = KEYS[1]
local global_lru = KEYS[2]
local global_expiry = KEYS[3]
local ttl_ms = tonumber(ARGV[1])
local key_prefix = ARGV[2]

local function canonical_positive_decimal(value)
  return type(value) == "string" and
         string.match(value, "^[1-9]%d*$") ~= nil
end

local function valid_quota_indexes(lru_key, expiry_key)
  if type(lru_key) ~= "string" or type(expiry_key) ~= "string" then
    return false
  end
  local identity_prefix = key_prefix .. "identity:"
  local prefix_length = string.len(identity_prefix)
  if string.sub(lru_key, 1, prefix_length) ~= identity_prefix or
     string.sub(lru_key, -4) ~= ":lru" then
    return false
  end
  local digest = string.sub(lru_key, prefix_length + 1, string.len(lru_key) - 4)
  if string.len(digest) ~= 64 or not string.match(digest, "^[0-9a-f]+$") then
    return false
  end
  return expiry_key == identity_prefix .. digest .. ":expiry"
end

local function server_time_ms()
  local value = redis.call("TIME")
  return tonumber(value[1]) * 1000 + math.floor(tonumber(value[2]) / 1000)
end

local state_type = redis.call("TYPE", state_key).ok
if state_type ~= "none" and state_type ~= "hash" then
  return {"corrupted"}
end
if state_type == "none" then
  local indexed_expiry
  if redis.call("TYPE", global_expiry).ok == "zset" then
    indexed_expiry = redis.call("ZSCORE", global_expiry, state_key)
  end
  if indexed_expiry and tonumber(indexed_expiry) <= server_time_ms() then
    -- Redis may have removed the state hash via PEXPIRE before this load
    -- reached the script. The expiry index still lets us preserve the
    -- manager's explicit expiry receipt without returning stale state.
    return {"expired", "", "0", "0"}
  end
  return {"missing"}
end

local revision = redis.call("HGET", state_key, "revision")
local generation = redis.call("HGET", state_key, "generation")
local payload = redis.call("HGET", state_key, "payload")
local quota_lru = redis.call("HGET", state_key, "quota_lru")
local quota_expiry = redis.call("HGET", state_key, "quota_expiry")
local expires_at = tonumber(redis.call("HGET", state_key, "expires_at_ms") or "")
if not canonical_positive_decimal(revision) or
   not canonical_positive_decimal(generation) or
   not payload or payload == "" or
   not quota_lru or quota_lru == "" or
   not quota_expiry or quota_expiry == "" or
   not valid_quota_indexes(quota_lru, quota_expiry) or
   not expires_at then
  return {"corrupted", payload or "", revision or "", generation or ""}
end
local now = server_time_ms()
if expires_at <= now then
  return {"expired", "", revision, generation}
end

-- Load is deliberately read-only. A successful manager turn always performs
-- CAS, which refreshes state and all four indexes in one trusted operation.
-- Returning the previous commit time keeps validation accurate without
-- extending a state that the current request may later reject.
local last_seen = expires_at - ttl_ms
if redis.call("TYPE", global_lru).ok == "zset" then
  local indexed_last_seen = tonumber(redis.call("ZSCORE", global_lru, state_key) or "")
  if indexed_last_seen then
    last_seen = indexed_last_seen
  end
end
return {"found", payload, revision, generation, tostring(last_seen), tostring(expires_at)}
`

const redisCompareAndSwapScript = `
local state_key = KEYS[1]
local global_lru = KEYS[2]
local global_expiry = KEYS[3]
local requested_quota_lru = KEYS[4]
local requested_quota_expiry = KEYS[5]
local generation_key = KEYS[6]
local revision_key = KEYS[7]

local expected_revision = ARGV[1]
local payload = ARGV[2]
local ttl_ms = tonumber(ARGV[3])
local max_sessions = tonumber(ARGV[4])
local max_identity_sessions = tonumber(ARGV[5])
local key_prefix = ARGV[6]

local function canonical_positive_decimal(value)
  return type(value) == "string" and
         string.match(value, "^[1-9]%d*$") ~= nil
end

local function owns_key(key)
  return type(key) == "string" and string.sub(key, 1, string.len(key_prefix)) == key_prefix
end

local function valid_quota_indexes(lru_key, expiry_key)
  if type(lru_key) ~= "string" or type(expiry_key) ~= "string" then
    return false
  end
  local identity_prefix = key_prefix .. "identity:"
  local prefix_length = string.len(identity_prefix)
  if string.sub(lru_key, 1, prefix_length) ~= identity_prefix or
     string.sub(lru_key, -4) ~= ":lru" then
    return false
  end
  local digest = string.sub(lru_key, prefix_length + 1, string.len(lru_key) - 4)
  if string.len(digest) ~= 64 or not string.match(digest, "^[0-9a-f]+$") then
    return false
  end
  return expiry_key == identity_prefix .. digest .. ":expiry"
end

local function valid_state_key(key)
  if type(key) ~= "string" then
    return false
  end
  local state_prefix = key_prefix .. "state:"
  if string.sub(key, 1, string.len(state_prefix)) ~= state_prefix then
    return false
  end
  local digest = string.sub(key, string.len(state_prefix) + 1)
  return string.len(digest) == 64 and string.match(digest, "^[0-9a-f]+$") ~= nil
end

local function decimal_less(left, right)
  if string.len(left) ~= string.len(right) then
    return string.len(left) < string.len(right)
  end
  return left < right
end

local function increment_sequence(key)
  local result = redis.pcall("INCR", key)
  if type(result) == "table" and result.err then
    return nil
  end
  local value = redis.call("GET", key)
  if not canonical_positive_decimal(value) then
    return nil
  end
  return value
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

local function usable_index_pair(lru_key, expiry_key, allow_missing)
  if not owns_key(lru_key) or not owns_key(expiry_key) then
    return false
  end
  local lru_type = redis.call("TYPE", lru_key).ok
  local expiry_type = redis.call("TYPE", expiry_key).ok
  if lru_type == "none" and expiry_type == "none" then
    return allow_missing
  end
  return lru_type == "zset" and expiry_type == "zset"
end

local function remove_index_member(index_key, member)
  if not owns_key(index_key) or redis.call("TYPE", index_key).ok ~= "zset" then
    return
  end
  redis.call("ZREM", index_key, member)
end

local function remove_state(key, cleanup_quota)
  if not valid_state_key(key) then
    return
  end
  local quota_lru
  local quota_expiry
  if redis.call("TYPE", key).ok == "hash" then
    quota_lru = redis.call("HGET", key, "quota_lru")
    quota_expiry = redis.call("HGET", key, "quota_expiry")
  end
  redis.call("DEL", key)
  remove_index_member(global_lru, key)
  remove_index_member(global_expiry, key)
  if cleanup_quota and valid_quota_indexes(quota_lru, quota_expiry) then
    remove_index_member(quota_lru, key)
    remove_index_member(quota_expiry, key)
  end
end

-- A state hash can expire before its quota indexes do. Before evicting a
-- member from a quota index, verify that the current state still points at
-- that exact index pair. Without this check, a delete-and-recreate cycle can
-- leave an old quota member that accidentally evicts the newly-created state
-- (an ABA race across Redis' independently-expiring keys).
local function remove_slot_member(lru_key, expiry_key, member, now, require_expired, check_quota)
  if not valid_state_key(member) then
    -- A damaged index must never make this store inspect or delete a key it
    -- does not own. Remove only the bad references from our own indexes.
    remove_index_member(lru_key, member)
    remove_index_member(expiry_key, member)
    return
  end
  local state_type = redis.call("TYPE", member).ok
  if state_type == "none" then
    remove_index_member(lru_key, member)
    remove_index_member(expiry_key, member)
    return
  end
  if state_type ~= "hash" then
    -- The index member is stale or the state key is corrupt. Remove only the
    -- index references; the state key itself is handled by its own load path.
    remove_index_member(lru_key, member)
    remove_index_member(expiry_key, member)
    return
  end

  if check_quota then
    local member_lru = redis.call("HGET", member, "quota_lru")
    local member_expiry = redis.call("HGET", member, "quota_expiry")
    if not member_lru or not member_expiry then
      remove_state(member, false)
      remove_index_member(lru_key, member)
      remove_index_member(expiry_key, member)
      return
    end
    if member_lru == member or member_expiry == member or member_lru == member_expiry then
      remove_state(member, false)
      remove_index_member(lru_key, member)
      remove_index_member(expiry_key, member)
      return
    end
    if member_lru ~= lru_key or member_expiry ~= expiry_key then
      -- This member belongs to a different quota bucket after a recreate. Do
      -- not delete the live state; only discard the stale index references.
      remove_index_member(lru_key, member)
      remove_index_member(expiry_key, member)
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
  remove_state(member, check_quota)
  remove_index_member(lru_key, member)
  remove_index_member(expiry_key, member)
end

local function ensure_slot(lru_key, expiry_key, limit, now, check_quota)
  if not usable_index_pair(lru_key, expiry_key, true) then
    return false
  end
  local cleanup_budget = 64
  while redis.call("ZCARD", lru_key) >= limit do
    if cleanup_budget == 0 then
      return false
    end
    cleanup_budget = cleanup_budget - 1
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

local function has_live_state_member(index_key)
  local members = redis.call("ZRANGE", index_key, 0, 64)
  if #members > 64 then
    return true
  end
  for _, member in ipairs(members) do
    if valid_state_key(member) and redis.call("TYPE", member).ok == "hash" then
      return true
    end
  end
  return false
end

local now = server_time_ms()
if not valid_state_key(state_key) or
   not owns_key(global_lru) or
   not owns_key(global_expiry) or
   not valid_quota_indexes(requested_quota_lru, requested_quota_expiry) or
   not (expected_revision == "0" or canonical_positive_decimal(expected_revision)) or
   not ttl_ms or ttl_ms <= 0 or
   not max_sessions or max_sessions <= 0 or
   not max_identity_sessions or max_identity_sessions <= 0 then
  return {"corrupted"}
end
if not usable_index_pair(global_lru, global_expiry, true) or
   not usable_index_pair(requested_quota_lru, requested_quota_expiry, true) then
  return {"corrupted"}
end
local generation_type = redis.call("TYPE", generation_key).ok
if generation_type ~= "none" and generation_type ~= "string" then
  return {"corrupted"}
end
local generation_value = redis.call("GET", generation_key)
if generation_value and not canonical_positive_decimal(generation_value) then
  return {"corrupted"}
end
local revision_type = redis.call("TYPE", revision_key).ok
if revision_type ~= "none" and revision_type ~= "string" then
  return {"corrupted"}
end
local revision_value = redis.call("GET", revision_key)
if revision_value and not canonical_positive_decimal(revision_value) then
  return {"corrupted"}
end
if (generation_value and not revision_value) or
   (revision_value and not generation_value) then
  return {"corrupted"}
end
local state_type = redis.call("TYPE", state_key).ok
if state_type ~= "none" and state_type ~= "hash" then
  redis.call("DEL", state_key)
  remove_index_member(global_lru, state_key)
  remove_index_member(global_expiry, state_key)
  if expected_revision ~= "0" then
    return {"corrupted"}
  end
  state_type = "none"
end
local revision = redis.call("HGET", state_key, "revision")
local exists = state_type ~= "none"
local expires_at = tonumber(redis.call("HGET", state_key, "expires_at_ms") or "")
if exists and (not canonical_positive_decimal(revision) or
               not expires_at) then
  remove_state(state_key, false)
  if expected_revision ~= "0" then
    return {"corrupted"}
  end
  revision = nil
  exists = false
end
if exists and expires_at <= now then
  remove_state(state_key, false)
  revision = nil
  exists = false
end

local generation
local next_revision
local quota_lru
local quota_expiry
if expected_revision == "0" then
  if revision or exists then
    return {"mismatch"}
  end
  if not generation_value and has_live_state_member(global_lru) then
    return {"corrupted"}
  end
  generation = increment_sequence(generation_key)
  if not generation then
    return {"corrupted"}
  end
  next_revision = increment_sequence(revision_key)
  if not next_revision then
    return {"corrupted"}
  end
  if not ensure_slot(requested_quota_lru, requested_quota_expiry, max_identity_sessions, now, true) or
     not ensure_slot(global_lru, global_expiry, max_sessions, now, false) then
    return {"mismatch"}
  end
  quota_lru = requested_quota_lru
  quota_expiry = requested_quota_expiry
else
  if not revision or revision ~= expected_revision then
    return {"mismatch"}
  end
  generation = redis.call("HGET", state_key, "generation")
  quota_lru = redis.call("HGET", state_key, "quota_lru")
  quota_expiry = redis.call("HGET", state_key, "quota_expiry")
  if not canonical_positive_decimal(generation) or
     not quota_lru or quota_lru == "" or
     not quota_expiry or quota_expiry == "" or
     not valid_quota_indexes(quota_lru, quota_expiry) then
    remove_state(state_key, false)
    return {"corrupted"}
  end
  if quota_lru ~= requested_quota_lru or quota_expiry ~= requested_quota_expiry then
    remove_state(state_key, false)
    return {"corrupted"}
  end
  if not generation_value or not revision_value or
     decimal_less(generation_value, generation) or
     decimal_less(revision_value, expected_revision) then
    remove_state(state_key, true)
    return {"corrupted"}
  end
  if not usable_index_pair(global_lru, global_expiry, false) or
     not usable_index_pair(quota_lru, quota_expiry, false) or
     not redis.call("ZSCORE", global_lru, state_key) or
     not redis.call("ZSCORE", global_expiry, state_key) or
     not redis.call("ZSCORE", quota_lru, state_key) or
     not redis.call("ZSCORE", quota_expiry, state_key) then
    remove_state(state_key, true)
    return {"corrupted"}
  end
  next_revision = increment_sequence(revision_key)
  if not next_revision then
    return {"corrupted"}
  end
end

-- GET returns the exact decimal string; converting revisions through Lua's
-- double representation would lose precision for large sequence values.
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
return {"applied", next_revision, generation, tostring(now), tostring(next_expiry)}
`

const redisDeleteScript = `
local state_key = KEYS[1]
local global_lru = KEYS[2]
local global_expiry = KEYS[3]
local key_prefix = ARGV[1]

local function owns_key(key)
  return type(key) == "string" and string.sub(key, 1, string.len(key_prefix)) == key_prefix
end

local function remove_index_member(index_key, member)
  if not owns_key(index_key) or redis.call("TYPE", index_key).ok ~= "zset" then
    return
  end
  redis.call("ZREM", index_key, member)
end

redis.call("DEL", state_key)
remove_index_member(global_lru, state_key)
remove_index_member(global_expiry, state_key)
return 1
`

const redisDeleteIfTokenScript = `
local state_key = KEYS[1]
local key_prefix = ARGV[3]

local function owns_key(key)
  return type(key) == "string" and string.sub(key, 1, string.len(key_prefix)) == key_prefix
end

local function remove_index_member(index_key, member)
  if not owns_key(index_key) or redis.call("TYPE", index_key).ok ~= "zset" then
    return
  end
  redis.call("ZREM", index_key, member)
end

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
redis.call("DEL", state_key)
remove_index_member(KEYS[2], state_key)
remove_index_member(KEYS[3], state_key)
return 1
`

const redisDeleteRawIfCurrentScript = `
local state_key = KEYS[1]
local key_prefix = ARGV[4]

local function owns_key(key)
  return type(key) == "string" and string.sub(key, 1, string.len(key_prefix)) == key_prefix
end

local function remove_index_member(index_key, member)
  if not owns_key(index_key) or redis.call("TYPE", index_key).ok ~= "zset" then
    return
  end
  redis.call("ZREM", index_key, member)
end

if redis.call("TYPE", state_key).ok ~= "hash" then
  return 0
end
if redis.call("HGET", state_key, "payload") ~= ARGV[1] or
   redis.call("HGET", state_key, "revision") ~= ARGV[2] or
   redis.call("HGET", state_key, "generation") ~= ARGV[3] then
  return 0
end
redis.call("DEL", state_key)
remove_index_member(KEYS[2], state_key)
remove_index_member(KEYS[3], state_key)
return 1
`
