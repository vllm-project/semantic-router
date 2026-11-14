package sessiontelemetry

import (
	"context"
	"time"

	"github.com/redis/go-redis/v9"
)

// RouterSessionBudgetStore is implemented by shared stores that hold token
// reservations for every replica of a session, so replicas cannot each admit
// a turn against the same remaining budget. Reserve checks and holds in one
// atomic step; Settle closes a hold once and adds its charge to the settled
// total.
type RouterSessionBudgetStore interface {
	ReserveTokens(r SharedReservation) error
	SettleTokens(sessionID, id string, charge int64, ttl time.Duration) error
}

// SharedReservation is one turn's hold in a RouterSessionBudgetStore.
type SharedReservation struct {
	SessionID string
	ID        string
	Tokens    int64
	// Recorded is the session usage this replica has recorded; the store
	// counts the larger of it and the total settled through the store.
	Recorded    int64
	Limit       int64
	MaxInFlight int
	// HoldFor bounds a hold whose turn never settles, for example when the
	// replica that took it exits.
	HoldFor time.Duration
	TTL     time.Duration
}

// The ledger is a hash beside the session snapshot: "settled" is the total
// charged through it, and each "r:<id>" field holds "<tokens>:<expires ms>".
// Session keys escape '#', so the suffix cannot collide with another session.
const redisBudgetKeySuffix = "#reservations"

var redisBudgetReserveScript = redis.NewScript(`
local now = redis.call('TIME')
local nowMs = tonumber(now[1]) * 1000 + math.floor(tonumber(now[2]) / 1000)
local settled, held, open = 0, 0, 0
local fields = redis.call('HGETALL', KEYS[1])
for i = 1, #fields, 2 do
  if fields[i] == 'settled' then
    settled = tonumber(fields[i + 1])
  else
    local tokens, expires = string.match(fields[i + 1], '^(%d+):(%d+)$')
    if tokens == nil or tonumber(expires) <= nowMs then
      redis.call('HDEL', KEYS[1], fields[i])
    else
      held = held + tonumber(tokens)
      open = open + 1
    end
  end
end
local maxInFlight = tonumber(ARGV[5])
if maxInFlight > 0 and open >= maxInFlight then
  return -2
end
local used = math.max(settled, tonumber(ARGV[3]))
if tonumber(ARGV[2]) > tonumber(ARGV[4]) - used - held then
  return -1
end
redis.call('HSET', KEYS[1], 'r:' .. ARGV[1], ARGV[2] .. ':' .. (nowMs + tonumber(ARGV[6])))
redis.call('PEXPIRE', KEYS[1], ARGV[7])
return 1
`)

var redisBudgetSettleScript = redis.NewScript(`
if redis.call('HDEL', KEYS[1], 'r:' .. ARGV[1]) == 0 then
  return 0
end
if tonumber(ARGV[2]) > 0 then
  redis.call('HINCRBY', KEYS[1], 'settled', ARGV[2])
end
redis.call('PEXPIRE', KEYS[1], ARGV[3])
return 1
`)

func (s *redisRouterSessionStore) ReserveTokens(r SharedReservation) error {
	if s.ttl > 0 {
		r.TTL = s.ttl
	}
	ctx, cancel := context.WithTimeout(context.Background(), s.timeout)
	defer cancel()
	result, err := redisBudgetReserveScript.Run(ctx, s.client, []string{s.keyPrefix + r.SessionID + redisBudgetKeySuffix},
		r.ID, r.Tokens, r.Recorded, r.Limit, r.MaxInFlight, r.HoldFor.Milliseconds(), r.TTL.Milliseconds()).Int()
	if err != nil {
		return err
	}
	switch result {
	case -1:
		return ErrSessionBudgetExceeded
	case -2:
		return ErrSessionReservationsFull
	}
	return nil
}

func (s *redisRouterSessionStore) SettleTokens(sessionID, id string, charge int64, ttl time.Duration) error {
	if s.ttl > 0 {
		ttl = s.ttl
	}
	ctx, cancel := context.WithTimeout(context.Background(), s.timeout)
	defer cancel()
	return redisBudgetSettleScript.Run(ctx, s.client, []string{s.keyPrefix + sessionID + redisBudgetKeySuffix},
		id, charge, ttl.Milliseconds()).Err()
}
