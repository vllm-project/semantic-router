package sessiontelemetry

import (
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"strconv"
	"sync/atomic"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

var (
	// ErrSessionBudgetExceeded means the reservation would take the session
	// past its token limit.
	ErrSessionBudgetExceeded = errors.New("session token budget exceeded")
	// ErrSessionReservationsFull means the session already has the allowed
	// number of open reservations.
	ErrSessionReservationsFull = errors.New("session has too many open token reservations")
)

// lastReservation numbers reservations across sessions, so an id from an
// evicted session never matches a later one.
var lastReservation atomic.Uint64

// sharedReservationHold bounds a shared-store hold whose turn never settles.
const sharedReservationHold = 15 * time.Minute

// reservationPrefix tells this process's reservation ids apart from other
// replicas' in a shared store.
var reservationPrefix = func() string {
	var b [8]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}()

func sharedReservationID(id uint64) string {
	return reservationPrefix + "-" + strconv.FormatUint(id, 10)
}

// SessionReservationParams sizes one turn's reservation against its session's
// token limit.
type SessionReservationParams struct {
	SessionID string
	// Tokens is the turn's demand: the prompt estimate plus the output reserve.
	Tokens int64
	// Limit caps the session's prompt plus completion tokens.
	Limit int64
	// MaxInFlight caps the session's open reservations; zero means no cap.
	MaxInFlight int
	Timestamp   time.Time
}

// ReserveSessionTokens holds tokens for one turn before dispatch. Recorded
// usage and open reservations both count against the limit, and the check and
// the hold happen under the session memory lock, so concurrent turns of one
// session cannot pass the limit with their reservations. The returned id is
// closed once, by RecordSessionUsage (SessionUsageParams.Reservation) or by
// ReleaseSessionTokens. Reservations are not part of RouterSessionSnapshot. A
// session this process has not seen yet starts from the usage in the shared
// store, when one is set; when that store is a RouterSessionBudgetStore, the
// reservation is also held there, against every replica's open holds.
func ReserveSessionTokens(p SessionReservationParams) (uint64, error) {
	if p.SessionID == "" || p.Tokens < 0 || p.Limit <= 0 || p.MaxInFlight < 0 {
		return 0, fmt.Errorf("invalid session token reservation for %q: tokens %d, limit %d, max in flight %d",
			p.SessionID, p.Tokens, p.Limit, p.MaxInFlight)
	}
	s := globalRouterSessionMemory
	s.mu.Lock()
	cold := s.sessions[p.SessionID] == nil
	s.mu.Unlock()
	if cold {
		loadSharedRouterSessionSnapshotMode(p.SessionID, p.Timestamp, true)
	}
	id, recorded, err := reserveLocal(p)
	if err != nil {
		return 0, err
	}
	if err := reserveShared(p, id, recorded); err != nil {
		releaseLocal(p.SessionID, id)
		return 0, err
	}
	return id, nil
}

func reserveLocal(p SessionReservationParams) (uint64, int64, error) {
	s := globalRouterSessionMemory
	s.mu.Lock()
	defer s.mu.Unlock()

	now := p.Timestamp
	if now.IsZero() {
		now = s.nowFn()
	}
	s.evictExpiredLocked(now)
	st := s.sessions[p.SessionID]
	if st != nil && now.Sub(st.lastSeen) > routerMemoryTTL {
		st = nil // expired but not swept yet: sessionLocked starts it over
	}
	var used int64
	if st != nil {
		if p.MaxInFlight > 0 && len(st.reservations) >= p.MaxInFlight {
			return 0, 0, ErrSessionReservationsFull
		}
		used = st.cumulativePrompt + st.cumulativeCompletion + st.reservedTokens
	}
	if p.Tokens > p.Limit-used {
		return 0, 0, ErrSessionBudgetExceeded
	}
	if st == nil {
		st = s.sessionLocked(p.SessionID, now)
	}
	if st.reservations == nil {
		st.reservations = make(map[uint64]int64)
	}
	id := lastReservation.Add(1)
	st.reservations[id] = p.Tokens
	st.reservedTokens += p.Tokens
	st.lastSeen = now
	return id, st.cumulativePrompt + st.cumulativeCompletion, nil
}

func reserveShared(p SessionReservationParams, id uint64, recorded int64) error {
	store, release, acquired := acquireCurrentRouterSessionStateStore()
	if !acquired {
		return nil
	}
	defer release()
	budget, ok := store.(RouterSessionBudgetStore)
	if !ok {
		return nil
	}
	err := budget.ReserveTokens(SharedReservation{
		SessionID: p.SessionID, ID: sharedReservationID(id), Tokens: p.Tokens, Recorded: recorded,
		Limit: p.Limit, MaxInFlight: p.MaxInFlight, HoldFor: sharedReservationHold, TTL: routerMemoryTTL,
	})
	if err != nil && !errors.Is(err, ErrSessionBudgetExceeded) && !errors.Is(err, ErrSessionReservationsFull) {
		return fmt.Errorf("reserve session tokens in the shared store: %w", err)
	}
	return err
}

// ReleaseSessionTokens closes a reservation without charging it, for a turn
// that ends with no usage to record. A reservation that is already closed is
// ignored.
func ReleaseSessionTokens(sessionID string, reservation uint64) {
	releaseLocal(sessionID, reservation)
	settleShared(sessionID, reservation, 0)
}

func releaseLocal(sessionID string, reservation uint64) {
	s := globalRouterSessionMemory
	s.mu.Lock()
	defer s.mu.Unlock()
	if st := s.sessions[sessionID]; st != nil {
		st.closeReservation(reservation)
	}
}

// settleShared closes a shared-store hold and charges it. A failed call is
// logged; the hold then lapses after sharedReservationHold.
func settleShared(sessionID string, reservation uint64, charge int64) {
	if reservation == 0 {
		return
	}
	store, release, acquired := acquireCurrentRouterSessionStateStore()
	if !acquired {
		return
	}
	defer release()
	budget, ok := store.(RouterSessionBudgetStore)
	if !ok {
		return
	}
	if err := budget.SettleTokens(sessionID, sharedReservationID(reservation), charge, routerMemoryTTL); err != nil {
		logging.ComponentWarnEvent("router", "session_budget_settle_failed", map[string]interface{}{
			"session_id": sessionID,
			"error":      err.Error(),
		})
	}
}

func (st *routerSessionState) closeReservation(id uint64) {
	tokens, ok := st.reservations[id]
	if !ok {
		return
	}
	delete(st.reservations, id)
	st.reservedTokens -= tokens
}
