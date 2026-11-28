package classification

import (
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

const (
	sharedVerdictRefreshInterval = 5 * time.Minute
	sharedVerdictRefreshLimit    = 1024
)

type sharedVerdictRefresh struct {
	view     *sharedWindowVerdicts
	cache    *windowVerdicts
	key      windowVerdictKey
	response modelservice.Response
	expires  time.Time
}

// touch never does network I/O or signing. A pending key retains one snapshot;
// further hits update only its expiry. Only hits that extend the last published
// expiry by at least five minutes enter the queue. Expiry is always last use
// + TTL, not the time the queued write eventually executes.
func (s *sharedWindowVerdicts) touch(cache *windowVerdicts, key windowVerdictKey, response modelservice.Response) {
	if s == nil || s.ctx.Err() != nil {
		return
	}
	name, expires := s.key(key), time.Now().Add(windowVerdictTTL)
	if published := cache.sharedExpiry(key); !published.IsZero() && expires.Sub(published) < sharedVerdictRefreshInterval {
		return
	}
	s.refreshMu.Lock()
	if s.ctx.Err() != nil {
		s.refreshMu.Unlock()
		return
	}
	if pending := s.refreshes[name]; pending != nil {
		if expires.After(pending.expires) {
			pending.expires = expires
		}
		s.refreshMu.Unlock()
		metrics.RecordGuardVerdictStoreOp("refresh", "coalesced")
		return
	}
	if len(s.refreshes) >= sharedVerdictRefreshLimit {
		s.refreshMu.Unlock()
		metrics.RecordGuardVerdictStoreOp("refresh", "queue_full")
		return
	}
	s.refreshes[name] = &sharedVerdictRefresh{
		view: s, cache: cache, key: key, response: cloneWindowResponse(response), expires: expires,
	}
	s.refreshMu.Unlock()
	metrics.RecordGuardVerdictStoreOp("refresh", "queued")
	select {
	case s.wake <- struct{}{}:
	default:
	}
}

func (s *sharedVerdictStore) refreshLoop() {
	defer s.worker.Done()
	for {
		select {
		case <-s.ctx.Done():
			return
		case <-s.wake:
		}
		for s.flushRefreshes() {
			if s.ctx.Err() != nil {
				return
			}
		}
	}
}

// flushRefreshes drains a bounded batch of ready work and reports whether the
// worker should continue. The queue contains no entries waiting for a timer.
func (s *sharedVerdictStore) flushRefreshes() bool {
	s.refreshMu.Lock()
	var ready []*sharedVerdictRefresh
	for name, pending := range s.refreshes {
		ready = append(ready, pending)
		delete(s.refreshes, name)
		if len(ready) == 32 {
			break
		}
	}
	more := len(s.refreshes) > 0
	s.refreshMu.Unlock()
	for _, pending := range ready {
		// Another write may have completed after this hit was queued.
		if published := pending.cache.sharedExpiry(pending.key); !published.IsZero() && pending.expires.Sub(published) < sharedVerdictRefreshInterval {
			continue
		}
		if pending.view.putUntil(s.ctx, pending.key, pending.response, pending.expires) {
			pending.cache.published(pending.key, pending.expires)
		}
	}
	return more
}

func (v *windowVerdicts) published(key windowVerdictKey, expires time.Time) {
	if expires.IsZero() {
		return
	}
	v.mu.Lock()
	defer v.mu.Unlock()
	if element := v.entries[key]; element != nil {
		entry := element.Value.(*keptWindowEntry)
		if expires.After(entry.sharedUntil) {
			entry.sharedUntil = expires
		}
	}
}

func (v *windowVerdicts) sharedExpiry(key windowVerdictKey) time.Time {
	v.mu.Lock()
	defer v.mu.Unlock()
	if element := v.entries[key]; element != nil {
		return element.Value.(*keptWindowEntry).sharedUntil
	}
	return time.Time{}
}
