package modelservice

import (
	"context"
	"encoding/json"
	"time"
)

// SystemOne invokes a concrete deployment from this retained generation. The
// public gateway must not authorize with one snapshot and call a newer one.
func (l *Lease) SystemOne(ctx context.Context, deployment string, body json.RawMessage) (SystemOneResult, error) {
	if l == nil || l.manager == nil {
		return SystemOneResult{}, ErrUnavailable
	}
	m := l.manager
	m.mu.Lock()
	l.mu.RLock()
	closed := len(l.groups) == 0
	l.mu.RUnlock()
	if m.closed || closed {
		m.mu.Unlock()
		return SystemOneResult{}, ErrUnavailable
	}
	selected, err := l.call(deployment)
	if err != nil {
		m.mu.Unlock()
		return SystemOneResult{}, err
	}
	retained := retainMemberLocked(selected)
	m.mu.Unlock()
	defer m.release(retained)
	started := time.Now()
	result, timing, err := selected.client().systemOne(ctx, selected.served.name, body)
	l.observe(selected, deployment, "decisions", started, timing, err)
	return result, err
}

// Caller holds Manager.mu, pinning candidates while an exchange is assembled.
func retainMemberLocked(selected member) []*group {
	groups := []*group{selected.group}
	if selected.pool != nil {
		groups = nil
		for _, worker := range selected.pool.workers {
			groups = append(groups, worker.member.group)
		}
	}
	for _, g := range groups {
		g.refs++
	}
	return groups
}
