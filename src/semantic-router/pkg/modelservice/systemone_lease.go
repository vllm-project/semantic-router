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
	member, err := l.call(deployment)
	if err != nil {
		m.mu.Unlock()
		return SystemOneResult{}, err
	}
	member.group.refs++
	m.mu.Unlock()
	defer m.release([]*group{member.group})
	started := time.Now()
	result, timing, err := member.group.client.systemOne(ctx, member.served.name, body)
	l.observe(member, deployment, "decisions", started, timing, err)
	return result, err
}
