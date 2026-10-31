package store

import (
	"sync/atomic"
	"time"
)

const retentionSweepInterval = time.Minute

// retentionCutoff is the oldest timestamp a read may return, or zero without a TTL.
func retentionCutoff(ttl time.Duration) time.Time {
	if ttl <= 0 {
		return time.Time{}
	}
	return time.Now().UTC().Add(-ttl)
}

func isExpired(record Record, ttl time.Duration) bool {
	return ttl > 0 && !record.Timestamp.IsZero() && record.Timestamp.Before(retentionCutoff(ttl))
}

// retentionSweep spaces out background deletes of expired records.
type retentionSweep struct {
	next atomic.Int64
}

func (s *retentionSweep) due() bool {
	now := time.Now().UnixNano()
	next := s.next.Load()
	return now >= next && s.next.CompareAndSwap(next, now+int64(retentionSweepInterval))
}
