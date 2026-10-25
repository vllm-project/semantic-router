package configprojection

import (
	"fmt"
	"sync/atomic"
	"time"
)

var activationVersions activationVersionClock

type activationVersionClock struct{ last atomic.Int64 }

// NewActivationVersion returns a process-unique deployment version for projection records.
// Backup filenames keep second precision; projection versions add nanoseconds to
// avoid same-second deploy/update collisions and rollback namespace overlap.
func NewActivationVersion() string {
	return activationVersions.next(time.Now())
}

func (c *activationVersionClock) next(now time.Time) string {
	next := now.UnixNano()
	for {
		last := c.last.Load()
		// Wall clocks may repeat a tick or move backward. Reserve one logical
		// nanosecond atomically instead of relying on their actual resolution.
		if next <= last {
			next = last + 1
		}
		if c.last.CompareAndSwap(last, next) {
			stamp := time.Unix(0, next).UTC()
			return fmt.Sprintf("%s.%09d", stamp.Format("20060102-150405"), stamp.Nanosecond())
		}
	}
}
