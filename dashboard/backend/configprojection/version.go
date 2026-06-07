package configprojection

import (
	"fmt"
	"time"
)

// NewActivationVersion returns a unique deployment version for projection records.
// Backup filenames keep second precision; projection versions add nanoseconds to
// avoid same-second deploy/update collisions and rollback namespace overlap.
func NewActivationVersion() string {
	now := time.Now().UTC()
	return fmt.Sprintf("%s.%09d", now.Format("20060102-150405"), now.Nanosecond())
}
