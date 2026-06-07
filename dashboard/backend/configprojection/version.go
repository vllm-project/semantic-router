package configprojection

import "time"

// NewActivationVersion returns a deployment version aligned with config backup naming.
func NewActivationVersion() string {
	return time.Now().UTC().Format("20060102-150405")
}
