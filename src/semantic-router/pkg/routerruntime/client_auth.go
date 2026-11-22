package routerruntime

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/clientauth"

// ActiveClientAuth reads the same live policy as the serving owner. The
// management snapshot can lag that owner during publication, so its policy
// is only a fallback for registries without a serving owner.
func (r *Registry) ActiveClientAuth() *clientauth.Policy {
	if r == nil {
		return nil
	}
	r.mu.RLock()
	active, snapshot := r.activeClientAuth, r.configSnapshot
	r.mu.RUnlock()
	if active != nil {
		return active()
	}
	if snapshot != nil {
		return snapshot.ClientAuth()
	}
	return nil
}
