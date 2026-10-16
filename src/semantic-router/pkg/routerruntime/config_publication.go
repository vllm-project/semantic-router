package routerruntime

// LockConfigPublication coordinates source persistence with the final identity
// check and publication of a prepared file generation. The registry belongs to
// one Router; there is no process-global or path-indexed lock. Never retain this
// lock during model preparation or while waiting for runtime activation.
func (r *Registry) LockConfigPublication() func() {
	if r == nil {
		return func() {}
	}
	r.configPublicationMu.Lock()
	return r.configPublicationMu.Unlock
}
