package routerreplay

// UpdateSessionName stores the session name a record's answer gave.
func (r *Recorder) UpdateSessionName(id string, name string) error {
	ctx, cancel := r.replayOperationContext()
	defer cancel()
	return r.storage.UpdateSessionName(ctx, id, name)
}
