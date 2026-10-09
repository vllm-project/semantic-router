package configsnapshot

import "context"

// UpdateSource delivers configuration updates to a Manager: the watched file,
// the management API and the Kubernetes controller today, and a control plane
// that pushes resources later. Run delivers each update through apply until
// ctx ends; apply returns the update's ACK, the activated snapshot, or its
// NACK, which the source can report back to where the update came from.
type UpdateSource interface {
	Run(ctx context.Context, apply func(context.Context, Update) (*Snapshot, error)) error
}

// Serve runs source until ctx ends, applying every update it delivers.
func (m *Manager) Serve(ctx context.Context, source UpdateSource) error {
	return source.Run(ctx, m.Apply)
}
