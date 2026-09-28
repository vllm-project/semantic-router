package admission

// Registry holds one Admissioner per Router Model deployment key. Deployments
// without a configured gate admit every request through Noop.
type Registry struct {
	gates map[string]Admissioner
}

func NewRegistry(gates map[string]Admissioner) *Registry {
	return &Registry{gates: gates}
}

func (r *Registry) For(deployment string) Admissioner {
	if r == nil {
		return Noop{}
	}
	if gate, ok := r.gates[deployment]; ok && gate != nil {
		return gate
	}
	return Noop{}
}
