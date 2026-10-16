package pluginruntime

// Dependency describes an already-published handle. Availability never implies
// a healthy remote service; health is only established by an explicit probe.
type Dependency struct {
	Name         string `json:"name"`
	Availability string `json:"availability"`
	Health       string `json:"health"`
}

type BindingInspector interface {
	InspectPluginBinding(Binding, string) ([]Dependency, error)
}
