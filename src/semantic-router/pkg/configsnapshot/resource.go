package configsnapshot

import "sort"

// Kind is the type of a resource.
type Kind string

const (
	// KindSecret is a credential that clusters or listeners use. A snapshot
	// records where the values come from, never the values.
	KindSecret Kind = "secret"
	// KindRuntimeModel is a model the model runtime serves for the Router
	// (global.model_catalog.deployments).
	KindRuntimeModel Kind = "runtime_model"
	// KindEndpoint is one backend of a cluster (providers.models[].backend_refs).
	KindEndpoint Kind = "endpoint"
	// KindCluster is one provider model and the policy its endpoints share
	// (providers.models[]).
	KindCluster Kind = "cluster"
	// KindProgram is one routing recipe: its signals, projections, decisions,
	// algorithms and plugins (routing, recipes[]).
	KindProgram Kind = "program"
	// KindRoute maps one request-facing model name to a program (entrypoints).
	KindRoute Kind = "route"
	// KindListener is one client-facing listener (listeners[]).
	KindListener Kind = "listener"
	// KindSettings is the one resource that holds everything else the Router
	// is configured with: the global services, stores and model catalog,
	// evaluation data and provider defaults. With it, every field of the
	// document belongs to exactly one resource.
	KindSettings Kind = "settings"
)

// SettingsName names the settings resource.
const SettingsName = "global"

// Kinds lists every kind so that a resource only references kinds listed
// before its own.
var Kinds = []Kind{
	KindSecret, KindRuntimeModel, KindEndpoint, KindCluster, KindSettings, KindProgram, KindRoute, KindListener,
}

// Ref names one resource.
type Ref struct {
	Kind Kind   `json:"kind"`
	Name string `json:"name"`
}

func (r Ref) String() string { return string(r.Kind) + "/" + r.Name }

// Resource is one named resource of a snapshot.
type Resource struct {
	Ref
	// Path locates the resource in the canonical document.
	Path string `json:"path"`
	// Hash identifies the resource's content: resources with equal hashes
	// were compiled from equal configuration.
	Hash string `json:"hash"`
	// Refs are the resources this one uses.
	Refs []Ref `json:"refs,omitempty"`
	// Spec is the typed view of the resource; its type follows Kind.
	Spec Spec `json:"spec"`
}

// Spec is the typed view of one kind of resource.
type Spec interface{ specKind() Kind }

// SecretSpec says where a credential's values come from.
type SecretSpec struct {
	// Env names the environment variables the values are read from.
	Env []string `json:"env,omitempty"`
	// Inline counts values written in the document itself.
	Inline int `json:"inline,omitempty"`
}

// RuntimeModelSpec is a model runtime deployment.
type RuntimeModelSpec struct {
	Provider string `json:"provider"`
	Artifact string `json:"artifact,omitempty"`
	Revision string `json:"revision,omitempty"`
	Device   string `json:"device,omitempty"`
	// Endpoint is set when the deployment attaches to an engine the Router
	// does not manage.
	Endpoint string `json:"endpoint,omitempty"`
}

// EndpointSpec is one backend.
type EndpointSpec struct {
	Cluster  string `json:"cluster"`
	Address  string `json:"address"`
	Port     int    `json:"port"`
	Weight   int    `json:"weight,omitempty"`
	Protocol string `json:"protocol,omitempty"`
}

// ClusterSpec is one provider model.
type ClusterSpec struct {
	APIFormat string `json:"api_format,omitempty"`
	LBPolicy  string `json:"lb_policy,omitempty"`
	// Order is the model's position in providers.models; the first model with
	// a backend serves the default route.
	Order int `json:"order"`
}

// SettingsSpec is the settings resource.
type SettingsSpec struct {
	// DefaultModel is providers.defaults.model.
	DefaultModel string `json:"default_model,omitempty"`
}

// ProgramSpec is one routing recipe.
type ProgramSpec struct {
	// Decisions are the recipe's decisions in priority order as authored.
	Decisions []string `json:"decisions,omitempty"`
}

// RouteSpec maps a request-facing model name to a program.
type RouteSpec struct {
	Model   string `json:"model"`
	Program string `json:"program"`
}

// ListenerSpec is one client-facing listener.
type ListenerSpec struct {
	Address string `json:"address"`
	Port    int    `json:"port"`
	Timeout string `json:"timeout,omitempty"`
}

func (SecretSpec) specKind() Kind       { return KindSecret }
func (RuntimeModelSpec) specKind() Kind { return KindRuntimeModel }
func (EndpointSpec) specKind() Kind     { return KindEndpoint }
func (ClusterSpec) specKind() Kind      { return KindCluster }
func (ProgramSpec) specKind() Kind      { return KindProgram }
func (RouteSpec) specKind() Kind        { return KindRoute }
func (ListenerSpec) specKind() Kind     { return KindListener }
func (SettingsSpec) specKind() Kind     { return KindSettings }

// Resources is the immutable set of a snapshot's resources.
type Resources struct {
	byRef  map[Ref]*Resource
	byKind map[Kind][]*Resource
}

// Get returns the resource ref names.
func (r *Resources) Get(ref Ref) (*Resource, bool) {
	if r == nil {
		return nil, false
	}
	resource, ok := r.byRef[ref]
	return resource, ok
}

// List returns the resources of one kind, sorted by name.
func (r *Resources) List(kind Kind) []*Resource {
	if r == nil {
		return nil
	}
	return append([]*Resource(nil), r.byKind[kind]...)
}

// Len is the number of resources.
func (r *Resources) Len() int {
	if r == nil {
		return 0
	}
	return len(r.byRef)
}

// Counts returns how many resources there are of each kind.
func (r *Resources) Counts() map[Kind]int {
	counts := make(map[Kind]int, len(Kinds))
	if r == nil {
		return counts
	}
	for kind, resources := range r.byKind {
		counts[kind] = len(resources)
	}
	return counts
}

func newResources(list []*Resource) *Resources {
	r := &Resources{byRef: make(map[Ref]*Resource, len(list)), byKind: make(map[Kind][]*Resource, len(Kinds))}
	for _, resource := range list {
		r.byRef[resource.Ref] = resource
		r.byKind[resource.Kind] = append(r.byKind[resource.Kind], resource)
	}
	for _, resources := range r.byKind {
		sort.Slice(resources, func(i, j int) bool { return resources[i].Name < resources[j].Name })
	}
	return r
}
