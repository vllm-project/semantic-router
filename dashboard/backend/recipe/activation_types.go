package recipe

const (
	ActivationModeHotSwitch       = "hot_switch"
	ActivationModeStackRecreation = "stack_recreation"
)

// ActivationListener is the public, credential-free listener topology shown
// before a managed Recipe activation.
type ActivationListener struct {
	Name     string `json:"name"`
	Address  string `json:"address"`
	Port     int    `json:"port"`
	HostPort int    `json:"host_port"`
}

// ActivationStorageDiff describes the exact managed sidecar set the target
// runtime will own. Container names and storage credentials are intentionally
// absent from this Dashboard response.
type ActivationStorageDiff struct {
	Before []string `json:"before"`
	After  []string `json:"after"`
	Add    []string `json:"add"`
	Remove []string `json:"remove"`
	Repair []string `json:"repair"`
}

// ActivationManagementAuth describes only the scoped role contract. The
// credential value and its private storage location never cross the API.
type ActivationManagementAuth struct {
	Mode               string   `json:"mode"`
	ServiceRole        string   `json:"service_role,omitempty"`
	ServicePermissions []string `json:"service_permissions"`
}

// ActivationManagementListener is the credential-free management endpoint
// preview. BindAddress and Port describe the Router container listener, while
// HostPort is the loopback-only host publication owned by the managed stack.
type ActivationManagementListener struct {
	BindAddress string `json:"bind_address"`
	Port        int    `json:"port"`
	HostPort    int    `json:"host_port"`
}

// ActivationPlan is a stable preview of every topology-visible activation
// effect. PlanDigest binds a later confirmation to these exact effects.
type ActivationPlan struct {
	RecipeDigest         string                       `json:"recipe_digest"`
	PlanDigest           string                       `json:"plan_digest"`
	Mode                 string                       `json:"mode"`
	RequiresConfirmation bool                         `json:"requires_confirmation"`
	ListenersBefore      []ActivationListener         `json:"listeners_before"`
	ListenersAfter       []ActivationListener         `json:"listeners_after"`
	Storage              ActivationStorageDiff        `json:"storage"`
	ManagementBefore     ActivationManagementListener `json:"management_before"`
	ManagementAfter      ActivationManagementListener `json:"management_after"`
	ManagementAuth       ActivationManagementAuth     `json:"management_auth"`
	Effects              []string                     `json:"effects"`
}

// ActivationTopologyState is private transaction state. It contains only
// deterministic managed container identities and no credential values.
type ActivationTopologyState struct {
	SchemaVersion string                          `json:"schema_version"`
	TransactionID string                          `json:"transaction_id"`
	PlanDigest    string                          `json:"plan_digest"`
	Listeners     []ActivationListener            `json:"listeners"`
	StorageBefore []string                        `json:"storage_before"`
	StorageAfter  []string                        `json:"storage_after"`
	Containers    []ActivationContainerTransition `json:"containers"`
	CredentialEnv string                          `json:"credential_env"`
}

type ActivationContainerTransition struct {
	Service    string `json:"service"`
	Name       string `json:"name"`
	BackupName string `json:"backup_name"`
	Action     string `json:"action"`
	WasRunning bool   `json:"was_running"`
}
