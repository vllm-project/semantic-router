// Package providercatalog composes provider integrations in the Management
// control plane. Product providers are control-plane data; inference receives
// only compiled provider-neutral routing backends.
package providercatalog

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"

type CredentialMode string

const (
	CredentialNone     CredentialMode = "none"
	CredentialOptional CredentialMode = "optional"
	CredentialRequired CredentialMode = "required"
)

type OriginMode string

const (
	OriginFixed        OriginMode = "fixed"
	OriginUserSupplied OriginMode = "user_supplied"
)

type FieldKind string

const (
	FieldText    FieldKind = "text"
	FieldBoolean FieldKind = "boolean"
	FieldInteger FieldKind = "integer"
	FieldSelect  FieldKind = "select"
)

type Definition struct {
	ID         string      `json:"providerId"`
	Revision   string      `json:"revision"`
	Display    Display     `json:"display"`
	Order      uint32      `json:"order"`
	Interfaces []Interface `json:"interfaces"`
	Credential Credential  `json:"credential"`
	Origin     Origin      `json:"origin"`
	Discovery  *Discovery  `json:"discovery,omitempty"`
	// Capabilities describes transport features supported by this Provider
	// integration. It is catalog/filter metadata, never evidence that every
	// model returned by discovery supports the same features.
	Capabilities     []string          `json:"capabilities,omitempty"`
	ConnectionFields []ConnectionField `json:"connectionFields,omitempty"`
}

// Interface is a user-facing connection preset owned by one Provider
// Integration. Its short ID is suitable for an Advanced "API style" choice;
// wire format and compiler details stay control-plane metadata and compile to
// exactly one immutable backend.
type Interface struct {
	ID         string                 `json:"id"`
	Label      string                 `json:"label"`
	Default    bool                   `json:"default,omitempty"`
	WireFormat llmprotocol.WireFormat `json:"wireFormat"`
	Compiler   Compiler               `json:"compiler"`
	// Capabilities describes features the selected wire interface can carry.
	// Model capability metadata is discovered or authored independently.
	Capabilities []string `json:"capabilities,omitempty"`
}

// Integration is the injection seam for built-in and application-provided
// control-plane integrations. NewRegistry evaluates it exactly once.
type Integration interface{ Definition() Definition }

type IntegrationFunc func() Definition

func (integration IntegrationFunc) Definition() Definition { return integration() }

type Compiler struct {
	AdapterID string         `json:"adapterId"`
	Config    map[string]any `json:"config"`
}

type Display struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Category    string `json:"category"`
	Icon        Icon   `json:"icon"`
	Monogram    string `json:"monogram,omitempty"`
	Accent      string `json:"accent,omitempty"`
}

// Icon is safe presentation metadata owned by the control-plane Integration.
// The Dashboard renders this descriptor generically, so installing a Provider
// does not require adding product-specific frontend code.
type Icon struct {
	Source string `json:"source"`
	Value  string `json:"value"`
	Color  bool   `json:"color"`
}

type Credential struct {
	Mode      CredentialMode `json:"mode"`
	AdapterID string         `json:"adapterId,omitempty"`
	Label     string         `json:"label,omitempty"`
	Hint      string         `json:"hint,omitempty"`
}

type Origin struct {
	Mode       OriginMode `json:"mode"`
	DefaultURL string     `json:"defaultUrl,omitempty"`
	Label      string     `json:"label,omitempty"`
	Hint       string     `json:"hint,omitempty"`
}

type Discovery struct {
	AdapterID string            `json:"adapterId"`
	Path      string            `json:"path,omitempty"`
	Headers   map[string]string `json:"headers,omitempty"`
}

type ConnectionField struct {
	Name        string        `json:"name"`
	Label       string        `json:"label"`
	Kind        FieldKind     `json:"kind"`
	Required    bool          `json:"required"`
	Advanced    bool          `json:"advanced"`
	Default     string        `json:"default,omitempty"`
	Hint        string        `json:"hint,omitempty"`
	Placeholder string        `json:"placeholder,omitempty"`
	Options     []FieldOption `json:"options,omitempty"`
}

type FieldOption struct {
	Value string `json:"value"`
	Label string `json:"label"`
}

type IntegrationReference struct {
	ProviderID string `json:"providerId"`
	Revision   string `json:"revision"`
}

type Snapshot struct {
	revision     string
	references   []IntegrationReference
	integrations []Definition
	providers    []Definition
	byID         map[string]Definition
}
