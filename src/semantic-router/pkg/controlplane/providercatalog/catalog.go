package providercatalog

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

const catalogRevisionSchema = "provider-catalog.revision.v2"

func buildSnapshot(definitions []Definition, registry adapterCapabilities, restoring bool) (*Snapshot, error) {
	if registry == nil {
		return nil, fmt.Errorf("provider integration registry is required")
	}
	if len(definitions) == 0 {
		return nil, fmt.Errorf("at least one provider integration is required")
	}

	providers := make([]Definition, 0, len(definitions))
	integrations := make([]Definition, 0, len(definitions))
	byID := make(map[string]Definition, len(definitions))
	for index := range definitions {
		definition, buildSnapshotErr := cloneDefinitionChecked(definitions[index])
		if buildSnapshotErr != nil {
			return nil, fmt.Errorf("provider integration %d: %w", index, buildSnapshotErr)
		}
		declaredRevision := definition.Revision
		if !restoring && declaredRevision != "" {
			return nil, fmt.Errorf("provider integration %q revision is registry-owned", definition.ID)
		}
		definition.Revision = ""
		if err := validateDefinition(&definition, registry); err != nil {
			return nil, fmt.Errorf("provider integration %q: %w", definition.ID, err)
		}
		revision, buildSnapshotErr := definitionRevision(definition)
		if buildSnapshotErr != nil {
			return nil, fmt.Errorf("provider integration %q revision: %w", definition.ID, buildSnapshotErr)
		}
		if restoring && declaredRevision != revision {
			return nil, fmt.Errorf("provider integration %q revision does not match immutable content", definition.ID)
		}
		definition.Revision = revision
		if _, exists := byID[definition.ID]; exists {
			return nil, fmt.Errorf("provider integration %q is duplicated", definition.ID)
		}
		byID[definition.ID] = cloneDefinition(definition)
		providers = append(providers, cloneDefinition(definition))
		integrations = append(integrations, cloneDefinition(definition))
	}

	sort.Slice(providers, func(i, j int) bool {
		if providers[i].Order == providers[j].Order {
			return providers[i].ID < providers[j].ID
		}
		return providers[i].Order < providers[j].Order
	})
	sort.Slice(integrations, func(i, j int) bool { return integrations[i].ID < integrations[j].ID })
	references := make([]IntegrationReference, len(integrations))
	for index, definition := range integrations {
		references[index] = IntegrationReference{ProviderID: definition.ID, Revision: definition.Revision}
	}
	revision, err := snapshotRevision(references, providers)
	if err != nil {
		return nil, err
	}
	return &Snapshot{
		revision: revision, references: references, integrations: integrations,
		providers: providers, byID: byID,
	}, nil
}

func definitionRevision(definition Definition) (string, error) {
	definition.Revision = ""
	payload, err := json.Marshal(definition)
	if err != nil {
		return "", err
	}
	digest := sha256.Sum256(payload)
	return "sha256:" + hex.EncodeToString(digest[:]), nil
}

func snapshotRevision(references []IntegrationReference, providers []Definition) (string, error) {
	payload, err := json.Marshal(struct {
		Schema       string                 `json:"schema"`
		Integrations []IntegrationReference `json:"integrations"`
		Providers    []Definition           `json:"providers"`
	}{Schema: catalogRevisionSchema, Integrations: references, Providers: providers})
	if err != nil {
		return "", fmt.Errorf("encode Provider Catalog revision: %w", err)
	}
	digest := sha256.Sum256(payload)
	return "sha256:" + hex.EncodeToString(digest[:]), nil
}

func (snapshot *Snapshot) Revision() string {
	if snapshot == nil {
		return ""
	}
	return snapshot.revision
}

func (snapshot *Snapshot) IntegrationReferences() []IntegrationReference {
	if snapshot == nil {
		return nil
	}
	return append([]IntegrationReference(nil), snapshot.references...)
}

func (snapshot *Snapshot) Get(providerID string) (Definition, bool) {
	if snapshot == nil {
		return Definition{}, false
	}
	definition, found := snapshot.byID[providerID]
	return cloneDefinition(definition), found
}

func (snapshot *Snapshot) List() []Definition {
	if snapshot == nil {
		return nil
	}
	return cloneDefinitions(snapshot.providers)
}

func cloneSnapshot(snapshot *Snapshot) *Snapshot {
	if snapshot == nil {
		return nil
	}
	cloned := &Snapshot{
		revision:     snapshot.revision,
		references:   append([]IntegrationReference(nil), snapshot.references...),
		integrations: cloneDefinitions(snapshot.integrations),
		providers:    cloneDefinitions(snapshot.providers),
		byID:         make(map[string]Definition, len(snapshot.byID)),
	}
	for id, definition := range snapshot.byID {
		cloned.byID[id] = cloneDefinition(definition)
	}
	return cloned
}

func cloneDefinitions(source []Definition) []Definition {
	if source == nil {
		return nil
	}
	result := make([]Definition, len(source))
	for index := range source {
		result[index] = cloneDefinition(source[index])
	}
	return result
}

func cloneDefinition(source Definition) Definition {
	cloned, err := cloneDefinitionChecked(source)
	if err != nil {
		return Definition{}
	}
	return cloned
}

func cloneDefinitionChecked(source Definition) (Definition, error) {
	interfaces := make([]Interface, len(source.Interfaces))
	for index, providerInterface := range source.Interfaces {
		compilerConfig, err := cloneCompilerConfig(providerInterface.Compiler.Config)
		if err != nil {
			return Definition{}, fmt.Errorf("interface %q compiler config: %w", providerInterface.ID, err)
		}
		providerInterface.Compiler.Config = compilerConfig
		providerInterface.Capabilities = append([]string(nil), providerInterface.Capabilities...)
		interfaces[index] = providerInterface
	}
	source.Interfaces = interfaces
	if source.Discovery != nil {
		discovery := *source.Discovery
		discovery.Headers = cloneStringMap(source.Discovery.Headers)
		source.Discovery = &discovery
	}
	source.Capabilities = append([]string(nil), source.Capabilities...)
	source.ConnectionFields = cloneConnectionFields(source.ConnectionFields)
	return source, nil
}

func validCatalogRevision(value string) bool {
	if !strings.HasPrefix(value, "sha256:") || len(value) != len("sha256:")+sha256.Size*2 {
		return false
	}
	_, err := hex.DecodeString(strings.TrimPrefix(value, "sha256:"))
	return err == nil
}
