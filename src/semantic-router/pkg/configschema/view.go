package configschema

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/url"
	"sort"
	"strings"
)

const (
	ViewFull    = "full"
	ViewIndex   = "index"
	ViewSection = "section"
	ViewSurface = "surface"
)

// ViewOptions selects a progressively disclosed representation of the
// canonical configuration contract. An empty View selects the compact index.
type ViewOptions struct {
	View        string
	Path        string
	SurfaceKind string
	SurfaceName string
}

// Representation is one cacheable schema response.
type Representation struct {
	Body        []byte
	ContentType string
	ETag        string
}

// ViewError reports a client-selectable schema view that does not exist.
type ViewError struct {
	Message string
}

func (err *ViewError) Error() string { return err.Message }

type indexSection struct {
	Path        string `json:"path"`
	Title       string `json:"title"`
	Description string `json:"description,omitempty"`
	Required    bool   `json:"required"`
	Href        string `json:"href"`
}

type surfaceIndex struct {
	Count        int      `json:"count"`
	Names        []string `json:"names"`
	HrefTemplate string   `json:"href_template"`
}

type viewDescriptor struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Href        string `json:"href"`
}

type schemaIndex struct {
	ContractVersion string                  `json:"contract_version"`
	ConfigVersion   string                  `json:"config_version"`
	SchemaID        string                  `json:"schema_id"`
	SchemaETag      string                  `json:"schema_etag"`
	DefaultView     string                  `json:"default_view"`
	Views           []viewDescriptor        `json:"views"`
	Sections        []indexSection          `json:"sections"`
	Surfaces        map[string]surfaceIndex `json:"surfaces"`
}

// Render returns the requested full, index, section, or routing-surface view.
func Render(options ViewOptions) (Representation, error) {
	view := strings.ToLower(strings.TrimSpace(options.View))
	if view == "" {
		view = ViewIndex
	}
	if view == ViewFull {
		return representation(Document(), "application/schema+json"), nil
	}

	document, err := decodeDocument()
	if err != nil {
		return Representation{}, err
	}

	var payload any
	contentType := "application/json"
	switch view {
	case ViewIndex:
		payload, err = buildIndex(document)
	case ViewSection:
		contentType = "application/schema+json"
		payload, err = buildSection(document, options.Path)
	case ViewSurface:
		contentType = "application/schema+json"
		payload, err = buildSurface(document, options.SurfaceKind, options.SurfaceName)
	default:
		err = &ViewError{Message: fmt.Sprintf("unsupported schema view %q; use full, index, section, or surface", options.View)}
	}
	if err != nil {
		return Representation{}, err
	}
	body, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return Representation{}, fmt.Errorf("encode schema %s view: %w", view, err)
	}
	return representation(append(body, '\n'), contentType), nil
}

func representation(body []byte, contentType string) Representation {
	digest := sha256.Sum256(body)
	return Representation{
		Body:        body,
		ContentType: contentType,
		ETag:        `"sha256:` + hex.EncodeToString(digest[:]) + `"`,
	}
}

func decodeDocument() (map[string]any, error) {
	var document map[string]any
	if err := json.Unmarshal(embeddedSchema, &document); err != nil {
		return nil, fmt.Errorf("decode embedded config schema: %w", err)
	}
	return document, nil
}

func buildIndex(document map[string]any) (schemaIndex, error) {
	properties, _ := document["properties"].(map[string]any)
	required := stringSet(document["required"])
	sectionNames := make([]string, 0, len(properties))
	for name := range properties {
		sectionNames = append(sectionNames, name)
	}
	sort.Strings(sectionNames)
	sections := make([]indexSection, 0, len(sectionNames))
	for _, name := range sectionNames {
		node, _ := properties[name].(map[string]any)
		resolved, resolveErr := resolveSchemaNode(document, node)
		if resolveErr != nil {
			return schemaIndex{}, resolveErr
		}
		sections = append(sections, indexSection{
			Path:        name,
			Title:       firstString(resolved, "title", displayName(name)),
			Description: firstString(resolved, "description", ""),
			Required:    required[name],
			Href:        SchemaEndpoint + "?view=section&path=" + url.QueryEscape(name),
		})
	}

	extension, ok := document["x-vllm-sr"].(map[string]any)
	if !ok {
		return schemaIndex{}, fmt.Errorf("embedded config schema is missing x-vllm-sr")
	}
	surfaces := make(map[string]surfaceIndex, 4)
	for _, catalog := range []struct {
		kind       string
		collection string
		nameKey    string
	}{
		{kind: "signal", collection: "signals", nameKey: "type"},
		{kind: "algorithm", collection: "algorithms", nameKey: "type"},
		{kind: "plugin", collection: "plugins", nameKey: "type"},
		{kind: "projection", collection: "projections", nameKey: "collection"},
	} {
		entries, _ := extension[catalog.collection].([]any)
		names := make([]string, 0, len(entries))
		for _, rawEntry := range entries {
			entry, _ := rawEntry.(map[string]any)
			if name, ok := entry[catalog.nameKey].(string); ok {
				names = append(names, name)
			}
		}
		sort.Strings(names)
		surfaces[catalog.kind] = surfaceIndex{
			Count:        len(names),
			Names:        names,
			HrefTemplate: SchemaEndpoint + "?view=surface&kind=" + catalog.kind + "&name={name}",
		}
	}

	return schemaIndex{
		ContractVersion: firstString(extension, "contract_version", ContractVersion),
		ConfigVersion:   firstString(extension, "config_version", ConfigVersion),
		SchemaID:        firstString(document, "$id", SchemaID),
		SchemaETag:      ETag(),
		DefaultView:     ViewIndex,
		Views: []viewDescriptor{
			{Name: ViewIndex, Description: "Compact section and routing-surface directory.", Href: SchemaEndpoint + "?view=index"},
			{Name: ViewSection, Description: "One config path with only its transitive schema definitions.", Href: SchemaEndpoint + "?view=section&path={path}"},
			{Name: ViewSurface, Description: "One signal, algorithm, plugin, or projection contract.", Href: SchemaEndpoint + "?view=surface&kind={kind}&name={name}"},
			{Name: ViewFull, Description: "Complete canonical JSON Schema.", Href: SchemaEndpoint + "?view=full"},
		},
		Sections: sections,
		Surfaces: surfaces,
	}, nil
}

func buildSection(document map[string]any, path string) (map[string]any, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return nil, &ViewError{Message: "section view requires a non-empty path"}
	}
	segments := splitPath(path)
	node, err := schemaAtPath(document, segments)
	if err != nil {
		return nil, &ViewError{Message: err.Error()}
	}
	return focusedSchema(document, node, map[string]any{
		"view": ViewSection,
		"path": strings.Join(segments, "."),
	})
}

func buildSurface(document map[string]any, kind, name string) (map[string]any, error) {
	kind = strings.ToLower(strings.TrimSpace(kind))
	name = strings.TrimSpace(name)
	collection, nameKey, ok := surfaceCatalog(kind)
	if !ok {
		return nil, &ViewError{Message: fmt.Sprintf("unsupported surface kind %q; use signal, algorithm, plugin, or projection", kind)}
	}
	if name == "" {
		return nil, &ViewError{Message: "surface view requires a non-empty name"}
	}
	extension, _ := document["x-vllm-sr"].(map[string]any)
	entries, _ := extension[collection].([]any)
	for _, rawEntry := range entries {
		entry, _ := rawEntry.(map[string]any)
		if entry[nameKey] != name {
			continue
		}
		ref, _ := entry["schema_ref"].(string)
		node := map[string]any{"type": "object"}
		if ref != "" {
			node = map[string]any{"$ref": ref}
		}
		focused, err := focusedSchema(document, node, map[string]any{
			"view": ViewSurface,
			"kind": kind,
			"name": name,
		})
		if err != nil {
			return nil, err
		}
		focused["x-vllm-sr-surface"] = entry
		return focused, nil
	}
	return nil, &ViewError{Message: fmt.Sprintf("unknown %s surface %q", kind, name)}
}

func surfaceCatalog(kind string) (collection, nameKey string, ok bool) {
	switch kind {
	case "signal", "signals":
		return "signals", "type", true
	case "algorithm", "algorithms":
		return "algorithms", "type", true
	case "plugin", "plugins":
		return "plugins", "type", true
	case "projection", "projections":
		return "projections", "collection", true
	default:
		return "", "", false
	}
}

func splitPath(path string) []string {
	parts := strings.FieldsFunc(path, func(r rune) bool { return r == '.' || r == '/' })
	segments := make([]string, 0, len(parts))
	for _, part := range parts {
		if trimmed := strings.TrimSpace(part); trimmed != "" {
			segments = append(segments, trimmed)
		}
	}
	return segments
}

func schemaAtPath(document map[string]any, segments []string) (map[string]any, error) {
	current := document
	for index, segment := range segments {
		resolved, err := resolveSchemaNode(document, current)
		if err != nil {
			return nil, err
		}
		if resolved["type"] == "array" {
			items, _ := resolved["items"].(map[string]any)
			resolved, err = resolveSchemaNode(document, items)
			if err != nil {
				return nil, err
			}
		}
		properties, _ := resolved["properties"].(map[string]any)
		next, ok := properties[segment].(map[string]any)
		if !ok {
			return nil, fmt.Errorf("unknown config section path %q at %q", strings.Join(segments, "."), strings.Join(segments[:index+1], "."))
		}
		current = next
	}
	return current, nil
}

func resolveSchemaNode(document, node map[string]any) (map[string]any, error) {
	if node == nil {
		return nil, fmt.Errorf("schema node is empty")
	}
	ref, _ := node["$ref"].(string)
	if ref == "" {
		return node, nil
	}
	const prefix = "#/$defs/"
	if !strings.HasPrefix(ref, prefix) {
		return nil, fmt.Errorf("unsupported config schema reference %q", ref)
	}
	definitions, _ := document["$defs"].(map[string]any)
	definition, ok := definitions[strings.TrimPrefix(ref, prefix)].(map[string]any)
	if !ok {
		return nil, fmt.Errorf("missing config schema definition %q", ref)
	}
	return definition, nil
}

func focusedSchema(document, node map[string]any, viewMetadata map[string]any) (map[string]any, error) {
	clone, err := cloneMap(node)
	if err != nil {
		return nil, err
	}
	if schemaVersion, ok := document["$schema"]; ok {
		clone["$schema"] = schemaVersion
	}
	clone["x-vllm-sr-view"] = viewMetadata
	definitions, err := referencedDefinitions(document, node)
	if err != nil {
		return nil, err
	}
	if len(definitions) > 0 {
		clone["$defs"] = definitions
	}
	return clone, nil
}

func referencedDefinitions(document, node map[string]any) (map[string]any, error) {
	allDefinitions, _ := document["$defs"].(map[string]any)
	selected := make(map[string]any)
	queue := localDefinitionRefs(node)
	seen := make(map[string]bool)
	for len(queue) > 0 {
		name := queue[0]
		queue = queue[1:]
		if seen[name] {
			continue
		}
		seen[name] = true
		definition, ok := allDefinitions[name]
		if !ok {
			return nil, fmt.Errorf("missing config schema definition %q", name)
		}
		selected[name] = definition
		queue = append(queue, localDefinitionRefs(definition)...)
	}
	return selected, nil
}

func localDefinitionRefs(value any) []string {
	refs := make([]string, 0)
	var walk func(any)
	walk = func(current any) {
		switch typed := current.(type) {
		case map[string]any:
			for key, child := range typed {
				if key == "$ref" {
					if ref, ok := child.(string); ok && strings.HasPrefix(ref, "#/$defs/") {
						refs = append(refs, strings.TrimPrefix(ref, "#/$defs/"))
					}
					continue
				}
				walk(child)
			}
		case []any:
			for _, child := range typed {
				walk(child)
			}
		}
	}
	walk(value)
	return refs
}

func cloneMap(source map[string]any) (map[string]any, error) {
	payload, err := json.Marshal(source)
	if err != nil {
		return nil, fmt.Errorf("clone config schema node: %w", err)
	}
	var clone map[string]any
	if err := json.Unmarshal(payload, &clone); err != nil {
		return nil, fmt.Errorf("clone config schema node: %w", err)
	}
	return clone, nil
}

func firstString(values map[string]any, key, fallback string) string {
	if value, ok := values[key].(string); ok && value != "" {
		return value
	}
	return fallback
}

func displayName(value string) string {
	words := strings.Fields(strings.ReplaceAll(value, "_", " "))
	for index, word := range words {
		if word != "" {
			words[index] = strings.ToUpper(word[:1]) + word[1:]
		}
	}
	return strings.Join(words, " ")
}

func stringSet(value any) map[string]bool {
	result := make(map[string]bool)
	entries, _ := value.([]any)
	for _, entry := range entries {
		if text, ok := entry.(string); ok {
			result[text] = true
		}
	}
	return result
}
