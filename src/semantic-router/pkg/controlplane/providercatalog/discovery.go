package providercatalog

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/google/uuid"
)

// CanonicalConnectionDigest binds the normalized, non-secret Provider form
// without exposing it to the routing data plane.
func CanonicalConnectionDigest(fields map[string]CanonicalConnectionValue) (string, error) {
	encoded, err := json.Marshal(fields)
	if err != nil {
		return "", fmt.Errorf("%w: encode canonical connection fields", ErrInvalidRequest)
	}
	// Domain separation and an explicit encoding version prevent this digest
	// from being confused with another SHA-256 claim binding if the canonical
	// form evolves.
	digest := sha256.New()
	_, _ = digest.Write([]byte("vllm-sr.provider-connection.v1\x00"))
	_, _ = digest.Write(encoded)
	return fmt.Sprintf("sha256:%x", digest.Sum(nil)), nil
}

var (
	ErrDiscoveryUnsupported       = errors.New("provider does not support model discovery")
	ErrDiscoveryPluginUnavailable = errors.New("provider discovery plugin is unavailable")
)

// DiscoveryRequestValidator is a plugin seam, not an executor. Implementations
// validate adapter-specific, non-secret plan fields and must not perform I/O.
// Network execution, credential resolution, and egress enforcement belong to a
// later privileged discovery executor.
type DiscoveryRequestValidator interface {
	AdapterID() string
	ValidateDiscovery(context.Context, DiscoveryPlan) error
}

type DiscoveryRegistry struct {
	plugins map[string]DiscoveryRequestValidator
}

func NewDiscoveryRegistry(validators []DiscoveryRequestValidator) (*DiscoveryRegistry, error) {
	registry := &DiscoveryRegistry{plugins: make(map[string]DiscoveryRequestValidator, len(validators))}
	for index, validator := range validators {
		if validator == nil {
			return nil, fmt.Errorf("discovery validator %d is nil", index)
		}
		adapterID := validator.AdapterID()
		if !idPattern.MatchString(adapterID) {
			return nil, fmt.Errorf("discovery validator %d has an invalid adapter ID", index)
		}
		if _, exists := registry.plugins[adapterID]; exists {
			return nil, fmt.Errorf("discovery adapter %q is registered more than once", adapterID)
		}
		registry.plugins[adapterID] = validator
	}
	return registry, nil
}

func (registry *DiscoveryRegistry) clone() *DiscoveryRegistry {
	result := &DiscoveryRegistry{plugins: make(map[string]DiscoveryRequestValidator)}
	if registry == nil {
		return result
	}
	for adapterID, plugin := range registry.plugins {
		result.plugins[adapterID] = plugin
	}
	return result
}

type DiscoverModelsRequest struct {
	NamespaceID      string
	CredentialID     string
	Origin           string
	ConnectionFields map[string]any
	Search           string
	PageSize         int
	ProviderCursor   string
}

type CanonicalConnectionValue struct {
	Kind  FieldKind
	Value string
}

// DiscoveryPlan is the validated hand-off to a privileged executor. It has no
// credential material and contains only integration-selected adapter identifiers.
type DiscoveryPlan struct {
	CatalogRevision     string
	NamespaceID         string
	ProviderID          string
	DiscoveryAdapterID  string
	CredentialMode      CredentialMode
	CredentialAdapterID string
	CredentialID        string
	NormalizedOrigin    string
	Path                string
	Headers             map[string]string
	Capabilities        []string
	ConnectionFields    map[string]CanonicalConnectionValue
	Search              string
	PageSize            int
	ProviderCursor      string
}

func (service *Service) PrepareDiscovery(
	ctx context.Context,
	providerID string,
	request DiscoverModelsRequest,
) (DiscoveryPlan, error) {
	if service == nil || !idPattern.MatchString(providerID) {
		return DiscoveryPlan{}, fmt.Errorf("%w: provider ID is invalid", ErrInvalidRequest)
	}
	snapshot, _, err := service.activeSnapshot(ctx)
	if err != nil {
		return DiscoveryPlan{}, err
	}
	provider, found := snapshot.Get(providerID)
	if !found {
		return DiscoveryPlan{}, ErrNotFound
	}
	if provider.Discovery == nil {
		return DiscoveryPlan{}, ErrDiscoveryUnsupported
	}
	validator, found := service.discovery.plugins[provider.Discovery.AdapterID]
	if !found {
		return DiscoveryPlan{}, fmt.Errorf("%w: adapter %q", ErrDiscoveryPluginUnavailable, provider.Discovery.AdapterID)
	}
	if !canonicalUUID(request.NamespaceID) {
		return DiscoveryPlan{}, fmt.Errorf("%w: namespace ID is invalid", ErrInvalidRequest)
	}
	if request.CredentialID != "" && !canonicalUUID(request.CredentialID) {
		return DiscoveryPlan{}, fmt.Errorf("%w: credential ID is invalid", ErrInvalidRequest)
	}
	switch provider.Credential.Mode {
	case CredentialNone:
		if request.CredentialID != "" {
			return DiscoveryPlan{}, fmt.Errorf("%w: credential is forbidden for this provider", ErrInvalidRequest)
		}
	case CredentialRequired:
		if request.CredentialID == "" {
			return DiscoveryPlan{}, fmt.Errorf("%w: credential is required for this provider", ErrInvalidRequest)
		}
	case CredentialOptional:
	default:
		return DiscoveryPlan{}, fmt.Errorf("%w: provider credential mode is invalid", ErrInvalidRequest)
	}

	origin, err := providerOrigin(provider, request.Origin)
	if err != nil {
		return DiscoveryPlan{}, err
	}
	fields, err := normalizeConnectionFields(provider.ConnectionFields, request.ConnectionFields)
	if err != nil {
		return DiscoveryPlan{}, err
	}
	pageSize := request.PageSize
	if pageSize == 0 {
		pageSize = defaultPageSize
	}
	if pageSize < 1 || pageSize > maximumPageSize {
		return DiscoveryPlan{}, fmt.Errorf("%w: page size must be between 1 and %d", ErrInvalidRequest, maximumPageSize)
	}
	if request.Search != "" && !canonicalText(request.Search, 1, 256) {
		return DiscoveryPlan{}, fmt.Errorf("%w: discovery search is invalid", ErrInvalidRequest)
	}
	if request.ProviderCursor != "" && !canonicalText(request.ProviderCursor, 1, 4096) {
		return DiscoveryPlan{}, fmt.Errorf("%w: provider cursor is invalid", ErrInvalidRequest)
	}
	plan := DiscoveryPlan{
		CatalogRevision: snapshot.Revision(), NamespaceID: request.NamespaceID,
		ProviderID: provider.ID, DiscoveryAdapterID: provider.Discovery.AdapterID,
		CredentialMode:      provider.Credential.Mode,
		CredentialAdapterID: provider.Credential.AdapterID, CredentialID: request.CredentialID,
		NormalizedOrigin: origin, Path: provider.Discovery.Path, ConnectionFields: fields,
		Headers:      cloneStringMap(provider.Discovery.Headers),
		Capabilities: append([]string(nil), provider.Capabilities...),
		Search:       request.Search, PageSize: pageSize, ProviderCursor: request.ProviderCursor,
	}
	if err := validator.ValidateDiscovery(ctx, cloneDiscoveryPlan(plan)); err != nil {
		return DiscoveryPlan{}, fmt.Errorf("%w: discovery adapter %q rejected the request: %w", ErrInvalidRequest, provider.Discovery.AdapterID, err)
	}
	return plan, nil
}

func cloneDiscoveryPlan(plan DiscoveryPlan) DiscoveryPlan {
	cloned := plan
	cloned.Headers = cloneStringMap(plan.Headers)
	cloned.Capabilities = append([]string(nil), plan.Capabilities...)
	cloned.ConnectionFields = make(map[string]CanonicalConnectionValue, len(plan.ConnectionFields))
	for name, value := range plan.ConnectionFields {
		cloned.ConnectionFields[name] = value
	}
	return cloned
}

func normalizeConnectionFields(
	schema []ConnectionField,
	provided map[string]any,
) (map[string]CanonicalConnectionValue, error) {
	if len(provided) > 64 {
		return nil, fmt.Errorf("%w: too many connection fields", ErrInvalidRequest)
	}
	byName := make(map[string]ConnectionField, len(schema))
	for _, field := range schema {
		byName[field.Name] = field
	}
	for name := range provided {
		if _, found := byName[name]; !found {
			return nil, fmt.Errorf("%w: connection field %q is not declared by the provider", ErrInvalidRequest, name)
		}
	}
	result := make(map[string]CanonicalConnectionValue, len(schema))
	for _, field := range schema {
		raw, supplied := provided[field.Name]
		if !supplied {
			if field.Default != "" {
				raw = defaultFieldValue(field)
				supplied = true
			} else if field.Required {
				return nil, fmt.Errorf("%w: connection field %q is required", ErrInvalidRequest, field.Name)
			}
		}
		if !supplied {
			continue
		}
		value, err := canonicalConnectionValue(field, raw)
		if err != nil {
			return nil, fmt.Errorf("%w: connection field %q: %w", ErrInvalidRequest, field.Name, err)
		}
		result[field.Name] = value
	}
	return result, nil
}

func defaultFieldValue(field ConnectionField) any {
	switch field.Kind {
	case FieldBoolean:
		return field.Default == "true"
	case FieldInteger:
		return json.Number(field.Default)
	default:
		return field.Default
	}
}

func canonicalConnectionValue(field ConnectionField, raw any) (CanonicalConnectionValue, error) {
	result := CanonicalConnectionValue{Kind: field.Kind}
	switch field.Kind {
	case FieldText:
		value, ok := raw.(string)
		if !ok || !canonicalText(value, 1, 2048) {
			return CanonicalConnectionValue{}, errors.New("must be a bounded string")
		}
		result.Value = value
	case FieldSelect:
		value, ok := raw.(string)
		if !ok {
			return CanonicalConnectionValue{}, errors.New("must be a string option")
		}
		for _, option := range field.Options {
			if value == option.Value {
				result.Value = value
				return result, nil
			}
		}
		return CanonicalConnectionValue{}, errors.New("does not name an allowed option")
	case FieldBoolean:
		value, ok := raw.(bool)
		if !ok {
			return CanonicalConnectionValue{}, errors.New("must be a boolean")
		}
		result.Value = strconv.FormatBool(value)
	case FieldInteger:
		value, err := canonicalInteger(raw)
		if err != nil {
			return CanonicalConnectionValue{}, err
		}
		result.Value = value
	default:
		return CanonicalConnectionValue{}, errors.New("has an unsupported kind")
	}
	return result, nil
}

func canonicalInteger(raw any) (string, error) {
	var value int64
	switch number := raw.(type) {
	case int:
		value = int64(number)
	case int8:
		value = int64(number)
	case int16:
		value = int64(number)
	case int32:
		value = int64(number)
	case int64:
		value = number
	case uint:
		if uint64(number) > math.MaxInt64 {
			return "", errors.New("integer exceeds int64")
		}
		value = int64(number)
	case uint8:
		value = int64(number)
	case uint16:
		value = int64(number)
	case uint32:
		value = int64(number)
	case uint64:
		if number > math.MaxInt64 {
			return "", errors.New("integer exceeds int64")
		}
		value = int64(number)
	case json.Number:
		parsed, err := strconv.ParseInt(string(number), 10, 64)
		if err != nil || strconv.FormatInt(parsed, 10) != string(number) {
			return "", errors.New("must be a canonical integer")
		}
		value = parsed
	default:
		return "", errors.New("must be an integer")
	}
	return strconv.FormatInt(value, 10), nil
}

func canonicalUUID(value string) bool {
	parsed, err := uuid.Parse(value)
	return err == nil && parsed.String() == strings.ToLower(value)
}
