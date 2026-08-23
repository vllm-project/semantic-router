package providercatalog

import (
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential"
)

var (
	idPattern         = regexp.MustCompile(`^[a-z][a-z0-9._-]{0,127}$`)
	fieldPattern      = regexp.MustCompile(`^[a-z][a-z0-9_]{0,63}$`)
	capabilityPattern = regexp.MustCompile(`^[a-z][a-z0-9._-]{0,127}$`)
	accentPattern     = regexp.MustCompile(`^#[0-9a-fA-F]{6}$`)
	pathPattern       = regexp.MustCompile(`^[A-Za-z0-9/._~!$&'()*+,;=:@-]+$`)
	iconNamePattern   = regexp.MustCompile(`^[a-z0-9][a-z0-9-]{0,127}$`)
)

var forbiddenHeaders = map[string]struct{}{
	"authorization": {}, "proxy-authorization": {}, "x-api-key": {}, "api-key": {},
	"cookie": {}, "set-cookie": {}, "host": {}, "content-length": {}, "transfer-encoding": {},
}

// adapterCapabilities is the package-private validation view of the process
// registry. It keeps provider product metadata on the control-plane side while
// validating references to stable compiler and wire-format capabilities.
type adapterCapabilities interface {
	BackendCompilerResolver
	HasWireFormat(string) bool
	HasCredentialAdapter(string) bool
	HasDiscoveryAdapter(string) bool
}

func validateDefinition(provider *Definition, adapters adapterCapabilities) error {
	if !idPattern.MatchString(provider.ID) {
		return errors.New("providerId is invalid")
	}
	if provider.Revision != "" {
		return errors.New("provider revision is registry-owned")
	}
	if err := validateDisplay(provider.Display); err != nil {
		return err
	}
	if len(provider.Interfaces) == 0 || len(provider.Interfaces) > 8 {
		return errors.New("provider requires 1-8 interfaces")
	}
	seenInterfaces := make(map[string]struct{}, len(provider.Interfaces))
	defaultInterfaces := 0
	for index := range provider.Interfaces {
		providerInterface := &provider.Interfaces[index]
		if !idPattern.MatchString(providerInterface.ID) || !canonicalText(providerInterface.Label, 1, 128) {
			return fmt.Errorf("interfaces[%d] identity or label is invalid", index)
		}
		if _, duplicate := seenInterfaces[providerInterface.ID]; duplicate {
			return fmt.Errorf("provider interface %q is duplicated", providerInterface.ID)
		}
		seenInterfaces[providerInterface.ID] = struct{}{}
		if providerInterface.Default {
			defaultInterfaces++
		}
		if !idPattern.MatchString(string(providerInterface.WireFormat)) || adapters == nil ||
			!adapters.HasWireFormat(string(providerInterface.WireFormat)) {
			return fmt.Errorf("interfaces[%d] wire format %q is not installed", index, providerInterface.WireFormat)
		}
		if !idPattern.MatchString(providerInterface.Compiler.AdapterID) {
			return fmt.Errorf("interfaces[%d] compiler.adapterId is invalid", index)
		}
		compiler, found := adapters.BackendCompiler(providerInterface.Compiler.AdapterID)
		if !found || compiler == nil || compiler.AdapterID() != providerInterface.Compiler.AdapterID {
			return fmt.Errorf("backend compiler %q is not installed", providerInterface.Compiler.AdapterID)
		}
		providerInterface.Capabilities = sortedUnique(providerInterface.Capabilities)
		for _, capability := range providerInterface.Capabilities {
			if !capabilityPattern.MatchString(capability) {
				return fmt.Errorf("interfaces[%d] capability %q is invalid", index, capability)
			}
		}
	}
	if defaultInterfaces != 1 {
		return errors.New("provider requires exactly one default interface")
	}
	if err := validateCredential(provider.Credential, adapters); err != nil {
		return err
	}
	if err := validateOrigin(&provider.Origin); err != nil {
		return err
	}
	if provider.Discovery != nil {
		if !idPattern.MatchString(provider.Discovery.AdapterID) || !adapters.HasDiscoveryAdapter(provider.Discovery.AdapterID) {
			return fmt.Errorf("discovery adapter %q is not installed", provider.Discovery.AdapterID)
		}
		if provider.Discovery.Path == "" {
			return errors.New("discovery.path is required")
		}
		if err := validatePath("discovery.path", provider.Discovery.Path); err != nil {
			return err
		}
		if err := validateHeaders(provider.Discovery.Headers); err != nil {
			return fmt.Errorf("discovery: %w", err)
		}
	}
	provider.Capabilities = sortedUnique(provider.Capabilities)
	for _, capability := range provider.Capabilities {
		if !capabilityPattern.MatchString(capability) {
			return fmt.Errorf("capability %q is invalid", capability)
		}
	}
	if len(provider.ConnectionFields) > 64 {
		return errors.New("provider cannot define more than 64 connection fields")
	}
	seenFields := make(map[string]struct{}, len(provider.ConnectionFields))
	for index := range provider.ConnectionFields {
		field := &provider.ConnectionFields[index]
		if _, exists := seenFields[field.Name]; exists {
			return fmt.Errorf("duplicate connection field %q", field.Name)
		}
		seenFields[field.Name] = struct{}{}
		if err := validateField(field); err != nil {
			return fmt.Errorf("connectionFields[%d]: %w", index, err)
		}
	}
	for index, providerInterface := range provider.Interfaces {
		compiler, _ := adapters.BackendCompiler(providerInterface.Compiler.AdapterID)
		compilerConfig, err := cloneCompilerConfig(providerInterface.Compiler.Config)
		if err != nil {
			return fmt.Errorf("interfaces[%d] compiler.config is invalid: %w", index, err)
		}
		if err := compiler.Validate(compilerConfig, cloneConnectionFields(provider.ConnectionFields)); err != nil {
			return fmt.Errorf("backend compiler %q rejected Provider interface %q: %w", providerInterface.Compiler.AdapterID, providerInterface.ID, err)
		}
	}
	return nil
}

func validateDisplay(display Display) error {
	for field, value := range map[string]string{
		"display.name": display.Name, "display.description": display.Description,
		"display.category": display.Category,
	} {
		if !canonicalText(value, 1, 512) {
			return fmt.Errorf("%s is required and must be canonical", field)
		}
	}
	if display.Monogram != "" && !canonicalText(display.Monogram, 1, 8) {
		return errors.New("display.monogram must be canonical and at most 8 bytes")
	}
	if display.Accent != "" && !accentPattern.MatchString(display.Accent) {
		return errors.New("display.accent must be a six-digit hex color")
	}
	return validateIcon(display.Icon)
}

func validateIcon(icon Icon) error {
	if !canonicalText(icon.Value, 1, 512) {
		return errors.New("display.icon.value is required and must be canonical")
	}
	switch icon.Source {
	case "lobe":
		if !iconNamePattern.MatchString(icon.Value) {
			return errors.New("display.icon lobe value is invalid")
		}
	case "asset":
		parsed, err := url.ParseRequestURI(icon.Value)
		if err != nil || !strings.HasPrefix(icon.Value, "/") || strings.HasPrefix(icon.Value, "//") ||
			parsed.Host != "" || parsed.RawQuery != "" || parsed.Fragment != "" || strings.Contains(parsed.Path, "..") {
			return errors.New("display.icon asset must be a safe absolute application path")
		}
	case "url":
		parsed, err := url.Parse(icon.Value)
		if err != nil || parsed.Scheme != "https" || parsed.Host == "" || parsed.User != nil ||
			parsed.RawQuery != "" || parsed.Fragment != "" {
			return errors.New("display.icon URL must be an absolute HTTPS URL without credentials, query parameters, or a fragment")
		}
	default:
		return errors.New("display.icon.source must be lobe, asset, or url")
	}
	return nil
}

func validateCredential(credential Credential, adapters adapterCapabilities) error {
	switch credential.Mode {
	case CredentialNone:
		if credential.AdapterID != "" || credential.Label != "" || credential.Hint != "" {
			return errors.New("credential none cannot define adapter or secret prompt metadata")
		}
	case CredentialOptional, CredentialRequired:
		if !idPattern.MatchString(credential.AdapterID) || !adapters.HasCredentialAdapter(credential.AdapterID) {
			return fmt.Errorf("credential adapter %q is not installed", credential.AdapterID)
		}
		if !canonicalText(credential.Label, 1, 128) || !canonicalOptionalText(credential.Hint, 512) {
			return errors.New("credential prompt metadata is invalid")
		}
	default:
		return errors.New("credential mode must be none, optional, or required")
	}
	return nil
}

func validateOrigin(origin *Origin) error {
	switch origin.Mode {
	case OriginFixed:
		if origin.DefaultURL == "" {
			return errors.New("fixed origin requires defaultUrl")
		}
		normalized, err := providercredential.NormalizeOrigin(origin.DefaultURL)
		if err != nil || normalized != origin.DefaultURL {
			return errors.New("fixed origin defaultUrl must be canonical")
		}
		if origin.Label != "" || origin.Hint != "" {
			return errors.New("fixed origin cannot render a URL prompt")
		}
	case OriginUserSupplied:
		if origin.DefaultURL != "" {
			return errors.New("user-supplied origin cannot define defaultUrl")
		}
		if !canonicalText(origin.Label, 1, 128) || !canonicalOptionalText(origin.Hint, 512) {
			return errors.New("user-supplied origin prompt metadata is invalid")
		}
	default:
		return errors.New("origin mode must be fixed or user_supplied")
	}
	return nil
}

func validateHeaders(headers map[string]string) error {
	for name, value := range headers {
		canonical := http.CanonicalHeaderKey(name)
		if canonical == "" || canonical != name || strings.ContainsAny(value, "\r\n") || !canonicalOptionalText(value, 1024) {
			return fmt.Errorf("connection header %q is not canonical", name)
		}
		if _, forbidden := forbiddenHeaders[strings.ToLower(name)]; forbidden {
			return fmt.Errorf("connection header %q is security-sensitive", name)
		}
	}
	return nil
}

func cloneConnectionFields(source []ConnectionField) []ConnectionField {
	if source == nil {
		return nil
	}
	cloned := make([]ConnectionField, len(source))
	for index, field := range source {
		cloned[index] = field
		cloned[index].Options = append([]FieldOption(nil), field.Options...)
	}
	return cloned
}

func validatePath(field, value string) error {
	if value == "" {
		return nil
	}
	if !strings.HasPrefix(value, "/") || !pathPattern.MatchString(value) || strings.Contains(value, "//") {
		return fmt.Errorf("%s must be a canonical absolute path", field)
	}
	for _, segment := range strings.Split(value, "/") {
		if segment == "." || segment == ".." {
			return fmt.Errorf("%s contains a dot segment", field)
		}
	}
	return nil
}

func validateField(field *ConnectionField) error {
	if !fieldPattern.MatchString(field.Name) || !canonicalText(field.Label, 1, 128) ||
		!canonicalOptionalText(field.Hint, 512) || !canonicalOptionalText(field.Placeholder, 256) {
		return errors.New("field name or display metadata is invalid")
	}
	switch field.Kind {
	case FieldText:
		if len(field.Options) != 0 {
			return errors.New("only select fields may define options")
		}
		if !canonicalOptionalText(field.Default, 256) {
			return errors.New("text field default is invalid")
		}
	case FieldBoolean:
		if len(field.Options) != 0 {
			return errors.New("only select fields may define options")
		}
		if field.Default != "" && field.Default != "true" && field.Default != "false" {
			return errors.New("boolean field default must be true or false")
		}
	case FieldInteger:
		if len(field.Options) != 0 {
			return errors.New("only select fields may define options")
		}
		if field.Default != "" {
			value, err := strconv.ParseInt(field.Default, 10, 64)
			if err != nil || strconv.FormatInt(value, 10) != field.Default {
				return errors.New("integer field default must be a canonical int64")
			}
		}
	case FieldSelect:
		if len(field.Options) == 0 || len(field.Options) > 64 {
			return errors.New("select field requires 1-64 options")
		}
		seen := make(map[string]struct{}, len(field.Options))
		for _, option := range field.Options {
			if !canonicalText(option.Value, 1, 128) || !canonicalText(option.Label, 1, 128) {
				return errors.New("select option is invalid")
			}
			if _, exists := seen[option.Value]; exists {
				return errors.New("select option values must be unique")
			}
			seen[option.Value] = struct{}{}
		}
		if field.Default != "" {
			if _, exists := seen[field.Default]; !exists {
				return errors.New("select default must name an option")
			}
		}
	default:
		return errors.New("field kind must be text, boolean, integer, or select")
	}
	return nil
}

func canonicalText(value string, minimum, maximum int) bool {
	return utf8.ValidString(value) && len(value) >= minimum && len(value) <= maximum && strings.TrimSpace(value) == value &&
		!strings.ContainsAny(value, "\x00\r\n")
}

func canonicalOptionalText(value string, maximum int) bool {
	return value == "" || canonicalText(value, 1, maximum)
}

func sortedUnique(values []string) []string {
	if len(values) == 0 {
		return nil
	}
	copy := append([]string(nil), values...)
	sort.Strings(copy)
	result := copy[:0]
	for _, value := range copy {
		if len(result) == 0 || result[len(result)-1] != value {
			result = append(result, value)
		}
	}
	return result
}
