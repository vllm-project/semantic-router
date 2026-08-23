package providercatalog

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

const StaticBackendCompilerID = "static.backend.v1"

const maximumBackendCompilerConfigBytes = 64 << 10

// BackendCompiler is a control-plane extension point. It turns a provider
// integration's validated configuration and typed form values into the final
// non-secret connection consumed by a stable data-plane wire codec.
// Implementations must be deterministic and side-effect free.
type BackendCompiler interface {
	AdapterID() string
	Validate(config map[string]any, fields []ConnectionField) error
	Compile(config map[string]any, values map[string]CanonicalConnectionValue) (routingsnapshot.BackendConnection, error)
}

// BackendCompilerResolver keeps compiler selection on the control-plane side
// of the publication boundary.
type BackendCompilerResolver interface {
	BackendCompiler(string) (BackendCompiler, bool)
}

// StaticBackendCompiler covers the common integration: Provider form values
// determine only origin/credential selection, while invocation path and safe
// non-secret headers are immutable integration metadata.
type StaticBackendCompiler struct{}

func (StaticBackendCompiler) AdapterID() string { return StaticBackendCompilerID }

type staticBackendCompilerConfig struct {
	Path    string            `json:"path"`
	Headers map[string]string `json:"headers,omitempty"`
}

func (StaticBackendCompiler) Validate(config map[string]any, fields []ConnectionField) error {
	if len(fields) != 0 {
		return errors.New("static backend compiler does not accept connection fields")
	}
	_, err := decodeStaticBackendCompilerConfig(config)
	return err
}

func (StaticBackendCompiler) Compile(
	config map[string]any,
	values map[string]CanonicalConnectionValue,
) (routingsnapshot.BackendConnection, error) {
	if len(values) != 0 {
		return routingsnapshot.BackendConnection{}, errors.New("static backend compiler received connection fields")
	}
	decoded, err := decodeStaticBackendCompilerConfig(config)
	if err != nil {
		return routingsnapshot.BackendConnection{}, err
	}
	return routingsnapshot.BackendConnection{
		Path: decoded.Path, Headers: cloneStringMap(decoded.Headers),
	}, nil
}

func decodeStaticBackendCompilerConfig(config map[string]any) (staticBackendCompilerConfig, error) {
	if config == nil {
		return staticBackendCompilerConfig{}, errors.New("static backend compiler config is required")
	}
	payload, err := json.Marshal(config)
	if err != nil {
		return staticBackendCompilerConfig{}, fmt.Errorf("encode static backend compiler config: %w", err)
	}
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.DisallowUnknownFields()
	var decoded staticBackendCompilerConfig
	if err := decoder.Decode(&decoded); err != nil {
		return staticBackendCompilerConfig{}, fmt.Errorf("decode static backend compiler config: %w", err)
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return staticBackendCompilerConfig{}, errors.New("static backend compiler config contains trailing data")
	}
	if decoded.Path == "" {
		return staticBackendCompilerConfig{}, errors.New("static backend compiler path is required")
	}
	if err := validatePath("compiler.config.path", decoded.Path); err != nil {
		return staticBackendCompilerConfig{}, err
	}
	if err := validateHeaders(decoded.Headers); err != nil {
		return staticBackendCompilerConfig{}, fmt.Errorf("compiler.config: %w", err)
	}
	return decoded, nil
}

func cloneCompilerConfig(source map[string]any) (map[string]any, error) {
	if source == nil {
		return nil, nil
	}
	payload, err := json.Marshal(source)
	if err != nil {
		return nil, fmt.Errorf("encode backend compiler config: %w", err)
	}
	if len(payload) > maximumBackendCompilerConfigBytes {
		return nil, fmt.Errorf("backend compiler config exceeds %d bytes", maximumBackendCompilerConfigBytes)
	}
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.UseNumber()
	var cloned map[string]any
	if err := decoder.Decode(&cloned); err != nil {
		return nil, fmt.Errorf("decode backend compiler config: %w", err)
	}
	return cloned, nil
}

func cloneCanonicalConnectionValues(
	source map[string]CanonicalConnectionValue,
) map[string]CanonicalConnectionValue {
	if source == nil {
		return nil
	}
	cloned := make(map[string]CanonicalConnectionValue, len(source))
	for name, value := range source {
		cloned[name] = value
	}
	return cloned
}
