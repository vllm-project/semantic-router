package backenddispatch

import (
	"crypto/sha256"
	"fmt"
	"net/http"
	"reflect"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

const maximumRequestBodyBytes int64 = 256 << 20

// Options contains the complete production dispatch boundary. New takes
// ownership of Transport and copies CapabilityKeyring on success. Service
// dependencies are borrowed and must remain valid until Runtime.Close returns.
type Options struct {
	Audience            string
	CapabilityKeyring   backendinvoker.SigningKeyring
	Snapshots           backendinvoker.RoutingSnapshotSource
	Credentials         backendinvoker.CredentialResolver
	Codecs              *protocolcodec.Registry
	Journal             backendinvoker.Journal
	Finalizer           backendinvoker.ResponseFinalizer
	Observer            backendinvoker.ResponseObserver
	Transport           backendinvoker.Transport
	MaxRequestBodyBytes int64
	Now                 func() time.Time
}

// Runtime is the stable internal HTTP dispatch boundary. Close waits for every
// active response stream before zeroing capability keys and closing Transport.
type Runtime struct {
	mu           sync.RWMutex
	handler      *backendinvoker.Handler
	transport    backendinvoker.Transport
	capabilities []protocolcodec.Capability
	closed       bool
	closeErr     error
}

// New validates and composes the immutable snapshot resolver, invoker, and
// capability-verifying HTTP handler. It never installs default dependencies.
func New(options Options) (*Runtime, error) {
	if !canonicalAudience(options.Audience) {
		return nil, fmt.Errorf("dispatch capability audience is invalid")
	}
	if isNil(options.Snapshots) {
		return nil, fmt.Errorf("routing snapshot source is required")
	}
	if isNil(options.Credentials) {
		return nil, fmt.Errorf("provider credential resolver is required")
	}
	if options.Codecs == nil {
		return nil, fmt.Errorf("protocol codec registry is required")
	}
	capabilities := options.Codecs.Capabilities()
	if len(capabilities) == 0 {
		return nil, fmt.Errorf("protocol codec registry is empty")
	}
	if err := validateCapabilities(capabilities); err != nil {
		return nil, err
	}
	if isNil(options.Journal) {
		return nil, fmt.Errorf("dispatch journal is required")
	}
	if isNil(options.Finalizer) {
		return nil, fmt.Errorf("response terminal finalizer is required")
	}
	if isNil(options.Observer) {
		return nil, fmt.Errorf("response observer is required")
	}
	if isNil(options.Transport) {
		return nil, fmt.Errorf("backend transport is required")
	}
	if options.MaxRequestBodyBytes <= 0 || options.MaxRequestBodyBytes > maximumRequestBodyBytes {
		return nil, fmt.Errorf("maximum request body must be between 1 and %d bytes", maximumRequestBodyBytes)
	}
	keyring, err := cloneCapabilityKeyring(options.CapabilityKeyring)
	if err != nil {
		return nil, err
	}

	clock := options.Now
	if clock == nil {
		clock = time.Now
	}
	plans := &backendinvoker.SnapshotPlanResolver{Source: options.Snapshots}
	invoker := &backendinvoker.Invoker{
		Transport: options.Transport, Credentials: options.Credentials,
		Codecs: options.Codecs, Journal: options.Journal,
		Finalizer: options.Finalizer, Now: clock,
	}
	handler := &backendinvoker.Handler{
		Audience: options.Audience, Keyring: keyring, Plans: plans,
		Invoker: invoker, Observer: options.Observer,
		MaxRequestBody: options.MaxRequestBodyBytes, Now: clock,
	}
	return &Runtime{
		handler: handler, transport: options.Transport,
		capabilities: append([]protocolcodec.Capability(nil), capabilities...),
	}, nil
}

// Handler returns the stable process-owned handler. It remains safe to mount
// for the Runtime lifetime and fails closed after Close.
func (runtime *Runtime) Handler() http.Handler {
	return runtime
}

func (runtime *Runtime) ServeHTTP(writer http.ResponseWriter, request *http.Request) {
	if runtime == nil {
		http.Error(writer, "backend dispatch is unavailable", http.StatusServiceUnavailable)
		return
	}
	runtime.mu.RLock()
	defer runtime.mu.RUnlock()
	if runtime.closed || runtime.handler == nil {
		http.Error(writer, "backend dispatch is unavailable", http.StatusServiceUnavailable)
		return
	}
	runtime.handler.ServeHTTP(writer, request)
}

// WireCapabilities returns a defensive copy of the stable codec capabilities
// installed at construction.
func (runtime *Runtime) WireCapabilities() []protocolcodec.Capability {
	if runtime == nil {
		return nil
	}
	runtime.mu.RLock()
	defer runtime.mu.RUnlock()
	return append([]protocolcodec.Capability(nil), runtime.capabilities...)
}

// Close is idempotent. It blocks until active dispatch streams finish, then
// zeroes the owned capability keys and tears down the owned transport.
func (runtime *Runtime) Close() error {
	if runtime == nil {
		return nil
	}
	runtime.mu.Lock()
	defer runtime.mu.Unlock()
	if runtime.closed {
		return runtime.closeErr
	}
	runtime.closed = true
	if runtime.handler != nil {
		zeroKeyring(&runtime.handler.Keyring)
		runtime.handler = nil
	}
	transport := runtime.transport
	runtime.transport = nil
	switch closer := transport.(type) {
	case interface{ Close() error }:
		runtime.closeErr = closer.Close()
	case interface{ CloseIdleConnections() }:
		closer.CloseIdleConnections()
	}
	return runtime.closeErr
}

func cloneCapabilityKeyring(source backendinvoker.SigningKeyring) (backendinvoker.SigningKeyring, error) {
	if !canonicalKeyVersion(source.ActiveVersion) || len(source.Keys) == 0 {
		return backendinvoker.SigningKeyring{}, fmt.Errorf("dispatch capability keyring is empty or invalid")
	}
	if source.MaxLifetime <= 0 {
		return backendinvoker.SigningKeyring{}, fmt.Errorf("dispatch capability keyring lifetime must be positive")
	}
	result := backendinvoker.SigningKeyring{
		ActiveVersion: source.ActiveVersion,
		Keys:          make(map[string][]byte, len(source.Keys)),
		MaxLifetime:   source.MaxLifetime,
	}
	for version, key := range source.Keys {
		if !canonicalKeyVersion(version) || len(key) < sha256.Size {
			zeroKeyring(&result)
			return backendinvoker.SigningKeyring{}, fmt.Errorf("dispatch capability key %q is invalid", version)
		}
		result.Keys[version] = append([]byte(nil), key...)
	}
	if _, found := result.Keys[result.ActiveVersion]; !found {
		zeroKeyring(&result)
		return backendinvoker.SigningKeyring{}, fmt.Errorf("active dispatch capability key is unavailable")
	}
	return result, nil
}

func canonicalKeyVersion(value string) bool {
	if value == "" || value != strings.TrimSpace(value) || len(value) > 128 {
		return false
	}
	for _, character := range value {
		if (character >= 'a' && character <= 'z') ||
			(character >= 'A' && character <= 'Z') ||
			(character >= '0' && character <= '9') ||
			strings.ContainsRune("_-", character) {
			continue
		}
		return false
	}
	return true
}

func zeroKeyring(keyring *backendinvoker.SigningKeyring) {
	if keyring == nil {
		return
	}
	for _, key := range keyring.Keys {
		clear(key)
	}
	keyring.ActiveVersion = ""
	keyring.Keys = nil
	keyring.MaxLifetime = 0
}

func validateCapabilities(capabilities []protocolcodec.Capability) error {
	seen := make(map[string]struct{}, len(capabilities))
	for _, capability := range capabilities {
		if !canonicalIdentifier(string(capability.Format)) {
			return fmt.Errorf("protocol codec capability has an invalid identity")
		}
		if _, duplicate := seen[string(capability.Format)]; duplicate {
			return fmt.Errorf("protocol codec capability %q is duplicated", capability.Format)
		}
		seen[string(capability.Format)] = struct{}{}
	}
	return nil
}

func canonicalAudience(value string) bool {
	if value == "" || value != strings.TrimSpace(value) || len(value) > 128 {
		return false
	}
	for index, character := range value {
		if (character >= 'a' && character <= 'z') ||
			(character >= '0' && character <= '9') ||
			(index > 0 && strings.ContainsRune("._:/-", character)) {
			continue
		}
		return false
	}
	return true
}

func canonicalIdentifier(value string) bool {
	if value == "" || value != strings.TrimSpace(value) || len(value) > 128 {
		return false
	}
	for index, character := range value {
		if (character >= 'a' && character <= 'z') ||
			(character >= 'A' && character <= 'Z') ||
			(character >= '0' && character <= '9') ||
			(index > 0 && strings.ContainsRune("._-", character)) {
			continue
		}
		return false
	}
	return true
}

func isNil(value any) bool {
	if value == nil {
		return true
	}
	reflected := reflect.ValueOf(value)
	switch reflected.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return reflected.IsNil()
	default:
		return false
	}
}
