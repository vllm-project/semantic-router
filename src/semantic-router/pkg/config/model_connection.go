package config

import (
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"math"
	"net"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"

	"golang.org/x/net/http/httpguts"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// The providers a deployment names: the API the Router speaks to it.
const (
	// OpenAIEmbeddingsProvider serves embeddings over an OpenAI-compatible
	// API: POST <endpoint>/embeddings.
	OpenAIEmbeddingsProvider = "openai_embeddings"
	// ClassifyProvider serves the Router's classification contract: POST
	// <endpoint>/classify answering labels, scores, spans or a score.
	ClassifyProvider = "classify"
	// OpenAIChatProvider asks an OpenAI-compatible chat model to classify or
	// judge: POST <endpoint>/chat/completions.
	OpenAIChatProvider = "openai_chat"
	// RerankProvider scores query-document pairs: POST <endpoint>/rerank.
	RerankProvider = "rerank"
	// LegacyHTTPProvider is the deprecated provider of a deployment that
	// names an external[] model; the Router reads it as classify or
	// openai_chat.
	LegacyHTTPProvider = "http"
)

// MaxRemoteTimeoutMs bounds timeout_ms on a remote classifier, chat,
// embedding or rerank deployment.
const MaxRemoteTimeoutMs = 300000

// MaxRemoteRetries bounds max_retries on any connection block.
const MaxRemoteRetries = 10

// IsRemoteProvider reports whether a provider serves a deployment over a
// remote API rather than the model runtime.
func IsRemoteProvider(provider string) bool {
	switch provider {
	case SystemOneProvider, OpenAIEmbeddingsProvider, ClassifyProvider, OpenAIChatProvider, RerankProvider:
		return true
	}
	return false
}

// IsRemote reports whether a remote API serves the deployment.
func (d ModelDeployment) IsRemote() bool { return IsRemoteProvider(d.Provider) }

// DeploymentTLS verifies the server of an https endpoint against CAFile
// (else the system roots) under ServerName (else the endpoint's host), and
// presents CertFile and KeyFile when the server asks for a client
// certificate. Verification is never skipped unless InsecureSkipVerify says
// so.
type DeploymentTLS struct {
	CAFile             string `yaml:"ca_file,omitempty" json:"ca_file,omitempty"`
	ServerName         string `yaml:"server_name,omitempty" json:"server_name,omitempty"`
	CertFile           string `yaml:"cert_file,omitempty" json:"cert_file,omitempty"`
	KeyFile            string `yaml:"key_file,omitempty" json:"key_file,omitempty"`
	InsecureSkipVerify bool   `yaml:"insecure_skip_verify,omitempty" json:"insecure_skip_verify,omitempty"`
}

// DeploymentLimits bounds the bodies of one exchange with a deployment.
type DeploymentLimits struct {
	MaxRequestBytes  int64 `yaml:"max_request_bytes,omitempty" json:"max_request_bytes,omitempty"`
	MaxResponseBytes int64 `yaml:"max_response_bytes,omitempty" json:"max_response_bytes,omitempty"`
}

// DeploymentCapabilities declares what a remote model answers when its API
// does not list it: question types and presets (systemone), and the vector
// dimensions of an embedding model.
type DeploymentCapabilities struct {
	QuestionTypes []string `yaml:"question_types,omitempty" json:"question_types,omitempty"`
	Presets       []string `yaml:"presets,omitempty" json:"presets,omitempty"`
	Dimensions    int      `yaml:"dimensions,omitempty" json:"dimensions,omitempty"`
}

// DeclaredQuestionTypes are the question types a deployment declares, in
// capabilities or the deprecated top-level question_types.
func (d ModelDeployment) DeclaredQuestionTypes() []string {
	if d.Capabilities != nil && len(d.Capabilities.QuestionTypes) > 0 {
		return slices.Clone(d.Capabilities.QuestionTypes)
	}
	return slices.Clone(d.QuestionTypes)
}

// DeclaredPresets are the presets a deployment declares.
func (d ModelDeployment) DeclaredPresets() []string {
	if d.Capabilities == nil {
		return nil
	}
	return slices.Clone(d.Capabilities.Presets)
}

// DeclaredDimensions is the embedding dimension a deployment declares, zero
// when it declares none.
func (d ModelDeployment) DeclaredDimensions() int {
	if d.Capabilities == nil {
		return 0
	}
	return d.Capabilities.Dimensions
}

// RetryCount is the number of transport retries after a first attempt.
func (d ModelDeployment) RetryCount(fallback int) int {
	if d.MaxRetries == nil {
		return fallback
	}
	return *d.MaxRetries
}

// TimeoutOr is the deployment's timeout_ms, or fallback when it sets none.
func (d ModelDeployment) TimeoutOr(fallback time.Duration) time.Duration {
	if d.TimeoutMs > 0 {
		return time.Duration(d.TimeoutMs) * time.Millisecond
	}
	return fallback
}

// RequestLimitOr and ResponseLimitOr are the deployment's body limits, or
// fallback when it sets none.
func (d ModelDeployment) RequestLimitOr(fallback int64) int64 {
	if d.Limits != nil && d.Limits.MaxRequestBytes > 0 {
		return d.Limits.MaxRequestBytes
	}
	return fallback
}

func (d ModelDeployment) ResponseLimitOr(fallback int64) int64 {
	if d.Limits != nil && d.Limits.MaxResponseBytes > 0 {
		return d.Limits.MaxResponseBytes
	}
	return fallback
}

// SplitEndpoint splits an http(s) endpoint into the URL calls are made to,
// without its query, and its encoded query, which every call carries.
func SplitEndpoint(endpoint string) (base string, query string, err error) {
	parsed, err := url.Parse(strings.TrimSpace(endpoint))
	if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" {
		return "", "", fmt.Errorf("endpoint must be an http:// or https:// URL")
	}
	query = parsed.RawQuery
	parsed.RawQuery, parsed.ForceQuery = "", false
	parsed.Path = strings.TrimRight(parsed.Path, "/")
	parsed.RawPath = ""
	return parsed.String(), query, nil
}

// LegacyEndpointURL is the URL of a deprecated llm_endpoint: an address that
// is already a URL is used as it is and its port is ignored; a host takes
// the port and protocol (http by default).
func LegacyEndpointURL(address string, port int, protocol string) string {
	address = strings.TrimSpace(address)
	if strings.Contains(address, "://") {
		return strings.TrimRight(address, "/")
	}
	protocol = strings.ToLower(strings.TrimSpace(protocol))
	if protocol == "" {
		protocol = "http"
	}
	if port > 0 {
		address = net.JoinHostPort(strings.Trim(address, "[]"), strconv.Itoa(port))
	}
	return protocol + "://" + address
}

// ClientConfig is the TLS client configuration of an https endpoint, nil
// when the deployment sets no tls block.
func (t *DeploymentTLS) ClientConfig() (*tls.Config, error) {
	if t == nil {
		return nil, nil
	}
	config := &tls.Config{MinVersion: tls.VersionTLS12, ServerName: t.ServerName, InsecureSkipVerify: t.InsecureSkipVerify} //nolint:gosec // G402: skipped verification is an explicit operator opt-in
	if t.CAFile != "" {
		pem, err := os.ReadFile(t.CAFile)
		if err != nil {
			return nil, fmt.Errorf("tls.ca_file %s cannot be read", t.CAFile)
		}
		pool := x509.NewCertPool()
		if !pool.AppendCertsFromPEM(pem) {
			return nil, fmt.Errorf("tls.ca_file %s holds no PEM certificate", t.CAFile)
		}
		config.RootCAs = pool
	}
	if t.CertFile != "" {
		certificate, err := tls.LoadX509KeyPair(t.CertFile, t.KeyFile)
		if err != nil {
			return nil, fmt.Errorf("tls.cert_file and tls.key_file cannot be loaded: %w", err)
		}
		config.Certificates = []tls.Certificate{certificate}
	}
	return config, nil
}

func (t *DeploymentTLS) validate(endpoint *url.URL) error {
	if t == nil {
		return nil
	}
	if endpoint == nil || endpoint.Scheme != "https" {
		return fmt.Errorf("tls applies to an https:// endpoint")
	}
	for field, path := range map[string]string{"ca_file": t.CAFile, "cert_file": t.CertFile, "key_file": t.KeyFile} {
		if path != "" && !filepath.IsAbs(path) {
			return fmt.Errorf("tls.%s must be an absolute path", field)
		}
	}
	if (t.CertFile == "") != (t.KeyFile == "") {
		return fmt.Errorf("tls.cert_file and tls.key_file go together")
	}
	if t.ServerName != strings.TrimSpace(t.ServerName) || strings.ContainsAny(t.ServerName, "/: ") {
		return fmt.Errorf("tls.server_name must be a host name")
	}
	return nil
}

// credentialHeaders cannot be set in headers: a credential goes in auth,
// read from an environment variable or a file.
var credentialHeaders = []string{
	headers.Authorization, "proxy-authorization", headers.XAPIKey, headers.APIKey, headers.OcpApimSubscriptionKey, "cookie",
}

func validateConnectionHeaders(headers map[string]string, auth *DeploymentAuth) error {
	for name, value := range headers {
		if !httpguts.ValidHeaderFieldName(name) || !httpguts.ValidHeaderFieldValue(value) {
			return fmt.Errorf("headers.%s must be an HTTP header name and value", name)
		}
		lower := strings.ToLower(name)
		if slices.Contains(credentialHeaders, lower) || (auth != nil && strings.EqualFold(auth.HeaderName(), name)) {
			return fmt.Errorf("headers.%s carries a credential; set auth with api_key_env or api_key_file instead", name)
		}
	}
	return nil
}

// validateConnection checks the connection block of an attached or remote
// deployment: endpoint is the parsed endpoint, or nil for a unix socket.
func (d ModelDeployment) validateConnection(endpoint *url.URL, maxTimeoutMs int) error {
	if endpoint != nil && endpoint.User != nil {
		return fmt.Errorf("endpoint must not carry credentials; set auth.api_key_env or auth.api_key_file")
	}
	if endpoint != nil && endpoint.Fragment != "" {
		return fmt.Errorf("endpoint must not carry a fragment")
	}
	if d.TimeoutMs < 0 || d.TimeoutMs > maxTimeoutMs {
		return fmt.Errorf("timeout_ms must be within [1, %d] when set", maxTimeoutMs)
	}
	if d.MaxRetries != nil && (*d.MaxRetries < 0 || *d.MaxRetries > MaxRemoteRetries) {
		return fmt.Errorf("max_retries must be within [0, %d]", MaxRemoteRetries)
	}
	if d.Limits != nil && (d.Limits.MaxRequestBytes < 0 || d.Limits.MaxResponseBytes < 0) {
		return fmt.Errorf("limits must be positive byte counts")
	}
	if err := d.TLS.validate(endpoint); err != nil {
		return err
	}
	if err := validateConnectionHeaders(d.Headers, d.Auth); err != nil {
		return err
	}
	return d.Auth.validate()
}

// hasConnectionBlock reports whether a deployment sets any connection field
// beyond its endpoint.
func (d ModelDeployment) hasConnectionBlock() bool {
	return d.Model != "" || d.Auth != nil || len(d.Headers) > 0 || d.TimeoutMs != 0 || d.MaxRetries != nil || d.TLS != nil || d.Limits != nil
}

// DeploymentGeneration sets an openai_chat deployment's completions.
type DeploymentGeneration struct {
	// MaxTokens bounds one completion; unset keeps each signal's own bound.
	MaxTokens int `yaml:"max_tokens,omitempty" json:"max_tokens,omitempty"`
	// Temperature is the sampling temperature; unset sends 0.
	Temperature *float64 `yaml:"temperature,omitempty" json:"temperature,omitempty"`
	// Reasoning turns the model's reasoning on or off for its answers, in
	// the request form of a built-in reasoning family.
	Reasoning *ExternalModelReasoningConfig `yaml:"reasoning,omitempty" json:"reasoning,omitempty"`
}

// MaxGenerationTokens bounds generation.max_tokens.
const MaxGenerationTokens = 32768

func (g *DeploymentGeneration) validate() error {
	if g == nil {
		return nil
	}
	if g.MaxTokens < 0 || g.MaxTokens > MaxGenerationTokens {
		return fmt.Errorf("generation.max_tokens must be within 1..%d", MaxGenerationTokens)
	}
	if g.Temperature != nil && (math.IsNaN(*g.Temperature) || *g.Temperature < 0 || *g.Temperature > 2) {
		return fmt.Errorf("generation.temperature must be within 0..2")
	}
	if g.Reasoning != nil && (strings.TrimSpace(g.Reasoning.Family) == "" || g.Reasoning.UseReasoning == nil) {
		return fmt.Errorf("generation.reasoning needs a family and use_reasoning")
	}
	return nil
}

// validateRemote checks a classify, openai_chat, openai_embeddings or rerank
// deployment: a connection block and nothing the model runtime owns.
func (d ModelDeployment) validateRemote() error {
	if d.Artifact != "" || d.Revision != "" || d.ExternalModel != "" || d.Device != "" || d.HasEngineArgs() ||
		len(d.Replicas) != 0 || d.ServedName != "" || d.PublicName != "" {
		return fmt.Errorf("a %s deployment takes a connection block (endpoint, model, auth, headers, timeout_ms, max_retries, tls, limits); artifact, revision, external_model, device, engine arguments, replicas, served_name and public_name belong to model_runtime", d.Provider)
	}
	if d.Input.MaxTokens != 0 || (d.Input.Overflow != "" && d.Input.Overflow != "reject") {
		return fmt.Errorf("a %s deployment takes no input budget: the remote model reads its own input", d.Provider)
	}
	if d.OperatingPoint != nil {
		return fmt.Errorf("operating_point pins a managed model_runtime head; a %s deployment answers with its own scores", d.Provider)
	}
	if d.Generation != nil {
		if d.Provider != OpenAIChatProvider {
			return fmt.Errorf("generation applies to openai_chat deployments")
		}
		if err := d.Generation.validate(); err != nil {
			return err
		}
	}
	if len(d.QuestionTypes) > 0 || (d.Capabilities != nil && (len(d.Capabilities.QuestionTypes) > 0 || len(d.Capabilities.Presets) > 0)) {
		return fmt.Errorf("question types and presets apply to systemone deployments")
	}
	if d.Capabilities != nil && d.Capabilities.Dimensions != 0 && d.Provider != OpenAIEmbeddingsProvider {
		return fmt.Errorf("capabilities.dimensions applies to openai_embeddings deployments")
	}
	if d.Capabilities != nil && d.Capabilities.Dimensions < 0 {
		return fmt.Errorf("capabilities.dimensions must be positive")
	}
	endpoint, err := url.Parse(strings.TrimSpace(d.Endpoint))
	if err != nil || (endpoint.Scheme != "http" && endpoint.Scheme != "https") || endpoint.Host == "" {
		return fmt.Errorf("endpoint must be the http:// or https:// base URL of the %s API, such as https://models.example.com/v1", d.Provider)
	}
	if endpoint.RawQuery != "" {
		return fmt.Errorf("endpoint must not carry a query")
	}
	switch d.Provider {
	case OpenAIEmbeddingsProvider, OpenAIChatProvider, RerankProvider:
		if strings.TrimSpace(d.Model) == "" || d.Model != strings.TrimSpace(d.Model) {
			return fmt.Errorf("model must name the model the %s API serves", d.Provider)
		}
	}
	return d.validateConnection(endpoint, MaxRemoteTimeoutMs)
}
