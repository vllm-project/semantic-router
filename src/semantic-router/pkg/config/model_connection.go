package config

import (
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"math"
	"net/url"
	"os"
	"path/filepath"
	"slices"
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
	// OpenAIChatProvider asks an OpenAI-compatible chat model, such as the
	// memory query rewrite: POST <endpoint>/chat/completions.
	OpenAIChatProvider = "openai_chat"
)

// MaxRemoteTimeoutMs bounds timeout_ms on a remote chat or embedding
// deployment.
const MaxRemoteTimeoutMs = 300000

// MaxRemoteRetries bounds max_retries on any connection block.
const MaxRemoteRetries = 10

// IsRemoteProvider reports whether a provider serves a deployment over a
// remote API rather than the model runtime.
func IsRemoteProvider(provider string) bool {
	switch provider {
	case SystemOneProvider, OpenAIEmbeddingsProvider, OpenAIChatProvider:
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
// does not list it: question types, presets and the input modalities a
// decisions request may carry (systemone), and the vector dimensions of an
// embedding model.
type DeploymentCapabilities struct {
	QuestionTypes []string `yaml:"question_types,omitempty" json:"question_types,omitempty"`
	Presets       []string `yaml:"presets,omitempty" json:"presets,omitempty"`
	Modalities    []string `yaml:"modalities,omitempty" json:"modalities,omitempty"`
	Dimensions    int      `yaml:"dimensions,omitempty" json:"dimensions,omitempty"`
}

// DeclaredQuestionTypes are the question types a deployment declares.
func (d ModelDeployment) DeclaredQuestionTypes() []string {
	if d.Capabilities == nil {
		return nil
	}
	return slices.Clone(d.Capabilities.QuestionTypes)
}

// DeclaredPresets are the presets a deployment declares.
func (d ModelDeployment) DeclaredPresets() []string {
	if d.Capabilities == nil {
		return nil
	}
	return slices.Clone(d.Capabilities.Presets)
}

// DeclaredModalities are the input modalities a deployment declares.
func (d ModelDeployment) DeclaredModalities() []string {
	if d.Capabilities == nil {
		return nil
	}
	return slices.Clone(d.Capabilities.Modalities)
}

// The input modalities an embedding deployment may declare it embeds.
var embeddingModalities = []string{"text", "image", "audio"}

// validateEmbeddingModalities checks the modalities an embedding deployment
// declares: known ones, each once.
func (d ModelDeployment) validateEmbeddingModalities() error {
	seen := map[string]bool{}
	for _, modality := range d.DeclaredModalities() {
		switch {
		case !slices.Contains(embeddingModalities, modality):
			return fmt.Errorf("capabilities.modalities lists %q; an embedding model embeds text, image or audio", modality)
		case seen[modality]:
			return fmt.Errorf("capabilities.modalities lists %q twice", modality)
		}
		seen[modality] = true
	}
	return nil
}

// EmbedsText reports whether an embedding deployment embeds text, which
// recall embeds: it declares no modalities, or text among them.
func (d ModelDeployment) EmbedsText() bool {
	declared := d.DeclaredModalities()
	return len(declared) == 0 || slices.Contains(declared, "text")
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
	if err := validateForwardRequestHeaders(d.ForwardRequestHeaders, d.Auth.HeaderName()); err != nil {
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
	Reasoning *GenerationReasoning `yaml:"reasoning,omitempty" json:"reasoning,omitempty"`
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

// validateRemote checks an openai_chat or openai_embeddings deployment: a
// connection block and nothing the model runtime owns.
func (d ModelDeployment) validateRemote() error {
	if d.Artifact != "" || d.Revision != "" || d.Device != "" || d.HasEngineArgs() || len(d.Replicas) != 0 || d.OutlierDetection != nil || d.PublicName != "" {
		return fmt.Errorf("a %s deployment takes a connection block (endpoint, model, auth, headers, timeout_ms, max_retries, tls, limits); artifact, revision, device, engine arguments, replicas, outlier_detection and public_name belong to model_runtime", d.Provider)
	}
	if d.Input.MaxTokens != 0 || (d.Input.Overflow != "" && d.Input.Overflow != "reject") {
		return fmt.Errorf("a %s deployment takes no input budget: the remote model reads its own input", d.Provider)
	}
	if d.Generation != nil {
		if d.Provider != OpenAIChatProvider {
			return fmt.Errorf("generation applies to openai_chat deployments")
		}
		if err := d.Generation.validate(); err != nil {
			return err
		}
	}
	if d.Capabilities != nil && (len(d.Capabilities.QuestionTypes) > 0 || len(d.Capabilities.Presets) > 0) {
		return fmt.Errorf("question types and presets apply to systemone deployments")
	}
	if d.Capabilities != nil && d.Provider != OpenAIEmbeddingsProvider && (d.Capabilities.Dimensions != 0 || len(d.Capabilities.Modalities) > 0) {
		return fmt.Errorf("capabilities.dimensions and capabilities.modalities describe an embedding model; they apply to openai_embeddings deployments")
	}
	if d.Capabilities != nil && d.Capabilities.Dimensions < 0 {
		return fmt.Errorf("capabilities.dimensions must be positive")
	}
	if err := d.validateEmbeddingModalities(); err != nil {
		return err
	}
	endpoint, err := url.Parse(strings.TrimSpace(d.Endpoint))
	if err != nil || (endpoint.Scheme != "http" && endpoint.Scheme != "https") || endpoint.Host == "" {
		return fmt.Errorf("endpoint must be the http:// or https:// base URL of the %s API, such as https://models.example.com/v1", d.Provider)
	}
	if endpoint.RawQuery != "" {
		return fmt.Errorf("endpoint must not carry a query")
	}
	switch d.Provider {
	case OpenAIEmbeddingsProvider, OpenAIChatProvider:
		if strings.TrimSpace(d.Model) == "" || d.Model != strings.TrimSpace(d.Model) {
			return fmt.Errorf("model must name the model the %s API serves", d.Provider)
		}
	}
	return d.validateConnection(endpoint, MaxRemoteTimeoutMs)
}
