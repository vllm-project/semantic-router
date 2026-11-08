package config

import (
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	"golang.org/x/net/http/httpguts"
)

// SystemOneProvider serves a deployment through a remote System One API. The
// Router asks it decision questions, as it asks a model_runtime deployment,
// and never routes a request to it.
const SystemOneProvider = "systemone"

// DeploymentAuth is the credential a remote deployment's API takes: a key
// read from an environment variable or a file, never written inline, sent in
// Header (default Authorization) after Prefix (default Bearer). Prefix is
// presence-aware: an authored empty prefix sends the bare key.
type DeploymentAuth struct {
	Header     string  `yaml:"header,omitempty" json:"header,omitempty"`
	Prefix     *string `yaml:"prefix,omitempty" json:"prefix,omitempty"`
	APIKeyEnv  string  `yaml:"api_key_env,omitempty" json:"api_key_env,omitempty"`
	APIKeyFile string  `yaml:"api_key_file,omitempty" json:"api_key_file,omitempty"`

	// inline is a key a deprecated field wrote into the configuration
	// (access_key, api_key); it is never read from or written to auth.
	inline string
}

// legacyInlineAuth is the credential of a deprecated inline key, sent as
// Authorization: Bearer <key> as the deprecated field was.
func legacyInlineAuth(key string) *DeploymentAuth {
	return &DeploymentAuth{inline: key}
}

// HasInlineKey reports whether the credential came from a deprecated inline
// key.
func (a *DeploymentAuth) HasInlineKey() bool { return a != nil && a.inline != "" }

// IsSystemOne reports whether a remote System One API serves the deployment.
func (d ModelDeployment) IsSystemOne() bool {
	return d.Provider == SystemOneProvider
}

// IsSystemOneDeployment reports whether name is a declared systemone deployment.
func (c *RouterConfig) IsSystemOneDeployment(name string) bool {
	if c == nil {
		return false
	}
	deployment, ok := c.ModelDeployments[name]
	return ok && deployment.IsSystemOne()
}

// CallTimeout is the deployment's own bound on a call, zero when it sets none.
func (d ModelDeployment) CallTimeout() time.Duration {
	return time.Duration(d.TimeoutMs) * time.Millisecond
}

// HeaderName is the header the credential goes in.
func (a *DeploymentAuth) HeaderName() string {
	if a == nil || a.Header == "" {
		return "Authorization"
	}
	return a.Header
}

// HeaderValue is the header value that carries key.
func (a *DeploymentAuth) HeaderValue(key string) string {
	prefix := "Bearer"
	if a != nil && a.Prefix != nil {
		prefix = *a.Prefix
	}
	if prefix == "" {
		return key
	}
	return prefix + " " + key
}

// Key reads the credential when a call needs it, so a rotated file or
// environment applies without another configuration.
func (a *DeploymentAuth) Key() (string, error) {
	if a == nil {
		return "", nil
	}
	if a.inline != "" {
		return a.inline, nil
	}
	if a.APIKeyFile != "" {
		data, err := os.ReadFile(a.APIKeyFile)
		if err != nil {
			return "", fmt.Errorf("auth.api_key_file %s cannot be read", a.APIKeyFile)
		}
		if key := strings.TrimSpace(string(data)); key != "" {
			return key, nil
		}
		return "", fmt.Errorf("auth.api_key_file %s is empty", a.APIKeyFile)
	}
	if key := strings.TrimSpace(os.Getenv(a.APIKeyEnv)); key != "" {
		return key, nil
	}
	return "", fmt.Errorf("auth.api_key_env %s is not set", a.APIKeyEnv)
}

func (d ModelDeployment) validateSystemOne() error {
	if d.Artifact != "" || d.Revision != "" || d.ExternalModel != "" || d.Device != "" || d.HasEngineArgs() ||
		len(d.Replicas) != 0 || d.ServedName != "" || d.PublicName != "" {
		return fmt.Errorf("a systemone deployment takes a connection block (endpoint, model, auth, headers, timeout_ms, max_retries, tls, limits) and capabilities; artifact, revision, external_model, device, engine arguments, replicas, served_name and public_name belong to model_runtime")
	}
	if d.Input.MaxTokens != 0 || (d.Input.Overflow != "" && d.Input.Overflow != "reject") {
		return fmt.Errorf("a systemone deployment takes no input budget: its API reads one bounded input")
	}
	if d.OperatingPoint != nil || d.Generation != nil {
		return fmt.Errorf("operating_point belongs to a managed model_runtime deployment and generation to an openai_chat one")
	}
	endpoint, err := url.Parse(d.Endpoint)
	if err != nil || (endpoint.Scheme != "http" && endpoint.Scheme != "https") || endpoint.Host == "" {
		return fmt.Errorf("endpoint must be the http:// or https:// URL the System One API answers POST requests on, such as https://api.example.com/v1/systemone")
	}
	if strings.TrimSpace(d.Model) == "" || d.Model != strings.TrimSpace(d.Model) {
		return fmt.Errorf("model must name the model the API answers with, such as vllm-sr/d3-ultra")
	}
	if len(d.QuestionTypes) > 0 && d.Capabilities != nil && len(d.Capabilities.QuestionTypes) > 0 {
		return fmt.Errorf("question_types is the deprecated spelling of capabilities.question_types; set one")
	}
	if d.Capabilities != nil && d.Capabilities.Dimensions != 0 {
		return fmt.Errorf("capabilities.dimensions applies to openai_embeddings deployments")
	}
	seen := make(map[string]bool)
	for _, questionType := range d.DeclaredQuestionTypes() {
		switch questionType {
		case DecisionQuestionChoice, DecisionQuestionNoul, DecisionQuestionScore, DecisionQuestionSet, DecisionQuestionSpan:
		default:
			return fmt.Errorf("question_types lists %q; the types are choice, noul, score, set and span", questionType)
		}
		if seen[questionType] {
			return fmt.Errorf("question_types lists %q twice", questionType)
		}
		seen[questionType] = true
	}
	presets := make(map[string]bool)
	for _, preset := range d.DeclaredPresets() {
		if strings.TrimSpace(preset) == "" || preset != strings.TrimSpace(preset) {
			return fmt.Errorf("capabilities.presets must name ready-made questions, such as pii and halu")
		}
		if presets[preset] {
			return fmt.Errorf("capabilities.presets lists %q twice", preset)
		}
		presets[preset] = true
	}
	return d.validateConnection(endpoint, MaxDecisionTimeoutMs)
}

func (a *DeploymentAuth) validate() error {
	if a == nil || a.inline != "" {
		return nil
	}
	if err := a.validateHeader(); err != nil {
		return err
	}
	if (a.APIKeyEnv == "") == (a.APIKeyFile == "") {
		return fmt.Errorf("auth takes exactly one of api_key_env and api_key_file; a key is never written inline")
	}
	if a.APIKeyFile != "" && !filepath.IsAbs(a.APIKeyFile) {
		return fmt.Errorf("auth.api_key_file must be an absolute path")
	}
	return nil
}

// validateHeader checks the header and prefix the key is sent in.
func (a *DeploymentAuth) validateHeader() error {
	if a.Header != "" && !httpguts.ValidHeaderFieldName(a.Header) {
		return fmt.Errorf("auth.header must be an HTTP header name, such as Authorization")
	}
	if a.Prefix != nil && (*a.Prefix != strings.TrimSpace(*a.Prefix) || strings.ContainsAny(*a.Prefix, " \t\r\n")) {
		return fmt.Errorf("auth.prefix must be one word, such as Bearer, or empty to send the bare key")
	}
	return nil
}
