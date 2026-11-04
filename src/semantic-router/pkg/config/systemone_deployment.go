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
}

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
		return fmt.Errorf("a systemone deployment takes endpoint, model, auth, timeout_ms and question_types; artifact, revision, external_model, device, engine arguments, replicas, served_name and public_name belong to other providers")
	}
	if d.Input.MaxTokens != 0 || (d.Input.Overflow != "" && d.Input.Overflow != "reject") {
		return fmt.Errorf("a systemone deployment takes no input budget: its API reads one bounded input")
	}
	endpoint, err := url.Parse(d.Endpoint)
	if err != nil || (endpoint.Scheme != "http" && endpoint.Scheme != "https") || endpoint.Host == "" || endpoint.Fragment != "" {
		return fmt.Errorf("endpoint must be the http:// or https:// URL the System One API answers POST requests on, such as https://api.example.com/v1/systemone")
	}
	if endpoint.User != nil {
		return fmt.Errorf("endpoint must not carry credentials; set auth.api_key_env or auth.api_key_file")
	}
	if strings.TrimSpace(d.Model) == "" || d.Model != strings.TrimSpace(d.Model) {
		return fmt.Errorf("model must name the model the API answers with, such as vllm-sr/d3-ultra")
	}
	if d.TimeoutMs < 0 || d.TimeoutMs > MaxDecisionTimeoutMs {
		return fmt.Errorf("timeout_ms must be within [1, %d] when set", MaxDecisionTimeoutMs)
	}
	seen := make(map[string]bool, len(d.QuestionTypes))
	for _, questionType := range d.QuestionTypes {
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
	return d.Auth.validate()
}

func (a *DeploymentAuth) validate() error {
	if a == nil {
		return nil
	}
	if a.Header != "" && !httpguts.ValidHeaderFieldName(a.Header) {
		return fmt.Errorf("auth.header must be an HTTP header name, such as Authorization")
	}
	if a.Prefix != nil && (*a.Prefix != strings.TrimSpace(*a.Prefix) || strings.ContainsAny(*a.Prefix, " \t\r\n")) {
		return fmt.Errorf("auth.prefix must be one word, such as Bearer, or empty to send the bare key")
	}
	if (a.APIKeyEnv == "") == (a.APIKeyFile == "") {
		return fmt.Errorf("auth takes exactly one of api_key_env and api_key_file; a key is never written inline")
	}
	if a.APIKeyFile != "" && !filepath.IsAbs(a.APIKeyFile) {
		return fmt.Errorf("auth.api_key_file must be an absolute path")
	}
	return nil
}
