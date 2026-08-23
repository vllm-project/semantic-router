package config

import (
	"fmt"
	"net/url"
	"strings"
)

// AgentServiceConfig binds Router-native Agent model calls to the ordinary
// public inference front door. It never identifies a physical backend.
type AgentServiceConfig struct {
	PublicInferenceEndpoint string `yaml:"public_inference_endpoint,omitempty"`
}

func validateAgentService(mode string, service AgentServiceConfig) error {
	endpoint := service.PublicInferenceEndpoint
	if mode == ControlPlaneModeStandalone {
		if endpoint != "" {
			return fmt.Errorf("global.services.agent is managed-only")
		}
		return nil
	}
	if endpoint == "" {
		return fmt.Errorf("managed mode requires global.services.agent.public_inference_endpoint")
	}
	if endpoint != strings.TrimSpace(endpoint) {
		return fmt.Errorf("global.services.agent.public_inference_endpoint must not contain surrounding whitespace")
	}
	parsed, err := url.Parse(endpoint)
	if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" ||
		parsed.Opaque != "" || parsed.User != nil || parsed.RawQuery != "" || parsed.ForceQuery ||
		parsed.Fragment != "" || parsed.EscapedPath() != "/v1/chat/completions" {
		return fmt.Errorf("global.services.agent.public_inference_endpoint must be an HTTP(S) /v1/chat/completions URL")
	}
	return nil
}
