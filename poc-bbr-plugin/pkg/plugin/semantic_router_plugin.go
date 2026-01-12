/*
Package plugin provides the SemanticRouterPlugin which implements the BBRPlugin interface
from the Gateway API Inference Extension (GIE) project.

This plugin combines three vSR capabilities into a single BBR plugin:
1. Category/Intent Classification
2. PII Detection
3. Jailbreak Detection

Headers Output:
- X-Gateway-Intent-Category: The classified category (coding, math, creative, etc.)
- X-Gateway-Intent-Confidence: Classification confidence (0.0-1.0)
- X-Gateway-PII-Detected: Whether PII was detected (true/false)
- X-Gateway-PII-Types: Comma-separated list of detected PII types
- X-Gateway-PII-Blocked: Whether request was blocked due to PII
- X-Gateway-Security-Threat: Type of security threat detected
- X-Gateway-Security-Blocked: Whether request was blocked due to jailbreak
- X-Gateway-Security-Confidence: Jailbreak detection confidence
*/
package plugin

import (
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/poc-bbr-plugin/pkg/classifier"
	bbrplugins "sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
)

const (
	// Plugin identification
	PluginType = "Guardrail"
	PluginName = "vsr-semantic-router"

	// Header names for classifier results
	HeaderIntentCategory   = "X-Gateway-Intent-Category"
	HeaderIntentConfidence = "X-Gateway-Intent-Confidence"
	HeaderIntentLatencyMs  = "X-Gateway-Intent-Latency-Ms"

	// Header names for PII detection results
	HeaderPIIDetected  = "X-Gateway-PII-Detected"
	HeaderPIITypes     = "X-Gateway-PII-Types"
	HeaderPIIBlocked   = "X-Gateway-PII-Blocked"
	HeaderPIILatencyMs = "X-Gateway-PII-Latency-Ms"

	// Header names for jailbreak detection results
	HeaderSecurityThreat     = "X-Gateway-Security-Threat"
	HeaderSecurityBlocked    = "X-Gateway-Security-Blocked"
	HeaderSecurityConfidence = "X-Gateway-Security-Confidence"
	HeaderSecurityLatencyMs  = "X-Gateway-Security-Latency-Ms"

	// Overall plugin header
	HeaderPluginLatencyMs = "X-Gateway-Semantic-Router-Latency-Ms"
)

// SemanticRouterPlugin implements BBRPlugin interface and combines
// classifier, PII detection, and jailbreak detection into a single plugin.
type SemanticRouterPlugin struct {
	typedName           plugins.TypedName
	requiresFullParsing bool

	// Configuration
	config Config

	// Classifiers (can be mock or real vSR implementations)
	// Uses interface to support both MockSemanticRouterClassifier and CandleSemanticRouterClassifier
	classifier classifier.SemanticRouterClassifier

	// Statistics
	stats Stats
	mu    sync.RWMutex
}

// Config holds the plugin configuration
type Config struct {
	// Enable/disable individual features
	ClassifierEnabled bool `json:"classifier_enabled"`
	PIIEnabled        bool `json:"pii_enabled"`
	JailbreakEnabled  bool `json:"jailbreak_enabled"`

	// Thresholds
	ClassifierThreshold float32 `json:"classifier_threshold"`
	PIIThreshold        float32 `json:"pii_threshold"`
	JailbreakThreshold  float32 `json:"jailbreak_threshold"`

	// Blocking behavior
	BlockOnPII       bool `json:"block_on_pii"`
	BlockOnJailbreak bool `json:"block_on_jailbreak"`
}

// Stats holds plugin execution statistics
type Stats struct {
	TotalRequests      int64 `json:"total_requests"`
	ClassifiedRequests int64 `json:"classified_requests"`
	PIIDetected        int64 `json:"pii_detected"`
	JailbreakDetected  int64 `json:"jailbreak_detected"`
	BlockedRequests    int64 `json:"blocked_requests"`
	TotalLatencyMs     int64 `json:"total_latency_ms"`
}

// DefaultConfig returns the default plugin configuration
func DefaultConfig() Config {
	return Config{
		ClassifierEnabled:   true,
		PIIEnabled:          true,
		JailbreakEnabled:    true,
		ClassifierThreshold: 0.7,
		PIIThreshold:        0.8,
		JailbreakThreshold:  0.9,
		BlockOnPII:          true,
		BlockOnJailbreak:    true,
	}
}

// NewSemanticRouterPlugin creates a new SemanticRouterPlugin with default configuration
// Uses mock classifier by default (for testing and non-Linux platforms)
func NewSemanticRouterPlugin() bbrplugins.BBRPlugin {
	return NewSemanticRouterPluginWithConfig(DefaultConfig())
}

// NewSemanticRouterPluginWithConfig creates a new SemanticRouterPlugin with custom configuration
// Uses mock classifier (for testing and non-Linux platforms)
func NewSemanticRouterPluginWithConfig(config Config) *SemanticRouterPlugin {
	mockClassifier := classifier.NewMockSemanticRouterClassifier()
	mockClassifier.SetClassifierEnabled(config.ClassifierEnabled)
	mockClassifier.SetPIIEnabled(config.PIIEnabled)
	mockClassifier.SetJailbreakEnabled(config.JailbreakEnabled)

	return &SemanticRouterPlugin{
		typedName: plugins.TypedName{
			Type: PluginType,
			Name: PluginName,
		},
		requiresFullParsing: true, // Need to parse body for user content
		config:              config,
		classifier:          mockClassifier,
	}
}

// NewSemanticRouterPluginWithClassifier creates a new SemanticRouterPlugin with a custom classifier
// Use this to inject the real CandleSemanticRouterClassifier for production
func NewSemanticRouterPluginWithClassifier(config Config, clf classifier.SemanticRouterClassifier) *SemanticRouterPlugin {
	return &SemanticRouterPlugin{
		typedName: plugins.TypedName{
			Type: PluginType,
			Name: PluginName,
		},
		requiresFullParsing: true,
		config:              config,
		classifier:          clf,
	}
}

// TypedName returns the plugin type and name (implements plugins.Plugin)
func (p *SemanticRouterPlugin) TypedName() plugins.TypedName {
	return p.typedName
}

// RequiresFullParsing indicates we need full body parsing (implements BBRPlugin)
func (p *SemanticRouterPlugin) RequiresFullParsing() bool {
	return p.requiresFullParsing
}

// Execute runs all enabled classifiers on the request body (implements BBRPlugin)
func (p *SemanticRouterPlugin) Execute(requestBodyBytes []byte) (
	headers map[string]string,
	mutatedBodyBytes []byte,
	err error,
) {
	start := time.Now()
	headers = make(map[string]string)
	mutatedBodyBytes = requestBodyBytes // We don't mutate the body

	// Update stats
	p.mu.Lock()
	p.stats.TotalRequests++
	p.mu.Unlock()

	// Extract user content from request body
	userContent, err := p.extractUserContent(requestBodyBytes)
	if err != nil {
		// Return original body on parse failure, but don't fail the request
		return headers, requestBodyBytes, nil
	}

	if userContent == "" {
		// No user content to analyze
		return headers, requestBodyBytes, nil
	}

	// Run all classifiers
	categoryResult, piiResult, jailbreakResult, err := p.classifier.ProcessAll(userContent)
	if err != nil {
		// Log error but don't fail the request
		return headers, requestBodyBytes, nil
	}

	// Set classifier headers
	if p.config.ClassifierEnabled {
		headers[HeaderIntentCategory] = categoryResult.Category
		headers[HeaderIntentConfidence] = fmt.Sprintf("%.4f", categoryResult.Confidence)
		headers[HeaderIntentLatencyMs] = fmt.Sprintf("%d", categoryResult.LatencyMs)

		p.mu.Lock()
		p.stats.ClassifiedRequests++
		p.mu.Unlock()
	}

	// Set PII headers
	if p.config.PIIEnabled {
		headers[HeaderPIIDetected] = fmt.Sprintf("%t", piiResult.HasPII)
		if piiResult.HasPII {
			headers[HeaderPIITypes] = strings.Join(piiResult.PIITypes, ",")
			headers[HeaderPIIBlocked] = fmt.Sprintf("%t", p.config.BlockOnPII)

			p.mu.Lock()
			p.stats.PIIDetected++
			if p.config.BlockOnPII {
				p.stats.BlockedRequests++
			}
			p.mu.Unlock()
		}
		headers[HeaderPIILatencyMs] = fmt.Sprintf("%d", piiResult.LatencyMs)
	}

	// Set jailbreak headers
	if p.config.JailbreakEnabled {
		if jailbreakResult.IsJailbreak {
			headers[HeaderSecurityThreat] = jailbreakResult.ThreatType
			headers[HeaderSecurityConfidence] = fmt.Sprintf("%.4f", jailbreakResult.Confidence)
			headers[HeaderSecurityBlocked] = fmt.Sprintf("%t", p.config.BlockOnJailbreak)

			p.mu.Lock()
			p.stats.JailbreakDetected++
			if p.config.BlockOnJailbreak {
				p.stats.BlockedRequests++
			}
			p.mu.Unlock()
		}
		headers[HeaderSecurityLatencyMs] = fmt.Sprintf("%d", jailbreakResult.LatencyMs)
	}

	// Set overall latency header
	totalLatency := time.Since(start).Milliseconds()
	headers[HeaderPluginLatencyMs] = fmt.Sprintf("%d", totalLatency)

	p.mu.Lock()
	p.stats.TotalLatencyMs += totalLatency
	p.mu.Unlock()

	// If blocked, return an error (optional - depends on desired behavior)
	if (p.config.BlockOnPII && piiResult.HasPII) ||
		(p.config.BlockOnJailbreak && jailbreakResult.IsJailbreak) {
		var blockReason string
		if piiResult.HasPII {
			blockReason = fmt.Sprintf("PII detected: %s", strings.Join(piiResult.PIITypes, ", "))
		}
		if jailbreakResult.IsJailbreak {
			if blockReason != "" {
				blockReason += "; "
			}
			blockReason += fmt.Sprintf("Jailbreak detected: %s", jailbreakResult.ThreatType)
		}
		return headers, requestBodyBytes, fmt.Errorf("request blocked: %s", blockReason)
	}

	return headers, requestBodyBytes, nil
}

// extractUserContent extracts user message content from the OpenAI-compatible request body
func (p *SemanticRouterPlugin) extractUserContent(bodyBytes []byte) (string, error) {
	var request struct {
		Messages []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"messages"`
	}

	if err := json.Unmarshal(bodyBytes, &request); err != nil {
		return "", fmt.Errorf("failed to parse request body: %w", err)
	}

	// Extract all user messages
	var userContent strings.Builder
	for _, msg := range request.Messages {
		if msg.Role == "user" {
			if userContent.Len() > 0 {
				userContent.WriteString(" ")
			}
			userContent.WriteString(msg.Content)
		}
	}

	return userContent.String(), nil
}

// GetStats returns the current plugin statistics
func (p *SemanticRouterPlugin) GetStats() Stats {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.stats
}

// GetConfig returns the current plugin configuration
func (p *SemanticRouterPlugin) GetConfig() Config {
	return p.config
}

// String returns a string representation of the plugin
func (p *SemanticRouterPlugin) String() string {
	return fmt.Sprintf("SemanticRouterPlugin{%s, classifier=%t, pii=%t, jailbreak=%t}",
		p.typedName.String(),
		p.config.ClassifierEnabled,
		p.config.PIIEnabled,
		p.config.JailbreakEnabled,
	)
}
