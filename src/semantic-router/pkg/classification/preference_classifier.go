package classification

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// PreferenceResult represents the result of preference classification
type PreferenceResult struct {
	Preference string  `json:"route"` // The matched route name
	Confidence float32 `json:"confidence,omitempty"`
}

// PreferenceClassifier handles route preference matching via external LLM
type PreferenceClassifier struct {
	client             *VLLMClient
	modelName          string
	timeout            time.Duration
	preferenceRules    []config.PreferenceRule
	systemPrompt       string
	userPromptTemplate string
}

// NewPreferenceClassifier creates a new preference classifier
func NewPreferenceClassifier(cfg *config.ExternalModelConfig, rules []config.PreferenceRule) (*PreferenceClassifier, error) {
	if cfg.ModelEndpoint.Address == "" {
		return nil, fmt.Errorf("external model endpoint address is required for preference")
	}
	if cfg.ModelName == "" {
		return nil, fmt.Errorf("external model name is required for preference")
	}

	// Create client with or without access key
	var client *VLLMClient
	if cfg.AccessKey != "" {
		client = NewVLLMClientWithAuth(&cfg.ModelEndpoint, cfg.AccessKey)
	} else {
		client = NewVLLMClient(&cfg.ModelEndpoint)
	}

	timeout := 30 * time.Second
	if cfg.TimeoutSeconds > 0 {
		timeout = time.Duration(cfg.TimeoutSeconds) * time.Second
	}

	// Default prompts
	systemPrompt := "You are a routing classifier. Output ONLY a JSON object like {\"route\":\"...\"} with no extra text."

	userPromptTemplate := `You are a helpful assistant designed to find the best suited route.
You are provided with route description within <routes></routes> XML tags:
<routes>
%s
</routes>

<conversation>
%s
</conversation>

Your task is to decide which route is best suit with user intent on the conversation in <conversation></conversation> XML tags. Follow the instruction:
1. If the latest intent from user is irrelevant or user intent is full filled, response with other route {"route": "other"}.
2. You must analyze the route descriptions and find the best match route for user latest intent.
3. You only response the name of the route that best matches the user's request, use the exact name in the <routes></routes>.
Return ONLY the JSON in the exact format:
{"route":"route_name"}`

	return &PreferenceClassifier{
		client:             client,
		modelName:          cfg.ModelName,
		timeout:            timeout,
		preferenceRules:    rules,
		systemPrompt:       systemPrompt,
		userPromptTemplate: userPromptTemplate,
	}, nil
}

// Classify determines the best route preference for the given conversation
func (p *PreferenceClassifier) Classify(conversationJSON string) (*PreferenceResult, error) {
	ctx, cancel := context.WithTimeout(context.Background(), p.timeout)
	defer cancel()

	start := time.Now()

	// Build routes JSON
	routesJSON, err := p.buildRoutesJSON()
	if err != nil {
		return nil, fmt.Errorf("failed to build routes JSON: %w", err)
	}

	// Build user prompt
	userPrompt := fmt.Sprintf(p.userPromptTemplate, routesJSON, conversationJSON)

	// Call external LLM with chat format
	resp, err := p.client.Generate(ctx, p.modelName, userPrompt, &GenerationOptions{
		MaxTokens:   1000,
		Temperature: 0.0,
	})
	if err != nil {
		return nil, fmt.Errorf("external LLM API call failed: %w", err)
	}

	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("no choices in LLM response")
	}

	// Parse JSON response
	output := resp.Choices[0].Message.Content
	logging.Infof("Preference classification response: %s", output)

	result, err := p.parsePreferenceOutput(output)
	if err != nil {
		return nil, fmt.Errorf("failed to parse preference output: %w", err)
	}

	logging.Infof("Preference classification: preference=%s, latency=%.3fs",
		result.Preference, time.Since(start).Seconds())

	return result, nil
}

// buildRoutesJSON builds the routes JSON array from preference rules
func (p *PreferenceClassifier) buildRoutesJSON() (string, error) {
	type Route struct {
		Name        string `json:"name"`
		Description string `json:"description"`
	}

	routes := make([]Route, 0, len(p.preferenceRules))
	for _, rule := range p.preferenceRules {
		routes = append(routes, Route{
			Name:        rule.Name,
			Description: rule.Description,
		})
	}

	data, err := json.Marshal(routes)
	if err != nil {
		return "", err
	}

	return string(data), nil
}

// parsePreferenceOutput parses the JSON output from LLM
func (p *PreferenceClassifier) parsePreferenceOutput(output string) (*PreferenceResult, error) {
	// Try to extract JSON from output
	output = strings.TrimSpace(output)

	// Find JSON object
	start := strings.Index(output, "{")
	end := strings.LastIndex(output, "}")
	if start == -1 || end == -1 || start >= end {
		return nil, fmt.Errorf("no valid JSON found in output")
	}

	jsonStr := output[start : end+1]

	// Replace single quotes with double quotes for JSON compatibility
	// Some LLMs return {'key': 'value'} instead of {"key": "value"}
	jsonStr = strings.ReplaceAll(jsonStr, "'", "\"")

	var result PreferenceResult
	if err := json.Unmarshal([]byte(jsonStr), &result); err != nil {
		return nil, fmt.Errorf("failed to parse JSON: %w", err)
	}

	if result.Preference == "" {
		return nil, fmt.Errorf("preference field is empty")
	}

	return &result, nil
}

// IsInitialized returns true if the classifier is initialized
func (p *PreferenceClassifier) IsInitialized() bool {
	return p != nil && p.client != nil
}
