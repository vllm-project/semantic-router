package classification

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	candle "github.com/vllm-project/semantic-router/candle-binding"
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

	useLocal      bool
	localModelID  string
	localUseCPU   bool
	localThresh   float32
	localInitOnce sync.Once
	localInitErr  error
	useQwen       bool
}

// NewPreferenceClassifier creates a new preference classifier
func NewPreferenceClassifier(externalCfg *config.ExternalModelConfig, rules []config.PreferenceRule, localCfg *config.PreferenceModelConfig) (*PreferenceClassifier, error) {
	// Prefer local (Candle) when configured
	if localCfg != nil && localCfg.ModelID != "" {
		return &PreferenceClassifier{
			preferenceRules: rules,
			useLocal:        true,
			useQwen:         localCfg.UseQwen3,
			localModelID:    localCfg.ModelID,
			localUseCPU:     localCfg.UseCPU,
			localThresh:     localCfg.Threshold,
		}, nil
	}

	if externalCfg == nil {
		return nil, fmt.Errorf("external model config is required when local preference model is not set")
	}

	if externalCfg.ModelEndpoint.Address == "" {
		return nil, fmt.Errorf("external model endpoint address is required for preference")
	}
	if externalCfg.ModelName == "" {
		return nil, fmt.Errorf("external model name is required for preference")
	}

	// Create client with or without access key
	var client *VLLMClient
	if externalCfg.AccessKey != "" {
		client = NewVLLMClientWithAuth(&externalCfg.ModelEndpoint, externalCfg.AccessKey)
	} else {
		client = NewVLLMClient(&externalCfg.ModelEndpoint)
	}

	timeout := 30 * time.Second
	if externalCfg.TimeoutSeconds > 0 {
		timeout = time.Duration(externalCfg.TimeoutSeconds) * time.Second
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
		modelName:          externalCfg.ModelName,
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

	if p.useLocal {
		result, err := p.classifyLocal(conversationJSON)
		if err != nil {
			return nil, err
		}
		logging.Infof("Preference classification: preference=%s, latency=%.3fs",
			result.Preference, time.Since(start).Seconds())
		return result, nil
	}

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

// classifyLocal runs preference classification using local Candle Qwen3
func (p *PreferenceClassifier) classifyLocal(conversationJSON string) (*PreferenceResult, error) {
	// Initialize model once
	p.localInitOnce.Do(func() {
		if p.useQwen {
			p.localInitErr = candle.InitQwen3PreferenceClassifier(p.localModelID, p.localUseCPU)
		} else {
			p.localInitErr = fmt.Errorf("qwen3 preference required but useQwen flag is false")
		}
	})

	if p.localInitErr != nil {
		return nil, fmt.Errorf("failed to initialize local preference model: %w", p.localInitErr)
	}

	labels := make([]string, 0, len(p.preferenceRules))
	for _, rule := range p.preferenceRules {
		labels = append(labels, rule.Name)
	}

	result, err := candle.ClassifyQwen3Preference(conversationJSON, labels)
	if err != nil {
		return nil, fmt.Errorf("local preference classification failed: %w", err)
	}

	if result.Class < 0 || result.Class >= len(p.preferenceRules) {
		return nil, fmt.Errorf("predicted class %d out of range for %d preference rules", result.Class, len(p.preferenceRules))
	}

	// Apply optional confidence threshold
	conf := result.Confidence
	if p.localThresh > 0 && conf < p.localThresh {
		return nil, fmt.Errorf("preference confidence %.3f below threshold %.3f", conf, p.localThresh)
	}

	matchedRule := p.preferenceRules[result.Class]

	return &PreferenceResult{
		Preference: matchedRule.Name,
		Confidence: conf,
	}, nil
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
	if p == nil {
		return false
	}

	if p.useLocal {
		// Initialization occurs lazily; presence of struct is enough
		return p.localModelID != ""
	}

	return p.client != nil
}
