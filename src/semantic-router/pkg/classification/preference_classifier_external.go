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

const defaultPreferenceSystemPrompt = "You are a routing classifier. Output ONLY a JSON object like {\"route\":\"...\"} with no extra text."

const defaultPreferenceUserPromptTemplate = `You are a helpful assistant designed to find the best suited route.
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

func newExternalPreferenceClassifier(
	externalCfg *config.ExternalModelConfig,
	rules []config.PreferenceRule,
) (*PreferenceClassifier, error) {
	if externalCfg == nil {
		return nil, fmt.Errorf("external model config is required when contrastive preference classifier is not enabled")
	}

	if externalCfg.ModelEndpoint.Address == "" {
		return nil, fmt.Errorf("external model endpoint address is required for preference")
	}
	if externalCfg.ModelName == "" {
		return nil, fmt.Errorf("external model name is required for preference")
	}

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

	return &PreferenceClassifier{
		client:             client,
		modelName:          externalCfg.ModelName,
		timeout:            timeout,
		preferenceRules:    rules,
		systemPrompt:       defaultPreferenceSystemPrompt,
		userPromptTemplate: defaultPreferenceUserPromptTemplate,
	}, nil
}

func (p *PreferenceClassifier) classifyExternal(conversationJSON string) (*PreferenceResult, error) {
	ctx, cancel := context.WithTimeout(context.Background(), p.timeout)
	defer cancel()

	start := time.Now()
	routesJSON, err := p.buildRoutesJSON()
	if err != nil {
		return nil, fmt.Errorf("failed to build routes JSON: %w", err)
	}

	userPrompt := fmt.Sprintf(p.userPromptTemplate, routesJSON, conversationJSON)
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

func (p *PreferenceClassifier) buildRoutesJSON() (string, error) {
	type route struct {
		Name        string `json:"name"`
		Description string `json:"description"`
	}

	routes := make([]route, 0, len(p.preferenceRules))
	for _, rule := range p.preferenceRules {
		routes = append(routes, route{
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

func (p *PreferenceClassifier) parsePreferenceOutput(output string) (*PreferenceResult, error) {
	output = strings.TrimSpace(output)

	start := strings.Index(output, "{")
	end := strings.LastIndex(output, "}")
	if start == -1 || end == -1 || start >= end {
		return nil, fmt.Errorf("no valid JSON found in output")
	}

	jsonStr := output[start : end+1]
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
