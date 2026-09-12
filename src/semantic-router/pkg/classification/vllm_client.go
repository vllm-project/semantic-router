package classification

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/shared"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/connector"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// VLLMClient handles communication with vLLM REST API for classifiers
type VLLMClient struct {
	connector        *connector.Client
	initErr          error
	endpoint         *config.ClassifierVLLMEndpoint
	timeout          time.Duration
	transportOptions connector.Options
	accessKey        string
}

// NewVLLMClient creates a new vLLM REST API client for classifiers
func NewVLLMClient(endpoint *config.ClassifierVLLMEndpoint) *VLLMClient {
	return newVLLMClient(endpoint, "", config.DefaultClassifyMaxResponseBytes, time.Duration(config.DefaultClassifierTimeoutSeconds)*time.Second)
}

// NewVLLMClientWithAuth creates a new vLLM REST API client with access key
func NewVLLMClientWithAuth(endpoint *config.ClassifierVLLMEndpoint, accessKey string) *VLLMClient {
	return newVLLMClient(endpoint, accessKey, config.DefaultClassifyMaxResponseBytes, time.Duration(config.DefaultClassifierTimeoutSeconds)*time.Second)
}

func newVLLMClientFromConfig(cfg *config.ExternalModelConfig) *VLLMClient {
	return newVLLMClientWithLimits(&cfg.ModelEndpoint, cfg.AccessKey, cfg.GetMaxRequestBytes(), cfg.GetMaxResponseBytes(), cfg.GetTimeout())
}

func newVLLMClient(endpoint *config.ClassifierVLLMEndpoint, accessKey string, maxResponseBytes int64, timeout time.Duration) *VLLMClient {
	return newVLLMClientWithLimits(endpoint, accessKey, config.DefaultClassifyMaxRequestBytes, maxResponseBytes, timeout)
}

func newVLLMClientWithLimits(endpoint *config.ClassifierVLLMEndpoint, accessKey string, maxRequestBytes, maxResponseBytes int64, timeout time.Duration) *VLLMClient {
	if endpoint == nil {
		return &VLLMClient{initErr: fmt.Errorf("classifier endpoint is required")}
	}
	endpointCopy := *endpoint
	scheme := endpointCopy.Protocol
	if scheme == "" {
		scheme = "http"
	}
	baseURL := fmt.Sprintf("%s://%s:%d", scheme, endpointCopy.Address, endpointCopy.Port)
	options := connector.Options{AttemptTimeout: timeout, MaxRequestBytes: maxRequestBytes, MaxResponseBytes: maxResponseBytes, MaxErrorBytes: maxClassifyErrorBodyBytes}
	transport, err := connector.New(baseURL, bearerAuthorizer(accessKey), options)
	return &VLLMClient{connector: transport, initErr: err, endpoint: &endpointCopy, timeout: timeout, transportOptions: options, accessKey: accessKey}
}

// vllmChatCompletionRequest extends openai.ChatCompletionNewParams with
// the vLLM-specific extra_body field for guided decoding, LoRA adapters, etc.
type vllmChatCompletionRequest struct {
	Model          string                                             `json:"model"`
	Messages       []openai.ChatCompletionMessageParamUnion           `json:"messages"`
	MaxTokens      int                                                `json:"max_tokens,omitempty"`
	Temperature    float64                                            `json:"temperature,omitempty"`
	Stream         bool                                               `json:"stream,omitempty"`
	ExtraBody      map[string]interface{}                             `json:"extra_body,omitempty"`
	ResponseFormat *openai.ChatCompletionNewParamsResponseFormatUnion `json:"response_format,omitempty"`
}

// GenerationOptions contains options for vLLM generation
type GenerationOptions struct {
	MaxTokens   int
	Temperature float64
	Stream      bool
	ExtraBody   map[string]interface{}
	JSONMode    bool
}

func (c *VLLMClient) buildMessages(prompt string) []openai.ChatCompletionMessageParamUnion {
	if c.endpoint.UseChatTemplate {
		return []openai.ChatCompletionMessageParamUnion{
			{OfSystem: &openai.ChatCompletionSystemMessageParam{
				Content: openai.ChatCompletionSystemMessageParamContentUnion{
					OfString: openai.String("You are a safety classifier."),
				},
			}},
			{OfUser: &openai.ChatCompletionUserMessageParam{
				Content: openai.ChatCompletionUserMessageParamContentUnion{
					OfString: openai.String(prompt),
				},
			}},
		}
	}

	content := prompt
	if c.endpoint.PromptTemplate != "" {
		content = fmt.Sprintf(c.endpoint.PromptTemplate, prompt)
	}
	return []openai.ChatCompletionMessageParamUnion{
		{OfUser: &openai.ChatCompletionUserMessageParam{
			Content: openai.ChatCompletionUserMessageParamContentUnion{
				OfString: openai.String(content),
			},
		}},
	}
}

// Generate sends a chat completion request to vLLM
func (c *VLLMClient) Generate(ctx context.Context, modelName string, prompt string, options *GenerationOptions) (*openai.ChatCompletion, error) {
	return c.generateWithMessages(ctx, modelName, c.buildMessages(prompt), options)
}

// GenerateWithSystemPrompt sends an explicit system/user pair. It is used by
// generic classifiers whose policy prompt must not inherit a domain-specific
// built-in system message.
func (c *VLLMClient) GenerateWithSystemPrompt(
	ctx context.Context,
	modelName string,
	systemPrompt string,
	userContent string,
	options *GenerationOptions,
) (*openai.ChatCompletion, error) {
	return c.generateWithMessages(
		ctx,
		modelName,
		[]openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage(systemPrompt),
			openai.UserMessage(userContent),
		},
		options,
	)
}

func (c *VLLMClient) generateWithMessages(
	ctx context.Context,
	modelName string,
	messages []openai.ChatCompletionMessageParamUnion,
	options *GenerationOptions,
) (*openai.ChatCompletion, error) {
	req := vllmChatCompletionRequest{
		Model:    modelName,
		Messages: messages,
	}

	if options != nil {
		req.MaxTokens = options.MaxTokens
		req.Temperature = options.Temperature
		req.Stream = options.Stream
		req.ExtraBody = options.ExtraBody
		if options.JSONMode {
			jsonObjectFormat := shared.NewResponseFormatJSONObjectParam()
			responseFormat := openai.ChatCompletionNewParamsResponseFormatUnion{
				OfJSONObject: &jsonObjectFormat,
			}
			req.ResponseFormat = &responseFormat
		}
	}

	if req.MaxTokens == 0 {
		req.MaxTokens = 512
	}

	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	if c.initErr != nil {
		return nil, c.initErr
	}
	body, err := c.connector.Do(ctx, connector.Operation{Name: "http_chat", Method: http.MethodPost, Path: "/v1/chat/completions"}, jsonData)
	if err != nil {
		var remoteErr *connector.Error
		if errors.As(err, &remoteErr) && remoteErr.StatusCode != 0 {
			response, truncated := remoteErr.ResponseBody()
			return nil, fmt.Errorf("vLLM API returned status %d (response body %d bytes, truncated=%t, not logged): %w", remoteErr.StatusCode, len(response), truncated, err)
		}
		return nil, fmt.Errorf("vLLM request failed: %w", err)
	}

	var chatResp openai.ChatCompletion
	if err := json.Unmarshal(body, &chatResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	logging.Debugf("vLLM API call successful: model=%s, choices=%d", modelName, len(chatResp.Choices))

	return &chatResp, nil
}

// Close releases only this connector's idle connections. Active requests are
// protected by the owning typed binding's close/drain lifecycle.
func (c *VLLMClient) Close() error {
	if c != nil && c.connector != nil {
		return c.connector.Close()
	}
	return nil
}
