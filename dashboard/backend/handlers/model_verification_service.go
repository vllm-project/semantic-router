package handlers

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const (
	modelVerificationTimeout          = 30 * time.Second
	modelVerificationMaxResponseBytes = 64 << 10
	modelVerificationMaxSummaryRunes  = 240
	modelVerificationMaxTokens        = 32
	modelVerificationPrompt           = "Reply only with OK."
)

type modelVerificationHTTPClient interface {
	Do(*http.Request) (*http.Response, error)
}

type modelVerificationOptions struct {
	client      modelVerificationHTTPClient
	loadConfig  func(string) (*routerconfig.RouterConfig, error)
	timeout     time.Duration
	now         func() time.Time
	rateLimiter *modelVerificationRateLimiter
	auditor     ModelVerificationAuditor
}

type modelVerificationService struct {
	configPath string
	client     modelVerificationHTTPClient
	loadConfig func(string) (*routerconfig.RouterConfig, error)
	timeout    time.Duration
	now        func() time.Time
	inflightMu sync.Mutex
	inflight   map[string]*modelVerificationCall
}

type modelVerificationCall struct {
	done     chan struct{}
	response ModelVerificationResponse
	err      error
}

type modelVerificationTarget struct {
	model         string
	providerModel string
	backend       string
	provider      string
	endpointURL   string
	dialect       string
	authHeader    string
	authPrefix    string
	apiKey        string
	extraHeaders  map[string]string
	secrets       []string
}

type modelVerificationError struct {
	status  int
	code    string
	message string
}

func (e *modelVerificationError) Error() string { return e.message }

func newModelVerificationService(configPath string, options modelVerificationOptions) *modelVerificationService {
	client := options.client
	if client == nil {
		client = &http.Client{
			CheckRedirect: func(_ *http.Request, _ []*http.Request) error {
				return http.ErrUseLastResponse
			},
		}
	}
	loadConfig := options.loadConfig
	if loadConfig == nil {
		loadConfig = routerconfig.Parse
	}
	timeout := options.timeout
	if timeout <= 0 || timeout > modelVerificationTimeout {
		timeout = modelVerificationTimeout
	}
	now := options.now
	if now == nil {
		now = time.Now
	}
	return &modelVerificationService{
		configPath: strings.TrimSpace(configPath),
		client:     client,
		loadConfig: loadConfig,
		timeout:    timeout,
		now:        now,
		inflight:   make(map[string]*modelVerificationCall),
	}
}

func (service *modelVerificationService) Verify(ctx context.Context, model string) (ModelVerificationResponse, error) {
	target, err := service.resolveTarget(model)
	if err != nil {
		return ModelVerificationResponse{}, err
	}
	inflightKey := modelVerificationTargetKey(target)

	service.inflightMu.Lock()
	if call, found := service.inflight[inflightKey]; found {
		service.inflightMu.Unlock()
		select {
		case <-ctx.Done():
			return ModelVerificationResponse{}, modelVerificationFailure(
				http.StatusGatewayTimeout,
				"verification_timeout",
				"Model inference verification was cancelled before the shared request completed.",
			)
		case <-call.done:
			return call.response, call.err
		}
	}
	call := &modelVerificationCall{done: make(chan struct{})}
	service.inflight[inflightKey] = call
	service.inflightMu.Unlock()

	call.response, call.err = service.verifyTarget(ctx, target)
	service.inflightMu.Lock()
	delete(service.inflight, inflightKey)
	close(call.done)
	service.inflightMu.Unlock()
	return call.response, call.err
}

func (service *modelVerificationService) resolveTarget(model string) (modelVerificationTarget, error) {
	if service.configPath == "" {
		return modelVerificationTarget{}, modelVerificationFailure(
			http.StatusServiceUnavailable,
			"config_unavailable",
			"Active model configuration is unavailable.",
		)
	}
	config, err := service.loadConfig(service.configPath)
	if err != nil {
		return modelVerificationTarget{}, modelVerificationFailure(
			http.StatusServiceUnavailable,
			"config_unavailable",
			"Active model configuration could not be loaded.",
		)
	}
	target, err := resolveModelVerificationTarget(config, model)
	if err != nil {
		return modelVerificationTarget{}, err
	}
	return target, nil
}

func (service *modelVerificationService) verifyTarget(ctx context.Context, target modelVerificationTarget) (ModelVerificationResponse, error) {
	callContext, cancel := context.WithTimeout(ctx, service.timeout)
	defer cancel()
	started := time.Now()
	summary, err := service.call(callContext, target)
	if err != nil {
		return ModelVerificationResponse{}, err
	}
	latency := time.Since(started).Milliseconds()
	if latency < 0 {
		latency = 0
	}
	return ModelVerificationResponse{
		Verified:      true,
		Model:         target.model,
		ProviderModel: target.providerModel,
		Backend:       target.backend,
		Provider:      target.provider,
		Summary:       summarizeModelVerificationText(summary, modelVerificationMaxSummaryRunes),
		VerifiedAt:    service.now().UTC().Format(time.RFC3339),
		LatencyMS:     latency,
	}, nil
}

func modelVerificationTargetKey(target modelVerificationTarget) string {
	payload, _ := json.Marshal(struct {
		Model         string            `json:"model"`
		ProviderModel string            `json:"provider_model"`
		Backend       string            `json:"backend"`
		Provider      string            `json:"provider"`
		EndpointURL   string            `json:"endpoint_url"`
		Dialect       string            `json:"dialect"`
		AuthHeader    string            `json:"auth_header"`
		AuthPrefix    string            `json:"auth_prefix"`
		APIKey        string            `json:"api_key"`
		ExtraHeaders  map[string]string `json:"extra_headers"`
	}{
		Model:         target.model,
		ProviderModel: target.providerModel,
		Backend:       target.backend,
		Provider:      target.provider,
		EndpointURL:   target.endpointURL,
		Dialect:       target.dialect,
		AuthHeader:    target.authHeader,
		AuthPrefix:    target.authPrefix,
		APIKey:        target.apiKey,
		ExtraHeaders:  target.extraHeaders,
	})
	return fmt.Sprintf("%x", sha256.Sum256(payload))
}

func resolveModelVerificationTarget(config *routerconfig.RouterConfig, model string) (modelVerificationTarget, error) {
	if config == nil {
		return modelVerificationTarget{}, modelVerificationFailure(
			http.StatusServiceUnavailable,
			"config_unavailable",
			"Active model configuration could not be loaded.",
		)
	}
	params, modelConfigured := config.ModelConfig[model]
	endpoints := config.GetEndpointsForModel(model)
	if len(endpoints) == 0 {
		for _, endpoint := range config.VLLMEndpoints {
			if strings.TrimSpace(endpoint.Model) == model {
				endpoints = append(endpoints, endpoint)
			}
		}
	}
	if !modelConfigured && len(endpoints) == 0 {
		return modelVerificationTarget{}, modelVerificationFailure(
			http.StatusNotFound,
			"model_not_found",
			fmt.Sprintf("Model %q is not present in the active provider configuration.", model),
		)
	}
	if len(endpoints) == 0 {
		return modelVerificationTarget{}, modelVerificationFailure(
			http.StatusUnprocessableEntity,
			"backend_unavailable",
			fmt.Sprintf("Model %q has no configured provider backend.", model),
		)
	}
	endpoint := endpoints[0]
	for _, candidate := range endpoints[1:] {
		if candidate.Weight > endpoint.Weight {
			endpoint = candidate
		}
	}

	profile, err := config.GetProviderProfileForEndpoint(endpoint.Name)
	if err != nil {
		return modelVerificationTarget{}, modelVerificationFailure(
			http.StatusUnprocessableEntity,
			"backend_invalid",
			fmt.Sprintf("Model %q has invalid provider backend metadata.", model),
		)
	}
	target := modelVerificationTarget{
		model:         model,
		providerModel: config.ResolveExternalModelID(model, endpoint.Name),
		backend:       endpoint.Name,
		provider:      strings.TrimSpace(endpoint.Type),
		dialect:       strings.ToLower(strings.TrimSpace(params.APIFormat)),
		apiKey:        strings.TrimSpace(endpoint.APIKey),
	}
	if target.providerModel == "" {
		target.providerModel = model
	}
	if target.provider == "" {
		target.provider = "vllm"
	}
	if target.apiKey == "" {
		target.apiKey = strings.TrimSpace(config.GetModelAccessKey(model))
	}

	if profile == nil {
		target.endpointURL, err = legacyModelVerificationURL(endpoint, target.dialect)
		target.authHeader = "Authorization"
		target.authPrefix = "Bearer"
	} else {
		target.provider, err = profile.ProviderType()
		if err == nil {
			target.endpointURL, err = profiledModelVerificationURL(profile)
		}
		if err == nil {
			target.authHeader, target.authPrefix, err = profile.ResolveAuthHeader()
		}
		target.extraHeaders = copyModelVerificationHeaders(profile.ExtraHeaders)
	}
	if err != nil {
		return modelVerificationTarget{}, modelVerificationFailure(
			http.StatusUnprocessableEntity,
			"backend_invalid",
			fmt.Sprintf("Model %q has invalid provider backend metadata.", model),
		)
	}
	if target.dialect == "" {
		target.dialect = target.provider
	}
	if target.dialect == "anthropic_messages" {
		target.dialect = "anthropic"
	}
	target.secrets = modelVerificationCredentialSecrets(target)
	return target, nil
}

func legacyModelVerificationURL(endpoint routerconfig.VLLMEndpoint, dialect string) (string, error) {
	protocol := strings.ToLower(strings.TrimSpace(endpoint.Protocol))
	if protocol == "" {
		protocol = "http"
	}
	if protocol != "http" && protocol != "https" {
		return "", fmt.Errorf("unsupported backend protocol")
	}
	host := strings.TrimSpace(endpoint.Address)
	if host == "" {
		return "", fmt.Errorf("backend address is empty")
	}
	if strings.ContainsFunc(host, unicode.IsSpace) {
		return "", fmt.Errorf("backend address contains whitespace")
	}
	if endpoint.Port > 0 {
		host = net.JoinHostPort(strings.Trim(host, "[]"), strconv.Itoa(endpoint.Port))
	}
	parsed, err := url.Parse(protocol + "://" + host)
	if err != nil || parsed.Host == "" || parsed.Hostname() == "" || parsed.User != nil || parsed.Path != "" || parsed.RawQuery != "" || parsed.Fragment != "" {
		return "", fmt.Errorf("backend address is invalid")
	}
	path := "/v1/chat/completions"
	if strings.EqualFold(strings.TrimSpace(dialect), "anthropic") || strings.EqualFold(strings.TrimSpace(endpoint.Type), "anthropic") {
		path = "/v1/messages"
	}
	return (&url.URL{Scheme: protocol, Host: host, Path: path}).String(), nil
}

func profiledModelVerificationURL(profile *routerconfig.ProviderProfile) (string, error) {
	baseURL, err := url.Parse(strings.TrimSpace(profile.BaseURL))
	if err != nil || baseURL.Scheme == "" || baseURL.Host == "" || baseURL.Hostname() == "" || baseURL.Opaque != "" {
		return "", fmt.Errorf("provider base URL is invalid")
	}
	if baseURL.Scheme != "http" && baseURL.Scheme != "https" {
		return "", fmt.Errorf("provider base URL scheme is unsupported")
	}
	if baseURL.User != nil {
		return "", fmt.Errorf("provider base URL user info is unsupported")
	}
	chatPath, err := profile.ResolveChatPath()
	if err != nil {
		return "", err
	}
	pathURL, err := url.Parse(chatPath)
	if err != nil || pathURL.IsAbs() || pathURL.Host != "" {
		return "", fmt.Errorf("provider chat path is invalid")
	}
	return (&url.URL{
		Scheme:   baseURL.Scheme,
		Host:     baseURL.Host,
		Path:     pathURL.Path,
		RawPath:  pathURL.RawPath,
		RawQuery: pathURL.RawQuery,
	}).String(), nil
}

func modelVerificationFailure(status int, code, message string) error {
	return &modelVerificationError{status: status, code: code, message: message}
}
