package handlers

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"path"
	"strconv"
	"strings"
	"time"
	"unicode"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const modelHealthMaxListBytes = 2 << 20

type modelHealthCheck int

const (
	// modelHealthCheckList reads the provider's model list with the
	// backend's credentials and looks for the model in it.
	modelHealthCheckList modelHealthCheck = iota
	// modelHealthCheckPath reads the configured health check path without
	// credentials, as the gateway's active health checks do.
	modelHealthCheckPath
	// modelHealthCheckReach reads the base URL of a provider without a model
	// list: any answer below 500 shows the backend is reachable.
	modelHealthCheckReach
)

type modelHealthPlan struct {
	model       string
	fingerprint string
	// reason says why the model has nothing to check.
	reason  string
	targets []modelHealthTarget
}

type modelHealthTarget struct {
	backend       string
	weight        int
	check         modelHealthCheck
	url           string
	headers       map[string]string
	providerModel string
	// callerAuth backends take each request's credential, so a rejected
	// check without one still shows the backend answering.
	callerAuth bool
	tls        *routerconfig.DeploymentTLS
	problem    string
}

type modelHealthProbe struct {
	backend    string
	weight     int
	status     string
	reason     string
	httpStatus int
	latencyMS  int64
	answered   bool
}

func resolveModelHealthPlan(config *routerconfig.RouterConfig, registry *modelcatalog.Registry, model string) modelHealthPlan {
	plan := modelHealthPlan{model: model}
	params := config.ModelConfig[model]
	endpoints := config.GetEndpointsForModel(model)
	if len(endpoints) == 0 {
		for _, endpoint := range config.VLLMEndpoints {
			if strings.TrimSpace(endpoint.Model) == model {
				endpoints = append(endpoints, endpoint)
			}
		}
	}
	switch {
	case len(endpoints) > 0:
		healthPath := strings.TrimSpace(params.Reliability.HealthCheckPath)
		for _, endpoint := range endpoints {
			plan.targets = append(plan.targets, resolveModelHealthTarget(config, registry, model, endpoint, healthPath))
		}
	case strings.TrimSpace(params.Deployment) != "":
		plan.reason = fmt.Sprintf("Served by the shared deployment %s.", strings.TrimSpace(params.Deployment))
	default:
		plan.reason = "No backend endpoint to check."
	}
	plan.fingerprint = modelHealthFingerprint(plan)
	return plan
}

func resolveModelHealthTarget(
	config *routerconfig.RouterConfig,
	registry *modelcatalog.Registry,
	model string,
	endpoint routerconfig.VLLMEndpoint,
	healthPath string,
) modelHealthTarget {
	target := modelHealthTarget{
		// Canonical import names an endpoint <model>_<backend>; show the backend.
		backend: strings.TrimPrefix(endpoint.Name, model+"_"),
		weight:  endpoint.Weight,
	}
	profile, err := config.GetProviderProfileForEndpoint(endpoint.Name)
	if err != nil {
		target.problem = "The backend's provider settings are invalid."
		return target
	}
	base, err := modelHealthBaseURL(endpoint, profile)
	if err != nil {
		target.problem = "The backend URL is invalid."
		return target
	}
	if profile != nil && profile.TLS != nil && base.Scheme == "https" {
		settings := *profile.TLS
		target.tls = &settings
	}
	if healthPath != "" {
		parsed, parseErr := url.Parse(healthPath)
		if parseErr != nil || parsed.IsAbs() || parsed.Host != "" || !strings.HasPrefix(parsed.Path, "/") {
			target.problem = "The health check path is invalid."
			return target
		}
		target.check = modelHealthCheckPath
		target.url = (&url.URL{Scheme: base.Scheme, Host: base.Host, Path: parsed.Path, RawQuery: parsed.RawQuery}).String()
		return target
	}

	providerID := strings.TrimSpace(endpoint.Type)
	protocol := ""
	if profile != nil {
		if resolved, typeErr := profile.ProviderType(); typeErr == nil {
			providerID = resolved
		}
		protocol = profile.Protocol
	}
	if providerID == "" {
		providerID = "vllm"
	}
	target.check = modelHealthCheckReach
	target.url = base.String()
	if registry == nil {
		return target
	}
	operation, err := registry.ResolveOperation(providerID, protocol, "list_models", base.Path)
	if err != nil {
		return target
	}
	listURL := url.URL{Scheme: base.Scheme, Host: base.Host, Path: path.Clean(operation.Path)}
	if operation.UseAPIVersionQuery && profile != nil && profile.APIVersion != "" {
		listURL.RawQuery = url.Values{"api-version": {profile.APIVersion}}.Encode()
	}
	target.check = modelHealthCheckList
	target.url = listURL.String()
	target.providerModel = strings.TrimSpace(config.ResolveExternalModelID(model, endpoint.Name))
	target.headers = modelHealthListHeaders(config, registry, model, endpoint, profile, providerID)
	target.callerAuth = profile.Delegated() || profile.Caller()
	return target
}

func modelHealthListHeaders(
	config *routerconfig.RouterConfig,
	registry *modelcatalog.Registry,
	model string,
	endpoint routerconfig.VLLMEndpoint,
	profile *routerconfig.ProviderProfile,
	providerID string,
) map[string]string {
	headers := map[string]string{}
	if definition, found := registry.Provider(providerID); found {
		for key, value := range definition.DefaultHeaders {
			headers[key] = value
		}
	}
	if profile != nil {
		for key, value := range profile.ExtraHeaders {
			headers[key] = value
		}
	}
	if !profile.Delegated() && !profile.Caller() {
		apiKey := strings.TrimSpace(endpoint.APIKey)
		if apiKey == "" {
			apiKey = strings.TrimSpace(config.GetModelAccessKey(model))
		}
		header, prefix := "Authorization", "Bearer"
		if profile != nil {
			if resolvedHeader, resolvedPrefix, err := profile.ResolveAuthHeader(); err == nil {
				header, prefix = resolvedHeader, resolvedPrefix
			}
		}
		if apiKey != "" && header != "" {
			if strings.TrimSpace(prefix) != "" {
				apiKey = strings.TrimSpace(prefix) + " " + apiKey
			}
			headers[header] = apiKey
		}
	}
	for key := range headers {
		if !modelVerificationHeaderAllowed(key) {
			delete(headers, key)
		}
	}
	return headers
}

func modelHealthBaseURL(endpoint routerconfig.VLLMEndpoint, profile *routerconfig.ProviderProfile) (*url.URL, error) {
	if profile != nil && strings.TrimSpace(profile.BaseURL) != "" {
		parsed, err := url.Parse(strings.TrimSpace(profile.BaseURL))
		if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Hostname() == "" || parsed.User != nil || parsed.Opaque != "" {
			return nil, errors.New("provider base URL is invalid")
		}
		return &url.URL{Scheme: parsed.Scheme, Host: parsed.Host, Path: strings.TrimRight(parsed.Path, "/")}, nil
	}
	protocol := strings.ToLower(strings.TrimSpace(endpoint.Protocol))
	if protocol == "" {
		protocol = "http"
	}
	host := strings.TrimSpace(endpoint.Address)
	if (protocol != "http" && protocol != "https") || host == "" || strings.ContainsFunc(host, unicode.IsSpace) {
		return nil, errors.New("backend address is invalid")
	}
	if endpoint.Port > 0 {
		host = net.JoinHostPort(strings.Trim(host, "[]"), strconv.Itoa(endpoint.Port))
	}
	parsed, err := url.Parse(protocol + "://" + host)
	if err != nil || parsed.Hostname() == "" || parsed.User != nil || parsed.Path != "" || parsed.RawQuery != "" || parsed.Fragment != "" {
		return nil, errors.New("backend address is invalid")
	}
	return parsed, nil
}

func modelHealthFingerprint(plan modelHealthPlan) string {
	type fingerprintTarget struct {
		Backend       string                      `json:"backend"`
		Weight        int                         `json:"weight"`
		Check         modelHealthCheck            `json:"check"`
		URL           string                      `json:"url"`
		Headers       map[string]string           `json:"headers"`
		ProviderModel string                      `json:"provider_model"`
		CallerAuth    bool                        `json:"caller_auth"`
		TLS           *routerconfig.DeploymentTLS `json:"tls"`
		Problem       string                      `json:"problem"`
	}
	targets := make([]fingerprintTarget, 0, len(plan.targets))
	for _, target := range plan.targets {
		targets = append(targets, fingerprintTarget{
			Backend: target.backend, Weight: target.weight, Check: target.check, URL: target.url,
			Headers: target.headers, ProviderModel: target.providerModel, CallerAuth: target.callerAuth,
			TLS: target.tls, Problem: target.problem,
		})
	}
	payload, _ := json.Marshal(struct {
		Model   string              `json:"model"`
		Reason  string              `json:"reason"`
		Targets []fingerprintTarget `json:"targets"`
	}{Model: plan.model, Reason: plan.reason, Targets: targets})
	return fmt.Sprintf("%x", sha256.Sum256(payload))
}

func (service *modelHealthService) probe(ctx context.Context, target modelHealthTarget) modelHealthProbe {
	result := modelHealthProbe{backend: target.backend, weight: target.weight, status: ModelHealthStatusUnchecked}
	if target.problem != "" {
		result.reason = target.problem
		return result
	}
	client, err := service.clientFor(target)
	if err != nil {
		result.reason = "The Dashboard cannot read this backend's TLS files."
		return result
	}
	callContext, cancel := context.WithTimeout(ctx, service.timeout)
	defer cancel()
	request, err := http.NewRequestWithContext(callContext, http.MethodGet, target.url, nil)
	if err != nil {
		result.reason = "The backend URL is invalid."
		return result
	}
	request.Header.Set("Accept", "application/json")
	for key, value := range target.headers {
		request.Header.Set(key, value)
	}
	started := time.Now()
	response, err := client.Do(request)
	if err != nil {
		result.status = ModelHealthStatusDown
		result.reason = "The backend could not be reached."
		if errors.Is(callContext.Err(), context.DeadlineExceeded) || errors.Is(err, context.DeadlineExceeded) {
			result.reason = fmt.Sprintf("No answer within %s.", formatModelHealthDuration(service.timeout))
		}
		return result
	}
	defer func() { _ = response.Body.Close() }()
	latency := time.Since(started)
	result.httpStatus = response.StatusCode
	result.latencyMS = max(latency.Milliseconds(), 0)
	classifyModelHealthAnswer(&result, target, response.StatusCode, response.Body, latency)
	return result
}

func (service *modelHealthService) clientFor(target modelHealthTarget) (modelVerificationHTTPClient, error) {
	tlsConfig, err := target.tls.ClientConfig()
	if err != nil {
		return nil, err
	}
	if service.client != nil {
		return service.client, nil
	}
	if tlsConfig == nil {
		return modelHealthDefaultClient, nil
	}
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.TLSClientConfig = tlsConfig
	transport.DisableKeepAlives = true
	return &http.Client{Transport: transport, CheckRedirect: modelHealthDefaultClient.CheckRedirect}, nil
}

func classifyModelHealthAnswer(result *modelHealthProbe, target modelHealthTarget, status int, body io.Reader, latency time.Duration) {
	switch {
	case status >= http.StatusOK && status < http.StatusMultipleChoices:
		result.status = ModelHealthStatusHealthy
		if target.check == modelHealthCheckList && target.providerModel != "" {
			if listed, known := modelHealthListsModel(body, target.providerModel); known && !listed {
				result.status = ModelHealthStatusDegraded
				result.reason = fmt.Sprintf("The backend does not list %s.", target.providerModel)
			}
		}
	case target.check == modelHealthCheckPath:
		result.status = ModelHealthStatusDown
		result.reason = fmt.Sprintf("The health check answered HTTP %d.", status)
	case target.check == modelHealthCheckReach && status < http.StatusInternalServerError:
		result.status = ModelHealthStatusHealthy
		result.reason = "Reachable; this provider has no model list to confirm the model."
	case (status == http.StatusUnauthorized || status == http.StatusForbidden) && target.callerAuth:
		result.status = ModelHealthStatusHealthy
		result.reason = "Reachable; requests carry the caller's credential."
	case status == http.StatusUnauthorized || status == http.StatusForbidden:
		result.status = ModelHealthStatusDown
		result.reason = fmt.Sprintf("The backend rejected its credentials (HTTP %d).", status)
	case status == http.StatusTooManyRequests:
		result.status = ModelHealthStatusDegraded
		result.reason = "The backend is rate limiting requests (HTTP 429)."
	case status >= http.StatusInternalServerError:
		result.status = ModelHealthStatusDown
		result.reason = fmt.Sprintf("The backend answered HTTP %d.", status)
	default:
		result.status = ModelHealthStatusDegraded
		result.reason = fmt.Sprintf("The model list answered HTTP %d.", status)
	}
	result.answered = result.status == ModelHealthStatusHealthy || result.status == ModelHealthStatusDegraded
	if result.status == ModelHealthStatusHealthy && latency > modelHealthSlowAfter {
		result.status = ModelHealthStatusDegraded
		result.reason = fmt.Sprintf("Slow: answered in %s.", formatModelHealthDuration(latency))
	}
}

// modelHealthListsModel reports whether a model list names the model; known
// is false when the answer is no complete list, such as one page of many.
func modelHealthListsModel(body io.Reader, providerModel string) (listed bool, known bool) {
	data, err := io.ReadAll(io.LimitReader(body, modelHealthMaxListBytes+1))
	if err != nil || len(data) > modelHealthMaxListBytes {
		return false, false
	}
	type listItem struct {
		ID   string `json:"id"`
		Name string `json:"name"`
	}
	var payload struct {
		Data          []listItem `json:"data"`
		Models        []listItem `json:"models"`
		HasMore       bool       `json:"has_more"`
		NextPageToken string     `json:"nextPageToken"`
	}
	if json.Unmarshal(data, &payload) != nil {
		return false, false
	}
	items := payload.Data
	if len(items) == 0 {
		items = payload.Models
	}
	if len(items) == 0 || payload.HasMore || payload.NextPageToken != "" {
		return false, false
	}
	want := strings.TrimPrefix(providerModel, "models/")
	for _, item := range items {
		for _, value := range []string{item.ID, item.Name} {
			if strings.EqualFold(strings.TrimPrefix(strings.TrimSpace(value), "models/"), want) {
				return true, true
			}
		}
	}
	return false, true
}

func aggregateModelHealth(plan modelHealthPlan, probes []modelHealthProbe) ModelHealthEntry {
	entry := ModelHealthEntry{Model: plan.model, Status: ModelHealthStatusUnchecked, Reason: plan.reason}
	if plan.reason != "" {
		return entry
	}
	counts := map[string]int{}
	var primary *modelHealthProbe
	for index := range probes {
		probe := &probes[index]
		counts[probe.status]++
		endpoint := ModelHealthEndpoint{Name: probe.backend, Status: probe.status, Reason: probe.reason, HTTPStatus: probe.httpStatus}
		if probe.answered {
			latency := probe.latencyMS
			endpoint.LatencyMS = &latency
			if primary == nil || probe.weight > primary.weight {
				primary = probe
			}
		}
		entry.Endpoints = append(entry.Endpoints, endpoint)
	}
	checked := len(probes) - counts[ModelHealthStatusUnchecked]
	switch {
	case checked == 0:
		entry.Reason = firstModelHealthReason(probes, ModelHealthStatusUnchecked)
	case counts[ModelHealthStatusHealthy] == checked:
		entry.Status = ModelHealthStatusHealthy
		entry.Reason = firstModelHealthReason(probes, ModelHealthStatusHealthy)
	case counts[ModelHealthStatusDown] == checked:
		entry.Status = ModelHealthStatusDown
		entry.Reason = firstModelHealthReason(probes, ModelHealthStatusDown)
	case counts[ModelHealthStatusDown] > 0:
		entry.Status = ModelHealthStatusDegraded
		entry.Reason = fmt.Sprintf("Down on %d of %d backends.", counts[ModelHealthStatusDown], checked)
	default:
		entry.Status = ModelHealthStatusDegraded
		entry.Reason = firstModelHealthReason(probes, ModelHealthStatusDegraded)
	}
	if primary != nil {
		latency := primary.latencyMS
		entry.LatencyMS = &latency
	}
	return entry
}

func firstModelHealthReason(probes []modelHealthProbe, status string) string {
	for _, probe := range probes {
		if probe.status == status && probe.reason != "" {
			return probe.reason
		}
	}
	return ""
}

func formatModelHealthDuration(duration time.Duration) string {
	if duration < time.Second {
		return fmt.Sprintf("%d ms", duration.Milliseconds())
	}
	return strconv.FormatFloat(duration.Seconds(), 'f', 1, 64) + " s"
}
