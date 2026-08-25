// Package modelprobe verifies one dynamically authored Model through the same stable wire
// adapters, credential resolver, and egress transport used by inference. It
// contains no Provider-product branches.
package modelprobe

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

const publicProbePath = "/v1/chat/completions"

type Options struct {
	Credentials CredentialResolver
	Codecs      *protocolcodec.Registry
	Transport   backendinvoker.Transport
	Now         func() time.Time
}

// CredentialResolver is the Management-only ProviderCredential contract used
// by an explicit probe. Unlike inference dispatch, a probe is not associated
// with an activated routing publication and may read management storage.
type CredentialResolver interface {
	Pin(context.Context, string, string, string) (string, error)
	ResolvePinned(context.Context, string, string, string, string) (backendinvoker.Credential, error)
}

type managementCredentialAdapter struct{ resolver CredentialResolver }

func (adapter managementCredentialAdapter) Pin(
	ctx context.Context,
	_ backendinvoker.CredentialPublication,
	credentialID string,
	providerID string,
	origin string,
) (string, error) {
	return adapter.resolver.Pin(ctx, credentialID, providerID, origin)
}

func (adapter managementCredentialAdapter) ResolvePinned(
	ctx context.Context,
	_ backendinvoker.CredentialPublication,
	credentialID string,
	versionID string,
	providerID string,
	origin string,
) (backendinvoker.Credential, error) {
	return adapter.resolver.ResolvePinned(ctx, credentialID, versionID, providerID, origin)
}

// Prober owns no transport or credential lifecycle. Management process
// composition lends those resources and closes them after Management stops.
type Prober struct {
	invoker *backendinvoker.Invoker
	now     func() time.Time
}

func New(options Options) (*Prober, error) {
	if options.Codecs == nil || options.Transport == nil {
		return nil, fmt.Errorf("model probe requires protocol codecs and an egress transport")
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	var credentials backendinvoker.CredentialResolver
	if options.Credentials != nil {
		credentials = managementCredentialAdapter{resolver: options.Credentials}
	}
	return &Prober{
		invoker: &backendinvoker.Invoker{
			Credentials: credentials,
			Codecs:      options.Codecs,
			Journal:     backendinvoker.ProcessLocalJournal{},
			Transport:   options.Transport,
		},
		now: now,
	}, nil
}

func (prober *Prober) Probe(
	ctx context.Context,
	request routingmanagement.ProbeRequest,
) (routingmanagement.ProbeResult, error) {
	if prober == nil || prober.invoker == nil || request.NamespaceID == "" ||
		request.Timeout <= 0 || request.Model.ID == "" || request.Model.Revision <= 0 ||
		len(request.Model.Backends) == 0 || request.Model.Execution.MaxRetries < 0 ||
		request.Model.Execution.MaxRetries > 5 {
		return routingmanagement.ProbeResult{}, fmt.Errorf("model probe request is invalid")
	}
	body, err := probeRequestBody(request.Model.ID)
	if err != nil {
		return routingmanagement.ProbeResult{}, err
	}
	started := time.Now()
	available := false
	for index, backend := range request.Model.Backends {
		plan := probePlan(request, backend, index, body)
		result, invokeErr := prober.invoker.Invoke(ctx, plan)
		if result.Response != nil && result.Response.Body != nil {
			status := result.Response.StatusCode
			_ = result.Response.Body.Close()
			if invokeErr == nil && status >= http.StatusOK && status < http.StatusMultipleChoices {
				available = true
				break
			}
		}
		if ctx.Err() != nil {
			break
		}
	}
	return routingmanagement.ProbeResult{
		Available: available,
		Latency:   time.Since(started),
		CheckedAt: prober.now().UTC(),
	}, nil
}

func probeRequestBody(modelID string) ([]byte, error) {
	payload := struct {
		Model    string `json:"model"`
		Messages []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"messages"`
		MaxTokens int `json:"max_tokens"`
	}{Model: modelID, MaxTokens: 1}
	payload.Messages = append(payload.Messages, struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}{Role: "user", Content: "Reply with OK."})
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("encode Model probe request: %w", err)
	}
	return body, nil
}

func probePlan(
	request routingmanagement.ProbeRequest,
	backend routingsnapshot.Backend,
	index int,
	body []byte,
) backendinvoker.Plan {
	dispatchID := "management-probe-" + strconv.Itoa(index)
	plan := backendinvoker.Plan{
		NamespaceID:    request.NamespaceID,
		QuotaPartition: "management-probe", PublicationID: "management-probe",
		RuntimeEpoch: 1, RoutingRevision: request.Model.Revision,
		RoutingDigest:   probeDigest("routing", request.NamespaceID, request.Model.ID, strconv.FormatInt(request.Model.Revision, 10)),
		AdmissionID:     "management-probe",
		AdmissionDigest: probeDigest("admission", request.NamespaceID, request.Model.ID),
		RequestID:       "management-probe", DispatchID: dispatchID, DispatchType: "model-probe",
		Ordinal: 0, Priority: 0,
		DispatchPlanDigest: probeDigest("dispatch", request.NamespaceID, request.Model.ID, backend.ID),
		ModelID:            request.Model.ID, ModelRevision: request.Model.Revision,
		Method: http.MethodPost, Path: publicProbePath, SourceFormat: llmprotocol.OpenAIChatV1, Headers: make(http.Header), Body: append([]byte(nil), body...),
		Execution: backendinvoker.Execution{
			MaxRetries:     request.Model.Execution.MaxRetries,
			RetryOn:        probeRetryTriggers(request.Model.Execution.RetryOn),
			RequestTimeout: request.Timeout,
			StreamTimeout:  request.Timeout,
		},
		Backends: []backendinvoker.Backend{{
			ID: backend.ID, Origin: backend.Origin, ProviderID: backend.ProviderID,
			WireFormat: backend.WireFormat, ProviderModelID: backend.ProviderModelID,
			ProviderCredentialID: backend.ProviderCredentialID,
			Connection: backendinvoker.Connection{
				Path: backend.Connection.Path, Headers: connectionHeaders(backend.Connection.Headers),
			},
			Weight: 1,
		}},
	}
	plan.RequestDigest = backendinvoker.RequestDigest(plan.Method, plan.Path, plan.Query, plan.Body)
	return plan
}

func probeRetryTriggers(source []string) []backendinvoker.FallbackTrigger {
	result := make([]backendinvoker.FallbackTrigger, len(source))
	for index, trigger := range source {
		result[index] = backendinvoker.FallbackTrigger(trigger)
	}
	return result
}

func connectionHeaders(source map[string]string) http.Header {
	result := make(http.Header, len(source))
	for name, value := range source {
		result.Set(name, value)
	}
	return result
}

func probeDigest(parts ...string) string {
	digest := sha256.New()
	for _, part := range parts {
		_, _ = digest.Write([]byte(strconv.Itoa(len(part))))
		_, _ = digest.Write([]byte{':'})
		_, _ = digest.Write([]byte(part))
		_, _ = digest.Write([]byte{';'})
	}
	return hex.EncodeToString(digest.Sum(nil))
}

var _ routingmanagement.Prober = (*Prober)(nil)
