package modelservice

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/url"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
)

func (p *replicaPool) Do(request *http.Request) (*http.Response, error) {
	if request.Body == nil || request.Body == http.NoBody {
		return nil, ErrRejected
	}
	defer request.Body.Close()
	body, err := io.ReadAll(io.LimitReader(request.Body, ManagedRequestBytes+1))
	if err != nil {
		return nil, err
	}
	if len(body) > ManagedRequestBytes {
		return nil, ErrRejected
	}
	worker, release, err := p.retain(int64(max(1, len(body))))
	if err != nil {
		return nil, err
	}
	outcome := replicaRejected
	defer func() { release(outcome) }()
	rewritten, err := rewriteExchangeModel(body, worker.member.served.name, request.URL.Path == "/v1/bundle")
	if err != nil {
		return nil, err
	}
	physical := worker.member.group.client
	base, err := url.Parse(physical.base)
	if err != nil {
		return nil, err
	}
	target := request.Clone(request.Context())
	copied := *request.URL
	target.URL = &copied
	target.URL.Scheme, target.URL.Host, target.URL.User = base.Scheme, base.Host, base.User
	target.URL.Path = strings.TrimRight(base.Path, "/") + request.URL.Path
	target.Host = base.Host
	target.Header = request.Header.Clone()
	target.Header.Del("Content-Length")
	target.GetBody = func() (io.ReadCloser, error) { return io.NopCloser(bytes.NewReader(rewritten)), nil }
	target.Body = io.NopCloser(bytes.NewReader(rewritten))
	target.ContentLength = int64(len(rewritten))
	outcome = replicaFailed
	response, err := physical.httpClient.Do(target)
	if err != nil {
		outcome = replicaTransportError(request.Context())
		return nil, err
	}
	// Hold the worker through network body consumption. Decode only transport
	// selectors; semantic payloads remain raw, including native answer fields.
	defer response.Body.Close()
	data, readErr := io.ReadAll(io.LimitReader(response.Body, ManagedRequestBytes+1))
	if readErr != nil {
		outcome = replicaTransportError(request.Context())
		return nil, readErr
	}
	if len(data) > ManagedRequestBytes {
		return nil, ErrFailed
	}
	if response.StatusCode >= 200 && response.StatusCode < 300 && worker.member.served.name != p.name {
		data, err = rewriteResponseModel(data, p.name)
		if err != nil {
			return nil, err
		}
	}
	response.Body = io.NopCloser(bytes.NewReader(data))
	response.ContentLength = int64(len(data))
	response.Header.Del("Content-Length")
	if response.StatusCode >= 200 && response.StatusCode < 300 {
		outcome = replicaOK
	} else if response.StatusCode < 500 {
		outcome = replicaRejected
	}
	return response, nil
}

func replicaTransportError(ctx context.Context) replicaOutcome {
	switch {
	case errors.Is(ctx.Err(), context.DeadlineExceeded):
		return replicaTimeout
	case errors.Is(ctx.Err(), context.Canceled):
		return replicaCanceled
	default:
		return replicaFailed
	}
}

func rewriteResponseModel(data []byte, model string) ([]byte, error) {
	var object map[string]json.RawMessage
	if json.Unmarshal(data, &object) != nil || object == nil {
		return nil, ErrFailed
	}
	if _, ok := object["model"]; ok {
		if err := api.AliasResponseModel(object, model); err != nil {
			return nil, ErrFailed
		}
		return json.Marshal(object)
	}
	if results, ok := object["results"]; ok {
		var bundle []map[string]json.RawMessage
		if json.Unmarshal(results, &bundle) != nil {
			return nil, ErrFailed
		}
		for _, result := range bundle {
			for _, surface := range []string{"decisions", "classify", "embeddings", "rerank"} {
				if part, ok := result[surface]; ok {
					rewritten, err := rewriteResponseModel(part, model)
					if err != nil {
						return nil, err
					}
					result[surface] = rewritten
				}
			}
		}
		object["results"], _ = json.Marshal(bundle)
	}
	return json.Marshal(object)
}

// Only transport-owned model selectors are rewritten. State, question and
// criteria JSON stays raw, including order-sensitive label objects.
func rewriteExchangeModel(body []byte, model string, bundleRequest bool) ([]byte, error) {
	var object map[string]json.RawMessage
	if err := json.Unmarshal(body, &object); err != nil || object == nil {
		return nil, ErrRejected
	}
	encoded, _ := json.Marshal(model)
	if bundleRequest {
		tasks, ok := object["tasks"]
		if !ok {
			return nil, ErrRejected
		}
		var bundle []map[string]json.RawMessage
		if err := json.Unmarshal(tasks, &bundle); err != nil {
			return nil, ErrRejected
		}
		for _, task := range bundle {
			for _, surface := range []string{"decisions", "classify", "embeddings", "rerank"} {
				if part, ok := task[surface]; ok {
					rewritten, err := rewriteExchangeModel(part, model, false)
					if err != nil {
						return nil, err
					}
					task[surface] = rewritten
				}
			}
		}
		object["tasks"], _ = json.Marshal(bundle)
	} else {
		object["model"] = encoded
	}
	return json.Marshal(object)
}
