//go:build !windows

package apiserver

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemone"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/upstream"
)

const systemOneForwardRequestLimit = systemOneRequestLimit + 8192

type nativeModelRuntime interface {
	SystemOne(context.Context, string, json.RawMessage) (modelservice.SystemOneResult, error)
}

// handleSystemOneForward applies public listener policy within the same retained
// generation as inference. Saved, pending and rejected configurations cannot
// authorize requests or rename the active deployment through this transport.
func (s *ClassificationAPIServer) handleSystemOneForward(w http.ResponseWriter, r *http.Request) {
	body, err := readJSONRequestBody(r, systemOneForwardRequestLimit)
	if err != nil {
		status := http.StatusBadRequest
		var limitError *http.MaxBytesError
		if errors.Is(err, errRequestBodyTooLarge) || errors.As(err, &limitError) {
			status = http.StatusRequestEntityTooLarge
		}
		writeSystemOneError(w, status, "invalid_request", "Unable to read bounded native request")
		return
	}
	var forwarded systemone.ForwardRequest
	if decodeStrictJSONBody(body, &forwarded) != nil || !forwarded.ValidOperation() {
		writeSystemOneError(w, http.StatusBadRequest, "invalid_request", "Provide a native inference or discovery operation")
		return
	}
	snapshot, router, release, ok := s.runtimeRegistry.AcquireSystemOne()
	if !ok {
		writeSystemOneError(w, http.StatusServiceUnavailable, "not_ready", "The frontend has no active configuration")
		return
	}
	defer release()
	cfg := snapshot.Config()
	listener, err := systemone.SelectListener(cfg.Listeners, forwarded.Listener)
	if err != nil {
		writeSystemOneError(w, http.StatusNotFound, "systemone_not_published", "System One is not published for the selected listener")
		return
	}
	request, err := http.NewRequestWithContext(r.Context(), forwarded.Method, forwarded.Path, bytes.NewReader(forwarded.Request))
	if err != nil {
		writeSystemOneError(w, http.StatusBadRequest, "invalid_request", "Invalid native operation")
		return
	}
	request.Header.Set("Authorization", forwarded.Authorization)
	request.Header.Set("Api-Key", forwarded.APIKey)
	if forwarded.BackendRequest {
		request.Header.Set(systemone.BackendRequestHeader, "1")
	}
	systemone.Handler(cfg, listener, retainedSystemOneInvoker(snapshot, router, listener.Name))(w, request)
}

// retainedSystemOneInvoker uses only services from the caller's acquired
// generation. Public forwarding and operator diagnostics share this owner.
func retainedSystemOneInvoker(snapshot *configsnapshot.Snapshot, router systemone.Router, listener string) systemone.Invoke {
	models, _ := snapshot.Part(configsnapshot.ComponentModelService).(nativeModelRuntime)
	remote, _ := snapshot.Part(configsnapshot.ComponentUpstream).(*upstream.Set)
	local := func(ctx context.Context, deployment string, body json.RawMessage) (int, []byte, error) {
		if models == nil {
			return 0, nil, modelservice.ErrUnavailable
		}
		result, err := models.SystemOne(ctx, deployment, body)
		return result.Status, result.Body, err
	}
	return systemone.ServingInvoker(snapshot.Config(), listener, router, local, remote)
}
