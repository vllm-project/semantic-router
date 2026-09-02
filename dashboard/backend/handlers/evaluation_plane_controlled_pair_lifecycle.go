package handlers

import (
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/vllm-project/semantic-router/dashboard/backend/evaluationplane"
)

// ControlledPairLifecycle owns aggregate-only mutations. Pair members reject
// generic run lifecycle calls, so clients never have to coordinate two run
// mutations or infer aggregate membership from names and baseline links.
func (h *EvaluationPlaneHandler) ControlledPairLifecycle(w http.ResponseWriter, r *http.Request) {
	if evaluationCORS(w, r) {
		return
	}
	pairID, action, err := controlledPairLifecyclePath(r.URL.Path)
	if err != nil {
		writeEvaluationError(w, err)
		return
	}
	allowed := []string{http.MethodGet, http.MethodDelete}
	if action == "cancel" {
		allowed = []string{http.MethodPost}
	}
	if !controlledPairMethodAllowed(r.Method, allowed) {
		methodNotAllowed(w, allowed...)
		return
	}
	actor, ok := h.evaluationActor(w, r)
	if !ok {
		return
	}
	if r.URL.RawQuery != "" {
		writeEvaluationError(w, fmt.Errorf("%w: controlled pair lifecycle path does not accept query parameters", evaluationplane.ErrInvalid))
		return
	}
	if err := requireEmptyControlledPairBody(r); err != nil {
		writeEvaluationError(w, err)
		return
	}
	switch {
	case r.Method == http.MethodGet && action == "":
		execution, err := h.service.GetControlledPairExecutionAs(actor, pairID)
		if err != nil {
			writeEvaluationError(w, err)
			return
		}
		writeEvaluationJSON(w, http.StatusOK, execution)
	case r.Method == http.MethodPost && action == "cancel":
		if h.denyReadonly(w) {
			return
		}
		execution, err := h.service.CancelControlledPairExecutionAs(actor, pairID)
		if err != nil {
			writeEvaluationError(w, err)
			return
		}
		writeEvaluationJSON(w, http.StatusOK, execution)
	case r.Method == http.MethodDelete && action == "":
		if h.denyReadonly(w) {
			return
		}
		if err := h.service.DeleteControlledPairExecutionAs(actor, pairID); err != nil {
			writeEvaluationError(w, err)
			return
		}
		w.WriteHeader(http.StatusNoContent)
	}
}

func controlledPairMethodAllowed(method string, allowed []string) bool {
	for _, candidate := range allowed {
		if method == candidate {
			return true
		}
	}
	return false
}

func requireEmptyControlledPairBody(r *http.Request) error {
	if r.Body == nil {
		return nil
	}
	var first [1]byte
	n, err := r.Body.Read(first[:])
	if n != 0 {
		return fmt.Errorf("%w: controlled pair lifecycle request body must be empty", evaluationplane.ErrInvalid)
	}
	if err != nil && err != io.EOF {
		return fmt.Errorf("%w: read controlled pair lifecycle request body", evaluationplane.ErrInvalid)
	}
	return nil
}

func controlledPairLifecyclePath(path string) (string, string, error) {
	prefix := evaluationAPIBase + "/controlled-pairs/"
	if !strings.HasPrefix(path, prefix) {
		return "", "", fmt.Errorf("%w: controlled pair lifecycle path is invalid", evaluationplane.ErrInvalid)
	}
	parts := strings.Split(strings.TrimPrefix(path, prefix), "/")
	if len(parts) == 1 && parts[0] != "" {
		return parts[0], "", nil
	}
	if len(parts) == 2 && parts[0] != "" && parts[1] == "cancel" {
		return parts[0], "cancel", nil
	}
	return "", "", fmt.Errorf("%w: controlled pair lifecycle path is invalid", evaluationplane.ErrInvalid)
}
