package systemone

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/apiregistry"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// RulesPath is where the vSR agent plugins ask the rules a listener binds to
// their apps (listeners[].apis.systemone.vsr.rules).
const RulesPath = "/v1/systemone/rules"

// The bounds of one rules request, and how many a listener evaluates at once.
const (
	maxRuleParts     = 16
	maxRuleText      = 256 << 10
	maxRuleBody      = 4 * maxRuleText
	maxRulePartID    = 128
	maxRulesInFlight = 8
)

// RulePart is one text a plugin asks about: a part of a shell command, or a
// piece of a tool's output.
type RulePart struct {
	ID   string `json:"id"`
	Text string `json:"text"`
}

// RuleSpan is the half-open range of code points a redact rule found, after
// overlapping ranges of every rule were merged into one.
type RuleSpan struct {
	Start int    `json:"start"`
	End   int    `json:"end"`
	Rule  string `json:"rule"`
}

// RuleUnchecked is a bound rule the Router could not check on a part.
type RuleUnchecked struct {
	Rule   string `json:"rule"`
	Reason string `json:"reason"`
}

// RulePartResult is what a part's rules came to: the guard rules that
// matched, the text the redact rules found, and the rules left unchecked.
// Coverage is complete when every bound rule was checked, failed when none
// was, else partial.
type RulePartResult struct {
	ID        string          `json:"id"`
	Coverage  string          `json:"coverage"`
	Matches   []string        `json:"matches"`
	Spans     []RuleSpan      `json:"spans"`
	Unchecked []RuleUnchecked `json:"unchecked"`
}

// ListenerRules evaluates the rules a listener binds to an app on each part,
// with the classifiers of the recipe the rules bind. The serving Router
// implements it.
type ListenerRules interface {
	EvaluateListenerRules(ctx context.Context, listener string, rules *routerconfig.VSRRules, app string, parts []RulePart) ([]RulePartResult, error)
}

// rulesInFlight holds each listener's slots, by listener name, for as long
// as the process runs: handlers are built per request.
var rulesInFlight sync.Map

func rulesSlots(listener string) chan struct{} {
	slots, _ := rulesInFlight.LoadOrStore(listener, make(chan struct{}, maxRulesInFlight))
	return slots.(chan struct{})
}

type rulesRequest struct {
	App   string     `json:"app"`
	Parts []RulePart `json:"parts"`
}

// serveRules answers POST /v1/systemone/rules: the listener's grant and key
// as for any System One call, then the bound rules of one app on each part.
// The answer is the Router's own JSON, outside the listener's codec.
func serveRules(w http.ResponseWriter, r *http.Request, config *routerconfig.RouterConfig, listener *routerconfig.Listener, rules ListenerRules) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	if r.Method != http.MethodPost {
		publicSystemOneError(w, http.StatusMethodNotAllowed, "method_not_allowed")
		return
	}
	if len(listenerSystemOneModels(config, listener)) == 0 {
		publicSystemOneError(w, http.StatusNotFound, "systemone_not_published")
		return
	}
	if !systemOneAuthorized(r, listener.APIKeys) {
		w.Header().Set("WWW-Authenticate", `Bearer realm="vllm-semantic-router"`)
		publicSystemOneError(w, http.StatusUnauthorized, "invalid_api_key")
		return
	}
	request, status, problem := readRulesRequest(w, r)
	if problem != "" {
		code := "invalid_request"
		if status == http.StatusRequestEntityTooLarge {
			code = "request_too_large"
		}
		publicSystemOneErrorMessage(w, status, code, problem)
		return
	}
	grant, _ := listener.API(apiregistry.SystemOne)
	var bound *routerconfig.VSRRules
	if grant.VSR != nil {
		bound = grant.VSR.Rules
	}
	var listed *routerconfig.VSRRulesView
	if bound != nil {
		listed = config.VSRListingOf(grant.VSR).Rules
	}
	if rules == nil || listed == nil || len(bound.App(request.App)) == 0 {
		publicSystemOneErrorMessage(w, http.StatusNotFound, "rules_not_published", fmt.Sprintf("this listener binds no rules to %s", request.App))
		return
	}
	slots := rulesSlots(listener.Name)
	select {
	case slots <- struct{}{}:
		defer func() { <-slots }()
	default:
		w.Header().Set("Retry-After", "1")
		publicSystemOneErrorMessage(w, http.StatusTooManyRequests, "overloaded", fmt.Sprintf("this listener evaluates %d rules calls at once", maxRulesInFlight))
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), min(publicWait, grant.MaxDeadline()))
	defer cancel()
	results, err := rules.EvaluateListenerRules(ctx, listener.Name, bound, request.App, request.Parts)
	if err != nil {
		publicSystemOneErrorMessage(w, http.StatusServiceUnavailable, "systemone_unavailable", "the rules' recipe is not ready")
		return
	}
	_ = json.NewEncoder(w).Encode(map[string]any{"revision": listed.Revision, "parts": results})
}

// readRulesRequest decodes a rules request and checks its bounds, returning
// the status and message of a request it refuses.
func readRulesRequest(w http.ResponseWriter, r *http.Request) (rulesRequest, int, string) {
	var request rulesRequest
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, maxRuleBody))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&request); err != nil {
		var tooLarge *http.MaxBytesError
		if errors.As(err, &tooLarge) {
			return request, http.StatusRequestEntityTooLarge, fmt.Sprintf("a rules request holds at most %d bytes", maxRuleBody)
		}
		return request, http.StatusBadRequest, "the body is not a rules request: {app, parts: [{id, text}]}"
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		var tooLarge *http.MaxBytesError
		if errors.As(err, &tooLarge) {
			return request, http.StatusRequestEntityTooLarge, fmt.Sprintf("a rules request holds at most %d bytes", maxRuleBody)
		}
		return request, http.StatusBadRequest, "the body holds more than one JSON value"
	}
	switch {
	case request.App != routerconfig.VSRAppGuard && request.App != routerconfig.VSRAppRedact:
		return request, http.StatusBadRequest, fmt.Sprintf("app %q takes no rules; use guard or redact", request.App)
	case len(request.Parts) == 0:
		return request, http.StatusBadRequest, "parts is empty"
	case len(request.Parts) > maxRuleParts:
		return request, http.StatusRequestEntityTooLarge, fmt.Sprintf("a rules request holds at most %d parts", maxRuleParts)
	}
	seen := make(map[string]bool, len(request.Parts))
	size := 0
	for _, part := range request.Parts {
		size += len(part.Text)
		switch {
		case part.ID == "" || len(part.ID) > maxRulePartID:
			return request, http.StatusBadRequest, fmt.Sprintf("each part needs an id of 1 to %d bytes", maxRulePartID)
		case seen[part.ID]:
			return request, http.StatusBadRequest, fmt.Sprintf("part id %q appears twice", part.ID)
		case size > maxRuleText:
			return request, http.StatusRequestEntityTooLarge, fmt.Sprintf("a rules request holds at most %d bytes of text", maxRuleText)
		}
		seen[part.ID] = true
	}
	return request, 0, ""
}
