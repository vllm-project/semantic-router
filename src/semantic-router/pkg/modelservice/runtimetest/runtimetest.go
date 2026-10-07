// Package runtimetest serves a fake model runtime for router tests. It speaks
// the runtime contract (health with per-model states, model cards, classify,
// decisions with Set and Span answers, embeddings, rerank and bundles) with
// deterministic answers, so tests exercise the real client, bundles and typed
// bindings without a Python process.
//
// Text is read as whitespace-separated words, each one token, plus two special
// tokens per window. Sequence heads put 0.9 on the first label the text names
// (else on the first label) and scores heads score every named label 0.9;
// token heads mark every word equal to an entity label (B-PERSON marks
// "person"); grounded heads mark answer words absent from the context.
package runtimetest

import (
	"encoding/json"
	"net/http"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
)

// Head is one classify head of a fake model.
type Head struct {
	Name                 string
	Kind                 string
	Labels               []string
	Inputs               []string
	Thresholds           []float64
	OperatingPointSHA256 string
}

// Model is one served model: classify with heads, embeddings with an
// Embedder, rerank with a Reranker, otherwise decisions.
type Model struct {
	ID             string
	Heads          []Head
	Embedding      *Embedder
	Rerank         *Reranker
	Labelled       *Labelled
	MaxInputTokens int
	Device         string
}

// Runtime is the fake runtime state; Handler serves it.
type Runtime struct {
	mu         sync.Mutex
	models     map[string]Model
	ready      map[string]bool
	failed     map[string]string
	apiVersion string
	delay      time.Duration
	bundles    int
	tasks      int
	surfaces   map[string]int
	limits     api.ProcessLimits
	timing     string
}

// APIVersion is the contract version the fake serves unless SetAPIVersion
// changes it.
const APIVersion = "2.0.0"

// defaultLimits are the runtime's default process limits. Like the runtime,
// the fake reports its limits in /v1/models and refuses a larger bundle or
// request body whole.
var defaultLimits = api.ProcessLimits{MaxBundleTasks: 64, MaxRequestBytes: 8 << 20}

// maxInputs is the classify input cap of a task_heads card.
var maxInputs = 2048

// New serves models, all ready.
func New(models ...Model) *Runtime {
	r := &Runtime{models: make(map[string]Model), ready: make(map[string]bool), failed: make(map[string]string), apiVersion: APIVersion, surfaces: make(map[string]int), limits: defaultLimits}
	for _, model := range models {
		if model.MaxInputTokens == 0 {
			model.MaxInputTokens = 8192
		}
		if model.Device == "" {
			model.Device = "cpu"
		}
		r.models[model.ID] = model
		r.ready[model.ID] = true
	}
	return r
}

// SetReady changes one model's readiness and clears a load failure.
func (r *Runtime) SetReady(id string, ready bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.ready[id] = ready
	delete(r.failed, id)
}

// SetFailed reports one model as failed to load, with a reason, as the
// runtime does when loading raised.
func (r *Runtime) SetFailed(id, reason string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.ready[id] = false
	r.failed[id] = reason
}

// SetAPIVersion changes the contract version /health and /v1/models report.
func (r *Runtime) SetAPIVersion(version string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.apiVersion = version
}

// SetDelay delays every surface answer (deadline tests).
func (r *Runtime) SetDelay(delay time.Duration) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.delay = delay
}

// Bundles reports how many /v1/bundle calls arrived and how many tasks they carried.
func (r *Runtime) Bundles() (calls, tasks int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.bundles, r.tasks
}

// Calls reports direct calls to a surface (classify, decisions, embeddings,
// rerank), outside bundles.
func (r *Runtime) Calls(surface string) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.surfaces[surface]
}

// SetLimits changes the limits the fake reports and enforces, as the
// runtime's --max-bundle-tasks and --max-request-bytes do.
func (r *Runtime) SetLimits(limits api.ProcessLimits) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.limits = limits
}

func (r *Runtime) processLimits() api.ProcessLimits {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.limits
}

// SetServerTiming sets the Server-Timing value of every surface and bundle
// response, as the runtime reports its own time; empty sends none.
func (r *Runtime) SetServerTiming(value string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.timing = value
}

func (r *Runtime) serverTiming() string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.timing
}

// Handler serves the contract.
func (r *Runtime) Handler() http.Handler {
	mux := r.routes()
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if timing := r.serverTiming(); timing != "" && req.Method == http.MethodPost {
			w.Header().Set("Server-Timing", timing)
		}
		limit := int64(r.processLimits().MaxRequestBytes)
		if req.ContentLength > limit {
			write(w, http.StatusRequestEntityTooLarge, nil, &api.ErrorBody{Code: "request_too_large", Message: "request body over the process limit"})
			return
		}
		req.Body = http.MaxBytesReader(w, req.Body, limit)
		mux.ServeHTTP(w, req)
	})
}

func (r *Runtime) routes() *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", r.health)
	mux.HandleFunc("/v1/models", r.listModels)
	mux.HandleFunc("/v1/classify", func(w http.ResponseWriter, req *http.Request) {
		var body api.ClassifyRequest
		if !decode(w, req, &body) {
			return
		}
		r.count("classify")
		status, response, errBody := r.classify(body)
		write(w, status, response, errBody)
	})
	mux.HandleFunc("/v1/decisions", func(w http.ResponseWriter, req *http.Request) {
		var body api.DecisionRequest
		if !decode(w, req, &body) {
			return
		}
		r.count("decisions")
		status, response, errBody := r.decide(body)
		write(w, status, response, errBody)
	})
	mux.HandleFunc("/v1/embeddings", func(w http.ResponseWriter, req *http.Request) {
		var body api.EmbeddingsRequest
		if !decode(w, req, &body) {
			return
		}
		r.count("embeddings")
		status, response, errBody := r.embeddings(body)
		write(w, status, response, errBody)
	})
	mux.HandleFunc("/v1/rerank", func(w http.ResponseWriter, req *http.Request) {
		var body api.RerankRequest
		if !decode(w, req, &body) {
			return
		}
		r.count("rerank")
		status, response, errBody := r.rerank(body)
		write(w, status, response, errBody)
	})
	mux.HandleFunc("/v1/bundle", r.bundle)
	return mux
}

func (r *Runtime) count(surface string) {
	r.mu.Lock()
	r.surfaces[surface]++
	delay := r.delay
	r.mu.Unlock()
	time.Sleep(delay)
}

func (r *Runtime) health(w http.ResponseWriter, _ *http.Request) {
	r.mu.Lock()
	models := make(map[string]api.ModelHealth, len(r.models))
	everyReady := true
	for id := range r.models {
		health := api.ModelHealth{Status: api.ModelHealthStatus("ready")}
		if reason, failed := r.failed[id]; failed {
			health.Status, health.Reason = api.ModelHealthStatus("failed"), &reason
		} else if !r.ready[id] {
			health.Status = api.ModelHealthStatus("loading")
		}
		everyReady = everyReady && health.Status == "ready"
		models[id] = health
	}
	apiVersion := r.apiVersion
	r.mu.Unlock()
	health := api.Health{ApiVersion: apiVersion, Status: api.HealthStatus("ready"), Models: &models}
	status := http.StatusOK
	if !everyReady {
		health.Status, status = api.HealthStatus("degraded"), http.StatusServiceUnavailable
	}
	write(w, status, health, nil)
}

func (r *Runtime) listModels(w http.ResponseWriter, _ *http.Request) {
	r.mu.Lock()
	ids := make([]string, 0, len(r.models))
	for id := range r.models {
		ids = append(ids, id)
	}
	slices.Sort(ids)
	cards := make([]api.ModelCard, 0, len(ids))
	for _, id := range ids {
		cards = append(cards, r.card(r.models[id], r.ready[id]))
	}
	apiVersion := r.apiVersion
	r.mu.Unlock()
	write(w, http.StatusOK, api.ModelList{ApiVersion: apiVersion, Object: "list", Data: cards, Limits: r.processLimits()}, nil)
}

func (r *Runtime) card(model Model, ready bool) api.ModelCard {
	sha := strings.Repeat("0", 64-len(model.ID)%64) + strings.Repeat("a", len(model.ID)%64)
	device, dtype, profile := model.Device, "float32", "exact"
	limits := api.ModelLimits{MaxInputTokens: &model.MaxInputTokens, MaxInputs: &maxInputs}
	card := api.ModelCard{Id: model.ID, Object: "model", Family: "task_heads", Ready: ready, ModelSha256: &sha, Device: &device, Dtype: &dtype, Profile: &profile, Limits: &limits}
	switch {
	case model.Embedding != nil:
		card.Surfaces, card.Embedding = []string{"embeddings"}, embeddingCard(model.Embedding)
		return card
	case model.Rerank != nil:
		card.Surfaces, card.Rerank = []string{"rerank"}, rerankCard(model.Rerank)
		return card
	case len(model.Heads) == 0:
		card.Family, card.Surfaces = "decision2", []string{"decisions"}
		if model.Labelled != nil {
			card.Family = "vela2"
		}
		types := questionTypes(model)
		card.QuestionTypes = &types
		if names := presets(model); len(names) > 0 {
			card.Presets = &names
		}
		return card
	}
	card.Surfaces = []string{"classify"}
	heads := make([]api.HeadCard, len(model.Heads))
	for i, head := range model.Heads {
		heads[i] = api.HeadCard{Name: head.Name, Kind: api.HeadCardKind(head.Kind), Labels: head.Labels}
		if len(head.Inputs) > 0 {
			inputs := slices.Clone(head.Inputs)
			heads[i].Inputs = &inputs
		}
		if len(head.Thresholds) > 0 {
			thresholds := slices.Clone(head.Thresholds)
			reduction := api.Max
			heads[i].Thresholds, heads[i].Reduction = &thresholds, &reduction
		}
		if head.OperatingPointSHA256 != "" {
			digest := head.OperatingPointSHA256
			heads[i].OperatingPointSha256 = &digest
		}
	}
	card.Heads = &heads
	return card
}

func (r *Runtime) bundle(w http.ResponseWriter, req *http.Request) {
	var body api.BundleRequest
	if !decode(w, req, &body) {
		return
	}
	r.mu.Lock()
	r.bundles++
	r.tasks += len(body.Tasks)
	delay := r.delay
	r.mu.Unlock()
	if len(body.Tasks) > r.processLimits().MaxBundleTasks {
		write(w, http.StatusRequestEntityTooLarge, nil, &api.ErrorBody{Code: "request_too_large", Message: "too many bundle tasks"})
		return
	}
	time.Sleep(delay)
	results := make([]api.BundleResult, len(body.Tasks))
	for i, task := range body.Tasks {
		result := api.BundleResult{Id: task.Id}
		switch {
		case task.Classify != nil:
			status, response, errBody := r.classify(*task.Classify)
			result.Status, result.Error = status, errBody
			if status == http.StatusOK {
				result.Classify = &response
			}
		case task.Decisions != nil:
			status, response, errBody := r.decide(*task.Decisions)
			result.Status, result.Error = status, errBody
			if status == http.StatusOK {
				result.Decisions = &response
			}
		case task.Embeddings != nil:
			status, response, errBody := r.embeddings(*task.Embeddings)
			result.Status, result.Error = status, errBody
			if status == http.StatusOK {
				result.Embeddings = &response
			}
		case task.Rerank != nil:
			status, response, errBody := r.rerank(*task.Rerank)
			result.Status, result.Error = status, errBody
			if status == http.StatusOK {
				result.Rerank = &response
			}
		default:
			result.Status, result.Error = http.StatusUnprocessableEntity, &api.ErrorBody{Code: "unsupported_surface", Message: "fake runtime"}
		}
		results[i] = result
	}
	write(w, http.StatusOK, api.BundleResponse{Results: results}, nil)
}

func (r *Runtime) model(id *string, surface string) (Model, int, *api.ErrorBody) {
	r.mu.Lock()
	defer r.mu.Unlock()
	name := ""
	if id != nil {
		name = *id
	}
	if name == "" && len(r.models) == 1 {
		for only := range r.models {
			name = only
		}
	}
	model, ok := r.models[name]
	switch {
	case !ok:
		return Model{}, http.StatusNotFound, &api.ErrorBody{Code: "model_not_found", Message: name}
	case !r.ready[name]:
		return Model{}, http.StatusServiceUnavailable, &api.ErrorBody{Code: "not_ready", Message: name}
	case !serves(model, surface):
		return Model{}, http.StatusUnprocessableEntity, &api.ErrorBody{Code: "unsupported_surface", Message: surface}
	}
	return model, http.StatusOK, nil
}

func (r *Runtime) decide(body api.DecisionRequest) (int, api.DecisionResponse, *api.ErrorBody) {
	model, status, errBody := r.model(body.Model, "decisions")
	if status != http.StatusOK {
		return status, api.DecisionResponse{}, errBody
	}
	revision := strings.Repeat("a", 40)
	if model.Labelled != nil {
		response := r.decideLabelled(model, body)
		response.Meta = &api.ResponseMeta{Revision: &revision}
		return http.StatusOK, response, nil
	}
	answers := make(map[string]api.Answer, len(body.Questions))
	for id, question := range body.Questions {
		answers[id] = answer(question)
		if question.Preset != nil || (question.Type != nil && !slices.Contains(SystemOneTypes, *question.Type)) {
			answers[id] = api.Answer{Type: question.Type, Error: itemError("invalid_question")}
		}
	}
	return http.StatusOK, api.DecisionResponse{Model: model.ID, Answers: answers, Usage: api.Usage{InputTokens: 10}, Meta: &api.ResponseMeta{Revision: &revision}}, nil
}

// answer: noul 0.8, the first choice at 0.7, the middle score level.
func answer(question api.Question) api.Answer {
	kind := ""
	if question.Type != nil {
		kind = *question.Type
	}
	result := api.Answer{Type: &kind}
	switch kind {
	case "noul":
		value := 0.8
		result.Noul = &value
	case "score":
		levels := 0
		if question.Levels != nil {
			levels = len(*question.Levels)
		}
		score, confidence := float64(levels-1)/2, 0.5
		result.Score, result.Confidence = &score, &confidence
	default:
		if question.Choices != nil && len(*question.Choices) > 0 {
			choices := *question.Choices
			probabilities := make(map[string]float64, len(choices))
			for _, choice := range choices {
				probabilities[choice.Key] = 0.3 / float64(max(len(choices)-1, 1))
			}
			probabilities[choices[0].Key] = 0.7
			confidence := 0.7
			result.Choice, result.Probabilities, result.Confidence = &choices[0].Key, &probabilities, &confidence
		}
	}
	return result
}

func decode(w http.ResponseWriter, req *http.Request, body any) bool {
	if err := json.NewDecoder(req.Body).Decode(body); err != nil {
		write(w, http.StatusBadRequest, nil, &api.ErrorBody{Code: "invalid_request", Message: err.Error()})
		return false
	}
	return true
}

func write(w http.ResponseWriter, status int, body any, errBody *api.ErrorBody) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if errBody != nil {
		_ = json.NewEncoder(w).Encode(api.ErrorResponse{Error: *errBody})
		return
	}
	_ = json.NewEncoder(w).Encode(body)
}

// words splits text into word tokens with their code-point offsets.
type word struct {
	text       string
	start, end int
}

func words(text string) []word {
	var out []word
	start, point, inWord := 0, 0, false
	for _, character := range text {
		space := character == ' ' || character == '\n' || character == '\t'
		switch {
		case !space && !inWord:
			start, inWord = point, true
		case space && inWord:
			out = append(out, wordAt(text, start, point))
			inWord = false
		}
		point++
	}
	if inWord {
		out = append(out, wordAt(text, start, point))
	}
	return out
}

func wordAt(text string, start, end int) word {
	byteStart, byteEnd, point := len(text), len(text), 0
	for offset := range text {
		if point == start {
			byteStart = offset
		}
		if point == end {
			byteEnd = offset
			break
		}
		point++
	}
	return word{text: text[byteStart:byteEnd], start: start, end: end}
}
