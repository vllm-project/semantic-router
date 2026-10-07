// Package modelruntime is the E2E client of the built-in model runtime. It
// calls every surface of the runtime contract
// (src/model-runtime/vllm_srun/api/openapi.yaml) over a port-forward,
// for runtimes the Router attaches to, or through the Router pod, for the
// runtimes the Router manages on private Unix sockets. Waits poll readiness;
// nothing sleeps for a fixed time.
package modelruntime

// The types below mirror the subset of the runtime's OpenAPI schemas the E2E
// contracts read. Unknown fields are ignored on decode.

// ModelList is the body of GET /v1/models.
type ModelList struct {
	APIVersion string         `json:"api_version"`
	Data       []ModelCard    `json:"data"`
	Limits     *ProcessLimits `json:"limits"`
}

// ProcessLimits is what one request to a runtime process may hold.
type ProcessLimits struct {
	MaxBundleTasks  int `json:"max_bundle_tasks"`
	MaxRequestBytes int `json:"max_request_bytes"`
}

// ModelCard describes one served model.
type ModelCard struct {
	ID          string         `json:"id"`
	Family      string         `json:"family"`
	Repo        *string        `json:"repo"`
	Revision    *string        `json:"revision"`
	ModelSHA256 string         `json:"model_sha256"`
	Surfaces    []string       `json:"surfaces"`
	Heads       []HeadCard     `json:"heads"`
	Embedding   *EmbeddingCard `json:"embedding"`
	Rerank      *RerankCard    `json:"rerank"`
	Profile     string         `json:"profile"`
	Engine      string         `json:"engine"`
	Accelerator string         `json:"accelerator"`
	Device      string         `json:"device"`
	Ready       bool           `json:"ready"`
	Status      string         `json:"status"`
	Reason      string         `json:"reason"`
	Golden      GoldenStatus   `json:"golden"`
	Plugins     []PluginInfo   `json:"plugins"`
}

// HasSurface reports whether the model serves the named surface.
func (c ModelCard) HasSurface(surface string) bool {
	for _, served := range c.Surfaces {
		if served == surface {
			return true
		}
	}
	return false
}

// HeadCard describes one classify head.
type HeadCard struct {
	Name             string   `json:"name"`
	Kind             string   `json:"kind"`
	Labels           []string `json:"labels"`
	Inputs           []string `json:"inputs"`
	DefaultThreshold *float64 `json:"default_threshold"`
}

// EmbeddingCard describes an embedding model's views.
type EmbeddingCard struct {
	Dimensions []int    `json:"dimensions"`
	Layers     []int    `json:"layers"`
	Modalities []string `json:"modalities"`
	Normalized bool     `json:"normalized"`
}

// RerankCard describes a reranker's pair-scorer exits.
type RerankCard struct {
	Exits   []RerankExit `json:"exits"`
	Default RerankExit   `json:"default"`
}

// RerankExit is one pair-scorer exit.
type RerankExit struct {
	Layer     int `json:"layer"`
	Dimension int `json:"dimension"`
}

// GoldenStatus is a model's readiness check against its golden answers.
type GoldenStatus struct {
	Status string `json:"status"`
}

// PluginInfo is one active plugin with its capability descriptor.
type PluginInfo struct {
	Group        string                 `json:"group"`
	Name         string                 `json:"name"`
	Distribution *string                `json:"distribution"`
	Version      *string                `json:"version"`
	Capabilities map[string]interface{} `json:"capabilities"`
}

// Health is the body of GET /health.
type Health struct {
	APIVersion string                 `json:"api_version"`
	Status     string                 `json:"status"`
	Reason     *string                `json:"reason"`
	Models     map[string]ModelHealth `json:"models"`
}

// Liveness is the body of GET /health/live.
type Liveness struct {
	APIVersion string `json:"api_version"`
	Status     string `json:"status"`
}

// ModelHealth is one model's state in a multi-model process.
type ModelHealth struct {
	Status string  `json:"status"`
	Reason *string `json:"reason"`
}

// ErrorBody is a request-level error; item-level errors are a code string.
type ErrorBody struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// ResponseMeta reports how a response was produced.
type ResponseMeta struct {
	Revision       *string         `json:"revision"`
	ModelSHA256    string          `json:"model_sha256"`
	Profile        string          `json:"profile"`
	Engine         string          `json:"engine"`
	Device         string          `json:"device"`
	Head           string          `json:"head"`
	Representation *Representation `json:"representation"`
}

// Representation identifies an embedding space.
type Representation struct {
	ModelSHA256 string `json:"model_sha256"`
	Layer       int    `json:"layer"`
	Dimension   int    `json:"dimension"`
	Normalized  bool   `json:"normalized"`
	Modality    string `json:"modality"`
}

// InputUsage reports the tokenizer facts of one input.
type InputUsage struct {
	Tokens          int  `json:"tokens"`
	ProcessedTokens int  `json:"processed_tokens"`
	Truncated       bool `json:"truncated"`
	Windows         int  `json:"windows"`
}

// ClassifyRequest is the body of POST /v1/classify. Input is a string, a list
// of strings, or a list of ClassifyItem.
type ClassifyRequest struct {
	Model   string           `json:"model,omitempty"`
	Input   interface{}      `json:"input"`
	Head    string           `json:"head,omitempty"`
	Options *ClassifyOptions `json:"options,omitempty"`
}

// ClassifyItem is one object-form classify input.
type ClassifyItem struct {
	Text     string `json:"text,omitempty"`
	TextPair string `json:"text_pair,omitempty"`
	Context  string `json:"context,omitempty"`
	Question string `json:"question,omitempty"`
	Answer   string `json:"answer,omitempty"`
}

// ClassifyOptions are the per-request classify options.
type ClassifyOptions struct {
	Overflow   string         `json:"overflow,omitempty"`
	MaxTokens  int            `json:"max_tokens,omitempty"`
	Window     *WindowOptions `json:"window,omitempty"`
	Threshold  *float64       `json:"threshold,omitempty"`
	ReturnMeta bool           `json:"return_meta,omitempty"`
}

// WindowOptions sizes the windows of a windowed input.
type WindowOptions struct {
	Tokens  int `json:"tokens"`
	Overlap int `json:"overlap,omitempty"`
}

// ClassifyResponse is the body of a classify response.
type ClassifyResponse struct {
	Model   string           `json:"model"`
	Head    string           `json:"head"`
	Kind    string           `json:"kind"`
	Labels  []string         `json:"labels"`
	Results []ClassifyResult `json:"results"`
	Meta    *ResponseMeta    `json:"meta"`
}

// ClassifyResult is one input's classify result.
type ClassifyResult struct {
	Index         int         `json:"index"`
	Label         string      `json:"label"`
	Probabilities []float64   `json:"probabilities"`
	Scores        []float64   `json:"scores"`
	Selected      []string    `json:"selected"`
	Spans         []Span      `json:"spans"`
	Input         *InputUsage `json:"input"`
	Error         string      `json:"error"`
}

// Probability returns the probability of label in a sequence result.
func (r ClassifyResult) Probability(labels []string, label string) (float64, bool) {
	for index, name := range labels {
		if name == label && index < len(r.Probabilities) {
			return r.Probabilities[index], true
		}
	}
	return 0, false
}

// Span is one labelled span, in Unicode code points, end exclusive.
type Span struct {
	Label       string  `json:"label"`
	Start       int     `json:"start"`
	End         int     `json:"end"`
	Text        string  `json:"text"`
	Probability float64 `json:"probability"`
}

// EmbeddingsRequest is the body of POST /v1/embeddings.
type EmbeddingsRequest struct {
	Model      string      `json:"model,omitempty"`
	Input      interface{} `json:"input"`
	Dimensions int         `json:"dimensions,omitempty"`
	Layer      int         `json:"layer,omitempty"`
	InputType  string      `json:"input_type,omitempty"`
}

// EmbeddingsResponse is the body of an embeddings response.
type EmbeddingsResponse struct {
	Model string        `json:"model"`
	Data  []Embedding   `json:"data"`
	Meta  *ResponseMeta `json:"meta"`
}

// Embedding is one input's vector.
type Embedding struct {
	Index     int         `json:"index"`
	Embedding []float64   `json:"embedding"`
	Input     *InputUsage `json:"input"`
	Error     string      `json:"error"`
}

// RerankRequest is the body of POST /v1/rerank.
type RerankRequest struct {
	Model      string   `json:"model,omitempty"`
	Query      string   `json:"query"`
	Documents  []string `json:"documents"`
	TopN       int      `json:"top_n,omitempty"`
	Layer      int      `json:"layer,omitempty"`
	Dimensions int      `json:"dimensions,omitempty"`
}

// RerankResponse is the body of a rerank response.
type RerankResponse struct {
	Model   string         `json:"model"`
	Results []RerankResult `json:"results"`
}

// RerankResult is one document's relevance.
type RerankResult struct {
	Index          int     `json:"index"`
	RelevanceScore float64 `json:"relevance_score"`
	Logit          float64 `json:"logit"`
	Error          string  `json:"error"`
}

// DecisionsRequest is the body of POST /v1/decisions.
type DecisionsRequest struct {
	Model     string              `json:"model,omitempty"`
	State     interface{}         `json:"state"`
	Questions map[string]Question `json:"questions"`
}

// Question is one System One question. Criteria encode in key order, so a
// Set or Span question built from it lists its labels sorted.
type Question struct {
	Type         string            `json:"type"`
	Instructions string            `json:"instructions,omitempty"`
	Choices      []Choice          `json:"choices,omitempty"`
	Criteria     map[string]string `json:"criteria,omitempty"`
	Levels       []string          `json:"levels,omitempty"`
	Threshold    *float64          `json:"threshold,omitempty"`
}

// Choice is one option of a Choice question.
type Choice struct {
	Key         string `json:"key"`
	Description string `json:"description,omitempty"`
}

// DecisionsResponse is the body of a decisions response: Set questions answer
// in Sets, Span questions in Spans as well as in Answers.
type DecisionsResponse struct {
	Answers map[string]Answer    `json:"answers"`
	Sets    map[string]SetAnswer `json:"sets"`
	Spans   map[string][]Span    `json:"spans"`
}

// answers reports whether the response answers question id where its type
// answers.
func (r DecisionsResponse) answers(id, questionType string) bool {
	_, answered := r.Answers[id]
	switch questionType {
	case "set":
		_, ok := r.Sets[id]
		return ok
	case "span":
		_, ok := r.Spans[id]
		return answered && ok
	default:
		return answered
	}
}

// SetAnswer is one Set question's selected labels and every label's probability.
type SetAnswer struct {
	Selected      []string           `json:"selected"`
	Probabilities map[string]float64 `json:"probabilities"`
}

// Answer is one question's answer.
type Answer struct {
	Choice        string             `json:"choice"`
	Noul          *float64           `json:"noul"`
	Probabilities map[string]float64 `json:"probabilities"`
	Error         string             `json:"error"`
}

// BundleRequest is the body of POST /v1/bundle.
type BundleRequest struct {
	Tasks   []BundleTask   `json:"tasks"`
	Options *BundleOptions `json:"options,omitempty"`
}

// BundleOptions apply to every task of a bundle.
type BundleOptions struct {
	DeadlineMS float64 `json:"deadline_ms,omitempty"`
}

// BundleTask is one task; exactly one surface field is set.
type BundleTask struct {
	ID         string             `json:"id"`
	Classify   *ClassifyRequest   `json:"classify,omitempty"`
	Embeddings *EmbeddingsRequest `json:"embeddings,omitempty"`
	Rerank     *RerankRequest     `json:"rerank,omitempty"`
	Decisions  *DecisionsRequest  `json:"decisions,omitempty"`
}

// BundleResponse is the body of a bundle response, in task order.
type BundleResponse struct {
	Results []BundleResult `json:"results"`
}

// BundleResult is one task's result.
type BundleResult struct {
	ID         string              `json:"id"`
	Status     int                 `json:"status"`
	Classify   *ClassifyResponse   `json:"classify"`
	Embeddings *EmbeddingsResponse `json:"embeddings"`
	Rerank     *RerankResponse     `json:"rerank"`
	Decisions  *DecisionsResponse  `json:"decisions"`
	Error      *ErrorBody          `json:"error"`
}
