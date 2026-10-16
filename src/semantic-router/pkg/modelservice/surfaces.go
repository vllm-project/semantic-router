package modelservice

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
)

// ClassifyInput is one classify input: Text alone, Text with TextPair, or a
// grounded Answer read against Context (and an optional Question).
type ClassifyInput struct {
	Text     string
	TextPair string
	Context  string
	Question string
	Answer   string
}

// Window is a head's declared windows, or the overlapping windows a classify
// request asks for: Tokens includes special tokens, Overlap counts content
// tokens shared by neighbouring windows.
type Window = api.HeadWindow

// ClassifyRequest runs one head of a model over its inputs.
type ClassifyRequest struct {
	Head      string
	Inputs    []ClassifyInput
	Overflow  string // "" (the head's default), reject, truncate or window
	MaxTokens int
	Window    *Window
	Threshold *float64
}

// Span is a labelled span. Start and End are Unicode code points into the
// text the head read (End exclusive); callers convert them once.
type Span = api.Span

// InputUsage holds one input's tokenizer facts, including special tokens;
// Windows is set only for an input read in windows.
type InputUsage = api.InputUsage

// ClassifyWindow is one window of a windowed input, in content-token offsets.
type ClassifyWindow struct {
	Start         int
	End           int
	Probabilities []float64
	Scores        []float64
}

// ClassifyResult is one input's result; Error is set instead of values when
// this input could not be classified.
type ClassifyResult struct {
	Index         int
	Label         string
	Probabilities []float64
	Scores        []float64
	Selected      []string
	Spans         []Span
	Windows       []ClassifyWindow
	Input         *InputUsage
	Error         string
}

// ClassifyResponse holds the results in input order.
type ClassifyResponse struct {
	Model       string
	Head        string
	Kind        string
	Labels      []string
	Results     []ClassifyResult
	InputTokens int
}

// EmbedInput is one embeddings input: Text, an image data URL, or base64 WAV audio.
type EmbedInput struct {
	Text     string
	ImageURL string
	AudioWAV string
}

// EmbedRequest asks for embeddings at an optional dimension and layer exit.
type EmbedRequest struct {
	Inputs     []EmbedInput
	Dimensions int
	Layer      int
	InputType  string
	Overflow   string
	MaxTokens  int
}

// EmbedResponse holds one vector per input; Errors[i] is set when input i failed.
type EmbedResponse struct {
	Model        string
	Embeddings   [][]float32
	Inputs       []*InputUsage
	Errors       []string
	PromptTokens int
}

// RerankRequest scores documents against a query at an optional pair-scorer exit.
type RerankRequest struct {
	Query      string
	Documents  []string
	Layer      int
	Dimensions int
	Overflow   string
	MaxTokens  int
}

// RerankResult is one document's raw logit and score.
type RerankResult struct {
	Index int
	Logit float64
	Score float64
	Input *InputUsage
	Error string
}

// RerankResponse holds the results in document order.
type RerankResponse struct {
	Model       string
	Results     []RerankResult
	InputTokens int
}

// Classify runs a classify request on model, inside the context's bundle when there is one.
func (c *Client) Classify(ctx context.Context, model string, request ClassifyRequest) (ClassifyResponse, error) {
	response, _, err := c.classify(ctx, model, request)
	return response, err
}

func (c *Client) classify(ctx context.Context, model string, request ClassifyRequest) (ClassifyResponse, exchangeTiming, error) {
	body, err := encodeClassify(ctx, model, request)
	if err != nil {
		return ClassifyResponse{}, exchangeTiming{}, err
	}
	result, timing, err := c.exchange(ctx, api.BundleTask{Classify: &body})
	if err != nil {
		return ClassifyResponse{}, timing, err
	}
	if result.Classify == nil {
		return ClassifyResponse{}, timing, fmt.Errorf("%w: missing classify response body", ErrFailed)
	}
	return decodeClassify(*result.Classify), timing, nil
}

// Embed runs an embeddings request on model, inside the context's bundle when there is one.
func (c *Client) Embed(ctx context.Context, model string, request EmbedRequest) (EmbedResponse, error) {
	response, _, err := c.embed(ctx, model, request)
	return response, err
}

func (c *Client) embed(ctx context.Context, model string, request EmbedRequest) (EmbedResponse, exchangeTiming, error) {
	body, err := encodeEmbed(ctx, model, request)
	if err != nil {
		return EmbedResponse{}, exchangeTiming{}, err
	}
	result, timing, err := c.exchange(ctx, api.BundleTask{Embeddings: &body})
	if err != nil {
		return EmbedResponse{}, timing, err
	}
	if result.Embeddings == nil {
		return EmbedResponse{}, timing, fmt.Errorf("%w: missing embeddings response body", ErrFailed)
	}
	response, err := decodeEmbed(*result.Embeddings)
	return response, timing, err
}

// Rerank runs a rerank request on model, inside the context's bundle when there is one.
func (c *Client) Rerank(ctx context.Context, model string, request RerankRequest) (RerankResponse, error) {
	response, _, err := c.rerank(ctx, model, request)
	return response, err
}

func (c *Client) rerank(ctx context.Context, model string, request RerankRequest) (RerankResponse, exchangeTiming, error) {
	body, err := encodeRerank(ctx, model, request)
	if err != nil {
		return RerankResponse{}, exchangeTiming{}, err
	}
	result, timing, err := c.exchange(ctx, api.BundleTask{Rerank: &body})
	if err != nil {
		return RerankResponse{}, timing, err
	}
	if result.Rerank == nil {
		return RerankResponse{}, timing, fmt.Errorf("%w: missing rerank response body", ErrFailed)
	}
	response, err := decodeRerank(*result.Rerank, len(request.Documents))
	return response, timing, err
}

// exchange sends one surface task, through the context's bundle when there is
// one; a non-200 status becomes the package error for it. The timing is that
// of the exchange that carried the task.
func (c *Client) exchange(ctx context.Context, task api.BundleTask) (api.BundleResult, exchangeTiming, error) {
	result, timing, err := c.send(ctx, task)
	if err != nil {
		return api.BundleResult{}, timing, err
	}
	if statusErr := statusError(result.Status, result.Error); statusErr != nil {
		return api.BundleResult{}, timing, statusErr
	}
	return result, timing, nil
}

func (c *Client) send(ctx context.Context, task api.BundleTask) (api.BundleResult, exchangeTiming, error) {
	if bundle := bundleFrom(ctx); bundle != nil {
		return bundle.submit(ctx, c, task)
	}
	started := time.Now()
	switch {
	case task.Decisions != nil:
		response, err := c.api.CreateDecisionsWithResponse(ctx, *task.Decisions)
		if err != nil {
			return api.BundleResult{}, exchangeTiming{}, transportError(ctx, err)
		}
		timing := timedExchange(started, response.HTTPResponse)
		return api.BundleResult{Status: response.StatusCode(), Decisions: response.JSON200, Error: errorBody(response.JSON400, response.JSON404, response.JSON413, response.JSON422, response.JSON429, response.JSON500, response.JSON503)}, timing, nil
	case task.Classify != nil:
		response, err := c.api.CreateClassificationWithResponse(ctx, *task.Classify)
		if err != nil {
			return api.BundleResult{}, exchangeTiming{}, transportError(ctx, err)
		}
		timing := timedExchange(started, response.HTTPResponse)
		return api.BundleResult{Status: response.StatusCode(), Classify: response.JSON200, Error: errorBody(response.JSON400, response.JSON404, response.JSON413, response.JSON422, response.JSON429, response.JSON500, response.JSON503)}, timing, nil
	case task.Embeddings != nil:
		response, err := c.api.CreateEmbeddingsWithResponse(ctx, *task.Embeddings)
		if err != nil {
			return api.BundleResult{}, exchangeTiming{}, transportError(ctx, err)
		}
		timing := timedExchange(started, response.HTTPResponse)
		return api.BundleResult{Status: response.StatusCode(), Embeddings: response.JSON200, Error: errorBody(response.JSON400, response.JSON404, response.JSON413, response.JSON422, response.JSON429, response.JSON500, response.JSON503)}, timing, nil
	case task.Rerank != nil:
		response, err := c.api.CreateRerankWithResponse(ctx, *task.Rerank)
		if err != nil {
			return api.BundleResult{}, exchangeTiming{}, transportError(ctx, err)
		}
		timing := timedExchange(started, response.HTTPResponse)
		return api.BundleResult{Status: response.StatusCode(), Rerank: response.JSON200, Error: errorBody(response.JSON400, response.JSON404, response.JSON413, response.JSON422, response.JSON429, response.JSON500, response.JSON503)}, timing, nil
	default:
		return api.BundleResult{}, exchangeTiming{}, fmt.Errorf("%w: a task names no surface", ErrRejected)
	}
}

// Bundle sends tasks in one /v1/bundle call and returns their results in task order.
func (c *Client) Bundle(ctx context.Context, tasks []api.BundleTask) ([]api.BundleResult, error) {
	results, _, err := c.sendBundle(ctx, tasks)
	return results, err
}

func (c *Client) sendBundle(ctx context.Context, tasks []api.BundleTask) ([]api.BundleResult, exchangeTiming, error) {
	started := time.Now()
	response, err := c.api.CreateBundleWithResponse(ctx, api.BundleRequest{Tasks: tasks})
	if err != nil {
		return nil, exchangeTiming{}, transportError(ctx, err)
	}
	timing := timedExchange(started, response.HTTPResponse)
	if err := statusError(response.StatusCode(), errorBody(response.JSON400, response.JSON413, response.JSON500)); err != nil {
		return nil, timing, err
	}
	if response.JSON200 == nil || len(response.JSON200.Results) != len(tasks) {
		return nil, timing, fmt.Errorf("%w: bundle response does not match its tasks", ErrFailed)
	}
	return response.JSON200.Results, timing, nil
}

func remainingMillis(ctx context.Context) (*float64, error) {
	deadline, ok := ctx.Deadline()
	if !ok {
		return nil, nil
	}
	remaining := float64(time.Until(deadline).Microseconds()) / 1000.0
	if remaining <= 0 {
		return nil, context.DeadlineExceeded
	}
	return &remaining, nil
}

func optionalString(value string) *string {
	if value == "" {
		return nil
	}
	return &value
}

func optionalInt(value int) *int {
	if value == 0 {
		return nil
	}
	return &value
}

func encodeClassify(ctx context.Context, model string, request ClassifyRequest) (api.ClassifyRequest, error) {
	if len(request.Inputs) == 0 {
		return api.ClassifyRequest{}, fmt.Errorf("%w: classify needs at least one input", ErrRejected)
	}
	deadline, err := remainingMillis(ctx)
	if err != nil {
		return api.ClassifyRequest{}, err
	}
	items := make(api.ClassifyItemList, len(request.Inputs))
	for index, input := range request.Inputs {
		items[index] = api.ClassifyItem{
			Text: optionalString(input.Text), TextPair: optionalString(input.TextPair),
			Context: optionalString(input.Context), Question: optionalString(input.Question), Answer: optionalString(input.Answer),
		}
	}
	var inputs api.ClassifyInput
	if err := inputs.FromClassifyItemList(items); err != nil {
		return api.ClassifyRequest{}, fmt.Errorf("%w: %w", ErrRejected, err)
	}
	options := api.ClassifyOptions{DeadlineMs: deadline, MaxTokens: optionalInt(request.MaxTokens), Threshold: request.Threshold}
	if request.Overflow != "" {
		overflow := api.ClassifyOptionsOverflow(request.Overflow)
		options.Overflow = &overflow
	}
	if request.Window != nil {
		overlap := request.Window.Overlap
		options.Window = &api.WindowOptions{Tokens: request.Window.Tokens, Overlap: &overlap}
	}
	return api.ClassifyRequest{Model: optionalString(model), Head: optionalString(request.Head), Input: inputs, Options: &options}, nil
}

func encodeEmbed(ctx context.Context, model string, request EmbedRequest) (api.EmbeddingsRequest, error) {
	if len(request.Inputs) == 0 {
		return api.EmbeddingsRequest{}, fmt.Errorf("%w: embeddings need at least one input", ErrRejected)
	}
	deadline, err := remainingMillis(ctx)
	if err != nil {
		return api.EmbeddingsRequest{}, err
	}
	parts := make(api.ContentPartList, len(request.Inputs))
	for index, input := range request.Inputs {
		switch {
		case input.ImageURL != "":
			parts[index] = api.ContentPart{Type: api.ImageUrl, ImageUrl: &api.ImagePart{Url: input.ImageURL}}
		case input.AudioWAV != "":
			format := "wav"
			parts[index] = api.ContentPart{Type: api.InputAudio, InputAudio: &api.AudioPart{Data: input.AudioWAV, Format: &format}}
		default:
			text := input.Text
			parts[index] = api.ContentPart{Type: api.Text, Text: &text}
		}
	}
	var inputs api.EmbeddingsInput
	if err := inputs.FromContentPartList(parts); err != nil {
		return api.EmbeddingsRequest{}, fmt.Errorf("%w: %w", ErrRejected, err)
	}
	options := api.EmbeddingsOptions{DeadlineMs: deadline, MaxTokens: optionalInt(request.MaxTokens)}
	if request.Overflow != "" {
		overflow := api.EmbeddingsOptionsOverflow(request.Overflow)
		options.Overflow = &overflow
	}
	encoding := api.EmbeddingsRequestEncodingFormat("base64")
	body := api.EmbeddingsRequest{
		Model: optionalString(model), Input: inputs, Dimensions: optionalInt(request.Dimensions),
		Layer: optionalInt(request.Layer), EncodingFormat: &encoding, Options: &options,
	}
	if request.InputType != "" {
		inputType := api.EmbeddingsRequestInputType(request.InputType)
		body.InputType = &inputType
	}
	return body, nil
}

func encodeRerank(ctx context.Context, model string, request RerankRequest) (api.RerankRequest, error) {
	if len(request.Documents) == 0 {
		return api.RerankRequest{}, fmt.Errorf("%w: rerank needs at least one document", ErrRejected)
	}
	deadline, err := remainingMillis(ctx)
	if err != nil {
		return api.RerankRequest{}, err
	}
	options := api.RerankOptions{DeadlineMs: deadline, MaxTokens: optionalInt(request.MaxTokens)}
	if request.Overflow != "" {
		overflow := api.RerankOptionsOverflow(request.Overflow)
		options.Overflow = &overflow
	}
	return api.RerankRequest{
		Model: optionalString(model), Query: request.Query, Documents: request.Documents,
		Layer: optionalInt(request.Layer), Dimensions: optionalInt(request.Dimensions), Options: &options,
	}, nil
}

func floats(values *[]float64) []float64 {
	if values == nil {
		return nil
	}
	return append([]float64(nil), (*values)...)
}

func finite(values ...[]float64) bool {
	for _, list := range values {
		for _, value := range list {
			if math.IsNaN(value) || math.IsInf(value, 0) {
				return false
			}
		}
	}
	return true
}

func decodeClassify(body api.ClassifyResponse) ClassifyResponse {
	decoded := ClassifyResponse{Model: body.Model, Head: body.Head, Kind: string(body.Kind), Labels: body.Labels, InputTokens: body.Usage.InputTokens}
	for _, result := range body.Results {
		item := ClassifyResult{Index: result.Index, Probabilities: floats(result.Probabilities), Scores: floats(result.Scores), Input: result.Input}
		if result.Error != nil {
			item.Error = string(*result.Error)
			decoded.Results = append(decoded.Results, item)
			continue
		}
		if result.Label != nil {
			item.Label = *result.Label
		}
		if result.Selected != nil {
			item.Selected = append([]string(nil), (*result.Selected)...)
		}
		if result.Spans != nil {
			item.Spans = append([]Span(nil), (*result.Spans)...)
		}
		if result.Windows != nil {
			for _, window := range *result.Windows {
				item.Windows = append(item.Windows, ClassifyWindow{Start: window.Start, End: window.End, Probabilities: floats(window.Probabilities), Scores: floats(window.Scores)})
			}
		}
		valid := finite(item.Probabilities, item.Scores)
		for _, span := range item.Spans {
			valid = valid && finite([]float64{span.Probability})
		}
		for _, window := range item.Windows {
			valid = valid && finite(window.Probabilities, window.Scores)
		}
		if !valid {
			item = ClassifyResult{Index: result.Index, Error: "invalid_model_output"}
		}
		decoded.Results = append(decoded.Results, item)
	}
	return decoded
}

func decodeEmbed(body api.EmbeddingsResponse) (EmbedResponse, error) {
	decoded := EmbedResponse{
		Model:        body.Model,
		Embeddings:   make([][]float32, len(body.Data)),
		Inputs:       make([]*InputUsage, len(body.Data)),
		Errors:       make([]string, len(body.Data)),
		PromptTokens: body.Usage.PromptTokens,
	}
	for _, item := range body.Data {
		if item.Index < 0 || item.Index >= len(body.Data) {
			return EmbedResponse{}, fmt.Errorf("%w: embedding index %d is out of range", ErrFailed, item.Index)
		}
		decoded.Inputs[item.Index] = item.Input
		if item.Error != nil {
			decoded.Errors[item.Index] = string(*item.Error)
			continue
		}
		if item.Embedding == nil {
			return EmbedResponse{}, fmt.Errorf("%w: embedding %d has no vector", ErrFailed, item.Index)
		}
		vector, err := decodeVector(*item.Embedding)
		if err != nil {
			return EmbedResponse{}, err
		}
		decoded.Embeddings[item.Index] = vector
	}
	return decoded, nil
}

// decodeVector reads a float list or a base64 string of little-endian float32 values.
func decodeVector(value api.EmbeddingVector) ([]float32, error) {
	raw, err := value.MarshalJSON()
	if err != nil || len(raw) == 0 {
		return nil, fmt.Errorf("%w: embedding has an unknown encoding", ErrFailed)
	}
	if raw[0] != '"' {
		vector, decodeErr := value.AsFloatVector()
		if decodeErr != nil {
			return nil, fmt.Errorf("%w: embedding holds a non-number", ErrFailed)
		}
		return checkVector(vector)
	}
	packed, err := value.AsBase64Vector()
	if err != nil || len(packed)%4 != 0 {
		return nil, fmt.Errorf("%w: embedding is not base64 float32", ErrFailed)
	}
	vector := make([]float32, len(packed)/4)
	for index := range vector {
		vector[index] = math.Float32frombits(binary.LittleEndian.Uint32(packed[index*4:]))
	}
	return checkVector(vector)
}

func checkVector(vector []float32) ([]float32, error) {
	if len(vector) == 0 {
		return nil, fmt.Errorf("%w: embedding is empty", ErrFailed)
	}
	for _, value := range vector {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, fmt.Errorf("%w: embedding is not finite", ErrFailed)
		}
	}
	return vector, nil
}

func decodeRerank(body api.RerankResponse, documents int) (RerankResponse, error) {
	decoded := RerankResponse{Model: body.Model, Results: make([]RerankResult, documents), InputTokens: body.Usage.InputTokens}
	seen := make([]bool, documents)
	for _, result := range body.Results {
		if result.Index < 0 || result.Index >= documents || seen[result.Index] {
			return RerankResponse{}, fmt.Errorf("%w: rerank result index %d is invalid", ErrFailed, result.Index)
		}
		seen[result.Index] = true
		item := RerankResult{Index: result.Index, Input: result.Input}
		switch {
		case result.Error != nil:
			item.Error = string(*result.Error)
		case result.Logit == nil || !finite([]float64{*result.Logit}):
			item.Error = "invalid_model_output"
		default:
			item.Logit = *result.Logit
			if result.RelevanceScore != nil {
				item.Score = *result.RelevanceScore
			}
		}
		decoded.Results[result.Index] = item
	}
	for index, ok := range seen {
		if !ok {
			return RerankResponse{}, fmt.Errorf("%w: rerank result for document %d is missing", ErrFailed, index)
		}
	}
	return decoded, nil
}

func errorBody(responses ...*api.Error) *api.ErrorBody {
	for _, response := range responses {
		if response != nil {
			return &response.Error
		}
	}
	return nil
}

// statusError maps a surface status to the package's errors (nil on 200).
func statusError(status int, body *api.ErrorBody) error {
	message := http.StatusText(status)
	if body != nil {
		message = fmt.Sprintf("%s (%s)", body.Code, body.Message)
	}
	switch status {
	case http.StatusOK:
		return nil
	case http.StatusTooManyRequests:
		return ErrOverloaded
	case http.StatusServiceUnavailable:
		return ErrUnavailable
	case http.StatusBadRequest, http.StatusNotFound, http.StatusRequestEntityTooLarge, http.StatusUnprocessableEntity:
		return fmt.Errorf("%w: %s", ErrRejected, message)
	default:
		return fmt.Errorf("%w: %s", ErrFailed, message)
	}
}

func transportError(ctx context.Context, err error) error {
	if ctx.Err() != nil {
		return ctx.Err()
	}
	return fmt.Errorf("%w: %w", ErrFailed, err)
}
