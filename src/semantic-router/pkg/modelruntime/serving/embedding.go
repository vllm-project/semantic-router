package serving

import (
	"cmp"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync/atomic"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

const (
	embeddingContract = "embedding.v1"
	embeddingTaskID   = "embedding_inputs.v1"
	// embeddingCacheEnv sizes the process-wide embedding cache in MiB (0 disables it).
	embeddingCacheEnv       = "VLLM_SR_EMBEDDING_CACHE_MB"
	defaultEmbeddingCacheMB = 64
)

// sharedVectors is the process-wide content-hash embedding cache. Its keys
// carry a package digest or preparation namespace plus numerics. Digest-backed
// models serving the same representation share vectors across consumers,
// deployments and router generations; unhashed models share only within one
// preparation, so a changed model never reuses another model's vectors.
var sharedVectors = embedding.NewVectorCache(embeddingCacheBytes())

// Without a digest, isolate every preparation: matching card IDs, revisions
// and dimensions cannot identify replacements. The counter is process-local,
// like the vector and request caches.
var unhashedEmbeddingNamespace atomic.Uint64

func embeddingCacheBytes() int {
	megabytes := defaultEmbeddingCacheMB
	if value, err := strconv.Atoi(strings.TrimSpace(os.Getenv(embeddingCacheEnv))); err == nil && value >= 0 {
		megabytes = value
	}
	return megabytes << 20
}

// embeddingRequest is one embeddings call: inputs of any served modality at
// one output view.
type embeddingRequest struct {
	Inputs    []modelservice.EmbedInput
	Options   embedding.Options
	FullInput bool
}

// EmbeddingProvider serves one embedding binding through a model_runtime
// deployment. It implements the embedding package's provider interfaces for
// text, image and audio input, dimension and layer views, and representation
// identities. Every input goes through the shared vector cache, so repeated
// and concurrent embeddings of the same content cost one runtime task.
type EmbeddingProvider struct {
	call           *binding.Resolved[embeddingRequest, []tasks.EmbeddingResult]
	recipe         string
	options        embedding.Options
	info           embedding.ModelInfo
	representation embedding.RuntimeDescriptor
	numerics       string
	maxInputs      int
	fullDimension  int
	closed         atomic.Bool
}

// Embedding prepares a binding on a model_runtime deployment that serves
// embeddings. dimension and layer select the binding's default view (zero:
// the model's full output and last layer); both must be ones the card declares.
func (r *Runtime) Embedding(ctx context.Context, spec config.ResolvedModelBinding, dimension, layer int) (_ *EmbeddingProvider, callErr error) {
	defer func() { observePreparationFailure(spec, callErr) }()
	ctx, cancel := preparationContext(ctx)
	defer cancel()
	card, err := r.card(ctx, spec)
	if err != nil {
		return nil, err
	}
	if !card.Serves("embeddings") || card.Embedding == nil {
		return nil, fmt.Errorf("%w: deployment %q does not serve embeddings", binding.ErrCapability, spec.Binding.Deployment)
	}
	view := embedding.Options{Dimension: dimension, Layer: layer}
	if err = checkView(card, view); err != nil {
		return nil, err
	}
	task, err := r.embeddingTask()
	if err != nil {
		return nil, err
	}
	input := spec.Deployment.WithDefaults().Input
	template := modelservice.EmbedRequest{Overflow: embeddingOverflow(input.Overflow), MaxTokens: input.MaxTokens}
	deployment, services := spec.Binding.Deployment, r.services
	embed := func(ctx context.Context, call embeddingRequest) ([]tasks.EmbeddingResult, error) {
		request := template
		request.Inputs, request.Dimensions, request.Layer = call.Inputs, call.Options.Dimension, call.Options.Layer
		if call.FullInput {
			request.Overflow = "reject"
		}
		response, embedErr := services.Embed(ctx, deployment, request)
		if embedErr != nil {
			return nil, embedErr
		}
		return embeddingResults(card.ID, response, len(call.Inputs), call.FullInput)
	}
	// One warmup through the transport, decoding and validation fixes the
	// default view's dimension before the binding is published.
	warmup := embeddingRequest{Inputs: []modelservice.EmbedInput{{Text: "warmup"}}, Options: view}
	warm, err := embed(ctx, warmup)
	if err == nil {
		err = validateEmbeddingResults(warmup, warm)
	}
	if err != nil {
		return nil, fmt.Errorf("warm up embedding deployment %q: %w", deployment, err)
	}
	width := len(warm[0].Embedding)
	capability := binding.Capability{
		Contract: spec.Binding.Contract, Provider: Provider, Device: card.Device, Precision: card.Dtype,
		Limits: binding.Limits{ModelTokens: card.MaxInputTokens, DeploymentTokens: input.MaxTokens, Overflow: template.Overflow},
		Embedding: &binding.EmbeddingCapability{
			AvailableDimensions: slices.Clone(card.Embedding.Dimensions), Dimension: width, Layer: layer,
			Pooling: card.Embedding.Pooling, Normalization: normalization(card), Modalities: modalities(card),
		},
	}
	if slices.Contains(capability.Embedding.Modalities, "audio") {
		capability.Embedding.Audio = &binding.AudioCapability{MaxSampleRate: 384000, MaxSeconds: 30, MaxChannels: 8, Layout: "channel-major-pcm"}
	}
	resource, err := r.acquire(ctx, spec, card)
	if err != nil {
		return nil, err
	}
	call, err := task.Resolve(taskIdentity(spec), capability, resource, func(ctx context.Context, _ io.Closer, request embeddingRequest) ([]tasks.EmbeddingResult, error) {
		return embed(ctx, request)
	})
	if err != nil {
		_ = resource.Close()
		return nil, err
	}
	call.Ready()
	full := width
	if view.Dimension > 0 {
		full = slices.Max(card.Embedding.Dimensions)
	}
	provider := &EmbeddingProvider{
		call: call, recipe: string(spec.Recipe), options: view, numerics: numerics(card, capability.Limits), maxInputs: card.MaxInputs, fullDimension: full,
		representation: embedding.RuntimeDescriptor{
			ModelType: cmp.Or(card.Repo, card.ID), Runtime: Provider, MaxSequenceLength: capability.Limits.EffectiveTokens(), PoolingContract: poolingContract(card),
			Artifacts: []embedding.ArtifactDigest{{Role: "package", SHA256: card.ModelSHA256}},
		},
		info: embedding.ModelInfo{
			Artifact: cmp.Or(card.Repo, card.ID), Backend: Provider, Dimension: width, Dimensions: slices.Clone(card.Embedding.Dimensions),
			Layers: slices.Clone(card.Embedding.Layers), MaxTokens: capability.Limits.EffectiveTokens(), Pooling: card.Embedding.Pooling,
			Normalization: normalization(card), Modalities: modalities(card), Audio: capability.Embedding.Audio,
		},
	}
	if card.ModelSHA256 == "" {
		provider.numerics += fmt.Sprintf(":unhashed=%d", unhashedEmbeddingNamespace.Add(1))
	}
	return provider, nil
}

func (r *Runtime) embeddingTask() (*binding.Task[embeddingRequest, []tasks.EmbeddingResult], error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if task, err := binding.Lookup[embeddingRequest, []tasks.EmbeddingResult](r.registry, embeddingTaskID); err == nil {
		return task, nil
	}
	return binding.RegisterTask(r.registry, embeddingTaskID, embeddingContract, validateEmbeddingRequest, validateEmbeddingResults)
}

func validateEmbeddingRequest(request embeddingRequest) error {
	if len(request.Inputs) == 0 {
		return fmt.Errorf("at least one embedding input is required")
	}
	if request.Options.Dimension < 0 || request.Options.Layer < 0 {
		return fmt.Errorf("embedding dimension and layer must be nonnegative")
	}
	for _, input := range request.Inputs {
		switch {
		case input.ImageURL != "" || input.AudioWAV != "":
			if input.Text != "" {
				return fmt.Errorf("an embedding input has one modality")
			}
		case !utf8.ValidString(input.Text):
			return fmt.Errorf("embedding text must be valid UTF-8")
		default:
			if err := validateText(input.Text); err != nil {
				return err
			}
		}
	}
	return nil
}

func validateEmbeddingResults(request embeddingRequest, results []tasks.EmbeddingResult) error {
	if len(results) != len(request.Inputs) {
		return fmt.Errorf("embedding results do not match the inputs")
	}
	for _, result := range results {
		if request.FullInput {
			if result.Input == nil {
				return fmt.Errorf("%w: embedding response omitted input coverage", binding.ErrCapability)
			}
			if result.Input.Truncated || result.Input.ProcessedTokens != result.Input.OriginalTokens {
				return fmt.Errorf("%w: embedding requires complete input coverage", binding.ErrInputLimit)
			}
			if result.Input.OriginalTokens <= 0 {
				return fmt.Errorf("%w: embedding response omitted token counts", binding.ErrCapability)
			}
		}
		if len(result.Embedding) == 0 {
			return fmt.Errorf("embedding vector is empty")
		}
		if request.Options.Dimension > 0 && len(result.Embedding) != request.Options.Dimension {
			return fmt.Errorf("embedding dimension %d differs from requested %d", len(result.Embedding), request.Options.Dimension)
		}
		for _, value := range result.Embedding {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return fmt.Errorf("embedding vector is not finite")
			}
		}
	}
	return nil
}

func embeddingResults(model string, response modelservice.EmbedResponse, inputs int, fullInput bool) ([]tasks.EmbeddingResult, error) {
	if len(response.Embeddings) != inputs {
		return nil, fmt.Errorf("%w: %d embeddings for %d inputs", binding.ErrInvalidResult, len(response.Embeddings), inputs)
	}
	results := make([]tasks.EmbeddingResult, inputs)
	for i := range results {
		if i < len(response.Errors) && response.Errors[i] != "" {
			return nil, itemError(response.Errors[i])
		}
		results[i] = tasks.EmbeddingResult{Embedding: response.Embeddings[i], ModelType: model}
		if i < len(response.Inputs) {
			usage := response.Inputs[i]
			// The task projection keeps token counts but not the worker's lower
			// bound flag. Reject partial tokenization before losing that proof.
			if fullInput && usage != nil && usage.TokensLowerBound != nil && *usage.TokensLowerBound {
				return nil, fmt.Errorf("%w: embedding token count is only a lower bound", binding.ErrInputLimit)
			}
			results[i].Input = inputUsage(usage)
		}
		if results[i].Input != nil {
			results[i].SequenceLength = results[i].Input.ProcessedTokens
		}
	}
	return results, nil
}

// checkView verifies a dimension and layer exit against the card.
func checkView(card modelservice.ModelCard, view embedding.Options) error {
	if view.Dimension < 0 || view.Layer < 0 {
		return fmt.Errorf("%w: embedding dimension and layer must be nonnegative", binding.ErrCapability)
	}
	if view.Dimension > 0 && !slices.Contains(card.Embedding.Dimensions, view.Dimension) {
		return fmt.Errorf("%w: model %q serves dimensions %v, not %d", binding.ErrCapability, card.ID, card.Embedding.Dimensions, view.Dimension)
	}
	if view.Layer > 0 && !slices.Contains(card.Embedding.Layers, view.Layer) {
		return fmt.Errorf("%w: model %q serves layer exits %v, not %d", binding.ErrCapability, card.ID, card.Embedding.Layers, view.Layer)
	}
	return nil
}

// embeddingOverflow maps the deployment's input policy to the embeddings
// surface, which truncates or rejects.
func embeddingOverflow(overflow string) string {
	if overflow == "reject" {
		return "reject"
	}
	return "truncate"
}

func modalities(card modelservice.ModelCard) []string {
	if len(card.Embedding.Modalities) == 0 {
		return []string{"text"}
	}
	return slices.Clone(card.Embedding.Modalities)
}

func normalization(card modelservice.ModelCard) string {
	if card.Embedding.Normalized {
		return "l2"
	}
	return "none"
}

func poolingContract(card modelservice.ModelCard) string {
	pooling := card.Embedding.Pooling
	if pooling == "" {
		pooling = "model"
	}
	return pooling + "+" + normalization(card)
}

// numerics names everything besides the input and view that determines a
// vector: package content, revision, engine, accelerator, dtype, profile and
// the input budget that truncates long inputs.
func numerics(card modelservice.ModelCard, limits binding.Limits) string {
	encoded, _ := json.Marshal([]string{
		card.ModelSHA256, card.Revision, card.Engine, card.Accelerator, card.Dtype, card.Profile,
		limits.Overflow, strconv.Itoa(limits.EffectiveTokens()),
	})
	return string(encoded)
}

// Close releases the binding's reference to its deployment. A closed
// provider fails every call, cached inputs included.
func (p *EmbeddingProvider) Close() error {
	p.closed.Store(true)
	return p.call.Close()
}

// Backend names the serving path.
func (p *EmbeddingProvider) Backend() string { return Provider }

// Dimension is the length of the default view's vectors.
func (p *EmbeddingProvider) Dimension() int { return p.info.Dimension }

// EmbeddingInfo describes the served model and the default view.
func (p *EmbeddingProvider) EmbeddingInfo() embedding.ModelInfo {
	info := p.info
	info.Dimensions = slices.Clone(p.info.Dimensions)
	info.Layers = slices.Clone(p.info.Layers)
	info.Modalities = slices.Clone(p.info.Modalities)
	if info.Audio != nil {
		audio := *info.Audio
		info.Audio = &audio
	}
	return info
}

// Embed embeds text at the default view.
func (p *EmbeddingProvider) Embed(ctx context.Context, text string) ([]float32, error) {
	return p.EmbedWithOptions(ctx, text, p.options)
}

// EmbedWithOptions embeds text at another declared view.
func (p *EmbeddingProvider) EmbedWithOptions(ctx context.Context, text string, options embedding.Options) ([]float32, error) {
	embedded, err := p.embed(ctx, []modelservice.EmbedInput{{Text: text}}, options)
	if err != nil {
		return nil, err
	}
	return embedded[0].Vector, nil
}

// EmbedFullInput rejects over-budget text before the runtime's model forward.
func (p *EmbeddingProvider) EmbedFullInput(ctx context.Context, text string) ([]float32, error) {
	return p.EmbedFullInputWithOptions(ctx, text, p.options)
}

// EmbedFullInputWithOptions keeps ordinary truncating embeddings independent
// from semantic comparisons that require complete tokenizer coverage.
func (p *EmbeddingProvider) EmbedFullInputWithOptions(ctx context.Context, text string, options embedding.Options) ([]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	embedded, err := p.embedWithPolicy(ctx, []modelservice.EmbedInput{{Text: text}}, options, true)
	if err != nil {
		return nil, err
	}
	return embedded[0].Vector, nil
}

// FitsInput reports whether the model reads text whole at the default view.
// The answer comes with the vector, which the cache keeps for the next call.
func (p *EmbeddingProvider) FitsInput(ctx context.Context, text string) (bool, error) {
	return p.FitsInputWithOptions(ctx, text, p.options)
}

// FitsInputWithOptions reports whether the model reads text whole, with the
// vector of another declared view.
func (p *EmbeddingProvider) FitsInputWithOptions(ctx context.Context, text string, options embedding.Options) (bool, error) {
	embedded, err := p.embed(ctx, []modelservice.EmbedInput{{Text: text}}, options)
	if err != nil {
		return false, err
	}
	if !embedded[0].InputKnown {
		return false, fmt.Errorf("%w: embedding response omitted input coverage", binding.ErrCapability)
	}
	return !embedded[0].Truncated, nil
}

// EmbedBatch embeds texts at the default view in as few runtime calls as the
// card's input limit allows.
func (p *EmbeddingProvider) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	inputs := make([]modelservice.EmbedInput, len(texts))
	for i, text := range texts {
		inputs[i] = modelservice.EmbedInput{Text: text}
	}
	embedded, err := p.embed(ctx, inputs, p.options)
	if err != nil {
		return nil, err
	}
	vectors := make([][]float32, len(embedded))
	for i := range embedded {
		vectors[i] = embedded[i].Vector
	}
	return vectors, nil
}

// EmbedImage embeds raw image bytes (any format the model's processor reads).
func (p *EmbeddingProvider) EmbedImage(ctx context.Context, data []byte, dimension int) ([]float32, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("%w: image is empty", binding.ErrInvalidInput)
	}
	url := "data:" + http.DetectContentType(data) + ";base64," + base64.StdEncoding.EncodeToString(data)
	embedded, err := p.embed(ctx, []modelservice.EmbedInput{{ImageURL: url}}, embedding.Options{Dimension: dimension})
	if err != nil {
		return nil, err
	}
	return embedded[0].Vector, nil
}

// EmbedAudio embeds decoded PCM, sent to the runtime as 32-bit float WAV so
// the samples arrive unchanged.
func (p *EmbeddingProvider) EmbedAudio(ctx context.Context, request embedding.AudioRequest) ([]float32, error) {
	if err := request.Validate(); err != nil {
		return nil, fmt.Errorf("%w: %w", binding.ErrInvalidInput, err)
	}
	wav := base64.StdEncoding.EncodeToString(embedding.EncodeFloatWAV(request))
	embedded, err := p.embed(ctx, []modelservice.EmbedInput{{AudioWAV: wav}}, embedding.Options{Dimension: request.Options.Dimension})
	if err != nil {
		return nil, err
	}
	return embedded[0].Vector, nil
}

// embed answers inputs from the shared cache and embeds the rest in calls of
// at most maxInputs inputs.
func (p *EmbeddingProvider) embed(ctx context.Context, inputs []modelservice.EmbedInput, options embedding.Options) ([]embedding.Embedded, error) {
	return p.embedWithPolicy(ctx, inputs, options, false)
}

func (p *EmbeddingProvider) embedWithPolicy(ctx context.Context, inputs []modelservice.EmbedInput, options embedding.Options, fullInput bool) ([]embedding.Embedded, error) {
	if p.closed.Load() {
		return nil, binding.ErrClosed
	}
	numerics := p.numerics
	if fullInput {
		// A cached truncated vector or response without coverage cannot satisfy
		// the stricter policy, including while an ordinary call is in flight.
		numerics += ":full-input:reject"
	}
	keys := make([]embedding.VectorKey, len(inputs))
	for i, input := range inputs {
		kind, content := embedding.InputText, input.Text
		switch {
		case input.ImageURL != "":
			kind, content = embedding.InputImage, input.ImageURL
		case input.AudioWAV != "":
			kind, content = embedding.InputAudio, input.AudioWAV
		}
		keys[i] = embedding.NewVectorKey(numerics, options, kind, []byte(content))
	}
	// Inside a request bundle, waiting on another participant's call would
	// hold the bundle until its window ends; the bundle carries both calls'
	// inputs in one round trip instead.
	resolve := sharedVectors.Resolve
	if modelservice.InBundle(ctx) {
		resolve = sharedVectors.ResolveAlone
	}
	return resolve(ctx, keys, func(ctx context.Context, missing []int) ([]embedding.Embedded, error) {
		return p.embedMissing(ctx, inputs, missing, options, fullInput)
	})
}

// concurrentEmbedBatches bounds the card-sized batches of one call that are
// in flight at once.
const concurrentEmbedBatches = 8

// embedMissing embeds inputs[missing] in batches of the card's input limit,
// up to concurrentEmbedBatches at a time. Inside a request bundle the
// concurrent batches park in it and travel in one round trip.
func (p *EmbeddingProvider) embedMissing(ctx context.Context, inputs []modelservice.EmbedInput, missing []int, options embedding.Options, fullInput bool) ([]embedding.Embedded, error) {
	size := len(missing)
	if p.maxInputs > 0 {
		size = min(size, p.maxInputs)
	}
	vectors := make([]embedding.Embedded, len(missing))
	embedBatch := func(start int) error {
		end := min(start+size, len(missing))
		batch := make([]modelservice.EmbedInput, 0, end-start)
		for _, i := range missing[start:end] {
			batch = append(batch, inputs[i])
		}
		results, err := p.call.Call(ctx, p.recipe, embeddingRequest{Inputs: batch, Options: options, FullInput: fullInput})
		if err != nil {
			return err
		}
		for j, result := range results {
			vectors[start+j] = embedding.Embedded{Vector: result.Embedding, Truncated: result.Input != nil && result.Input.Truncated, InputKnown: result.Input != nil}
		}
		return nil
	}
	if size == len(missing) {
		return vectors, embedBatch(0)
	}
	batches := (len(missing) + size - 1) / size
	errs := make([]error, batches)
	for wave := 0; wave < batches; wave += concurrentEmbedBatches {
		modelservice.Fan(ctx, min(concurrentEmbedBatches, batches-wave), func(i int) {
			errs[wave+i] = embedBatch((wave + i) * size)
		})
		if err := errors.Join(errs[wave:min(wave+concurrentEmbedBatches, batches)]...); err != nil {
			return nil, err
		}
	}
	return vectors, nil
}

// CacheIdentity names the default view's vector space for request-local caches.
func (p *EmbeddingProvider) CacheIdentity() string { return p.CacheIdentityForOptions(p.options) }

// CacheIdentityForOptions names a view's vector space.
func (p *EmbeddingProvider) CacheIdentityForOptions(options embedding.Options) string {
	return fmt.Sprintf("%s:layer=%d:dimension=%d", p.numerics, options.Layer, options.Dimension)
}

// RepresentationIdentity identifies a view's vectors for persistent stores.
// It names the model package's content, not a path or a revision label.
func (p *EmbeddingProvider) RepresentationIdentity(options embedding.Options, inputPolicy string) (embedding.ContentIdentity, error) {
	descriptor := p.representation
	descriptor.Layer, descriptor.Dimension = options.Layer, cmp.Or(options.Dimension, p.fullDimension)
	return embedding.IdentityForRuntime(descriptor, inputPolicy)
}

// DiagnoseEmbedding runs a recipe's prepared embedding binding on text at its
// default view.
func (r *Runtime) DiagnoseEmbedding(ctx context.Context, recipe, name, text string) (DiagnosticResult[tasks.EmbeddingResult], error) {
	var response DiagnosticResult[tasks.EmbeddingResult]
	handle, metadata, err := binding.LookupPrepared[embeddingRequest, []tasks.EmbeddingResult](r.prepared, recipe, name, embeddingContract)
	if err != nil {
		return response, err
	}
	options := embedding.Options{}
	if metadata.Capability.Embedding != nil {
		options.Dimension = metadata.Capability.Embedding.Dimension
		options.Layer = metadata.Capability.Embedding.Layer
	}
	results, err := handle.Call(ctx, recipe, embeddingRequest{Inputs: []modelservice.EmbedInput{{Text: text}}, Options: options})
	response.Binding = metadata
	if err == nil {
		response.Result = results[0]
	}
	return response, err
}
