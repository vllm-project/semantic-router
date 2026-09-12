package native

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	candle "github.com/vllm-project/semantic-router/candle-binding"
	ort "github.com/vllm-project/semantic-router/onnx-binding/instance"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

// EmbeddingProvider owns one binding reference to a shared physical model.
// Calls hold the resource through admission and non-preemptible native work.
type EmbeddingProvider struct {
	info      embedding.ModelInfo
	identity  string
	resource  *binding.Resource
	text      *binding.Resolved[embedding.TextRequest, tasks.EmbeddingResult]
	recipe    string
	backend   string
	dimension int
	options   embedding.Options
}

type embeddingEngine struct {
	remote *embedding.OpenAICompatibleProvider
	candle *candle.EmbeddingModel
	ort    *ort.EmbeddingModel
	multi  *ort.MultiModalModel
}

func (e *embeddingEngine) Close() error {
	if e.remote != nil {
		return e.remote.Close()
	}
	if e.candle != nil {
		return e.candle.Close()
	}
	if e.ort != nil {
		return e.ort.Close()
	}
	return e.multi.Close()
}

func (e *embeddingEngine) embed(text string, options embedding.Options) (tasks.EmbeddingResult, error) {
	if e.candle != nil {
		out, err := e.candle.EmbedAtLayer(text, options.Dimension, options.Layer)
		return tasks.EmbeddingResult{Embedding: out.Values, Input: &tasks.InputUsage{OriginalTokens: out.Input.InputTokens, ProcessedTokens: out.Input.ProcessedTokens, Truncated: out.Input.Truncated}}, nativeError(err)
	}
	if e.ort != nil {
		out, err := e.ort.Encode(text, options.Layer, options.Dimension)
		return embeddingORTResult(out), ortError(err)
	}
	if options.Layer != 0 {
		return tasks.EmbeddingResult{}, fmt.Errorf("%w: ORT multimodal embedding has no layer early exit", binding.ErrCapability)
	}
	out, err := e.multi.EncodeText(text, options.Dimension)
	return embeddingORTResult(out), ortError(err)
}

func embeddingORTResult(out ort.EmbeddingResult) tasks.EmbeddingResult {
	result := tasks.EmbeddingResult{Embedding: out.Values}
	if out.Input != nil {
		result.Input = &tasks.InputUsage{OriginalTokens: out.Input.OriginalTokens, ProcessedTokens: out.Input.ProcessedTokens, Truncated: out.Input.Truncated}
	}
	return result
}

func validateEmbeddingResult(input embedding.TextRequest, result tasks.EmbeddingResult) error {
	return validateEmbedding(input, result.Embedding)
}

func validateEmbedding(input embedding.TextRequest, vector []float32) error {
	if len(vector) == 0 {
		return fmt.Errorf("embedding vector is empty")
	}
	if input.Options.Dimension > 0 && len(vector) != input.Options.Dimension {
		return fmt.Errorf("embedding dimension %d differs from requested %d", len(vector), input.Options.Dimension)
	}
	for _, value := range vector {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return fmt.Errorf("embedding vector is not finite")
		}
	}
	return nil
}

func (r *Runtime) embeddingTask() (*binding.Task[embedding.TextRequest, tasks.EmbeddingResult], error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	task, err := binding.Lookup[embedding.TextRequest, tasks.EmbeddingResult](r.registry, "embedding.v1")
	if err == nil {
		return task, nil
	}
	return binding.Register(r.registry, "embedding.v1", func(input embedding.TextRequest) error {
		if input.Options.Dimension < 0 || input.Options.Layer < 0 {
			return fmt.Errorf("embedding dimension and layer must be nonnegative")
		}
		return validateText(input.Text)
	}, validateEmbeddingResult)
}

func (r *Runtime) Embedding(ctx context.Context, spec config.ResolvedModelBinding, dimension, layer int) (prepared *EmbeddingProvider, resultErr error) {
	defer func() { observePreparationFailure(spec, resultErr) }()
	if spec.Deployment.Provider != "candle" && spec.Deployment.Provider != "ort" {
		return nil, fmt.Errorf("%w: native embedding provider %q", binding.ErrCapability, spec.Deployment.Provider)
	}
	options := candleOptions(spec)
	revision, err := r.artifactRevision(ctx, options.ModelPath)
	if err != nil {
		return nil, err
	}
	execution, _ := json.Marshal(options)
	if spec.Deployment.Provider == "candle" && spec.Binding.Head != "" {
		return nil, fmt.Errorf("%w: Candle embedding does not support a separate classifier head", binding.ErrCapability)
	}
	if spec.Deployment.Provider == "ort" {
		selected, optionErr := ortOptions(spec)
		if optionErr != nil {
			return nil, optionErr
		}
		headRevision := ""
		if selected.ModelFile != "" {
			path := selected.ModelFile
			if !filepath.IsAbs(path) {
				path = filepath.Join(selected.ModelPath, path)
			}
			headRevision, optionErr = r.artifactRevision(ctx, filepath.Dir(path))
			if optionErr != nil {
				return nil, optionErr
			}
		}
		execution, _ = json.Marshal(struct {
			Options               ort.Options
			HeadRevision, Adapter string
		}{selected, headRevision, spec.Binding.Adapter})
	}
	id := binding.ResourceIdentity{Artifact: options.ModelPath, Revision: spec.Deployment.Revision + ":" + revision, Provider: spec.Deployment.Provider, Device: options.Device, Precision: options.Precision, Execution: "embedding:" + string(execution)}
	budget, gate := resourceAdmission(spec)
	var capability binding.Capability
	var layers []int
	resource, err := r.Pool.Acquire(ctx, id, budget, gate, func(context.Context) (io.Closer, error) {
		if spec.Deployment.Provider == "candle" {
			model, loadErr := candle.LoadEmbeddingModel(options)
			if loadErr != nil {
				return nil, nativeError(loadErr)
			}
			return &embeddingEngine{candle: model}, nil
		}
		ortOptions, optionErr := ortOptions(spec)
		if optionErr != nil {
			return nil, optionErr
		}
		if spec.Binding.Adapter == "multimodal" {
			model, loadErr := ort.LoadMultiModal(ortOptions)
			if loadErr != nil {
				return nil, loadErr
			}
			return &embeddingEngine{multi: model}, nil
		}
		model, loadErr := ort.LoadEmbeddingModel(ortOptions)
		if loadErr != nil {
			return nil, loadErr
		}
		return &embeddingEngine{ort: model}, nil
	})
	if err != nil {
		return nil, err
	}
	capability = binding.Capability{Contract: "embedding.v1", Provider: spec.Deployment.Provider, Device: spec.Deployment.Device, Precision: spec.Deployment.Precision}
	err = resource.Use(ctx, func(value io.Closer) error {
		engine := value.(*embeddingEngine)
		if engine.candle != nil {
			info, infoErr := engine.candle.Info()
			if infoErr != nil {
				return infoErr
			}
			capability.Device = info.Device
			capability.Precision = info.Precision
			capability.Limits = binding.Limits{ModelTokens: info.ArchitecturalMaxTokens, TaskTokens: info.ArchitecturalMaxTokens, DeploymentTokens: spec.Deployment.Input.MaxTokens, Overflow: spec.Deployment.Input.Overflow}
			capability.Embedding = candleEmbeddingSemantics(info.ModelType, layer)
			capability.Embedding.Modalities = append([]string(nil), info.Modalities...)
			if info.ModelType == "mmbert" || info.ModelType == "mmbert_embedding" {
				layers = candleEmbeddingLayers(options.ModelPath)
			}
		} else {
			var info ort.Info
			var infoErr error
			if engine.ort != nil {
				info, infoErr = engine.ort.Info()
			} else {
				info, infoErr = engine.multi.Info()
			}
			if infoErr != nil {
				return infoErr
			}
			capability, infoErr = ortCapability(spec, info)
			if infoErr != nil {
				return infoErr
			}
			if engine.ort != nil {
				layers = ortEmbeddingLayers(info, options.ModelPath)
			}
			capability.Embedding = &binding.EmbeddingCapability{Layer: layer, Pooling: "graph_defined", Normalization: "l2", Modalities: []string{"text"}}
			if engine.multi != nil {
				capability.Embedding.Modalities = []string{"text", "image", "audio"}
			}
		}
		request := embedding.TextRequest{Text: "warmup", Options: embedding.Options{Dimension: dimension, Layer: layer}}
		warm, warmErr := engine.embed(request.Text, request.Options)
		if warmErr != nil {
			return warmErr
		}
		if warmErr = validateEmbeddingResult(request, warm); warmErr != nil {
			return fmt.Errorf("%w: %w", binding.ErrInvalidResult, warmErr)
		}
		capability.Embedding.Dimension = len(warm.Embedding)
		return nil
	})
	if err != nil {
		_ = resource.Close()
		return nil, err
	}
	task, err := r.embeddingTask()
	if err != nil {
		_ = resource.Close()
		return nil, err
	}
	text, err := task.Resolve(taskIdentity(spec), capability, resource, func(_ context.Context, value io.Closer, input embedding.TextRequest) (tasks.EmbeddingResult, error) {
		return value.(*embeddingEngine).embed(input.Text, input.Options)
	})
	if err != nil {
		_ = resource.Close()
		return nil, err
	}
	key, _ := id.Key()
	provider := &EmbeddingProvider{identity: key, resource: resource, text: text, recipe: string(spec.Recipe), backend: spec.Deployment.Provider, options: embedding.Options{Dimension: dimension, Layer: layer}}
	provider.text.Ready()
	provider.dimension = capability.Embedding.Dimension
	provider.info = embedding.ModelInfo{Layers: layers, Artifact: options.ModelPath, Backend: provider.backend, Dimension: provider.dimension, MaxTokens: capability.Limits.EffectiveTokens(), Pooling: capability.Embedding.Pooling, Normalization: capability.Embedding.Normalization, Modalities: append([]string(nil), capability.Embedding.Modalities...)}
	return provider, nil
}
func (p *EmbeddingProvider) Close() error    { return p.text.Close() }
func (p *EmbeddingProvider) Backend() string { return p.backend }
func (p *EmbeddingProvider) Dimension() int  { return p.dimension }
func (p *EmbeddingProvider) Embed(ctx context.Context, text string) ([]float32, error) {
	return p.EmbedWithOptions(ctx, text, p.options)
}

func (p *EmbeddingProvider) EmbedWithOptions(ctx context.Context, text string, options embedding.Options) ([]float32, error) {
	result, err := p.text.Call(ctx, p.recipe, embedding.TextRequest{Text: text, Options: options})
	return result.Embedding, err
}

func (p *EmbeddingProvider) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	result := make([][]float32, len(texts))
	for i, text := range texts {
		value, err := p.Embed(ctx, text)
		if err != nil {
			return nil, err
		}
		result[i] = value
	}
	return result, nil
}

func (p *EmbeddingProvider) EmbedImage(ctx context.Context, data []byte, dimension int) ([]float32, error) {
	var vector []float32
	err := p.resource.Use(ctx, func(value io.Closer) error {
		engine := value.(*embeddingEngine)
		var err error
		if engine.candle != nil {
			var out candle.InstanceEmbeddingOutput
			out, err = engine.candle.EmbedImage(data, dimension)
			vector = out.Values
		} else if engine.multi != nil {
			var out ort.EmbeddingResult
			out, err = engine.multi.EncodeImageBytes(data, dimension)
			vector = out.Values
		} else {
			return fmt.Errorf("%w: embedding provider has no image encoder", binding.ErrCapability)
		}
		if err != nil {
			return nativeError(ortError(err))
		}
		return validateEmbedding(embedding.TextRequest{Options: embedding.Options{Dimension: dimension}}, vector)
	})
	return vector, err
}

func (p *EmbeddingProvider) EmbedAudio(ctx context.Context, data []float32, bins, frames, dimension int) ([]float32, error) {
	var vector []float32
	err := p.resource.Use(ctx, func(value io.Closer) error {
		engine := value.(*embeddingEngine)
		var err error
		if engine.candle != nil {
			var out candle.InstanceEmbeddingOutput
			out, err = engine.candle.EmbedAudio(data, bins, frames, dimension)
			vector = out.Values
		} else if engine.multi != nil {
			var out ort.EmbeddingResult
			out, err = engine.multi.EncodeAudio(data, bins, frames, dimension)
			vector = out.Values
		} else {
			return fmt.Errorf("%w: embedding provider has no audio encoder", binding.ErrCapability)
		}
		if err != nil {
			return nativeError(ortError(err))
		}
		return validateEmbedding(embedding.TextRequest{Options: embedding.Options{Dimension: dimension}}, vector)
	})
	return vector, err
}

func (p *EmbeddingProvider) Windows(ctx context.Context, text string, limit int) ([]embedding.Window, error) {
	var windows []embedding.Window
	err := p.resource.Use(ctx, func(value io.Closer) error {
		engine := value.(*embeddingEngine)
		if engine.candle != nil {
			ranges, err := engine.candle.Windows(text, limit)
			for _, r := range ranges {
				windows = append(windows, embedding.Window{Start: r.Start, End: r.End})
			}
			return nativeError(err)
		}
		var ranges []ort.TextWindow
		var err error
		if engine.ort != nil {
			ranges, err = engine.ort.Windows(text, limit)
		} else if engine.multi != nil {
			ranges, err = engine.multi.Windows(text, limit)
		} else {
			return fmt.Errorf("%w: remote embedding token windows unavailable", binding.ErrCapability)
		}
		for _, r := range ranges {
			windows = append(windows, embedding.Window{Start: r.Start, End: r.End})
		}
		return ortError(err)
	})
	return windows, err
}

// RemoteEmbedding owns the HTTP connector and admission reference, never the
// external model process. The declared service is warmed before publication.
func (r *Runtime) RemoteEmbedding(ctx context.Context, spec config.ResolvedModelBinding, cfg embedding.OpenAICompatibleConfig) (prepared *EmbeddingProvider, resultErr error) {
	defer func() { observePreparationFailure(spec, resultErr) }()
	if spec.Deployment.Input.MaxTokens > 0 {
		return nil, fmt.Errorf("%w: remote embedding endpoint does not report tokenizer counts", binding.ErrCapability)
	}
	if cfg.APIKeyEnv != "" {
		cfg.APIKey = os.Getenv(cfg.APIKeyEnv)
		if cfg.APIKey == "" {
			return nil, fmt.Errorf("embedding API key environment is unset")
		}
		cfg.APIKeyEnv = ""
	}
	execution, err := remoteEmbeddingExecution(cfg)
	if err != nil {
		return nil, err
	}
	identity := binding.ResourceIdentity{Artifact: cfg.BaseURL, Revision: cfg.Model, Provider: "http", Device: "external", Precision: "external", Execution: execution}
	budget, gate := resourceAdmission(spec)
	resource, err := r.Pool.Acquire(ctx, identity, budget, gate, func(context.Context) (io.Closer, error) {
		provider, loadErr := embedding.NewOpenAICompatibleProvider(cfg)
		if loadErr != nil {
			return nil, loadErr
		}
		return &embeddingEngine{remote: provider}, nil
	})
	if err != nil {
		return nil, err
	}
	task, err := r.embeddingTask()
	if err != nil {
		_ = resource.Close()
		return nil, err
	}
	var warm []float32
	err = resource.Use(ctx, func(value io.Closer) error {
		var callErr error
		warm, callErr = value.(*embeddingEngine).remote.Embed(ctx, "semantic router embedding warmup")
		if callErr != nil {
			return callErr
		}
		return validateEmbedding(embedding.TextRequest{}, warm)
	})
	if err != nil {
		_ = resource.Close()
		return nil, err
	}
	capability := binding.Capability{Contract: "embedding.v1", Provider: "http", Device: "external", Precision: "external", Embedding: &binding.EmbeddingCapability{Dimension: len(warm), Modalities: []string{"text"}}}
	text, err := task.Resolve(taskIdentity(spec), capability, resource, func(ctx context.Context, value io.Closer, input embedding.TextRequest) (tasks.EmbeddingResult, error) {
		vector, callErr := value.(*embeddingEngine).remote.Embed(ctx, input.Text)
		return tasks.EmbeddingResult{Embedding: vector}, callErr
	})
	if err != nil {
		_ = resource.Close()
		return nil, err
	}
	key, _ := identity.Key()
	p := &EmbeddingProvider{identity: key, resource: resource, text: text, recipe: string(spec.Recipe), backend: config.EmbeddingBackendOpenAICompatible}
	p.text.Ready()
	p.dimension = len(warm)
	p.info = embedding.ModelInfo{Artifact: cfg.Model, Backend: p.backend, Dimension: p.dimension, Modalities: []string{"text"}}
	return p, nil
}

func (p *EmbeddingProvider) CacheIdentity() string { return p.identity }

func (p *EmbeddingProvider) EmbeddingInfo() embedding.ModelInfo {
	info := p.info
	info.Modalities = append([]string(nil), p.info.Modalities...)
	info.Layers = append([]int(nil), p.info.Layers...)
	return info
}

// These semantics describe the concrete adapters instantiated by the Candle
// loader, rather than assumptions about an arbitrary artifact name.
func candleEmbeddingSemantics(modelType string, layer int) *binding.EmbeddingCapability {
	info := &binding.EmbeddingCapability{Layer: layer, Normalization: "l2", Modalities: []string{"text"}}
	switch modelType {
	case "qwen3":
		info.Pooling = "last_token"
	case "bert", "gemma", "gemma3", "modernbert", "mmbert", "mmbert_embedding":
		info.Pooling = "mean"
	case "multimodal":
		info.Pooling = "mean_text;attention_image;mean_audio"
		info.Modalities = []string{"text", "image", "audio"}
	}
	return info
}

// HTTP clients may contain functions and transports; serialize only immutable
// connector settings. A caller-supplied client is shared only with itself.
func remoteEmbeddingExecution(cfg embedding.OpenAICompatibleConfig) (string, error) {
	auth := sha256.Sum256([]byte(cfg.APIKey))
	projected := struct {
		Endpoint, Model, Auth, Client                         string
		TimeoutSeconds, Retries, Dimension, ExpectedDimension int
		MaxResponseBytes                                      int64
	}{cfg.BaseURL, cfg.Model, hex.EncodeToString(auth[:]), fmt.Sprintf("%p", cfg.HTTPClient), cfg.TimeoutSeconds, cfg.MaxRetries, cfg.Dimensions, cfg.ExpectedDimension, cfg.MaxResponseBytes}
	encoded, err := json.Marshal(projected)
	if err != nil {
		return "", err
	}
	digest := sha256.Sum256(encoded)
	return hex.EncodeToString(digest[:]), nil
}

// Candle mmBERT owns every encoder layer. Keep the artifact's advertised exit
// layers within its loaded architecture, and include its final output layer.
func candleEmbeddingLayers(modelPath string) []int {
	data, err := os.ReadFile(filepath.Join(modelPath, "config.json"))
	if err != nil {
		return nil
	}
	var architecture struct {
		Layers int `json:"num_hidden_layers"`
	}
	if json.Unmarshal(data, &architecture) != nil || architecture.Layers <= 0 {
		return nil
	}
	seen := map[int]bool{architecture.Layers: true}
	for _, layer := range config.MmBertAvailableLayers(modelPath) {
		if layer > 0 && layer <= architecture.Layers {
			seen[layer] = true
		}
	}
	layers := make([]int, 0, len(seen))
	for layer := range seen {
		layers = append(layers, layer)
	}
	sort.Ints(layers)
	return layers
}

// Only loaded ORT exit sessions are exposed. An artifact manifest can describe
// additional graphs that are not present in this prepared execution.
func ortEmbeddingLayers(info ort.Info, modelPath string) []int {
	seen := map[int]bool{}
	for _, session := range info.Sessions {
		hasExit := false
		for _, component := range strings.Split(filepath.ToSlash(session.Graph), "/") {
			if strings.HasPrefix(component, "layer-") {
				layer, err := strconv.Atoi(strings.TrimPrefix(component, "layer-"))
				if err == nil && layer > 0 {
					seen[layer] = true
					hasExit = true
				}
			}
		}
		if !hasExit {
			data, err := os.ReadFile(filepath.Join(modelPath, "config.json"))
			var architecture struct {
				Layers int `json:"num_hidden_layers"`
			}
			if err == nil && json.Unmarshal(data, &architecture) == nil && architecture.Layers > 0 {
				seen[architecture.Layers] = true
			}
		}
	}
	layers := make([]int, 0, len(seen))
	for layer := range seen {
		layers = append(layers, layer)
	}
	sort.Ints(layers)
	return layers
}
