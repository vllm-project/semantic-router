package extproc

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	. "github.com/onsi/ginkgo/v2"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/dispatchauthority"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tools"
)

type lifecycleDispatchRuntime struct {
	dispatchCapabilityRuntimeStub

	mu        sync.Mutex
	issued    map[string]dispatchauthority.RoutingOnlyChainIssueRequest
	terminals backendinvoker.ResponseTerminalStore
}

func newLifecycleDispatchRuntime() *lifecycleDispatchRuntime {
	return &lifecycleDispatchRuntime{
		issued:    make(map[string]dispatchauthority.RoutingOnlyChainIssueRequest),
		terminals: backendinvoker.NewLocalResponseTerminalStore(),
	}
}

func (runtime *lifecycleDispatchRuntime) IssueRoutingOnlyChain(
	_ context.Context,
	request dispatchauthority.RoutingOnlyChainIssueRequest,
) (string, error) {
	runtime.mu.Lock()
	runtime.issued[request.RequestID] = request
	runtime.mu.Unlock()
	return "test-capability", nil
}

func (runtime *lifecycleDispatchRuntime) VerifyDispatchOutcome(
	ctx context.Context,
	token string,
	request dispatchauthority.OutcomeVerificationRequest,
) (backendinvoker.DispatchOutcome, error) {
	if token != "test-outcome" {
		return backendinvoker.DispatchOutcome{}, fmt.Errorf("unexpected test dispatch outcome")
	}
	runtime.mu.Lock()
	issued, ok := runtime.issued[request.RequestID]
	runtime.mu.Unlock()
	if !ok || len(issued.Candidates) == 0 {
		return backendinvoker.DispatchOutcome{}, fmt.Errorf("test dispatch was not issued")
	}
	candidate := issued.Candidates[0]
	admissionID, admissionDigest := dispatchauthority.RoutingOnlyAdmissionIdentity(
		issued.Generation, request.RequestID,
	)
	attempt := backendinvoker.AttemptResult{
		Attempt: backendinvoker.Attempt{ID: "test-attempt", Number: 1, BackendID: "test-backend"},
		State:   backendinvoker.AttemptResponseStarted, StatusCode: 200,
	}
	if err := runtime.terminals.Finalize(ctx, backendinvoker.Plan{
		NamespaceID: issued.Generation.NamespaceID, QuotaPartition: issued.Generation.QuotaPartition,
		PublicationID: issued.Generation.PublicationID, RuntimeEpoch: issued.Generation.RuntimeEpoch,
		RoutingRevision: issued.Generation.SnapshotRevision, RoutingDigest: issued.Generation.RoutingDigest,
		AdmissionID: admissionID, AdmissionDigest: admissionDigest,
		RequestID: request.RequestID, DispatchID: candidate.Dispatch.DispatchID,
		DispatchType: "primary", Ordinal: int(candidate.Dispatch.Ordinal), Priority: candidate.Priority,
		DispatchPlanDigest: candidate.Dispatch.DispatchPlanDigest,
		ModelID:            candidate.Model.ID, ModelRevision: candidate.Model.Revision,
	}, attempt, backendinvoker.ResponseTerminal{
		Usage: testAuthoritativeUsage(5, 3), StopReason: llmprotocol.StopEndTurn,
	}); err != nil {
		return backendinvoker.DispatchOutcome{}, err
	}
	return backendinvoker.DispatchOutcome{
		NamespaceID: issued.Generation.NamespaceID, QuotaPartition: issued.Generation.QuotaPartition,
		PublicationID: issued.Generation.PublicationID, RuntimeEpoch: issued.Generation.RuntimeEpoch,
		RoutingRevision: issued.Generation.SnapshotRevision, RoutingDigest: issued.Generation.RoutingDigest,
		RequestID: request.RequestID,
		RequestDigest: backendinvoker.RequestDigest(
			issued.Final.Method, issued.Final.Path, issued.Final.Query, issued.Final.Body,
		),
		Attempted: []backendinvoker.DispatchOutcomeCandidate{{
			DispatchID: candidate.Dispatch.DispatchID, DispatchType: "primary",
			Ordinal: int(candidate.Dispatch.Ordinal), DispatchPlanDigest: candidate.Dispatch.DispatchPlanDigest,
			ModelID: candidate.Model.ID, ModelRevision: candidate.Model.Revision,
			Priority: candidate.Priority, State: backendinvoker.AttemptResponseStarted, AttemptCount: 1,
		}},
		SelectedDispatchID: candidate.Dispatch.DispatchID,
	}, nil
}

func bindAuthoritativeTestTerminal(
	router *OpenAIRouter,
	ctx *RequestContext,
	inputTokens, outputTokens int64,
) error {
	if router == nil || ctx == nil || ctx.RequestID == "" {
		return fmt.Errorf("test response lifecycle is incomplete")
	}
	dispatchID := "dispatch-" + ctx.RequestID
	store := backendinvoker.NewLocalResponseTerminalStore()
	router.ResponseTerminals = store
	reference := testResponseTerminalReference(ctx.RequestID, dispatchID, ctx.RequestModel)
	ctx.DispatchState = &requestDispatchState{
		requestID: ctx.RequestID, primaryDispatchID: dispatchID, selectedDispatchID: dispatchID,
		dispatches: []*inferenceDispatch{{
			id: dispatchID, terminalReference: reference,
		}},
	}
	return store.Finalize(context.Background(), testResponseTerminalPlan(reference), backendinvoker.AttemptResult{
		Attempt: backendinvoker.Attempt{ID: "test-attempt", Number: 1, BackendID: "test-backend"},
		State:   backendinvoker.AttemptResponseStarted, StatusCode: 200,
	}, backendinvoker.ResponseTerminal{
		Usage: testAuthoritativeUsage(inputTokens, outputTokens), StopReason: llmprotocol.StopEndTurn,
	})
}

func testResponseTerminalReference(
	requestID string,
	dispatchID string,
	modelID string,
) backendinvoker.ResponseTerminalReference {
	return backendinvoker.ResponseTerminalReference{
		NamespaceID: "namespace", QuotaPartition: "partition", PublicationID: "publication",
		RuntimeEpoch: 1, RoutingRevision: 1, RoutingDigest: strings.Repeat("a", 64),
		AdmissionID: "admission", AdmissionDigest: strings.Repeat("b", 64),
		RequestID: requestID, DispatchID: dispatchID, DispatchType: "primary",
		Ordinal: 0, Priority: 0, DispatchPlanDigest: strings.Repeat("c", 64),
		ModelID: modelID, ModelRevision: 1,
	}
}

func testResponseTerminalPlan(reference backendinvoker.ResponseTerminalReference) backendinvoker.Plan {
	return backendinvoker.Plan{
		NamespaceID: reference.NamespaceID, QuotaPartition: reference.QuotaPartition,
		PublicationID: reference.PublicationID, RuntimeEpoch: reference.RuntimeEpoch,
		RoutingRevision: reference.RoutingRevision, RoutingDigest: reference.RoutingDigest,
		AdmissionID: reference.AdmissionID, AdmissionDigest: reference.AdmissionDigest,
		RequestID: reference.RequestID, DispatchID: reference.DispatchID,
		DispatchType: reference.DispatchType, Ordinal: reference.Ordinal, Priority: reference.Priority,
		DispatchPlanDigest: reference.DispatchPlanDigest,
		ModelID:            reference.ModelID, ModelRevision: reference.ModelRevision,
	}
}

func testAuthoritativeUsage(inputTokens, outputTokens int64) llmprotocol.Usage {
	totalTokens := inputTokens + outputTokens
	zero := int64(0)
	count := func(value *int64) llmprotocol.TokenCount {
		return llmprotocol.TokenCount{Value: value, Provenance: llmprotocol.UsageAuthoritative}
	}
	return llmprotocol.Usage{
		State:         llmprotocol.UsageAvailable,
		InputUncached: count(&inputTokens), InputCacheRead: count(&zero), InputCacheWrite: count(&zero),
		OutputReasoning: count(&zero), OutputOther: count(&outputTokens),
		InputTotal: count(&inputTokens), OutputTotal: count(&outputTokens), Total: count(&totalTokens),
	}
}

func testNeutralRequest(model, text string) *llmprotocol.Request {
	return &llmprotocol.Request{
		Generation: 1,
		Model:      model,
		Messages: []llmprotocol.Message{{
			Role:    llmprotocol.RoleUser,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: text}},
		}},
	}
}

var extprocTestModelWeightCandidates = []string{
	"model.safetensors",
	"model.safetensors.index.json",
	"pytorch_model.bin",
	"adapter_model.safetensors",
}

// CreateTestRouter creates a properly initialized router for testing.
func CreateTestRouter(cfg *config.RouterConfig) (*OpenAIRouter, error) {
	classifierCfg := cloneRouterConfigForTest(cfg)
	categoryMapping, err := loadTestCategoryMapping(classifierCfg)
	if err != nil {
		return nil, err
	}
	if !extprocTestModelArtifactsAvailable(classifierCfg.CategoryModel.ModelID) {
		classifierCfg.CategoryModel.ModelID = ""
		classifierCfg.CategoryMappingPath = ""
		categoryMapping = nil
	}

	piiMapping, err := loadTestPIIMapping(classifierCfg)
	if err != nil {
		return nil, err
	}

	err = initTestBERTModel(classifierCfg)
	if err != nil {
		return nil, err
	}

	semanticCache, err := newTestSemanticCache(classifierCfg)
	if err != nil {
		return nil, err
	}

	toolsDatabase, err := newTestToolsDatabase(classifierCfg)
	if err != nil {
		return nil, err
	}

	classifier, err := classification.NewClassifier(classifierCfg, categoryMapping, piiMapping, nil)
	if err != nil {
		return nil, err
	}

	dispatch := newLifecycleDispatchRuntime()
	return &OpenAIRouter{
		Config:               cfg,
		CategoryDescriptions: cfg.GetCategoryDescriptions(),
		Classifier:           classifier,
		Cache:                semanticCache,
		ToolsDatabase:        toolsDatabase,
		ResponseAPIFilter:    newTestResponseAPIFilter(cfg),
		DispatchCapabilities: dispatch,
		ResponseTerminals:    dispatch.terminals,
		ProtocolCodecs:       protocolcodec.NewBuiltinRegistry(),
	}, nil
}

func cloneRouterConfigForTest(cfg *config.RouterConfig) *config.RouterConfig {
	if cfg == nil {
		return nil
	}
	clone := *cfg
	return &clone
}

func loadTestCategoryMapping(cfg *config.RouterConfig) (*classification.CategoryMapping, error) {
	if cfg == nil || cfg.CategoryMappingPath == "" {
		return nil, nil
	}
	if _, err := os.Stat(cfg.CategoryMappingPath); err != nil {
		return nil, nil
	}
	return classification.LoadCategoryMapping(cfg.CategoryMappingPath)
}

func loadTestPIIMapping(cfg *config.RouterConfig) (*classification.PIIMapping, error) {
	if cfg.PIIMappingPath == "" {
		return nil, nil
	}
	if _, err := os.Stat(cfg.PIIMappingPath); err != nil {
		return nil, nil
	}
	return classification.LoadPIIMapping(cfg.PIIMappingPath)
}

func initTestBERTModel(cfg *config.RouterConfig) error {
	if err := candle_binding.InitModel(cfg.BertModelPath, cfg.UseCPU); err != nil {
		return fmt.Errorf("failed to initialize BERT model: %w", err)
	}
	return nil
}

func newTestSemanticCache(cfg *config.RouterConfig) (cache.CacheBackend, error) {
	return cache.NewCacheBackend(cache.CacheConfig{
		BackendType:         cache.InMemoryCacheType,
		Enabled:             cfg.Enabled,
		SimilarityThreshold: cfg.GetCacheSimilarityThreshold(),
		MaxEntries:          cfg.MaxEntries,
		TTLSeconds:          cfg.TTLSeconds,
		EvictionPolicy:      cache.EvictionPolicyType(cfg.EvictionPolicy),
		EmbeddingModel:      cfg.EmbeddingModel,
	})
}

func newTestToolsDatabase(cfg *config.RouterConfig) (*tools.ToolsDatabase, error) {
	toolCfg := cfg.Tools
	toolsSimilarityThreshold := float32(0.2)
	if toolCfg.SimilarityThreshold != nil {
		toolsSimilarityThreshold = *toolCfg.SimilarityThreshold
	}

	toolsDatabase := tools.NewToolsDatabase(tools.ToolsDatabaseOptions{
		SimilarityThreshold: toolsSimilarityThreshold,
		Enabled:             toolCfg.Enabled,
		ModelType:           cfg.EmbeddingConfig.ModelType,
		TargetDimension:     cfg.EmbeddingConfig.TargetDimension,
	})
	if !toolCfg.Enabled || toolCfg.ToolsDBPath == "" {
		return toolsDatabase, nil
	}
	if err := toolsDatabase.LoadToolsFromFile(toolCfg.ToolsDBPath); err != nil {
		return nil, fmt.Errorf("failed to load tools database: %w", err)
	}
	return toolsDatabase, nil
}

func newTestResponseAPIFilter(cfg *config.RouterConfig) *ResponseAPIFilter {
	if !cfg.ResponseAPI.Enabled {
		return nil
	}
	return NewResponseAPIFilter(NewMockResponseStore())
}

func testInferenceRequestAccess(userID, teamID string) *inferenceRequestAccess {
	return &inferenceRequestAccess{tenant: accessruntime.TenantContext{
		NamespaceID: "test-namespace",
		APIKeyID:    "test-api-key",
		UserID:      userID,
		TeamID:      teamID,
	}}
}

func findExtprocTestProjectRoot() string {
	wd, err := os.Getwd()
	if err != nil {
		return ""
	}
	for current := wd; current != filepath.Dir(current); current = filepath.Dir(current) {
		if _, err := os.Stat(filepath.Join(current, "models")); err == nil {
			return current
		}
	}
	return ""
}

func resolveExtprocTestPath(relativePath string) string {
	if _, err := os.Stat(relativePath); err == nil {
		return relativePath
	}
	root := findExtprocTestProjectRoot()
	if root == "" {
		return relativePath
	}
	trimmed := strings.TrimPrefix(relativePath, "../../../../")
	trimmed = strings.TrimPrefix(trimmed, "../../../../../")
	absolute := filepath.Join(root, trimmed)
	if _, err := os.Stat(absolute); err == nil {
		return absolute
	}
	return relativePath
}

func extprocTestModelArtifactsAvailable(modelPath string) bool {
	info, err := os.Stat(modelPath)
	if err != nil || !info.IsDir() {
		return false
	}
	if _, err := os.Stat(filepath.Join(modelPath, "config.json")); err != nil {
		return false
	}
	for _, candidate := range extprocTestModelWeightCandidates {
		if _, err := os.Stat(filepath.Join(modelPath, candidate)); err == nil {
			return true
		}
	}
	if matches, _ := filepath.Glob(filepath.Join(modelPath, "*.safetensors")); len(matches) > 0 {
		return true
	}
	if matches, _ := filepath.Glob(filepath.Join(modelPath, "*.bin")); len(matches) > 0 {
		return true
	}
	return false
}

func skipExtprocSpecIfModelArtifactsMissing(label string, modelPath string) {
	if extprocTestModelArtifactsAvailable(modelPath) {
		return
	}
	Skip(fmt.Sprintf("%s artifacts not available at %s (missing model weights)", label, modelPath))
}
