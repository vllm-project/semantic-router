package modelruntime

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

type runtimeTaskManifestCase struct {
	name              string
	configure         func(*config.RouterConfig)
	plan              embedding.RuntimePlan
	wantConsumerPaths embeddingPaths
	wantTasks         []string
	wantRequiredTasks []string
}

func TestRouterRuntimeEmbeddingTaskManifestIncludesLocalConsumers(t *testing.T) {
	for _, tt := range runtimeTaskManifestCases() {
		t.Run(tt.name, func(t *testing.T) {
			cfg := remoteEmbeddingRuntimeConfig("http://embedding-service:8000/v1")
			tt.configure(cfg)
			configuredPaths := resolveEmbeddingPaths(cfg)
			got, err := localConsumerEmbeddingPaths(cfg, configuredPaths)
			if err != nil {
				t.Fatalf("localConsumerEmbeddingPaths() error = %v", err)
			}
			if got != tt.wantConsumerPaths {
				t.Fatalf("local consumer paths = %+v, want %+v", got, tt.wantConsumerPaths)
			}

			_, tasks, _, err := routerRuntimeEmbeddingTasks(cfg, "test", configuredPaths, tt.plan)
			if err != nil {
				t.Fatalf("routerRuntimeEmbeddingTasks() error = %v", err)
			}
			assertRuntimeTaskManifest(t, tasks, tt.wantTasks, tt.wantRequiredTasks)
		})
	}
}

func TestRouterRuntimeEmbeddingTasksRejectMissingLocalConsumerModelPath(t *testing.T) {
	tests := []struct {
		model       string
		missingPath string
	}{
		{model: "qwen3", missingPath: "qwen3_model_path"},
		{model: "gemma", missingPath: "gemma_model_path"},
		{model: "mmbert", missingPath: "mmbert_model_path"},
		{model: "multimodal", missingPath: "multimodal_model_path"},
	}
	plan := embedding.RuntimePlan{
		Backend:   config.EmbeddingBackendOpenAICompatible,
		ModelType: config.EmbeddingModelTypeRemote,
	}
	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			cfg := remoteEmbeddingRuntimeConfig("http://embedding-service:8000/v1")
			cfg.MmBertModelPath = ""
			cfg.Memory.Enabled = true
			cfg.Memory.EmbeddingModel = tt.model

			_, tasks, _, err := routerRuntimeEmbeddingTasks(cfg, "test", resolveEmbeddingPaths(cfg), plan)
			if err == nil || !strings.Contains(err.Error(), tt.missingPath) {
				t.Fatalf("routerRuntimeEmbeddingTasks() error = %v, want missing %s", err, tt.missingPath)
			}
			if len(tasks) != 0 {
				t.Fatalf("tasks = %+v, want none after validation failure", tasks)
			}
		})
	}
}

func TestClassifierEmbeddingPathsIncludeDistinctContrastivePreferenceModel(t *testing.T) {
	t.Setenv("EMBEDDING_MODEL_TYPE_OVERRIDE", "")
	t.Setenv("EMBEDDING_MODEL_OVERRIDE", "")
	useContrastive := true
	cfg := remoteEmbeddingRuntimeConfig("http://embedding-service:8000/v1")
	cfg.EmbeddingModels.EmbeddingConfig = config.HNSWConfig{
		Backend:   config.EmbeddingBackendCandle,
		ModelType: config.EmbeddingModelTypeQwen3,
	}
	cfg.Qwen3ModelPath = "models/qwen3"
	cfg.MmBertModelPath = "models/mmbert"
	cfg.ReaskRules = []config.ReaskRule{{Name: "repeat", LookbackTurns: 1}}
	cfg.PreferenceRules = []config.PreferenceRule{{Name: "secure"}}
	cfg.PreferenceModel.UseContrastive = &useContrastive
	cfg.PreferenceModel.EmbeddingModel = "mmbert"

	paths, err := classifierEmbeddingPaths(cfg, resolveEmbeddingPaths(cfg), config.EmbeddingBackendCandle)
	if err != nil {
		t.Fatalf("classifierEmbeddingPaths: %v", err)
	}
	want := embeddingPaths{
		qwen3:  config.ResolveModelPath("models/qwen3"),
		mmBert: config.ResolveModelPath("models/mmbert"),
	}
	if paths != want {
		t.Fatalf("classifier paths = %+v, want %+v", paths, want)
	}

	plan := embedding.RuntimePlan{
		Backend:   config.EmbeddingBackendCandle,
		ModelType: config.EmbeddingModelTypeQwen3,
		ModelPath: cfg.Qwen3ModelPath,
	}
	_, tasks, _, err := routerRuntimeEmbeddingTasks(cfg, "test", resolveEmbeddingPaths(cfg), plan)
	if err != nil {
		t.Fatalf("routerRuntimeEmbeddingTasks: %v", err)
	}
	assertRuntimeTaskManifest(t, tasks, []string{"router.embedding.unified_factory"}, []string{"router.embedding.unified_factory"})
}

func TestToolsMultimodalReadinessUsesEffectiveRuntimePlan(t *testing.T) {
	if !toolsUseMultiModalEmbeddings(embedding.RuntimePlan{ModelType: "multimodal"}) {
		t.Fatal("effective multimodal plan was not recognized")
	}
	if toolsUseMultiModalEmbeddings(embedding.RuntimePlan{ModelType: config.EmbeddingModelTypeQwen3}) {
		t.Fatal("effective qwen3 plan was treated as multimodal")
	}
}

func runtimeTaskManifestCases() []runtimeTaskManifestCase {
	cases := runtimeTaskManifestCoreCases()
	return append(cases, runtimeTaskManifestUnionCases()...)
}

func runtimeTaskManifestCoreCases() []runtimeTaskManifestCase {
	remotePlan := embedding.RuntimePlan{
		Backend:   config.EmbeddingBackendOpenAICompatible,
		ModelType: config.EmbeddingModelTypeRemote,
	}
	defaultBERT := config.ResolveModelPath("sentence-transformers/all-MiniLM-L6-v2")
	return []runtimeTaskManifestCase{
		{
			name: "remote plan and semantic cache default BERT",
			configure: func(cfg *config.RouterConfig) {
				cfg.Enabled = true
				cfg.MmBertModelPath = ""
			},
			plan:              remotePlan,
			wantConsumerPaths: embeddingPaths{bert: defaultBERT},
			wantTasks:         []string{"router.embedding.remote_provider", "router.local_consumers.bert"},
			wantRequiredTasks: []string{"router.local_consumers.bert"},
		},
		{
			name: "remote plan and memory mmbert",
			configure: func(cfg *config.RouterConfig) {
				cfg.Memory.Enabled = true
				cfg.Memory.EmbeddingModel = "mmbert"
			},
			plan:              remotePlan,
			wantConsumerPaths: embeddingPaths{mmBert: config.ResolveModelPath("models/mmbert-embed-32k-2d-matryoshka")},
			wantTasks:         []string{"router.embedding.remote_provider", "router.local_consumers.unified_factory"},
			wantRequiredTasks: []string{"router.local_consumers.unified_factory"},
		},
		{
			name: "local qwen override and vector store BERT",
			configure: func(cfg *config.RouterConfig) {
				cfg.Qwen3ModelPath = "models/local-qwen3"
				cfg.VectorStore = localVectorStoreConfig("bert")
			},
			plan: embedding.RuntimePlan{
				Backend:       config.EmbeddingBackendCandle,
				ModelType:     config.EmbeddingModelTypeQwen3,
				ModelPath:     "models/local-qwen3",
				LocalOverride: true,
			},
			wantConsumerPaths: embeddingPaths{bert: defaultBERT},
			wantTasks:         []string{"router.embedding.unified_factory", "router.embedding.bert"},
			wantRequiredTasks: []string{"router.embedding.bert"},
		},
	}
}

func runtimeTaskManifestUnionCases() []runtimeTaskManifestCase {
	remotePlan := embedding.RuntimePlan{
		Backend:   config.EmbeddingBackendOpenAICompatible,
		ModelType: config.EmbeddingModelTypeRemote,
	}
	return []runtimeTaskManifestCase{
		{
			name: "three consumers use the union of distinct models",
			configure: func(cfg *config.RouterConfig) {
				cfg.Qwen3ModelPath = "models/local-qwen3"
				cfg.Enabled = true
				cfg.EmbeddingModel = "bert"
				cfg.VectorStore = localVectorStoreConfig("qwen3")
				cfg.Memory.Enabled = true
				cfg.Memory.EmbeddingModel = "mmbert"
			},
			plan: remotePlan,
			wantConsumerPaths: embeddingPaths{
				qwen3:  config.ResolveModelPath("models/local-qwen3"),
				mmBert: config.ResolveModelPath("models/mmbert-embed-32k-2d-matryoshka"),
				bert:   config.ResolveModelPath("sentence-transformers/all-MiniLM-L6-v2"),
			},
			wantTasks: []string{
				"router.embedding.remote_provider",
				"router.local_consumers.unified_factory",
				"router.local_consumers.bert",
			},
			wantRequiredTasks: []string{
				"router.local_consumers.unified_factory",
				"router.local_consumers.bert",
			},
		},
		{
			name: "shared model is initialized once",
			configure: func(cfg *config.RouterConfig) {
				cfg.Enabled = true
				cfg.EmbeddingModel = "mmbert"
				cfg.VectorStore = localVectorStoreConfig("mmbert")
				cfg.Memory.Enabled = true
				cfg.Memory.EmbeddingModel = "mmbert"
			},
			plan:              remotePlan,
			wantConsumerPaths: embeddingPaths{mmBert: config.ResolveModelPath("models/mmbert-embed-32k-2d-matryoshka")},
			wantTasks:         []string{"router.embedding.remote_provider", "router.local_consumers.unified_factory"},
			wantRequiredTasks: []string{"router.local_consumers.unified_factory"},
		},
		{
			name: "Llama Stack vector store owns its embedder",
			configure: func(cfg *config.RouterConfig) {
				cfg.VectorStore = &config.VectorStoreConfig{
					Enabled:        true,
					BackendType:    "llama_stack",
					EmbeddingModel: "bert",
				}
			},
			plan:              remotePlan,
			wantConsumerPaths: embeddingPaths{},
			wantTasks:         []string{"router.embedding.remote_provider"},
		},
	}
}

func localVectorStoreConfig(model string) *config.VectorStoreConfig {
	return &config.VectorStoreConfig{
		Enabled:        true,
		BackendType:    "memory",
		EmbeddingModel: model,
	}
}

func assertRuntimeTaskManifest(t *testing.T, tasks []Task, wantNames, wantRequiredNames []string) {
	t.Helper()
	if len(tasks) != len(wantNames) {
		t.Fatalf("task count = %d, want %d; tasks = %+v", len(tasks), len(wantNames), tasks)
	}
	required := make(map[string]struct{}, len(wantRequiredNames))
	for _, name := range wantRequiredNames {
		required[name] = struct{}{}
	}
	seen := make(map[string]struct{}, len(tasks))
	for i, task := range tasks {
		if task.Name != wantNames[i] {
			t.Fatalf("task[%d] = %q, want %q", i, task.Name, wantNames[i])
		}
		if _, duplicate := seen[task.Name]; duplicate {
			t.Fatalf("duplicate task name %q", task.Name)
		}
		seen[task.Name] = struct{}{}
		_, mustSucceed := required[task.Name]
		if task.BestEffort == mustSucceed {
			t.Fatalf("task %q BestEffort = %t, required = %t", task.Name, task.BestEffort, mustSucceed)
		}
	}
}
