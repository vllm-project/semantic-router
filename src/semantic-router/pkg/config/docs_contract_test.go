package config

import (
	"path/filepath"
	"testing"
)

type docNeedles struct {
	path    string
	needles []string
}

func repoRel(parts ...string) string {
	return filepath.Join(parts...)
}

// These stay unfenced so the assertion survives the page moving an endpoint between
// prose and a table, which is what broke it in #2773.
var apiserverDocNeedles = []string{
	"http://localhost:8080",
	"/openapi.json",
	"one read-only manifest",
	"/management/v1",
}

var configContractRequiredDocs = []docNeedles{
	{
		path: "config/README.md",
		needles: []string{
			"version: v0.3",
			"`providers.models`",
			"structured invocation `control`",
			"`global.billing.currency`",
			"`routing.modelCards`",
			"`model_names`",
			"`recipes[].routing.projections`",
			"`config/config.yaml`",
			"exhaustive canonical reference config",
			"`config/fragments/signal/`",
			"`tutorials/signal/heuristic/`",
			"`tutorials/signal/learned/`",
			"`config/fragments/decision/`",
			"`config/fragments/algorithm/`",
			"`config/fragments/plugin/`",
			"`tutorials/global/`",
			"`go test ./pkg/config/...`",
			"`make agent-lint`",
		},
	},
	{
		path: repoRel("website", "docs", "installation", "configuration.md"),
		needles: []string{
			"version:\nlisteners:\nproviders:\nrouting:\nrecipes:\nentrypoints:\nglobal:",
			"`entrypoints[].model_names`",
			"backend_refs:",
			"control:",
			"global.billing.currency",
			"vllm-sr validate --config config.yaml",
			"Environment references and secrets",
			"Entrypoints and recipes",
			"exhaustive canonical example",
			"`config/fragments/`",
		},
	},
	{
		path: repoRel("website", "docs", "installation", "configuration-workflows.md"),
		needles: []string{
			"`global.stores.management`",
			"vllm-sr serve --target k8s",
			"immutable routing authority",
			"Routing DSL",
			"Avoid split ownership",
		},
	},
	{
		path: repoRel("website", "docs", "tutorials", "global", "models-entrypoints-serving.md"),
		needles: []string{
			"vllm-sr serve",
			"Build → Models",
			"Build → Recipes",
			"Build → Mixture of Models",
			"Router Management API",
			"vllm-sr status",
			"vllm-sr logs router",
			"vllm-sr stop",
			"curl -sS http://localhost:8899/v1/models",
			"VLLM_SR_API_KEY",
			"vllm-sr recipe pack",
			"vllm-sr serve --config /path/to/config.yaml",
		},
	},
	{
		path: repoRel("website", "docs", "installation", "milvus.md"),
		needles: []string{
			"global:\n  stores:\n    response_cache:",
		},
	},
	{
		path: repoRel("website", "docs", "overview", "mom-model-family.md"),
		needles: []string{
			"Mixture of Experts",
			"**Model**",
			"**Virtual model**",
			"**Router system model**",
			"`vllm-sr/mom-v1-blend`",
		},
	},
	{
		path: repoRel("website", "docs", "training", "ml-model-selection.md"),
		needles: []string{
			"`algorithm.ml`",
			"recipes:\n  - name: ml-selection",
			"routing:",
			"model_names:",
			"assignments:",
		},
	},
	{
		path: repoRel("website", "docs", "tutorials", "global", "api-and-observability.md"),
		needles: []string{
			"`global.stores.management`",
			"`providers.models[].pricing`",
			"GET /management/v1/usage",
			"global:\n  services:\n    observability:",
		},
	},
	{
		path: repoRel("website", "docs", "training", "model-performance-eval.md"),
		needles: []string{
			"listeners: []",
			"providers:\n  models: []",
			"routing:\n  modelCards: []",
			"recipes: []",
			"entrypoints: []",
			"one provider binding, structured invocation control, and connection-free",
		},
	},
	{
		path: repoRel("website", "docs", "proposals", "hallucination-mitigation-milestone.md"),
		needles: []string{
			"global:\n  model_catalog:\n    modules:\n      hallucination_mitigation:",
			"routing:\n  decisions:",
			"hallucination_action:",
		},
	},
	{
		path: repoRel("website", "docs", "api", "router.md"),
		needles: []string{
			"providers:\n  models:\n    - name: local-small",
			"backend_refs:",
			"pricing:",
			"Entrypoint's `model_names`",
		},
	},
	{
		path:    repoRel("website", "docs", "api", "apiserver.md"),
		needles: apiserverDocNeedles,
	},
	{
		path: repoRel("website", "docs", "troubleshooting", "common-errors.md"),
		needles: []string{
			"backend_refs:",
			"endpoint: http://10.0.0.1:8000/v1",
			"[config/config.yaml]",
			"global:\n  stores:\n    response_cache:",
			"global:\n  model_catalog:\n    modules:\n      classifier:",
			"routing:\n  decisions:",
		},
	},
	{
		path: repoRel("website", "docs", "overview", "semantic-router-overview.md"),
		needles: []string{
			"Envoy presents the request to the Router.",
			"**Entrypoint**",
			"**Recipe**",
			"direct selection",
			"configured default provider model",
		},
	},
	{
		path: repoRel("src", "vllm-sr", "README.md"),
		needles: []string{
			"`recipes[].routing.decisions[]`",
			"recipes:\n  - name: local",
			"`providers.models[].backend_refs`",
		},
	},
	{
		path: repoRel("bench", "README.md"),
		needles: []string{
			"`vsr_canonical_patch.yaml`",
			"`vsr_canonical_patch_recommendation.json`",
			"providers:",
			"backend_refs:",
			"modelCards:",
			"entrypoints:",
			"model_names:",
			"reasoning: {enabled: true, effort: high}",
		},
	},
	{
		path: repoRel("bench", "hallucination", "README.md"),
		needles: []string{
			"providers:",
			"backend_refs:",
			"modelCards:",
			"global:\n  model_catalog:\n    modules:\n      prompt_guard:",
			"global:\n  model_catalog:\n    modules:\n      hallucination_mitigation:",
		},
	},
	{
		path: repoRel("bench", "cpu-vs-gpu", "README.md"),
		needles: []string{
			"`config-bench.yaml`",
			"`config-bench-candle.yaml`",
			"`global.router.streamed_body.enabled`",
			"`bench-3way.sh`",
		},
	},
	{
		path: repoRel("website", "docs", "proposals", "nvidia-dynamo-integration.md"),
		needles: []string{
			"global:\n  model_catalog:\n    modules:\n      classifier:",
			"global:\n  model_catalog:\n    modules:\n      prompt_guard:",
		},
	},
	{
		path: repoRel("website", "docs", "installation", "k8s", "operator.md"),
		needles: []string{
			"kind: SemanticRouter",
			"`spec.bootstrap.configMapRef`",
			"immutable: true",
			"`global.stores.management.postgres`",
			"subsequent desired-state changes use the",
			"does not create an",
			"secretKeyRef:",
		},
	},
	{
		path: repoRel("website", "docs", "api", "semantic-router-crd.md"),
		needles: []string{
			"kind: SemanticRouter",
			"kubectl explain semanticrouter.spec --recursive",
			"`bootstrap`",
			"`bootstrap.configMapRef`",
			"immutable: true",
			"Management API owns",
			"`status.observedGeneration`",
		},
	},
	{
		path: "deploy/helm/README.md",
		needles: []string{
			"version: v0.3",
			"providers:",
			"backend_refs:",
			"control:",
			"modelCards:",
			"recipes:",
			"routing:",
			"entrypoints:",
			"model_names:",
			"assignments:",
		},
	},
	{
		path: "tools/mcp-classifier-server/README.md",
		needles: []string{
			"version: v0.3",
			"providers:",
			"backend_refs:",
			"control:",
			"modelCards:",
			"recipes:",
			"routing:",
			"entrypoints:",
			"model_names:",
			"assignments:",
			"global:\n  model_catalog:\n    modules:\n      classifier:",
		},
	},
	{
		path: repoRel("src", "semantic-router", "pkg", "modelselection", "README.md"),
		needles: []string{
			"`algorithm.ml`",
			"providers:",
			"backend_refs:",
			"control:",
			"modelCards:",
			"recipes:",
			"routing:",
			"entrypoints:",
			"model_names:",
			"assignments:",
		},
	},
}

var configContractForbiddenDocs = []docNeedles{
	{
		path: "config/README.md",
		needles: []string{
			"version: v0.4",
			"\nmodels:\n",
			"connections:",
			"recipes[].document",
			"model_bindings",
			"control_plane:",
		},
	},
	{
		path: repoRel("website", "docs", "installation", "configuration.md"),
		needles: []string{
			"model_bindings",
			"control_plane:",
			"providers.models[].runtime",
			"providers.models[].reliability",
		},
	},
	{
		path: repoRel("website", "docs", "installation", "milvus.md"),
		needles: []string{
			"global:\n  semantic_cache:",
		},
	},
	{
		path: repoRel("website", "docs", "overview", "mom-model-family.md"),
		needles: []string{
			"global:\n  classifier:",
			"global:\n  prompt_guard:",
			"global:\n  modules:",
		},
	},
	{
		path: repoRel("website", "docs", "training", "ml-model-selection.md"),
		needles: []string{
			"config:\n  model_selection:",
			"global:\n  router:\n    model_selection:",
			"\nmodel_selection:\n",
			"\nembedding_models:\n",
			"\n    document:\n",
			"standalone Router manifest",
			"Recipe document",
		},
	},
	{
		path: repoRel("website", "docs", "api", "apiserver.md"),
		needles: []string{
			"Standalone mode",
			"Managed mode",
			"active listener mode",
		},
	},
	{
		path: repoRel("website", "docs", "tutorials", "global", "api-and-observability.md"),
		needles: []string{
			"Standalone",
			"Managed deployments",
			"`models[].pricing`",
		},
	},
	{
		path: repoRel("website", "docs", "tutorials", "signal", "heuristic", "authz.md"),
		needles: []string{
			"Standalone routing",
		},
	},
	{
		path: repoRel("src", "semantic-router", "pkg", "modelselection", "README.md"),
		needles: []string{
			"global:\n  router:\n    model_selection:",
			"version: v0.4",
			"\nmodels:\n",
			"\n    document:\n",
			"connections:",
		},
	},
	{
		path: repoRel("website", "docs", "training", "model-performance-eval.md"),
		needles: []string{
			"\nvllm_endpoints:\n",
			"\nmodel_config:\n",
			"\nprompt_guard:\n",
			"\nclassifier:\n",
			"version: v0.4",
			"\nmodels:\n",
			"\n    document:\n",
			"connections:",
		},
	},
	{
		path: repoRel("website", "docs", "training", "training-overview.md"),
		needles: []string{
			"\nvllm_endpoints:\n",
			"\nmodel_config:\n",
			"router-defaults.yaml",
		},
	},
	{
		path: repoRel("website", "docs", "proposals", "hallucination-mitigation-milestone.md"),
		needles: []string{
			"\nhallucination:\n",
			"\n  - name: \"medical_assistant\"\n",
		},
	},
	{
		path: repoRel("website", "docs", "api", "router.md"),
		needles: []string{
			"\nmodel_config:\n",
			"vllm_endpoints[].models",
		},
	},
	{
		path: repoRel("website", "docs", "api", "apiserver.md"),
		needles: []string{
			"\nclassifier:\n",
			"\ncategories:\n",
			"\ndecisions:\n",
		},
	},
	{
		path: repoRel("website", "docs", "troubleshooting", "common-errors.md"),
		needles: []string{
			"\nvllm_endpoints:\n",
			"#vllm_endpoints",
			"\nsemantic_cache:\n",
			"\nclassifier:\n",
			"\nplugins:\n",
			"\nmodels:\n",
			"\ndocument:\n",
			"connections:",
		},
	},
	{
		path: repoRel("website", "docs", "overview", "semantic-router-overview.md"),
		needles: []string{
			"\nplugins:\n",
		},
	},
	{
		path: repoRel("src", "vllm-sr", "README.md"),
		needles: []string{
			"\nplugins:\n",
			"make generate     - Generate configurations",
			"make show-config",
		},
	},
	{
		path: repoRel("bench", "README.md"),
		needles: []string{
			"\nmodel_config:\n",
			"vsr_model_config.yaml",
			"vsr_model_config_recommendation.json",
			"config.yaml model_config section",
			"preferred_endpoints:",
			"\ndefault_reasoning_effort:",
			"\ncategories:\n",
			"version: v0.4",
			"\nmodels:\n",
			"connections:",
		},
	},
	{
		path: repoRel("bench", "hallucination", "README.md"),
		needles: []string{
			"\nvllm_endpoints:\n",
			"\nmodel_config:\n",
			"\nhallucination_mitigation:\n",
			"version: v0.4",
			"\nmodels:\n",
			"connections:",
		},
	},
	{
		path: repoRel("bench", "cpu-vs-gpu", "README.md"),
		needles: []string{
			"streamed_body_mode",
		},
	},
	{
		path: repoRel("website", "docs", "proposals", "nvidia-dynamo-integration.md"),
		needles: []string{
			"\nclassifier:\n",
			"\nprompt_guard:\n",
		},
	},
	{
		path: repoRel("website", "docs", "installation", "k8s", "operator.md"),
		needles: []string{
			"spec:\n  config:\n    semantic_cache:",
			"spec:\n  config:\n    classifier:",
			"spec:\n  config:\n    prompt_guard:",
		},
	},
	{
		path: "deploy/helm/README.md",
		needles: []string{
			"providers:\n    default_model:",
		},
	},
	{
		path: "tools/mcp-classifier-server/README.md",
		needles: []string{
			"\nclassifier:\n",
			"categories: []",
			"version: v0.4",
			"\nmodels:\n",
			"\n    document:\n",
			"connections:",
		},
	},
	{
		path: repoRel("src", "semantic-router", "pkg", "modelselection", "README.md"),
		needles: []string{
			"\nvllm_endpoints:\n",
			"\nmodel_config:\n",
			"access_key:",
			"version: v0.4",
			"\nmodels:\n",
			"\n    document:\n",
			"connections:",
		},
	},
	{
		path: repoRel("website", "docs", "tutorials", "algorithm", "overview.md"),
		needles: []string{
			"computer_science",
		},
	},
	{
		path: repoRel("website", "docs", "overview", "signal-driven-decisions.md"),
		needles: []string{
			"computer_science",
		},
	},
	{
		path: repoRel("website", "docs", "training", "training-overview.md"),
		needles: []string{
			"computer_science",
		},
	},
	{
		path: repoRel("website", "docs", "troubleshooting", "vsr-headers.md"),
		needles: []string{
			"computer_science",
		},
	},
	{
		path: repoRel("website", "docs", "proposals", "nvidia-dynamo-integration.md"),
		needles: []string{
			"computer_science",
		},
	},
}

var latestTutorialOverviewDocs = []docNeedles{
	{
		path: repoRel("website", "docs", "tutorials", "signal", "overview.md"),
		needles: []string{
			"Signals turn request facts into names",
			"### Heuristic Signals",
			"### Learned Signals",
			"[Keyword](./heuristic/keyword)",
			"[Domain](./learned/domain)",
		},
	},
	{
		path: repoRel("website", "docs", "tutorials", "decision", "overview.md"),
		needles: []string{
			"Signals tell the Router what it detected",
			"`recipes[].routing.decisions`",
			"`decision.algorithm`",
			"`decision.plugins`",
		},
	},
	{
		path: repoRel("website", "docs", "tutorials", "algorithm", "overview.md"),
		needles: []string{
			"An algorithm runs after a decision matches",
			"### Selection Algorithms",
			"### Looper Algorithms",
			"[Static](./selection/static)",
			"[Confidence](./looper/confidence)",
		},
	},
	{
		path: repoRel("website", "docs", "tutorials", "plugin", "overview.md"),
		needles: []string{
			"Plugins add route-local behavior after a decision matches",
			"`recipes[].routing.decisions[].plugins`",
			"[Fast Response](./fast-response)",
			"[Response Cache](./response-cache)",
		},
	},
	{
		path: repoRel("website", "docs", "tutorials", "global", "overview.md"),
		needles: []string{
			"`global:`",
			"`global.services`",
			"`global.stores`",
		},
	},
}

var latestTutorialOverviewForbidden = []docNeedles{
	{
		path:    repoRel("website", "docs", "tutorials", "signal", "overview.md"),
		needles: []string{"`config/fragments/signal/`"},
	},
	{
		path: repoRel("website", "docs", "tutorials", "decision", "overview.md"),
		needles: []string{
			"`config/fragments/decision/`",
			"`recipes[].document",
			"\ndocument:\n",
		},
	},
	{
		path:    repoRel("website", "docs", "tutorials", "algorithm", "overview.md"),
		needles: []string{"`config/fragments/algorithm/`"},
	},
	{
		path: repoRel("website", "docs", "tutorials", "plugin", "overview.md"),
		needles: []string{
			"`config/fragments/plugin/`",
			"`recipes[].document",
			"\ndocument:\n",
		},
	},
}

var latestTutorialSidebarRequired = []string{
	"label: 'Signals'",
	"label: 'Heuristic'",
	"label: 'Learned'",
	"label: 'Decisions'",
	"label: 'Algorithms'",
	"label: 'Selection'",
	"label: 'Looper'",
	"label: 'Plugins'",
	"label: 'Response and Mutation'",
	"label: 'Retrieval and Memory'",
	"label: 'Safety and Generation'",
	"label: 'Entrypoints'",
	"label: 'Shared Services'",
	"'tutorials/signal/overview'",
	"'tutorials/decision/overview'",
	"'tutorials/algorithm/overview'",
	"'tutorials/plugin/overview'",
	"'tutorials/global/entrypoints-and-recipes'",
	"'tutorials/global/models-entrypoints-serving'",
	"'tutorials/global/entrypoints'",
	"'tutorials/global/recipes'",
	"'tutorials/global/playground-builder'",
	"'tutorials/global/overview'",
}

var latestTutorialSidebarForbidden = []string{
	"'tutorials/signal/routing'",
	"'tutorials/signal/safety'",
	"'tutorials/signal/operational'",
	"'tutorials/intelligent-route/",
	"'tutorials/content-safety/",
	"'tutorials/semantic-cache/",
	"'tutorials/observability/",
	"'tutorials/response-api/",
	"'tutorials/performance-tuning/",
	"'tutorials/runtime/",
	"'tutorials/algorithm/selection'",
	"'tutorials/algorithm/looper'",
	"'tutorials/plugin/response-and-mutation'",
	"'tutorials/plugin/retrieval-and-memory'",
	"'tutorials/plugin/safety-and-generation'",
}

var proposalSidebarRequired = []string{
	"label: 'Proposals'",
	"'proposals/router-native-access-control'",
	"'proposals/multi-protocol-adaptor'",
	"'proposals/router-native-agent-runtime'",
}

var latestTutorialRequiredSections = []string{
	"## Overview",
	"## What Problem Does It Solve?",
	"## When to Use",
	"## Configuration",
}

var latestTutorialAllowedDirectories = map[string]bool{
	"signal":     true,
	"decision":   true,
	"algorithm":  true,
	"learning":   true,
	"plugin":     true,
	"global":     true,
	"projection": true,
}

// currentTranslationFallbackDocs are deliberately absent from the latest
// zh-Hans overrides. Docusaurus serves the canonical current English page when
// an override is missing; historical versioned translations remain untouched.
var currentTranslationFallbackDocs = []string{
	repoRel("website", "i18n", "zh-Hans", "docusaurus-plugin-content-docs", "current", "cookbook", "classifier-tuning.md"),
	repoRel("website", "i18n", "zh-Hans", "docusaurus-plugin-content-docs", "current", "cookbook", "pii-policy.md"),
	repoRel("website", "i18n", "zh-Hans", "docusaurus-plugin-content-docs", "current", "cookbook", "vllm-endpoints.md"),
	repoRel("website", "i18n", "zh-Hans", "docusaurus-plugin-content-docs", "current", "api", "apiserver.md"),
	repoRel("website", "i18n", "zh-Hans", "docusaurus-plugin-content-docs", "current", "training", "training-overview.md"),
	repoRel("website", "i18n", "zh-Hans", "docusaurus-plugin-content-docs", "current", "training", "model-performance-eval.md"),
	repoRel("website", "i18n", "zh-Hans", "docusaurus-plugin-content-docs", "current", "troubleshooting", "common-errors.md"),
	repoRel("website", "i18n", "zh-Hans", "docusaurus-plugin-content-docs", "current", "troubleshooting", "vsr-headers.md"),
}

func TestConfigContractDocsStayAligned(t *testing.T) {
	assertDocsContainAll(t, repoRootFromTestFile(t), configContractRequiredDocs)
}

func TestCurrentConfigDocsAvoidRetiredCanonicalExamples(t *testing.T) {
	assertDocsDoNotContainAny(t, repoRootFromTestFile(t), configContractForbiddenDocs)
}

func TestCurrentTutorialDocsDoNotReferenceRemovedConfigFiles(t *testing.T) {
	root := repoRootFromTestFile(t)
	for _, docRoot := range tutorialDocRoots(root) {
		assertMarkdownTreeDoesNotContainAny(t, docRoot, []string{"router-config.yaml", "router-defaults.yaml"})
	}
}

func TestLatestTutorialTaxonomyMatchesConfigHierarchy(t *testing.T) {
	root := repoRootFromTestFile(t)

	assertTutorialSidebarTaxonomy(t, root)
	assertDocsContainAll(t, root, latestTutorialOverviewDocs)
	assertDocsDoNotContainAny(t, root, latestTutorialOverviewForbidden)
	assertTutorialFilesContainRequiredSections(t, root)
	assertTutorialRootDirectories(t, root)
	assertSignalTutorialDocsMatchConfigHierarchy(t, root)
	assertAlgorithmTutorialDocsMatchConfigHierarchy(t, root)
	assertPluginTutorialDocsMatchConfigHierarchy(t, root)
	assertPathsDoNotExist(t, root, currentTranslationFallbackDocs)
}

func TestConfigProposalIsReachableFromSidebar(t *testing.T) {
	root := repoRootFromTestFile(t)
	sidebarPath := repoRel("website", "sidebars.ts")
	content := readRepoFile(t, root, sidebarPath)
	assertStringContainsAll(t, content, sidebarPath, proposalSidebarRequired)
}
