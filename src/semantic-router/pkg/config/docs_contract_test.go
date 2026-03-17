package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type docNeedles struct {
	path    string
	needles []string
}

func repoRel(parts ...string) string {
	return filepath.Join(parts...)
}

var configContractRequiredDocs = []docNeedles{
	{
		path: "config/README.md",
		needles: []string{
			"`config/config.yaml`",
			"exhaustive canonical reference config",
			"`config/signal/`",
			"`tutorials/signal/heuristic/`",
			"`tutorials/signal/learned/`",
			"`config/decision/`",
			"`config/algorithm/`",
			"`config/plugin/`",
			"`tutorials/global/`",
			"`go test ./pkg/config/...`",
			"`make agent-lint`",
		},
	},
	{
		path: repoRel("website", "docs", "installation", "configuration.md"),
		needles: []string{
			"`version/listeners/providers/routing/global`",
			"`routing.modelCards`",
			"`routing.modelCards[].loras`",
			"`providers.defaults`",
			"`providers.models[*]`",
			"`global.router`",
			"`global.router.config_source`",
			"`global.services`",
			"`global.stores`",
			"`global.integrations`",
			"`global.model_catalog`",
			"`global.model_catalog.modules`",
			"`config/algorithm/`",
			"`tutorials/global/`",
			"`tutorials/signal/heuristic/`",
			"`tutorials/signal/learned/`",
			"vllm-sr config migrate --config old-config.yaml",
			"v0.3. The steady-state file is `config.yaml`",
			"`lora_name`",
			"`make agent-lint`",
			"exhaustive canonical reference config",
			"`global.router.config_source: kubernetes`",
		},
	},
	{
		path: repoRel("website", "docs", "proposals", "unified-config-contract-v0-3.md"),
		needles: []string{
			"version:\nlisteners:\nproviders:\nrouting:\nglobal:",
			"`routing.modelCards`",
			"`routing.modelCards[].loras`",
			"`config/algorithm/`",
			"`providers.defaults`",
			"`providers.models[].backend_refs[]`",
			"`lora_name`",
			"`global.router.config_source`",
			"vllm-sr init",
			"exhaustive canonical reference config",
			"`make agent-lint`",
		},
	},
	{
		path: repoRel("website", "docs", "installation", "milvus.md"),
		needles: []string{
			"global:\n  stores:\n    semantic_cache:",
		},
	},
	{
		path: repoRel("website", "docs", "overview", "mom-model-family.md"),
		needles: []string{
			"`global.model_catalog`",
			"`global.model_catalog.modules`",
			"model_ref",
		},
	},
	{
		path: repoRel("website", "docs", "training", "ml-model-selection.md"),
		needles: []string{
			"global:\n  router:\n    model_selection:",
			"routing:\n  decisions:",
		},
	},
	{
		path: repoRel("website", "docs", "training", "model-performance-eval.md"),
		needles: []string{
			"listeners: []",
			"providers:\n  defaults:",
			"routing:\n  modelCards:",
			"model_catalog:",
			"modules:",
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
		path: repoRel("website", "i18n", "zh-Hans", "docusaurus-plugin-content-docs", "current", "proposals", "hallucination-mitigation-milestone.md"),
		needles: []string{
			"global:\n  model_catalog:\n    modules:\n      hallucination_mitigation:",
			"routing:\n  decisions:",
			"hallucination_action:",
		},
	},
	{
		path: repoRel("website", "docs", "api", "router.md"),
		needles: []string{
			"providers:\n  models:",
			"pricing:",
			"`providers.models[].backend_refs[]`",
		},
	},
	{
		path: repoRel("website", "docs", "api", "classification.md"),
		needles: []string{
			"global:\n  model_catalog:\n    modules:\n      classifier:",
			"routing:\n  signals:\n    domains:",
			"decisions:",
		},
	},
	{
		path: repoRel("website", "docs", "troubleshooting", "common-errors.md"),
		needles: []string{
			"backend_refs:",
			"`10.0.0.1:8000`",
			"[config/config.yaml]",
			"global:\n  stores:\n    semantic_cache:",
			"global:\n  model_catalog:\n    modules:\n      classifier:",
			"routing:\n  decisions:",
		},
	},
	{
		path: repoRel("website", "docs", "overview", "semantic-router-overview.md"),
		needles: []string{
			"routing:\n  decisions:",
			"      plugins:",
		},
	},
	{
		path: repoRel("website", "docs", "overview", "collective-intelligence.md"),
		needles: []string{
			"routing:\n  decisions:",
			"      plugins:",
		},
	},
	{
		path: repoRel("src", "vllm-sr", "README.md"),
		needles: []string{
			"`routing.decisions[]`",
			"routing:\n  decisions:",
			"vllm-sr config migrate --config old-config.yaml",
		},
	},
	{
		path: repoRel("bench", "README.md"),
		needles: []string{
			"`vsr_canonical_patch.yaml`",
			"`vsr_canonical_patch_recommendation.json`",
			"providers:\n  defaults:\n    reasoning_families:",
			"routing:\n  modelCards:",
			"routing:\n  decisions:",
			"default_reasoning_effort: medium",
		},
	},
	{
		path: repoRel("bench", "hallucination", "README.md"),
		needles: []string{
			"providers:\n  models:",
			"backend_refs:",
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
		path: repoRel("website", "i18n", "zh-Hans", "docusaurus-plugin-content-docs", "current", "api", "classification.md"),
		needles: []string{
			"global:\n  model_catalog:\n    modules:\n      classifier:",
			"routing:\n  signals:\n    domains:",
			"decisions:",
		},
	},
	{
		path: repoRel("website", "i18n", "zh-Hans", "docusaurus-plugin-content-docs", "current", "proposals", "nvidia-dynamo-integration.md"),
		needles: []string{
			"global:\n  model_catalog:\n    modules:\n      classifier:",
			"global:\n  model_catalog:\n    modules:\n      prompt_guard:",
		},
	},
	{
		path: repoRel("website", "docs", "installation", "k8s", "operator.md"),
		needles: []string{
			"providers:\n      defaults:",
			"global:\n      model_catalog:",
			"      stores:",
			"        modules:",
			"      services:",
			"loras:",
		},
	},
	{
		path: "deploy/helm/README.md",
		needles: []string{
			"providers:\n    defaults:",
		},
	},
	{
		path: "deploy/addons/mcp-classifier-server/README.md",
		needles: []string{
			"providers:\n  defaults:",
			"routing:\n  modelCards:",
			"global:\n  model_catalog:\n    modules:\n      classifier:",
		},
	},
	{
		path: repoRel("src", "semantic-router", "pkg", "modelselection", "README.md"),
		needles: []string{
			"global:\n  router:\n    model_selection:",
			"providers:\n  models:",
			"backend_refs:",
			"routing:\n  decisions:",
		},
	},
}

var configContractForbiddenDocs = []docNeedles{
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
			"\nmodel_selection:\n",
			"\nembedding_models:\n",
		},
	},
	{
		path: repoRel("website", "docs", "training", "model-performance-eval.md"),
		needles: []string{
			"\nvllm_endpoints:\n",
			"\nmodel_config:\n",
			"\nprompt_guard:\n",
			"\nclassifier:\n",
		},
	},
	{
		path: repoRel("website", "docs", "training", "training-overview.md"),
		needles: []string{
			"\nvllm_endpoints:\n",
			"\nmodel_config:\n",
			"\nprovider_profiles:\n",
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
		path: repoRel("website", "i18n", "zh-Hans", "docusaurus-plugin-content-docs", "current", "proposals", "hallucination-mitigation-milestone.md"),
		needles: []string{
			"\nhallucination:\n",
			"\n  - name: \"medical_assistant\"\n",
		},
	},
	{
		path: repoRel("website", "i18n", "zh-Hans", "docusaurus-plugin-content-docs", "current", "training", "training-overview.md"),
		needles: []string{
			"\nvllm_endpoints:\n",
			"\nmodel_config:\n",
			"\nprovider_profiles:\n",
			"router-defaults.yaml",
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
		path: repoRel("website", "docs", "api", "classification.md"),
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
		},
	},
	{
		path: repoRel("website", "docs", "overview", "semantic-router-overview.md"),
		needles: []string{
			"\nplugins:\n",
		},
	},
	{
		path: repoRel("website", "docs", "overview", "collective-intelligence.md"),
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
		},
	},
	{
		path: repoRel("bench", "hallucination", "README.md"),
		needles: []string{
			"\nvllm_endpoints:\n",
			"\nmodel_config:\n",
			"\nhallucination_mitigation:\n",
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
		path: repoRel("website", "i18n", "zh-Hans", "docusaurus-plugin-content-docs", "current", "api", "classification.md"),
		needles: []string{
			"\nclassifier:\n",
			"\ncategories:\n",
			"\ndecisions:\n",
		},
	},
	{
		path: repoRel("website", "i18n", "zh-Hans", "docusaurus-plugin-content-docs", "current", "proposals", "nvidia-dynamo-integration.md"),
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
		path: "deploy/addons/mcp-classifier-server/README.md",
		needles: []string{
			"\nclassifier:\n",
			"categories: []",
		},
	},
	{
		path: repoRel("src", "semantic-router", "pkg", "modelselection", "README.md"),
		needles: []string{
			"\nvllm_endpoints:\n",
			"\nmodel_config:\n",
			"access_key:",
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
	{
		path: repoRel("website", "i18n", "zh-Hans", "docusaurus-plugin-content-docs", "current", "training", "training-overview.md"),
		needles: []string{
			"computer_science",
		},
	},
	{
		path: repoRel("website", "i18n", "zh-Hans", "docusaurus-plugin-content-docs", "current", "troubleshooting", "vsr-headers.md"),
		needles: []string{
			"computer_science",
		},
	},
}

var latestTutorialOverviewDocs = []docNeedles{
	{
		path: repoRel("website", "docs", "tutorials", "signal", "overview.md"),
		needles: []string{
			"`config/signal/`",
			"### Heuristic Signals",
			"### Learned Signals",
			"[Keyword](./heuristic/keyword)",
			"[Domain](./learned/domain)",
		},
	},
	{
		path: repoRel("website", "docs", "tutorials", "decision", "overview.md"),
		needles: []string{
			"`config/decision/`",
			"`decision.algorithm`",
			"`decision.plugins`",
		},
	},
	{
		path: repoRel("website", "docs", "tutorials", "algorithm", "overview.md"),
		needles: []string{
			"`config/algorithm/`",
			"### Selection Algorithms",
			"### Looper Algorithms",
			"[Static](./selection/static)",
			"[Confidence](./looper/confidence)",
		},
	},
	{
		path: repoRel("website", "docs", "tutorials", "plugin", "overview.md"),
		needles: []string{
			"`config/plugin/`",
			"`routing.decisions[].plugins`",
			"[Fast Response](./fast-response)",
			"[Semantic Cache](./semantic-cache)",
		},
	},
	{
		path: repoRel("website", "docs", "tutorials", "global", "overview.md"),
		needles: []string{
			"`global:`",
			"`signal/`",
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
	"label: 'Global'",
	"'tutorials/signal/overview'",
	"'tutorials/decision/overview'",
	"'tutorials/algorithm/overview'",
	"'tutorials/plugin/overview'",
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
	"'proposals/unified-config-contract-v0-3'",
}

var latestTutorialRequiredSections = []string{
	"## Overview",
	"## Key Advantages",
	"## What Problem Does It Solve?",
	"## When to Use",
	"## Configuration",
}

var latestTutorialAllowedDirectories = map[string]bool{
	"signal":    true,
	"decision":  true,
	"algorithm": true,
	"plugin":    true,
	"global":    true,
}

var retiredCurrentTranslationOverrides = []string{
	repoRel("website", "i18n", "zh-Hans", "docusaurus-plugin-content-docs", "current", "cookbook", "classifier-tuning.md"),
	repoRel("website", "i18n", "zh-Hans", "docusaurus-plugin-content-docs", "current", "cookbook", "pii-policy.md"),
	repoRel("website", "i18n", "zh-Hans", "docusaurus-plugin-content-docs", "current", "cookbook", "vllm-endpoints.md"),
	repoRel("website", "i18n", "zh-Hans", "docusaurus-plugin-content-docs", "current", "training", "model-performance-eval.md"),
	repoRel("website", "i18n", "zh-Hans", "docusaurus-plugin-content-docs", "current", "troubleshooting", "common-errors.md"),
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
	assertTutorialFilesContainRequiredSections(t, root)
	assertTutorialRootDirectories(t, root)
	assertSignalTutorialDocsMatchConfigHierarchy(t, root)
	assertAlgorithmTutorialDocsMatchConfigHierarchy(t, root)
	assertPluginTutorialDocsMatchConfigHierarchy(t, root)
	assertPathsDoNotExist(t, root, retiredCurrentTranslationOverrides)
}

func TestConfigProposalIsReachableFromSidebar(t *testing.T) {
	root := repoRootFromTestFile(t)
	sidebarPath := repoRel("website", "sidebars.ts")
	content := readRepoFile(t, root, sidebarPath)
	assertStringContainsAll(t, content, sidebarPath, proposalSidebarRequired)
}

func assertDocsContainAll(t *testing.T, root string, docs []docNeedles) {
	t.Helper()
	for _, doc := range docs {
		assertStringContainsAll(t, readRepoFile(t, root, doc.path), doc.path, doc.needles)
	}
}

func assertDocsDoNotContainAny(t *testing.T, root string, docs []docNeedles) {
	t.Helper()
	for _, doc := range docs {
		assertStringContainsNone(t, readRepoFile(t, root, doc.path), doc.path, doc.needles)
	}
}

func assertTutorialSidebarTaxonomy(t *testing.T, root string) {
	t.Helper()
	sidebarPath := repoRel("website", "sidebars.ts")
	content := readRepoFile(t, root, sidebarPath)
	required := append([]string(nil), latestTutorialSidebarRequired...)
	required = append(required, signalTutorialSidebarEntries()...)
	required = append(required, algorithmTutorialSidebarEntries()...)
	required = append(required, pluginTutorialSidebarEntries()...)
	assertStringContainsAll(t, content, sidebarPath, required)
	assertStringContainsNone(t, content, sidebarPath, latestTutorialSidebarForbidden)
}

func assertTutorialFilesContainRequiredSections(t *testing.T, root string) {
	t.Helper()
	tutorialRoot := filepath.Join(root, repoRel("website", "docs", "tutorials"))
	err := filepath.Walk(tutorialRoot, func(path string, info os.FileInfo, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if info.IsDir() || filepath.Ext(path) != ".md" {
			return nil
		}
		contentBytes, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		assertStringContainsAll(t, string(contentBytes), path, latestTutorialRequiredSections)
		return nil
	})
	if err != nil {
		t.Fatalf("failed to walk latest tutorial files: %v", err)
	}
}

func assertTutorialRootDirectories(t *testing.T, root string) {
	t.Helper()
	tutorialRoot := filepath.Join(root, repoRel("website", "docs", "tutorials"))
	entries, err := os.ReadDir(tutorialRoot)
	if err != nil {
		t.Fatalf("failed to read latest tutorial root: %v", err)
	}

	allowed := copyStringBoolMap(latestTutorialAllowedDirectories)
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		if !allowed[entry.Name()] {
			t.Fatalf("%s contains retired top-level directory %q", tutorialRoot, entry.Name())
		}
		delete(allowed, entry.Name())
	}
	for remaining := range allowed {
		t.Fatalf("%s is missing required top-level directory %q", tutorialRoot, remaining)
	}
}

func assertMarkdownTreeDoesNotContainAny(t *testing.T, root string, forbidden []string) {
	t.Helper()
	if _, err := os.Stat(root); os.IsNotExist(err) {
		return
	}
	err := filepath.Walk(root, func(path string, info os.FileInfo, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if info.IsDir() || filepath.Ext(path) != ".md" {
			return nil
		}
		contentBytes, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		assertStringContainsNone(t, string(contentBytes), path, forbidden)
		return nil
	})
	if err != nil {
		t.Fatalf("failed to walk tutorial docs under %s: %v", root, err)
	}
}

func assertPathsDoNotExist(t *testing.T, root string, relPaths []string) {
	t.Helper()
	for _, relPath := range relPaths {
		fullPath := filepath.Join(root, relPath)
		if _, err := os.Stat(fullPath); err == nil {
			t.Fatalf("%s should not exist; let latest docs fall back to the canonical current source instead", relPath)
		} else if !os.IsNotExist(err) {
			t.Fatalf("failed to stat %s: %v", relPath, err)
		}
	}
}

func tutorialDocRoots(root string) []string {
	return []string{
		filepath.Join(root, repoRel("website", "docs", "tutorials")),
		filepath.Join(root, repoRel("website", "i18n", "zh-Hans", "docusaurus-plugin-content-docs", "current", "tutorials")),
	}
}

func readRepoFile(t *testing.T, root string, relPath string) string {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(root, relPath))
	if err != nil {
		t.Fatalf("failed to read %s: %v", relPath, err)
	}
	return string(data)
}

func assertStringContainsAll(t *testing.T, content string, label string, needles []string) {
	t.Helper()
	for _, needle := range needles {
		if !strings.Contains(content, needle) {
			t.Fatalf("%s is missing required text %q", label, needle)
		}
	}
}

func assertStringContainsNone(t *testing.T, content string, label string, needles []string) {
	t.Helper()
	for _, needle := range needles {
		if strings.Contains(content, needle) {
			t.Fatalf("%s still contains retired text %q", label, needle)
		}
	}
}

func copyStringBoolMap(source map[string]bool) map[string]bool {
	clone := make(map[string]bool, len(source))
	for key, value := range source {
		clone[key] = value
	}
	return clone
}
