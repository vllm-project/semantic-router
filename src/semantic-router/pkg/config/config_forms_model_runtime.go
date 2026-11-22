package config

import (
	"fmt"
	"path"
	"strings"

	"gopkg.in/yaml.v3"
)

// modelRuntimeForms are the model execution settings from before the
// built-in model runtime served every local model: the candle, ONNX Runtime
// and OpenVINO providers with their execution fields and the earlier task
// models they ran, the runtime's numerics profiles, the Omni bundles router
// images shipped, and the retired NLI explainer with its fields.
var modelRuntimeForms = []ConfigForm{
	{
		Code:    CodeRemovedField,
		Paths:   []string{"global.model_catalog.deployments.*"},
		Match:   func(node *yaml.Node, _ string) bool { return retiredNLIModel(scalarAt(node, "artifact")) },
		What:    "the deployment of the retired NLI explainer is removed",
		Becomes: "drop it; nothing explains a hallucination with an NLI model",
		Migrate: func(m *Migration, at FormMatch) error {
			if model := providerModelOfDeployment(m.Root(), at.Key); model != "" {
				return fmt.Errorf("the retired NLI explainer's deployment is removed, but providers.models[%s].deployment names it; point the model at the deployment that serves it first", model)
			}
			m.Delete(at.Parent, at.Key)
			deleteEmpty(m.Root(), "global", "model_catalog", "deployments")
			return nil
		},
	},
	{
		Code:  CodeInvalidValue,
		Paths: []string{"global.model_catalog.deployments.*"},
		Match: func(node *yaml.Node, _ string) bool { return removedModelProvider(scalarAt(node, "provider")) },
		Describe: func(at FormMatch) (string, string) {
			return fmt.Sprintf("provider %s is removed: the built-in model runtime serves every local model", scalarAt(at.Node, "provider")),
				"provider: model_runtime, on its device, with its artifact's Hub repository"
		},
		Migrate: migrateRemovedModelProvider,
	},
	{
		Code:  CodeRemovedField,
		Paths: []string{"global.model_catalog.deployments.*.precision", "global.model_catalog.deployments.*.custom_ops_profile", "global.model_catalog.deployments.*.compilation_cache_dir"},
		Describe: func(at FormMatch) (string, string) {
			return at.Key + " is removed", "nothing; the model runtime selects numerics and kernels, and dtype sets a precision"
		},
		Migrate: migrateExecutionField,
	},
	{
		Code:    CodeRemovedField,
		Paths:   []string{"global.model_catalog.deployments.*.profile"},
		What:    "profile is removed",
		Becomes: "the engine arguments it stood for: max_num_seqs, max_num_batched_tokens, batch_window_ms, enable_shared_context, dtype",
		Migrate: migrateRuntimeProfile,
	},
	{
		Code:  CodeInvalidValue,
		Paths: []string{"global.model_catalog.deployments.*.artifact"},
		Match: func(node *yaml.Node, _ string) bool {
			return retiredOmniBundles[strings.TrimRight(node.Value, "/")] != ""
		},
		Describe: func(at FormMatch) (string, string) {
			return fmt.Sprintf("artifact %s is an Omni bundle router images no longer ship", at.Node.Value),
				fmt.Sprintf("artifact: %s, the published repository the runtime serves", retiredOmniBundles[strings.TrimRight(at.Node.Value, "/")])
		},
		Migrate: func(m *Migration, at FormMatch) error {
			m.Set(at.Parent, "artifact", Scalar(retiredOmniBundles[strings.TrimRight(at.Node.Value, "/")]))
			m.Delete(at.Parent, "revision")
			return nil
		},
	},
	{
		Code:    CodeRemovedField,
		Paths:   RoutingScopes("signals.hallucination[].use_nli"),
		What:    "use_nli is removed: NLI explanations are retired",
		Becomes: "nothing; the rule's spans stay in the response",
		Migrate: func(m *Migration, at FormMatch) error {
			m.Delete(at.Parent, "use_nli")
			return nil
		},
	},
	{
		Code:    CodeRemovedField,
		Paths:   RoutingScopes("decisions[].algorithm.fusion.grounding.nli_contradiction_penalty"),
		What:    "nli_contradiction_penalty is removed",
		Becomes: "contradiction_penalty; grounding reads the hallucination signal, not an NLI model",
		Migrate: func(m *Migration, at FormMatch) error {
			if mappingValue(at.Parent, "contradiction_penalty") != nil {
				m.Delete(at.Parent, "nli_contradiction_penalty")
				return nil
			}
			return m.Rename(at.Parent, "nli_contradiction_penalty", "contradiction_penalty")
		},
	},
	{
		Code:    CodeRemovedField,
		Paths:   []string{"global.stores.response_cache.polarity_guard"},
		What:    "polarity_guard is removed: the NLI tier is retired",
		Becomes: "nothing; the lexical polarity guard always runs",
		Migrate: func(m *Migration, at FormMatch) error {
			m.Delete(at.Parent, "polarity_guard")
			return nil
		},
	},
}

func removedModelProvider(provider string) bool {
	switch strings.ToLower(strings.TrimSpace(provider)) {
	case "candle", "ort", "openvino":
		return true
	}
	return false
}

// migrateRemovedModelProvider moves a deployment onto the model runtime: the
// runtime device of its earlier one and the Hub repository of its model.
func migrateRemovedModelProvider(m *Migration, at FormMatch) error {
	provider := strings.ToLower(strings.TrimSpace(scalarAt(at.Node, "provider")))
	device, problem := runtimeDevice(provider, scalarAt(at.Node, "device"))
	m.Set(at.Node, "provider", Scalar(ModelRuntimeProvider))
	m.Set(at.Node, "device", Scalar(device))
	if problem != "" {
		m.Changed(at.Path+".device", problem)
	}
	artifact := scalarAt(at.Node, "artifact")
	if artifact == "" {
		return nil
	}
	target, note := runtimeArtifact(artifact)
	switch {
	case target != "":
		m.Set(at.Node, "artifact", Scalar(target))
		if note != "" {
			m.Delete(at.Node, "revision")
			m.Changed(at.Path+".artifact", fmt.Sprintf("%s -> %s: %s", artifact, target, note))
		}
	case !path.IsAbs(artifact) && (!hubRepositoryID.MatchString(artifact) || strings.HasPrefix(artifact, "models/")):
		m.Note(at.Path+".artifact", fmt.Sprintf("%q is a relative path; a model_runtime artifact is a Hub repository or an absolute package path", artifact))
	}
	return nil
}

// runtimeDevice is the model runtime device of an earlier provider's device,
// and what changed when it means something else.
func runtimeDevice(provider, device string) (string, string) {
	value := strings.TrimSpace(device)
	if provider == "openvino" {
		if value == "" || strings.EqualFold(value, "CPU") {
			return "cpu", ""
		}
		return "cpu", fmt.Sprintf("OpenVINO device %q has no runtime equivalent, so the model runs on CPU; Intel GPUs can use device xpu:N", value)
	}
	if value == "" || value == "cpu" {
		return "cpu", ""
	}
	prefix, index, _ := strings.Cut(value, ":")
	if index == "" {
		index = "0"
	}
	accelerators := map[string]string{"cuda": "cuda", "rocm": "rocm", "migraphx": "rocm"}
	switch {
	case prefix == "metal":
		return "mps", ""
	case accelerators[prefix] != "" && strings.Trim(index, "0123456789") == "":
		return accelerators[prefix] + ":" + index, ""
	}
	return "cpu", fmt.Sprintf("unknown device %q; the model runs on CPU", value)
}

func migrateExecutionField(m *Migration, at FormMatch) error {
	value := strings.TrimSpace(at.Node.Value)
	switch {
	case at.Key == "precision" && value == "fp16":
		m.Changed(at.Path, "the model runs in its published precision (dtype: auto); dtype: float16 runs it in half precision on a GPU")
	case at.Key == "precision" && value != "" && value != "native" && value != "fp32":
		m.Changed(at.Path, fmt.Sprintf("the unsupported value %q is dropped", value))
	}
	m.Delete(at.Parent, at.Key)
	return nil
}

// runtimeProfiles are the engine arguments that replace each former numerics
// profile, and what an operator should know about the change.
var runtimeProfiles = map[string]struct {
	sharedContext bool
	note          string
}{
	"exact":          {false, "the runtime batches concurrent requests by default; no engine argument restores one request per forward"},
	"batching":       {false, "batching concurrent requests is the runtime's default"},
	"shared_context": {true, "rows of concurrent requests now also share forwards"},
	"max_speed":      {true, "its batching, fused kernels and consented copies are the runtime's defaults"},
}

func migrateRuntimeProfile(m *Migration, at FormMatch) error {
	profile, known := runtimeProfiles[at.Node.Value]
	m.Delete(at.Parent, "profile")
	if !known {
		m.Note(at.Path, fmt.Sprintf("profile %q is dropped; set the engine arguments it stood for (max_num_seqs, max_num_batched_tokens, batch_window_ms, enable_shared_context, dtype)", at.Node.Value))
		return nil
	}
	if profile.sharedContext && mappingValue(at.Parent, "enable_shared_context") == nil {
		m.Set(at.Parent, "enable_shared_context", boolNode(true))
	}
	m.Changed(at.Path, fmt.Sprintf("profile %s: %s", at.Node.Value, profile.note))
	return nil
}

// runtimeArtifact is the Hub repository the model runtime serves for an
// earlier registry path or alias, with the note of a replacement model; empty
// when the artifact names none.
func runtimeArtifact(artifact string) (string, string) {
	reference := strings.TrimRight(strings.TrimSpace(artifact), "/")
	note := ""
	if replacement, ok := legacyModelReplacement(reference); ok {
		reference, note = replacement.target, replacement.note
	}
	if repository := retiredOmniBundles[reference]; repository != "" {
		return repository, "router images no longer ship prepared Omni bundles; the runtime serves the published repository"
	}
	if spec := GetModelByPath(reference); spec != nil && servedBuiltIn(spec) && spec.RepoID != reference {
		return spec.RepoID, note
	}
	return "", ""
}

// retiredOmniBundles are the Omni bundles earlier router images shipped, by
// the repository the runtime serves instead.
var retiredOmniBundles = map[string]string{
	"/opt/router-model-artifacts/vela-1.0-omni-nano": "vllm-sr/Vela-1.0-Omni-Nano",
	"/opt/router-model-artifacts/vela-1.0-omni-mini": "vllm-sr/Vela-1.0-Omni-Mini",
}

type modelReplacement struct{ target, note string }

// legacyModelReplacements are the Vela 1.0 releases that replace the task
// models the native bindings served, which keep the label names of the
// models they replace, by every path and alias the earlier registry
// accepted.
var legacyModelReplacements = func() map[string]modelReplacement {
	const vela = "models/Vela-1.0-Encoder-307M-"
	groups := []struct {
		replacement modelReplacement
		aliases     []string
	}{
		{modelReplacement{vela + "Domain", "Vela Domain keeps the 14 MMLU-Pro domain labels"}, []string{
			"models/mom-domain-classifier", "domain-classifier", "intent-classifier", "category-classifier",
			"category_classifier_modernbert-base_model", "lora_intent_classifier_bert-base-uncased_model",
			"models/mmbert32k-intent-classifier-lora", "mmbert32k-intent", "mmbert-32k-intent", "intent-classifier-32k",
			"models/mmbert32k-intent-classifier-merged", "mmbert32k-intent-merged", "intent-classifier-32k-merged",
		}},
		{modelReplacement{vela + "PII", "Vela PII keeps the 35 BIO labels of the 17 PII types"}, []string{
			"models/mom-pii-classifier", "pii-detector", "pii-classifier", "privacy-guard", "lora_pii_detector_bert-base-uncased_model",
			"models/mom-mmbert-pii-detector", "mmbert-pii-detector", "mmbert-pii-detector-merged",
			"pii_classifier_modernbert-base_presidio_token_model", "pii_classifier_modernbert-base_model", "pii_classifier_modernbert_model",
			"pii_classifier_modernbert_ai4privacy_token_model", "models/mmbert32k-pii-detector-merged", "mmbert32k-pii-merged",
			"pii-detector-32k-merged", "models/mmbert32k-pii-detector-lora", "mmbert32k-pii", "mmbert-32k-pii", "pii-detector-32k",
		}},
		{modelReplacement{vela + "Guard", "Vela Guard keeps the benign / jailbreak labels"}, []string{
			"models/mom-jailbreak-classifier", "jailbreak-detector", "prompt-guard", "safety-classifier",
			"jailbreak_classifier_modernbert-base_model", "lora_jailbreak_classifier_bert-base-uncased_model",
			"jailbreak_classifier_modernbert_model", "models/mmbert32k-jailbreak-detector-lora", "mmbert32k-jailbreak",
			"mmbert-32k-jailbreak", "jailbreak-detector-32k", "prompt-guard-32k", "models/mmbert32k-jailbreak-detector-merged",
			"mmbert32k-jailbreak-merged", "jailbreak-detector-32k-merged",
		}},
		{modelReplacement{vela + "FactCheck", "Vela FactCheck keeps the FACT_CHECK_NEEDED / NO_FACT_CHECK_NEEDED labels"}, []string{
			"models/mom-halugate-sentinel", "hallucination-sentinel", "halugate-sentinel", "models/mmbert32k-factcheck-classifier-lora",
			"mmbert32k-factcheck", "mmbert-32k-factcheck", "factcheck-classifier-32k", "fact-check-32k",
			"models/mmbert32k-factcheck-classifier-merged", "mmbert32k-factcheck-merged", "factcheck-classifier-32k-merged",
		}},
		{modelReplacement{vela + "Halu", "Vela Halu reads the same context, question and answer and returns answer spans"}, []string{
			"models/mom-halugate-detector", "hallucination-detector", "halugate-detector", "lettucedect",
			"KRLabsOrg/lettucedect-base-modernbert-en-v1", "models/lettucedect-v2-mmbert-base", "hallucination-detector-multilingual",
			"lettucedect-v2-mmbert", "KRLabsOrg/lettucedect-v2-mmbert-base",
		}},
		{modelReplacement{vela + "Feedback", "Vela Feedback keeps the four feedback labels and adds NO_FEEDBACK for messages without feedback"}, []string{
			"models/mom-feedback-detector", "feedback-detector", "user-feedback-classifier", "models/mmbert32k-feedback-detector-lora",
			"mmbert32k-feedback", "mmbert-32k-feedback", "feedback-detector-32k", "models/mmbert32k-feedback-detector-merged",
			"mmbert32k-feedback-merged", "feedback-detector-32k-merged",
		}},
		{modelReplacement{vela + "Modality", "Vela Modality keeps the AR / DIFFUSION / BOTH labels"}, []string{
			"models/mmbert32k-modality-router-merged", "modality-classifier", "modality-router", "mmbert32k-modality-router",
		}},
		{modelReplacement{vela + "Embedding", "Vela Embedding replaces this embedding model; stored vectors must be re-embedded and similarity thresholds re-checked"}, []string{
			"models/mmbert-embed-32k-2d-matryoshka", "mom-embedding-ultra", "mmbert-embed-32k-2d-matryoshka", "mmbert-embedding",
			"embedding-mmbert", "embedding-ultra", "models/mom-embedding-flash", "embeddinggemma-300m", "embedding-flash",
			"google/embeddinggemma-300m", "models/mom-embedding-light", "all-MiniLM-L12-v2", "embedding-light", "bert-light",
			"sentence-transformers/all-MiniLM-L12-v2",
		}},
		{modelReplacement{"models/vela-1.0-omni-nano", "Vela Omni Nano replaces this multimodal embedding model; stored vectors must be re-embedded and similarity thresholds re-checked"}, []string{
			"models/mom-embedding-multimodal", "multi-modal-embed-small", "multimodal-embedding", "embedding-multimodal",
			"mom-embedding-multimodal", "vllm-sr/multi-modal-embed-small", "vllm-sr/multi-modal-embed-large", "multi-modal-embed-large",
		}},
	}
	replacements := map[string]modelReplacement{}
	for _, group := range groups {
		for _, alias := range group.aliases {
			for _, key := range legacyAliasKeys(alias) {
				replacements[key] = group.replacement
			}
		}
	}
	return replacements
}()

// retiredNLIAliases name the NLI explainer, which has no runtime replacement.
var retiredNLIAliases = func() map[string]bool {
	keys := map[string]bool{}
	for _, alias := range []string{"models/mom-halugate-explainer", "hallucination-explainer", "halugate-explainer", "nli-explainer", "tasksource/ModernBERT-base-nli"} {
		for _, key := range legacyAliasKeys(alias) {
			keys[key] = true
		}
	}
	return keys
}()

// legacyAliasKeys are the references an earlier alias was accepted by: a
// bare name also under models/.
func legacyAliasKeys(alias string) []string {
	if strings.Contains(alias, "/") {
		return []string{alias}
	}
	return []string{alias, "models/" + alias}
}

func legacyModelReplacement(reference string) (modelReplacement, bool) {
	replacement, ok := legacyModelReplacements[strings.TrimRight(strings.TrimSpace(reference), "/")]
	return replacement, ok
}

func retiredNLIModel(reference string) bool {
	return retiredNLIAliases[strings.TrimRight(strings.TrimSpace(reference), "/")]
}

// providerModelOfDeployment is the first providers.models[] entry whose
// deployment is the named one.
func providerModelOfDeployment(root *yaml.Node, deployment string) string {
	for _, model := range sequenceAt(root, "providers", "models") {
		if model = resolveAlias(model); scalarAt(model, "deployment") == deployment {
			return scalarAt(model, "name")
		}
	}
	return ""
}
