package config

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// A binding names the deployment that answers one signal type, written as
// the deployment's name (`pii: local-vela`). The object form also takes a
// head and an operating point; its contract, adapter, mapping_path and
// pair_scorer fields are deprecated: the signal decides the contract, the
// provider the adapter, and the RAG plugin's rerank block a reranker's
// layer and dimension.

// bindingConsumers maps a binding's signal-type key to the consumer that
// reads it. A key that is not listed names its consumer itself: preference,
// reask, complexity, embedding, safety, hazard, classifier, decision and the
// rule keys safety.<rule>, safety.<rule>.hazard and classifier.<rule>.
var bindingConsumers = map[string]string{
	SignalTypeDomain:        "domain_classifier",
	SignalTypeJailbreak:     "prompt_guard",
	SignalTypePII:           "pii_classifier",
	SignalTypeFactCheck:     "fact_check_classifier",
	SignalTypeUserFeedback:  "feedback_detector",
	SignalTypeModality:      "modality_detector",
	SignalTypeHallucination: "hallucination_detector",
}

// bindingKeyOfConsumer is the inverse of bindingConsumers.
var bindingKeyOfConsumer = func() map[string]string {
	keys := make(map[string]string, len(bindingConsumers))
	for key, consumer := range bindingConsumers {
		keys[consumer] = key
	}
	return keys
}()

// Type-level binding keys of the signals whose rules each ask a question of
// their own: a rule's deployment, then its rule key, then these.
const (
	bindingKeySafety     = SignalTypeSafety
	bindingKeyHazard     = "hazard"
	bindingKeyClassifier = SignalTypeClassifier
	bindingKeySystemOne  = SignalTypeSystemOne
)

// BindingConsumer is the consumer a binding key names: the signal type's
// consumer, or the key itself for a consumer name or rule key.
func BindingConsumer(key string) string {
	if consumer, ok := bindingConsumers[key]; ok {
		return consumer
	}
	return key
}

// BindingKey is the signal-type key a configuration writes for a consumer.
func BindingKey(consumer string) string {
	if key, ok := bindingKeyOfConsumer[consumer]; ok {
		return key
	}
	return consumer
}

// knownBindingKey reports whether a consumer key names a signal a binding
// can serve.
func knownBindingKey(key string) bool {
	switch key {
	case "domain_classifier", "prompt_guard", "pii_classifier", "fact_check_classifier", "feedback_detector",
		"modality_detector", "hallucination_detector", "preference", "reask", "complexity", "embedding",
		RAGRerankerConsumer, bindingKeySafety, bindingKeyHazard, bindingKeyClassifier, bindingKeySystemOne:
		return true
	}
	for _, prefix := range []string{"safety.", "classifier.", "complexity.", "systemone."} {
		if strings.HasPrefix(key, prefix) && len(key) > len(prefix) {
			return true
		}
	}
	return false
}

// ruleConsumer is the consumer a binding key is checked as: a type-level
// key binds every rule of its type, so it is checked as one such rule.
func ruleConsumer(key string) string {
	switch key {
	case bindingKeySafety:
		return "safety._"
	case bindingKeyHazard:
		return "safety._.hazard"
	case bindingKeyClassifier:
		return "classifier._"
	case bindingKeySystemOne:
		return "systemone._"
	}
	return key
}

// IsDeprecatedBindingKey reports whether key is a consumer name that a
// signal-type key replaces.
func IsDeprecatedBindingKey(key string) bool {
	_, ok := bindingKeyOfConsumer[key]
	return ok
}

// normalizeBindingKeys keys bindings by consumer, accepting both the signal
// type and the consumer name of earlier releases, but not both for one
// consumer.
func normalizeBindingKeys(path string, bindings map[string]ModelBinding) (map[string]ModelBinding, error) {
	if len(bindings) == 0 {
		return bindings, nil
	}
	normalized := make(map[string]ModelBinding, len(bindings))
	written := make(map[string]string, len(bindings))
	keys := make([]string, 0, len(bindings))
	for key := range bindings {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		consumer := BindingConsumer(key)
		if previous, seen := written[consumer]; seen {
			return nil, fmt.Errorf("%s: %s and %s bind the same signal; keep %s", path, previous, key, BindingKey(consumer))
		}
		written[consumer] = key
		normalized[consumer] = bindings[key]
	}
	return normalized, nil
}

// PublicBindings keys bindings by signal type, as a configuration writes
// them.
func PublicBindings(bindings map[string]ModelBinding) map[string]ModelBinding {
	return publicBindings(bindings)
}

// publicBindings keys bindings by signal type for a written configuration.
func publicBindings(bindings map[string]ModelBinding) map[string]ModelBinding {
	if bindings == nil {
		return nil
	}
	public := make(map[string]ModelBinding, len(bindings))
	for consumer, binding := range bindings {
		public[BindingKey(consumer)] = binding
	}
	return public
}

// modelBindingObject is the object form of a binding.
type modelBindingObject struct {
	Deployment     string                   `yaml:"deployment" json:"deployment"`
	Contract       string                   `yaml:"contract,omitempty" json:"contract,omitempty"`
	Adapter        string                   `yaml:"adapter,omitempty" json:"adapter,omitempty"`
	Head           string                   `yaml:"head,omitempty" json:"head,omitempty"`
	MappingPath    string                   `yaml:"mapping_path,omitempty" json:"mapping_path,omitempty"`
	PairScorer     *PairScorerSelection     `yaml:"pair_scorer,omitempty" json:"pair_scorer,omitempty"`
	OperatingPoint *OperatingPointReference `yaml:"operating_point,omitempty" json:"operating_point,omitempty"`
}

// OnlyDeployment reports whether a binding sets nothing but its deployment,
// so it is written as the deployment's name.
func (b ModelBinding) OnlyDeployment() bool {
	return b.Contract == "" && b.Adapter == "" && b.Head == "" && b.MappingPath == "" && b.PairScorer == nil && b.OperatingPoint == nil
}

// DeprecatedFields lists the deprecated fields a binding sets. A binding
// still selects a head and an operating point; the RAG plugin's rerank
// block selects a reranker's layer and dimension.
func (b ModelBinding) DeprecatedFields() []string {
	var fields []string
	for name, set := range map[string]bool{
		"contract": b.Contract != "", "adapter": b.Adapter != "", "mapping_path": b.MappingPath != "",
		"pair_scorer": b.PairScorer != nil,
	} {
		if set {
			fields = append(fields, name)
		}
	}
	sort.Strings(fields)
	return fields
}

// UnmarshalYAML reads a binding written as a deployment name or an object.
func (b *ModelBinding) UnmarshalYAML(unmarshal func(interface{}) error) error {
	var name string
	if err := unmarshal(&name); err == nil {
		*b = ModelBinding{Deployment: strings.TrimSpace(name)}
		return nil
	}
	var object modelBindingObject
	if err := unmarshal(&object); err != nil {
		return err
	}
	*b = object.binding()
	return nil
}

// MarshalYAML writes a binding that sets only its deployment as its name.
func (b ModelBinding) MarshalYAML() (interface{}, error) {
	if b.OnlyDeployment() {
		return b.Deployment, nil
	}
	return objectOf(b), nil
}

// UnmarshalJSON reads a binding written as a deployment name or an object,
// which may set no field a binding does not have.
func (b *ModelBinding) UnmarshalJSON(data []byte) error {
	var name string
	if err := json.Unmarshal(data, &name); err == nil {
		*b = ModelBinding{Deployment: strings.TrimSpace(name)}
		return nil
	}
	var object modelBindingObject
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&object); err != nil {
		return err
	}
	*b = object.binding()
	return nil
}

// MarshalJSON writes a binding that sets only its deployment as its name.
func (b ModelBinding) MarshalJSON() ([]byte, error) {
	if b.OnlyDeployment() {
		return json.Marshal(b.Deployment)
	}
	return json.Marshal(objectOf(b))
}

func (o modelBindingObject) binding() ModelBinding {
	return ModelBinding{Deployment: o.Deployment, Contract: o.Contract, Adapter: o.Adapter, Head: o.Head, MappingPath: o.MappingPath, PairScorer: o.PairScorer, OperatingPoint: o.OperatingPoint}
}

func objectOf(b ModelBinding) modelBindingObject {
	return modelBindingObject{Deployment: b.Deployment, Contract: b.Contract, Adapter: b.Adapter, Head: b.Head, MappingPath: b.MappingPath, PairScorer: b.PairScorer, OperatingPoint: b.OperatingPoint}
}
