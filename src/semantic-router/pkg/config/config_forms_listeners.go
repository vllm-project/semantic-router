package config

import (
	"fmt"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/apiregistry"
)

// listenerForms are the listener fields apis replaced: models, the model
// allow-list of the LLM APIs, and the systemone block. migrate moves both
// into apis, which then names every API the listener served: the LLM APIs,
// each with the models, and systemone in the schema its clients spoke,
// vllm-sr/4.0.
var listenerForms = []ConfigForm{
	{
		Code:    CodeRemovedField,
		Paths:   []string{"listeners[].models"},
		What:    "a listener's models are the models of each API it serves",
		Becomes: "apis: {chat_completions: {models: [...]}, responses: {models: [...]}, messages: {models: [...]}, images: {models: [...]}}",
		Migrate: func(m *Migration, at FormMatch) error {
			if mappingValue(at.Parent, "apis") != nil {
				return fmt.Errorf("the listener sets both models and apis; give each apis entry its models and remove models")
			}
			apis := defaultListenerAPIsNode()
			for index := 0; index+1 < len(apis.Content); index += 2 {
				m.Set(apis.Content[index+1], "models", cloneNode(at.Node))
			}
			m.Set(at.Parent, "apis", apis)
			m.Delete(at.Parent, "models")
			return nil
		},
	},
	{
		Code:    CodeRemovedField,
		Paths:   []string{"listeners[].systemone"},
		What:    "a listener's systemone block is its systemone API",
		Becomes: "apis: {systemone: {schema: vllm-sr/4.0, models: [...], max_deadline_ms: ...}} next to the LLM APIs it serves",
		Migrate: func(m *Migration, at FormMatch) error {
			apis := mappingValue(at.Parent, "apis")
			if apis == nil {
				apis = defaultListenerAPIsNode()
				m.Set(at.Parent, "apis", apis)
			}
			if mappingValue(apis, string(apiregistry.SystemOne)) != nil {
				return fmt.Errorf("the listener sets both systemone and apis.systemone; keep apis.systemone")
			}
			entry := &yaml.Node{Kind: yaml.MappingNode, Tag: "!!map"}
			m.Set(entry, "schema", Scalar(apiregistry.SchemaVLLMSR40))
			for _, key := range []string{"models", "max_deadline_ms"} {
				if value := mappingValue(at.Node, key); value != nil {
					m.Set(entry, key, cloneNode(value))
				}
			}
			m.Set(apis, string(apiregistry.SystemOne), entry)
			m.Delete(at.Parent, "systemone")
			m.Changed(at.Path, "the systemone API keeps the vllm-sr/4.0 contract its clients speak; set schema: vllm-sr/5.0 once they speak 5.0")
			return nil
		},
	},
}

// defaultListenerAPIsNode is an apis mapping of the APIs a listener without
// apis serves, each with no settings.
func defaultListenerAPIsNode() *yaml.Node {
	apis := &yaml.Node{Kind: yaml.MappingNode, Tag: "!!map"}
	for _, api := range apiregistry.All() {
		if api.DefaultOn {
			setNode(apis, string(api.Name), &yaml.Node{Kind: yaml.MappingNode, Tag: "!!map", Style: yaml.FlowStyle})
		}
	}
	return apis
}
