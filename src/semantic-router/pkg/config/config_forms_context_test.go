package config

import (
	"strings"
	"testing"
)

const contextFormsDoc = `version: v0.5
providers:
  defaults: {model: general}
  models:
    - name: general
      backend_refs: [{name: local, endpoint: "127.0.0.1:8000", protocol: http, weight: 1}]
recipes:
  - name: default
    plugins: RECIPE
    decisions:
      - name: assistant
        priority: 100
        rules: {operator: AND, conditions: []}
        modelRefs: [{model: general}]
        plugins: [PLUGINS]
global: GLOBAL
`

// The context plugin a decision inherits: none, the one with a memory source
// the memory store's switch becomes, global.plugins' or its recipe's with a
// memory source, and global.plugins' or its recipe's disabled over a memory
// source. listed is whether it lists a memory source, enabled whether it is
// enabled.
var contextFormsInherited = []struct {
	name, global, recipe string
	listed, enabled      bool
}{
	{"off", `{stores: {memory: {enabled: false, backend: milvus, milvus: {address: "127.0.0.1:19530"}}, vector_store: {enabled: true, backend_type: memory}}}`, "[]", false, false},
	{"store switch", `{stores: {memory: {enabled: true, backend: milvus, milvus: {address: "127.0.0.1:19530"}}, vector_store: {enabled: true, backend_type: memory}}}`, "[]", true, true},
	{"global.plugins", contextFormsGlobalMemory, "[]", true, true},
	{"recipe", `{stores: {memory: {backend: milvus, milvus: {address: "127.0.0.1:19530"}}, vector_store: {enabled: true, backend_type: memory}}}`, `[{type: context, configuration: {enabled: true, sources: [{type: memory}]}}]`, true, true},
	{"global.plugins disabled", `{stores: {memory: {backend: milvus, milvus: {address: "127.0.0.1:19530"}}, vector_store: {enabled: true, backend_type: memory}}, plugins: [{type: context, configuration: {enabled: false, sources: [{type: memory}]}}]}`, "[]", true, false},
	{"recipe disabled", contextFormsGlobalMemory, `[{type: context, configuration: {enabled: false}}]`, true, false},
}

const contextFormsGlobalMemory = `{stores: {memory: {backend: milvus, milvus: {address: "127.0.0.1:19530"}}, vector_store: {enabled: true, backend_type: memory}}, plugins: [{type: context, configuration: {enabled: true, sources: [{type: memory}]}}]}`

// The old plugins of a decision that turn memory on or off, or bring other
// context, and context plugins that list no sources and compress, set the
// injection, or both.
var contextFormsPlugins = map[string]string{
	"memory on":   `{type: memory, configuration: {enabled: true}}`,
	"memory off":  `{type: memory, configuration: {enabled: false}}`,
	"rag":         `{type: rag, configuration: {enabled: true, backend: vectorstore, backend_config: {vector_store_id: vs_docs}}}`,
	"compression": `{type: context_compression, configuration: {enabled: true, mode: auto}}`,
	"context":     `{type: context, configuration: {enabled: true, compress: {mode: auto}}}`,
	"injection":   `{type: context, configuration: {enabled: true, injection: {mode: system_prompt}}}`,
	"both":        `{type: context, configuration: {enabled: true, compress: {mode: auto}, injection: {mode: system_prompt}}}`,
}

// After migrate and load, every decision retrieves memories and compresses
// as it did: memory where an enabled memory plugin turned it on, or none
// turned it off and its context plugin, inherited or its own over an
// inherited memory source, retrieved it, in either plugin order.
func TestMigrateKeepsWhetherADecisionRetrievesMemory(t *testing.T) {
	for _, inherited := range contextFormsInherited {
		for _, memory := range []string{"", "memory on", "memory off"} {
			for _, other := range []string{"", "rag", "compression", "context", "injection", "both"} {
				if (other == "injection" || other == "both") && !inherited.listed {
					continue // a plugin that sets the injection loads over a source
				}
				orders := [][]string{{memory, other}}
				if memory != "" && other != "" {
					orders = append(orders, []string{other, memory})
				}
				for _, order := range orders {
					var plugins []string
					for _, name := range order {
						if name != "" {
							plugins = append(plugins, contextFormsPlugins[name])
						}
					}
					t.Run(inherited.name+"/"+strings.Join(order, ","), func(t *testing.T) {
						doc := strings.NewReplacer("PLUGINS", strings.Join(plugins, ", "), "GLOBAL", inherited.global, "RECIPE", inherited.recipe).Replace(contextFormsDoc)
						migrated, _, err := MigrateConfig([]byte(doc))
						if err != nil {
							t.Fatalf("migrate: %v", err)
						}
						cfg, err := ParseYAMLBytes(migrated)
						if err != nil {
							t.Fatalf("load: %v\n%s", err, migrated)
						}
						context := cfg.ContextPlugin(cfg.GetDecisionByName("assistant"))
						ownContext := other == "context" || other == "injection" || other == "both"
						wantMemory := memory == "memory on" || memory == "" && inherited.listed && (inherited.enabled || ownContext)
						if got := context.Source(ContextSourceMemory) != nil; got != wantMemory {
							t.Errorf("retrieves memory = %t, want %t\n%s", got, wantMemory, migrated)
						}
						wantCompress := other == "compression" || other == "context" || other == "both"
						if got := context != nil && context.Compress.CompressesRequest(); got != wantCompress {
							t.Errorf("compresses = %t, want %t\n%s", got, wantCompress, migrated)
						}
					})
				}
			}
		}
	}
}

// A decision whose only source was an mcp RAG source, which migrate drops,
// still does not retrieve the memory it would inherit.
func TestMigrateKeepsMemoryOffWhenItDropsTheOnlyMCPSource(t *testing.T) {
	mcp := `{type: context, configuration: {enabled: true, sources: [{type: rag, backend: mcp, backend_config: {server_name: docs-server, tool_name: retrieve_docs}}], compress: {mode: auto}}}`
	global := `{stores: {memory: {backend: milvus, milvus: {address: "127.0.0.1:19530"}}, vector_store: {enabled: true, backend_type: memory}}, plugins: [{type: context, configuration: {enabled: true, sources: [{type: memory}]}}]}`
	doc := strings.NewReplacer("PLUGINS", mcp, "GLOBAL", global, "RECIPE", "[]").Replace(contextFormsDoc)
	migrated, _, err := MigrateConfig([]byte(doc))
	if err != nil {
		t.Fatalf("migrate: %v", err)
	}
	cfg, err := ParseYAMLBytes(migrated)
	if err != nil {
		t.Fatalf("load: %v\n%s", err, migrated)
	}
	context := cfg.ContextPlugin(cfg.GetDecisionByName("assistant"))
	if context.Source(ContextSourceMemory) != nil || !context.Compress.CompressesRequest() {
		t.Fatalf("context = %+v, want compression and no memory\n%s", context, migrated)
	}
}

// A decision that migrate gives a context plugin of its own lists the memory
// source it inherited as its recipe wrote it: its sources replace the
// inherited ones, so a bare memory source would drop the recall bounds and
// write memories the inherited source did not.
func TestMigrateKeepsTheInheritedMemorySourceSettings(t *testing.T) {
	global := `{stores: {memory: {auto_store: true, backend: milvus, milvus: {address: "127.0.0.1:19530"}}, vector_store: {enabled: true, backend_type: memory}}}`
	recipe := `[{type: context, configuration: {enabled: true, sources: [{type: memory, match: {recall: {top_k: 9, min_similarity: 0.6}}, store: {auto: false}}]}}]`
	for _, other := range []string{"rag", "compression"} {
		t.Run(other, func(t *testing.T) {
			doc := strings.NewReplacer("PLUGINS", contextFormsPlugins[other], "GLOBAL", global, "RECIPE", recipe).Replace(contextFormsDoc)
			migrated, _, err := MigrateConfig([]byte(doc))
			if err != nil {
				t.Fatalf("migrate: %v", err)
			}
			cfg, err := ParseYAMLBytes(migrated)
			if err != nil {
				t.Fatalf("load: %v\n%s", err, migrated)
			}
			memory := cfg.ContextPlugin(cfg.GetDecisionByName("assistant")).Source(ContextSourceMemory)
			if memory == nil || memory.Match == nil || memory.Match.Recall == nil || memory.Match.Recall.TopK != 9 ||
				memory.AutoStore() == nil || *memory.AutoStore() {
				t.Fatalf("memory source = %+v, want the recipe's top_k 9 and store.auto false\n%s", memory, migrated)
			}
		})
	}
}
