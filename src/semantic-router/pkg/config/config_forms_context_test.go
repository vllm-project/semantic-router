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
    decisions:
      - name: assistant
        priority: 100
        rules: {operator: AND, conditions: []}
        modelRefs: [{model: general}]
        plugins: [PLUGINS]
global: GLOBAL
`

// The global memory off, on by the memory store's switch, and on as the
// memory source of global.plugins' context plugin.
var contextFormsGlobals = map[string]string{
	"off":            `{stores: {memory: {enabled: false, backend: milvus, milvus: {address: "127.0.0.1:19530"}}}}`,
	"store switch":   `{stores: {memory: {enabled: true, backend: milvus, milvus: {address: "127.0.0.1:19530"}}}}`,
	"global.plugins": `{stores: {memory: {backend: milvus, milvus: {address: "127.0.0.1:19530"}}}, plugins: [{type: context, configuration: {enabled: true, sources: [{type: memory}]}}]}`,
}

// The old plugins of a decision that turn memory on or off, or bring other
// context, and a context plugin that compresses and lists no sources.
var contextFormsPlugins = map[string]string{
	"memory on":   `{type: memory, configuration: {enabled: true}}`,
	"memory off":  `{type: memory, configuration: {enabled: false}}`,
	"rag":         `{type: rag, configuration: {enabled: true, backend: vectorstore, backend_config: {vector_store_id: vs_docs}}}`,
	"compression": `{type: context_compression, configuration: {enabled: true, mode: auto}}`,
	"context":     `{type: context, configuration: {enabled: true, compress: {mode: auto}}}`,
}

// After migrate and load, every decision retrieves memories and compresses
// as it did: memory where the global memory was on and no memory plugin
// turned it off, or an enabled memory plugin turned it on, in either plugin
// order.
func TestMigrateKeepsWhetherADecisionRetrievesMemory(t *testing.T) {
	for _, global := range []string{"off", "store switch", "global.plugins"} {
		for _, memory := range []string{"", "memory on", "memory off"} {
			for _, other := range []string{"", "rag", "compression", "context"} {
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
					t.Run(global+"/"+strings.Join(order, ","), func(t *testing.T) {
						doc := strings.NewReplacer("PLUGINS", strings.Join(plugins, ", "), "GLOBAL", contextFormsGlobals[global]).Replace(contextFormsDoc)
						migrated, _, err := MigrateConfig([]byte(doc))
						if err != nil {
							t.Fatalf("migrate: %v", err)
						}
						cfg, err := ParseYAMLBytes(migrated)
						if err != nil {
							t.Fatalf("load: %v\n%s", err, migrated)
						}
						context := cfg.ContextPlugin(cfg.GetDecisionByName("assistant"))
						wantMemory := memory == "memory on" || memory == "" && global != "off"
						if got := context.Source(ContextSourceMemory) != nil; got != wantMemory {
							t.Errorf("retrieves memory = %t, want %t\n%s", got, wantMemory, migrated)
						}
						wantCompress := other == "compression" || other == "context"
						if got := context != nil && context.Compress.CompressesRequest(); got != wantCompress {
							t.Errorf("compresses = %t, want %t\n%s", got, wantCompress, migrated)
						}
					})
				}
			}
		}
	}
}
