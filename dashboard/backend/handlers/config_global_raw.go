package handlers

import (
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"

	"gopkg.in/yaml.v3"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// GlobalConfigYAMLHandler returns the effective canonical global config as raw
// YAML. The response body contains the merged result of router-owned defaults
// plus any config.yaml `global:` overrides, without the surrounding top-level
// `global:` key.
func GlobalConfigYAMLHandler(configPath string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		globalYAML, err := readEffectiveGlobalYAML(configPath)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to read global config: %v", err), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "text/yaml; charset=utf-8")
		w.Header().Set("Cache-Control", "no-cache, no-store, must-revalidate")
		_, _ = w.Write(globalYAML)
	}
}

// UpdateGlobalConfigYAMLHandler replaces the config.yaml global block using a
// raw YAML payload that represents the contents nested under `global:`.
func UpdateGlobalConfigYAMLHandler(configPath string, readonlyMode bool, configDir string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost && r.Method != http.MethodPut {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		if readonlyMode {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusForbidden)
			if err := writeYAMLTaggedJSON(w, map[string]string{
				"error":   "readonly_mode",
				"message": "Dashboard is in read-only mode. Configuration editing is disabled.",
			}); err != nil {
				log.Printf("Error encoding readonly response: %v", err)
			}
			return
		}

		existingData, err := os.ReadFile(configPath)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to read config: %v", err), http.StatusInternalServerError)
			return
		}

		rawBody, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to read request body: %v", err), http.StatusBadRequest)
			return
		}

		updatedYAML, err := replaceGlobalOverrideYAML(existingData, rawBody)
		if err != nil {
			http.Error(w, fmt.Sprintf("Global config validation failed: %v", err), http.StatusBadRequest)
			return
		}

		if err := writeConfigAtomically(configPath, updatedYAML); err != nil {
			http.Error(w, fmt.Sprintf("Failed to write config: %v", err), http.StatusInternalServerError)
			return
		}

		if err := propagateConfigToRuntime(configPath, configDir); err != nil {
			if restoreErr := restorePreviousRuntimeConfig(configPath, configDir, existingData); restoreErr != nil {
				http.Error(w, fmt.Sprintf("Failed to apply config to runtime: %v. Failed to restore previous config: %v", err, restoreErr), http.StatusInternalServerError)
				return
			}
			http.Error(w, fmt.Sprintf("Failed to apply config to runtime: %v. Previous config restored.", err), http.StatusInternalServerError)
			return
		}

		if err := writeYAMLTaggedJSON(w, map[string]string{"status": "success"}); err != nil {
			log.Printf("Error encoding response: %v", err)
		}
	}
}

func readEffectiveGlobalYAML(configPath string) ([]byte, error) {
	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, err
	}

	parsed, err := routerconfig.ParseYAMLBytes(data)
	if err != nil {
		return nil, fmt.Errorf("parse config.yaml: %w", err)
	}

	global := routerconfig.CanonicalGlobalFromRouterConfig(parsed)
	if global == nil {
		defaults := routerconfig.DefaultCanonicalGlobal()
		global = &defaults
	}

	globalYAML, err := yaml.Marshal(global)
	if err != nil {
		return nil, fmt.Errorf("marshal effective global config: %w", err)
	}
	return globalYAML, nil
}

func replaceGlobalOverrideYAML(existingData, rawGlobalYAML []byte) ([]byte, error) {
	existingDoc, err := parseYAMLDocument(existingData)
	if err != nil {
		return nil, fmt.Errorf("parse config.yaml: %w", err)
	}
	root, err := documentMappingNode(existingDoc)
	if err != nil {
		return nil, err
	}

	globalOverride, hasOverride, err := parseRawGlobalOverrideNode(rawGlobalYAML)
	if err != nil {
		return nil, err
	}

	if hasOverride {
		setMappingValueNode(root, "global", globalOverride)
	} else {
		deleteMappingValueNode(root, "global")
	}

	updatedYAML, err := marshalYAMLDocument(existingDoc)
	if err != nil {
		return nil, fmt.Errorf("marshal updated config: %w", err)
	}

	if _, err := routerconfig.ParseYAMLBytes(updatedYAML); err != nil {
		return nil, err
	}

	return updatedYAML, nil
}

func parseRawGlobalOverrideNode(raw []byte) (*yaml.Node, bool, error) {
	if strings.TrimSpace(string(raw)) == "" {
		return nil, false, nil
	}

	doc, err := parseYAMLDocument(raw)
	if err != nil {
		return nil, false, fmt.Errorf("invalid YAML: %w", err)
	}
	root, err := documentMappingNode(doc)
	if err != nil {
		return nil, false, fmt.Errorf("invalid YAML: %w", err)
	}

	if len(root.Content) == 0 {
		return nil, false, nil
	}

	return cloneYAMLNode(root), true, nil
}

func mergeGlobalOverridePatchYAML(existingData, rawPatch []byte) ([]byte, error) {
	existingDoc, err := parseYAMLDocument(existingData)
	if err != nil {
		return nil, fmt.Errorf("parse config.yaml: %w", err)
	}
	root, err := documentMappingNode(existingDoc)
	if err != nil {
		return nil, err
	}

	patchDoc, err := parseYAMLDocument(rawPatch)
	if err != nil {
		return nil, fmt.Errorf("invalid request body: %w", err)
	}
	patchRoot, err := documentMappingNode(patchDoc)
	if err != nil {
		return nil, fmt.Errorf("invalid request body: %w", err)
	}

	existingGlobal := mappingValueNode(root, "global")
	if existingGlobal == nil {
		existingGlobal = &yaml.Node{Kind: yaml.MappingNode, Tag: "!!map"}
		setMappingValueNode(root, "global", existingGlobal)
		existingGlobal = mappingValueNode(root, "global")
	}
	if existingGlobal.Kind != yaml.MappingNode {
		return nil, fmt.Errorf("config.yaml global block must be a YAML mapping")
	}
	if err := mergeMappingNodes(existingGlobal, patchRoot); err != nil {
		return nil, err
	}

	return marshalYAMLDocument(existingDoc)
}
