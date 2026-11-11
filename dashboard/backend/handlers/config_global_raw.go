package handlers

import (
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// GlobalConfigYAMLHandler returns the effective canonical global config as raw
// YAML. The response body contains the merged result of router-owned defaults
// plus any config.yaml `global:` overrides, without the surrounding top-level
// `global:` key.
func GlobalConfigYAMLHandler(configPath string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		globalYAML, err := readEffectiveGlobalYAML(configPath)
		if err != nil {
			log.Printf("Failed to read global config: %v", err)
			http.Error(w, "Failed to read global config", http.StatusInternalServerError)
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

		rawBody, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to read request body: %v", err), http.StatusBadRequest)
			return
		}
		release, lockErr := beginOrdinaryRuntimeConfigMutation(configDir)
		if lockErr != nil {
			writeRuntimeConfigMutationError(w, lockErr)
			return
		}
		defer release()

		existingData, err := readPersistedDashboardConfig(configPath)
		if err != nil {
			log.Printf("Failed to read config: %v", err)
			http.Error(w, "Failed to read config", http.StatusInternalServerError)
			return
		}

		updatedYAML, warnings, err := replaceGlobalOverrideYAML(existingData, rawBody)
		if err != nil {
			http.Error(w, fmt.Sprintf("Global config validation failed: %v", err), http.StatusBadRequest)
			return
		}

		if auth.RejectRevokedMutation(w, r) {
			return
		}
		if err := writeConfigAtomically(configPath, updatedYAML); err != nil {
			writeConfigPersistenceError(w, err)
			return
		}
		if rejectRevokedConfigAndRestore(w, r, configPath, configDir, existingData) {
			return
		}

		restartMessage, applyErr := applyWrittenConfig(configPath, configDir, existingData, true)
		if applyErr != nil {
			http.Error(w, formatRuntimeApplyError("Failed to apply config to runtime", applyErr), http.StatusInternalServerError)
			return
		}
		if rejectRevokedConfigAndRestore(w, r, configPath, configDir, existingData) {
			return
		}
		if configActivationDeferred() {
			writeDeferredConfigResponse(w, warnings)
			return
		}
		if restartMessage != "" {
			writeRestartRequiredResponse(w, "", restartMessage, warnings)
			return
		}

		writeConfigSavedResponse(w, warnings)
	}
}

func readEffectiveGlobalYAML(configPath string) ([]byte, error) {
	data, err := readPersistedDashboardConfig(configPath)
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

	var effective yaml.Node
	if err = effective.Encode(global); err != nil {
		return nil, fmt.Errorf("marshal effective global config: %w", err)
	}
	// The typed global omits an explicit false or empty value, which a save of
	// this view would turn back into its default: show what config.yaml writes.
	written, err := decodeConfigDocument[canonicalConfigTransport](data)
	if err != nil {
		return nil, err
	}
	if written.globalOverrideRaw != nil {
		if err = mergeMappingNodes(&effective, written.globalOverrideRaw); err != nil {
			return nil, err
		}
	}
	globalYAML, err := yaml.Marshal(&effective)
	if err != nil {
		return nil, fmt.Errorf("marshal effective global config: %w", err)
	}
	return globalYAML, nil
}

// replaceGlobalOverrideYAML returns the config with its global block
// replaced, once the Router's validator accepts it, and the Router's warnings.
func replaceGlobalOverrideYAML(existingData, rawGlobalYAML []byte) ([]byte, []routerconfig.ConfigWarning, error) {
	existingDoc, err := parseYAMLDocument(existingData)
	if err != nil {
		return nil, nil, fmt.Errorf("parse config.yaml: %w", err)
	}
	root, err := documentMappingNode(existingDoc)
	if err != nil {
		return nil, nil, err
	}

	globalOverride, hasOverride, err := parseRawGlobalOverrideNode(rawGlobalYAML)
	if err != nil {
		return nil, nil, err
	}

	if hasOverride {
		setMappingValueNode(root, "global", globalOverride)
	} else {
		deleteMappingValueNode(root, "global")
	}

	updatedYAML, err := marshalYAMLDocument(existingDoc)
	if err != nil {
		return nil, nil, fmt.Errorf("marshal updated config: %w", err)
	}

	parsed, err := routerconfig.ParseYAMLBytes(updatedYAML)
	if err != nil {
		return nil, nil, err
	}

	return updatedYAML, savedConfigWarnings(parsed), nil
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
	if err := mergeMappingPatch(existingGlobal, patchRoot); err != nil {
		return nil, err
	}

	return marshalYAMLDocument(existingDoc)
}

// mergeMappingPatch applies src to dst as a JSON merge patch (RFC 7396): a
// null value removes its key, a mapping merges into the mapping at its key,
// and any other value replaces it. Removing keys from a mapping that does not
// exist writes nothing.
func mergeMappingPatch(dst, src *yaml.Node) error {
	if dst.Kind != yaml.MappingNode || src.Kind != yaml.MappingNode {
		return fmt.Errorf("merge requires YAML mapping nodes")
	}
	for i := 0; i+1 < len(src.Content); i += 2 {
		key, value := src.Content[i].Value, src.Content[i+1]
		switch {
		case value.Kind == yaml.ScalarNode && value.ShortTag() == "!!null":
			deleteMappingValueNode(dst, key)
		case value.Kind == yaml.MappingNode:
			target := mappingValueNode(dst, key)
			if target != nil && target.Kind == yaml.MappingNode {
				if err := mergeMappingPatch(target, value); err != nil {
					return err
				}
				continue
			}
			created := &yaml.Node{Kind: yaml.MappingNode, Tag: "!!map"}
			if err := mergeMappingPatch(created, value); err != nil {
				return err
			}
			if target != nil || len(created.Content) > 0 || len(value.Content) == 0 {
				setMappingValueNode(dst, key, created)
			}
		default:
			setMappingValueNode(dst, key, value)
		}
	}
	return nil
}
