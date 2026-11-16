package handlers

import (
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/k8s/configwriter"
)

// A key typed into the Dashboard, as Connect Models takes one, is saved as a
// reference. Each literal providers.models[].backend_refs[].api_key moves to
// an owner-only file in this directory beside the config, and the backend
// names that file in auth.api_key_file, which the Router reads at every load.
// The directory sits next to the file the Router reads, which `vllm-sr serve`
// mounts into both containers at one path. An environment reference such as
// ${OPENAI_API_KEY} stays as written.
const providerKeyDirectory = "provider-keys"

// persistConfig writes a saved document with its literal backend keys stored
// as files, and returns the bytes it wrote. A Kubernetes ConfigMap target is
// written as given: the Router pod cannot read a file this pod writes.
func persistConfig(configPath string, data []byte) ([]byte, error) {
	if _, ok := configwriter.ConfigMapTargetFromEnv(); ok {
		return data, writeConfigAtomically(configPath, data)
	}
	stored, written, err := storeProviderKeys(filepath.Join(filepath.Dir(configPath), providerKeyDirectory), data)
	if err == nil {
		err = writeConfigAtomically(configPath, stored)
	}
	if err != nil {
		removeProviderKeyFiles(written)
		return nil, err
	}
	return stored, nil
}

// storeProviderKeys returns data with each literal backend key replaced by a
// new key file, and the files it wrote. One key used by several backends is
// stored once. data comes back unchanged when it holds no literal key.
func storeProviderKeys(dir string, data []byte) ([]byte, []string, error) {
	doc, err := parseYAMLDocument(data)
	if err != nil {
		return nil, nil, err
	}
	files := map[string]string{}
	var written []string
	for _, ref := range providerBackendRefs(doc) {
		key := mappingValueNode(ref, "api_key")
		if key == nil || key.Kind != yaml.ScalarNode || key.Value == "" || configEnvReference.MatchString(key.Value) {
			continue
		}
		path, ok := files[key.Value]
		if !ok {
			if path, err = writeProviderKeyFile(dir, key.Value); err != nil {
				removeProviderKeyFiles(written)
				return nil, nil, err
			}
			files[key.Value] = path
			written = append(written, path)
		}
		deleteMappingValueNode(ref, "api_key")
		if auth := mappingValueNode(ref, "auth"); auth == nil || auth.Kind != yaml.MappingNode {
			setMappingValueNode(ref, "auth", &yaml.Node{Kind: yaml.MappingNode, Tag: "!!map"})
		}
		setMappingValueNode(mappingValueNode(ref, "auth"), "api_key_file", &yaml.Node{Kind: yaml.ScalarNode, Tag: "!!str", Value: path})
	}
	if len(written) == 0 {
		return data, nil, nil
	}
	stored, err := yaml.Marshal(doc)
	if err != nil {
		removeProviderKeyFiles(written)
		return nil, nil, err
	}
	return stored, written, nil
}

func providerBackendRefs(doc *yaml.Node) []*yaml.Node {
	root, err := documentMappingNode(doc)
	if err != nil {
		return nil
	}
	models := mappingValueNode(mappingValueNode(root, "providers"), "models")
	if models == nil || models.Kind != yaml.SequenceNode {
		return nil
	}
	var refs []*yaml.Node
	for _, model := range models.Content {
		if backends := mappingValueNode(model, "backend_refs"); backends != nil && backends.Kind == yaml.SequenceNode {
			refs = append(refs, backends.Content...)
		}
	}
	return refs
}

// writeProviderKeyFile stores key in a new mode 0600 file in an owner-only
// directory and returns its absolute path.
func writeProviderKeyFile(dir, key string) (string, error) {
	dir, err := filepath.Abs(dir)
	if err != nil {
		return "", err
	}
	if err = os.MkdirAll(dir, 0o700); err != nil {
		return "", err
	}
	file, err := os.CreateTemp(dir, "key-*")
	if err != nil {
		return "", err
	}
	_, err = file.WriteString(key)
	if err == nil {
		err = file.Sync()
	}
	if closeErr := file.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		_ = os.Remove(file.Name())
		return "", err
	}
	return file.Name(), nil
}

func removeProviderKeyFiles(paths []string) {
	for _, path := range paths {
		_ = os.Remove(path)
	}
}
