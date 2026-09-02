package evaluationplane

import (
	"fmt"
	"os"
	"path/filepath"
)

type workerStaging struct {
	root                string
	storePath           string
	manifestPath        string
	destinationStore    string
	runID               string
	manifestBytes       []byte
	evidencePublication func(func() error) error
}

func prepareWorkerStaging(spec ProcessSpec) (*workerStaging, error) {
	manifestPath, err := filepath.Abs(spec.ManifestPath)
	if err != nil {
		return nil, fmt.Errorf("resolve worker manifest: %w", err)
	}
	storePath, err := filepath.Abs(spec.StorePath)
	if err != nil {
		return nil, fmt.Errorf("resolve worker store: %w", err)
	}
	if validationErr := requirePrivateDirectory(storePath); validationErr != nil {
		return nil, fmt.Errorf("validate worker destination store: %w", validationErr)
	}
	manifest, raw, err := readRunManifestStrict(manifestPath)
	if err != nil {
		return nil, fmt.Errorf("read worker manifest: %w", err)
	}
	if _, contractErr := spec.executionContracts.resolve(manifest); contractErr != nil {
		return nil, fmt.Errorf("resolve worker execution contract: %w", contractErr)
	}
	expectedManifest := filepath.Join(storePath, "runs", manifest.RunID, manifestFileName)
	if manifestPath != expectedManifest {
		return nil, fmt.Errorf("worker manifest is outside its canonical run bundle")
	}

	root, err := os.MkdirTemp("", "vllm-sr-evaluation-worker-")
	if err != nil {
		return nil, fmt.Errorf("create private worker staging root: %w", err)
	}
	staging := &workerStaging{
		root: root, storePath: filepath.Join(root, "store"), destinationStore: storePath,
		runID: manifest.RunID, manifestBytes: raw, evidencePublication: spec.evidencePublication,
	}
	if err := staging.initialize(); err != nil {
		staging.cleanup()
		return nil, err
	}
	return staging, nil
}

func (staging *workerStaging) initialize() error {
	if err := os.Chmod(staging.root, 0o700); err != nil {
		return fmt.Errorf("protect worker staging root: %w", err)
	}
	paths := []string{
		staging.storePath,
		filepath.Join(staging.storePath, "runs"),
		filepath.Join(staging.storePath, "runs", staging.runID),
		filepath.Join(staging.root, "home"),
		filepath.Join(staging.root, "tmp"),
	}
	for _, path := range paths {
		if err := os.Mkdir(path, 0o700); err != nil {
			return fmt.Errorf("create worker staging directory: %w", err)
		}
		if err := requirePrivateDirectory(path); err != nil {
			return fmt.Errorf("protect worker staging directory: %w", err)
		}
	}
	staging.manifestPath = filepath.Join(staging.storePath, "runs", staging.runID, manifestFileName)
	//nolint:gosec // The canonical path is constructed under a private mktemp root.
	file, err := os.OpenFile(staging.manifestPath, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return fmt.Errorf("stage immutable worker manifest: %w", err)
	}
	if _, err = file.Write(staging.manifestBytes); err == nil {
		err = file.Sync()
	}
	closeErr := file.Close()
	if err != nil {
		return fmt.Errorf("write immutable worker manifest: %w", err)
	}
	if closeErr != nil {
		return fmt.Errorf("close immutable worker manifest: %w", closeErr)
	}
	return nil
}

func (staging *workerStaging) cleanup() {
	if staging != nil && staging.root != "" {
		_ = os.RemoveAll(staging.root)
	}
}
