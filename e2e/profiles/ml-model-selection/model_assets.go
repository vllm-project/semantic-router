/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package mlmodelselection

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

const (
	mlModelSourceDir   = ".cache/ml-models"
	mlModelTrainingDir = "src/training/model_selection/ml_model_selection"
	mlModelHostDir     = "/tmp/kind-ml-models"
	mlModelNodeDir     = "/tmp/ml-models"
	mlModelRepository  = "abdallah1008/semantic-router-ml-models"
)

var mlModelFiles = []string{
	"knn_model.json",
	"kmeans_model.json",
	"svm_model.json",
	"mlp_model.json",
}

// prepareMLModels ensures ML models are downloaded and available to Kind.
func (p *Profile) prepareMLModels(ctx context.Context) error {
	absSourceDir, err := filepath.Abs(mlModelSourceDir)
	if err != nil {
		return fmt.Errorf("failed to get absolute path: %w", err)
	}
	p.log("ML models directory: %s", absSourceDir)

	if err := p.ensureMLModelsDownloaded(ctx); err != nil {
		return err
	}
	if err := p.copyMLModelsToHost(); err != nil {
		return err
	}
	if err := p.copyMLModelsToKind(ctx); err != nil {
		return err
	}

	p.log("✓ ML models ready in Kind containers")
	return nil
}

func (p *Profile) ensureMLModelsDownloaded(ctx context.Context) error {
	if mlModelsExist() {
		p.log("✓ ML models found in %s", mlModelSourceDir)
		return nil
	}

	p.log("ML models not found locally, downloading from HuggingFace...")
	p.ensureHuggingFaceHub(ctx)
	if err := os.MkdirAll(mlModelSourceDir, 0755); err != nil {
		return fmt.Errorf("create ML models directory: %w", err)
	}
	return p.downloadMLModels(ctx)
}

func mlModelsExist() bool {
	for _, name := range mlModelFiles {
		if _, err := os.Stat(filepath.Join(mlModelSourceDir, name)); os.IsNotExist(err) {
			return false
		}
	}
	return true
}

func (p *Profile) ensureHuggingFaceHub(ctx context.Context) {
	p.log("Ensuring huggingface-hub is installed...")
	var lastErr error
	for _, executable := range []string{"pip", "pip3"} {
		cmd := loggedCommand(ctx, executable, "install", "--quiet", "huggingface-hub>=0.20.0")
		if err := cmd.Run(); err == nil {
			return
		} else {
			lastErr = err
		}
	}
	p.log("Warning: could not install huggingface-hub: %v", lastErr)
}

func (p *Profile) downloadMLModels(ctx context.Context) error {
	p.log("Downloading pretrained ML models from HuggingFace...")
	var lastErr error
	for _, executable := range []string{"python3", "python"} {
		cmd := loggedCommand(
			ctx,
			executable,
			"download_model.py",
			"--output-dir",
			"../../../../.cache/ml-models",
			"--repo-id",
			mlModelRepository,
		)
		cmd.Dir = mlModelTrainingDir
		if err := cmd.Run(); err == nil {
			p.log("✓ ML models downloaded from HuggingFace")
			return nil
		} else {
			lastErr = err
		}
	}
	return fmt.Errorf(
		"failed to download ML models from HuggingFace: %w\nPlease ensure models are uploaded to %s",
		lastErr,
		mlModelRepository,
	)
}

func loggedCommand(ctx context.Context, name string, args ...string) *exec.Cmd {
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd
}

func (p *Profile) copyMLModelsToHost() error {
	p.log("Copying models to host directory %s...", mlModelHostDir)
	if err := os.MkdirAll(mlModelHostDir, 0755); err != nil {
		p.log("  Warning: could not create host directory: %v (may need sudo on some systems)", err)
		return nil
	}

	for _, name := range mlModelFiles {
		source := filepath.Join(mlModelSourceDir, name)
		destination := filepath.Join(mlModelHostDir, name)
		data, err := os.ReadFile(source)
		if err != nil {
			return fmt.Errorf("failed to read %s: %w", source, err)
		}
		if err := os.WriteFile(destination, data, 0644); err != nil {
			p.log("  Warning: could not write %s: %v", destination, err)
			continue
		}
		p.log("  ✓ Copied %s to host", name)
	}
	return nil
}

func (p *Profile) copyMLModelsToKind(ctx context.Context) error {
	p.log("Copying models into Kind node containers...")
	nodes, err := p.getKindNodes(ctx)
	if err != nil {
		return fmt.Errorf("failed to get Kind nodes: %w", err)
	}
	p.log("  Found %d Kind nodes: %v", len(nodes), nodes)

	for _, node := range nodes {
		if err := p.copyMLModelsToKindNode(ctx, node); err != nil {
			return err
		}
	}
	return nil
}

func (p *Profile) copyMLModelsToKindNode(ctx context.Context, node string) error {
	mkdirCmd := exec.CommandContext(ctx, "docker", "exec", node, "mkdir", "-p", mlModelNodeDir)
	if err := mkdirCmd.Run(); err != nil {
		p.log("  Warning: could not create directory in %s: %v", node, err)
		return nil
	}

	for _, name := range mlModelFiles {
		source := filepath.Join(mlModelSourceDir, name)
		data, err := os.ReadFile(source)
		if err != nil {
			return fmt.Errorf("failed to read %s: %w", source, err)
		}

		destination := filepath.Join(mlModelNodeDir, name)
		copyCmd := exec.CommandContext(ctx, "docker", "exec", "-i", node, "tee", destination)
		copyCmd.Stdin = bytes.NewReader(data)
		if err := copyCmd.Run(); err != nil {
			p.log("  Warning: failed to copy %s to %s: %v", name, node, err)
		}
	}
	p.log("  ✓ Copied models to %s", node)
	return nil
}

// getKindNodes returns the node names for the E2E Kind cluster.
func (p *Profile) getKindNodes(ctx context.Context) ([]string, error) {
	cmd := exec.CommandContext(ctx, "kind", "get", "nodes", "--name", "semantic-router-e2e")
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("kind get nodes failed: %w", err)
	}

	var nodes []string
	for _, line := range strings.Split(strings.TrimSpace(string(output)), "\n") {
		if line != "" {
			nodes = append(nodes, line)
		}
	}
	if len(nodes) == 0 {
		return nil, fmt.Errorf("no Kind nodes found")
	}
	return nodes, nil
}
