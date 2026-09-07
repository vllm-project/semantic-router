package handlers

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

func realizeRecipeRuntimeConfigWithCLI(raw []byte, targetPath string) ([]byte, error) {
	parent := filepath.Dir(targetPath)
	source, err := os.CreateTemp(parent, ".recipe-package-raw-*.yaml")
	if err != nil {
		return nil, err
	}
	sourcePath := source.Name()
	defer func() { _ = os.Remove(sourcePath) }()
	err = source.Chmod(0o600)
	if err == nil {
		_, err = source.Write(raw)
	}
	if err == nil {
		err = source.Sync()
	}
	if closeErr := source.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return nil, err
	}
	output, err := os.CreateTemp(parent, ".recipe-package-realized-*.yaml")
	if err != nil {
		return nil, err
	}
	outputPath := output.Name()
	defer func() { _ = os.Remove(outputPath) }()
	if closeErr := output.Close(); closeErr != nil {
		return nil, closeErr
	}
	cliRoot := detectPythonCLIRoot()
	if cliRoot == "" {
		return nil, errors.New("python CLI root not found for Recipe runtime realization")
	}
	pythonBinary, err := runtimeSyncPythonBinary()
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	// #nosec G204 -- pythonBinary is constrained to resolved Python interpreters;
	// module and argv boundaries are fixed, and config paths are process-owned.
	args := recipeRuntimeMaterializeArgs(sourcePath, outputPath)
	cmd := exec.CommandContext(ctx, pythonBinary, args...)
	if algorithm := strings.TrimSpace(os.Getenv("VLLM_SR_ALGORITHM_OVERRIDE")); algorithm != "" {
		cmd.Args = append(cmd.Args, "--algorithm", algorithm)
	}
	platform := strings.TrimSpace(os.Getenv("VLLM_SR_PLATFORM"))
	if platform == "" {
		platform = strings.TrimSpace(os.Getenv("DASHBOARD_PLATFORM"))
	}
	if platform != "" {
		cmd.Args = append(cmd.Args, "--platform", platform)
	}
	cmd.Dir = cliRoot
	cmd.Env = append(os.Environ(), "PYTHONPATH="+cliRoot)
	_, err = cmd.CombinedOutput()
	if err != nil {
		return nil, errors.New("recipe runtime realization failed")
	}
	return readActivationConfig(outputPath)
}

func recipeRuntimeMaterializeArgs(sourcePath, targetPath string) []string {
	return []string{
		"-m",
		"cli.commands.runtime_materialize",
		"--source",
		sourcePath,
		"--target",
		targetPath,
		"--package-activation",
	}
}
