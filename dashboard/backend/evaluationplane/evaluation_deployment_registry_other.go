//go:build !linux

package evaluationplane

import "fmt"

type deploymentRegistryRoot struct{}

func openDeploymentRegistryRoot(string) (*deploymentRegistryRoot, error) {
	return nil, fmt.Errorf("evaluation deployment registries require descriptor-relative no-follow filesystem support")
}

func (*deploymentRegistryRoot) Close() {}

func (*deploymentRegistryRoot) ReadFile(string, int64) ([]byte, error) {
	return nil, fmt.Errorf("evaluation deployment registries require descriptor-relative no-follow filesystem support")
}
