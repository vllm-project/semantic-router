//go:build !windows && cgo

package apiserver

import (
	"context"
	"fmt"
	"os"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/k8s/configwriter"
)

// readPersistedSourceConfig reads the document that the next pod will load.
// A ConfigMap subPath mount stays at its old revision until the pod restarts.
func readPersistedSourceConfig(path string) ([]byte, error) {
	target, ok := configwriter.ConfigMapTargetFromEnv()
	if !ok {
		return os.ReadFile(path)
	}
	writer, err := resolvedConfigMapWriter()
	if err != nil {
		return nil, fmt.Errorf("resolve config ConfigMap client: %w", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), configMapWriteTimeout)
	defer cancel()
	data, found, err := writer.Read(ctx, target)
	if err != nil {
		return nil, err
	}
	if !found {
		return nil, os.ErrNotExist
	}
	return data, nil
}
