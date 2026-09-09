package framework

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// ResolveRunResources supplies isolated defaults without changing explicit reuse.
func ResolveRunResources(opts *TestOptions) error {
	if opts.UseExistingCluster && strings.TrimSpace(opts.ClusterName) == "" {
		return fmt.Errorf("--use-existing-cluster requires an explicit --cluster")
	}
	if opts.SkipSetup && !opts.UseExistingCluster {
		return fmt.Errorf("--skip-setup requires --use-existing-cluster and --cluster")
	}
	var random [8]byte
	if _, err := rand.Read(random[:]); err != nil {
		return fmt.Errorf("allocate E2E run identity: %w", err)
	}
	id := hex.EncodeToString(random[:])
	if opts.ClusterName == "" {
		opts.ClusterName = "semantic-router-e2e-" + id
	}
	if opts.ImageTag == "" {
		opts.ImageTag = "e2e-" + id
	}
	if opts.OutputDir == "" {
		opts.OutputDir = filepath.Join(".agent-harness", "runs", "e2e-"+id)
	}
	return os.MkdirAll(opts.OutputDir, 0o755)
}

func runImageReference(reference, tag string) string {
	// Local build references always carry a tag; do not alter registry ports.
	if colon := strings.LastIndex(reference, ":"); colon > strings.LastIndex(reference, "/") {
		return reference[:colon+1] + tag
	}
	return reference + ":" + tag
}

func (r *Runner) localImageReferences() map[string]string {
	references := make(map[string]string, len(r.profileCapabilities.LocalImages))
	for _, image := range r.profileCapabilities.LocalImages {
		references[image.Tag] = runImageReference(image.Tag, r.opts.ImageTag)
	}
	return references
}

// WithLocalImages stages a manifest for one apply, leaving checked-in assets intact.
func WithLocalImages(path string, images map[string]string, apply func(string) error) error {
	if len(images) == 0 {
		return apply(path)
	}
	content, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	updated := string(content)
	for original, replacement := range images {
		updated = strings.ReplaceAll(updated, original, replacement)
	}
	if updated == string(content) {
		return apply(path)
	}
	file, err := os.CreateTemp("", "e2e-images-*.yaml")
	if err != nil {
		return err
	}
	defer os.Remove(file.Name())
	if _, err := file.WriteString(updated); err != nil {
		file.Close()
		return err
	}
	if err := file.Close(); err != nil {
		return err
	}
	return apply(file.Name())
}
