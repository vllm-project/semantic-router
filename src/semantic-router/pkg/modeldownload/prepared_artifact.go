package modeldownload

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

const omniManifestName = "vela_omni_manifest.json"

// Prepared bundles are built and checked against their pinned source before
// deployment. Provisioning copies only inventoried tensors and processor data;
// it never installs Python, executes remote model code, or exports at startup.
type preparedManifest struct {
	FormatVersion int    `json:"format_version"`
	Adapter       string `json:"adapter"`
	Variant       string `json:"variant"`
	Source        struct {
		RepoID   string `json:"repo_id"`
		Revision string `json:"revision"`
	} `json:"source"`
	Tokenizer string `json:"tokenizer"`
	Graphs    map[string]struct {
		File string `json:"file"`
	} `json:"graphs"`
	Processors struct {
		Audio struct {
			File string `json:"file"`
		} `json:"audio"`
	} `json:"processors"`
	Files           map[string]string `json:"files"`
	ReferenceParity struct {
		File   string `json:"file"`
		Passed bool   `json:"passed"`
	} `json:"reference_parity"`
}

func verifyPreparedArtifact(spec ModelSpec) (*preparedManifest, error) {
	if spec.PreparedArtifact != "vela_omni" {
		return nil, fmt.Errorf("unsupported prepared artifact format %q", spec.PreparedArtifact)
	}
	data, err := os.ReadFile(filepath.Join(spec.LocalPath, omniManifestName))
	if err != nil {
		return nil, err
	}
	var manifest preparedManifest
	if decodeErr := json.Unmarshal(data, &manifest); decodeErr != nil {
		return nil, fmt.Errorf("decode prepared manifest: %w", decodeErr)
	}
	if manifest.FormatVersion != 1 || manifest.Adapter != spec.PreparedArtifact ||
		(manifest.Variant != "nano" && manifest.Variant != "mini") || !manifest.ReferenceParity.Passed {
		return nil, fmt.Errorf("unsupported or unverified prepared artifact in %s", spec.LocalPath)
	}
	if manifest.Source.RepoID != spec.RepoID || !isImmutableRevision(manifest.Source.Revision) ||
		(spec.Revision != "" && manifest.Source.Revision != spec.Revision) {
		return nil, fmt.Errorf("prepared source identity does not match %s at revision %s", spec.RepoID, spec.Revision)
	}
	if len(manifest.Graphs) != 4 {
		return nil, fmt.Errorf("prepared Omni artifact requires exactly four graphs")
	}
	required := []string{manifest.Tokenizer, manifest.Processors.Audio.File, manifest.ReferenceParity.File}
	for _, name := range []string{"text", "image", "clap", "audio"} {
		graph, ok := manifest.Graphs[name]
		if !ok {
			return nil, fmt.Errorf("prepared Omni artifact omits %s graph", name)
		}
		required = append(required, graph.File)
	}
	for _, name := range required {
		if name == "" || manifest.Files[name] == "" {
			return nil, fmt.Errorf("prepared inventory omits required file %q", name)
		}
	}
	for name, digest := range manifest.Files {
		if name == omniManifestName || strings.HasSuffix(name, ".py") || strings.HasSuffix(name, ".pyc") || strings.HasSuffix(name, ".safetensors") {
			return nil, fmt.Errorf("prepared artifact contains unexpected runtime payload %q", name)
		}
		want, decodeErr := hex.DecodeString(digest)
		if decodeErr != nil || len(want) != sha256.Size {
			return nil, fmt.Errorf("invalid SHA256 for prepared file %q", name)
		}
		path, pathErr := preparedFile(spec.LocalPath, name)
		if pathErr != nil {
			return nil, pathErr
		}
		file, openErr := os.Open(path)
		if openErr != nil {
			return nil, openErr
		}
		hash := sha256.New()
		_, readErr := io.Copy(hash, file)
		closeErr := file.Close()
		if readErr != nil {
			return nil, readErr
		}
		if closeErr != nil {
			return nil, closeErr
		}
		if hex.EncodeToString(hash.Sum(nil)) != strings.ToLower(digest) {
			return nil, fmt.Errorf("prepared file %q failed SHA256 verification", name)
		}
	}
	// The receipt is inventory-bound and must describe this exact source. The
	// typed runtime subsequently validates tensor shapes and processor semantics.
	reportData, err := os.ReadFile(filepath.Join(spec.LocalPath, manifest.ReferenceParity.File))
	if err != nil {
		return nil, err
	}
	var report struct {
		Passed  bool   `json:"passed"`
		Variant string `json:"variant"`
		Source  struct {
			RepoID   string `json:"repo_id"`
			Revision string `json:"revision"`
		} `json:"source"`
		Tests []struct {
			Passed bool `json:"passed"`
		} `json:"tests"`
	}
	if err := json.Unmarshal(reportData, &report); err != nil {
		return nil, err
	}
	if !report.Passed || report.Source != manifest.Source || report.Variant != manifest.Variant || len(report.Tests) == 0 {
		return nil, fmt.Errorf("invalid prepared artifact parity receipt")
	}
	for _, test := range report.Tests {
		if !test.Passed {
			return nil, fmt.Errorf("prepared artifact has a failed reference parity case")
		}
	}
	return &manifest, nil
}

func preparedFile(root, name string) (string, error) {
	if name == "" || strings.Contains(name, "\\") || filepath.IsAbs(name) || filepath.ToSlash(filepath.Clean(name)) != name || name == "." || strings.HasPrefix(name, "../") {
		return "", fmt.Errorf("invalid prepared file path %q", name)
	}
	rootPath, err := filepath.EvalSymlinks(root)
	if err != nil {
		return "", err
	}
	path, err := filepath.EvalSymlinks(filepath.Join(rootPath, name))
	if err != nil {
		return "", err
	}
	relative, err := filepath.Rel(rootPath, path)
	if err != nil || relative == ".." || strings.HasPrefix(relative, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("prepared file %q escapes artifact", name)
	}
	info, err := os.Stat(path)
	if err != nil {
		return "", err
	}
	if !info.Mode().IsRegular() {
		return "", fmt.Errorf("prepared file %q is not regular", name)
	}
	return path, nil
}

func provisionPreparedArtifact(ctx context.Context, spec ModelSpec) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if ready, err := isSpecComplete(spec); err != nil || ready {
		return err
	}
	root := os.Getenv("ROUTER_MODEL_ARTIFACTS")
	if root == "" {
		root = "/opt/router-model-artifacts"
	}
	if spec.ArtifactBundle == "" || filepath.Base(spec.ArtifactBundle) != spec.ArtifactBundle || spec.ArtifactBundle == "." || spec.ArtifactBundle == ".." {
		return fmt.Errorf("%s requires a complete prepared %s artifact mounted at its configured path", spec.LocalPath, spec.PreparedArtifact)
	}
	source := spec
	source.LocalPath = filepath.Join(root, spec.ArtifactBundle)
	manifest, err := verifyPreparedArtifact(source)
	if err != nil {
		return fmt.Errorf("prepared artifact %s is unavailable or invalid: %w; build the image with VELA_OMNI_VARIANTS including this model, or prepare and mount its verified artifact (tools/models/vela_omni/prepare.py)", source.LocalPath, err)
	}
	if mkdirErr := os.MkdirAll(filepath.Dir(spec.LocalPath), 0o755); mkdirErr != nil {
		return mkdirErr
	}
	stage, err := os.MkdirTemp(filepath.Dir(spec.LocalPath), ".router-artifact-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(stage)
	for name := range manifest.Files {
		if err := ctx.Err(); err != nil {
			return err
		}
		path, err := preparedFile(source.LocalPath, name)
		if err != nil {
			return err
		}
		if err := copyPreparedFile(ctx, path, filepath.Join(stage, name)); err != nil {
			return err
		}
	}
	if err := copyPreparedFile(ctx, filepath.Join(source.LocalPath, omniManifestName), filepath.Join(stage, omniManifestName)); err != nil {
		return err
	}
	staged := spec
	staged.LocalPath = stage
	if _, err := verifyPreparedArtifact(staged); err != nil {
		return fmt.Errorf("verify staged prepared artifact: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if entries, err := os.ReadDir(spec.LocalPath); err == nil {
		if len(entries) > 0 {
			if ready, verifyErr := isSpecComplete(spec); ready && verifyErr == nil {
				return nil
			}
			return fmt.Errorf("refusing to overwrite incomplete prepared artifact directory %s; mount a complete verified bundle or use an empty model cache", spec.LocalPath)
		}
		if removeErr := os.Remove(spec.LocalPath); removeErr != nil {
			return removeErr
		}
	} else if !os.IsNotExist(err) {
		return err
	}
	if err := os.Rename(stage, spec.LocalPath); err != nil {
		return fmt.Errorf("publish prepared artifact: %w", err)
	}
	return nil
}

// Context-aware copying avoids leaving a partially visible cache on shutdown.
func copyPreparedFile(ctx context.Context, source, target string) error {
	input, err := os.Open(source)
	if err != nil {
		return err
	}
	defer input.Close()
	if mkdirErr := os.MkdirAll(filepath.Dir(target), 0o755); mkdirErr != nil {
		return mkdirErr
	}
	output, err := os.OpenFile(target, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	buffer := make([]byte, 1024*1024)
	for {
		if err := ctx.Err(); err != nil {
			_ = output.Close()
			return err
		}
		n, readErr := input.Read(buffer)
		if n > 0 {
			if _, err := output.Write(buffer[:n]); err != nil {
				_ = output.Close()
				return err
			}
		}
		if readErr == io.EOF {
			return output.Close()
		}
		if readErr != nil {
			_ = output.Close()
			return readErr
		}
	}
}

func isImmutableRevision(revision string) bool {
	decoded, err := hex.DecodeString(revision)
	return err == nil && len(decoded) == 20
}
