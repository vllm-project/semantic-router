// Package modelassets resolves performance artifacts from the router's canonical catalog.
package modelassets

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const ManifestEnv = "VLLM_SR_MODEL_MANIFEST"

type Artifact struct {
	Name     string `json:"name"`
	Env      string `json:"env"`
	Path     string `json:"path"`
	RepoID   string `json:"repo_id"`
	Revision string `json:"revision"`
}

type Manifest struct {
	Provider string     `json:"provider"`
	Models   []Artifact `json:"models"`
}

func Resolve(name, root string) (Artifact, error) {
	env := "VLLM_SR_" + strings.ToUpper(name) + "_MODEL"
	var artifact Artifact
	if path := os.Getenv(ManifestEnv); path != "" {
		data, err := os.ReadFile(path)
		if err != nil {
			return artifact, err
		}
		var manifest Manifest
		if err = json.Unmarshal(data, &manifest); err != nil {
			return artifact, err
		}
		if manifest.Provider != "candle" {
			return artifact, fmt.Errorf("performance models require the candle manifest")
		}
		for _, candidate := range manifest.Models {
			if candidate.Env == env {
				artifact = candidate
				break
			}
		}
	} else {
		defaults := config.DefaultGlobalConfig()
		paths := map[string]string{"domain": defaults.CategoryModel.ModelID, "pii": defaults.PIIModel.ModelID, "jailbreak": defaults.PromptGuard.ModelID, "embedding": defaults.EmbeddingModels.MmBertModelPath}
		spec := config.GetModelByPath(paths[name])
		if spec != nil {
			artifact = Artifact{Name: name, Env: env, Path: paths[name], RepoID: spec.RepoID, Revision: spec.Revision}
		}
	}
	if artifact.RepoID == "" || artifact.Revision == "" || artifact.Path == "" {
		return artifact, fmt.Errorf("missing pinned performance artifact %q", name)
	}
	if override := os.Getenv(env); override != "" {
		artifact.Path = override
	}
	if !filepath.IsAbs(artifact.Path) {
		artifact.Path = filepath.Join(root, artifact.Path)
	}
	return artifact, nil
}

// ContentsSHA256 measures the actual local artifact, not just its declared revision.
// Download bookkeeping is excluded; all model, tokenizer and mapping files count.
func ContentsSHA256(path string) (string, error) {
	resolved, err := filepath.EvalSymlinks(path)
	if err != nil {
		return "", err
	}
	directory, err := os.OpenRoot(resolved)
	if err != nil {
		return "", err
	}
	defer directory.Close()
	hash := sha256.New()
	files := 0
	err = fs.WalkDir(directory.FS(), ".", func(pathname string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.IsDir() {
			if entry.Name() == ".cache" || entry.Name() == ".git" {
				return fs.SkipDir
			}
			return nil
		}
		file, openErr := openArtifactFile(directory, filepath.FromSlash(pathname), entry.Type())
		if openErr != nil {
			return openErr
		}
		digest := sha256.New()
		_, copyErr := io.Copy(digest, file)
		closeErr := file.Close()
		if copyErr != nil {
			return copyErr
		}
		if closeErr != nil {
			return closeErr
		}
		fmt.Fprintf(hash, "%s\x00%x\x00", pathname, digest.Sum(nil))
		files++
		return nil
	})
	if err != nil {
		return "", err
	}
	if files == 0 {
		return "", fmt.Errorf("empty model artifact: %s", path)
	}
	return hex.EncodeToString(hash.Sum(nil)), nil
}

func openArtifactFile(directory *os.Root, name string, mode fs.FileMode) (*os.File, error) {
	if mode&os.ModeSymlink != 0 {
		// HF snapshots deliberately reference ../../blobs. Resolve that explicit
		// link and anchor the final open to the blob's parent directory.
		target, err := directory.Readlink(name)
		if err != nil {
			return nil, err
		}
		if !filepath.IsAbs(target) {
			target = filepath.Join(directory.Name(), filepath.Dir(name), target)
		}
		target, err = filepath.EvalSymlinks(target)
		if err != nil {
			return nil, err
		}
		blobDirectory, err := os.OpenRoot(filepath.Dir(target))
		if err != nil {
			return nil, err
		}
		defer blobDirectory.Close()
		directory, name = blobDirectory, filepath.Base(target)
	}
	info, err := directory.Stat(name)
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() {
		return nil, fmt.Errorf("model artifact is not a regular file: %s", name)
	}
	file, err := directory.Open(name)
	if err != nil {
		return nil, err
	}
	info, err = file.Stat()
	if err != nil || !info.Mode().IsRegular() {
		_ = file.Close()
		if err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("model artifact is not a regular file: %s", name)
	}
	return file, nil
}
