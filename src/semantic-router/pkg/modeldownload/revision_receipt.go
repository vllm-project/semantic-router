package modeldownload

import (
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

const modelRevisionReceipt = ".router-model-revision.json"

type revisionReceipt struct {
	// HF local-dir metadata establishes commit/content, not repository identity.
	RequestedRepoID string            `json:"requested_repo_id"`
	Revision        string            `json:"revision"`
	VerifiedFiles   map[string]string `json:"verified_files"`
}

func immutableModelRevision(spec ModelSpec) bool {
	if len(spec.Revision) != 40 {
		return false
	}
	_, err := hex.DecodeString(spec.Revision)
	return err == nil
}

// Ordinary HF CLI metadata works directly, including read-only pre-mounts.
// Receipts are informational: identity alone never bypasses content validation.
func hasCurrentModelRevision(spec ModelSpec) (bool, error) {
	if !immutableModelRevision(spec) {
		return true, nil
	}
	_, err := verifyHFModelRevision(spec)
	if errors.Is(err, errUnverifiedModelRevision) {
		return false, nil
	}
	return err == nil, err
}

func invalidateModelRevision(spec ModelSpec) error {
	// A custom/mutable download into a formerly pinned directory invalidates
	// its old provenance too, even though it will not write a new receipt.
	err := os.Remove(filepath.Join(spec.LocalPath, modelRevisionReceipt))
	if os.IsNotExist(err) {
		return nil
	}
	return err
}

// The HF CLI can return success for an old nonempty local-dir while offline.
// Independently verify the requested commit and bytes before recording anything.
func recordModelRevision(spec ModelSpec) error {
	if !immutableModelRevision(spec) {
		return nil
	}
	complete, err := IsModelComplete(spec.LocalPath, spec.RequiredFiles)
	if err != nil {
		return err
	}
	if !complete {
		return fmt.Errorf("downloaded model is missing required artifacts: %s", spec.LocalPath)
	}
	verified, err := verifyHFModelRevision(spec)
	if err != nil {
		return err
	}
	data, err := json.Marshal(revisionReceipt{RequestedRepoID: spec.RepoID, Revision: spec.Revision, VerifiedFiles: verified})
	if err != nil {
		return err
	}
	file, err := os.CreateTemp(spec.LocalPath, ".router-revision-*")
	if err != nil {
		return err
	}
	defer os.Remove(file.Name())
	if _, err = file.Write(data); err != nil {
		file.Close()
		return err
	}
	if err = file.Close(); err != nil {
		return err
	}
	return os.Rename(file.Name(), filepath.Join(spec.LocalPath, modelRevisionReceipt))
}
