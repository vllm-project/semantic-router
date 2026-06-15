//go:build !windows && cgo

package apiserver

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type managedKnowledgeBaseAssetsTxn struct {
	finalDir   string
	backupDir  string
	removeOnly bool
}

func stageManagedKnowledgeBaseAssets(baseDir string, sourcePath string, payload knowledgeBaseUpsertRequest) (*managedKnowledgeBaseAssetsTxn, error) {
	finalDir := managedKnowledgeBaseDirForSource(baseDir, sourcePath, payload.Name)
	parentDir := filepath.Dir(finalDir)
	if err := os.MkdirAll(parentDir, 0o755); err != nil {
		return nil, err
	}

	stageDir := finalDir + ".tmp-" + fmt.Sprintf("%d", time.Now().UnixNano())
	if err := os.MkdirAll(stageDir, 0o755); err != nil {
		return nil, err
	}

	if err := writeKnowledgeBaseAssets(stageDir, payload); err != nil {
		_ = os.RemoveAll(stageDir)
		return nil, err
	}

	txn := &managedKnowledgeBaseAssetsTxn{finalDir: finalDir}
	if _, err := os.Stat(finalDir); err == nil {
		txn.backupDir = finalDir + ".bak-" + fmt.Sprintf("%d", time.Now().UnixNano())
		if err := os.Rename(finalDir, txn.backupDir); err != nil {
			_ = os.RemoveAll(stageDir)
			return nil, err
		}
	}

	if err := os.Rename(stageDir, finalDir); err != nil {
		if txn.backupDir != "" {
			_ = os.Rename(txn.backupDir, finalDir)
		}
		_ = os.RemoveAll(stageDir)
		return nil, err
	}
	return txn, nil
}

func stageManagedKnowledgeBaseRemoval(baseDir string, sourcePath string, name string) (*managedKnowledgeBaseAssetsTxn, error) {
	finalDir := managedKnowledgeBaseDirForSource(baseDir, sourcePath, name)
	if _, err := os.Stat(finalDir); os.IsNotExist(err) {
		return nil, nil
	}
	backupDir := finalDir + ".bak-" + fmt.Sprintf("%d", time.Now().UnixNano())
	if err := os.Rename(finalDir, backupDir); err != nil {
		return nil, err
	}
	return &managedKnowledgeBaseAssetsTxn{
		finalDir:   finalDir,
		backupDir:  backupDir,
		removeOnly: true,
	}, nil
}

func (txn *managedKnowledgeBaseAssetsTxn) Commit() {
	if txn == nil || txn.backupDir == "" {
		return
	}
	_ = os.RemoveAll(txn.backupDir)
}

func (txn *managedKnowledgeBaseAssetsTxn) Rollback() {
	if txn == nil {
		return
	}
	if txn.removeOnly {
		if txn.backupDir != "" {
			_ = os.RemoveAll(txn.finalDir)
			_ = os.Rename(txn.backupDir, txn.finalDir)
		}
		return
	}
	_ = os.RemoveAll(txn.finalDir)
	if txn.backupDir != "" {
		_ = os.Rename(txn.backupDir, txn.finalDir)
	}
}

func writeKnowledgeBaseAssets(root string, payload knowledgeBaseUpsertRequest) error {
	definition := config.KnowledgeBaseDefinition{
		Version:     knowledgeBaseManifestVersion,
		Description: payload.Description,
		Labels:      make(map[string]config.KnowledgeBaseLabelDef, len(payload.Labels)),
	}
	for _, label := range payload.Labels {
		definition.Labels[label.Name] = config.KnowledgeBaseLabelDef{
			Description: label.Description,
			Exemplars:   append([]string(nil), label.Exemplars...),
		}
	}

	definitionBytes, err := json.MarshalIndent(definition, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(root, knowledgeBaseManifestName), definitionBytes, 0o644)
}
