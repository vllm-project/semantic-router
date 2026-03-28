//go:build !windows && cgo

package apiserver

import (
	"fmt"
	"net/http"
	"os"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func (s *ClassificationAPIServer) handleListKnowledgeBases(w http.ResponseWriter, _ *http.Request) {
	cfg := s.currentConfig()
	if cfg == nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "CONFIG_UNAVAILABLE", "Classification config not available")
		return
	}

	documents, err := listKnowledgeBaseDocuments(cfg, knowledgeBaseConfigBaseDir(cfg, s.configPath))
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "KB_LIST_ERROR", err.Error())
		return
	}
	s.writeJSONResponse(w, http.StatusOK, knowledgeBaseListResponse{Items: documents})
}

func (s *ClassificationAPIServer) handleGetKnowledgeBase(w http.ResponseWriter, r *http.Request) {
	cfg := s.currentConfig()
	if cfg == nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "CONFIG_UNAVAILABLE", "Classification config not available")
		return
	}

	name := strings.TrimSpace(r.PathValue("name"))
	kb, ok := knowledgeBaseByName(cfg.KnowledgeBases, name)
	if !ok {
		s.writeErrorResponse(w, http.StatusNotFound, "KB_NOT_FOUND", fmt.Sprintf("knowledge base %q was not found", name))
		return
	}

	document, err := buildKnowledgeBaseDocument(cfg, knowledgeBaseConfigBaseDir(cfg, s.configPath), kb)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "KB_READ_ERROR", err.Error())
		return
	}
	s.writeJSONResponse(w, http.StatusOK, document)
}

func (s *ClassificationAPIServer) handleCreateKnowledgeBase(w http.ResponseWriter, r *http.Request) {
	cfg, ok := s.writableKnowledgeBaseConfig(w)
	if !ok {
		return
	}

	var payload knowledgeBaseUpsertRequest
	if err := s.parseJSONRequest(r, &payload); err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", err.Error())
		return
	}
	payload, err := normalizeKnowledgeBaseRequest(payload)
	if err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", err.Error())
		return
	}
	if _, exists := knowledgeBaseByName(cfg.KnowledgeBases, payload.Name); exists {
		s.writeErrorResponse(w, http.StatusConflict, "KB_EXISTS", fmt.Sprintf("knowledge base %q already exists", payload.Name))
		return
	}

	newKB := config.KnowledgeBaseConfig{
		Name:            payload.Name,
		Threshold:       payload.Threshold,
		LabelThresholds: cloneLabelThresholds(payload.LabelThresholds),
		Groups:          cloneKnowledgeBaseGroups(payload.Groups),
		Metrics:         cloneKnowledgeBaseMetrics(payload.Metrics),
		Source: config.KnowledgeBaseSource{
			Path:     managedKnowledgeBaseSourcePath(payload.Name),
			Manifest: knowledgeBaseManifestName,
		},
	}

	if err := s.persistManagedKnowledgeBase(w, cfg, payload, newKB, desiredKnowledgeBases(cfg.KnowledgeBases, newKB), http.StatusCreated); err != nil {
		return
	}
}

func (s *ClassificationAPIServer) handleUpdateKnowledgeBase(w http.ResponseWriter, r *http.Request) {
	cfg, ok := s.writableKnowledgeBaseConfig(w)
	if !ok {
		return
	}

	name := strings.TrimSpace(r.PathValue("name"))
	existingKB, ok := knowledgeBaseByName(cfg.KnowledgeBases, name)
	if !ok {
		s.writeErrorResponse(w, http.StatusNotFound, "KB_NOT_FOUND", fmt.Sprintf("knowledge base %q was not found", name))
		return
	}
	if !existingKnowledgeBaseEditable(existingKB) {
		s.writeErrorResponse(w, http.StatusForbidden, "KB_READ_ONLY", fmt.Sprintf("knowledge base %q is router-managed and cannot be updated", name))
		return
	}

	var payload knowledgeBaseUpsertRequest
	if err := s.parseJSONRequest(r, &payload); err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", err.Error())
		return
	}
	payload.Name = name
	payload, err := normalizeKnowledgeBaseRequest(payload)
	if err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", err.Error())
		return
	}

	updatedKB := existingKB
	updatedKB.Threshold = payload.Threshold
	updatedKB.LabelThresholds = cloneLabelThresholds(payload.LabelThresholds)
	updatedKB.Groups = cloneKnowledgeBaseGroups(payload.Groups)
	updatedKB.Metrics = cloneKnowledgeBaseMetrics(payload.Metrics)
	updatedKB.Source = config.KnowledgeBaseSource{
		Path:     managedKnowledgeBaseSourcePath(name),
		Manifest: knowledgeBaseManifestName,
	}

	if err := s.persistManagedKnowledgeBase(w, cfg, payload, updatedKB, desiredKnowledgeBases(cfg.KnowledgeBases, updatedKB), http.StatusOK); err != nil {
		return
	}
}

func (s *ClassificationAPIServer) handleDeleteKnowledgeBase(w http.ResponseWriter, r *http.Request) {
	cfg, ok := s.writableKnowledgeBaseConfig(w)
	if !ok {
		return
	}

	name := strings.TrimSpace(r.PathValue("name"))
	existingKB, ok := knowledgeBaseByName(cfg.KnowledgeBases, name)
	if !ok {
		s.writeErrorResponse(w, http.StatusNotFound, "KB_NOT_FOUND", fmt.Sprintf("knowledge base %q was not found", name))
		return
	}
	if !existingKnowledgeBaseEditable(existingKB) {
		s.writeErrorResponse(w, http.StatusForbidden, "KB_READ_ONLY", fmt.Sprintf("knowledge base %q is router-managed and cannot be deleted", name))
		return
	}

	paths := resolveConfigPersistencePaths(s.configPath)
	baseDir := knowledgeBaseConfigBaseDir(cfg, s.configPath)
	existingData, err := os.ReadFile(paths.sourcePath)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "READ_ERROR", fmt.Sprintf("failed to read config: %v", err))
		return
	}

	removeTxn, err := stageManagedKnowledgeBaseRemoval(baseDir, name)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "ASSET_STAGE_ERROR", err.Error())
		return
	}
	committed := false
	defer rollbackManagedKnowledgeBaseRemoval(removeTxn, &committed)

	updatedYAML, err := knowledgeBaseOverrideYAML(existingData, removeKnowledgeBase(cfg.KnowledgeBases, name))
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "CONFIG_PATCH_ERROR", err.Error())
		return
	}

	newCfg, err := validateConfigWithBaseDir(baseDir, updatedYAML)
	if err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "CONFIG_PARSE_ERROR", err.Error())
		return
	}

	if err := persistConfigAndSync(s, paths, existingData, updatedYAML, newCfg); err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "CONFIG_PERSIST_ERROR", err.Error())
		return
	}
	if removeTxn != nil {
		removeTxn.Commit()
		committed = true
	}
	s.writeJSONResponse(w, http.StatusOK, knowledgeBaseDeleteResponse{
		Status: "deleted",
		Name:   name,
	})
}

func rollbackManagedKnowledgeBaseRemoval(txn *managedKnowledgeBaseAssetsTxn, committed *bool) {
	if txn == nil || committed == nil || *committed {
		return
	}
	txn.Rollback()
}

func (s *ClassificationAPIServer) writableKnowledgeBaseConfig(w http.ResponseWriter) (*config.RouterConfig, bool) {
	cfg := s.currentConfig()
	if cfg == nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "CONFIG_UNAVAILABLE", "Classification config not available")
		return nil, false
	}
	if s.configPath == "" {
		s.writeErrorResponse(w, http.StatusInternalServerError, "NO_CONFIG_PATH", "Router configPath not set")
		return nil, false
	}
	return cfg, true
}

func (s *ClassificationAPIServer) persistManagedKnowledgeBase(
	w http.ResponseWriter,
	cfg *config.RouterConfig,
	payload knowledgeBaseUpsertRequest,
	kb config.KnowledgeBaseConfig,
	desired []config.KnowledgeBaseConfig,
	successStatus int,
) error {
	paths := resolveConfigPersistencePaths(s.configPath)
	baseDir := knowledgeBaseConfigBaseDir(cfg, s.configPath)
	existingData, err := os.ReadFile(paths.sourcePath)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "READ_ERROR", fmt.Sprintf("failed to read config: %v", err))
		return err
	}

	assetTxn, err := stageManagedKnowledgeBaseAssets(baseDir, payload)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "ASSET_STAGE_ERROR", err.Error())
		return err
	}

	updatedYAML, err := knowledgeBaseOverrideYAML(existingData, desired)
	if err != nil {
		assetTxn.Rollback()
		s.writeErrorResponse(w, http.StatusInternalServerError, "CONFIG_PATCH_ERROR", err.Error())
		return err
	}

	newCfg, err := validateConfigWithBaseDir(baseDir, updatedYAML)
	if err != nil {
		assetTxn.Rollback()
		s.writeErrorResponse(w, http.StatusBadRequest, "CONFIG_PARSE_ERROR", err.Error())
		return err
	}

	persistErr := persistConfigAndSync(s, paths, existingData, updatedYAML, newCfg)
	if persistErr != nil {
		assetTxn.Rollback()
		s.writeErrorResponse(w, http.StatusInternalServerError, "CONFIG_PERSIST_ERROR", persistErr.Error())
		return persistErr
	}
	assetTxn.Commit()

	document, err := buildKnowledgeBaseDocument(newCfg, baseDir, kb)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "KB_READ_ERROR", err.Error())
		return err
	}
	s.writeJSONResponse(w, successStatus, document)
	return nil
}

func knowledgeBaseByName(kbs []config.KnowledgeBaseConfig, name string) (config.KnowledgeBaseConfig, bool) {
	for _, kb := range kbs {
		if kb.Name == name {
			return kb, true
		}
	}
	return config.KnowledgeBaseConfig{}, false
}

func existingKnowledgeBaseEditable(kb config.KnowledgeBaseConfig) bool {
	return kb.Name != ""
}
