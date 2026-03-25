//go:build !windows && cgo

package apiserver

import (
	"fmt"
	"net/http"
	"os"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func (s *ClassificationAPIServer) handleListTaxonomyClassifiers(w http.ResponseWriter, _ *http.Request) {
	cfg := s.currentConfig()
	if cfg == nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "CONFIG_UNAVAILABLE", "Classification config not available")
		return
	}

	documents, err := listTaxonomyClassifierDocuments(cfg, taxonomyClassifierConfigBaseDir(cfg, s.configPath))
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "CLASSIFIER_LIST_ERROR", err.Error())
		return
	}
	s.writeJSONResponse(w, http.StatusOK, taxonomyClassifierListResponse{Items: documents})
}

func (s *ClassificationAPIServer) handleGetTaxonomyClassifier(w http.ResponseWriter, r *http.Request) {
	cfg := s.currentConfig()
	if cfg == nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "CONFIG_UNAVAILABLE", "Classification config not available")
		return
	}

	name := strings.TrimSpace(r.PathValue("name"))
	classifier, ok := taxonomyClassifierByName(cfg.TaxonomyClassifiers, name)
	if !ok {
		s.writeErrorResponse(w, http.StatusNotFound, "CLASSIFIER_NOT_FOUND", fmt.Sprintf("taxonomy classifier %q was not found", name))
		return
	}

	document, err := buildTaxonomyClassifierDocument(cfg, taxonomyClassifierConfigBaseDir(cfg, s.configPath), classifier)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "CLASSIFIER_READ_ERROR", err.Error())
		return
	}
	s.writeJSONResponse(w, http.StatusOK, document)
}

func (s *ClassificationAPIServer) handleCreateTaxonomyClassifier(w http.ResponseWriter, r *http.Request) {
	cfg := s.currentConfig()
	if cfg == nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "CONFIG_UNAVAILABLE", "Classification config not available")
		return
	}
	if s.configPath == "" {
		s.writeErrorResponse(w, http.StatusInternalServerError, "NO_CONFIG_PATH", "Router configPath not set")
		return
	}

	var payload taxonomyClassifierUpsertRequest
	if err := s.parseJSONRequest(r, &payload); err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", err.Error())
		return
	}
	payload, err := normalizeTaxonomyClassifierRequest(payload)
	if err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", err.Error())
		return
	}
	if _, exists := taxonomyClassifierByName(cfg.TaxonomyClassifiers, payload.Name); exists {
		s.writeErrorResponse(w, http.StatusConflict, "CLASSIFIER_EXISTS", fmt.Sprintf("taxonomy classifier %q already exists", payload.Name))
		return
	}

	newClassifier := config.TaxonomyClassifierConfig{
		Name:              payload.Name,
		Type:              config.ClassifierTypeTaxonomy,
		Threshold:         payload.Threshold,
		SecurityThreshold: payload.SecurityThreshold,
		Source: config.TaxonomyClassifierSource{
			Path:         managedTaxonomyClassifierSourcePath(payload.Name),
			TaxonomyFile: "taxonomy.json",
		},
	}

	if err := s.persistManagedTaxonomyClassifier(w, cfg, payload, newClassifier, desiredTaxonomyClassifiers(cfg.TaxonomyClassifiers, newClassifier), http.StatusCreated); err != nil {
		return
	}
}

func (s *ClassificationAPIServer) handleUpdateTaxonomyClassifier(w http.ResponseWriter, r *http.Request) {
	cfg := s.currentConfig()
	if cfg == nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "CONFIG_UNAVAILABLE", "Classification config not available")
		return
	}
	if s.configPath == "" {
		s.writeErrorResponse(w, http.StatusInternalServerError, "NO_CONFIG_PATH", "Router configPath not set")
		return
	}

	name := strings.TrimSpace(r.PathValue("name"))
	existingClassifier, ok := taxonomyClassifierByName(cfg.TaxonomyClassifiers, name)
	if !ok {
		s.writeErrorResponse(w, http.StatusNotFound, "CLASSIFIER_NOT_FOUND", fmt.Sprintf("taxonomy classifier %q was not found", name))
		return
	}
	if !existingClassifierEditable(existingClassifier) {
		s.writeErrorResponse(w, http.StatusForbidden, "CLASSIFIER_READ_ONLY", fmt.Sprintf("taxonomy classifier %q is router-managed and cannot be updated", name))
		return
	}

	var payload taxonomyClassifierUpsertRequest
	if err := s.parseJSONRequest(r, &payload); err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", err.Error())
		return
	}
	payload.Name = name
	payload, err := normalizeTaxonomyClassifierRequest(payload)
	if err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", err.Error())
		return
	}

	updatedClassifier := existingClassifier
	updatedClassifier.Type = config.ClassifierTypeTaxonomy
	updatedClassifier.Threshold = payload.Threshold
	updatedClassifier.SecurityThreshold = payload.SecurityThreshold
	updatedClassifier.Source = config.TaxonomyClassifierSource{
		Path:         managedTaxonomyClassifierSourcePath(name),
		TaxonomyFile: "taxonomy.json",
	}

	if err := s.persistManagedTaxonomyClassifier(w, cfg, payload, updatedClassifier, desiredTaxonomyClassifiers(cfg.TaxonomyClassifiers, updatedClassifier), http.StatusOK); err != nil {
		return
	}
}

func (s *ClassificationAPIServer) handleDeleteTaxonomyClassifier(w http.ResponseWriter, r *http.Request) {
	cfg := s.currentConfig()
	if cfg == nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "CONFIG_UNAVAILABLE", "Classification config not available")
		return
	}
	if s.configPath == "" {
		s.writeErrorResponse(w, http.StatusInternalServerError, "NO_CONFIG_PATH", "Router configPath not set")
		return
	}

	name := strings.TrimSpace(r.PathValue("name"))
	existingClassifier, ok := taxonomyClassifierByName(cfg.TaxonomyClassifiers, name)
	if !ok {
		s.writeErrorResponse(w, http.StatusNotFound, "CLASSIFIER_NOT_FOUND", fmt.Sprintf("taxonomy classifier %q was not found", name))
		return
	}
	if !existingClassifierEditable(existingClassifier) {
		s.writeErrorResponse(w, http.StatusForbidden, "CLASSIFIER_READ_ONLY", fmt.Sprintf("taxonomy classifier %q is router-managed and cannot be deleted", name))
		return
	}

	paths := resolveConfigPersistencePaths(s.configPath)
	baseDir := taxonomyClassifierConfigBaseDir(cfg, s.configPath)
	existingData, err := os.ReadFile(paths.sourcePath)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "READ_ERROR", fmt.Sprintf("failed to read config: %v", err))
		return
	}

	removeTxn, err := stageManagedTaxonomyClassifierRemoval(baseDir, name)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "ASSET_STAGE_ERROR", err.Error())
		return
	}

	updatedYAML, err := taxonomyClassifierOverrideYAML(existingData, removeTaxonomyClassifier(cfg.TaxonomyClassifiers, name))
	if err != nil {
		if removeTxn != nil {
			removeTxn.Rollback()
		}
		s.writeErrorResponse(w, http.StatusInternalServerError, "CONFIG_PATCH_ERROR", err.Error())
		return
	}

	newCfg, err := validateConfigWithBaseDir(baseDir, updatedYAML)
	if err != nil {
		if removeTxn != nil {
			removeTxn.Rollback()
		}
		s.writeErrorResponse(w, http.StatusBadRequest, "CONFIG_PARSE_ERROR", err.Error())
		return
	}

	persistErr := persistConfigAndSync(s, paths, existingData, updatedYAML, newCfg)
	if persistErr != nil {
		if removeTxn != nil {
			removeTxn.Rollback()
		}
		s.writeErrorResponse(w, http.StatusInternalServerError, "CONFIG_PERSIST_ERROR", persistErr.Error())
		return
	}
	if removeTxn != nil {
		removeTxn.Commit()
	}
	s.writeJSONResponse(w, http.StatusOK, taxonomyClassifierDeleteResponse{
		Status: "deleted",
		Name:   name,
	})
}

func (s *ClassificationAPIServer) persistManagedTaxonomyClassifier(
	w http.ResponseWriter,
	cfg *config.RouterConfig,
	payload taxonomyClassifierUpsertRequest,
	classifier config.TaxonomyClassifierConfig,
	desired []config.TaxonomyClassifierConfig,
	successStatus int,
) error {
	paths := resolveConfigPersistencePaths(s.configPath)
	baseDir := taxonomyClassifierConfigBaseDir(cfg, s.configPath)
	existingData, err := os.ReadFile(paths.sourcePath)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "READ_ERROR", fmt.Sprintf("failed to read config: %v", err))
		return err
	}

	assetTxn, err := stageManagedTaxonomyClassifierAssets(baseDir, payload)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "ASSET_STAGE_ERROR", err.Error())
		return err
	}

	updatedYAML, err := taxonomyClassifierOverrideYAML(existingData, desired)
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

	document, err := buildTaxonomyClassifierDocument(newCfg, baseDir, classifier)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "CLASSIFIER_READ_ERROR", err.Error())
		return err
	}
	s.writeJSONResponse(w, successStatus, document)
	return nil
}

func taxonomyClassifierByName(
	classifiers []config.TaxonomyClassifierConfig,
	name string,
) (config.TaxonomyClassifierConfig, bool) {
	for _, classifier := range classifiers {
		if classifier.Name == name {
			return classifier, true
		}
	}
	return config.TaxonomyClassifierConfig{}, false
}

func existingClassifierEditable(classifier config.TaxonomyClassifierConfig) bool {
	return isManagedTaxonomyClassifier(classifier) && !isBuiltinTaxonomyClassifier(classifier)
}
