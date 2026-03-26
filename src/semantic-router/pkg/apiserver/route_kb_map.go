//go:build !windows && cgo

package apiserver

import (
	"fmt"
	"net/http"
	"strings"
)

func (s *ClassificationAPIServer) handleGetKnowledgeBaseMapMetadata(w http.ResponseWriter, r *http.Request) {
	artifacts, ok := s.loadKnowledgeBaseMapArtifacts(w, r)
	if !ok {
		return
	}
	w.Header().Set("Cache-Control", "no-cache")
	s.writeJSONResponse(w, http.StatusOK, artifacts.metadata)
}

func (s *ClassificationAPIServer) handleGetKnowledgeBaseMapData(w http.ResponseWriter, r *http.Request) {
	artifacts, ok := s.loadKnowledgeBaseMapArtifacts(w, r)
	if !ok {
		return
	}
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(artifacts.pointData)
}

func (s *ClassificationAPIServer) handleGetKnowledgeBaseMapGrid(w http.ResponseWriter, r *http.Request) {
	artifacts, ok := s.loadKnowledgeBaseMapArtifacts(w, r)
	if !ok {
		return
	}
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(artifacts.gridData)
}

func (s *ClassificationAPIServer) handleGetKnowledgeBaseMapTopic(w http.ResponseWriter, r *http.Request) {
	artifacts, ok := s.loadKnowledgeBaseMapArtifacts(w, r)
	if !ok {
		return
	}
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(artifacts.topicData)
}

func (s *ClassificationAPIServer) loadKnowledgeBaseMapArtifacts(
	w http.ResponseWriter,
	r *http.Request,
) (*knowledgeBaseMapArtifacts, bool) {
	cfg := s.currentConfig()
	if cfg == nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "CONFIG_UNAVAILABLE", "Classification config not available")
		return nil, false
	}

	name := strings.TrimSpace(r.PathValue("name"))
	kb, ok := knowledgeBaseByName(cfg.KnowledgeBases, name)
	if !ok {
		s.writeErrorResponse(w, http.StatusNotFound, "KB_NOT_FOUND", fmt.Sprintf("knowledge base %q was not found", name))
		return nil, false
	}

	artifacts, err := s.ensureKnowledgeBaseMapArtifacts(cfg, knowledgeBaseConfigBaseDir(cfg, s.configPath), kb)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "KB_MAP_BUILD_ERROR", err.Error())
		return nil, false
	}
	return artifacts, true
}
