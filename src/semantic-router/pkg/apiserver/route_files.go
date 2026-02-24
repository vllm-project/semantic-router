//go:build !windows && cgo

/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package apiserver

import (
	"fmt"
	"io"
	"net/http"
	"path/filepath"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/vectorstore"
)

const maxUploadSize = 50 * 1024 * 1024 // 50MB

// allowedExtensions defines the file types that can be uploaded.
var allowedExtensions = map[string]bool{
	".txt":  true,
	".md":   true,
	".json": true,
	".csv":  true,
	".html": true,
	".htm":  true,
}

// globalFileStore is the global file store instance.
// It is set during initialization via SetFileStore.
var globalFileStore *vectorstore.FileStore

// SetFileStore sets the global file store for the API server.
func SetFileStore(fs *vectorstore.FileStore) {
	globalFileStore = fs
}

// registerFileRoutes registers file management routes.
func registerFileRoutes(mux *http.ServeMux, s *ClassificationAPIServer) {
	mux.HandleFunc("POST /v1/files", s.handleUploadFile)
	mux.HandleFunc("GET /v1/files", s.handleListFiles)
	mux.HandleFunc("GET /v1/files/{id}", s.handleGetFile)
	mux.HandleFunc("DELETE /v1/files/{id}", s.handleDeleteFile)
	mux.HandleFunc("GET /v1/files/{id}/content", s.handleGetFileContent)
}

func (s *ClassificationAPIServer) handleUploadFile(w http.ResponseWriter, r *http.Request) {
	if globalFileStore == nil {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "FILE_STORE_DISABLED", "file storage is not enabled")
		return
	}

	// Limit upload size.
	r.Body = http.MaxBytesReader(w, r.Body, maxUploadSize)

	if err := r.ParseMultipartForm(maxUploadSize); err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT",
			fmt.Sprintf("failed to parse multipart form (max size: %dMB): %s", maxUploadSize/(1024*1024), err.Error()))
		return
	}

	file, header, err := r.FormFile("file")
	if err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", "file field is required")
		return
	}
	defer file.Close()

	purpose := r.FormValue("purpose")
	if purpose == "" {
		purpose = "assistants"
	}

	// Validate extension.
	ext := strings.ToLower(filepath.Ext(header.Filename))
	if !allowedExtensions[ext] {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_FILE_TYPE",
			fmt.Sprintf("unsupported file type: %s (allowed: .txt, .md, .json, .csv, .html)", ext))
		return
	}

	content, err := io.ReadAll(file)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "READ_ERROR", "failed to read file content")
		return
	}

	record, err := globalFileStore.Save(header.Filename, content, purpose)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "SAVE_ERROR", "failed to save file")
		return
	}

	s.writeJSONResponse(w, http.StatusOK, record)
}

func (s *ClassificationAPIServer) handleListFiles(w http.ResponseWriter, r *http.Request) {
	if globalFileStore == nil {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "FILE_STORE_DISABLED", "file storage is not enabled")
		return
	}

	purposeFilter := r.URL.Query().Get("purpose")
	records := globalFileStore.List()

	// Filter by purpose if specified.
	if purposeFilter != "" {
		filtered := make([]*vectorstore.FileRecord, 0)
		for _, r := range records {
			if r.Purpose == purposeFilter {
				filtered = append(filtered, r)
			}
		}
		records = filtered
	}

	response := map[string]interface{}{
		"object": "list",
		"data":   records,
	}
	s.writeJSONResponse(w, http.StatusOK, response)
}

func (s *ClassificationAPIServer) handleGetFile(w http.ResponseWriter, r *http.Request) {
	if globalFileStore == nil {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "FILE_STORE_DISABLED", "file storage is not enabled")
		return
	}

	id := extractPathParam(r.URL.Path, "/v1/files/")
	if id == "" {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", "file ID is required")
		return
	}

	record, err := globalFileStore.Get(id)
	if err != nil {
		s.writeErrorResponse(w, http.StatusNotFound, "NOT_FOUND", err.Error())
		return
	}

	s.writeJSONResponse(w, http.StatusOK, record)
}

func (s *ClassificationAPIServer) handleDeleteFile(w http.ResponseWriter, r *http.Request) {
	if globalFileStore == nil {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "FILE_STORE_DISABLED", "file storage is not enabled")
		return
	}

	id := extractPathParam(r.URL.Path, "/v1/files/")
	if id == "" {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", "file ID is required")
		return
	}

	if err := globalFileStore.Delete(id); err != nil {
		s.writeErrorResponse(w, http.StatusNotFound, "NOT_FOUND", err.Error())
		return
	}

	s.writeJSONResponse(w, http.StatusOK, map[string]interface{}{
		"id":      id,
		"object":  "file",
		"deleted": true,
	})
}

func (s *ClassificationAPIServer) handleGetFileContent(w http.ResponseWriter, r *http.Request) {
	if globalFileStore == nil {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "FILE_STORE_DISABLED", "file storage is not enabled")
		return
	}

	// Extract file ID from /v1/files/{id}/content
	path := strings.TrimPrefix(r.URL.Path, "/v1/files/")
	id := strings.TrimSuffix(path, "/content")
	if id == "" || id == path {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", "file ID is required")
		return
	}

	record, err := globalFileStore.Get(id)
	if err != nil {
		s.writeErrorResponse(w, http.StatusNotFound, "NOT_FOUND", "file not found")
		return
	}

	content, err := globalFileStore.Read(id)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "READ_ERROR", "failed to read file content")
		return
	}

	w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=%q", record.Filename))
	w.Header().Set("Content-Type", "application/octet-stream")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(content)
}
