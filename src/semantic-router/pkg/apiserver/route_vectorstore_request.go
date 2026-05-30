//go:build !windows && cgo

package apiserver

import (
	"net/http"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/vectorstore"
)

const (
	defaultVectorStoreListLimit = 20
	maxVectorStoreListLimit     = 100
)

func (s *ClassificationAPIServer) parseVectorStoreListParams(
	w http.ResponseWriter,
	r *http.Request,
) (vectorstore.ListStoresParams, bool) {
	query := r.URL.Query()

	limit, ok := s.parseVectorStoreListLimit(w, query.Get("limit"))
	if !ok {
		return vectorstore.ListStoresParams{}, false
	}
	order, ok := s.parseVectorStoreListOrder(w, query.Get("order"))
	if !ok {
		return vectorstore.ListStoresParams{}, false
	}

	after := strings.TrimSpace(query.Get("after"))
	before := strings.TrimSpace(query.Get("before"))
	if after != "" && before != "" {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_CURSOR", "after and before cursors are mutually exclusive")
		return vectorstore.ListStoresParams{}, false
	}

	return vectorstore.ListStoresParams{
		Limit:  limit,
		Order:  order,
		After:  after,
		Before: before,
	}, true
}

func (s *ClassificationAPIServer) parseVectorStoreListLimit(
	w http.ResponseWriter,
	limitStr string,
) (int, bool) {
	if limitStr == "" {
		return defaultVectorStoreListLimit, true
	}

	limit, err := strconv.Atoi(limitStr)
	if err != nil || limit <= 0 {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_LIMIT", "limit must be a positive integer")
		return 0, false
	}
	if limit > maxVectorStoreListLimit {
		return maxVectorStoreListLimit, true
	}
	return limit, true
}

func (s *ClassificationAPIServer) parseVectorStoreListOrder(
	w http.ResponseWriter,
	order string,
) (string, bool) {
	order = strings.TrimSpace(order)
	if order == "" {
		return "desc", true
	}
	if order == "asc" || order == "desc" {
		return order, true
	}

	s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_ORDER", "order must be asc or desc")
	return "", false
}
