package handlers

import (
	"net/http"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
)

func rejectRevokedMutation(w http.ResponseWriter, r *http.Request) bool {
	return auth.RejectRevokedMutation(w, r)
}
