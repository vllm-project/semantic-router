package handlers

import "net/http"

func (h *OpenClawHandler) canManageOpenClaw() bool {
	return !h.readOnly
}

func (h *OpenClawHandler) canRepairWorkerChatEndpoint() bool {
	return h.canManageOpenClaw()
}

func (h *OpenClawHandler) canSendRoomMessages() bool {
	return true
}

func (h *OpenClawHandler) writeReadOnlyError(w http.ResponseWriter) {
	http.Error(w, `{"error":"Read-only mode enabled"}`, http.StatusForbidden)
}
