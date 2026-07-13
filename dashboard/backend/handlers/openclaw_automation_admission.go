package handlers

import (
	"log"
	"strings"
)

type roomAutomationAdmission struct {
	pendingMessageID string
}

// enqueueRoomAutomation admits at most one worker per room and a bounded
// number globally. While a room is active, only its newest pending trigger is
// retained, which provides deterministic coalescing without accumulating a
// goroutine for every authenticated message.
func (h *OpenClawHandler) enqueueRoomAutomation(roomID, messageID string) bool {
	roomID = strings.TrimSpace(roomID)
	messageID = strings.TrimSpace(messageID)
	if h == nil || roomID == "" || messageID == "" {
		return false
	}

	h.roomAutomationAdmissionMu.Lock()
	if current, ok := h.roomAutomationAdmissions[roomID]; ok {
		current.pendingMessageID = messageID
		h.roomAutomationAdmissionMu.Unlock()
		return true
	}
	select {
	case h.roomAutomationSlots <- struct{}{}:
		admission := &roomAutomationAdmission{}
		h.roomAutomationAdmissions[roomID] = admission
		h.roomAutomationAdmissionMu.Unlock()
		go h.runRoomAutomationAdmission(roomID, messageID, admission)
		return true
	default:
		h.roomAutomationAdmissionMu.Unlock()
		log.Printf("openclaw: automation admission full room=%q", roomID)
		return false
	}
}

func (h *OpenClawHandler) runRoomAutomationAdmission(
	roomID string,
	messageID string,
	admission *roomAutomationAdmission,
) {
	released := false
	defer func() {
		if recovered := recover(); recovered != nil {
			log.Printf("openclaw: automation worker panic type=%T", recovered)
		}
		if !released {
			h.releaseRoomAutomationAdmission(roomID, admission)
		}
	}()

	currentMessageID := messageID
	for {
		processor := h.roomAutomationProcess
		if processor == nil {
			processor = h.processRoomUserMessage
		}
		processor(roomID, currentMessageID)

		h.roomAutomationAdmissionMu.Lock()
		current, ok := h.roomAutomationAdmissions[roomID]
		if !ok || current != admission {
			h.roomAutomationAdmissionMu.Unlock()
			return
		}
		if current.pendingMessageID != "" {
			currentMessageID = current.pendingMessageID
			current.pendingMessageID = ""
			h.roomAutomationAdmissionMu.Unlock()
			continue
		}
		delete(h.roomAutomationAdmissions, roomID)
		<-h.roomAutomationSlots
		released = true
		h.roomAutomationAdmissionMu.Unlock()
		return
	}
}

func (h *OpenClawHandler) releaseRoomAutomationAdmission(
	roomID string,
	admission *roomAutomationAdmission,
) {
	h.roomAutomationAdmissionMu.Lock()
	defer h.roomAutomationAdmissionMu.Unlock()
	if current, ok := h.roomAutomationAdmissions[roomID]; ok && current == admission {
		delete(h.roomAutomationAdmissions, roomID)
		<-h.roomAutomationSlots
	}
}
