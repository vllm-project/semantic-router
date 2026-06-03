package handlers

import (
	"context"
	"log"
	"strings"
	"time"
)

func (h *OpenClawHandler) resolveOpenClawSessionToolTraceFile(
	ctx context.Context,
	worker ContainerEntry,
	sessionKeys []string,
	knownSessionFile string,
) string {
	sessionFile := strings.TrimSpace(knownSessionFile)
	if sessionFile != "" {
		return sessionFile
	}

	deadline := time.Now().Add(openClawSessionPollTimeout)
	for sessionFile == "" && time.Now().Before(deadline) {
		select {
		case <-ctx.Done():
			return ""
		default:
		}
		resolved, err := h.resolveOpenClawSessionFile(worker.Name, sessionKeys)
		if err == nil {
			return resolved
		}
		time.Sleep(openClawSessionPollInterval)
	}
	return ""
}

func (h *OpenClawHandler) readOpenClawSessionToolTraceStartOffset(
	worker ContainerEntry,
	sessionFile string,
	baselineOffset int64,
) int64 {
	offset := baselineOffset
	if data, err := h.readOpenClawContainerFile(worker.Name, sessionFile); err == nil {
		return openClawSessionToolTraceInitialOffset(baselineOffset, int64(len(data)))
	}
	if offset < 0 {
		return 0
	}
	return offset
}

func (h *OpenClawHandler) publishOpenClawSessionToolTraceUpdate(
	roomID string,
	room ClawRoomEntry,
	worker ContainerEntry,
	messageID string,
	steps map[string]openClawSessionToolStep,
	stepOrder []string,
	revision int,
) int {
	revision++
	orderedSteps := orderedOpenClawSessionToolSteps(steps, stepOrder)
	payload := openClawSessionToolTracePayload{
		Revision: revision,
		Steps:    orderedSteps,
	}
	openClawSessionToolTraceStepRegistry.Store(messageID, append([]openClawSessionToolStep(nil), orderedSteps...))
	h.publishRoomCollaborationEvent(
		roomID,
		toolTraceUpdateCollaborationEvent(room, worker, messageID, payload),
	)
	return revision
}

func (h *OpenClawHandler) pollOpenClawSessionToolTraceOnce(
	roomID string,
	room ClawRoomEntry,
	worker ContainerEntry,
	messageID string,
	sessionFile string,
	steps map[string]openClawSessionToolStep,
	stepOrder *[]string,
	offset *int64,
	revision *int,
) {
	data, err := h.readOpenClawContainerFile(worker.Name, sessionFile)
	if err != nil {
		return
	}

	if int64(len(data)) < *offset {
		*offset = 0
		for key := range steps {
			delete(steps, key)
		}
		*stepOrder = (*stepOrder)[:0]
	}
	if int64(len(data)) == *offset {
		return
	}

	newData := data[*offset:]
	*offset = int64(len(data))

	lines, err := splitOpenClawSessionToolTraceLines(newData)
	if err != nil {
		log.Printf("openclaw: failed to tail session tool trace lines in %s: %v", sessionFile, err)
	}

	var changed bool
	*stepOrder, changed = parseOpenClawSessionToolTraceLines(lines, steps, *stepOrder)
	if !changed {
		return
	}

	*revision = h.publishOpenClawSessionToolTraceUpdate(
		roomID,
		room,
		worker,
		messageID,
		steps,
		*stepOrder,
		*revision,
	)
}

func (h *OpenClawHandler) streamOpenClawSessionToolTrace(
	ctx context.Context,
	roomID string,
	room ClawRoomEntry,
	worker ContainerEntry,
	messageID string,
	knownSessionFile string,
	baselineOffset int64,
) {
	sessionUser := roomScopedSessionUser(room, worker)
	sessionKeys := openClawChatCompletionsSessionKeys(sessionUser)
	sessionFile := h.resolveOpenClawSessionToolTraceFile(ctx, worker, sessionKeys, knownSessionFile)
	if sessionFile == "" {
		return
	}

	steps := make(map[string]openClawSessionToolStep)
	stepOrder := make([]string, 0)
	offset := h.readOpenClawSessionToolTraceStartOffset(worker, sessionFile, baselineOffset)
	var revision int

	for {
		select {
		case <-ctx.Done():
			return
		default:
		}

		h.pollOpenClawSessionToolTraceOnce(
			roomID,
			room,
			worker,
			messageID,
			sessionFile,
			steps,
			&stepOrder,
			&offset,
			&revision,
		)
		time.Sleep(openClawSessionPollInterval)
	}
}
