package handlers

func buildWorkerChatRequest(messages []openAIChatMessage, sessionUser string, stream bool) openAIChatRequest {
	requestMessages := append([]openAIChatMessage(nil), messages...)
	return openAIChatRequest{
		Model:    openClawPrimaryAgentModel,
		Messages: requestMessages,
		Stream:   stream,
		User:     sessionUser,
	}
}

func roomScopedSessionUser(room ClawRoomEntry, worker ContainerEntry) string {
	roomID := sanitizeRoomID(room.ID)
	if roomID == "" {
		roomID = "room"
	}

	workerID := sanitizeContainerName(worker.Name)
	if workerID == "" {
		workerID = "claw"
	}

	return roomID + ":" + workerID
}
