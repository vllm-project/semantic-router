package decisionmedia

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"

// FromRequest retains user attachment references in input order. Validation and
// limits belong to Select; extraction never fetches or decodes media.
func FromRequest(request *llmprotocol.Request) []*Source {
	if request == nil {
		return nil
	}
	var sources []*Source
	for _, message := range request.Messages {
		if message.Role != llmprotocol.RoleUser {
			continue
		}
		for _, part := range message.Content {
			if part.Kind == llmprotocol.ContentImage || part.Kind == llmprotocol.ContentVideo {
				sources = append(sources, &Source{Kind: string(part.Kind), URL: part.URL, MediaType: part.MediaType, Data: part.Data})
			}
		}
	}
	return sources
}
