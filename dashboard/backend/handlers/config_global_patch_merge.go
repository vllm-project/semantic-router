package handlers

import "gopkg.in/yaml.v3"

var remoteEmbeddingEndpointPath = []string{
	"model_catalog",
	"embeddings",
	"semantic",
	"endpoint",
}

func isRemoteEmbeddingEndpointDelete(path []string, _ *yaml.Node) bool {
	if len(path) == len(remoteEmbeddingEndpointPath) {
		for index, segment := range remoteEmbeddingEndpointPath {
			if path[index] != segment {
				return false
			}
		}
		return true
	}
	if len(path) != len(remoteEmbeddingEndpointPath)+1 {
		return false
	}
	for index, segment := range remoteEmbeddingEndpointPath {
		if path[index] != segment {
			return false
		}
	}
	switch path[len(path)-1] {
	case "api_key_env", "timeout_seconds", "max_retries", "dimensions":
		return true
	default:
		return false
	}
}
