package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

type omniInventoryModel struct {
	Recipe   string            `json:"recipe"`
	Type     string            `json:"type"`
	Loaded   bool              `json:"loaded"`
	Metadata map[string]string `json:"metadata"`
}

// The profile's default recipe has no embedding binding. Named owners must
// still appear here, using the same metadata as the general model inventory.
func (p omniProbe) checkEmbeddingOwners(ctx context.Context) error {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, p.apiURL+"/api/v1/inventory/embedding-models", nil)
	if err != nil {
		return err
	}
	response, err := p.client.Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("embedding inventory returned HTTP %d", response.StatusCode)
	}
	var inventory struct {
		Models []omniInventoryModel `json:"models"`
		Count  int                  `json:"count"`
	}
	if err := json.NewDecoder(io.LimitReader(response.Body, 1<<20)).Decode(&inventory); err != nil {
		return err
	}
	if inventory.Count != len(inventory.Models) {
		return fmt.Errorf("embedding inventory count %d differs from %d owner rows", inventory.Count, len(inventory.Models))
	}
	for recipe, dimension := range map[string]string{"nano": "384", "mini": "768"} {
		owners := 0
		for _, model := range inventory.Models {
			if model.Recipe != recipe || model.Metadata["binding"] != "embedding" {
				continue
			}
			owners++
			if model.Type != "embedding" || !model.Loaded {
				return fmt.Errorf("%s embedding owner is not a loaded embedding: %+v", recipe, model)
			}
			for key, expected := range map[string]string{
				"deployment": recipe, "contract": "embedding.v1", "model_type": "vela_omni",
				"provider": "ort", "default_dimension": dimension, "modalities": "text,image,audio",
			} {
				if model.Metadata[key] != expected {
					return fmt.Errorf("%s embedding owner metadata %s=%q, expected %q", recipe, key, model.Metadata[key], expected)
				}
			}
		}
		if owners != 1 {
			return fmt.Errorf("embedding inventory has %d %s/embedding owners, expected 1", owners, recipe)
		}
	}
	return nil
}
