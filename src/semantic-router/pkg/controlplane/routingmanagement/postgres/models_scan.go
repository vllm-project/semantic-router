package postgres

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
)

type rowScanner interface {
	Scan(...any) error
}

func scanModelRow(row rowScanner) (routingmanagement.Model, error) {
	var result routingmanagement.Model
	var aliases, capabilities, reasoning, loras, tags, execution, pricing []byte
	if err := row.Scan(
		&result.NamespaceID, &result.ID, &result.Name, &result.Status, &result.Revision,
		&result.CreatedAt, &result.UpdatedAt, &result.Current.Revision, &result.Current.CatalogRevision,
		&aliases, &result.Current.ParamSize, &result.Current.ContextWindowSize, &result.Current.Description,
		&capabilities, &reasoning, &loras, &result.Current.QualityScore, &result.Current.Modality, &tags,
		&execution, &pricing,
	); err != nil {
		return routingmanagement.Model{}, err
	}
	result.Current.ID, result.Current.Name = result.ID, result.Name
	for _, field := range []struct {
		payload []byte
		target  any
	}{
		{aliases, &result.Current.Aliases},
		{capabilities, &result.Current.Capabilities},
		{reasoning, &result.Current.Reasoning},
		{loras, &result.Current.LoRAs},
		{tags, &result.Current.Tags},
		{execution, &result.Current.Execution},
		{pricing, &result.Current.Pricing},
	} {
		if err := strictJSON(field.payload, field.target); err != nil {
			return routingmanagement.Model{}, fmt.Errorf("decode routing Model: %w", err)
		}
	}
	return result, nil
}
