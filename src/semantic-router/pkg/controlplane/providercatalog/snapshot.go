package providercatalog

import (
	"bytes"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"io"
)

const snapshotSchema = "provider-catalog.snapshot.v2"

type snapshotEnvelope struct {
	Schema       string       `json:"schema"`
	Revision     string       `json:"revision"`
	Integrations []Definition `json:"integrations"`
}

func (snapshot *Snapshot) MarshalBinary() ([]byte, error) {
	if snapshot == nil || !validCatalogRevision(snapshot.revision) || len(snapshot.integrations) == 0 {
		return nil, fmt.Errorf("provider Catalog snapshot is invalid")
	}
	return json.Marshal(snapshotEnvelope{
		Schema: snapshotSchema, Revision: snapshot.revision,
		Integrations: cloneDefinitions(snapshot.integrations),
	})
}

func RestoreSnapshot(payload []byte, registry *Registry) (*Snapshot, error) {
	if registry == nil {
		return nil, fmt.Errorf("provider integration registry is required")
	}
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.DisallowUnknownFields()
	var envelope snapshotEnvelope
	if err := decoder.Decode(&envelope); err != nil {
		return nil, fmt.Errorf("decode Provider Catalog snapshot: %w", err)
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return nil, fmt.Errorf("provider Catalog snapshot contains trailing data")
	}
	if envelope.Schema != snapshotSchema || !validCatalogRevision(envelope.Revision) {
		return nil, fmt.Errorf("provider Catalog snapshot schema or revision is invalid")
	}
	restored, err := buildSnapshot(envelope.Integrations, registry, true)
	if err != nil {
		return nil, err
	}
	if subtle.ConstantTimeCompare([]byte(restored.Revision()), []byte(envelope.Revision)) != 1 {
		return nil, fmt.Errorf("provider Catalog snapshot revision does not match immutable content")
	}
	return restored, nil
}
