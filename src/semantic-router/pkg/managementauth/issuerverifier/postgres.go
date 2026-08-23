package issuerverifier

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"io"
)

type PostgresRepository struct {
	database *sql.DB
}

func NewPostgresRepository(database *sql.DB) (*PostgresRepository, error) {
	if database == nil {
		return nil, ErrUnavailable
	}
	return &PostgresRepository{database: database}, nil
}

func (repository *PostgresRepository) LoadActive(
	ctx context.Context,
	issuerID string,
) (TrustedIssuer, error) {
	if repository == nil || repository.database == nil {
		return TrustedIssuer{}, ErrUnavailable
	}
	var issuer TrustedIssuer
	var audiences, claimMapping, assuranceMapping []byte
	var discoveryURL, jwksURL sql.NullString
	var revision int64
	err := repository.database.QueryRowContext(ctx, `SELECT id::text,issuer,kind,discovery_url,jwks_url,
audiences,claim_mapping,assurance_mapping,revision
FROM trusted_identity_issuers WHERE id=$1 AND status='active'`, issuerID).Scan(
		&issuer.ID, &issuer.Issuer, &issuer.Kind, &discoveryURL, &jwksURL,
		&audiences, &claimMapping, &assuranceMapping, &revision,
	)
	if errors.Is(err, sql.ErrNoRows) {
		return TrustedIssuer{}, ErrDenied
	}
	if err != nil || revision <= 0 {
		return TrustedIssuer{}, fmt.Errorf("%w: load trusted issuer", ErrUnavailable)
	}
	issuer.DiscoveryURL, issuer.JWKSURL, issuer.Revision = discoveryURL.String, jwksURL.String, uint64(revision)
	if err := decodeStrictJSON(audiences, &issuer.Audiences); err != nil {
		return TrustedIssuer{}, fmt.Errorf("%w: decode issuer audiences", ErrUnavailable)
	}
	if err := decodeStrictJSON(claimMapping, &issuer.ClaimMapping); err != nil {
		return TrustedIssuer{}, fmt.Errorf("%w: decode issuer claim mapping", ErrUnavailable)
	}
	if err := decodeStrictJSON(assuranceMapping, &issuer.AssuranceMapping); err != nil {
		return TrustedIssuer{}, fmt.Errorf("%w: decode issuer assurance mapping", ErrUnavailable)
	}
	if issuer.ClaimMapping == nil {
		issuer.ClaimMapping = map[string]string{}
	}
	if issuer.AssuranceMapping == nil {
		issuer.AssuranceMapping = map[string]string{}
	}
	if err := issuer.Validate(); err != nil {
		return TrustedIssuer{}, err
	}
	return issuer, nil
}

func decodeStrictJSON(data []byte, target any) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return err
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return errors.New("JSON contains trailing values")
	}
	return nil
}

var _ Repository = (*PostgresRepository)(nil)
