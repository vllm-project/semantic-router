package managementapi

import (
	"fmt"
	"math"
	"strconv"
	"time"
)

type ProviderCatalogBootstrapRequest struct {
	ExpectedGeneration WholeQuantity `json:"expectedGeneration"`
}

type ProviderCatalogActivateRequest struct {
	Revision           string        `json:"revision"`
	ExpectedGeneration WholeQuantity `json:"expectedGeneration"`
}

type ProviderCatalogPublication struct {
	DesiredRevision string        `json:"desiredRevision"`
	ActiveRevision  string        `json:"activeRevision,omitempty"`
	Generation      WholeQuantity `json:"generation"`
	UpdatedAt       time.Time     `json:"updatedAt"`
}

func (request ProviderCatalogBootstrapRequest) Generation() (uint64, error) {
	return providerCatalogGeneration(request.ExpectedGeneration)
}

func (request ProviderCatalogActivateRequest) Generation() (uint64, error) {
	return providerCatalogGeneration(request.ExpectedGeneration)
}

func providerCatalogGeneration(value WholeQuantity) (uint64, error) {
	if _, err := ParseWholeQuantity(string(value)); err != nil {
		return 0, err
	}
	parsed, err := strconv.ParseUint(string(value), 10, 64)
	if err != nil || parsed == 0 || parsed > math.MaxInt64 {
		return 0, fmt.Errorf("provider catalog generation must fit a positive PostgreSQL BIGINT")
	}
	return parsed, nil
}
