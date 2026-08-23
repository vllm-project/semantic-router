package postgres

import (
	"database/sql"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
)

func scanRecipeRow(row rowScanner) (routingmanagement.Recipe, error) {
	var result routingmanagement.Recipe
	var distributionID, distributionVersion, sourceRecipeID sql.NullString
	var assetDigest, recipeDigest []byte
	var sourceRevision sql.NullInt64
	var installedAt sql.NullTime
	if err := row.Scan(
		&result.NamespaceID, &result.ID, &result.Name, &result.Description, &result.Status,
		&result.Revision, &result.CreatedAt, &result.UpdatedAt, &result.Current.Revision,
		&result.Current.Description, &result.Current.Document,
		&distributionID, &distributionVersion, &assetDigest, &sourceRecipeID, &sourceRevision,
		&recipeDigest, &installedAt,
	); err != nil {
		return routingmanagement.Recipe{}, err
	}
	result.Current.ID, result.Current.Name = result.ID, result.Name
	result.Description = result.Current.Description
	result.Origin = routingmanagement.RecipeOriginCustom
	if distributionID.Valid {
		result.Origin = routingmanagement.RecipeOriginDistribution
		result.Provenance = &routingmanagement.RecipeProvenance{
			DistributionID: distributionID.String, DistributionVersion: distributionVersion.String,
			AssetDigest: encodedDigest(assetDigest), SourceRecipeID: sourceRecipeID.String,
			SourceRevision: sourceRevision.Int64, RecipeDigest: encodedDigest(recipeDigest),
			InstalledAt: installedAt.Time,
		}
	}
	return result, nil
}
