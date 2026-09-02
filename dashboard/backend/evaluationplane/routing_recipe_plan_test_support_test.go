package evaluationplane

func mustFreezeTestRoutingRecipePlan(mixture *ManifestMixture) {
	targetSnapshotDigest, err := routingRecipeTargetSnapshotDigest(*mixture)
	if err != nil {
		panic(err)
	}
	armIDs := make([]string, 0, len(mixture.ModelArms))
	for _, arm := range mixture.ModelArms {
		armIDs = append(armIDs, arm.ID)
	}
	plan, err := canonicalRoutingRecipePlan(RoutingRecipePlan{
		ContractVersion:      RoutingRecipePlanContractVersion,
		TargetSnapshotDigest: targetSnapshotDigest,
		ArmIDs:               armIDs,
		FallbackArmID:        mixture.FallbackArmID,
		Signals:              []RoutingRecipeInputSpec{},
		Projections:          []RoutingRecipeProjectionSpec{},
		TopK:                 routingRecipeTopK(len(armIDs)),
	})
	if err != nil {
		panic(err)
	}
	mixture.RoutingRecipePlan = plan
}
