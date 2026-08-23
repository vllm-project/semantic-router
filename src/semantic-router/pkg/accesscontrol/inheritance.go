package accesscontrol

type InheritanceLayer string

const (
	InheritanceLayerNone InheritanceLayer = "none"
	InheritanceLayerKey  InheritanceLayer = "key"
	InheritanceLayerUser InheritanceLayer = "user"
	InheritanceLayerTeam InheritanceLayer = "team"
)

type EffectiveAccessBindings struct {
	Source   InheritanceLayer
	Bindings []AccessPolicyBinding
}

// ResolveAccessBindings selects the first layer containing an active binding.
// Policies within that one layer are later evaluated as an allow union with
// explicit deny precedence; lower layers never supplement an explicit layer.
func ResolveAccessBindings(
	keyBindings []AccessPolicyBinding,
	userBindings []AccessPolicyBinding,
	teamBindings []AccessPolicyBinding,
) (EffectiveAccessBindings, error) {
	layers := []struct {
		source InheritanceLayer
		kind   SubjectKind
		items  []AccessPolicyBinding
	}{
		{source: InheritanceLayerKey, kind: SubjectKindAPIKey, items: keyBindings},
		{source: InheritanceLayerUser, kind: SubjectKindUser, items: userBindings},
		{source: InheritanceLayerTeam, kind: SubjectKindTeam, items: teamBindings},
	}
	for _, layer := range layers {
		active, err := activeAccessBindings(layer.kind, layer.items)
		if err != nil {
			return EffectiveAccessBindings{}, err
		}
		if len(active) > 0 {
			return EffectiveAccessBindings{Source: layer.source, Bindings: active}, nil
		}
	}
	return EffectiveAccessBindings{Source: InheritanceLayerNone}, nil
}

func activeAccessBindings(kind SubjectKind, bindings []AccessPolicyBinding) ([]AccessPolicyBinding, error) {
	active := make([]AccessPolicyBinding, 0, len(bindings))
	var subject *SubjectRef
	for _, binding := range bindings {
		if err := binding.Validate(); err != nil {
			return nil, err
		}
		if binding.Subject.Kind != kind {
			return nil, invalid("subject.kind", "does not match its inheritance layer")
		}
		if subject == nil {
			candidate := binding.Subject
			subject = &candidate
		} else if binding.Subject != *subject {
			return nil, invalid("subject", "an inheritance layer must contain bindings for exactly one subject")
		}
		if binding.Status == BindingStatusActive {
			active = append(active, binding)
		}
	}
	return active, nil
}

type ResolvedRateBinding struct {
	Source  InheritanceLayer
	Binding RateLimitBinding
}

type EffectiveRateBindings struct {
	Allocation *ResolvedRateBinding
	HardCaps   []ResolvedRateBinding
}

// ResolveRateBindings selects one allocation from Key, User, then Team, while
// retaining every applicable hard cap from all three layers.
func ResolveRateBindings(
	keyBindings []RateLimitBinding,
	userBindings []RateLimitBinding,
	teamBindings []RateLimitBinding,
) (EffectiveRateBindings, error) {
	result := EffectiveRateBindings{}
	var namespaceID NamespaceID
	var partitionID QuotaPartitionID
	layers := []struct {
		source InheritanceLayer
		kind   SubjectKind
		items  []RateLimitBinding
	}{
		{source: InheritanceLayerKey, kind: SubjectKindAPIKey, items: keyBindings},
		{source: InheritanceLayerUser, kind: SubjectKindUser, items: userBindings},
		{source: InheritanceLayerTeam, kind: SubjectKindTeam, items: teamBindings},
	}

	for _, layer := range layers {
		resolved, err := resolveRateLayer(layer.source, layer.kind, layer.items)
		if err != nil {
			return EffectiveRateBindings{}, err
		}
		if resolved.hasBindings {
			if namespaceID == "" {
				namespaceID, partitionID = resolved.namespaceID, resolved.partitionID
			} else if resolved.namespaceID != namespaceID || resolved.partitionID != partitionID {
				return EffectiveRateBindings{}, invalid("quota_partition_id", "all effective bindings must share one namespace partition")
			}
		}
		result.HardCaps = append(result.HardCaps, resolved.hardCaps...)
		if result.Allocation == nil {
			result.Allocation = resolved.allocation
		}
	}

	return result, nil
}

type rateLayerResolution struct {
	hasBindings bool
	namespaceID NamespaceID
	partitionID QuotaPartitionID
	allocation  *ResolvedRateBinding
	hardCaps    []ResolvedRateBinding
}

func resolveRateLayer(source InheritanceLayer, kind SubjectKind, bindings []RateLimitBinding) (rateLayerResolution, error) {
	result := rateLayerResolution{}
	var subject *SubjectRef
	for _, binding := range bindings {
		if err := binding.Validate(); err != nil {
			return rateLayerResolution{}, err
		}
		if err := validateRateLayerSubject(kind, binding.Subject, &subject); err != nil {
			return rateLayerResolution{}, err
		}
		if !result.hasBindings {
			result.hasBindings = true
			result.namespaceID = binding.NamespaceID
			result.partitionID = binding.QuotaPartitionID
		} else if binding.NamespaceID != result.namespaceID || binding.QuotaPartitionID != result.partitionID {
			return rateLayerResolution{}, invalid("quota_partition_id", "all bindings in a layer must share one namespace partition")
		}
		if binding.Status != BindingStatusActive {
			continue
		}
		resolved := ResolvedRateBinding{Source: source, Binding: binding}
		if binding.Mode == RateBindingHardCap {
			result.hardCaps = append(result.hardCaps, resolved)
			continue
		}
		if result.allocation != nil {
			return rateLayerResolution{}, invalid("allocation", "a subject may have at most one active allocation")
		}
		result.allocation = &resolved
	}
	return result, nil
}

func validateRateLayerSubject(kind SubjectKind, candidate SubjectRef, expected **SubjectRef) error {
	if candidate.Kind != kind {
		return invalid("subject.kind", "does not match its inheritance layer")
	}
	if *expected == nil {
		copy := candidate
		*expected = &copy
		return nil
	}
	if candidate != **expected {
		return invalid("subject", "an inheritance layer must contain bindings for exactly one subject")
	}
	return nil
}
