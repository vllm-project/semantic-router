package accesscontrol

type AccessPolicyBinding struct {
	ID          PolicyBindingID
	NamespaceID NamespaceID
	Subject     SubjectRef
	PolicyID    AccessPolicyID
	Status      BindingStatus
	Revision    Revision
}

func (b AccessPolicyBinding) Validate() error {
	var subjectErr, statusErr error
	if err := b.Subject.Validate(); err != nil {
		subjectErr = invalid("subject", err.Error())
	} else if b.Subject.NamespaceID != b.NamespaceID {
		subjectErr = invalid("subject", "must be in the binding namespace")
	}
	if !b.Status.Valid() {
		statusErr = invalid("status", "is not a valid binding status")
	}
	return joinValidation(
		validateRequired("id", string(b.ID)),
		validateRequired("namespace_id", string(b.NamespaceID)),
		subjectErr,
		validateRequired("policy_id", string(b.PolicyID)),
		statusErr,
		validateRevision(b.Revision),
	)
}

type RateBindingMode string

const (
	RateBindingAllocation RateBindingMode = "allocation"
	RateBindingHardCap    RateBindingMode = "hard_cap"
)

func (m RateBindingMode) Valid() bool {
	return m == RateBindingAllocation || m == RateBindingHardCap
}

type RateLimitBinding struct {
	ID               PolicyBindingID
	NamespaceID      NamespaceID
	Subject          SubjectRef
	PolicyID         RateLimitPolicyID
	Mode             RateBindingMode
	QuotaPartitionID QuotaPartitionID
	Status           BindingStatus
	Revision         Revision
}

func (b RateLimitBinding) Validate() error {
	var subjectErr, modeErr, statusErr error
	if err := b.Subject.Validate(); err != nil {
		subjectErr = invalid("subject", err.Error())
	} else if b.Subject.NamespaceID != b.NamespaceID {
		subjectErr = invalid("subject", "must be in the binding namespace")
	}
	if !b.Mode.Valid() {
		modeErr = invalid("mode", "is not allocation or hard_cap")
	}
	if !b.Status.Valid() {
		statusErr = invalid("status", "is not a valid binding status")
	}
	return joinValidation(
		validateRequired("id", string(b.ID)),
		validateRequired("namespace_id", string(b.NamespaceID)),
		subjectErr,
		validateRequired("policy_id", string(b.PolicyID)),
		modeErr,
		validateRequired("quota_partition_id", string(b.QuotaPartitionID)),
		statusErr,
		validateRevision(b.Revision),
	)
}

// CounterID makes the counter-ownership rule explicit: reusing a policy never
// shares state; the binding is the counter identity.
func (b RateLimitBinding) CounterID() PolicyBindingID { return b.ID }

func ValidateAccessBindingReferences(binding AccessPolicyBinding, policy AccessPolicy, subject Subject) error {
	if err := binding.Validate(); err != nil {
		return err
	}
	if policy.ID != binding.PolicyID || policy.NamespaceID != binding.NamespaceID {
		return invalid("policy_id", "must reference a policy in the binding namespace")
	}
	if subject.Ref() != binding.Subject {
		return invalid("subject", "must reference the exact typed subject")
	}
	return nil
}

func ValidateRateBindingReferences(binding RateLimitBinding, policy RateLimitPolicy, subject Subject, namespace Namespace) error {
	if err := binding.Validate(); err != nil {
		return err
	}
	if policy.ID != binding.PolicyID || policy.NamespaceID != binding.NamespaceID {
		return invalid("policy_id", "must reference a policy in the binding namespace")
	}
	if subject.Ref() != binding.Subject {
		return invalid("subject", "must reference the exact typed subject")
	}
	if namespace.ID != binding.NamespaceID || namespace.QuotaPartitionID != binding.QuotaPartitionID {
		return invalid("quota_partition_id", "must equal the namespace canonical partition")
	}
	return nil
}
