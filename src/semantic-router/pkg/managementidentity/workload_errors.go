package managementidentity

import "errors"

var (
	ErrInvalidWorkloadRequest       = errors.New("management workload identity request is invalid")
	ErrWorkloadUnavailable          = errors.New("management workload identity is unavailable")
	ErrWorkloadDependency           = errors.New("management workload identity has dependent resources")
	ErrServiceCredentialUnavailable = errors.New("management service credential is unavailable")
	ErrWorkloadSecretExpired        = errors.New("management workload secret result expired")
	ErrMTLSListenerUnavailable      = errors.New("management mTLS listener verification is unavailable")
)
