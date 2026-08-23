package postgres

var (
	_ NamespaceRepository          = (*Store)(nil)
	_ UserRepository               = (*Store)(nil)
	_ TeamRepository               = (*Store)(nil)
	_ MembershipRepository         = (*Store)(nil)
	_ APIKeyRepository             = (*Store)(nil)
	_ PolicyRepository             = (*Store)(nil)
	_ BindingRepository            = (*Store)(nil)
	_ ProviderCredentialRepository = (*Store)(nil)
)
