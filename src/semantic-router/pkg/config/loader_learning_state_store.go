package config

func rejectUnsupportedRouterLearningStateStoreFields(
	learning map[string]interface{},
) error {
	stateStore := nestedStringMap(learning["state_store"])
	if err := rejectUnknownMapFields(
		"global.router.learning.state_store",
		stateStore,
		[]string{"backend", "ttl_seconds", "timeout_ms", "redis"},
	); err != nil {
		return err
	}
	return rejectUnknownMapFields(
		"global.router.learning.state_store.redis",
		nestedStringMap(stateStore["redis"]),
		[]string{"address", "password", "database", "key_prefix"},
	)
}
