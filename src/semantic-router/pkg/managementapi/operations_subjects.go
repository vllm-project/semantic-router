package managementapi

func subjectOperations() []OperationContract {
	var operations []OperationContract

	userRead := Require("user.read", "user")
	userManage := Require("user.manage", "user")
	userCRUD := resourceCRUD("Users", BasePath+"/users", "userId", userRead, userManage, ScopeSubject)
	userCRUD[len(userCRUD)-1].Revision = RevisionCAS
	operations = append(operations, userCRUD...)
	operations = append(operations,
		operation(MethodGET, BasePath+"/users/{userId}/effective-policy", "Users", ScopeCompound,
			RequireAll(userRead, Require("access_policy.read", "user"), Require("rate_policy.read", "user"))),
		operation(MethodGET, BasePath+"/users/{userId}/routing-context", "Routing Context", ScopeCompound,
			RequireAll(userRead, Require("routing_context.read", "user"))),
		operation(MethodPUT, BasePath+"/users/{userId}/routing-context", "Routing Context", ScopeCompound,
			RequireAll(userManage, Require("routing_context.manage", "user")), casRevision()),
		operation(MethodGET, BasePath+"/users/{userId}/quota", "Quota", ScopeCompound,
			RequireAll(userRead, Require("quota.read", "all_returned_bindings"))),
		operation(MethodGET, BasePath+"/users/{userId}/usage", "Usage", ScopeCompound,
			RequireAll(userRead, Require("usage.read", "user"))),
		operation(MethodGET, BasePath+"/users/{userId}/memberships", "Users", ScopeResultSet,
			RequireAll(userRead, RequireWhen("user_membership_row", Require("team.read", "all_returned_resources"))), paginated()),
	)

	teamRead := Require("team.read", "team")
	teamManage := Require("team.manage", "team")
	teamCRUD := resourceCRUD("Teams", BasePath+"/teams", "teamId", teamRead, teamManage, ScopeSubject)
	teamCRUD[1].Permission = RequireAll(teamManage,
		Require("access_policy.manage", "access_policy"),
		Require("rate_policy.manage", "rate_policy"))
	teamCRUD[len(teamCRUD)-1].Revision = RevisionCAS
	operations = append(operations, teamCRUD...)
	operations = append(operations,
		operation(MethodGET, BasePath+"/teams/{teamId}/effective-policy", "Teams", ScopeCompound,
			RequireAll(teamRead, Require("access_policy.read", "team"), Require("rate_policy.read", "team"))),
		operation(MethodGET, BasePath+"/teams/{teamId}/routing-context", "Routing Context", ScopeCompound,
			RequireAll(teamRead, Require("routing_context.read", "team"))),
		operation(MethodPUT, BasePath+"/teams/{teamId}/routing-context", "Routing Context", ScopeCompound,
			RequireAll(teamManage, Require("routing_context.manage", "team")), casRevision()),
		operation(MethodGET, BasePath+"/teams/{teamId}/quota", "Quota", ScopeCompound,
			RequireAll(teamRead, Require("quota.read", "all_returned_bindings"))),
		operation(MethodGET, BasePath+"/teams/{teamId}/usage", "Usage", ScopeCompound,
			RequireAll(teamRead, Require("usage.read", "team"))),
		operation(MethodGET, BasePath+"/teams/{teamId}/members", "Teams", ScopeSubject, teamRead, paginated()),
		operation(MethodPUT, BasePath+"/teams/{teamId}/members/{userId}", "Teams", ScopeSubject,
			Require("membership.manage", "team")),
		operation(MethodPATCH, BasePath+"/teams/{teamId}/members/{userId}", "Teams", ScopeSubject,
			Require("membership.manage", "team")),
		operation(MethodDELETE, BasePath+"/teams/{teamId}/members/{userId}", "Teams", ScopeSubject,
			Require("membership.manage", "team"), casRevision()),
	)

	return operations
}
