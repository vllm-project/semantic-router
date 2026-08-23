// Package managementauthorization evaluates Management API permission
// expressions against authoritative, live role and membership facts.
//
// Authentication, database lookup, and resource-graph expansion deliberately
// stay outside this package. Callers must build an EvaluationContext from the
// current Management session and the resources resolved for the operation.
package managementauthorization
