// Package quota defines the exact, store-independent quota domain contract.
//
// Quantities in this package are deliberately represented without floating
// point. Storage adapters may encode them differently, but must preserve the
// same canonical decimal and overflow semantics.
package quota
