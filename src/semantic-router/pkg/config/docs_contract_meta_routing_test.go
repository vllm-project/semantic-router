package config

func metaRoutingTutorialOverviewDoc() docNeedles {
	return docNeedles{
		path: repoRel("website", "docs", "tutorials", "meta-routing", "overview.md"),
		needles: []string{
			"`routing.meta`",
			"`observe`",
			"`shadow`",
			"`active`",
			"[Modes](./modes)",
			"[Design](./design)",
			"[Usage](./usage)",
		},
	}
}

func metaRoutingTutorialModesDoc() docNeedles {
	return docNeedles{
		path: repoRel("website", "docs", "tutorials", "meta-routing", "modes.md"),
		needles: []string{
			"`observe` means:",
			"`shadow` means:",
			"`active` means:",
			"## What Counts as Fragile?",
			"## Is This Learning?",
		},
	}
}

func metaRoutingTutorialSidebarEntries() []string {
	return []string{
		"'tutorials/meta-routing/overview'",
		"'tutorials/meta-routing/modes'",
		"'tutorials/meta-routing/design'",
		"'tutorials/meta-routing/problems'",
		"'tutorials/meta-routing/usage'",
	}
}
