package main

// Corrections avoid repeating the old value, so a stale hit always means the
// superseded memory itself was injected.
var builtinScenario = []session{
	{
		Day: 0,
		Probes: []probe{
			{Phase: phaseNoMemory, Query: "What is my dog's name?"},
			{Phase: phaseNoMemory, Query: "Which city do I live in?"},
		},
		Turns: []turn{
			{User: "My dog is a beagle named Biscuit.", Assistant: "Biscuit the beagle, noted."},
			{User: "I live in Boston, near the Charles River.", Assistant: "Got it, you live in Boston."},
			{User: "I work as a nurse at the children's hospital.", Assistant: "Thanks, I'll remember you're a nurse."},
			{User: "My budget for the Japan trip is $4,000.", Assistant: "I'll plan the Japan trip around $4,000."},
		},
	},
	{
		Day: 2,
		Probes: []probe{
			{Phase: phaseFirstSeen, Query: "What is my dog's name?", Expect: []string{"biscuit"}},
			{Phase: phaseFirstSeen, Query: "Which city do I live in?", Expect: []string{"boston"}},
			{Phase: phaseFirstSeen, Query: "What is my budget for the Japan trip?", Expect: []string{"4,000"}},
			{Phase: phaseNoMemory, Query: "When is my dentist appointment?"},
		},
		Turns: []turn{
			{User: "I'm allergic to peanuts, so keep them out of any recipe.", Assistant: "Understood, no peanuts."},
			{User: "I drive a 2019 Honda Civic.", Assistant: "A 2019 Civic, got it."},
			{User: "I take my coffee black with no sugar.", Assistant: "Black coffee, no sugar."},
		},
	},
	{
		Day: 9,
		Probes: []probe{
			{Phase: phaseRecurring, Query: "What breed is my dog?", Expect: []string{"beagle"}},
			{Phase: phaseRecurring, Query: "Can you suggest a snack recipe for me?", Expect: []string{"peanut"}},
			{Phase: phaseRecurring, Query: "How do I like my coffee?", Expect: []string{"black"}},
			{Phase: phaseRecurring, Query: "What car do I drive?", Expect: []string{"civic"}},
			{Phase: phaseNoMemory, Query: "What is my sister's name?"},
		},
		Turns: []turn{
			{User: "Biscuit turned three today, so I bought my dog a new toy.", Assistant: "Happy birthday to Biscuit!"},
			{User: "I'm learning Rust for a side project.", Assistant: "Rust is a good pick for that."},
		},
	},
	{
		Day: 21,
		Probes: []probe{
			{Phase: phaseRecurring, Query: "What is my dog's name?", Expect: []string{"biscuit"}},
			{Phase: phaseRecurring, Query: "Which programming language am I learning?", Expect: []string{"rust"}},
		},
		Turns: []turn{
			{User: "I just moved to Denver, and I live there now.", Assistant: "Welcome to Denver!"},
			{User: "I changed jobs and now work as a paramedic.", Assistant: "Congratulations on the paramedic job!"},
			{User: "I raised my budget for the Japan trip to $6,000.", Assistant: "Updated, the Japan trip budget is $6,000."},
			{User: "My sister Maya is visiting me next month.", Assistant: "Say hi to Maya for me."},
		},
	},
	{
		Day: 30,
		Probes: []probe{
			{Phase: phaseStale, Query: "Which city do I live in now?", Expect: []string{"denver"}, Stale: []string{"boston"}},
			{Phase: phaseStale, Query: "What do I do for work?", Expect: []string{"paramedic"}, Stale: []string{"nurse"}},
			{Phase: phaseStale, Query: "What is my budget for the Japan trip now?", Expect: []string{"6,000"}, Stale: []string{"4,000"}},
			{Phase: phaseFirstSeen, Query: "What is my sister's name?", Expect: []string{"maya"}},
			{Phase: phaseRecurring, Query: "What breed is my dog?", Expect: []string{"beagle"}},
		},
	},
}
