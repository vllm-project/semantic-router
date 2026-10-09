"""SYN1 taxonomy: sectors and domains, task archetypes, conditions, styles and languages.

Everything here is data. The planner (``plan.py``) samples seeds from it deterministically, and the
prompts (``prompts.py``) quote the descriptions verbatim, so editing a description changes the
generation prompt hash recorded in every row.

Public Decision Index benchmarks must not be imitated (report §1.2). Domains therefore avoid the
suites' own task framings: no banking-app intent catalogues (BANKING77), virtual-assistant intent
catalogues (CLINC150), smart-home device commands (Home appliances), function-calling or tool
selection (BFCL, API-Bank, ToolRet, When2Call), phishing email verdicts (PhishNChips), sarcasm
detection (iSarcasmEval), stance on social-media targets (VAST), product-search relevance (ESCI),
math word problems (GSM8K), code-output prediction (CRUXEval), humour, music, colour or chess.
"""

from __future__ import annotations

SECTORS: dict[str, tuple[str, ...]] = {
    "business_operations": (
        "procurement and vendor onboarding",
        "facilities management for office buildings",
        "corporate event planning",
        "warehouse inventory control",
        "manufacturing quality assurance",
        "small business bookkeeping",
        "franchise operations compliance",
        "corporate travel desk",
        "meeting room and desk booking",
        "internal communications approvals",
        "B2B customer onboarding",
        "sales operations and CRM data hygiene",
        "partnership and reseller management",
        "subscription box fulfilment",
        "print shop order intake",
    ),
    "law_and_compliance": (
        "residential tenancy disputes",
        "employment law compliance",
        "personal data access requests",
        "commercial contract renewals",
        "trademark filing review",
        "small claims court intake",
        "immigration paperwork preparation",
        "anti-money-laundering customer reviews",
        "export control screening",
        "consumer protection complaints",
        "board meeting governance",
        "legal aid intake",
        "notary appointments",
        "litigation hold notices",
        "regulatory reporting deadlines",
    ),
    "medicine_and_triage": (
        "emergency department triage",
        "telehealth symptom intake",
        "pharmacy refill requests",
        "vaccination clinic scheduling",
        "dental practice front desk",
        "physiotherapy referrals",
        "mental health intake screening",
        "maternity ward admissions",
        "pediatric clinic phone triage",
        "hospital bed management",
        "clinical laboratory result routing",
        "home health nursing visits",
        "medical billing and coding queries",
        "allergy clinic management",
        "occupational health checks",
    ),
    "veterinary_and_animals": (
        "veterinary clinic triage",
        "animal shelter adoptions",
        "livestock veterinary records",
        "pet boarding kennels",
        "wildlife rescue hotline",
        "dog grooming salon bookings",
        "equine stable management",
        "pet microchip registry",
        "aquarium fish health",
        "zoo animal care logs",
        "pet food recalls",
        "service animal certification",
    ),
    "finance": (
        "small business loan underwriting",
        "card payment dispute adjudication",
        "household budgeting apps",
        "individual tax filing preparation",
        "corporate expense audits",
        "accounts payable invoice matching",
        "treasury cash forecasting",
        "microfinance lending",
        "payroll processing",
        "pension plan administration",
        "investment suitability reviews",
        "mortgage servicing",
        "accounts receivable collections",
        "crypto exchange compliance",
        "credit union membership",
    ),
    "human_resources": (
        "recruitment screening",
        "interview scheduling",
        "parental and sick leave requests",
        "performance review calibration",
        "workplace accommodation requests",
        "benefits enrollment",
        "new hire onboarding checklists",
        "employee relations complaints",
        "hospital shift swaps",
        "overtime approval",
        "mandatory training compliance",
        "remote work eligibility",
        "relocation assistance",
        "contractor timesheet approval",
        "volunteer program coordination",
    ),
    "logistics_and_supply_chain": (
        "freight forwarding",
        "last-mile delivery exceptions",
        "cold chain temperature monitoring",
        "customs brokerage",
        "warehouse slotting",
        "truck fleet maintenance",
        "container port operations",
        "rail freight scheduling",
        "air cargo acceptance",
        "reverse logistics returns processing",
        "bicycle courier dispatch",
        "hazardous materials shipping",
        "supplier lead-time management",
        "shipping container tracking",
        "rural postal services",
    ),
    "education": (
        "university admissions",
        "course prerequisite checks",
        "student financial aid eligibility",
        "school bus routing",
        "special education services",
        "exam accommodations",
        "academic library services",
        "academic integrity review",
        "teacher timetabling",
        "student conduct hearings",
        "scholarship committees",
        "continuing education enrollment",
        "early childhood centre enrollment",
        "language school placement",
        "school field trip approvals",
    ),
    "science_and_research": (
        "laboratory safety incident reports",
        "research grant compliance",
        "journal manuscript triage",
        "field ecology surveys",
        "telescope observation scheduling",
        "chemical inventory management",
        "research data management plans",
        "research ethics board review",
        "biobank sample requests",
        "weather station operations",
        "geology field logs",
        "materials testing laboratory",
        "citizen science submission moderation",
        "core facility equipment booking",
        "archaeological excavation records",
    ),
    "engineering": (
        "construction site safety",
        "building code plan review",
        "electrical grid maintenance",
        "HVAC service calls",
        "water treatment plant operations",
        "aircraft maintenance logs",
        "independent auto repair shops",
        "bridge structural inspection",
        "manufacturing line changeovers",
        "industrial robot cell faults",
        "telecom tower maintenance",
        "underground mining operations",
        "solar farm monitoring",
        "elevator maintenance contracts",
        "railway signalling maintenance",
    ),
    "software_and_devops": (
        "on-call incident triage",
        "continuous integration failures",
        "code review feedback",
        "dependency vulnerability alerts",
        "cloud cost anomalies",
        "database migration planning",
        "feature flag rollouts",
        "internal access request approvals",
        "application log anomalies",
        "release note categorisation",
        "bug report deduplication",
        "service level objective breaches",
        "container orchestration events",
        "mobile app store review responses",
        "open-source project issue triage",
    ),
    "security": (
        "security operations centre alert triage",
        "physical security incident reports",
        "badge access anomalies",
        "vulnerability disclosure intake",
        "insider risk reviews",
        "help desk identity verification",
        "data loss prevention alerts",
        "firewall change requests",
        "penetration test findings",
        "security awareness training records",
        "third-party vendor risk assessments",
        "cloud storage misconfiguration reports",
        "AI agent guardrail reviews",
        "event venue crowd security",
        "museum artwork security",
    ),
    "customer_support": (
        "mobile network operator support",
        "home internet outage reports",
        "SaaS product support",
        "online order problems",
        "airline customer service",
        "electricity and gas billing support",
        "hotel guest services",
        "car rental counter disputes",
        "gym membership support",
        "video streaming service support",
        "food delivery app support",
        "consumer electronics warranty",
        "furniture delivery scheduling",
        "appliance repair booking",
        "online marketplace seller support",
    ),
    "government_services": (
        "municipal building permits",
        "social benefits eligibility",
        "vehicle registration offices",
        "passport applications",
        "public housing waitlists",
        "tax authority correspondence",
        "city non-emergency service requests",
        "jury duty administration",
        "voter registration",
        "public library services",
        "parks and recreation bookings",
        "household waste collection",
        "food truck licensing",
        "disaster relief assistance",
        "public records requests",
    ),
    "travel_and_hospitality": (
        "hotel revenue management",
        "guided tour operators",
        "cruise line operations",
        "travel insurance claims",
        "entry visa requirements",
        "airline crew scheduling",
        "restaurant reservations",
        "vacation rental hosting",
        "ski resort operations",
        "theme park operations",
        "travel agency itineraries",
        "airport ground handling",
        "conference venue management",
        "hostel management",
        "national park campground reservations",
    ),
    "retail_and_ecommerce": (
        "store returns and refunds policy",
        "retail pricing and promotions",
        "customer loyalty programmes",
        "grocery store operations",
        "fashion retail sizing and exchanges",
        "marketplace listing compliance",
        "gift card issues",
        "consumer electronics retail",
        "independent bookstores",
        "pet supply stores",
        "hardware store special orders",
        "flash sale operations",
        "secondhand and resale shops",
        "jewellery repair counters",
        "garden centre plant care",
    ),
    "real_estate_and_property": (
        "rental application screening",
        "apartment maintenance requests",
        "homeowners association rules",
        "commercial office leasing",
        "property valuation reviews",
        "home inspection reports",
        "short-term rental compliance",
        "building amenity bookings",
        "property tax appeals",
        "coworking space memberships",
        "student housing allocation",
        "retirement community admissions",
        "new-build construction warranty claims",
        "self-storage facilities",
        "parking permit allocation",
    ),
    "insurance": (
        "auto insurance claims",
        "home insurance claims",
        "health insurance prior authorisation",
        "life insurance underwriting",
        "pet insurance claims",
        "crop insurance",
        "workers' compensation",
        "general liability claims",
        "claims fraud screening",
        "policy renewals",
        "flood insurance",
        "cyber insurance",
        "dental insurance claims",
        "event cancellation insurance",
        "marine cargo insurance",
    ),
    "agriculture_and_food": (
        "crop disease reports",
        "dairy herd health",
        "irrigation scheduling",
        "farm equipment rental",
        "food safety inspections",
        "organic certification",
        "grain elevator operations",
        "commercial fisheries management",
        "beekeeping associations",
        "greenhouse operations",
        "vineyard management",
        "restaurant kitchen operations",
        "food bank distribution",
        "agricultural subsidy applications",
        "seed bank requests",
    ),
    "sports_and_recreation": (
        "amateur league scheduling",
        "athlete eligibility",
        "referee match reports",
        "return-to-play after injury",
        "fitness class bookings",
        "marathon registration",
        "outdoor equipment rental",
        "esports tournament administration",
        "sailing club operations",
        "golf course operations",
        "youth sports safeguarding",
        "sports ticketing",
        "stadium operations",
        "martial arts grading",
        "climbing gym safety",
    ),
    "media_and_publishing": (
        "newsroom corrections desk",
        "hobby forum rules on spam, duplicates and off-topic posts",
        "podcast production",
        "film production scheduling",
        "music licensing for video",
        "book manuscript submissions",
        "streaming catalogue metadata",
        "advertising standards review",
        "brand social media management",
        "video game quality assurance",
        "stock photography licensing",
        "theatre box office",
        "museum collections management",
        "local radio broadcast scheduling",
        "influencer sponsorship disclosure",
    ),
    "everyday_life": (
        "family calendar conflicts",
        "household bill splitting",
        "meal planning with dietary restrictions",
        "childcare arrangements",
        "neighbourhood pet sitting",
        "home repair contractor selection",
        "moving house logistics",
        "neighbour disputes",
        "recipe ingredient substitutions",
        "personal fitness goals",
        "allotment gardening",
        "wedding planning",
        "used car purchases",
        "clothing care and laundry",
        "caring for an elderly parent",
    ),
    "transportation_and_mobility": (
        "public transit service alerts",
        "ride-hailing trip disputes",
        "bike-share operations",
        "parking enforcement appeals",
        "traffic incident reports",
        "driver licensing",
        "electric vehicle charging networks",
        "taxi dispatch",
        "general aviation flight planning",
        "harbour navigation notices",
        "toll road billing",
        "car-sharing memberships",
        "truck driver hours-of-service compliance",
        "paratransit booking",
        "ferry operations",
    ),
    "energy_and_utilities": (
        "power outage management",
        "meter reading anomalies",
        "gas leak reports",
        "water main break response",
        "demand response programmes",
        "rooftop solar permits",
        "utility payment plans",
        "grid interconnection requests",
        "home energy audits",
        "district heating networks",
        "battery storage operations",
        "heating oil delivery",
        "utility disconnection protections",
        "wind farm maintenance",
        "telecom fibre rollout",
    ),
    "nonprofit_and_community": (
        "grant-making foundations",
        "donor relations",
        "community centre bookings",
        "homeless shelter intake",
        "humanitarian aid logistics",
        "faith community event planning",
        "mutual aid networks",
        "youth clubs",
        "refugee resettlement services",
        "environmental campaigns",
        "food cooperatives",
        "arts council grants",
        "neighbourhood associations",
        "blood donation drives",
        "senior centre programmes",
    ),
    "manufacturing_and_industry": (
        "supplier quality audits",
        "product recall management",
        "production planning",
        "maintenance work orders",
        "ISO audit findings",
        "packaging line operations",
        "textile mills",
        "pharmaceutical manufacturing deviations",
        "semiconductor fab operations",
        "food processing plants",
        "chemical plant permits",
        "3D printing service bureaus",
        "furniture workshops",
        "automotive parts suppliers",
        "glass and ceramics kilns",
    ),
    "environment_and_sustainability": (
        "corporate emissions reporting",
        "municipal recycling contamination",
        "environmental impact assessments",
        "wildlife rehabilitation centres",
        "carbon offset verification",
        "river water quality monitoring",
        "wildfire risk management",
        "industrial pollution complaints",
        "sustainable procurement",
        "urban tree management",
        "noise complaint investigations",
        "beach clean-up coordination",
        "invasive species reporting",
        "e-waste collection",
        "community composting",
    ),
    "arts_culture_and_design": (
        "art gallery submissions",
        "painting conservation",
        "architecture design reviews",
        "graphic design briefs",
        "interior design consultations",
        "private music lessons",
        "dance studio scheduling",
        "handmade crafts marketplace",
        "professional translation services",
        "literary magazine submissions",
        "heritage site management",
        "costume rental",
        "portrait photography studios",
        "public art commissions",
        "film festival programming",
    ),
}

DOMAINS: tuple[tuple[str, str], ...] = tuple(
    (sector, domain) for sector, names in SECTORS.items() for domain in names
)


# Each archetype: a description quoted to the generator, the question types it supports, choice
# option-count range, whether the options are an ordered scale (never shuffled), its weight in the
# plan, and phenomena the planner samples as the family's focus.
ARCHETYPES: dict[str, dict] = {
    "criteria_classification": {
        "description": (
            "Classify the case into exactly one category. Every option has a descriptive criterion "
            "(what qualifies, and where neighbouring categories end), not just a name, and the "
            "decisive evidence in the state maps to exactly one criterion."
        ),
        "types": ("choice",),
        "options": (3, 100),
        "ordinal": False,
        "weight": 1.3,
        "phenomena": (
            "near-boundary cases between neighbouring categories",
            "primary versus secondary topic",
            "misleading keywords that point to the wrong category",
            "multiple issues with one primary",
            "category defined by an exclusion clause",
        ),
    },
    "policy_exceptions": {
        "description": (
            "Apply an explicitly supplied policy (rules plus exceptions and their own exceptions) to "
            "a request. The policy is fictional and complete; the answer follows only from it and the "
            "facts of the case."
        ),
        "types": ("choice", "noul"),
        "options": (3, 6),
        "ordinal": False,
        "weight": 1.3,
        "phenomena": (
            "an exception that overrides the general rule",
            "an exception to the exception",
            "a condition that is mentioned but not met",
            "a deadline or quantity limit",
            "a superseded policy version",
            "facts that look relevant but are not decisive",
        ),
    },
    "eligibility": {
        "description": (
            "Decide whether a person, organisation or item meets every stated requirement for a "
            "programme, service, discount, permit or benefit. Requirements are stated in the "
            "question or the state; one failed requirement makes the case ineligible."
        ),
        "types": ("noul", "choice"),
        "options": (2, 6),
        "ordinal": False,
        "weight": 1.2,
        "phenomena": (
            "all criteria met except one",
            "a waiver that substitutes for a criterion",
            "an age, income or date limit near the boundary",
            "documentation still pending",
            "requirements with 'either/or' structure",
        ),
    },
    "routing": {
        "description": (
            "Route a request, message or case to the single team, queue, form or desk that should "
            "handle its primary need. Each option describes its responsibility; neighbouring "
            "options overlap in vocabulary but differ in responsibility."
        ),
        "types": ("choice",),
        "options": (4, 100),
        "ordinal": False,
        "weight": 1.2,
        "phenomena": (
            "quoted or reported request versus the actual request",
            "a correction later in the message",
            "background context versus current need",
            "two needs with one explicitly primary",
            "indirect request",
            "a team named in the text that is not the right one",
        ),
    },
    "priority_urgency": {
        "description": (
            "Assign a priority, severity or urgency level using explicit level definitions (impact, "
            "time sensitivity, safety). Levels form an ordered scale."
        ),
        "types": ("choice", "noul"),
        "options": (3, 6),
        "ordinal": True,
        "weight": 1.0,
        "phenomena": (
            "alarming language with low actual impact",
            "calm language with high actual impact",
            "a workaround that lowers severity",
            "a deadline that raises urgency",
            "safety risk overriding other factors",
        ),
    },
    "evidence_verification": {
        "description": (
            "Check a claim, summary or reported fact against supplied records or source excerpts: "
            "fully supported, contradicted, or not established. Partial support is not support."
        ),
        "types": ("choice", "noul"),
        "options": (3, 5),
        "ordinal": False,
        "weight": 1.2,
        "phenomena": (
            "a date or number that differs slightly",
            "a different entity with a similar name",
            "a plan or forecast reported as completed fact",
            "support split across two sources",
            "a claim broader than the evidence",
            "an outdated record superseded later",
        ),
    },
    "answerability": {
        "description": (
            "Decide whether a specific question can be answered from the supplied material alone, or "
            "pick the supported answer value with an explicit 'not stated' option. Missing "
            "information must be intentional and unambiguous."
        ),
        "types": ("noul", "choice"),
        "options": (3, 8),
        "ordinal": False,
        "weight": 1.2,
        "phenomena": (
            "information about a similar but different entity",
            "an answer that needs one unstated assumption",
            "a value given only as a range",
            "the fact appears only as a hypothetical",
            "the fact is stated for a different time period",
        ),
    },
    "none_of_the_above": {
        "description": (
            "Choose the matching option, where the list includes an explicit 'none of these' option "
            "that is correct exactly when no specific option fits the case."
        ),
        "types": ("choice",),
        "options": (3, 12),
        "ordinal": False,
        "weight": 1.0,
        "phenomena": (
            "a case close to one option but failing its key condition",
            "a case that clearly fits one option",
            "a case belonging to a category that is not listed",
            "two partial fits, neither complete",
        ),
    },
    "long_state_lookup": {
        "description": (
            "The state is a long structured record (roster, ledger, log, inventory, schedule, case "
            "file) with many similar entries. The question asks which entry has a property, or "
            "what a specific entry's attribute is, among candidate options."
        ),
        "types": ("choice", "noul"),
        "options": (3, 60),
        "ordinal": False,
        "weight": 1.0,
        "phenomena": (
            "similar entity names",
            "an entry updated later in the record",
            "a filter on two attributes",
            "a value stated in different units",
            "the queried entry appears only once among distractors",
        ),
    },
    "temporal_numeric": {
        "description": (
            "Decide using dates, durations, deadlines, amounts, counts, rates or unit conversions "
            "stated in the state, against an explicit threshold or tier table. Arithmetic is short "
            "and exact; the decisive computation crosses (or does not cross) the threshold."
        ),
        "types": ("noul", "choice"),
        "options": (2, 8),
        "ordinal": False,
        "weight": 1.2,
        "phenomena": (
            "a deadline counted in business days",
            "a limit that is inclusive versus exclusive",
            "a unit conversion",
            "a cumulative total across several entries",
            "a time-zone or date-boundary case",
            "a percentage change",
        ),
    },
    "comparison_choice": {
        "description": (
            "Choose the candidate (vendor, plan, route, slot, applicant, product) that best satisfies "
            "explicit constraints and a stated objective. Exactly one candidate satisfies every hard "
            "constraint and wins on the objective."
        ),
        "types": ("choice",),
        "options": (2, 12),
        "ordinal": False,
        "weight": 1.0,
        "phenomena": (
            "the cheapest option fails a hard constraint",
            "a tie broken by a secondary criterion",
            "a constraint stated as an exclusion",
            "candidates described in different formats",
        ),
    },
    "preference_judgement": {
        "description": (
            "Compare two or three candidate work products written by people in the domain (notices, "
            "schedules, plans, quotes, replies drafted by staff) against an explicit rubric and pick "
            "the one that satisfies it best. Differences are concrete (a missing required item, a "
            "wrong figure, a policy violation, wrong tone), not stylistic taste. Never AI or chatbot "
            "outputs, and no tie option."
        ),
        "types": ("choice",),
        "options": (2, 3),
        "ordinal": False,
        "weight": 0.9,
        "phenomena": (
            "the longer draft violates a rule",
            "a polite draft with a factual error",
            "one draft omits a required element",
            "both drafts flawed but one acceptable",
        ),
    },
    "tone_style": {
        "description": (
            "Judge the tone, register or style of a message against explicit definitions (for example "
            "calm / frustrated / hostile, or whether it meets a stated style guide). Variants keep the "
            "same content and change only tone or style."
        ),
        "types": ("choice", "noul"),
        "options": (3, 6),
        "ordinal": False,
        "weight": 0.9,
        "phenomena": (
            "polite wording with an underlying complaint",
            "blunt but not hostile wording",
            "formal register that hides urgency",
            "style-guide rule violated in one sentence",
        ),
    },
    "injection_resistance": {
        "description": (
            "The state is data from an untrusted source (an email, web page, document, form or log) "
            "and may contain text addressed to an automated system that tries to change the decision "
            "or asks it to do something. The question asks either a normal decision about the "
            "content, whose answer must follow the real evidence and ignore embedded instructions, "
            "or whether the content contains such an embedded instruction."
        ),
        "types": ("choice", "noul"),
        "options": (2, 6),
        "ordinal": False,
        "weight": 0.9,
        "phenomena": (
            "an instruction disguised as a system note",
            "an instruction in a footer or hidden field",
            "a benign instruction meant for a human colleague",
            "an instruction that claims special authority",
        ),
    },
    "multi_question": {
        "description": (
            "One rich state (case file, report, transcript or record) with several different questions "
            "about it, mixing choice and yes/no questions that test different details."
        ),
        "types": ("multi",),
        "options": (2, 10),
        "ordinal": False,
        "weight": 0.9,
        "phenomena": (
            "questions about different parties in the same case",
            "one question whose answer is 'not stated'",
            "questions that depend on the order of events",
        ),
    },
    "knowledge_application": {
        "description": (
            "A practical decision in the domain that needs widely accepted domain knowledge that is "
            "not in the state (for example which material, regulation type, drug class, species, "
            "unit or standard practice applies). The knowledge must be stable, uncontroversial and "
            "verifiable; never exam-style trivia copied from a test."
        ),
        "types": ("choice", "noul"),
        "options": (3, 10),
        "ordinal": False,
        "weight": 1.3,
        "phenomena": (
            "a frequently confused procedure",
            "two similar-sounding terms",
            "a safety-relevant fact",
            "a rule of thumb with a known exception",
            "a classification by scientific property",
        ),
    },
    "pii_sensitivity": {
        "description": (
            "Decide whether text contains a specific kind of personal or confidential data, or which "
            "kind it contains, using explicit definitions. All data is invented."
        ),
        "types": ("noul", "choice"),
        "options": (3, 8),
        "ordinal": False,
        "weight": 0.7,
        "phenomena": (
            "a number that looks like an ID but is not",
            "a partially masked identifier",
            "personal data about a third party",
            "a public business contact versus personal data",
        ),
    },
    "agent_trace": {
        "description": (
            "The state is a log of an automated agent or workflow run (steps, observations, results). "
            "The question asks whether the run achieved its goal, which step failed, whether a rule "
            "was violated, or what the final state is."
        ),
        "types": ("choice", "noul"),
        "options": (2, 10),
        "ordinal": False,
        "weight": 0.8,
        "phenomena": (
            "a step reported as successful but contradicted later",
            "a retry that succeeded",
            "the goal partially achieved",
            "a constraint violated silently",
        ),
    },
}

CONDITIONS: dict[str, str] = {
    "clean": "Natural, direct wording; 40-150 words of state.",
    "noisy": (
        "Realistic noise: typos, shorthand, missing punctuation, inconsistent casing, transcription "
        "or OCR slips, but the decisive facts stay understandable; 40-150 words of state."
    ),
    "negated": (
        "Meaningful negation, exceptions, contrast or retractions that matter to the answer "
        "(for example 'not', 'except', 'unless', 'no longer', 'rather than'); 40-150 words of state."
    ),
    "long": (
        "300-600 words of state with plausible, related but non-decisive context (history, other "
        "items, side remarks, boilerplate); the decisive evidence stays clear and appears once, "
        "anywhere in the text."
    ),
    "shifted_prior": (
        "Ordinary, natural cases of the requested answers; 40-200 words of state. Never mention how "
        "often an answer occurs."
    ),
}

STYLES: tuple[str, ...] = (
    "email",
    "chat transcript",
    "support ticket",
    "web form submission",
    "internal memo",
    "case notes",
    "table or spreadsheet excerpt",
    "log excerpt",
    "letter",
    "voicemail transcript",
    "meeting minutes",
    "checklist",
    "JSON record",
    "bulleted notes",
    "incident report",
    "application form",
    "text message thread",
    "policy excerpt plus case description",
)

# Weighted languages for state text; questions and options stay in the same language as the state.
LANGUAGES: tuple[tuple[str, float], ...] = (
    ("English", 0.82),
    ("German", 0.03),
    ("French", 0.03),
    ("Spanish", 0.03),
    ("Portuguese", 0.02),
    ("Italian", 0.02),
    ("Chinese", 0.02),
    ("Japanese", 0.015),
    ("Dutch", 0.015),
)

# Choice option-count buckets (inclusive ranges) and weights; clipped to each archetype's range.
OPTION_BUCKETS: tuple[tuple[int, int, float], ...] = (
    (2, 2, 0.08),
    (3, 4, 0.30),
    (5, 8, 0.30),
    (9, 16, 0.17),
    (17, 30, 0.10),
    (31, 60, 0.035),
    (61, 100, 0.015),
)

# Variants per condition and the label pattern for each. "pair": two variants with different
# answers. "skew": four variants, three sharing a dominant answer.
CONDITION_LAYOUT: dict[str, tuple[int, str]] = {
    "clean": (2, "pair"),
    "noisy": (2, "pair"),
    "negated": (2, "pair"),
    "long": (2, "pair"),
    "shifted_prior": (4, "skew"),
}

NOUL_SHARE = 0.42
JSON_STATE_SHARE = 0.35
