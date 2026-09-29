"""Declarative domain packs for F2 ``hs1_policy_packet`` (records/hs1-prereg-2026-09-29.md §1).

A pack supplies the lexicon and domain-specific templates that the generic
engine in ``f2_policy`` renders into a policy document: the organisation and
document names, the annex scope attribute, the case attributes that
exceptions and definitions test, the two case dates, one outcome variable per
interface (Choice: a value with units; Noul: a numeric limit compared with a
requested quantity; Score: an ordinal level), question templates and at least
twenty distractor sections. Distractor, static and extra-definition texts
never mention an outcome variable, so they cannot change a label.

Template slots: {org} {office} {member} {members} {Member} {Members} {person}
{Person} {case} {cases} {Case} {Cases} {title} and, in distractors only, the
filler numbers {wd} (working days) {dd} (days) {mm} (months) {hh} (hours)
{nn} (a small count) {pct} {ext} {yy} (year) and the names {name1} {name2}.
Outcome templates use {val}; conditions use {t} (threshold) and {v} (case
value); Noul quantities use {q}; case sentences use {name}.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class Quantity:
    """A numeric outcome variable. ``unit`` is ``money`` or a format with ``{n}``."""

    key: str
    heading: str
    noun: str
    unit: str
    ladder: tuple[int, ...]
    option: str
    frags: tuple[str, ...]
    base: tuple[str, ...]
    ask: tuple[str, ...] = ()
    ask_label: str = ""


@dataclasses.dataclass(frozen=True)
class Levels:
    """An ordinal outcome: level ``i`` is stated by ``frags[i]`` and described by ``criteria[i]``."""

    key: str
    heading: str
    noun: str
    frags: tuple[str, ...]
    needles: tuple[str, ...]
    criteria: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class Condition:
    """A case attribute an exception can test: ``min`` (v >= t), ``max`` (v <= t) or ``is`` (v == t)."""

    key: str
    kind: str
    text: str
    term: str
    term_def: str
    term_use: str
    case: tuple[str, ...]
    label: str
    lo: int = 0
    hi: int = 0
    thresholds: tuple = ()
    choices: tuple[str, ...] = ()
    unit: str = "{n}"


@dataclasses.dataclass(frozen=True)
class Scope:
    """The attribute annexes are keyed on (a region, site, channel, zone, range ...)."""

    label: str
    values: tuple[str, ...]
    annex_title: str
    annex_scope: str
    case: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class Noise:
    label: str
    values: tuple[str, ...]
    case: str


@dataclasses.dataclass(frozen=True)
class DatePair:
    """Two case dates; the second follows the first by ``lag`` days."""

    a_key: str
    a_label: str
    a_phrase: str
    a_case: str
    b_key: str
    b_label: str
    b_phrase: str
    b_case: str
    lag: tuple[int, int]


@dataclasses.dataclass(frozen=True)
class Domain:
    key: str
    titles: tuple[str, ...]
    orgs: tuple[str, ...]
    member: str
    members: str
    person: str
    claimant_label: str
    case: str
    cases: str
    offices: tuple[str, ...]
    currency: tuple[str, ...]
    purpose: tuple[str, ...]
    scope_text: tuple[str, ...]
    scope: Scope
    conditions: tuple[Condition, ...]
    noise: tuple[Noise, ...]
    dates: DatePair
    choice: Quantity
    noul: Quantity
    score: Levels
    questions: dict
    extra_defs: tuple[tuple[str, str], ...]
    static: tuple[str, ...]
    distractors: tuple[tuple[str, str], ...]


# ------------------------------------------------------------ shared texts

GENERIC_DISTRACTORS: tuple[tuple[str, str], ...] = (
    (
        "Document control",
        "This document is owned by the {office}. It is reviewed at least once every {mm} months and whenever a change in the "
        "law or in the structure of {org} makes a review necessary. Printed copies are uncontrolled.",
    ),
    (
        "Equality and accessibility",
        "This policy is applied consistently to everyone it covers. Anyone who needs the document in another format, such "
        "as large print or an audio version, can ask the {office}, which will normally provide it within {wd} working days.",
    ),
    (
        "Complaints about the handling of {a_case}",
        "Anyone who is unhappy with the way {a_case} has been handled may ask for the handling to be looked at again by a "
        "senior member of the {office}. The request should be made in writing within {dd} days and should explain what "
        "went wrong. A written reply is sent within {wd} working days.",
    ),
    (
        "Records and retention",
        "The {office} keeps a record of every {case} and of the documents supplied with it. Records are held for {mm} "
        "months after the matter is closed and are then securely destroyed, unless an audit or a dispute requires them to "
        "be kept longer.",
    ),
    (
        "Data protection",
        "Personal information supplied with {a_case} is used only to process it, to meet audit and legal obligations and "
        "to produce anonymous statistics. It is not shared outside {org} except where the law requires it. Questions about "
        "personal data can be sent to the data protection officer, {name1}.",
    ),
    (
        "Related documents",
        "This document should be read together with the {org} Code of Conduct, the Data Protection Notice and the "
        "Anti-Fraud Statement. Those documents are available from the {office} and on the intranet.",
    ),
    (
        "Misuse and false information",
        "Knowingly giving false or misleading information in {a_case} is treated as a serious matter and may lead to "
        "disciplinary action or to the recovery of any amount paid. Honest mistakes should be reported to the {office} as "
        "soon as they are noticed so that they can be corrected.",
    ),
    (
        "Service standards",
        "The {office} aims to acknowledge every {case} within {wd} working days of receipt and to deal with most of them "
        "within {dd} days. Performance against these targets is published every quarter.",
    ),
    (
        "Consultation on this edition",
        "This edition was prepared after consultation with staff representatives and with a sample of the people who "
        "use it most. Comments received during the consultation are summarised in a separate note kept by the {office}.",
    ),
    (
        "Translations",
        "Translations of this document are provided for convenience. If a translation differs from the English text, "
        "the English text should be used.",
    ),
    (
        "Monitoring and reporting",
        "The {office} reports the number of {cases} received, the time taken to process them and the main reasons for "
        "delay to the management board twice a year. The report does not identify individuals.",
    ),
    (
        "Confidentiality",
        "Staff who handle {cases} must keep the information in them confidential and use it only for the purpose for "
        "which it was supplied. Paper documents are stored in locked cabinets and electronic files in restricted folders.",
    ),
    (
        "Electronic forms and signatures",
        "Where this policy refers to a form, the online version may be used. An electronic signature or confirmation "
        "given through the {org} portal has the same standing as a handwritten signature.",
    ),
    (
        "Suggestions for improvement",
        "Suggestions for improving this policy or the forms that support it are welcome. They can be sent to the {office} "
        "at any time and are considered at the next scheduled review.",
    ),
    (
        "Business continuity",
        "If the {office} is unable to work normally, for example because of a system outage, {cases} are logged manually "
        "and processed in date order once normal working resumes. Nobody is disadvantaged by a delay of this kind.",
    ),
    (
        "Environmental note",
        "{org} prefers electronic copies of documents. Please do not print this policy unless a paper copy is essential, "
        "and recycle printed copies when a new edition is issued.",
    ),
    (
        "Contact points",
        "The {office} can be reached on extension {ext} between 9 a.m. and 4 p.m., Monday to Friday. Written enquiries "
        "should be addressed to {name1}, who coordinates the team, or in their absence to {name2}.",
    ),
    (
        "Status of guidance notes",
        "The {office} may publish guidance notes explaining how to complete forms or where to send documents. Guidance "
        "notes describe procedure only; they do not add to or change any entitlement or limit in this policy.",
    ),
)

GENERIC_DEFS: tuple[tuple[str, str], ...] = (
    ("Working day", "any day from Monday to Friday other than a public holiday"),
    (
        "Portal",
        "the online system provided by {org} for submitting and tracking {cases}",
    ),
    ("Month", "a calendar month"),
    (
        "Written",
        "on paper or in electronic form, including email and messages sent through the portal",
    ),
)

LAYOUTS = ("numbered", "handbook", "consolidated", "memo")


def _t(*items: str) -> tuple[str, ...]:
    return tuple(items)


# ------------------------------------------------------------ travel_expense

TRAVEL = Domain(
    key="travel_expense",
    titles=_t(
        "Travel and Expense Policy",
        "Business Travel and Expenses Policy",
        "Staff Travel Reimbursement Policy",
        "Policy on Travel and Subsistence Expenses",
    ),
    orgs=_t(
        "Halberd Engineering Ltd",
        "Corvane Logistics",
        "Meridian Clinical Services",
        "Ashcombe Partners LLP",
        "Tessellate Software",
        "Brightwater Housing Trust",
    ),
    member="employee",
    members="employees",
    person="the traveller",
    claimant_label="Traveller",
    case="claim",
    cases="claims",
    offices=_t(
        "Finance Office", "Travel Desk", "Expenses Unit", "Accounts Payable team"
    ),
    currency=_t("usd", "eur", "gbp"),
    purpose=_t(
        "This policy sets out what {members} of {org} may claim when they travel on the organisation's business and "
        "how their {cases} are checked and paid.",
        "The purpose of this policy is to make sure that business travel is necessary, safe and reasonably priced, and "
        "that {members} are repaid fairly and promptly for costs they meet on behalf of {org}.",
        "{org} expects business travel to be planned sensibly and paid for at fair prices. This policy explains which "
        "travel costs are repaid and on what terms.",
    ),
    scope_text=_t(
        "This policy applies to all {members} of {org}, and to contractors engaged through approved agencies, when they "
        "travel on authorised business. Relocation costs are dealt with under a separate procedure.",
        "The policy covers every business trip made by {members} of {org}, whether in the home country or abroad. It "
        "does not cover travel by board members, which is governed by the board's own rules.",
    ),
    scope=Scope(
        label="Home office",
        values=_t("Leeds", "Porto", "Gdansk", "Rotterdam", "Lyon", "Cork"),
        annex_title="Staff based at the {v} office",
        annex_scope="{members} whose home office is {v}",
        case=_t("{name} is based at the {v} office.", "{name}'s home office is {v}."),
    ),
    conditions=(
        Condition(
            "nights",
            "min",
            "the trip lasts at least {t} consecutive nights",
            "Extended Trip",
            "a business trip of at least {t} consecutive nights away from the traveller's home office",
            "the trip is an Extended Trip",
            _t("The trip lasted {v}.", "{name} was away for {v}."),
            "Nights away",
            lo=1,
            hi=21,
            thresholds=(4, 5, 6, 7, 8, 10),
            unit="{n} nights",
        ),
        Condition(
            "grade",
            "min",
            "the traveller is on salary grade {t} or above",
            "Senior Traveller",
            "an employee whose salary grade is {t} or above",
            "the traveller is a Senior Traveller",
            _t(
                "{name} is on salary grade {v}.",
            ),
            "Salary grade",
            lo=1,
            hi=12,
            thresholds=(6, 7, 8, 9),
        ),
        Condition(
            "purpose",
            "is",
            "the main purpose of the trip is {t}",
            "Designated Trip",
            "a trip whose main purpose is {t}",
            "the trip is a Designated Trip",
            _t(
                "The main purpose of the trip was {v}.",
            ),
            "Purpose of trip",
            choices=_t(
                "a client meeting",
                "a trade conference",
                "internal training",
                "a site inspection",
                "a recruitment fair",
            ),
        ),
        Condition(
            "distance",
            "min",
            "the destination is at least {t} from the traveller's home office",
            "Long-distance Trip",
            "a trip to a destination at least {t} from the traveller's home office",
            "the trip is a Long-distance Trip",
            _t(
                "The destination is {v} from {name}'s home office.",
            ),
            "Distance from home office",
            lo=20,
            hi=900,
            thresholds=(150, 200, 250, 300, 400),
            unit="{n} km",
        ),
    ),
    noise=(
        Noise(
            "Mode of travel",
            _t("rail", "air", "hire car", "coach", "ferry and rail"),
            "{name} travelled by {v}.",
        ),
        Noise(
            "Cost centre",
            _t(
                "CC-{n}",
            ),
            "The costs are charged to cost centre {v}.",
        ),
        Noise(
            "Project code",
            _t("PRJ-{n}", "WB-{n}"),
            "The trip was booked against project {v}.",
        ),
    ),
    dates=DatePair(
        "incurred",
        "Expense incurred",
        "the date on which the expense was incurred",
        "The expense was incurred on {d}.",
        "submitted",
        "Claim submitted",
        "the date on which the claim was submitted",
        "The claim was submitted on {d}.",
        (4, 60),
    ),
    choice=Quantity(
        key="meal_allowance",
        heading="Meals",
        noun="daily meal allowance",
        unit="money",
        ladder=tuple(range(35, 100, 5)),
        option="{val} per day",
        frags=_t(
            "the daily meal allowance is {val}",
            "travellers may claim a daily meal allowance of up to {val}",
            "a daily meal allowance of {val} applies",
        ),
        base=_t(
            "{Members} travelling on business may claim a daily meal allowance of up to {val} for each full day "
            "away. The allowance covers breakfast, lunch and dinner.",
            "The daily meal allowance is {val}. It is paid for each full day away from the home office and is "
            "meant to cover all meals on that day.",
        ),
    ),
    noul=Quantity(
        key="hotel_cap",
        heading="Accommodation",
        noun="maximum nightly hotel rate",
        unit="money",
        ladder=tuple(range(110, 340, 10)),
        option="{val} per night",
        frags=_t(
            "the maximum nightly hotel rate is {val}",
            "hotel accommodation may be claimed up to {val} per night",
            "the nightly hotel rate may not exceed {val}",
        ),
        base=_t(
            "Hotel accommodation is repaid at the actual cost up to a maximum nightly hotel rate of {val}, "
            "excluding breakfast.",
            "The maximum nightly hotel rate is {val}. Any amount above it is not repaid, whatever the reason for "
            "choosing the hotel.",
        ),
        ask=_t(
            "The hotel charged {q} per night.",
            "{name} paid {q} per night for the hotel.",
        ),
        ask_label="Nightly hotel rate claimed",
    ),
    score=Levels(
        key="approval",
        heading="Approval of claims",
        noun="approval the claim needs",
        frags=_t(
            "claims need no approval beyond the traveller's own certification",
            "claims must be approved by the traveller's line manager",
            "claims must be approved by the head of department",
            "claims must be approved by the finance director",
        ),
        needles=_t(
            "own certification",
            "line manager",
            "head of department",
            "finance director",
        ),
        criteria=_t(
            "No approval is needed beyond the traveller's own certification",
            "The traveller's line manager must approve the claim",
            "The head of department must approve the claim",
            "The finance director must approve the claim",
        ),
    ),
    questions={
        "choice": _t(
            "Under the policy, what daily meal allowance applies to {name}'s claim?",
            "Which daily meal allowance should the {office} apply when it assesses this claim?",
            "Taking every relevant provision into account, what is the daily meal allowance for this claim?",
        ),
        "noul": _t(
            "Is the nightly hotel rate that {name} claimed within the maximum nightly rate that applies to this "
            "claim?",
            "Can the {office} repay the hotel cost in full, that is, is the nightly rate claimed no higher than "
            "the maximum that applies?",
        ),
        "score": _t(
            "What approval does {name}'s claim need under the policy?",
            "Before the {office} pays this claim, whose approval is required?",
        ),
    },
    extra_defs=(
        (
            "Home office",
            "the office named in the employee's contract as their normal place of work",
        ),
        (
            "Business trip",
            "travel away from the home office that is undertaken for the purposes of {org}",
        ),
        (
            "Approved agency",
            "a travel agency with which {org} holds a current framework contract",
        ),
        (
            "Itinerary",
            "the planned sequence of journeys and overnight stays for a trip",
        ),
    ),
    static=_t(
        "Mileage for the use of a private car is repaid at the rate published each year by the {office}.",
        "Rail travel is booked in standard class unless the journey lasts more than four hours.",
        "Receipts must be attached to every claim, except for items covered by a flat-rate allowance.",
        "Taxi fares are repaid where public transport is not available or is unsafe at the time of travel.",
    ),
    distractors=(
        (
            "Booking travel",
            "Rail and air journeys should be booked through the approved agency at least {wd} working "
            "days before departure where possible. Tickets bought directly from a carrier are repaid only if the agency "
            "could not offer a comparable fare.",
        ),
        (
            "Travel insurance",
            "{org} holds a group travel insurance policy that covers medical emergencies, lost luggage "
            "and cancelled flights. {Members} should carry the assistance card and call the emergency number before "
            "arranging treatment abroad.",
        ),
        (
            "Hire cars",
            "Hire cars are booked in the smallest category suitable for the journey and the number of "
            "passengers. The car must be refuelled before it is returned; refuelling charges added by the hire company are "
            "not repaid.",
        ),
        (
            "Private vehicles",
            "Anyone who uses their own car for business must hold insurance that covers business use. "
            "Parking fines and penalties for traffic offences are the driver's own responsibility.",
        ),
        (
            "Receipts and evidence",
            "Keep original receipts or clear photographs of them. A card statement on its own is "
            "not accepted as evidence, because it does not show what was bought.",
        ),
        (
            "Foreign currency",
            "Costs paid in another currency are converted at the rate shown on the card statement or, "
            "for cash payments, at the published rate on the day of purchase. Bank charges for cash withdrawals abroad "
            "are repaid.",
        ),
        (
            "Corporate cards",
            "Corporate cards may be used only for business costs. A personal item charged to the card "
            "by mistake must be repaid within {dd} days of the statement date.",
        ),
        (
            "Loyalty schemes",
            "{Members} may keep loyalty points earned on business travel, provided that the choice of "
            "airline, hotel or hire company was not influenced by the scheme.",
        ),
        (
            "Combining business and personal travel",
            "Where a trip combines business days and personal days, only the "
            "costs that would have arisen on the business part alone are repaid.",
        ),
        (
            "Visas and vaccinations",
            "The cost of visas, and of vaccinations recommended for the destination, is repaid "
            "on production of a receipt. Allow at least {dd} days for visa applications.",
        ),
        (
            "Personal safety",
            "Before travelling to a destination on the security watch list, the itinerary must be "
            "registered with the security team, and the traveller must check in with the team on arrival.",
        ),
        (
            "Sustainable travel",
            "Rail is preferred to air for journeys that take less than {hh} hours by train. Teams "
            "are encouraged to replace routine visits with video meetings where the purpose of the visit allows it.",
        ),
        (
            "Gratuities",
            "Reasonable gratuities for taxis and restaurant service are repaid in line with local custom. "
            "Alcohol is not repaid except at client hospitality events agreed in advance.",
        ),
        (
            "Cancelled trips",
            "If a trip is cancelled for business reasons, non-refundable costs are met by {org}. If it "
            "is cancelled for personal reasons, the traveller may be asked to meet them.",
        ),
        (
            "Submitting a claim",
            "{Cases} are submitted through the portal. Each line must show the date, the supplier, "
            "the amount and a short description of the business purpose, and the supporting receipts must be attached.",
        ),
        (
            "Checks and audit",
            "The {office} checks a sample of {cases} every month. Anyone whose claim is selected will "
            "be asked to provide the original receipts within {wd} working days.",
        ),
        (
            "Lost receipts",
            "If a receipt is lost, the traveller should complete a missing-receipt declaration giving the "
            "date, the supplier and the amount. Repeated use of declarations is reported to the {office}.",
        ),
        (
            "Frequently asked questions",
            "Q: Can I book a taxi to the airport? A: Yes, where public transport is "
            "impractical or the journey starts before 6 a.m. Q: Can I claim for a replacement phone charger? A: No, "
            "personal items are not repaid.",
        ),
        (
            "Contacts for travel",
            "Questions about bookings go to the approved agency's helpline. Questions about {cases} "
            "go to the {office} on extension {ext}. Out of hours, the agency's emergency line is available.",
        ),
        (
            "Tax treatment",
            "Some benefits connected with travel may be taxable in the employee's country of residence. "
            "The payroll team reports them where required; individuals remain responsible for their own tax returns.",
        ),
        (
            "Luggage",
            "Charges for one checked bag are repaid on flights where a bag is needed for the business purpose. "
            "Charges for extra legroom or priority boarding are not repaid.",
        ),
        (
            "Advances",
            "A cash advance may be requested for a long trip where card payment is not practical. The advance "
            "is set against the first claim submitted after the trip.",
        ),
    ),
)


# ------------------------------------------------------------ returns_refund

RETURNS = Domain(
    key="returns_refund",
    titles=_t(
        "Returns and Refunds Policy",
        "Customer Returns Policy",
        "Refund and Exchange Policy",
        "Policy on Returned Goods",
    ),
    orgs=_t(
        "Larkspur Home Goods",
        "Veloce Cycles",
        "Bramble & Finch Booksellers",
        "Northlight Electronics",
        "Copperleaf Outdoor",
        "Harbourside Kitchenware",
    ),
    member="customer",
    members="customers",
    person="the customer",
    claimant_label="Customer",
    case="return",
    cases="returns",
    offices=_t("Customer Service team", "Returns Desk", "Customer Care Centre"),
    currency=_t("usd", "eur", "gbp"),
    purpose=_t(
        "This policy explains how {members} can return goods bought from {org}, what they receive in return and "
        "which charges apply.",
        "{org} wants every purchase to be the right one. When it is not, this policy sets out the terms on which goods "
        "may be returned and money refunded.",
        "This document sets out the returns terms offered by {org} in addition to the customer's legal rights, which "
        "are not affected.",
    ),
    scope_text=_t(
        "This policy applies to goods bought from {org} by {members} for their own use, through any of the sales "
        "channels listed in this policy. It does not apply to goods bought for resale.",
        "The policy covers new goods sold by {org}. Gift cards, services and goods sold as part of a clearance auction "
        "are outside its scope.",
    ),
    scope=Scope(
        label="Sales channel",
        values=_t(
            "online shop",
            "flagship store",
            "outlet store network",
            "partner marketplace",
            "catalogue service",
        ),
        annex_title="Purchases made through the {v}",
        annex_scope="{members} who bought the goods through the {v}",
        case=_t(
            "{name} bought the item through the {v}.",
            "The purchase was made through the {v}.",
        ),
    ),
    conditions=(
        Condition(
            "loyalty",
            "min",
            "the customer has been a loyalty-club member for at least {t}",
            "Established Member",
            "a loyalty-club member whose membership has lasted at least {t}",
            "the customer is an Established Member",
            _t(
                "{name} has been a loyalty-club member for {v}.",
            ),
            "Loyalty-club membership",
            lo=0,
            hi=15,
            thresholds=(2, 3, 4, 5),
            unit="{n} years",
        ),
        Condition(
            "order_value",
            "min",
            "the order value was at least {t}",
            "Major Order",
            "an order whose total value was at least {t}",
            "the purchase was a Major Order",
            _t(
                "The total value of the order was {v}.",
            ),
            "Order value",
            lo=20,
            hi=2400,
            thresholds=(250, 400, 500, 750, 1000),
            unit="money",
        ),
        Condition(
            "category",
            "is",
            "the item is {t}",
            "Special Category Item",
            "an item that is {t}",
            "the item is a Special Category Item",
            _t(
                "The item being returned is {v}.",
            ),
            "Item",
            choices=_t(
                "a large appliance",
                "a mattress",
                "a made-to-measure item",
                "a software licence",
                "a personal-care product",
            ),
        ),
        Condition(
            "packaging",
            "is",
            "the item is returned {t}",
            "Resaleable Condition",
            "the condition of an item returned {t}",
            "the item is in Resaleable Condition",
            _t(
                "The item was returned {v}.",
            ),
            "Condition on return",
            choices=_t(
                "unopened in its original packaging",
                "opened but unused",
                "used and undamaged",
                "with its packaging missing",
            ),
        ),
    ),
    noise=(
        Noise("Order number", _t("ORD-{n}", "WEB-{n}"), "The order number is {v}."),
        Noise(
            "Payment method",
            _t("credit card", "debit card", "gift card", "bank transfer"),
            "The order was paid for by {v}.",
        ),
        Noise(
            "Reason given",
            _t(
                "wrong size",
                "changed mind",
                "arrived later than expected",
                "duplicate order",
                "colour not as expected",
            ),
            "The reason given for the return was: {v}.",
        ),
    ),
    dates=DatePair(
        "delivered",
        "Delivered",
        "the date on which the goods were delivered",
        "The goods were delivered on {d}.",
        "requested",
        "Return requested",
        "the date on which the return was requested",
        "The return was requested on {d}.",
        (3, 70),
    ),
    choice=Quantity(
        key="return_window",
        heading="Return period",
        noun="return period",
        unit="{n} days",
        ladder=(14, 20, 28, 30, 35, 40, 45, 60, 75, 90),
        option="{val}",
        frags=_t(
            "the return period is {val}",
            "goods may be returned within {val}",
            "a return period of {val} applies",
        ),
        base=_t(
            "Goods may be returned within {val} of delivery for any reason, provided that they are complete.",
            "The return period is {val}. It runs from the day after delivery.",
        ),
    ),
    noul=Quantity(
        key="no_receipt_cap",
        heading="Returns without proof of purchase",
        noun="limit on refunds without proof of purchase",
        unit="money",
        ladder=tuple(range(20, 210, 10)),
        option="{val}",
        frags=_t(
            "refunds without proof of purchase are limited to {val} per return",
            "the limit on refunds without proof of purchase is {val}",
            "a return without proof of purchase may be refunded up to {val}",
        ),
        base=_t(
            "Where the customer cannot show proof of purchase, a refund may be given up to a limit of {val} per "
            "return, at the current selling price.",
            "The limit on refunds without proof of purchase is {val} per return. Larger amounts are not refunded "
            "without proof.",
        ),
        ask=_t(
            "{name} has no proof of purchase and asks for a refund of {q}.",
            "Without a receipt, the refund asked for is {q}.",
        ),
        ask_label="Refund asked for without proof of purchase",
    ),
    score=Levels(
        key="remedy",
        heading="Remedies",
        noun="remedy offered",
        frags=_t(
            "returned goods are not accepted and no refund or exchange is given",
            "the customer may exchange the goods, but no refund is given",
            "the customer receives a credit note for the full price",
            "the customer receives a refund to the original payment method",
        ),
        needles=_t(
            "no refund or exchange",
            "exchange the goods",
            "credit note",
            "original payment method",
        ),
        criteria=_t(
            "No refund or exchange is given",
            "An exchange only, with no refund",
            "A credit note for the full price",
            "A refund to the original payment method",
        ),
    ),
    questions={
        "choice": _t(
            "What return period applies to {name}'s purchase under the policy?",
            "Which return period should the {office} apply to this return?",
        ),
        "noul": _t(
            "Is the refund that {name} asks for within the limit on refunds without proof of purchase that "
            "applies?",
            "Can the {office} give the refund asked for without proof of purchase, that is, is it within the "
            "applicable limit?",
        ),
        "score": _t(
            "What remedy is {name} entitled to under the policy?",
            "Under the policy, which remedy should the {office} offer for this return?",
        ),
    },
    extra_defs=(
        (
            "Proof of purchase",
            "a till receipt, an order confirmation or a card statement showing the purchase",
        ),
        (
            "Complete",
            "returned with all parts, accessories and manuals supplied with the goods",
        ),
        (
            "Selling price",
            "the price at which the goods are offered by {org} on the day of the return",
        ),
        (
            "Faulty goods",
            "goods that do not work as described; they are dealt with under the statutory rules",
        ),
    ),
    static=_t(
        "Faulty goods are dealt with under the customer's statutory rights, whatever the time since delivery.",
        "Returns can be made in any store or by post using the prepaid label supplied with online orders.",
        "Refunds are processed within ten days of the goods being received back.",
    ),
    distractors=(
        (
            "How to return goods",
            "Pack the goods securely, include the returns slip and take the parcel to any "
            "collection point listed on the portal. Keep the proof of postage until the refund has been processed.",
        ),
        (
            "Collection of large items",
            "For goods too large to post, the {office} arranges a collection within {wd} "
            "working days. Someone must be at the address to hand the goods over.",
        ),
        (
            "Gift purchases",
            "A person who received goods as a gift may return them with the gift receipt. Any refund is "
            "paid to the original purchaser unless the purchaser agrees otherwise.",
        ),
        (
            "Damaged in transit",
            "If goods arrive damaged, please photograph the packaging and the goods and tell the "
            "{office} within {wd} working days so that a claim can be made against the carrier.",
        ),
        (
            "Missing parts",
            "Where a part is missing from a delivery, the {office} sends the part free of charge. The "
            "rest of the order does not need to be returned.",
        ),
        (
            "Price changes",
            "If the price of an item falls after purchase, the difference is not paid back, except during "
            "a published price-match promotion.",
        ),
        (
            "Hygiene",
            "For hygiene reasons, pierced jewellery and swimwear with the hygiene seal removed cannot be "
            "resold. Staff will explain the position when goods of this kind are brought back.",
        ),
        (
            "Checking returned goods",
            "Returned goods are checked on arrival at the returns centre. The check normally "
            "takes {wd} working days, and the customer is emailed when it is complete.",
        ),
        (
            "Refunds to gift cards",
            "Where an order was paid for with a gift card, the value is restored to the same "
            "card or, if the card has expired, to a new card of the same value.",
        ),
        (
            "Marketplace sellers",
            "Goods sold by independent sellers on the marketplace are covered by the seller's own "
            "terms. The {office} can help customers contact the seller.",
        ),
        (
            "Store opening hours",
            "Returns desks are open during normal store hours. On public holidays some desks close "
            "early; the portal lists the hours for each store.",
        ),
        (
            "Questions and answers",
            "Q: Do I need the original box? A: It helps, but goods can be returned in other "
            "secure packaging. Q: Can someone else return goods for me? A: Yes, if they bring the order number.",
        ),
        (
            "Customer accounts",
            "Customers with an online account can follow each return on the portal, print "
            "replacement labels and see the date on which a refund was issued.",
        ),
        (
            "Fraud prevention",
            "{org} keeps a record of returns in order to detect abuse. Customers who return goods "
            "unusually often may be asked to explain the pattern.",
        ),
        (
            "Personal data on devices",
            "Before returning a phone, tablet or computer, please remove all personal data "
            "and sign out of every account. {org} is not responsible for data left on returned devices.",
        ),
        (
            "Returns by post from abroad",
            "Customers outside the country pay the cost of postage for returns unless the "
            "goods were sent in error.",
        ),
        (
            "Bulky furniture",
            "Furniture that has been assembled should be dismantled before collection where possible. "
            "Assembly instructions can be downloaded from the portal.",
        ),
        (
            "Contacting the {office}",
            "The {office} can be reached by live chat on the portal, by email or on extension "
            "{ext}. Please have the order number ready.",
        ),
        (
            "Environmental commitment",
            "Returned goods that cannot be resold as new are repaired, sold as refurbished or "
            "recycled. Very few returned items go to landfill.",
        ),
        (
            "Promotions",
            "Where goods were bought as part of a multi-buy offer, returning one item may change the price "
            "of the others. The adjustment is shown in the returns confirmation.",
        ),
        (
            "Loyalty points",
            "Loyalty points earned on returned goods are removed from the member's balance when the "
            "return is processed.",
        ),
    ),
)


# ------------------------------------------------------------ leave_policy

LEAVE = Domain(
    key="leave_policy",
    titles=_t(
        "Leave Policy",
        "Annual and Special Leave Policy",
        "Staff Leave and Absence Policy",
        "Policy on Paid and Special Leave",
    ),
    orgs=_t(
        "Fenwick County Council",
        "Orrery Research Institute",
        "Stellan Medical Group",
        "Kestrel Freight",
        "Juniper Schools Trust",
        "Aldersgate Housing Association",
    ),
    member="employee",
    members="employees",
    person="the employee",
    claimant_label="Employee",
    case="leave request",
    cases="leave requests",
    offices=_t("People Team", "HR Service Centre", "Human Resources office"),
    currency=_t("usd", "eur", "gbp"),
    purpose=_t(
        "This policy explains the leave that {members} of {org} may take, how {cases} are made and the limits that "
        "apply to each kind of leave.",
        "{org} recognises that time away from work is essential. This policy sets out entitlements to annual leave and "
        "to paid special leave and explains how {cases} are handled.",
    ),
    scope_text=_t(
        "This policy applies to every {member} of {org}, whatever their length of service. Agency workers are covered "
        "by their agency's terms and are outside its scope.",
        "The policy covers all {members} on the {org} payroll. Sickness absence and parental leave are dealt with in "
        "separate policies.",
    ),
    scope=Scope(
        label="Place of work",
        values=_t(
            "Harbour Street site",
            "Riverside campus",
            "North depot",
            "City Hall annex",
            "Westfield clinic",
            "Quayside laboratory",
        ),
        annex_title="Staff at the {v}",
        annex_scope="{members} whose contractual place of work is the {v}",
        case=_t(
            "{name} works at the {v}.", "{name}'s contractual place of work is the {v}."
        ),
    ),
    conditions=(
        Condition(
            "service",
            "min",
            "the employee has at least {t} of continuous service",
            "Long-serving Employee",
            "an employee with at least {t} of continuous service",
            "the employee is a Long-serving Employee",
            _t(
                "{name} has {v} of continuous service.",
            ),
            "Continuous service",
            lo=0,
            hi=25,
            thresholds=(3, 5, 7, 10),
            unit="{n} years",
        ),
        Condition(
            "contract",
            "is",
            "the employee works under {t}",
            "Eligible Contract",
            "employment under {t}",
            "the employee holds an Eligible Contract",
            _t(
                "{name} is employed under {v}.",
            ),
            "Contract type",
            choices=_t(
                "a permanent contract",
                "a fixed-term contract",
                "a zero-hours contract",
                "an apprenticeship agreement",
            ),
        ),
        Condition(
            "hours",
            "min",
            "the employee is contracted for at least {t} a week",
            "Full-time Employee",
            "an employee contracted for at least {t} a week",
            "the employee is a Full-time Employee",
            _t(
                "{name} is contracted for {v} a week.",
            ),
            "Contracted hours",
            lo=8,
            hi=40,
            thresholds=(30, 32, 35, 37),
            unit="{n} hours",
        ),
        Condition(
            "reason",
            "is",
            "the leave is needed for {t}",
            "Qualifying Reason",
            "{t}",
            "the leave is needed for a Qualifying Reason",
            _t(
                "The leave is needed for {v}.",
            ),
            "Reason for leave",
            choices=_t(
                "caring for a dependant",
                "a bereavement",
                "jury service",
                "a house move",
                "a medical appointment",
            ),
        ),
    ),
    noise=(
        Noise(
            "Department",
            _t("Finance", "Facilities", "Planning", "Customer Services", "Transport"),
            "{name} works in the {v} department.",
        ),
        Noise(
            "Employee number", _t("E{n}", "S-{n}"), "{name}'s employee number is {v}."
        ),
    ),
    dates=DatePair(
        "submitted",
        "Request submitted",
        "the date on which the leave request is submitted",
        "The request was submitted on {d}.",
        "starts",
        "First day of leave",
        "the first day of the leave",
        "The leave is due to start on {d}.",
        (7, 70),
    ),
    choice=Quantity(
        key="special_leave",
        heading="Paid special leave",
        noun="paid special leave for one occasion",
        unit="{n} days",
        ladder=(1, 2, 3, 4, 5, 6, 7, 8, 10, 12),
        option="{val}",
        frags=_t(
            "paid special leave of up to {val} may be granted for one occasion",
            "the allowance of paid special leave for one occasion is {val}",
            "up to {val} of paid special leave may be granted for each occasion",
        ),
        base=_t(
            "A manager may grant paid special leave of up to {val} for one occasion where an employee needs time "
            "off for an urgent personal matter.",
            "The allowance of paid special leave for one occasion is {val}. Further time off is taken as annual "
            "leave or as unpaid leave.",
        ),
    ),
    noul=Quantity(
        key="leave_block",
        heading="Length of annual leave blocks",
        noun="longest block of annual leave",
        unit="{n} working days",
        ladder=(10, 12, 15, 18, 20, 22, 25, 30),
        option="{val}",
        frags=_t(
            "annual leave may be taken in blocks of up to {val}",
            "the longest block of annual leave is {val}",
            "a single block of annual leave may not exceed {val}",
        ),
        base=_t(
            "Annual leave may be taken in blocks of up to {val} at a time, so that teams can plan cover.",
            "The longest block of annual leave that may be taken at one time is {val}.",
        ),
        ask=_t(
            "{name} has asked for a block of {q} of annual leave.",
            "The block of annual leave requested is {q}.",
        ),
        ask_label="Annual leave block requested",
    ),
    score=Levels(
        key="approver",
        heading="Approval of leave",
        noun="approval the request needs",
        frags=_t(
            "requests of this kind may be approved by the team leader",
            "requests of this kind must be approved by the service manager",
            "requests of this kind must be approved by the head of service",
            "requests of this kind must be approved by the director of people",
        ),
        needles=_t(
            "team leader", "service manager", "head of service", "director of people"
        ),
        criteria=_t(
            "The team leader may approve the request",
            "The service manager must approve the request",
            "The head of service must approve the request",
            "The director of people must approve the request",
        ),
    ),
    questions={
        "choice": _t(
            "How much paid special leave may {name} be granted for this occasion under the policy?",
            "What is the most paid special leave the {office} can grant for this request?",
        ),
        "noul": _t(
            "Is the block of annual leave that {name} requested within the longest block the policy allows in "
            "this case?",
            "Can the request be granted as asked, that is, is the requested block no longer than the maximum "
            "that applies?",
        ),
        "score": _t(
            "Who must approve {name}'s leave request under the policy?",
            "What level of approval does this leave request need?",
        ),
    },
    extra_defs=(
        ("Leave year", "the period from 1 January to 31 December"),
        (
            "Continuous service",
            "service with {org} without a break of more than one week",
        ),
        (
            "Dependant",
            "a spouse, partner, child, parent or a person who relies on the employee for care",
        ),
        (
            "Public holiday",
            "a day declared a public holiday in the country where the employee works",
        ),
    ),
    static=_t(
        "Annual leave is booked through the portal and is not taken until it has been confirmed.",
        "Leave not taken by the end of the leave year is lost unless the carry-over rules allow otherwise.",
        "Employees who work part time receive annual leave in proportion to their contracted hours.",
    ),
    distractors=(
        (
            "Booking leave",
            "Leave is booked through the portal. The request shows the dates, the kind of leave and any "
            "note for the approver. Employees receive an automatic message when the request has been decided.",
        ),
        (
            "Public holidays",
            "Public holidays are in addition to annual leave. Employees who must work on a public "
            "holiday receive a day off in lieu, to be taken within {mm} months.",
        ),
        (
            "Sickness during annual leave",
            "An employee who falls ill during annual leave may treat the days of illness "
            "as sick leave if they report the illness on the first day and provide a medical certificate.",
        ),
        (
            "Unpaid leave",
            "Unpaid leave may be agreed for personal reasons that are not covered elsewhere. It does not "
            "break continuous service, but it reduces the annual leave that accrues in the leave year.",
        ),
        (
            "Volunteering days",
            "Employees may take up to {nn} volunteering days each year with a registered charity. "
            "Volunteering days are recorded separately and do not count as annual leave.",
        ),
        (
            "Religious observance",
            "{org} will try to accommodate requests for time off for religious festivals, "
            "normally from annual leave or through a temporary change of working pattern.",
        ),
        (
            "Adverse weather",
            "When severe weather prevents travel, employees should contact their team as early as "
            "possible. Where the work allows, they may work from home for the day.",
        ),
        (
            "Leaving {org}",
            "On leaving, any annual leave accrued but not taken is paid with the final salary. Leave "
            "taken in excess of the accrued amount is recovered from the final salary.",
        ),
        (
            "Records of leave",
            "The {office} keeps records of every kind of leave taken for {mm} months after an "
            "employee leaves. Employees can see their own records on the portal at any time.",
        ),
        (
            "Shift workers",
            "For employees who work shifts, leave is calculated in hours rather than days so that shifts "
            "of different lengths are treated fairly.",
        ),
        (
            "Medical appointments",
            "Routine medical and dental appointments should be arranged outside working hours "
            "where possible. Where that is not possible, time off is agreed locally.",
        ),
        (
            "Emergency contact details",
            "Employees should keep their emergency contact details up to date on the portal "
            "so that the {office} can reach their family in an emergency.",
        ),
        (
            "Planning cover",
            "Teams keep a shared leave calendar so that enough people are at work at all times. Busy "
            "periods are published at the start of each leave year.",
        ),
        (
            "Questions and answers",
            "Q: Can I cancel leave I have booked? A: Yes, through the portal, as long as the "
            "leave has not started. Q: Can I buy extra annual leave? A: Yes, through the salary-sacrifice scheme.",
        ),
        (
            "Returning to work",
            "After any absence of more than {nn} weeks, a short return-to-work conversation takes "
            "place to update the employee on changes in the team.",
        ),
        (
            "Working elsewhere during leave",
            "Employees must not work for another employer during paid leave from {org} "
            "without the written agreement of the {office}.",
        ),
        (
            "Contacts for leave questions",
            "Questions about leave can be raised with the {office} on extension {ext} or "
            "through the portal's enquiry form. Replies are normally given within {wd} working days.",
        ),
        (
            "Well-being support",
            "The employee assistance programme offers confidential advice on personal, financial "
            "and legal matters, twenty-four hours a day.",
        ),
        (
            "Carers' network",
            "{org} supports a staff network for employees with caring responsibilities. The network "
            "meets every {nn} weeks and can be contacted through the intranet.",
        ),
        (
            "Training days",
            "Mandatory training days are working time and are not deducted from annual leave, even when "
            "they take place away from the normal place of work.",
        ),
        (
            "Flexible working",
            "Requests to change working hours or patterns are dealt with under the flexible working "
            "procedure, which is separate from this policy.",
        ),
    ),
)


# ------------------------------------------------------------ library_lending

LIBRARY = Domain(
    key="library_lending",
    titles=_t(
        "Lending Policy",
        "Library Borrowing Regulations",
        "Loans and Renewals Policy",
        "Rules for Borrowing Library Materials",
    ),
    orgs=_t(
        "Ashgrove Public Libraries",
        "Marlow University Library",
        "Tidewater Community Libraries",
        "St Aldric's College Library",
        "Kilbride City Libraries",
    ),
    member="member",
    members="members",
    person="the borrower",
    claimant_label="Borrower",
    case="request",
    cases="requests",
    offices=_t("Lending Services team", "Circulation Desk", "Library Services office"),
    currency=_t("usd", "eur", "gbp"),
    purpose=_t(
        "These regulations explain how {members} of {org} may borrow, renew and return library materials, and the "
        "limits that apply.",
        "{org} lends its collections as widely as possible while keeping them available to everyone. This policy "
        "sets out the terms on which materials are lent.",
    ),
    scope_text=_t(
        "This policy applies to all registered {members} of {org} and to every item that the catalogue shows as "
        "available for loan. Reference-only items are outside its scope.",
        "The policy covers loans made to registered {members} at any service point of {org}. Inter-library loans are "
        "governed by the lending library's own terms.",
    ),
    scope=Scope(
        label="Issuing branch",
        values=_t(
            "Central Library",
            "Eastgate branch",
            "Harbour branch",
            "Science Library",
            "Mobile Library",
            "Law Library",
        ),
        annex_title="Loans issued by the {v}",
        annex_scope="loans issued by the {v}",
        case=_t(
            "The item was issued by the {v}.", "{name} borrowed the item from the {v}."
        ),
    ),
    conditions=(
        Condition(
            "years",
            "min",
            "the borrower has held membership for at least {t}",
            "Established Member",
            "a member who has held membership for at least {t}",
            "the borrower is an Established Member",
            _t(
                "{name} has held membership for {v}.",
            ),
            "Length of membership",
            lo=0,
            hi=20,
            thresholds=(2, 3, 5, 6),
            unit="{n} years",
        ),
        Condition(
            "category",
            "is",
            "the borrower is {t}",
            "Research Borrower",
            "a borrower who is {t}",
            "the borrower is a Research Borrower",
            _t(
                "{name} is {v}.",
            ),
            "Borrower category",
            choices=_t(
                "a postgraduate researcher",
                "an undergraduate student",
                "a member of staff",
                "an external reader",
                "a visiting scholar",
            ),
        ),
        Condition(
            "format",
            "is",
            "the item is {t}",
            "Designated Item",
            "an item that is {t}",
            "the item is a Designated Item",
            _t(
                "The item is {v}.",
            ),
            "Item format",
            choices=_t(
                "a printed book",
                "a DVD",
                "a laptop",
                "a bound journal volume",
                "a board game",
            ),
        ),
        Condition(
            "ontime",
            "min",
            "the borrower has returned at least {t} on time in the past year",
            "Reliable Borrower",
            "a member who has returned at least {t} on time in the past year",
            "the borrower is a Reliable Borrower",
            _t(
                "In the past year {name} returned {v} on time.",
            ),
            "Items returned on time (past year)",
            lo=0,
            hi=60,
            thresholds=(10, 15, 20, 25),
            unit="{n} items",
        ),
    ),
    noise=(
        Noise(
            "Membership number",
            _t("LIB-{n}", "M{n}"),
            "{name}'s membership number is {v}.",
        ),
        Noise(
            "Item barcode",
            _t(
                "3900{n}",
            ),
            "The item's barcode is {v}.",
        ),
        Noise(
            "Collection",
            _t(
                "general lending", "local history", "children's library", "music scores"
            ),
            "The item belongs to the {v} collection.",
        ),
    ),
    dates=DatePair(
        "borrowed",
        "Date borrowed",
        "the date on which the item was borrowed",
        "The item was borrowed on {d}.",
        "renewal",
        "Renewal requested",
        "the date on which the renewal is requested",
        "The renewal was requested on {d}.",
        (5, 45),
    ),
    choice=Quantity(
        key="loan_period",
        heading="Loan periods",
        noun="loan period",
        unit="{n} days",
        ladder=(7, 10, 14, 21, 28, 35, 42, 56),
        option="{val}",
        frags=_t(
            "the loan period is {val}",
            "items may be borrowed for {val}",
            "a loan period of {val} applies",
        ),
        base=_t(
            "Items are lent for a loan period of {val}. The due date is printed on the issue receipt.",
            "The loan period is {val} from the date of issue.",
        ),
    ),
    noul=Quantity(
        key="loan_limit",
        heading="Number of items on loan",
        noun="maximum number of items on loan",
        unit="{n} items",
        ladder=(4, 6, 8, 10, 12, 15, 20, 25, 30),
        option="{val}",
        frags=_t(
            "the maximum number of items on loan at one time is {val}",
            "a member may have up to {val} on loan at one time",
            "no more than {val} may be on loan to a member at one time",
        ),
        base=_t(
            "A member may have up to {val} on loan at one time, counting items on every card they hold.",
            "The maximum number of items on loan to one member at one time is {val}.",
        ),
        ask=_t(
            "If the request is granted, {name} will have {q} on loan.",
            "Granting the request would bring {name}'s loans to {q}.",
        ),
        ask_label="Items on loan if the request is granted",
    ),
    score=Levels(
        key="renewals",
        heading="Renewals",
        noun="renewals allowed",
        frags=_t(
            "loans may not be renewed",
            "a loan may be renewed once",
            "a loan may be renewed twice",
            "loans may be renewed without limit",
        ),
        needles=_t(
            "may not be renewed", "renewed once", "renewed twice", "without limit"
        ),
        criteria=_t(
            "No renewal is allowed",
            "One renewal is allowed",
            "Two renewals are allowed",
            "Renewals are allowed without limit",
        ),
    ),
    questions={
        "choice": _t(
            "What loan period applies to the item in {name}'s case under the policy?",
            "Which loan period should the {office} apply to this item?",
        ),
        "noul": _t(
            "Is the number of items {name} would have on loan within the maximum that applies?",
            "Can the {office} grant the request without exceeding the limit on items on loan that applies?",
        ),
        "score": _t(
            "How many renewals does the policy allow for {name}'s loan?",
            "Under the policy, what renewal allowance applies to this loan?",
        ),
    },
    extra_defs=(
        ("Due date", "the date by which a loan must be returned or renewed"),
        (
            "Service point",
            "any desk, self-service machine or locker at which loans are issued or returned",
        ),
        (
            "Reservation",
            "a request by another member to borrow an item that is on loan",
        ),
        (
            "Reference-only item",
            "an item that may be used in the library but not borrowed",
        ),
    ),
    static=_t(
        "Items may be returned at any service point, whatever branch issued them.",
        "An item reserved by another member must be returned by its due date.",
        "Members are responsible for the items issued on their card until the items are returned.",
    ),
    distractors=(
        (
            "Joining the library",
            "Membership is free for residents, students and staff. Applicants show proof of "
            "address or a valid student or staff card, and the card is issued on the spot.",
        ),
        (
            "Lost cards",
            "A lost card should be reported at once so that it can be blocked. A replacement card is "
            "issued at any service point on proof of identity.",
        ),
        (
            "Reservations",
            "Members may reserve items that are on loan. When the item is returned it is held for "
            "{wd} working days at the chosen branch, after which it passes to the next member in the queue.",
        ),
        (
            "Self-service machines",
            "Self-service machines issue and return most items. Items with a security case are "
            "unlocked automatically when issued.",
        ),
        (
            "Damaged items",
            "Please tell staff if an item is damaged when you borrow it, so that you are not held "
            "responsible. Do not try to repair damaged items yourself.",
        ),
        (
            "Lost items",
            "If an item is lost, the member is asked to pay the replacement cost shown in the catalogue "
            "together with a processing fee. The charge is cancelled if the item is found within {mm} months.",
        ),
        (
            "Study spaces",
            "Silent study areas are on the upper floors of the larger branches. Group study rooms can be "
            "booked for up to {hh} hours through the portal.",
        ),
        (
            "Printing and copying",
            "Printers and scanners are available in every branch. Copying must respect "
            "copyright law; notices beside each machine explain the limits.",
        ),
        (
            "Children's membership",
            "Children under twelve can join with the consent of a parent or carer, who is "
            "responsible for the items borrowed on the child's card.",
        ),
        (
            "Accessibility services",
            "Large-print books, audio books and a home delivery service are available for "
            "members who cannot easily visit a branch.",
        ),
        (
            "Digital collections",
            "E-books and e-journals are available through the catalogue. They are returned "
            "automatically at the end of the digital lending period.",
        ),
        (
            "Donations",
            "{org} welcomes donations of books in good condition. Donated items are added to the "
            "collection or sold to raise funds for new stock.",
        ),
        (
            "Food and drink",
            "Covered drinks may be taken into most areas. Food may be eaten only in the café and in "
            "the designated break area.",
        ),
        (
            "Questions and answers",
            "Q: Can I renew by telephone? A: Yes, on extension {ext} during opening hours. "
            "Q: Can I return items when the library is closed? A: Yes, using the external returns box.",
        ),
        (
            "Opening hours",
            "Opening hours are displayed at each branch and on the portal. Branches close early on the "
            "day before a public holiday.",
        ),
        (
            "Behaviour in the library",
            "Members are asked to respect others using the library. Staff may ask anyone who "
            "disturbs other users to leave for the rest of the day.",
        ),
        (
            "Local history enquiries",
            "Enquiries about local history and family records are answered by the archive "
            "team. Complex enquiries may take up to {dd} days.",
        ),
        (
            "Wi-Fi",
            "Free Wi-Fi is available in every branch. Users accept the acceptable-use terms when they connect.",
        ),
        (
            "Events",
            "Author talks, reading groups and homework clubs run throughout the year. Most events are free "
            "and are listed on the portal.",
        ),
        (
            "Contacting the {office}",
            "The {office} can be reached on extension {ext}, by email or at any service "
            "desk. Messages left out of hours are answered on the next working day.",
        ),
        (
            "Volunteers",
            "Volunteers help with events, the home delivery service and shelving. They receive training "
            "and are supervised by a named member of staff.",
        ),
    ),
)


# ------------------------------------------------------------ it_access

ITACCESS = Domain(
    key="it_access",
    titles=_t(
        "Information Systems Access Policy",
        "IT Access Control Policy",
        "Policy on Access to Computer Systems",
        "User Access Management Standard",
    ),
    orgs=_t(
        "Quillon Bank",
        "Harrowgate Health Trust",
        "Parallax Insurance",
        "Obsidian Analytics",
        "Wrenfield City Council",
        "Calloway Engineering Group",
    ),
    member="user",
    members="users",
    person="the requester",
    claimant_label="Requester",
    case="access request",
    cases="access requests",
    offices=_t(
        "IT Service Desk", "Identity and Access team", "Information Security Office"
    ),
    currency=_t("usd", "eur", "gbp"),
    purpose=_t(
        "This policy controls how {members} obtain access to the information systems of {org}, so that access is "
        "granted only where it is needed and only for as long as it is needed.",
        "{org} protects its information by limiting who can use each system. This policy sets out how {cases} are "
        "made, checked and granted.",
    ),
    scope_text=_t(
        "This policy applies to every {member} of the information systems of {org}, including employees, contractors "
        "and staff of partner organisations who are given an account.",
        "The policy covers all accounts on systems owned or managed by {org}. Accounts on suppliers' own systems are "
        "outside its scope.",
    ),
    scope=Scope(
        label="Business unit",
        values=_t(
            "Retail Operations",
            "Claims Processing",
            "Research and Development",
            "Corporate Finance",
            "Customer Contact",
            "Estates and Facilities",
        ),
        annex_title="The {v} business unit",
        annex_scope="{members} who work in the {v} business unit",
        case=_t(
            "{name} works in the {v} business unit.",
            "{name} belongs to the {v} business unit.",
        ),
    ),
    conditions=(
        Condition(
            "clearance",
            "min",
            "the requester holds security clearance level {t} or higher",
            "Cleared User",
            "a user who holds security clearance level {t} or higher",
            "the requester is a Cleared User",
            _t(
                "{name} holds security clearance level {v}.",
            ),
            "Security clearance level",
            lo=1,
            hi=5,
            thresholds=(2, 3, 4),
        ),
        Condition(
            "training",
            "max",
            "the requester completed security training no more than {t} ago",
            "Current Training",
            "security awareness training completed no more than {t} earlier",
            "the requester has Current Training",
            _t(
                "{name} last completed security training {v} ago.",
            ),
            "Security training completed",
            lo=1,
            hi=30,
            thresholds=(6, 9, 12, 18),
            unit="{n} months",
        ),
        Condition(
            "worker",
            "is",
            "the requester is {t}",
            "Internal User",
            "a requester who is {t}",
            "the requester is an Internal User",
            _t(
                "{name} is {v}.",
            ),
            "Worker type",
            choices=_t(
                "a permanent employee",
                "a contractor",
                "a temporary agency worker",
                "a secondee",
                "an intern",
            ),
        ),
        Condition(
            "system",
            "is",
            "the request concerns {t}",
            "Sensitive System",
            "{t}",
            "the request concerns a Sensitive System",
            _t(
                "The request concerns {v}.",
            ),
            "System",
            choices=_t(
                "the payroll system",
                "the customer database",
                "the document archive",
                "the source-code repository",
                "the general ledger",
            ),
        ),
    ),
    noise=(
        Noise(
            "Ticket number",
            _t("REQ-{n}", "INC-{n}"),
            "The request was logged as ticket {v}.",
        ),
        Noise(
            "Device",
            _t(
                "a managed laptop",
                "a desktop workstation",
                "a thin client",
                "a managed tablet",
            ),
            "Access will be used from {v}.",
        ),
    ),
    dates=DatePair(
        "logged",
        "Request logged",
        "the date on which the access request is logged",
        "The request was logged on {d}.",
        "start",
        "Access needed from",
        "the date from which access is needed",
        "Access is needed from {d}.",
        (2, 45),
    ),
    choice=Quantity(
        key="temp_access",
        heading="Temporary access",
        noun="longest temporary access grant",
        unit="{n} days",
        ladder=(7, 14, 21, 30, 45, 60, 90, 120, 180),
        option="{val}",
        frags=_t(
            "temporary access may be granted for up to {val}",
            "the longest temporary access grant is {val}",
            "a temporary access grant may last no more than {val}",
        ),
        base=_t(
            "Temporary access may be granted for up to {val}, after which the account is suspended "
            "automatically unless a new request is approved.",
            "The longest temporary access grant is {val}.",
        ),
    ),
    noul=Quantity(
        key="quota",
        heading="Storage quotas",
        noun="maximum storage quota for a project share",
        unit="{n} GB",
        ladder=(50, 100, 150, 200, 250, 300, 400, 500, 750, 1000),
        option="{val}",
        frags=_t(
            "the maximum storage quota for a project share is {val}",
            "a project share may be given a quota of up to {val}",
            "no project share may be given a quota above {val}",
        ),
        base=_t(
            "A project share may be given a storage quota of up to {val}. Larger datasets belong in the "
            "research data store.",
            "The maximum storage quota for a project share is {val}.",
        ),
        ask=_t(
            "{name} has asked for a project-share quota of {q}.",
            "The quota requested for the share is {q}.",
        ),
        ask_label="Quota requested",
    ),
    score=Levels(
        key="approval",
        heading="Approval of access",
        noun="approval the request needs",
        frags=_t(
            "requests of this kind may be approved by the requester's team leader",
            "requests of this kind must be approved by the system owner",
            "requests of this kind must be approved by the Information Security Officer",
            "requests of this kind must be approved by the Chief Information Officer",
        ),
        needles=_t(
            "team leader",
            "system owner",
            "Information Security Officer",
            "Chief Information Officer",
        ),
        criteria=_t(
            "The requester's team leader may approve it",
            "The system owner must approve it",
            "The Information Security Officer must approve it",
            "The Chief Information Officer must approve it",
        ),
    ),
    questions={
        "choice": _t(
            "For how long may temporary access be granted on {name}'s request under the policy?",
            "What is the longest temporary access grant the {office} may make for this request?",
        ),
        "noul": _t(
            "Is the storage quota that {name} requested within the maximum that applies to this request?",
            "Can the {office} grant the quota as requested, that is, is it within the applicable maximum?",
        ),
        "score": _t(
            "Whose approval does {name}'s access request need under the policy?",
            "Under the policy, which approval is required before this access is granted?",
        ),
    },
    extra_defs=(
        (
            "Account",
            "a set of credentials that allows a person to use one or more systems",
        ),
        (
            "Privileged access",
            "access that allows a user to change the configuration of a system",
        ),
        ("Leaver", "a user whose employment or contract with {org} has ended"),
        (
            "Multi-factor authentication",
            "sign-in that requires something the user knows and something the user has",
        ),
    ),
    static=_t(
        "Every account belongs to one named person; shared accounts are not permitted.",
        "Passwords must meet the length and complexity rules published by the {office}.",
        "Access is removed on the last working day of any user who leaves {org}.",
    ),
    distractors=(
        (
            "Password rules",
            "Passwords must be at least twelve characters long and must not be reused across systems. "
            "Password managers approved by the {office} may be used to store them.",
        ),
        (
            "Multi-factor sign-in",
            "Every remote sign-in requires a second factor, such as an authenticator app or a "
            "hardware key. Codes sent by text message are used only as a fallback.",
        ),
        (
            "Leavers",
            "Line managers must tell the {office} of every leaver at least {wd} working days before the last "
            "day. Accounts are disabled at the end of that day and deleted after {mm} months.",
        ),
        (
            "Movers",
            "When a user changes role, the access needed for the old role is removed and the access for the new "
            "role is requested afresh. Access is never simply added to.",
        ),
        (
            "Access reviews",
            "System owners review the list of users of their systems every {mm} months and confirm "
            "that each account is still needed.",
        ),
        (
            "Lost or stolen devices",
            "A lost or stolen device must be reported to the {office} immediately so that it "
            "can be wiped remotely. Users are not penalised for prompt reporting.",
        ),
        (
            "Phishing",
            "Suspicious emails should be reported with the report button and then deleted. Staff should "
            "never enter their password on a page reached from a link in an unexpected email.",
        ),
        (
            "Remote working",
            "Users working away from the office must connect through the virtual private network and "
            "must not leave devices unattended in public places.",
        ),
        (
            "Software installation",
            "Only software from the approved catalogue may be installed. Requests for new "
            "software are assessed by the {office} for licensing and security.",
        ),
        (
            "Removable media",
            "Data may be copied to removable media only if the media is encrypted and issued by the "
            "{office}. Personal memory sticks must not be used.",
        ),
        (
            "Logging and monitoring",
            "Sign-ins and changes to sensitive data are logged. Logs are kept for {mm} months "
            "and reviewed when an incident is investigated.",
        ),
        (
            "Incident reporting",
            "Any suspected security incident, however minor, is reported to the {office} on "
            "extension {ext}. Reports can also be made through the portal.",
        ),
        (
            "Clear desk and screen",
            "Screens are locked whenever a device is left unattended, and papers containing "
            "personal data are locked away at the end of the day.",
        ),
        (
            "Supplier access",
            "Suppliers who need to reach {org} systems for maintenance use a monitored connection "
            "that is opened for each session and closed at its end.",
        ),
        (
            "Questions and answers",
            "Q: Can I share my account with a colleague who is covering my work? A: No; the "
            "colleague must request access in their own name. Q: How do I reset my password? A: Through the portal.",
        ),
        (
            "Test accounts",
            "Test accounts are created only in test environments and never hold real personal data. "
            "They are removed when the test is complete.",
        ),
        (
            "Printing",
            "Documents sent to shared printers are released only when the user taps their card at the "
            "printer, so that printouts are not left in trays.",
        ),
        (
            "Encryption",
            "All laptops and mobile devices issued by {org} are encrypted. Users must not disable the "
            "encryption or the automatic updates.",
        ),
        (
            "Training",
            "All users complete the security awareness course when they join and at intervals set by the "
            "{office}. Completion is recorded on the portal.",
        ),
        (
            "Contacts",
            "The {office} is staffed from 8 a.m. to 6 p.m. on working days on extension {ext}. Urgent "
            "security matters out of hours go to the on-call engineer.",
        ),
        (
            "Cloud services",
            "Cloud services may be used for {org} data only if they appear on the approved list. "
            "Personal cloud storage accounts must not be used for work files.",
        ),
    ),
)


# ------------------------------------------------------------ grant_funding

GRANTS = Domain(
    key="grant_funding",
    titles=_t(
        "Small Grants Programme Rules",
        "Research Grant Funding Policy",
        "Community Fund Grant Guidelines",
        "Rules of the Project Grants Scheme",
    ),
    orgs=_t(
        "Calderbrook Foundation",
        "Innis Arts Council",
        "Verity Science Trust",
        "Northmere Community Fund",
        "Hollins Heritage Trust",
    ),
    member="applicant",
    members="applicants",
    person="the applicant",
    claimant_label="Applicant",
    case="application",
    cases="applications",
    offices=_t("Grants Office", "Programme Team", "Funding Unit"),
    currency=_t("usd", "eur", "gbp"),
    purpose=_t(
        "These rules explain who may apply to {org} for a grant, how much may be requested and how {cases} are "
        "assessed.",
        "{org} funds projects that benefit the communities it serves. This document sets out the terms of the "
        "scheme and the limits on what it will fund.",
    ),
    scope_text=_t(
        "These rules apply to every {case} made to the scheme by an eligible organisation. Individuals may not "
        "apply in their own name.",
        "The rules cover all {cases} received by {org} under this scheme. Commissioned work and emergency awards are "
        "handled separately.",
    ),
    scope=Scope(
        label="Area",
        values=_t(
            "the Coastal Districts",
            "the Upland Districts",
            "the Metropolitan Area",
            "the Island Communities",
            "the Border Counties",
            "the River Valley",
        ),
        annex_title="Applicants based in {v}",
        annex_scope="{members} whose registered address is in {v}",
        case=_t(
            "{name}'s organisation is based in {v}.",
            "The applicant organisation is registered in {v}.",
        ),
    ),
    conditions=(
        Condition(
            "partners",
            "min",
            "the project involves at least {t}",
            "Collaborative Project",
            "a project that involves at least {t}",
            "the project is a Collaborative Project",
            _t(
                "The project involves {v}.",
            ),
            "Partner organisations",
            lo=0,
            hi=9,
            thresholds=(2, 3, 4),
            unit="{n} partner organisations",
        ),
        Condition(
            "age",
            "min",
            "the applicant organisation has been operating for at least {t}",
            "Established Organisation",
            "an organisation that has been operating for at least {t}",
            "the applicant is an Established Organisation",
            _t(
                "The organisation has been operating for {v}.",
            ),
            "Years operating",
            lo=0,
            hi=30,
            thresholds=(2, 3, 5, 8),
            unit="{n} years",
        ),
        Condition(
            "body",
            "is",
            "the applicant is {t}",
            "Eligible Body",
            "an applicant that is {t}",
            "the applicant is an Eligible Body",
            _t(
                "The applicant is {v}.",
            ),
            "Type of organisation",
            choices=_t(
                "a registered charity",
                "a university department",
                "a social enterprise",
                "a community interest company",
                "an informal community group",
            ),
        ),
        Condition(
            "duration",
            "max",
            "the project lasts no more than {t}",
            "Short Project",
            "a project lasting no more than {t}",
            "the project is a Short Project",
            _t(
                "The project is planned to last {v}.",
            ),
            "Project length",
            lo=3,
            hi=48,
            thresholds=(12, 18, 24),
            unit="{n} months",
        ),
    ),
    noise=(
        Noise(
            "Reference", _t("GR-{n}", "APP-{n}"), "The application reference is {v}."
        ),
        Noise(
            "Theme",
            _t(
                "youth participation",
                "heritage skills",
                "public health",
                "digital inclusion",
                "local food",
            ),
            "The project theme is {v}.",
        ),
    ),
    dates=DatePair(
        "received",
        "Application received",
        "the date on which the application is received",
        "The application was received on {d}.",
        "start",
        "Proposed start",
        "the proposed start date of the project",
        "The project is proposed to start on {d}.",
        (20, 120),
    ),
    choice=Quantity(
        key="max_award",
        heading="Size of awards",
        noun="maximum award",
        unit="money",
        ladder=(5000, 7500, 10000, 12500, 15000, 20000, 25000, 30000, 40000, 50000),
        option="{val}",
        frags=_t(
            "the maximum award is {val}",
            "grants of up to {val} may be made",
            "an award may not exceed {val}",
        ),
        base=_t(
            "Grants of up to {val} may be made for a single project, including any contribution to overheads.",
            "The maximum award is {val} per project.",
        ),
    ),
    noul=Quantity(
        key="indirect_rate",
        heading="Indirect costs",
        noun="maximum indirect-cost rate",
        unit="{n}%",
        ladder=(5, 8, 10, 12, 15, 18, 20, 25),
        option="{val} of direct costs",
        frags=_t(
            "indirect costs may be charged at up to {val} of direct costs",
            "the maximum indirect-cost rate is {val} of direct costs",
            "the indirect-cost rate may not exceed {val} of direct costs",
        ),
        base=_t(
            "Indirect costs may be charged at up to {val} of direct costs. The rate must be shown in the budget.",
            "The maximum indirect-cost rate is {val} of direct costs.",
        ),
        ask=_t(
            "The budget charges indirect costs at {q} of direct costs.",
            "{name}'s budget includes indirect costs at {q} of direct costs.",
        ),
        ask_label="Indirect-cost rate in budget",
    ),
    score=Levels(
        key="review",
        heading="Assessment route",
        noun="assessment route",
        frags=_t(
            "applications are decided on a desk review by a grants officer",
            "applications are assessed by two external reviewers",
            "applications are assessed by the full assessment panel",
            "applications are decided by the board of trustees",
        ),
        needles=_t(
            "desk review",
            "two external reviewers",
            "full assessment panel",
            "board of trustees",
        ),
        criteria=_t(
            "A desk review by a grants officer",
            "Assessment by two external reviewers",
            "Assessment by the full assessment panel",
            "A decision by the board of trustees",
        ),
    ),
    questions={
        "choice": _t(
            "What is the maximum award {name}'s application can receive under the rules?",
            "Which maximum award applies to this application?",
        ),
        "noul": _t(
            "Is the indirect-cost rate in {name}'s budget within the maximum rate that applies?",
            "Can the {office} accept the budget as submitted, that is, is its indirect-cost rate within the "
            "applicable maximum?",
        ),
        "score": _t(
            "Which assessment route does {name}'s application follow under the rules?",
            "Under the rules, how is this application to be assessed?",
        ),
    },
    extra_defs=(
        (
            "Direct costs",
            "costs incurred only because the project takes place, such as staff time and materials",
        ),
        (
            "Lead applicant",
            "the organisation that submits the application and receives the grant",
        ),
        (
            "Match funding",
            "money or time contributed to the project from sources other than {org}",
        ),
        (
            "Grant agreement",
            "the contract signed by the lead applicant when an award is accepted",
        ),
    ),
    static=_t(
        "Only one application per organisation may be under consideration at any time.",
        "Grants are paid in instalments against the milestones set out in the grant agreement.",
        "Retrospective costs, incurred before the grant agreement is signed, are not funded.",
    ),
    distractors=(
        (
            "How to apply",
            "Applications are made on the portal. The form asks for a project summary, a timetable, a "
            "budget and the names of two referees. Incomplete forms cannot be submitted.",
        ),
        (
            "Deadlines",
            "The scheme has rolling deadlines. Applications received after the close of a round are "
            "considered in the next round without any need to resubmit.",
        ),
        (
            "Budgets",
            "Budgets should be realistic and show how each figure was worked out. Quotes should be attached "
            "for any single item of equipment costing more than a small amount.",
        ),
        (
            "Referees",
            "Referees must know the applicant's work but must not be employed by the applicant. The "
            "{office} contacts referees directly and may ask them for a short written statement.",
        ),
        (
            "Monitoring",
            "Grant holders submit a short progress report every {mm} months and a final report within "
            "{dd} days of the end of the project.",
        ),
        (
            "Publicity",
            "Grant holders acknowledge the support of {org} in publicity about the project, using the "
            "logo supplied by the {office}.",
        ),
        (
            "Safeguarding",
            "Projects that work with children or vulnerable adults must have a safeguarding policy and "
            "must carry out the checks required by law.",
        ),
        (
            "Equipment",
            "Equipment bought with a grant remains the property of the grant holder, but it must be used for "
            "the purposes of the project for its useful life.",
        ),
        (
            "Changes to a project",
            "A grant holder who needs to change the timetable or the activities of a project "
            "must ask the {office} before making the change.",
        ),
        (
            "Unspent funds",
            "Any part of a grant not spent on the project by the end date must be returned to {org} "
            "within {dd} days.",
        ),
        (
            "Conflicts of interest",
            "Panel members declare any interest in an application before it is discussed and "
            "take no part in the discussion of an application in which they have an interest.",
        ),
        (
            "Feedback",
            "Unsuccessful applicants may ask for feedback within {dd} days of the decision. Feedback is given "
            "in writing and is intended to help with future applications.",
        ),
        (
            "Questions and answers",
            "Q: Can we apply for the salary of an existing post? A: Only for the time spent on "
            "the project. Q: Can we submit supporting videos? A: Yes, as a link in the application form.",
        ),
        (
            "Environmental impact",
            "Applicants are asked to describe how the project will limit its environmental "
            "impact, for example through travel choices and the reuse of materials.",
        ),
        (
            "Accessibility of events",
            "Events funded by the scheme should be held in accessible venues and publicised "
            "in formats that everyone can use.",
        ),
        (
            "Evaluation",
            "Larger projects are encouraged to set aside time for evaluation and to share what they learn "
            "with other organisations working in the same field.",
        ),
        (
            "Advice sessions",
            "The {office} runs online advice sessions every {nn} weeks for organisations thinking of "
            "applying. Booking is through the portal.",
        ),
        (
            "Bank details",
            "Grants are paid only into a bank account held in the name of the lead applicant. Changes to "
            "bank details must be confirmed by two signatories.",
        ),
        (
            "Record keeping",
            "Grant holders keep invoices and payroll records relating to the project for {mm} months "
            "after the final report and make them available on request.",
        ),
        (
            "Contacts",
            "The {office} can be reached on extension {ext} or through the enquiry form on the portal. "
            "Named contacts are {name1} and {name2}.",
        ),
        (
            "Acknowledging receipt",
            "Every application receives an automatic acknowledgement with a reference number. "
            "Please quote the reference in all correspondence.",
        ),
    ),
)


# ------------------------------------------------------------ parking_permits

PARKING = Domain(
    key="parking_permits",
    titles=_t(
        "Parking Permit Scheme Rules",
        "Residents' and Visitors' Parking Policy",
        "Permit Parking Policy",
        "Controlled Parking Zone Permit Rules",
    ),
    orgs=_t(
        "Ellesmere Borough Council",
        "Port Carrow Town Council",
        "Linton Vale District Council",
        "Kingsreach City Council",
        "Asheby Parish Council",
    ),
    member="resident",
    members="residents",
    person="the applicant",
    claimant_label="Applicant",
    case="permit application",
    cases="permit applications",
    offices=_t("Parking Services office", "Permits Office", "Traffic Management team"),
    currency=_t("usd", "eur", "gbp"),
    purpose=_t(
        "These rules explain how {members} obtain parking permits from {org}, what the permits cost and the limits "
        "that apply to each kind of permit.",
        "{org} manages kerbside parking so that {members} can park near their homes. This document sets out the "
        "terms of the permit scheme.",
    ),
    scope_text=_t(
        "These rules apply to every {case} for an address inside a controlled parking zone of {org}. Business "
        "permits are covered by a separate scheme.",
        "The rules cover resident and visitor permits for all controlled zones. Permits for the council's own car "
        "parks are outside their scope.",
    ),
    scope=Scope(
        label="Parking zone",
        values=_t(
            "Zone A (Old Town)",
            "Zone B (Harbourside)",
            "Zone C (Station Quarter)",
            "Zone D (Hillcrest)",
            "Zone E (Riverside)",
            "Zone F (Market Hill)",
        ),
        annex_title="Addresses in {v}",
        annex_scope="{members} whose address is in {v}",
        case=_t("{name}'s address is in {v}.", "The address is in {v}."),
    ),
    conditions=(
        Condition(
            "emissions",
            "max",
            "the vehicle's CO2 emissions are no more than {t}",
            "Low-emission Vehicle",
            "a vehicle whose CO2 emissions are no more than {t}",
            "the vehicle is a Low-emission Vehicle",
            _t(
                "The vehicle's CO2 emissions are {v}.",
            ),
            "CO2 emissions",
            lo=0,
            hi=240,
            thresholds=(50, 75, 100, 120),
            unit="{n} g/km",
        ),
        Condition(
            "residence",
            "min",
            "the applicant has lived at the address for at least {t}",
            "Settled Resident",
            "a resident who has lived at the address for at least {t}",
            "the applicant is a Settled Resident",
            _t(
                "{name} has lived at the address for {v}.",
            ),
            "Time at address",
            lo=1,
            hi=120,
            thresholds=(6, 12, 24, 36),
            unit="{n} months",
        ),
        Condition(
            "status",
            "is",
            "the applicant is {t}",
            "Priority Applicant",
            "an applicant who is {t}",
            "the applicant is a Priority Applicant",
            _t(
                "{name} is {v}.",
            ),
            "Applicant status",
            choices=_t(
                "a disabled badge holder",
                "a registered carer",
                "a key worker",
                "a car-club member",
                "a sole trader working from home",
            ),
        ),
        Condition(
            "vehicles",
            "max",
            "the household has no more than {t} registered at the address",
            "Low-car Household",
            "a household with no more than {t} registered at the address",
            "the household is a Low-car Household",
            _t(
                "The household has {v} registered at the address.",
            ),
            "Vehicles at the address",
            lo=1,
            hi=4,
            thresholds=(1, 2),
            unit="{n} vehicles",
        ),
    ),
    noise=(
        Noise(
            "Vehicle registration",
            _t("KX{n}", "LB{n}", "TR{n}"),
            "The vehicle registration is {v}.",
        ),
        Noise(
            "Payment",
            _t("by card online", "by direct debit", "at the customer service centre"),
            "Payment will be made {v}.",
        ),
    ),
    dates=DatePair(
        "received",
        "Application received",
        "the date on which the application is received",
        "The application was received on {d}.",
        "start",
        "Permit to start",
        "the date on which the permit is to start",
        "The permit is to start on {d}.",
        (3, 40),
    ),
    choice=Quantity(
        key="permit_fee",
        heading="Permit fees",
        noun="annual fee for a resident permit",
        unit="money",
        ladder=(40, 60, 75, 90, 110, 130, 150, 180, 210, 250),
        option="{val} a year",
        frags=_t(
            "the annual fee for a resident permit is {val}",
            "a resident permit costs {val} a year",
            "resident permits are charged at {val} a year",
        ),
        base=_t(
            "A resident permit costs {val} a year and is valid for twelve months from its start date.",
            "The annual fee for a resident permit is {val}.",
        ),
    ),
    noul=Quantity(
        key="visitor_days",
        heading="Visitor permits",
        noun="maximum visitor-permit days a year",
        unit="{n} days",
        ladder=(20, 30, 40, 50, 60, 75, 90, 120),
        option="{val} a year",
        frags=_t(
            "each household may buy up to {val} of visitor permits a year",
            "the maximum visitor-permit allowance is {val} a year",
            "no household may hold more than {val} of visitor permits in a year",
        ),
        base=_t(
            "Each household may buy up to {val} of visitor permits in any permit year.",
            "The maximum visitor-permit allowance is {val} a year for each household.",
        ),
        ask=_t(
            "{name} has asked for {q} of visitor permits for the year.",
            "The application is for {q} of visitor permits.",
        ),
        ask_label="Visitor-permit days requested",
    ),
    score=Levels(
        key="band",
        heading="Waiting-list bands",
        noun="waiting-list band",
        frags=_t(
            "applications are placed in Band D of the waiting list",
            "applications are placed in Band C of the waiting list",
            "applications are placed in Band B of the waiting list",
            "applications are placed in Band A of the waiting list",
        ),
        needles=_t("Band D", "Band C", "Band B", "Band A"),
        criteria=_t(
            "Band D, the lowest priority on the waiting list",
            "Band C, below-average priority",
            "Band B, above-average priority",
            "Band A, the highest priority on the waiting list",
        ),
    ),
    questions={
        "choice": _t(
            "What annual fee applies to {name}'s resident permit under the rules?",
            "Which annual fee should the {office} charge for this permit?",
        ),
        "noul": _t(
            "Is the number of visitor-permit days {name} asked for within the maximum that applies?",
            "Can the {office} issue the visitor permits as requested without exceeding the applicable "
            "maximum?",
        ),
        "score": _t(
            "In which waiting-list band is {name}'s application placed under the rules?",
            "Under the rules, which band of the waiting list does this application go into?",
        ),
    },
    extra_defs=(
        (
            "Controlled parking zone",
            "an area in which parking is restricted to permit holders at signed times",
        ),
        (
            "Household",
            "the people who live at one address and share its living accommodation",
        ),
        (
            "Permit year",
            "the twelve months starting on the date on which a permit is first issued",
        ),
        (
            "Registered keeper",
            "the person shown as the keeper on the vehicle's registration document",
        ),
    ),
    static=_t(
        "A permit is valid only for the vehicle and the zone shown on it.",
        "Permits are issued in electronic form; no paper permit needs to be displayed.",
        "A permit does not guarantee that a parking space will be available.",
    ),
    distractors=(
        (
            "How to apply",
            "Applications are made on the portal. Applicants upload proof of address and the vehicle's "
            "registration document, and pay online. Most applications are decided within {wd} working days.",
        ),
        (
            "Proof of address",
            "Acceptable proof of address includes a council tax bill, a tenancy agreement or a "
            "utility bill dated within the last {nn} months.",
        ),
        (
            "Changing vehicles",
            "A permit holder who changes vehicle can transfer the permit on the portal. The new "
            "vehicle's details must be entered before it is parked in the zone.",
        ),
        (
            "Moving house",
            "A permit holder who moves to a new address inside the same zone updates the address on the "
            "portal. A move to another zone requires a new application.",
        ),
        (
            "Suspensions",
            "Bays may be suspended for roadworks, removals or events. Signs are put up at least {wd} "
            "working days before a planned suspension.",
        ),
        (
            "Enforcement",
            "Civil enforcement officers patrol every zone. A penalty charge notice may be issued to a "
            "vehicle parked without a valid permit during the controlled hours.",
        ),
        (
            "Challenging a penalty",
            "A penalty charge notice can be challenged within {dd} days on the portal. The "
            "challenge should explain the circumstances and include any evidence.",
        ),
        (
            "Electric vehicle charging",
            "Charging bays are reserved for vehicles that are charging. Permit holders may "
            "use them only while their vehicle is connected.",
        ),
        (
            "Car clubs",
            "Car-club vehicles have their own reserved bays. Residents who join a car club may find that "
            "they no longer need a permit of their own.",
        ),
        (
            "Motorcycles",
            "Solo motorcycles may park in the dedicated motorcycle bays without a permit. Motorcycles "
            "parked in permit bays need a permit.",
        ),
        (
            "Coaches and large vehicles",
            "Vehicles over a certain length may not park in permit bays. Owners of such "
            "vehicles should ask the {office} about alternatives.",
        ),
        (
            "Refunds on surrender",
            "A permit that is no longer needed can be surrendered on the portal. Any refund is "
            "worked out from the number of whole months remaining.",
        ),
        (
            "Lost access to the portal",
            "Residents who cannot use the portal can apply by telephone on extension {ext} "
            "or in person at the customer service centre.",
        ),
        (
            "Questions and answers",
            "Q: Can my carer park outside my home? A: Yes, with a carer's permit. Q: Do I need "
            "a permit at weekends? A: Only if the signs show controlled hours at weekends.",
        ),
        (
            "Consultation on new zones",
            "New zones are introduced only after consultation with the {members} of the "
            "area. Results of consultations are published on the portal.",
        ),
        (
            "Signs and markings",
            "The times of control are shown on signs at the entry to each zone and on plates beside "
            "the bays. Where signs differ, the plate beside the bay should be followed.",
        ),
        (
            "Dispensations for tradespeople",
            "Tradespeople working at an address may apply for a short-term "
            "dispensation to park near the property while the work is carried out.",
        ),
        (
            "Street cleaning",
            "On street-cleaning days, bays are cleared for a few hours. The dates are published on "
            "the portal every {nn} weeks.",
        ),
        (
            "Data sharing",
            "Vehicle details are checked with the national vehicle register. The data is used only to "
            "run the scheme and to enforce parking restrictions.",
        ),
        (
            "Contacts",
            "The {office} can be reached on extension {ext} from 9 a.m. to 5 p.m. on working days. Written "
            "enquiries go to {name1}, the permits coordinator.",
        ),
        (
            "Accessibility",
            "Disabled badge holders may park in marked disabled bays without a resident permit. Badges "
            "must be displayed clearly.",
        ),
    ),
)


# ------------------------------------------------------------ warranty_service

WARRANTY = Domain(
    key="warranty_service",
    titles=_t(
        "Warranty and Service Policy",
        "Limited Warranty Terms and Service Policy",
        "Product Warranty Policy",
        "After-Sales Service and Warranty Terms",
    ),
    orgs=_t(
        "Ostrander Appliances",
        "Kelvin & Marsh Tools",
        "Aurora Audio",
        "Pellucid Optics",
        "Stonebridge Furniture",
        "Vantage Garden Machinery",
    ),
    member="customer",
    members="customers",
    person="the customer",
    claimant_label="Customer",
    case="warranty claim",
    cases="warranty claims",
    offices=_t("Service Centre", "Warranty Desk", "After-Sales team"),
    currency=_t("usd", "eur", "gbp"),
    purpose=_t(
        "This policy sets out the warranty that {org} gives on its products and how {cases} are handled.",
        "{org} stands behind the quality of its products. This document explains what the warranty covers, how long "
        "it lasts and how service is provided.",
    ),
    scope_text=_t(
        "This policy applies to new products sold by {org} or its authorised dealers to {members} for their own use. "
        "Products sold as refurbished carry their own terms.",
        "The policy covers every product that bears the {org} name and was bought new. It does not affect the "
        "customer's statutory rights.",
    ),
    scope=Scope(
        label="Product range",
        values=_t(
            "kitchen appliances",
            "power tools",
            "audio equipment",
            "garden machinery",
            "office furniture",
            "optical instruments",
        ),
        annex_title="The {v} range",
        annex_scope="products in the {v} range",
        case=_t(
            "The product belongs to the {v} range.",
            "The product is part of the {v} range.",
        ),
    ),
    conditions=(
        Condition(
            "plan",
            "is",
            "the customer holds {t}",
            "Plan Holder",
            "a customer who holds {t}",
            "the customer is a Plan Holder",
            _t(
                "{name} holds {v}.",
            ),
            "Care plan",
            choices=_t(
                "the Gold care plan",
                "the Silver care plan",
                "the Basic care plan",
                "no care plan",
            ),
        ),
        Condition(
            "repairs",
            "max",
            "the product has had no more than {t}",
            "Low-repair Product",
            "a product that has had no more than {t}",
            "the product is a Low-repair Product",
            _t(
                "The product has had {v}.",
            ),
            "Previous repairs",
            lo=0,
            hi=6,
            thresholds=(1, 2, 3),
            unit="{n} previous repairs",
        ),
        Condition(
            "price",
            "min",
            "the purchase price was at least {t}",
            "Premium Product",
            "a product whose purchase price was at least {t}",
            "the product is a Premium Product",
            _t(
                "The purchase price was {v}.",
            ),
            "Purchase price",
            lo=60,
            hi=3000,
            thresholds=(400, 600, 800, 1000, 1500),
            unit="money",
        ),
        Condition(
            "fault",
            "is",
            "the fault is {t}",
            "Covered Fault",
            "{t}",
            "the fault is a Covered Fault",
            _t(
                "The fault reported is {v}.",
            ),
            "Fault",
            choices=_t(
                "an electrical fault",
                "a mechanical fault",
                "cosmetic damage",
                "water damage",
                "a software fault",
            ),
        ),
    ),
    noise=(
        Noise("Serial number", _t("SN-{n}", "K{n}"), "The serial number is {v}."),
        Noise(
            "Retailer",
            _t("an authorised dealer", "the {org} online shop", "a department store"),
            "The product was bought from {v}.",
        ),
    ),
    dates=DatePair(
        "purchased",
        "Date of purchase",
        "the date of purchase",
        "The product was bought on {d}.",
        "reported",
        "Fault reported",
        "the date on which the fault is reported",
        "The fault was reported on {d}.",
        (40, 400),
    ),
    choice=Quantity(
        key="warranty_period",
        heading="Warranty period",
        noun="warranty period",
        unit="{n} months",
        ladder=(6, 12, 18, 24, 30, 36, 48, 60),
        option="{val}",
        frags=_t(
            "the warranty period is {val}",
            "products are covered for {val}",
            "a warranty period of {val} applies",
        ),
        base=_t(
            "Products are covered for {val} from the date of purchase against defects in materials and "
            "workmanship.",
            "The warranty period is {val}, counted from the date of purchase.",
        ),
    ),
    noul=Quantity(
        key="repair_limit",
        heading="Authorising repairs",
        noun="repair cost the service centre may authorise",
        unit="money",
        ladder=(100, 150, 200, 250, 300, 350, 400, 500, 600),
        option="{val}",
        frags=_t(
            "the service centre may authorise repairs costing up to {val} without referral",
            "repairs costing up to {val} may be authorised locally without referral",
            "the limit for repairs authorised without referral is {val}",
        ),
        base=_t(
            "The service centre may authorise repairs costing up to {val} without referral to the warranty "
            "manager.",
            "The limit for repairs authorised without referral is {val} per claim.",
        ),
        ask=_t(
            "The estimated cost of the repair is {q}.",
            "The service centre estimates the repair at {q}.",
        ),
        ask_label="Estimated repair cost",
    ),
    score=Levels(
        key="service",
        heading="Level of service",
        noun="level of service",
        frags=_t(
            "repairs are not covered and are charged at standard rates",
            "parts are covered but labour is charged",
            "parts and labour are both covered",
            "the product is replaced with a new unit",
        ),
        needles=_t(
            "charged at standard rates",
            "labour is charged",
            "parts and labour are both covered",
            "replaced with a new unit",
        ),
        criteria=_t(
            "Not covered: the repair is charged at standard rates",
            "Parts covered, labour charged",
            "Parts and labour covered",
            "Replacement with a new unit",
        ),
    ),
    questions={
        "choice": _t(
            "What warranty period applies to {name}'s product under the policy?",
            "Which warranty period should the {office} apply to this product?",
        ),
        "noul": _t(
            "Is the estimated repair cost within the amount the service centre may authorise without "
            "referral in this case?",
            "Can the service centre authorise this repair without referral, that is, is the estimate within "
            "the applicable limit?",
        ),
        "score": _t(
            "What level of service is {name} entitled to under the policy?",
            "Under the policy, which level of service applies to this warranty claim?",
        ),
    },
    extra_defs=(
        (
            "Authorised dealer",
            "a retailer appointed by {org} to sell and service its products",
        ),
        (
            "Defect",
            "a fault caused by the materials used or by the way the product was made",
        ),
        (
            "Proof of purchase",
            "a receipt or invoice showing the date and place of purchase",
        ),
        (
            "Consumable part",
            "a part that wears out in normal use, such as a filter, blade or battery",
        ),
    ),
    static=_t(
        "The warranty does not cover damage caused by misuse, accident or unauthorised repair.",
        "Consumable parts are covered only if they fail within the first month.",
        "Proof of purchase must be provided with every warranty claim.",
    ),
    distractors=(
        (
            "Registering a product",
            "Products can be registered on the portal using the serial number. Registration "
            "helps the {office} contact owners about safety notices.",
        ),
        (
            "Making a claim",
            "Claims are made on the portal or by telephone on extension {ext}. The customer describes "
            "the fault and uploads proof of purchase and, where useful, a photograph.",
        ),
        (
            "Collection and delivery",
            "Large products are collected from the customer's address within {wd} working "
            "days. Smaller products are sent to the service centre in packaging supplied by the {office}.",
        ),
        (
            "Repair times",
            "Most repairs are completed within {dd} days of the product arriving at the service centre. "
            "Customers can follow progress on the portal.",
        ),
        (
            "Loan units",
            "Where a repair is expected to take longer than usual, a loan unit may be offered while the "
            "customer's product is away.",
        ),
        (
            "Safety notices",
            "If a safety notice is issued for a product, owners are contacted directly and the "
            "necessary work is carried out free of charge.",
        ),
        (
            "Moving abroad",
            "Customers who take a product abroad should check whether it is compatible with the local "
            "power supply. Service abroad is provided through local partners where they exist.",
        ),
        (
            "Accessories",
            "Accessories bought separately carry their own warranty, which is stated on their packaging.",
        ),
        (
            "Data on devices",
            "Customers should back up any data stored on a product before sending it for repair. "
            "{org} cannot guarantee that stored data will be preserved.",
        ),
        (
            "Cleaning before collection",
            "Products should be clean and free of food, soil or chemicals when they are "
            "collected. Heavily soiled products may be returned uncleaned.",
        ),
        (
            "Care and maintenance",
            "Regular cleaning and the use of genuine spare parts help products last longer. "
            "Maintenance guides are available on the portal.",
        ),
        (
            "Spare parts",
            "Spare parts are available for at least {nn} years after a model is discontinued. Parts can "
            "be ordered from the portal or from authorised dealers.",
        ),
        (
            "Questions and answers",
            "Q: Can I repair the product myself? A: Doing so may affect the warranty. Q: Do I "
            "need the original packaging? A: No, but the product must be packed securely.",
        ),
        (
            "Environmental policy",
            "Faulty parts removed during repairs are recycled. Products that cannot be repaired "
            "are dismantled so that materials can be recovered.",
        ),
        (
            "Customer satisfaction",
            "After each repair, customers receive a short survey. The results are reviewed "
            "every {mm} months and used to improve the service.",
        ),
        (
            "Home visits",
            "Engineers visiting a home show identification on arrival. Customers may ask the {office} to "
            "confirm an engineer's identity before letting them in.",
        ),
        (
            "Second-hand products",
            "The warranty passes to a new owner if the product is sold, provided that the "
            "original proof of purchase is handed over.",
        ),
        (
            "Commercial use",
            "Products used for commercial purposes are serviced under separate business terms, which "
            "are available from the {office}.",
        ),
        (
            "Contact details",
            "The {office} is open from 8 a.m. to 8 p.m. on working days. The service manager for "
            "the region is {name1}.",
        ),
        (
            "Software updates",
            "Some products receive software updates. Installing them keeps the product secure and "
            "may add features.",
        ),
        (
            "Complaints about repairs",
            "A customer unhappy with a repair should contact the {office} within {dd} days, "
            "so that the product can be inspected again.",
        ),
    ),
)


# ------------------------------------------------------------ tuition_refund (dev-ood only)

TUITION = Domain(
    key="tuition_refund",
    titles=_t(
        "Tuition Fee Refund Policy",
        "Fees, Withdrawals and Refunds Regulations",
        "Student Fee Refund Policy",
        "Regulations on the Refund of Tuition Fees",
    ),
    orgs=_t(
        "Whitmore College",
        "Lakeshore Institute of Technology",
        "University of Carrowmore",
        "Brackenfield School of Art",
        "Merriton Open College",
    ),
    member="student",
    members="students",
    person="the student",
    claimant_label="Student",
    case="refund request",
    cases="refund requests",
    offices=_t("Student Fees Office", "Registry", "Student Finance team"),
    currency=_t("usd", "eur", "gbp"),
    purpose=_t(
        "These regulations explain when {members} of {org} who withdraw from their studies are entitled to a refund "
        "of tuition fees, and how much is kept back.",
        "{org} wants students who leave a course early to be treated fairly. This policy sets out how tuition fees "
        "are refunded on withdrawal.",
    ),
    scope_text=_t(
        "These regulations apply to every {member} who pays tuition fees to {org} directly. Fees paid by a sponsor "
        "are refunded to the sponsor under the same rules.",
        "The regulations cover tuition fees for all taught programmes. Accommodation charges and fees for "
        "short courses are dealt with separately.",
    ),
    scope=Scope(
        label="School",
        values=_t(
            "School of Engineering",
            "School of Health Sciences",
            "Business School",
            "School of Arts",
            "Language Centre",
            "School of Law",
        ),
        annex_title="Students of the {v}",
        annex_scope="{members} registered with the {v}",
        case=_t("{name} is registered with the {v}.", "{name} studies at the {v}."),
    ),
    conditions=(
        Condition(
            "credits",
            "min",
            "the student is registered for at least {t}",
            "Full-time Student",
            "a student registered for at least {t} in the academic year",
            "the student is a Full-time Student",
            _t(
                "{name} is registered for {v}.",
            ),
            "Credits registered",
            lo=10,
            hi=120,
            thresholds=(60, 75, 90),
            unit="{n} credits",
        ),
        Condition(
            "reason",
            "is",
            "the withdrawal is due to {t}",
            "Qualifying Circumstance",
            "{t}",
            "the withdrawal is due to a Qualifying Circumstance",
            _t(
                "The withdrawal is due to {v}.",
            ),
            "Reason for withdrawal",
            choices=_t(
                "a serious illness",
                "a family bereavement",
                "a military posting",
                "a change of employer",
                "financial hardship",
            ),
        ),
        Condition(
            "mode",
            "is",
            "the student is enrolled for {t}",
            "Campus Enrolment",
            "enrolment for {t}",
            "the student holds a Campus Enrolment",
            _t(
                "{name} is enrolled for {v}.",
            ),
            "Mode of study",
            choices=_t(
                "on-campus study",
                "distance learning",
                "a work-based placement",
                "an exchange programme",
            ),
        ),
        Condition(
            "weeks",
            "max",
            "the student attended no more than {t} of teaching",
            "Early Withdrawal",
            "a withdrawal after no more than {t} of teaching",
            "the withdrawal is an Early Withdrawal",
            _t(
                "{name} attended {v} of teaching.",
            ),
            "Teaching attended",
            lo=0,
            hi=14,
            thresholds=(2, 3, 4, 6),
            unit="{n} weeks",
        ),
    ),
    noise=(
        Noise("Student number", _t("ST{n}", "U{n}"), "{name}'s student number is {v}."),
        Noise(
            "Programme",
            _t(
                "BSc Civil Engineering",
                "BA Illustration",
                "MSc Public Health",
                "LLB Law",
                "MBA",
                "Diploma in Spanish",
            ),
            "{name} is enrolled on the {v} programme.",
        ),
    ),
    dates=DatePair(
        "withdrew",
        "Date of withdrawal",
        "the date on which the student withdraws",
        "{name} withdrew on {d}.",
        "requested",
        "Refund requested",
        "the date on which the refund is requested",
        "The refund was requested on {d}.",
        (5, 60),
    ),
    choice=Quantity(
        key="admin_fee",
        heading="Administration fee",
        noun="administration fee kept on withdrawal",
        unit="money",
        ladder=(50, 75, 100, 125, 150, 200, 250, 300, 400),
        option="{val}",
        frags=_t(
            "an administration fee of {val} is kept from any refund",
            "the administration fee kept is {val}",
            "{org} keeps an administration fee of {val}",
        ),
        base=_t(
            "Where a refund is made, {org} keeps an administration fee of {val} to cover the cost of processing "
            "the withdrawal.",
            "The administration fee kept on withdrawal is {val}.",
        ),
    ),
    noul=Quantity(
        key="refund_deadline",
        heading="Refund deadline",
        noun="refund deadline",
        unit="{n} days",
        ladder=(7, 10, 14, 21, 28, 35, 42, 56),
        option="{val} after teaching starts",
        frags=_t(
            "the refund deadline is {val} after teaching starts",
            "a withdrawal qualifies for a refund if it takes place within {val} of the start of teaching",
            "refunds are available for withdrawals made within {val} of the start of teaching",
        ),
        base=_t(
            "A withdrawal qualifies for a refund if it takes place within {val} of the start of teaching.",
            "The refund deadline is {val} after teaching starts.",
        ),
        ask=_t(
            "{name} withdrew {q} after teaching began.",
            "The withdrawal took place {q} after the start of " "teaching.",
        ),
        ask_label="Time from start of teaching to withdrawal",
    ),
    score=Levels(
        key="refund_band",
        heading="Refund rates",
        noun="refund rate",
        frags=_t(
            "no part of the tuition fee is refunded",
            "25% of the tuition fee is refunded",
            "50% of the tuition fee is refunded",
            "75% of the tuition fee is refunded",
            "the tuition fee is refunded in full",
        ),
        needles=_t(
            "no part of the tuition fee",
            "25% of the tuition fee",
            "50% of the tuition fee",
            "75% of the tuition fee",
            "refunded in full",
        ),
        criteria=_t(
            "No refund of the tuition fee",
            "A refund of 25% of the tuition fee",
            "A refund of 50% of the tuition fee",
            "A refund of 75% of the tuition fee",
            "A full refund of the tuition fee",
        ),
    ),
    questions={
        "choice": _t(
            "What administration fee does {org} keep from {name}'s refund under the regulations?",
            "Which administration fee applies to this withdrawal?",
        ),
        "noul": _t(
            "Did {name} withdraw within the refund deadline that applies?",
            "Does the withdrawal fall within the refund deadline that applies to this case?",
        ),
        "score": _t(
            "What share of the tuition fee is refunded to {name} under the regulations?",
            "Under the regulations, which refund rate applies to this request?",
        ),
    },
    extra_defs=(
        (
            "Academic year",
            "the period from the first day of teaching in the autumn to the last day of the summer term",
        ),
        (
            "Sponsor",
            "an employer or other body that has agreed in writing to pay a student's fees",
        ),
        (
            "Withdrawal",
            "the formal ending of a student's registration, confirmed by the Registry",
        ),
        (
            "Tuition fee",
            "the fee charged for teaching, excluding accommodation and other charges",
        ),
    ),
    static=_t(
        "Refunds are paid to the person or body that paid the fee.",
        "A student who suspends studies rather than withdrawing keeps their fee credit for the following year.",
        "Refunds are made within thirty days of the Registry confirming the withdrawal.",
    ),
    distractors=(
        (
            "How to withdraw",
            "A student who wishes to withdraw completes the withdrawal form on the portal and "
            "discusses the decision with their personal tutor. The Registry confirms the withdrawal in writing.",
        ),
        (
            "Suspending studies",
            "Students facing temporary difficulties may be able to suspend their studies for up to "
            "{nn} terms instead of withdrawing. Advice is available from the student support service.",
        ),
        (
            "International students",
            "Students who hold a student visa should speak to the international advice team "
            "before withdrawing, because withdrawal is reported to the immigration authorities.",
        ),
        (
            "Student loans",
            "Where fees are paid by a student loan, the loan company is told of the withdrawal and any "
            "refund is returned to it.",
        ),
        (
            "Accommodation",
            "Students living in college accommodation give notice under their accommodation contract. "
            "Accommodation charges are not part of this policy.",
        ),
        (
            "Library and IT accounts",
            "Library and IT accounts are closed {wd} working days after a withdrawal is "
            "confirmed. Students should save any personal files before then.",
        ),
        (
            "Transcripts",
            "Students who withdraw receive a transcript of the credits they have completed. The "
            "transcript can be used to transfer to another institution.",
        ),
        (
            "Returning later",
            "A student who withdraws may apply to return in a later year. Credits already completed "
            "may be counted, subject to the programme's rules.",
        ),
        (
            "Support services",
            "Counselling, financial advice and disability support remain available to students for "
            "{dd} days after withdrawal.",
        ),
        (
            "Bursaries and scholarships",
            "Bursaries and scholarships stop from the date of withdrawal. Instalments "
            "already paid are not normally recovered.",
        ),
        (
            "Graduation ceremonies",
            "Students who complete an interim award before withdrawing may attend the next "
            "graduation ceremony for that award.",
        ),
        (
            "Questions and answers",
            "Q: Can I change programme instead of withdrawing? A: Yes, if a place is available. "
            "Q: Will my sponsor be told? A: Yes, sponsors are told when a withdrawal is confirmed.",
        ),
        (
            "Appeals about academic decisions",
            "Decisions about progression and assessment are dealt with under the "
            "academic appeals procedure, not under these regulations.",
        ),
        (
            "Payment plans",
            "Students paying by instalments continue to receive statements until the account is settled. "
            "The Student Fees Office can agree a revised plan where needed.",
        ),
        (
            "Placement years",
            "Students on a placement year pay a reduced fee. Placement providers are informed if a "
            "student withdraws during a placement.",
        ),
        (
            "Equipment loans",
            "Laptops and equipment lent by {org} must be returned within {wd} working days of "
            "withdrawal.",
        ),
        (
            "Student identity cards",
            "Identity cards should be returned to the Registry on withdrawal. Access to "
            "buildings stops on the date of withdrawal.",
        ),
        (
            "Data held about students",
            "{org} keeps the student record for {mm} years after withdrawal so that "
            "transcripts and references can be provided.",
        ),
        (
            "Contacts",
            "The {office} can be reached on extension {ext} or at the student hub. The fees adviser is "
            "{name1}.",
        ),
        (
            "Student union",
            "The student union's advice centre offers free, confidential help to students thinking "
            "about withdrawal.",
        ),
        (
            "Council tax",
            "Students who withdraw may become liable for council tax. The local authority may ask for "
            "confirmation of the withdrawal date.",
        ),
    ),
)


# ------------------------------------------------------------ venue_booking (dev-ood only)

VENUE = Domain(
    key="venue_booking",
    titles=_t(
        "Venue Hire Policy",
        "Conditions of Hire for Community Venues",
        "Room and Venue Booking Policy",
        "Terms of Hire for Halls and Meeting Rooms",
    ),
    orgs=_t(
        "Castlegate Civic Centre",
        "Millbrook Community Halls",
        "Saltmarsh Arts Centre",
        "Greyfriars Conference Venues",
        "Oldbury Town Hall Trust",
    ),
    member="hirer",
    members="hirers",
    person="the hirer",
    claimant_label="Hirer",
    case="booking",
    cases="bookings",
    offices=_t("Bookings Office", "Venue Management team", "Events Office"),
    currency=_t("usd", "eur", "gbp"),
    purpose=_t(
        "This policy sets out the terms on which {org} hires its rooms and halls to {members}, including deposits, "
        "capacity limits and charges.",
        "{org} wants its venues to be used as widely as possible and left in good condition. This document explains "
        "the conditions of hire.",
    ),
    scope_text=_t(
        "This policy applies to every {case} of a room or hall managed by {org}, whether for a private, community "
        "or commercial event.",
        "The policy covers all hires of the venues listed in it. Filming and outdoor events are dealt with under "
        "separate terms.",
    ),
    scope=Scope(
        label="Venue",
        values=_t(
            "Main Hall",
            "Riverside Room",
            "Garden Pavilion",
            "Studio Theatre",
            "Committee Suite",
            "Long Gallery",
        ),
        annex_title="The {v}",
        annex_scope="bookings of the {v}",
        case=_t("The booking is for the {v}.", "{name} has booked the {v}."),
    ),
    conditions=(
        Condition(
            "hirer",
            "is",
            "the hirer is {t}",
            "Community Hirer",
            "a hirer that is {t}",
            "the hirer is a Community Hirer",
            _t(
                "The hirer is {v}.",
            ),
            "Type of hirer",
            choices=_t(
                "a registered charity",
                "a local school",
                "a commercial organiser",
                "a residents' association",
                "a private individual",
            ),
        ),
        Condition(
            "duration",
            "min",
            "the booking lasts at least {t}",
            "Extended Booking",
            "a booking lasting at least {t}",
            "the booking is an Extended Booking",
            _t(
                "The booking lasts {v}.",
            ),
            "Length of booking",
            lo=1,
            hi=14,
            thresholds=(4, 5, 6, 8),
            unit="{n} hours",
        ),
        Condition(
            "previous",
            "min",
            "the hirer has made at least {t} with {org}",
            "Regular Hirer",
            "a hirer who has made at least {t} with {org}",
            "the hirer is a Regular Hirer",
            _t(
                "{name} has made {v} with {org}.",
            ),
            "Previous bookings",
            lo=0,
            hi=30,
            thresholds=(3, 5, 8, 10),
            unit="{n} previous bookings",
        ),
        Condition(
            "event",
            "is",
            "the event is {t}",
            "Designated Event",
            "an event that is {t}",
            "the event is a Designated Event",
            _t(
                "The event is {v}.",
            ),
            "Type of event",
            choices=_t(
                "a wedding reception",
                "a public meeting",
                "a concert",
                "a children's party",
                "a trade fair",
            ),
        ),
    ),
    noise=(
        Noise(
            "Booking reference", _t("BK-{n}", "VH-{n}"), "The booking reference is {v}."
        ),
        Noise(
            "Catering",
            _t("the in-house caterer", "an approved outside caterer", "no catering"),
            "Catering will be provided by {v}.",
        ),
    ),
    dates=DatePair(
        "confirmed",
        "Booking confirmed",
        "the date on which the booking is confirmed",
        "The booking was confirmed on {d}.",
        "event",
        "Date of event",
        "the date of the event",
        "The event takes place on {d}.",
        (10, 150),
    ),
    choice=Quantity(
        key="deposit",
        heading="Deposits",
        noun="security deposit",
        unit="money",
        ladder=(100, 150, 200, 250, 300, 400, 500, 750, 1000),
        option="{val}",
        frags=_t(
            "the security deposit is {val}",
            "a security deposit of {val} is payable",
            "hirers pay a security deposit of {val}",
        ),
        base=_t(
            "A security deposit of {val} is payable when the booking is confirmed and is returned after the "
            "event if the venue is left in good order.",
            "The security deposit is {val} for each booking.",
        ),
    ),
    noul=Quantity(
        key="capacity",
        heading="Capacity",
        noun="maximum attendance",
        unit="{n} people",
        ladder=(60, 80, 100, 120, 150, 180, 200, 250, 300),
        option="{val}",
        frags=_t(
            "the maximum attendance is {val}",
            "no more than {val} may attend",
            "attendance is limited to {val}",
        ),
        base=_t(
            "For safety reasons no more than {val} may attend an event, including performers and staff.",
            "The maximum attendance is {val}.",
        ),
        ask=_t("{name} expects {q} to attend.", "The expected attendance is {q}."),
        ask_label="Expected attendance",
    ),
    score=Levels(
        key="staffing",
        heading="Staffing charges",
        noun="staffing charge",
        frags=_t(
            "no staffing charge applies",
            "the basic staffing charge applies",
            "the standard staffing charge applies",
            "the enhanced staffing charge applies",
        ),
        needles=_t(
            "no staffing charge",
            "basic staffing charge",
            "standard staffing charge",
            "enhanced staffing charge",
        ),
        criteria=_t(
            "No staffing charge",
            "The basic staffing charge",
            "The standard staffing charge",
            "The enhanced staffing charge",
        ),
    ),
    questions={
        "choice": _t(
            "What security deposit applies to {name}'s booking under the policy?",
            "Which security deposit should the {office} ask for on this booking?",
        ),
        "noul": _t(
            "Is the expected attendance at {name}'s event within the maximum that applies?",
            "Can the {office} accept the booking with the expected attendance, that is, is it within the "
            "applicable maximum?",
        ),
        "score": _t(
            "Which staffing charge applies to {name}'s booking under the policy?",
            "Under the policy, what staffing charge does this booking attract?",
        ),
    },
    extra_defs=(
        ("Hire period", "the time booked, including setting up and clearing away"),
        (
            "Duty manager",
            "the member of staff responsible for the building during a hire",
        ),
        (
            "Licensable activity",
            "an activity for which a premises licence is required by law",
        ),
        ("Hire fee", "the charge for the use of a room, excluding deposits and extras"),
    ),
    static=_t(
        "The hirer must be at least eighteen years old and must be present throughout the event.",
        "Decorations must not be fixed to walls or ceilings except on the hanging rails provided.",
        "Music must end by eleven o'clock at night in all venues.",
    ),
    distractors=(
        (
            "Making a booking",
            "Bookings are made on the portal or through the {office}. A booking is provisional "
            "until the booking form has been signed and the hire fee paid.",
        ),
        (
            "Setting up and clearing away",
            "Hirers must include time for setting up and clearing away in the hire "
            "period. Rooms must be left as they were found, with chairs and tables stacked.",
        ),
        (
            "Fire safety",
            "Fire exits must be kept clear at all times. The hirer must know the location of the fire "
            "alarm points and the assembly point, which are shown on the plan in each room.",
        ),
        (
            "First aid",
            "A first-aid kit is kept in each venue. Accidents must be recorded in the accident book held by "
            "the duty manager.",
        ),
        (
            "Licensing",
            "Where an event includes a licensable activity, the hirer must check with the {office} that the "
            "venue's licence covers it.",
        ),
        (
            "Insurance",
            "{org} insures its buildings. Hirers are responsible for insuring their own equipment and, for "
            "commercial events, for holding public liability insurance.",
        ),
        (
            "Noise and neighbours",
            "Hirers must keep noise to a reasonable level and ask guests to leave quietly. "
            "Complaints from neighbours may affect future bookings.",
        ),
        (
            "Parking",
            "Parking at most venues is limited. Guests should be encouraged to use public transport or the "
            "public car parks listed on the portal.",
        ),
        (
            "Catering",
            "Food may be supplied by the in-house caterer or by an approved outside caterer. Kitchens must be "
            "left clean at the end of the hire.",
        ),
        (
            "Accessibility",
            "All venues have step-free access and an accessible toilet. Hearing loops are installed in "
            "the larger rooms.",
        ),
        (
            "Equipment",
            "Projectors, microphones and staging can be booked with the room. Equipment must be returned "
            "to the duty manager in working order.",
        ),
        (
            "Keys and access",
            "The duty manager opens and closes the building. Keys are not issued to hirers.",
        ),
        (
            "Children's events",
            "At events for children, the hirer must provide enough adult supervisors and must "
            "follow the safeguarding guidance issued by {org}.",
        ),
        (
            "Questions and answers",
            "Q: Can we bring our own sound system? A: Yes, if it has been safety tested. Q: Can "
            "we use candles? A: Only battery-operated candles are allowed.",
        ),
        (
            "Smoking",
            "Smoking and vaping are not allowed inside any venue or within {nn} metres of the entrances.",
        ),
        (
            "Lost property",
            "Lost property is kept for {dd} days. Hirers should check the rooms before leaving.",
        ),
        (
            "Damage",
            "The hirer must report any damage to the duty manager before leaving. Repairs are arranged by "
            "{org}.",
        ),
        (
            "Charitable discounts",
            "Registered charities may be eligible for discounted hire fees. Details are "
            "available from the {office}.",
        ),
        (
            "Contacts",
            "The {office} is open from 9 a.m. to 5 p.m. on working days and can be reached on extension "
            "{ext}. The venue coordinator is {name1}.",
        ),
        (
            "Environmental practice",
            "Hirers are asked to recycle waste using the bins provided and to avoid single-use "
            "plastics where possible.",
        ),
        (
            "Regular hirings",
            "Groups that meet weekly may book a series of dates up to {mm} months ahead.",
        ),
    ),
)


DOMAINS: dict[str, Domain] = {
    pack.key: pack
    for pack in (
        TRAVEL,
        RETURNS,
        LEAVE,
        LIBRARY,
        ITACCESS,
        GRANTS,
        PARKING,
        WARRANTY,
        TUITION,
        VENUE,
    )
}
