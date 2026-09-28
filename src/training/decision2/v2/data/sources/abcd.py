"""ABCD v1.1 TRAIN as A1 rows: the customer's subflow (Choice) and whether the
issue belongs to a named flow (Noul).

The state is the customer/agent transcript; `action` turns name system
operations and are dropped. Subflow descriptions are an explicit table checked
against the ontology. TRAIN labels the two FAQ flows per question
(`boots_how_1`), and the guideline question lists are not always in label
order, so those entries follow what the TRAIN conversations of each label ask.
"""

from __future__ import annotations

import collections
import gzip
import json
from pathlib import Path
from typing import Any

from training.model.data import file_sha256

from v2.data.sources.common import choice_options, make_row, noul_options, rotate, sha

ARM = "a1"
SOURCE = "abcd_v1.1_train"
CHOICE_FAMILY = "abcd_subflow"
NOUL_FAMILY = "abcd_flow"
SEED = "a1-abcd-v1"
CAPS = {CHOICE_FAMILY: 1500, NOUL_FAMILY: 1500}
DATA = "data/abcd_v1.1.json.gz"
ONTOLOGY = "data/ontology.json"
GUIDELINES = "data/guidelines.json"
CHOICE_INSTRUCTIONS = "Which request is the customer making in this conversation?"
NOUL_INSTRUCTIONS = "Is the customer's issue about {}?"
SPEAKERS = {"customer": "Customer", "agent": "Agent"}
FAQ_FLOWS = ("single_item_query", "storewide_query")
# TRAIN never uses the ontology name `status_active`; its conversations carry `status_questions`.
LABEL_ALIASES = {"status_questions": "status_active"}

FLOW_NAMES = {
    "account_access": "Account Access",
    "manage_account": "Manage Account",
    "order_issue": "Order Issue",
    "product_defect": "Product Defect",
    "purchase_dispute": "Purchase Dispute",
    "shipping_issue": "Shipping Issue",
    "single_item_query": "Single-Item Query",
    "storewide_query": "Storewide Query",
    "subscription_inquiry": "Subscription Inquiry",
    "troubleshoot_site": "Troubleshoot Site",
}

# Applied only on an exact match: a typo, ungrammatical text, and "sweater" for the jacket product.
DESCRIPTION_FIXES = {
    "check our update a shipment of an item": "checking or updating a shipment of an item",
    "get status of an order or change an order, possibly shipping": (
        "getting the status of an order or changing an order, possibly its shipping"
    ),
    "FAQ questions about jeans, boots, shirt or sweater": (
        "FAQ questions about jeans, boots, shirts or jackets"
    ),
}

SUBFLOWS: dict[str, dict[str, str]] = {
    "account_access": {
        "recover_username": "Recover a forgotten username",
        "recover_password": "Recover a forgotten password",
        "reset_2fa": "Reset two-factor authentication",
    },
    "manage_account": {
        "status_service_added": "Ask why an unfamiliar service was added to the account",
        "status_service_removed": "Ask why a service was removed from the account",
        "status_shipping_question": "Ask whether the account includes free international shipping",
        "status_credit_missing": "Report credit that is missing from the account",
        "manage_change_address": "Change the address on the account",
        "manage_change_name": "Change the name on the account",
        "manage_change_phone": "Change the phone number on the account",
        "manage_payment_method": "Change the payment method on the account",
    },
    "order_issue": {
        "status_mystery_fee": "Ask about an unexpected fee on an order",
        "status_delivery_time": "Fix the delivery time of an order",
        "status_payment_method": "Fix the payment method used for an order",
        "status_quantity": "Fix the quantity of an item in an order",
        "manage_upgrade": "Upgrade an order to overnight shipping",
        "manage_downgrade": "Downgrade an order to slower, cheaper shipping",
        "manage_create": "Add an item to an existing order",
        "manage_cancel": "Cancel an item that was ordered by mistake",
    },
    "product_defect": {
        "refund_initiate": "Start a refund",
        "refund_update": "Add an item to an existing refund",
        "refund_status": "Check the status of a refund",
        "return_stain": "Return an item because of a stain",
        "return_color": "Return an item because of its color",
        "return_size": "Return an item because of its size",
    },
    "purchase_dispute": {
        "bad_price_competitor": "Complain that a competitor offers a lower price",
        "bad_price_yesterday": "Complain that the price was lower yesterday",
        "out_of_stock_general": "Complain that items are out of stock",
        "out_of_stock_one_item": "Complain that one particular item is out of stock",
        "promo_code_invalid": "Complain that a promo code is invalid",
        "promo_code_out_of_date": "Complain that a promo code has expired",
        "mistimed_billing_already_returned": "Dispute a bill for an item that was already returned",
        "mistimed_billing_never_bought": "Dispute a bill for something that was never bought",
    },
    "shipping_issue": {
        "status": "Check on the status of a shipment",
        "manage": "Change the address or the item of a shipment",
        "missing": "Report an item that never arrived",
        "cost": "Get the shipping fee waived or refunded",
    },
    "single_item_query": {
        "boots_how_1": "Ask how to remove a paint stain from the boots",
        "boots_how_2": "Ask how wide the boots are",
        "boots_how_3": "Ask how to remove gum from the boots",
        "boots_how_4": "Ask how long the boots take to wear in",
        "boots_other_1": "Ask whether the boots are waterproof",
        "boots_other_2": "Ask what color the laces of the boots are",
        "boots_other_3": "Ask whether the boots are in stock in a certain size",
        "boots_other_4": "Ask whether the boots come with a warranty",
        "shirt_how_1": "Ask how to remove a stain from the shirt",
        "shirt_how_2": "Ask how to wash the shirt",
        "shirt_how_3": "Ask about the arm length of the shirt",
        "shirt_how_4": "Ask how wide the collar of the shirt is",
        "shirt_other_1": "Ask whether the shirt shrinks after washing",
        "shirt_other_2": "Ask whether the shirt is in stock in a certain size",
        "shirt_other_3": "Ask whether the buttons on the shirt are brown or black",
        "shirt_other_4": "Ask what material the shirt is made of",
        "jeans_how_1": "Ask how to remove a grass stain from the jeans",
        "jeans_how_2": "Ask how often the jeans need to be washed",
        "jeans_how_3": "Ask about the leg length of the jeans",
        "jeans_how_4": "Ask how much it costs to get the jeans tailored",
        "jeans_other_1": "Ask whether the jeans shrink after washing",
        "jeans_other_2": "Ask whether the jeans are dark blue or black",
        "jeans_other_3": "Ask whether the jeans come in a larger or smaller size",
        "jeans_other_4": "Ask whether the jeans come in a design with ripped holes",
        "jacket_how_1": "Ask how to remove a wine stain from the jacket",
        "jacket_how_2": "Ask whether the jacket is machine washable or dry-clean only",
        "jacket_how_3": "Ask how often the jacket needs to be washed",
        "jacket_how_4": "Ask how to detach the hood of the jacket",
        "jacket_other_1": "Ask whether the jacket shrinks after washing",
        "jacket_other_2": "Ask whether the jacket is in stock in a certain size",
        "jacket_other_3": "Ask whether the jacket is warm enough for windy weather",
        "jacket_other_4": "Ask what material the jacket is made of",
    },
    "storewide_query": {
        "pricing_1": "Ask how much gift wrapping costs",
        "pricing_2": "Ask how much it costs to have a name stitched on an item",
        "pricing_3": "Ask how much overnight shipping costs",
        "pricing_4": "Ask whether the shopping cart qualifies for free shipping",
        "membership_1": "Ask what the membership levels are",
        "membership_2": "Ask how to qualify for premium membership",
        "membership_3": "Ask about the benefits of membership",
        "membership_4": "Ask how long a membership lasts",
        "timing_1": "Ask when the new seasonal collection comes out",
        "timing_2": "Ask about the opening hours of the local store",
        "timing_3": "Ask when the annual sale takes place",
        "timing_4": "Ask how long promo codes stay valid",
        "policy_1": "Ask about the return policy",
        "policy_2": "Ask about the refund policy",
        "policy_3": "Ask how to cancel a subscription",
        "policy_4": "Ask what happens when a subscription payment is late",
    },
    "subscription_inquiry": {
        "status_active": "Check whether the subscription is still active",
        "status_due_amount": "Ask how much is due on the subscription",
        "status_due_date": "Ask when the subscription payment is due",
        "manage_pay_bill": "Pay or renew the subscription",
        "manage_extension": "Ask for more time to pay for the subscription",
        "manage_dispute_bill": "Dispute a subscription bill, such as a double charge",
    },
    "troubleshoot_site": {
        "credit_card": "Fix a credit card that keeps being rejected",
        "shopping_cart": "Fix a shopping cart that is not updating",
        "search_results": "Fix a site search that returns no results",
        "slow_speed": "Fix a website that is too slow",
    },
}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def transcript(turns: list[list[str]]) -> str:
    lines = []
    for speaker, text in turns:
        if speaker == "action":
            continue
        if speaker not in SPEAKERS:
            raise ValueError(f"unknown ABCD speaker {speaker!r}")
        text = " ".join(text.split())
        if text:
            lines.append(f"{SPEAKERS[speaker]}: {text}")
    return "\n".join(lines)


def ontology_name(flow: str, label: str) -> str:
    return label.split("_", 1)[0] if flow in FAQ_FLOWS else label


def check_ontology(ontology: dict[str, Any]) -> list[str]:
    intents = ontology["intents"]
    if sorted(intents["flows"]) != sorted(SUBFLOWS):
        raise ValueError("ABCD ontology flows differ from the subflow table")
    for flow, labels in SUBFLOWS.items():
        if set(intents["subflows"][flow]) != {
            ontology_name(flow, label) for label in labels
        }:
            raise ValueError(f"ABCD ontology subflows of {flow} differ from the table")
    return sorted(SUBFLOWS)


def flow_topics(guidelines: dict[str, Any]) -> tuple[dict[str, str], list[str]]:
    topics, fixed = {}, []
    for flow in sorted(FLOW_NAMES):
        text = " ".join(guidelines[FLOW_NAMES[flow]]["description"].split())
        if text in DESCRIPTION_FIXES:
            text = DESCRIPTION_FIXES[text]
            fixed.append(flow)
        topics[flow] = text
    if len({text.lower() for text in topics.values()}) != len(topics):
        raise ValueError("ABCD flow descriptions are not distinct")
    return topics, fixed


def choice_flows(
    table: dict[str, dict[str, str]] = SUBFLOWS,
) -> tuple[list[str], dict[str, str]]:
    eligible, skipped = [], {}
    for flow, labels in sorted(table.items()):
        if len(labels) < 3:
            skipped[flow] = "fewer_than_two_siblings"
        elif len({text.lower() for text in labels.values()}) != len(labels):
            skipped[flow] = "indistinct_descriptions"
        else:
            eligible.append(flow)
    return eligible, skipped


def _choice_row(descriptions: list[str], gold: int, **fields: Any) -> dict[str, Any]:
    probe = make_row(options=choice_options(descriptions), label=gold, **fields)
    options, label = rotate(probe["options"], gold, f"{fields['arm']}-v1:{probe['id']}")
    return make_row(options=options, label=label, **fields)


def build(root: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    inputs = {name: file_sha256(root / name) for name in (DATA, ONTOLOGY, GUIDELINES)}
    splits = json.loads(gzip.decompress((root / DATA).read_bytes()))
    conversations = splits.pop("train")
    ignored = sorted(splits)
    del splits
    flows = check_ontology(_json(root / ONTOLOGY))
    topics, fixed = flow_topics(_json(root / GUIDELINES))
    eligible, skipped = choice_flows()
    rows: dict[str, list[dict[str, Any]]] = {CHOICE_FAMILY: [], NOUL_FAMILY: []}
    dropped: dict[str, collections.Counter[str]] = {
        family: collections.Counter() for family in rows
    }
    aliased: collections.Counter[str] = collections.Counter()
    seen: set[str] = set()
    for conversation in conversations:
        local_id = str(conversation["convo_id"])
        if local_id in seen:
            raise ValueError(f"duplicate ABCD convo_id {local_id}")
        seen.add(local_id)
        flow = conversation["scenario"]["flow"]
        subflow = conversation["scenario"]["subflow"]
        state = transcript(conversation["original"])
        if flow not in SUBFLOWS or not state:
            for counter in dropped.values():
                counter[
                    "unknown_flow" if flow not in SUBFLOWS else "empty_transcript"
                ] += 1
            continue
        shared = {
            "arm": ARM,
            "source": SOURCE,
            "language": "en",
            "group_key": local_id,
            "local_id": local_id,
            "state": state,
        }
        label = LABEL_ALIASES.get(subflow, subflow)
        if flow not in eligible:
            dropped[CHOICE_FAMILY][f"flow_{skipped[flow]}"] += 1
        elif label not in SUBFLOWS[flow]:
            dropped[CHOICE_FAMILY]["subflow_not_in_ontology"] += 1
        else:
            if label != subflow:
                aliased[f"{subflow}->{label}"] += 1
            names = list(SUBFLOWS[flow])
            rows[CHOICE_FAMILY].append(
                _choice_row(
                    [SUBFLOWS[flow][name] for name in names],
                    names.index(label),
                    family=CHOICE_FAMILY,
                    task_type="choice",
                    instructions=CHOICE_INSTRUCTIONS,
                    render_template=f"{CHOICE_FAMILY}/v1",
                    audit={
                        "convo_id": conversation["convo_id"],
                        "flow": flow,
                        "subflow": subflow,
                    },
                    **shared,
                )
            )
        truth = int(sha(f"{SEED}:truth:{local_id}"), 16) % 2
        asked = (
            flow
            if truth
            else min(
                (other for other in flows if other != flow),
                key=lambda other: sha(f"{SEED}:false-flow:{local_id}:{other}"),
            )
        )
        rows[NOUL_FAMILY].append(
            make_row(
                family=NOUL_FAMILY,
                task_type="noul",
                instructions=NOUL_INSTRUCTIONS.format(topics[asked]),
                options=noul_options("en"),
                label=truth,
                render_template=f"{NOUL_FAMILY}/v1",
                audit={
                    "convo_id": conversation["convo_id"],
                    "gold_flow": flow,
                    "asked_flow": asked,
                },
                **shared,
            )
        )
    return rows, {
        "inputs": inputs,
        "splits_ignored": ignored,
        "dropped": {
            family: dict(sorted(counter.items())) for family, counter in dropped.items()
        },
        "label_aliases": dict(sorted(aliased.items())),
        "flow_description_fixes": fixed,
        "choice_flows_skipped": skipped,
    }
