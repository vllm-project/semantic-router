# MASSIVE v2 six-language blind review pilot

This is a **preregistered data audit**, not a training run. It uses only the
English-filtered, still-unapproved MASSIVE v2 TRAIN candidate: manifest SHA-256
`193b68b92f588f1d1e13352ff4752eab9048fbdd37e1f0ea88d1b514b5431a10`,
TRAIN SHA-256
`55d7d711a130165be5a1d43930c836c7356938c33577e50fa22fdc937d976b80`.
The input has 481 complete seven-locale source groups and 59 retained intents.
Its official DEV, MASSIVE TEST, and JevArena FINAL are not used here.

Before any new localized semantic verdict is inspected, select exactly one
source group from each of these **18 fixed intents**, for 18 independent groups
and all six non-English translations per group: **108 English/localized pairs**,
18 per locale. `cooking_query` cannot enter because no English-approved v2
source group remains. The fixed strata are:

| Stratum | Six intent labels | Reason |
| --- | --- | --- |
| Sparse | `audio_volume_other`, `general_quirky`, `transport_query`, `music_dislikeness`, `recommendation_movies`, `iot_hue_lighton` | Few retained source groups; one mistranslation has high leverage. |
| Prior risk | `datetime_convert`, `qa_definition`, `play_podcasts`, `social_post`, `transport_ticket`, `recommendation_events` | Initial review identified a drift, ambiguity, or missing detail near these intents. |
| Broad scenarios | `alarm_set`, `email_sendemail`, `weather_query`, `lists_query`, `takeaway_order`, `iot_hue_lightoff` | Spread the pilot across routine domains and speech acts. |

Within each intent, choose the source ID with the smallest SHA-256 rank of
`decision2-massive-v2-locale-pilot-18-groups-v1\0sample\0<intent>\0<source_id>`;
ties use the source ID. There is no replacement after seeing review outcomes.
The builder fails if an intent has no retained group, any selected group lacks
one of the seven locales, its six options or label drift across translations,
or any input hash changes.

The private blind packet contains an opaque review ID and parallel-group token,
the English and localized utterances, both instructions, the locale, and the
**exact shared six option descriptions**. It contains no source ID, source
intent, label, answer key, or selection stratum. A separate private key holds
those fields. The reviewer should record whether the translation preserves the
English meaning and which option it supports **before** the key is opened.
Any adverse result quarantines the entire seven-locale source group pending
adjudication. The packet and its independent review do not themselves set
`training_approved=true`.

```bash
python -m training.data.build_massive_locale_review \
  --candidate /private/decision20/data/massive_multilingual_review_filtered_v2 \
  --output /private/decision20/reviews/massive-v2-locale-pilot-v1
python -m unittest training.data.tests.test_massive_locale_review
```

This pilot samples only 18 of 481 retained groups. It can discover systematic
or high-leverage localization problems, but a clean pilot would not certify
all six languages or the remaining 463 groups. The original MASSIVE CC BY 4.0
LICENSE and NOTICE, including SLURP attribution, remain with the private
packet. No raw source row belongs in Git or a public progress gist.
