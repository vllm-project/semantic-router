"""Small, author reviewed, controlled multilingual development cases.

The seven versions of each case are one semantic group. This is a diagnostic
pilot, not a claim of native speaker review or a production benchmark.
"""

LANGUAGES = ("en", "zh", "es", "fr", "de", "ja", "ar")

CHOICE_INSTRUCTION = {
    "en": "Which action does the customer's message request?",
    "zh": "客户的消息请求哪项操作？",
    "es": "¿Qué acción solicita el mensaje del cliente?",
    "fr": "Quelle action le message du client demande-t-il ?",
    "de": "Welche Handlung verlangt die Nachricht des Kunden?",
    "ja": "顧客のメッセージはどの対応を求めていますか。",
    "ar": "ما الإجراء الذي تطلبه رسالة العميل؟",
}

CHOICE_OPTIONS = {
    "en": (
        "Track the parcel",
        "Cancel the order",
        "Change the delivery address",
        "Request a refund",
    ),
    "zh": ("查询包裹状态", "取消订单", "更改收货地址", "申请退款"),
    "es": (
        "Consultar el estado del paquete",
        "Cancelar el pedido",
        "Cambiar la dirección de entrega",
        "Solicitar un reembolso",
    ),
    "fr": (
        "Suivre le colis",
        "Annuler la commande",
        "Changer l'adresse de livraison",
        "Demander un remboursement",
    ),
    "de": (
        "Das Paket verfolgen",
        "Die Bestellung stornieren",
        "Die Lieferadresse ändern",
        "Eine Rückerstattung beantragen",
    ),
    "ja": (
        "荷物を追跡する",
        "注文をキャンセルする",
        "配送先の住所を変更する",
        "返金を求める",
    ),
    "ar": ("تتبّع الطرد", "إلغاء الطلب", "تغيير عنوان التسليم", "طلب استرداد المبلغ"),
}

# Semantic option index: track=0, cancel=1, address=2, refund=3.
CHOICE_CASES = (
    (
        "parcel_status",
        0,
        {
            "en": "Where is my parcel? Tell me its current delivery status.",
            "zh": "我的包裹在哪里？请告诉我目前的配送状态。",
            "es": "¿Dónde está mi paquete? Dígame su estado actual de entrega.",
            "fr": "Où est mon colis ? Dites-moi son statut de livraison actuel.",
            "de": "Wo ist mein Paket? Bitte teilen Sie mir den aktuellen Lieferstatus mit.",
            "ja": "荷物はどこにありますか。現在の配送状況を教えてください。",
            "ar": "أين طردي؟ أخبرني بحالة التوصيل الحالية.",
        },
    ),
    (
        "cancel_order",
        1,
        {
            "en": "Please cancel my order before it is sent.",
            "zh": "请在订单发出前取消它。",
            "es": "Cancele mi pedido antes de que lo envíen, por favor.",
            "fr": "Veuillez annuler ma commande avant son expédition.",
            "de": "Bitte stornieren Sie meine Bestellung, bevor sie versendet wird.",
            "ja": "発送される前に注文をキャンセルしてください。",
            "ar": "يرجى إلغاء طلبي قبل إرساله.",
        },
    ),
    (
        "new_address",
        2,
        {
            "en": "Please send my order to my new address instead of the old one.",
            "zh": "请把订单送到我的新地址，而不是旧地址。",
            "es": "Envíen mi pedido a mi nueva dirección en lugar de la anterior.",
            "fr": "Veuillez envoyer ma commande à ma nouvelle adresse, pas à l'ancienne.",
            "de": "Bitte senden Sie meine Bestellung an meine neue Adresse statt an die alte.",
            "ja": "注文品は古い住所ではなく新しい住所に送ってください。",
            "ar": "أرسلوا طلبي إلى عنواني الجديد بدلاً من العنوان القديم.",
        },
    ),
    (
        "money_back",
        3,
        {
            "en": "The item arrived broken. I want my money back.",
            "zh": "商品送来时已经损坏了。我想退回付款。",
            "es": "El artículo llegó roto. Quiero que me devuelvan el dinero.",
            "fr": "L'article est arrivé cassé. Je veux récupérer mon argent.",
            "de": "Der Artikel kam beschädigt an. Ich möchte mein Geld zurück.",
            "ja": "届いた商品が壊れていました。返金してほしいです。",
            "ar": "وصلت السلعة مكسورة. أريد استرداد مالي.",
        },
    ),
    (
        "not_cancel_track",
        0,
        {
            "en": "Do not cancel the order. I only want to know where the parcel is.",
            "zh": "不要取消订单。我只想知道包裹在哪里。",
            "es": "No cancelen el pedido. Solo quiero saber dónde está el paquete.",
            "fr": "N'annulez pas la commande. Je veux seulement savoir où est le colis.",
            "de": "Stornieren Sie die Bestellung nicht. Ich möchte nur wissen, wo das Paket ist.",
            "ja": "注文をキャンセルしないでください。荷物がどこにあるかだけ知りたいです。",
            "ar": "لا تلغوا الطلب. أريد فقط معرفة مكان الطرد.",
        },
    ),
    (
        "address_only",
        2,
        {
            "en": "The item is fine; I only need to change its delivery address.",
            "zh": "商品没有问题；我只需要更改它的收货地址。",
            "es": "El artículo está bien; solo necesito cambiar su dirección de entrega.",
            "fr": "L'article convient ; je dois seulement changer son adresse de livraison.",
            "de": "Der Artikel ist in Ordnung; ich muss nur die Lieferadresse ändern.",
            "ja": "商品に問題はありません。配送先の住所だけ変更したいです。",
            "ar": "السلعة سليمة؛ أحتاج فقط إلى تغيير عنوان تسليمها.",
        },
    ),
)

NOUL_LEAD = {
    "en": "Answer only from the stated facts.",
    "zh": "仅根据陈述的事实作答。",
    "es": "Responda solo con los hechos indicados.",
    "fr": "Répondez uniquement d'après les faits indiqués.",
    "de": "Antworten Sie nur anhand der genannten Fakten.",
    "ja": "述べられた事実だけに基づいて答えてください。",
    "ar": "أجب اعتماداً على الحقائق المذكورة فقط.",
}

NOUL_OPTIONS = {
    "en": {"false": "No", "true": "Yes"},
    "zh": {"false": "否", "true": "是"},
    "es": {"false": "No", "true": "Sí"},
    "fr": {"false": "Non", "true": "Oui"},
    "de": {"false": "Nein", "true": "Ja"},
    "ja": {"false": "いいえ", "true": "はい"},
    "ar": {"false": "لا", "true": "نعم"},
}

NOUL_ALTERNATE_OPTIONS = {
    "en": {"false": "The answer is no", "true": "The answer is yes"},
    "zh": {"false": "答案是否", "true": "答案是肯定"},
    "es": {"false": "La respuesta es no", "true": "La respuesta es sí"},
    "fr": {"false": "La réponse est non", "true": "La réponse est oui"},
    "de": {"false": "Die Antwort ist nein", "true": "Die Antwort ist ja"},
    "ja": {"false": "答えはいいえ", "true": "答えははい"},
    "ar": {"false": "الإجابة لا", "true": "الإجابة نعم"},
}

# Each value holds (state, question) in the given language.
NOUL_CASES = (
    (
        "shipped",
        True,
        {
            "en": ("The parcel was shipped on Monday.", "Has the parcel been shipped?"),
            "zh": ("包裹已在周一发出。", "包裹已经发出了吗？"),
            "es": ("El paquete se envió el lunes.", "¿Se ha enviado el paquete?"),
            "fr": ("Le colis a été expédié lundi.", "Le colis a-t-il été expédié ?"),
            "de": (
                "Das Paket wurde am Montag versendet.",
                "Wurde das Paket versendet?",
            ),
            "ja": ("荷物は月曜日に発送されました。", "荷物は発送済みですか。"),
            "ar": ("أُرسل الطرد يوم الاثنين.", "هل أُرسل الطرد؟"),
        },
    ),
    (
        "meeting_time",
        False,
        {
            "en": ("The meeting starts at 15:00.", "Does the meeting start at 16:00?"),
            "zh": ("会议在 15:00 开始。", "会议在 16:00 开始吗？"),
            "es": (
                "La reunión comienza a las 15:00.",
                "¿Comienza la reunión a las 16:00?",
            ),
            "fr": (
                "La réunion commence à 15:00.",
                "La réunion commence-t-elle à 16:00 ?",
            ),
            "de": (
                "Die Besprechung beginnt um 15:00.",
                "Beginnt die Besprechung um 16:00?",
            ),
            "ja": ("会議は 15:00 に始まります。", "会議は 16:00 に始まりますか。"),
            "ar": ("يبدأ الاجتماع الساعة 15:00.", "هل يبدأ الاجتماع الساعة 16:00؟"),
        },
    ),
    (
        "apple_count",
        True,
        {
            "en": (
                "Mina had 2 apples and bought 1 more.",
                "Does Mina now have 3 apples?",
            ),
            "zh": ("Mina 原有 2 个苹果，又买了 1 个。", "Mina 现在有 3 个苹果吗？"),
            "es": (
                "Mina tenía 2 manzanas y compró 1 más.",
                "¿Tiene Mina ahora 3 manzanas?",
            ),
            "fr": (
                "Mina avait 2 pommes et en a acheté 1 de plus.",
                "Mina a-t-elle maintenant 3 pommes ?",
            ),
            "de": (
                "Mina hatte 2 Äpfel und kaufte 1 weiteren.",
                "Hat Mina jetzt 3 Äpfel?",
            ),
            "ja": (
                "Mina はリンゴを 2 個持っていて、さらに 1 個買いました。",
                "Mina は今リンゴを 3 個持っていますか。",
            ),
            "ar": (
                "كان لدى Mina 2 من التفاح واشترت 1 أخرى.",
                "هل لدى Mina الآن 3 تفاحات؟",
            ),
        },
    ),
    (
        "private_file",
        False,
        {
            "en": ("The file is marked private.", "Is the file marked public?"),
            "zh": ("文件标记为私有。", "文件标记为公开吗？"),
            "es": (
                "El archivo está marcado como privado.",
                "¿Está marcado como público?",
            ),
            "fr": (
                "Le fichier est marqué comme privé.",
                "Le fichier est-il marqué comme public ?",
            ),
            "de": (
                "Die Datei ist als privat gekennzeichnet.",
                "Ist die Datei als öffentlich gekennzeichnet?",
            ),
            "ja": (
                "ファイルには非公開の印が付いています。",
                "ファイルには公開の印が付いていますか。",
            ),
            "ar": ("وُسم الملف بأنه خاص.", "هل وُسم الملف بأنه عام؟"),
        },
    ),
    (
        "approved",
        True,
        {
            "en": ("The request was approved yesterday.", "Was the request approved?"),
            "zh": ("这项请求昨天已获批准。", "这项请求获批准了吗？"),
            "es": ("La solicitud fue aprobada ayer.", "¿Fue aprobada la solicitud?"),
            "fr": (
                "La demande a été approuvée hier.",
                "La demande a-t-elle été approuvée ?",
            ),
            "de": (
                "Der Antrag wurde gestern genehmigt.",
                "Wurde der Antrag genehmigt?",
            ),
            "ja": ("申請は昨日承認されました。", "申請は承認されましたか。"),
            "ar": ("تمت الموافقة على الطلب أمس.", "هل تمت الموافقة على الطلب؟"),
        },
    ),
    (
        "closing_time",
        False,
        {
            "en": ("The store closes at 18:00.", "Does the store close at 20:00?"),
            "zh": ("商店在 18:00 关门。", "商店在 20:00 关门吗？"),
            "es": ("La tienda cierra a las 18:00.", "¿Cierra la tienda a las 20:00?"),
            "fr": ("Le magasin ferme à 18:00.", "Le magasin ferme-t-il à 20:00 ?"),
            "de": (
                "Das Geschäft schließt um 18:00.",
                "Schließt das Geschäft um 20:00?",
            ),
            "ja": ("店は 18:00 に閉まります。", "店は 20:00 に閉まりますか。"),
            "ar": ("يغلق المتجر الساعة 18:00.", "هل يغلق المتجر الساعة 20:00؟"),
        },
    ),
)

SCORE_STATE = {
    "en": "Out of 5 checks, {n} failed.",
    "zh": "5 项检查中，有 {n} 项未通过。",
    "es": "De 5 comprobaciones, {n} fallaron.",
    "fr": "Sur 5 vérifications, {n} ont échoué.",
    "de": "Von 5 Prüfungen sind {n} fehlgeschlagen.",
    "ja": "5 件の確認のうち、{n} 件が不合格でした。",
    "ar": "من أصل 5 فحوص، فشل {n} منها.",
}

SCORE_INSTRUCTION = {
    "en": "Assign the level that matches the number of failed checks.",
    "zh": "根据未通过的检查数量选择对应等级。",
    "es": "Elija el nivel correspondiente al número de comprobaciones fallidas.",
    "fr": "Choisissez le niveau correspondant au nombre de vérifications échouées.",
    "de": "Wählen Sie die Stufe, die zur Anzahl fehlgeschlagener Prüfungen passt.",
    "ja": "不合格になった確認の件数に対応する段階を選んでください。",
    "ar": "اختر المستوى المطابق لعدد الفحوص الفاشلة.",
}

SCORE_CRITERIA = {
    "en": (
        "0 failed checks",
        "1 failed check",
        "2 or 3 failed checks",
        "4 or 5 failed checks",
    ),
    "zh": ("0 项未通过", "1 项未通过", "2 或 3 项未通过", "4 或 5 项未通过"),
    "es": (
        "0 comprobaciones fallidas",
        "1 comprobación fallida",
        "2 o 3 comprobaciones fallidas",
        "4 o 5 comprobaciones fallidas",
    ),
    "fr": (
        "0 vérification échouée",
        "1 vérification échouée",
        "2 ou 3 vérifications échouées",
        "4 ou 5 vérifications échouées",
    ),
    "de": (
        "0 fehlgeschlagene Prüfungen",
        "1 fehlgeschlagene Prüfung",
        "2 oder 3 fehlgeschlagene Prüfungen",
        "4 oder 5 fehlgeschlagene Prüfungen",
    ),
    "ja": (
        "不合格 0 件",
        "不合格 1 件",
        "不合格 2 または 3 件",
        "不合格 4 または 5 件",
    ),
    "ar": ("0 فحوص فاشلة", "1 فحص فاشل", "2 أو 3 فحوص فاشلة", "4 أو 5 فحوص فاشلة"),
}
