from __future__ import annotations

# Twist's ordering app. There is no documented deep link that pre-fills a
# sequence, so the UI sends the user here with the CDS on their clipboard rather
# than pretending an order was placed.
#
# Both of these are Twist's own links, not guesses: ORDER is where the "START
# ORDER" button on the pricing page below points. Take them from there again if
# they rot — checked 2026-09-14, when ORDER was still ".../app/genes" here and
# rendered a permanently blank page, because the route is "gene", singular.
#
# ORDER redirects to id.twistdna.com unless the visitor already has a Twist
# session, so the card says an account is needed rather than promising a page.
TWIST_ORDER_URL = "https://ecommerce.twistdna.com/app/gene"
TWIST_PRICING_URL = "https://www.twistbioscience.com/products/genes"


class TwistService:
    """The hand-off to Twist Bioscience: a route to their order page, nothing more.

    **This deliberately quotes no price and no lead time.** It used to do both, and
    both were invented:

      * With no API key — which is what a public deployment runs — ``price_usd`` came
        from a hard-coded ``$0.10–0.15/bp + $50`` band table. It was labelled as an
        estimate, but it was still a fabricated number rendered in the same stat tile
        as everything the app actually computes.
      * ``lead_time_days`` was ``10 if length < 1500 else 15``. That one carried no
        estimate flag at all, so a made-up figure sat next to a labelled-estimate
        price and read as fact.
      * With a key, a live call parsed the response as ``data["price"]`` or
        ``data["total_price"]`` — guesses at a schema that no test ever exercised.
        A miss set ``price_usd = None`` while still reporting ``twist_status:
        "quoted"`` and ``price_is_estimate: False``: a missing number presented as a
        real quote.

    Gene synthesis pricing depends on the product line, sequence complexity, volume
    and an institution's contract, and it changes. There is no honest way to put a
    number on it from here, so the card shows what this app genuinely computed — the
    CDS, its CAI against a stated reference, GC content, feasibility flags — and
    sends the user to Twist for anything commercial.
    """

    def order_route(self) -> dict:
        return {
            "vendor": "Twist Bioscience",
            "order_url": TWIST_ORDER_URL,
            "pricing_url": TWIST_PRICING_URL,
        }
