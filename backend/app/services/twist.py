from __future__ import annotations

import httpx

from app.core.config import settings

# Twist's ordering app. There is no documented deep link that pre-fills a
# sequence, so the UI sends the user here with the CDS on their clipboard rather
# than pretending an order was placed.
TWIST_ORDER_URL = "https://ecommerce.twistdna.com/app/genes"
TWIST_PRICING_URL = "https://www.twistbioscience.com/products/genes"


class TwistService:
    """Twist Bioscience synthesis: real quote when a key is configured, otherwise
    a clearly-labelled local estimate plus a route to Twist's own order page.

    The distinction is carried in the payload, not just in prose. Previously both
    paths returned a bare ``price_usd`` and the UI rendered it as a headline
    figure; the only trace that it was a made-up ``0.09 × length + 50`` was the
    string ``estimated_offline`` in small print. On a deployment with no API key —
    which is what docs/OPERATIONS.md tells you to run publicly — that meant every
    user saw a fabricated number formatted exactly like a quote.
    """

    # Rough per-base list-price bands for synthetic genes, used only to put an
    # order-of-magnitude on the screen. These are NOT Twist's live prices: gene
    # synthesis pricing depends on the product line, sequence complexity, volume
    # and your institution's contract, and it changes. Anything shown from these
    # numbers must be labelled an estimate and paired with TWIST_PRICING_URL.
    _BANDS = (
        # (max_bp, usd_per_bp, product)
        (300, 0.15, "Gene fragment"),
        (1_800, 0.10, "Clonal gene"),
        (5_000, 0.12, "Clonal gene (long)"),
    )
    _HANDLING_USD = 50.0

    def get_quote(self, dna_sequence: str) -> dict:
        if not settings.twist_api_key:
            return self._offline_estimate(dna_sequence)

        payload = {
            "sequences": [{"name": "pyroforge_design", "sequence": dna_sequence}],
            "type": "non_clonal",
        }
        headers = {
            "Authorization": f"Bearer {settings.twist_api_key}",
            "Content-Type": "application/json",
        }
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    f"{settings.twist_api_base}/v1/genes/quotes",
                    json=payload,
                    headers=headers,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return {
                        "vendor": "Twist Bioscience",
                        "price_usd": data.get("price", data.get("total_price")),
                        "lead_time_days": data.get(
                            "lead_time_days", data.get("turnaround_days")
                        ),
                        "twist_status": "quoted",
                        "price_is_estimate": False,
                        "price_basis": "Live quote from the Twist Bioscience API.",
                        "order_url": TWIST_ORDER_URL,
                        "twist_response": data,
                    }
        except httpx.HTTPError:
            pass
        return self._offline_estimate(dna_sequence)

    @classmethod
    def _offline_estimate(cls, dna_sequence: str) -> dict:
        length = len(dna_sequence)
        per_bp, product = next(
            ((rate, name) for cap, rate, name in cls._BANDS if length <= cap),
            (cls._BANDS[-1][1], cls._BANDS[-1][2]),
        )
        price = round(per_bp * length + cls._HANDLING_USD, 2)
        return {
            "vendor": "Twist Bioscience",
            "price_usd": price,
            "lead_time_days": 10 if length < 1_500 else 15,
            "twist_status": "estimated_offline",
            # The two fields that stop this being read as a quote.
            "price_is_estimate": True,
            "price_basis": (
                f"Local estimate only — no Twist API key is configured, so Twist was "
                f"never contacted. Assumes ~${per_bp:.2f}/bp for a {length:,} bp "
                f"{product.lower()} plus ${cls._HANDLING_USD:.0f} handling. Real "
                f"pricing depends on sequence complexity, volume and your "
                f"institution's contract — get the actual price from Twist."
            ),
            "synthesis_product": product,
            "order_url": TWIST_ORDER_URL,
            "pricing_url": TWIST_PRICING_URL,
            "twist_response": None,
        }
