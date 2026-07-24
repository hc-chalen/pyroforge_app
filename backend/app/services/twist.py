from __future__ import annotations

import httpx

from app.core.config import settings


class TwistService:
    """Twist Bioscience gene synthesis quote integration."""

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
                        "price_usd": data.get("price", data.get("total_price")),
                        "lead_time_days": data.get("lead_time_days", data.get("turnaround_days")),
                        "twist_status": "quoted",
                        "twist_response": data,
                    }
        except httpx.HTTPError:
            pass
        return self._offline_estimate(dna_sequence)

    @staticmethod
    def _offline_estimate(dna_sequence: str) -> dict:
        length = len(dna_sequence)
        price = round(0.09 * length + 50, 2)
        lead_time = 10 if length < 1500 else 15
        return {
            "price_usd": price,
            "lead_time_days": lead_time,
            "twist_status": "estimated_offline",
            "twist_response": None,
        }
