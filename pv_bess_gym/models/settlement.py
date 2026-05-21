from __future__ import annotations

from dataclasses import dataclass


def kwh_from_kw(kw: float, dt_h: float) -> float:
    return float(kw * dt_h)


def eur_from_kwh(kwh: float, price_eur_mwh: float) -> float:
    return float((kwh / 1000.0) * price_eur_mwh)


def reserve_capacity_eur(cap_kw: float, price_eur_mw_h: float, dt_h: float) -> float:
    return float((cap_kw / 1000.0) * price_eur_mw_h * dt_h)


@dataclass
class SettlementBreakdown:
    dam_eur: float
    id_eur: float
    imbalance_eur: float
    afrr_capacity_eur: float
    afrr_activation_eur: float
    degradation_eur: float
    penalties_eur: float

    @property
    def total_eur(self) -> float:
        return (
            self.dam_eur
            + self.id_eur
            + self.imbalance_eur
            + self.afrr_capacity_eur
            + self.afrr_activation_eur
            - self.degradation_eur
            - self.penalties_eur
        )


def settle_energy_layers(
    dam_position_kw: float,
    id_adjustment_kw: float,
    actual_export_kw: float,
    actual_import_kw: float,
    price_dam_eur_mwh: float,
    price_id_eur_mwh: float,
    price_imb_eur_mwh: float,
    dt_h: float,
) -> dict:
    """Commercial settlement layers for the simulator benchmark.

    The net commercial schedule at the point of common coupling is:

        schedule_kw = dam_position_kw + id_adjustment_kw

    The DAM leg settles the D-1 day-ahead position, the IDM leg settles the
    additive intraday correction, and imbalance settlement applies to metered
    physical delivery minus the final commercial schedule. There is therefore
    no extra hidden "DAM deviation penalty" here; any research penalty for
    non-compliance should be added explicitly in the environment.
    """
    dam_kwh = kwh_from_kw(dam_position_kw, dt_h)
    id_kwh = kwh_from_kw(id_adjustment_kw, dt_h)
    actual_signed_kwh = kwh_from_kw(actual_export_kw - actual_import_kw, dt_h)

    scheduled_signed_kwh = dam_kwh + id_kwh
    imbalance_kwh = actual_signed_kwh - scheduled_signed_kwh

    return {
        "dam_eur": eur_from_kwh(dam_kwh, price_dam_eur_mwh),
        "id_eur": eur_from_kwh(id_kwh, price_id_eur_mwh),
        "imbalance_eur": eur_from_kwh(imbalance_kwh, price_imb_eur_mwh),
        "actual_signed_kwh": actual_signed_kwh,
        "scheduled_signed_kwh": scheduled_signed_kwh,
        "imbalance_kwh": imbalance_kwh,
    }
