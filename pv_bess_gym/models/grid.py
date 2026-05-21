from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class GridConfig:
    export_cap_kw: float = 373.0
    import_cap_kw: float = 373.0
    allow_export: bool = True
    allow_import: bool = True
    loss_frac: float = 0.01


class Grid:
    """PCC model with import/export caps and simple metering losses."""

    def __init__(self, cfg: GridConfig):
        self.cfg = cfg

    def apply(self, net_inj_kw: float) -> dict:
        cfg = self.cfg
        inj = float(net_inj_kw)

        if not cfg.allow_export:
            inj = min(inj, 0.0)
        if not cfg.allow_import:
            inj = max(inj, 0.0)

        export_kw = float(np.clip(inj, 0.0, cfg.export_cap_kw))
        import_kw = float(np.clip(-inj, 0.0, cfg.import_cap_kw))
        inj_capped = export_kw - import_kw

        if inj_capped >= 0:
            inj_metered = inj_capped * (1.0 - cfg.loss_frac)
        else:
            inj_metered = inj_capped * (1.0 + cfg.loss_frac)

        curtailed_kw = float(max(0.0, inj - export_kw)) if inj > 0 else 0.0
        unserved_import_kw = float(max(0.0, -inj - import_kw)) if inj < 0 else 0.0

        return {
            "inj_req_kw": float(net_inj_kw),
            "inj_capped_kw": float(inj_capped),
            "inj_metered_kw": float(inj_metered),
            "export_kw": float(export_kw),
            "import_kw": float(import_kw),
            "curtailed_kw": float(curtailed_kw),
            "unserved_import_kw": float(unserved_import_kw),
        }
