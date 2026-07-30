# PyroForge — all-models scorecard (held-out)

dG is built as *baseline(wt) + ΔΔG delta* so the within-protein signal comes
from the difference-embedding model (this de-bands the scatter). The pooled dG
correlation therefore inherits its between-protein spread from the absolute
baseline and its within-protein signal from ΔΔG — read it as the anchored
construction's skill, not a standalone absolute-dG predictor. Absolute Tm is
model_tm scored on its Meltome held-out split (its valid assay).

| Model / output | Parent-val | Held-out test |
|---|---|---|
| absolute dG = dG(wt)+ΔΔG | r=0.656, MAE=1.003, n=42473, within-protein r=0.76 (22p) | r=0.738, MAE=1.048, n=3561, within-protein r=0.52 (3p) |
| absolute Tm (model_tm · Meltome) | — | r=0.834, MAE=5.433, n=1106 |
| ΔΔG (model_ddg) | r=0.758, MAE=0.596, n=39264, within-protein r=0.79 (22p) | r=0.688, MAE=0.591, n=2676, within-protein r=-0.19 (3p) |

**Per-protein reliability.** model_ddg's ΔΔG is not uniform across proteins: on the validation proteins its within-protein r has median 0.79. The held-out test file covers only a few proteins, so its pooled r is noisier than the 266-protein grouped-CV estimate.
