# basket_recommender.py
import json, joblib, numpy as np, pandas as pd
from pathlib import Path

ARTIFACT_DIR = Path("model_artifacts")

# ── 1. Артефакты ─────────────────────────────────────────────
aisle_columns   = json.load(open(ARTIFACT_DIR / "aisle_columns.json"))
frequent_aisles = set(json.load(open(ARTIFACT_DIR / "frequent_aisles.json")))
product2aisle   = joblib.load(ARTIFACT_DIR / "product2aisle.pkl")
scaler          = joblib.load(ARTIFACT_DIR / "scaler.pkl")
kmeans          = joblib.load(ARTIFACT_DIR / "kmeans.pkl")
seg_rule_lists  = joblib.load(ARTIFACT_DIR / "seg_rule_lists.pkl")

aisle2idx = {a: i for i, a in enumerate(aisle_columns)}

aggregate_rules = (
    pd.concat([df for df in seg_rule_lists.values() if not df.empty])
      .drop_duplicates(["antecedents", "consequents"])
)

# ── 2. Рекомендации ─────────────────────────────────────────
def recommend_aisles(product_ids: list[int], top_k: int = 5, *, verbose: bool = False) -> list[str]:
    log = (print if verbose else (lambda *a, **k: None))

    # Шаг‑1: товары → категории
    cart_aisles = {product2aisle[pid] for pid in product_ids if pid in product2aisle}
    log(f"🔹 Categories in cart (raw): {cart_aisles}")

    if not cart_aisles:
        log("⛔ Ничего не нашли по product2aisle.")
        return []

    # Шаг‑2: фильтр только frequent‑категорий
    cart_aisles &= frequent_aisles
    log(f"🔹 After frequent‑filter: {cart_aisles}")
    if not cart_aisles:
        log("⛔ После фильтрации ничего не осталось.")
        return []

    # Шаг‑3: сегментация корзины
    vec = np.zeros(len(aisle_columns), dtype=np.float32)
    for a in cart_aisles:
        vec[aisle2idx[a]] = 1
    seg = int(kmeans.predict(scaler.transform([vec]))[0])
    log(f"🔹 Predicted segment: {seg}")

    rules_df = seg_rule_lists.get(seg)
    if rules_df is None or rules_df.empty:
        log("⚠️  У сегмента нет правил ― берём агрегированные.")
        rules_df = aggregate_rules

    # Шаг‑4: выбираем правила, где хотя бы один antecedent присутствует
    hits = rules_df[rules_df["antecedents"].apply(lambda s: any(a in s for a in cart_aisles))]
    log(f"🔹 Rules hit: {len(hits)}")
    if hits.empty:
        log("⛔ Совпадений правил нет.")
        return []

    # Шаг‑5: ранжируем consequent‑ы
    recommended = []
    for _, r in hits.sort_values("lift", ascending=False).iterrows():
        for cons in r["consequents"]:
            if cons not in cart_aisles and cons not in recommended:
                recommended.append(cons)
            if len(recommended) >= top_k:
                break
        if len(recommended) >= top_k:
            break

    # --- если ничего не нашли, пробуем агрегированные правила ---
    if not recommended:
        log("⚠️  В своём сегменте рекомендаций нет — пробуем агрегированные правила.")
        hits_all = aggregate_rules[aggregate_rules["antecedents"].apply(
            lambda s: any(a in s for a in cart_aisles)
        )]
        log(f"🔹 Rules hit in global rules: {len(hits_all)}")

        for _, r in hits_all.sort_values("lift", ascending=False).iterrows():
            for cons in r["consequents"]:
                if cons not in cart_aisles and cons not in recommended:
                    recommended.append(cons)
                if len(recommended) >= top_k:
                    break
            if len(recommended) >= top_k:
                break

    if recommended:
        log(f"🔹 Recommended: {recommended}")
    else:
        log("⛔ Не удалось найти рекомендации даже в агрегированных правилах.")

    return recommended[:top_k]

# ── тест ―───────────────────────────────────────────────────
if __name__ == "__main__":
    demo = [24852, 13176, 21137]  # бананы + шпинат + клубника
    print("Demo:", recommend_aisles(demo, verbose=True))
