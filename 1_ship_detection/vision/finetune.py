"""Adapt the ShipsNet detector to Sentinel-2 chips.

The notebook trained `ship_detector_fixed.keras` in three stages: frozen head,
full fine-tune at 1e-5, then hard-negative mining against a real port scene.
This is a fourth stage in the same spirit, for a different distribution shift -
not land false positives this time, but a change of sensor.

    python -m vision.finetune --train

Design choices that matter for a set this small (hundreds of chips, not
thousands):

  * Freeze the convolutional backbone. With ~400 labelled chips a full
    fine-tune of EfficientNetB0's 4M parameters memorises the set. Only the
    classifier head is retrained, so the number of free parameters stays in
    the low thousands.
  * Split by SCENE, not at random. Chips from one date share illumination, sea
    state and sun glint; a random split puts near-identical chips on both
    sides and reports a precision that will not survive a new date.
  * Augment with flips and 90-degree rotations only. Vessel headings are
    uniformly distributed, so rotation is label-preserving here, while
    brightness jitter is not - the fixed radiometric stretch is what makes
    dates comparable, and teaching the model to ignore brightness would throw
    that away.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_labelled(chips_dir):
    d = np.load(Path(chips_dir) / "labelled.npz")
    return d["X"], d["y"], d["idx"]


def scene_split(chips_dir, idx, holdout_frac=0.3, seed=0, y_for_check=None):
    """Group chips by source scene and hold whole scenes out."""
    import csv

    scene_of = {}
    with open(Path(chips_dir) / "chips.csv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            scene_of[int(row["idx"])] = row["scene"]
    scenes = sorted({scene_of[int(i)] for i in idx})
    rng = np.random.default_rng(seed)
    rng.shuffle(scenes)
    n_hold = max(1, int(round(len(scenes) * holdout_frac)))

    # A scene can contribute only negatives - 2022-04-23 has heavy surface
    # texture and drew 58 negative chips with no positives. Holding out a set
    # with no ships in it makes recall undefined and the report meaningless,
    # so rotate the shuffle until the test side actually contains positives.
    for start in range(len(scenes)):
        held = set((scenes + scenes)[start:start + n_hold])
        is_test = np.array([scene_of[int(i)] in held for i in idx])
        if y_for_check is None:
            break
        if y_for_check[is_test].sum() >= 5 and y_for_check[~is_test].sum() >= 10:
            break
    return is_test, sorted(held)


def augment(X, y, seed=0):
    """8-fold dihedral expansion: 4 rotations x optional flip."""
    rng = np.random.default_rng(seed)
    outs_x, outs_y = [], []
    for k in range(4):
        r = np.rot90(X, k, axes=(1, 2))
        outs_x.append(r); outs_y.append(y)
        outs_x.append(r[:, :, ::-1]); outs_y.append(y)
    Xa = np.concatenate(outs_x); ya = np.concatenate(outs_y)
    p = rng.permutation(len(ya))
    return Xa[p], ya[p]


def build_head_model(base_path):
    """Load the trained detector and rebuild it as frozen backbone + new head."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    base = tf.keras.models.load_model(str(base_path))

    # The saved model is backbone + head; find the last pooling/flatten so the
    # learned convolutional features are reused and only the top is replaced.
    cut = None
    for layer in reversed(base.layers):
        if isinstance(layer, (layers.GlobalAveragePooling2D, layers.GlobalMaxPooling2D,
                              layers.Flatten)):
            cut = layer
            break
    if cut is None:
        raise RuntimeError("could not find a pooling layer to cut the head at")

    features = models.Model(base.input, cut.output, name="backbone")
    features.trainable = False

    inp = layers.Input(shape=base.input_shape[1:])
    x = features(inp, training=False)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inp, out, name="ship_detector_s2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def metrics_at(y_true, scores, threshold):
    pred = (scores > threshold).astype("int32")
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)}


def roc_auc(y_true, scores):
    """Rank-based AUC; no sklearn (its extensions are blocked here)."""
    y_true = np.asarray(y_true)
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype="float64")
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s = np.asarray(scores)[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def train(chips_dir, base_path, out_path, epochs=25, holdout_frac=0.3, seed=0,
          report_path=None):
    import tensorflow as tf

    X, y, idx = load_labelled(chips_dir)
    is_test, held = scene_split(chips_dir, idx, holdout_frac=holdout_frac,
                                seed=seed, y_for_check=y)
    Xtr, ytr = X[~is_test].astype("float32") / 255.0, y[~is_test]
    Xte, yte = X[is_test].astype("float32") / 255.0, y[is_test]
    print(f"train {len(ytr)} chips ({int(ytr.sum())} ship) | "
          f"test {len(yte)} chips ({int(yte.sum())} ship)")
    print(f"held-out scenes: {', '.join(held)}")

    Xa, ya = augment(Xtr, ytr, seed=seed)
    print(f"after 8x dihedral augmentation: {len(ya)} training chips")

    model = build_head_model(base_path)

    # The chip set is dominated by water, so weight the positives back up;
    # otherwise the head can reach high accuracy by never predicting "ship".
    pos = float(ya.sum()); neg = float(len(ya) - pos)
    class_weight = {0: 1.0, 1: (neg / pos) if pos else 1.0}
    print(f"class weight for ship: {class_weight[1]:.2f}")

    model.fit(
        Xa, ya, epochs=epochs, batch_size=32, verbose=2,
        class_weight=class_weight,
        validation_data=(Xte, yte) if len(yte) else None,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss" if len(yte) else "loss",
            patience=6, restore_best_weights=True)],
    )

    report = {"held_out_scenes": held, "n_train": int(len(ytr)), "n_test": int(len(yte))}
    if len(yte):
        scores = model.predict(Xte, verbose=0).flatten()
        report["auc"] = round(roc_auc(yte, scores), 3)
        report["sweep"] = [metrics_at(yte, scores, t)
                           for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)]
        best = max(report["sweep"], key=lambda m: m["f1"])
        report["best_threshold"] = best["threshold"]
        print(f"\nHELD-OUT AUC {report['auc']}")
        for m in report["sweep"]:
            print(f"  thr {m['threshold']}: P {m['precision']}  R {m['recall']}  "
                  f"F1 {m['f1']}   (tp {m['tp']} fp {m['fp']} fn {m['fn']})")

    model.save(str(out_path))
    Path(out_path).with_suffix(".report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nsaved {out_path}")
    return report


def main(argv=None):
    from .config import DATA, DETECTOR_REPORT, SHIP_DETECTOR, SHIP_DETECTOR_S2

    p = argparse.ArgumentParser(description="Fine-tune the detector for Sentinel-2.")
    p.add_argument("--chips", type=Path, default=DATA / "chips")
    p.add_argument("--base", type=Path, default=SHIP_DETECTOR)
    p.add_argument("--out", type=Path, default=SHIP_DETECTOR_S2)
    p.add_argument("--report", type=Path, default=DETECTOR_REPORT)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--holdout", type=float, default=0.3)
    p.add_argument("--train", action="store_true")
    args = p.parse_args(argv)

    if args.train:
        train(args.chips, args.base, args.out, epochs=args.epochs,
              holdout_frac=args.holdout, report_path=args.report)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
