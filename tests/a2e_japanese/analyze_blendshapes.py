"""
A2Eブレンドシェイプ出力分析ツール

A2E推論結果（52次元ARKitブレンドシェイプ）を分析し、
日本語音声に対するリップシンク品質を評価する。

使い方:
    # A2E推論後に出力されたnpyファイルを分析
    python analyze_blendshapes.py --input blendshape_outputs/vowels_aiueo.npy

    # 複数ファイルを比較
    python analyze_blendshapes.py --input-dir blendshape_outputs/

    # CSVエクスポート
    python analyze_blendshapes.py --input-dir blendshape_outputs/ --export-csv
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# ARKit 52 ブレンドシェイプ名
ARKIT_NAMES = [
    "eyeBlinkLeft", "eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft",
    "eyeLookUpLeft", "eyeSquintLeft", "eyeWideLeft",
    "eyeBlinkRight", "eyeLookDownRight", "eyeLookInRight", "eyeLookOutRight",
    "eyeLookUpRight", "eyeSquintRight", "eyeWideRight",
    "jawForward", "jawLeft", "jawRight", "jawOpen",
    "mouthClose", "mouthFunnel", "mouthPucker", "mouthLeft", "mouthRight",
    "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthDimpleLeft", "mouthDimpleRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthPressLeft", "mouthPressRight", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "noseSneerLeft", "noseSneerRight",
    "tongueOut",
]

# カテゴリ分け
CATEGORIES = {
    "jaw": [i for i, n in enumerate(ARKIT_NAMES) if n.startswith("jaw")],
    "mouth": [i for i, n in enumerate(ARKIT_NAMES) if n.startswith("mouth")],
    "eye": [i for i, n in enumerate(ARKIT_NAMES) if n.startswith("eye")],
    "brow": [i for i, n in enumerate(ARKIT_NAMES) if n.startswith("brow")],
    "cheek": [i for i, n in enumerate(ARKIT_NAMES) if n.startswith("cheek")],
    "nose": [i for i, n in enumerate(ARKIT_NAMES) if n.startswith("nose")],
    "tongue": [i for i, n in enumerate(ARKIT_NAMES) if n.startswith("tongue")],
}

# リップシンクに重要なブレンドシェイプ
LIP_SYNC_CRITICAL = {
    "jawOpen": ARKIT_NAMES.index("jawOpen"),
    "mouthClose": ARKIT_NAMES.index("mouthClose"),
    "mouthFunnel": ARKIT_NAMES.index("mouthFunnel"),
    "mouthPucker": ARKIT_NAMES.index("mouthPucker"),
    "mouthSmileLeft": ARKIT_NAMES.index("mouthSmileLeft"),
    "mouthSmileRight": ARKIT_NAMES.index("mouthSmileRight"),
    "mouthLowerDownLeft": ARKIT_NAMES.index("mouthLowerDownLeft"),
    "mouthLowerDownRight": ARKIT_NAMES.index("mouthLowerDownRight"),
    "mouthUpperUpLeft": ARKIT_NAMES.index("mouthUpperUpLeft"),
    "mouthUpperUpRight": ARKIT_NAMES.index("mouthUpperUpRight"),
}


def analyze_single(data: np.ndarray, name: str, fps: float = 30.0) -> dict:
    """単一ブレンドシェイプ出力の分析"""
    if data.ndim != 2 or data.shape[1] != 52:
        raise ValueError(f"Expected shape (N, 52), got {data.shape}")

    num_frames = data.shape[0]
    duration = num_frames / fps

    result = {
        "name": name,
        "num_frames": num_frames,
        "duration_s": round(duration, 2),
        "fps": fps,
    }

    # 全体統計
    result["global"] = {
        "mean": round(float(data.mean()), 6),
        "std": round(float(data.std()), 6),
        "min": round(float(data.min()), 6),
        "max": round(float(data.max()), 6),
        "abs_mean": round(float(np.abs(data).mean()), 6),
    }

    # カテゴリ別統計
    result["categories"] = {}
    for cat_name, indices in CATEGORIES.items():
        cat_data = data[:, indices]
        result["categories"][cat_name] = {
            "mean_activation": round(float(np.abs(cat_data).mean()), 6),
            "max_activation": round(float(np.abs(cat_data).max()), 6),
            "active_ratio": round(float((np.abs(cat_data) > 0.01).any(axis=0).mean()), 4),
        }

    # リップシンク品質指標
    lip_indices = CATEGORIES["jaw"] + CATEGORIES["mouth"]
    lip_data = data[:, lip_indices]

    # 1. 動的範囲 (Dynamic Range): リップが動いている幅
    lip_range = float(lip_data.max() - lip_data.min())

    # 2. 時間変動 (Temporal Variation): フレーム間の変化量
    if num_frames > 1:
        lip_diff = np.diff(lip_data, axis=0)
        temporal_var = float(np.abs(lip_diff).mean())
    else:
        temporal_var = 0.0

    # 3. 活性度 (Activation Level): リップの平均活性度
    lip_activation = float(np.abs(lip_data).mean())

    # 4. 対称性 (Symmetry): 左右のブレンドシェイプの対称度
    symmetry_pairs = [
        ("mouthSmileLeft", "mouthSmileRight"),
        ("mouthFrownLeft", "mouthFrownRight"),
        ("mouthLowerDownLeft", "mouthLowerDownRight"),
        ("mouthUpperUpLeft", "mouthUpperUpRight"),
        ("mouthPressLeft", "mouthPressRight"),
    ]
    symmetry_scores = []
    for left_name, right_name in symmetry_pairs:
        if left_name in ARKIT_NAMES and right_name in ARKIT_NAMES:
            left_idx = ARKIT_NAMES.index(left_name)
            right_idx = ARKIT_NAMES.index(right_name)
            diff = np.abs(data[:, left_idx] - data[:, right_idx]).mean()
            symmetry_scores.append(1.0 - min(diff, 1.0))

    symmetry = float(np.mean(symmetry_scores)) if symmetry_scores else 0.0

    # 5. jawOpenの活性パターン
    jaw_open_idx = ARKIT_NAMES.index("jawOpen")
    jaw_data = data[:, jaw_open_idx]
    jaw_peaks = len(_find_peaks(jaw_data, threshold=0.1))

    result["lip_sync"] = {
        "dynamic_range": round(lip_range, 4),
        "temporal_variation": round(temporal_var, 6),
        "activation_level": round(lip_activation, 6),
        "symmetry": round(symmetry, 4),
        "jaw_open_peaks": jaw_peaks,
        "jaw_open_peaks_per_sec": round(jaw_peaks / max(duration, 0.01), 2),
    }

    # リップシンク品質スコア (0-100)
    # 高い temporal_variation = 口が動いている
    # 適度な dynamic_range = 表現力がある
    # 高い symmetry = 自然な動き
    quality_score = min(100, (
        min(temporal_var * 500, 30) +
        min(lip_range * 20, 25) +
        min(lip_activation * 200, 20) +
        symmetry * 25
    ))
    result["lip_sync"]["quality_score"] = round(quality_score, 1)

    # Top 10 最活性ブレンドシェイプ
    mean_abs = np.abs(data).mean(axis=0)
    top_indices = np.argsort(-mean_abs)[:10]
    result["top10_blendshapes"] = [
        {"rank": rank + 1, "name": ARKIT_NAMES[i], "mean_abs": round(float(mean_abs[i]), 6)}
        for rank, i in enumerate(top_indices)
    ]

    # リップシンク重要ブレンドシェイプの詳細
    result["critical_blendshapes"] = {}
    for bs_name, bs_idx in LIP_SYNC_CRITICAL.items():
        bs_data = data[:, bs_idx]
        result["critical_blendshapes"][bs_name] = {
            "mean": round(float(bs_data.mean()), 6),
            "std": round(float(bs_data.std()), 6),
            "min": round(float(bs_data.min()), 6),
            "max": round(float(bs_data.max()), 6),
            "active_frames_pct": round(float((np.abs(bs_data) > 0.01).mean()) * 100, 1),
        }

    return result


def _find_peaks(data: np.ndarray, threshold: float = 0.1) -> list:
    """簡易ピーク検出"""
    peaks = []
    for i in range(1, len(data) - 1):
        if data[i] > threshold and data[i] > data[i - 1] and data[i] > data[i + 1]:
            peaks.append(i)
    return peaks


def compare_languages(results: dict) -> dict:
    """言語間のリップシンク品質比較"""
    comparison = {}

    # カテゴリを推測
    ja_results = {k: v for k, v in results.items() if not k.endswith(("_compare", "_baseline"))}
    en_results = {k: v for k, v in results.items() if "english" in k}
    zh_results = {k: v for k, v in results.items() if "chinese" in k}

    for lang_name, lang_results in [("japanese", ja_results), ("english", en_results), ("chinese", zh_results)]:
        if not lang_results:
            continue

        scores = [r["lip_sync"]["quality_score"] for r in lang_results.values()]
        temporal_vars = [r["lip_sync"]["temporal_variation"] for r in lang_results.values()]
        jaw_rates = [r["lip_sync"]["jaw_open_peaks_per_sec"] for r in lang_results.values()]

        comparison[lang_name] = {
            "num_samples": len(scores),
            "avg_quality_score": round(float(np.mean(scores)), 1),
            "avg_temporal_variation": round(float(np.mean(temporal_vars)), 6),
            "avg_jaw_peaks_per_sec": round(float(np.mean(jaw_rates)), 2),
        }

    return comparison


def print_report(result: dict):
    """分析結果を見やすく表示"""
    print(f"\n{'=' * 60}")
    print(f"  {result['name']}")
    print(f"  {result['num_frames']} frames, {result['duration_s']}s @ {result['fps']}fps")
    print(f"{'=' * 60}")

    ls = result["lip_sync"]
    print(f"\n  Lip Sync Quality Score: {ls['quality_score']}/100")
    print(f"    Dynamic Range:      {ls['dynamic_range']:.4f}")
    print(f"    Temporal Variation:  {ls['temporal_variation']:.6f}")
    print(f"    Activation Level:   {ls['activation_level']:.6f}")
    print(f"    Symmetry:           {ls['symmetry']:.4f}")
    print(f"    Jaw Open Peaks:     {ls['jaw_open_peaks']} ({ls['jaw_open_peaks_per_sec']}/sec)")

    print(f"\n  Category Activation:")
    for cat, stats in result["categories"].items():
        bar = "█" * int(stats["mean_activation"] * 100)
        print(f"    {cat:8s}: {stats['mean_activation']:.4f} {bar}")

    print(f"\n  Top 10 Active Blendshapes:")
    for bs in result["top10_blendshapes"]:
        print(f"    {bs['rank']:2d}. {bs['name']:25s} {bs['mean_abs']:.6f}")

    print(f"\n  Critical Lip Sync Blendshapes:")
    for name, stats in result["critical_blendshapes"].items():
        print(f"    {name:25s} mean={stats['mean']:.4f} std={stats['std']:.4f} "
              f"active={stats['active_frames_pct']:.1f}%")


def export_csv(results: dict, output_path: str):
    """結果をCSVにエクスポート"""
    import csv
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # ヘッダー
        writer.writerow(["name", "frames", "duration_s", "quality_score",
                        "dynamic_range", "temporal_variation", "activation_level",
                        "symmetry", "jaw_peaks_per_sec"])
        for name, result in results.items():
            ls = result["lip_sync"]
            writer.writerow([
                name, result["num_frames"], result["duration_s"],
                ls["quality_score"], ls["dynamic_range"], ls["temporal_variation"],
                ls["activation_level"], ls["symmetry"], ls["jaw_open_peaks_per_sec"],
            ])
    print(f"\nCSV exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="A2E Blendshape Output Analyzer")
    parser.add_argument("--input", type=str, help="Single .npy file to analyze")
    parser.add_argument("--input-dir", type=str, help="Directory of .npy files to analyze")
    parser.add_argument("--fps", type=float, default=30.0, help="Frames per second (default: 30)")
    parser.add_argument("--export-csv", action="store_true", help="Export results to CSV")
    parser.add_argument("--export-json", action="store_true", help="Export results to JSON")
    args = parser.parse_args()

    if not args.input and not args.input_dir:
        # デモモード
        print("No input specified. Running demo with synthetic data.\n")
        print("Usage:")
        print("  python analyze_blendshapes.py --input output.npy")
        print("  python analyze_blendshapes.py --input-dir blendshape_outputs/")
        print("\nExpected input format: numpy array of shape (num_frames, 52)")
        print("\nRunning demo with synthetic data...\n")

        # デモ: 合成データで分析例を表示
        np.random.seed(42)
        demo_data = np.random.rand(90, 52).astype(np.float32) * 0.3
        # jawOpenに周期的なパターンを追加
        t = np.linspace(0, 3, 90)
        demo_data[:, ARKIT_NAMES.index("jawOpen")] = 0.3 * np.abs(np.sin(2 * np.pi * t))
        demo_data[:, ARKIT_NAMES.index("mouthFunnel")] = 0.15 * np.abs(np.sin(2 * np.pi * t + 0.5))

        result = analyze_single(demo_data, "demo_synthetic", fps=args.fps)
        print_report(result)
        return

    results = {}

    if args.input:
        data = np.load(args.input)
        name = Path(args.input).stem
        result = analyze_single(data, name, fps=args.fps)
        results[name] = result
        print_report(result)

    if args.input_dir:
        input_dir = Path(args.input_dir)
        for npy_path in sorted(input_dir.glob("*.npy")):
            data = np.load(str(npy_path))
            name = npy_path.stem
            try:
                result = analyze_single(data, name, fps=args.fps)
                results[name] = result
                print_report(result)
            except ValueError as e:
                print(f"\n  [SKIP] {name}: {e}")

    if len(results) > 1:
        print("\n" + "=" * 60)
        print("LANGUAGE COMPARISON")
        print("=" * 60)
        comparison = compare_languages(results)
        for lang, stats in comparison.items():
            print(f"\n  {lang}:")
            for k, v in stats.items():
                print(f"    {k}: {v}")

    if args.export_csv and results:
        csv_path = str(Path(args.input_dir or ".") / "analysis_results.csv")
        export_csv(results, csv_path)

    if args.export_json and results:
        json_path = str(Path(args.input_dir or ".") / "analysis_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nJSON exported to: {json_path}")


if __name__ == "__main__":
    main()
