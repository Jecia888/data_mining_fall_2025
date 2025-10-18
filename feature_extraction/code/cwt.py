#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CWT (连续小波变换) 日级特征（加强版；仅 r_close 与 volume_z_by_tod）

输入（--input_dir 下按股票一个 CSV）：
  由 preprocess 脚本生成的每分钟数据，至少包含：
    timestamp (tz-aware ET, 读入后设为 index),
    close, volume_z_by_tod, is_gap, session, day_id
  （其余列可有可无）

输出（每个股票一份）：
  <out_dir>/<TICKER>/cwt/cwt_daily.csv  （每天 1 行）
  [可选] <out_dir>/<TICKER>/cwt/cwt_ridges.csv  （脊线明细，按日多行）

特征（两条序列均提取；X ∈ {r_close, volume_z_by_tod}）：
  - CWT_X_power_lo/mid/hi                     : 分频带能量占比（>60m / 15–60m / <15m）
  - CWT_X_centroid_period_min                 : 能量加权“周期质心”（分钟）
  - CWT_X_flatness                            : 谱平坦度（0~1）
  - CWT_X_peak_period_min / _peak_time_min... : 全局主峰周期与出现时点（分钟 since open）
  - CWT_X_peak_ratio                          : 主峰能量占比（0~1）
  - CWT_X_energy_open/mid/late                : 开/午/尾盘能量占比（时间分布）
  - CWT_X_temporal_compactness                : 时间紧致度（Gini；越高越“集中爆发”）
  - CWT_X_ridge_count / _ridge_avg_period_min / _ridge_period_std :
      脊线条数/平均周期/周期标准差（阈值基于能量分位数）
质量与元信息：
  - ticker, day_id, day_start_ts, day_end_ts, feature_ts
  - CWT_flag_low_conf, CWT_gap_ratio, CWT_max_consecutive_gap, CWT_N, CWT_cover_ratio

用法示例：
  python build_cwt_features.py \
    --input_dir D:\...\intermediate_dataset \
    --out_dir D:\...\features \
    --wavelet morl --period_min 5 --period_max 120 --num_scales 36 \
    --cover_ratio_thresh 0.7 --gap_ratio_thresh 0.2 --max_gap_run_thresh 8 --n_thresh 300 \
    --ridge_percentile 95 --save_ridges
"""

import os
import glob
import argparse
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
import pywt  # PyWavelets

try:
    from zoneinfo import ZoneInfo
    NY_TZ = ZoneInfo("America/New_York")
except Exception:
    import pytz
    NY_TZ = pytz.timezone("America/New_York")


# ------------------ IO 与时间 ------------------ #

def read_minutes_csv(path: str) -> pd.DataFrame:
    """读入 CSV，解析 tz-aware ET timestamp 为 index；丢弃无效时间并排序。"""
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"'timestamp' column not found in {path}")
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(NY_TZ, nonexistent="shift_forward", ambiguous="NaT")
    else:
        ts = ts.dt.tz_convert(NY_TZ)
    df = df.drop(columns=["timestamp"])
    df.index = ts
    df = df[~df.index.isna()].sort_index()
    return df


def et_day_bounds(day: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """返回给定 ET 日期的 09:30 与 16:00 tz-aware 时间戳。"""
    d = pd.Timestamp(day)
    d = d.tz_localize(NY_TZ) if d.tz is None else d.tz_convert(NY_TZ)
    start = d.replace(hour=9, minute=30, second=0, microsecond=0)
    end = d.replace(hour=16, minute=0, second=0, microsecond=0)
    return start, end


# ------------------ 工具函数 ------------------ #

def gap_quality_metrics(day_df: pd.DataFrame) -> Tuple[float, int]:
    """基于 preprocess 写入的 is_gap 计算缺口比例与最长连续缺口。"""
    if "is_gap" not in day_df.columns:
        return 0.0, 0
    g = day_df["is_gap"].astype(int).to_numpy()
    gap_ratio = float(np.mean(g)) if len(g) else 0.0
    max_run = 0
    cur = 0
    for v in g:
        if v == 1:
            cur += 1
            if cur > max_run:
                max_run = cur
        else:
            cur = 0
    return gap_ratio, int(max_run)


def prep_series_for_cwt(s: pd.Series) -> np.ndarray:
    """去均值、NaN→0、std=1 标准化（std=0 则全 0）。"""
    x = s.to_numpy(dtype=float)
    mu = np.nanmean(x)
    if not np.isfinite(mu):
        mu = 0.0
    x = x - mu
    x[~np.isfinite(x)] = 0.0
    std = np.nanstd(x)
    if not np.isfinite(std) or std == 0.0:
        return np.zeros_like(x, dtype=float)
    return x / std


def build_scales_periods(period_min: float, period_max: float, num_scales: int,
                         wavelet: str, dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    构造对数均匀周期网格，并映射到 PyWavelets 的 scale：
      scale = central_frequency(wavelet) * period / dt
    返回：scales, periods(分钟)
    """
    periods = np.geomspace(period_min, period_max, num=num_scales)
    cfreq = pywt.central_frequency(wavelet)
    scales = (cfreq * periods) / dt
    return scales, periods


def coi_mask(scales: np.ndarray, nT: int, k: float = np.sqrt(2.0)) -> np.ndarray:
    """
    生成 COI 掩码（True=有效；False=COI 内无效）。
    近似 e-folding 半宽：halfwidth ≈ k * scale（单位=样本）。
    """
    mask = np.ones((len(scales), nT), dtype=bool)
    for i, sc in enumerate(scales):
        hw = int(np.ceil(k * sc))
        if hw > 0:
            hw = min(hw, nT // 2)
            mask[i, :hw] = False
            mask[i, nT - hw:] = False
    return mask


def cwt_power(x: np.ndarray, scales: np.ndarray, wavelet: str = "morl", dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算 CWT 并返回 (power, periods_min, freqs_cpm)。
    power = |coeff|^2；periods 由 pywt.scale2frequency 反推；freqs 单位 cycles/min。
    """
    coeffs, freqs = pywt.cwt(x, scales, wavelet, sampling_period=dt)  # freqs: cycles per sampling_period
    power = np.abs(coeffs) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        periods = 1.0 / freqs
    return power, periods, freqs


def band_masks_by_period(periods: np.ndarray) -> Dict[str, np.ndarray]:
    """按周期（分钟）构建分频带掩码：lo>60, mid:15–60, hi<15。"""
    lo = periods > 60.0
    mid = (periods >= 15.0) & (periods <= 60.0)
    hi = periods < 15.0
    return {"lo": lo, "mid": mid, "hi": hi}


def spectral_flatness(P: np.ndarray, eps: float = 1e-12) -> float:
    """谱平坦度（几何均值/算术均值）"""
    P = P[P > 0]
    if P.size == 0:
        return np.nan
    return float(np.exp(np.mean(np.log(P + eps))) / (np.mean(P) + eps))


def safe_period_from_freq(freq: float) -> float:
    if freq is None or not np.isfinite(freq) or freq <= 0:
        return np.nan
    return 1.0 / freq


def temporal_compactness_gini(energy_t: np.ndarray) -> float:
    """
    时间紧致度（Gini 系数）。
    输入：按时间聚合后的能量（非负）。
    输出：0=均匀分布，1=极端集中。
    """
    x = np.asarray(energy_t, dtype=float)
    x[x < 0] = 0.0
    s = x.sum()
    if s <= 0 or x.size == 0:
        return np.nan
    x = np.sort(x) / s
    n = x.size
    # Gini = 1 - 2 * sum_{i=1..n} (cummean at i)
    cum = np.cumsum(x)
    gini = 1.0 - 2.0 * np.mean(cum)
    # 归一到 [0,1]（该公式已在[0,1]）
    return float(gini)


def segment_energy_fractions(energy_t: np.ndarray, N: int,
                             open_end: int = 60, mid_end: int = 300) -> Tuple[float, float, float]:
    """
    将时间分三段（开盘/午间/尾盘）并计算能量占比。
    边界自动截断到 [0, N]。
    """
    open_end = max(0, min(open_end, N))
    mid_end = max(open_end, min(mid_end, N))
    late_end = N

    def frac(a: int, b: int) -> float:
        e = energy_t[a:b].sum()
        tot = energy_t.sum()
        return float(e / tot) if tot > 0 else np.nan

    return frac(0, open_end), frac(open_end, mid_end), frac(mid_end, late_end)


# ------------------ 单日：序列的全量特征 ------------------ #

def cwt_features_for_series(day_df: pd.DataFrame,
                            series_name: str,
                            wavelet: str,
                            period_min: float,
                            period_max: float,
                            num_scales: int,
                            dt: float,
                            ridge_percentile: float = 95.0) -> Tuple[Dict[str, float], List[Dict]]:
    """
    对某条序列提取全套特征，并返回 (特征字典, 脊线明细列表)。
    支持的 series_name: "r_close", "volume_z_by_tod"
    """
    if series_name == "r_close":
        close = day_df["close"].astype(float)
        s = np.log(close) - np.log(close.shift(1))
        if len(s) > 0:
            s.iloc[0] = 0.0
    elif series_name == "volume_z_by_tod":
        if "volume_z_by_tod" not in day_df.columns:
            # 若缺失则直接返回 NaN 特征
            return {f"CWT_{series_name}_power_lo": np.nan,
                    f"CWT_{series_name}_power_mid": np.nan,
                    f"CWT_{series_name}_power_hi": np.nan,
                    f"CWT_{series_name}_centroid_period_min": np.nan,
                    f"CWT_{series_name}_flatness": np.nan,
                    f"CWT_{series_name}_peak_period_min": np.nan,
                    f"CWT_{series_name}_peak_time_min_since_open": np.nan,
                    f"CWT_{series_name}_peak_ratio": np.nan,
                    f"CWT_{series_name}_energy_open": np.nan,
                    f"CWT_{series_name}_energy_mid": np.nan,
                    f"CWT_{series_name}_energy_late": np.nan,
                    f"CWT_{series_name}_temporal_compactness": np.nan,
                    f"CWT_{series_name}_ridge_count": 0,
                    f"CWT_{series_name}_ridge_avg_period_min": np.nan,
                    f"CWT_{series_name}_ridge_period_std": np.nan}, []
        s = day_df["volume_z_by_tod"].astype(float)
    else:
        raise ValueError(f"Unsupported series_name={series_name}")

    x = prep_series_for_cwt(s)
    N = len(x)
    if N < 16 or np.all(x == 0.0):
        # 返回空特征
        feats = {f"CWT_{series_name}_power_lo": np.nan,
                 f"CWT_{series_name}_power_mid": np.nan,
                 f"CWT_{series_name}_power_hi": np.nan,
                 f"CWT_{series_name}_centroid_period_min": np.nan,
                 f"CWT_{series_name}_flatness": np.nan,
                 f"CWT_{series_name}_peak_period_min": np.nan,
                 f"CWT_{series_name}_peak_time_min_since_open": np.nan,
                 f"CWT_{series_name}_peak_ratio": np.nan,
                 f"CWT_{series_name}_energy_open": np.nan,
                 f"CWT_{series_name}_energy_mid": np.nan,
                 f"CWT_{series_name}_energy_late": np.nan,
                 f"CWT_{series_name}_temporal_compactness": np.nan,
                 f"CWT_{series_name}_ridge_count": 0,
                 f"CWT_{series_name}_ridge_avg_period_min": np.nan,
                 f"CWT_{series_name}_ridge_period_std": np.nan}
        return feats, []

    # 计算 CWT
    scales, _ = build_scales_periods(period_min, period_max, num_scales, wavelet, dt)
    power, periods, freqs = cwt_power(x, scales, wavelet, dt)  # power: [S, T]
    valid = coi_mask(scales, N, k=np.sqrt(2.0))
    P = np.where(valid, power, 0.0)
    total_energy = P.sum()
    if not np.isfinite(total_energy) or total_energy <= 0:
        feats = {f"CWT_{series_name}_power_lo": np.nan,
                 f"CWT_{series_name}_power_mid": np.nan,
                 f"CWT_{series_name}_power_hi": np.nan,
                 f"CWT_{series_name}_centroid_period_min": np.nan,
                 f"CWT_{series_name}_flatness": np.nan,
                 f"CWT_{series_name}_peak_period_min": np.nan,
                 f"CWT_{series_name}_peak_time_min_since_open": np.nan,
                 f"CWT_{series_name}_peak_ratio": np.nan,
                 f"CWT_{series_name}_energy_open": np.nan,
                 f"CWT_{series_name}_energy_mid": np.nan,
                 f"CWT_{series_name}_energy_late": np.nan,
                 f"CWT_{series_name}_temporal_compactness": np.nan,
                 f"CWT_{series_name}_ridge_count": 0,
                 f"CWT_{series_name}_ridge_avg_period_min": np.nan,
                 f"CWT_{series_name}_ridge_period_std": np.nan}
        return feats, []

    # 分带能量占比
    masks = band_masks_by_period(periods)
    e_lo = P[masks["lo"], :].sum()
    e_mid = P[masks["mid"], :].sum()
    e_hi = P[masks["hi"], :].sum()
    power_lo = float(e_lo / total_energy)
    power_mid = float(e_mid / total_energy)
    power_hi = float(e_hi / total_energy)

    # 质心（用频率加权，再转周期）
    f_all = np.where(P > 0, np.repeat(freqs[:, None], N, axis=1), 0.0)
    centroid_freq = float((f_all * P).sum() / total_energy) if total_energy > 0 else np.nan
    centroid_period = safe_period_from_freq(centroid_freq)

    # 平坦度（对有效区域向量化）
    flatness = spectral_flatness(P[P > 0])

    # 全局主峰（有效区域）
    idx = np.unravel_index(np.argmax(P), P.shape)
    peak_s, peak_t = int(idx[0]), int(idx[1])
    peak_period = float(periods[peak_s])
    peak_ratio = float(P[peak_s, peak_t] / total_energy)
    peak_time_min = float(peak_t)  # since open

    # 时间分布与紧致度
    energy_t = P.sum(axis=0)  # 按时间聚合能量
    e_open, e_mid, e_late = segment_energy_fractions(energy_t, N, open_end=60, mid_end=300)
    compactness = temporal_compactness_gini(energy_t)

    # 粗粒度“脊线”检测：以“主导尺度能量”+分位阈值的连通片近似
    dom_scale_idx = np.argmax(P, axis=0)              # 每个时间点的主导尺度
    dom_power = P[dom_scale_idx, np.arange(N)]        # 对应能量
    thr = float(np.percentile(P[P > 0], ridge_percentile)) if (P > 0).any() else np.inf
    strong = dom_power >= thr
    ridges = []  # 明细：每条脊的 {t_start, t_end, avg_period, max_power}
    if np.isfinite(thr):
        i = 0
        while i < N:
            if strong[i]:
                j = i + 1
                while j < N and strong[j]:
                    j += 1
                t_slice = slice(i, j)
                # 该段的平均周期（按 dom_scale_idx 映射）
                sc = dom_scale_idx[t_slice]
                avg_period = float(np.mean(periods[sc])) if sc.size else np.nan
                max_pow = float(np.max(dom_power[t_slice])) if sc.size else np.nan
                ridges.append({
                    "t_start": int(i),
                    "t_end": int(j - 1),
                    "avg_period_min": avg_period,
                    "max_dom_power": max_pow
                })
                i = j
            else:
                i += 1

    ridge_count = int(len(ridges))
    ridge_avg_period = float(np.mean([r["avg_period_min"] for r in ridges])) if ridges else np.nan
    ridge_period_std = float(np.std([r["avg_period_min"] for r in ridges])) if ridges else np.nan

    feats = {
        f"CWT_{series_name}_power_lo": power_lo,
        f"CWT_{series_name}_power_mid": power_mid,
        f"CWT_{series_name}_power_hi": power_hi,
        f"CWT_{series_name}_centroid_period_min": centroid_period,
        f"CWT_{series_name}_flatness": flatness,
        f"CWT_{series_name}_peak_period_min": peak_period,
        f"CWT_{series_name}_peak_time_min_since_open": peak_time_min,
        f"CWT_{series_name}_peak_ratio": peak_ratio,
        f"CWT_{series_name}_energy_open": e_open,
        f"CWT_{series_name}_energy_mid": e_mid,
        f"CWT_{series_name}_energy_late": e_late,
        f"CWT_{series_name}_temporal_compactness": compactness,
        f"CWT_{series_name}_ridge_count": ridge_count,
        f"CWT_{series_name}_ridge_avg_period_min": ridge_avg_period,
        f"CWT_{series_name}_ridge_period_std": ridge_period_std,
    }

    # 生成脊线明细（按需要写盘）
    ridge_rows = []
    for r in ridges:
        ridge_rows.append({
            "series": series_name,
            "t_start": r["t_start"],
            "t_end": r["t_end"],
            "avg_period_min": r["avg_period_min"],
            "max_dom_power": r["max_dom_power"]
        })

    return feats, ridge_rows


# ------------------ 每个股票的处理 ------------------ #

def build_cwt_for_ticker(ticker: str,
                         in_path: str,
                         out_dir: str,
                         # 低置信阈值
                         cover_ratio_thresh: float = 0.70,
                         gap_ratio_thresh: float = 0.20,
                         max_gap_run_thresh: int = 8,
                         n_thresh: int = 300,
                         # CWT 参数
                         wavelet: str = "morl",
                         period_min: float = 5.0,
                         period_max: float = 120.0,
                         num_scales: int = 36,
                         ridge_percentile: float = 95.0,
                         save_ridges: bool = False):
    print(f"[{ticker}] reading:", in_path)
    df = read_minutes_csv(in_path)

    # 只用 RTH
    if "session" in df.columns:
        df = df.loc[df["session"] == "rth"]

    # 必备列检查
    for c in ["close", "day_id", "is_gap"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {in_path}")

    # day_id 标准化为 tz-aware ET 日期（00:00）
    if not pd.api.types.is_datetime64_any_dtype(df["day_id"]):
        df["day_id"] = pd.to_datetime(df["day_id"], errors="coerce")
    if getattr(df["day_id"].dt, "tz", None) is None:
        df["day_id"] = df["day_id"].dt.tz_localize(NY_TZ)

    # 按 ET 日期分组
    groups = dict(tuple(df.groupby(df["day_id"].dt.date)))

    rows = []
    ridges_all = []

    for d_date, g in groups.items():
        day = pd.Timestamp(d_date).tz_localize(NY_TZ)
        g = g.sort_index()
        N = len(g)
        gap_ratio, max_run = gap_quality_metrics(g)

        # 覆盖度：基于 r_close 的 CWT 有效掩码（不同序列差异通常很小，这里以 r_close 为准）
        # 先计算一次 r_close 的掩码
        rc = np.log(g["close"].astype(float)) - np.log(g["close"].astype(float).shift(1))
        if len(rc) > 0:
            rc.iloc[0] = 0.0
        xr = prep_series_for_cwt(rc)
        scales, _ = build_scales_periods(period_min, period_max, num_scales, wavelet, 1.0)
        valid = coi_mask(scales, N, k=np.sqrt(2.0))
        cover_ratio = float(np.mean(valid)) if valid.size else 0.0

        # 低置信判定
        low_conf = int(
            (cover_ratio < cover_ratio_thresh) or
            (gap_ratio > gap_ratio_thresh) or
            (max_run > max_gap_run_thresh) or
            (N < n_thresh) or
            (np.std(xr) < 1e-8)               # r_close 退化保护
        )

        # 特征：r_close
        feats_rc, ridges_rc = cwt_features_for_series(
            g, "r_close", wavelet, period_min, period_max, num_scales, 1.0, ridge_percentile
        )

        # 特征：volume_z_by_tod（若不存在会返回 NaN）
        feats_vol, ridges_vol = cwt_features_for_series(
            g, "volume_z_by_tod", wavelet, period_min, period_max, num_scales, 1.0, ridge_percentile
        )

        # 日时间戳
        day_start_ts, day_end_ts = et_day_bounds(day)

        row = {
            "ticker": ticker,
            "day_id": day.date(),                # YYYY-MM-DD
            "day_start_ts": day_start_ts,        # 09:30 ET tz-aware
            "day_end_ts": day_end_ts,            # 16:00 ET tz-aware
            "feature_ts": day_end_ts,            # 当日特征计时点
            "CWT_flag_low_conf": low_conf,
            "CWT_gap_ratio": float(gap_ratio),
            "CWT_max_consecutive_gap": int(max_run),
            "CWT_N": int(N),
            "CWT_cover_ratio": float(cover_ratio),
            **feats_rc,
            **feats_vol,
        }
        rows.append(row)

        if save_ridges:
            for r in ridges_rc:
                ridges_all.append({
                    "ticker": ticker,
                    "day_id": day.date(),
                    "series": "r_close",
                    **r
                })
            for r in ridges_vol:
                ridges_all.append({
                    "ticker": ticker,
                    "day_id": day.date(),
                    "series": "volume_z_by_tod",
                    **r
                })

    # 写日级特征
    out_ticker_dir = os.path.join(out_dir, ticker, "cwt")
    os.makedirs(out_ticker_dir, exist_ok=True)
    out_path = os.path.join(out_ticker_dir, "cwt_daily.csv")
    out_df = pd.DataFrame(rows).sort_values(["day_id"])
    out_df.to_csv(out_path, index=False)
    print(f"[{ticker}] CWT → {out_path} (days={len(out_df)})")

    # 写脊线明细（可选）
    if save_ridges and ridges_all:
        ridges_path = os.path.join(out_ticker_dir, "cwt_ridges.csv")
        pd.DataFrame(ridges_all).to_csv(ridges_path, index=False)
        print(f"[{ticker}] ridges → {ridges_path} (rows={len(ridges_all)})")


# ------------------ main ------------------ #

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build daily CWT features per ticker (enhanced; r_close & volume only).")
    ap.add_argument("--input_dir",
        default=r"D:\Data mining project\data_mining_fall_2025\feature_extraction\output\intermediate_dataset",
        help="Path to merged/preprocessed per-ticker minutes (CSV files).")
    ap.add_argument("--out_dir",
        default=r"D:\Data mining project\data_mining_fall_2025\feature_extraction\output\features",
        help="Root output directory. Will create <out_dir>/<ticker>/cwt/*.csv.")
    # CWT 参数
    ap.add_argument("--wavelet", default="morl", help="Wavelet name, e.g., 'morl' or complex 'cmor1.5-1.0'.")
    ap.add_argument("--period_min", type=float, default=5.0, help="Min period (minutes).")
    ap.add_argument("--period_max", type=float, default=120.0, help="Max period (minutes).")
    ap.add_argument("--num_scales", type=int, default=36, help="Number of log-spaced scales.")
    ap.add_argument("--ridge_percentile", type=float, default=95.0, help="Energy percentile for ridge threshold.")
    # 低置信度阈值
    ap.add_argument("--cover_ratio_thresh", type=float, default=0.70, help="CWT coverage threshold for low confidence.")
    ap.add_argument("--gap_ratio_thresh", type=float, default=0.20, help="Gap ratio threshold for low confidence.")
    ap.add_argument("--max_gap_run_thresh", type=int, default=8, help="Max consecutive gap minutes for low confidence.")
    ap.add_argument("--n_thresh", type=float, default=300, help="Min per-day samples for low confidence.")
    # 诊断输出
    ap.add_argument("--save_ridges", action="store_true", help="Save ridge details CSV per ticker.")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, "*.csv")))
    if not files:
        raise SystemExit(f"No CSV files found under --input_dir={args.input_dir}")

    for p in files:
        ticker = os.path.splitext(os.path.basename(p))[0]
        try:
            build_cwt_for_ticker(
                ticker=ticker,
                in_path=p,
                out_dir=args.out_dir,
                cover_ratio_thresh=args.cover_ratio_thresh,
                gap_ratio_thresh=args.gap_ratio_thresh,
                max_gap_run_thresh=args.max_gap_run_thresh,
                n_thresh=args.n_thresh,
                wavelet=args.wavelet,
                period_min=args.period_min,
                period_max=args.period_max,
                num_scales=args.num_scales,
                ridge_percentile=args.ridge_percentile,
                save_ridges=args.save_ridges
            )
        except Exception as e:
            print(f"[{ticker}] ERROR: {e}")
