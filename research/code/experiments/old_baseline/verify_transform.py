"""Check the fast re-implementation against the literal notebook transform on one signal-day slice.

The literal version is copied verbatim from baseline/inference.ipynb (transform_events +
get_all_valid_sequences); the fast version is src/old_baseline/run_old_baseline.py.
"""
import glob
import os
import sys

import numpy as np
import polars as pl

from pathlib import Path
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402  -- research/code/** and model/ on sys.path
from run_old_baseline import CACHE, FEATS, MAX_DELTA, MIN_ACT, SEQ, detector_windows, phase_state_frame  # noqa


def transform_events_literal(df: pl.DataFrame, max_parameter: int = 64) -> pl.DataFrame:
    df = df.filter(
        (pl.col('EventId').is_in([1, 7, 43, 44, 81, 82])) & (pl.col('Parameter') <= max_parameter)
    ).with_columns([pl.col('EventId').cast(pl.Int8), pl.col('Parameter').cast(pl.Int8)])
    device_id = df.select('DeviceId').unique()
    detector_events = (df.filter(pl.col('EventId').is_in([81, 82]))
                       .with_columns(pl.col('Parameter').alias('Detector')))
    phase_events = df.filter(pl.col('EventId').is_in([1, 7, 43, 44]))
    min_timestamp = df['Timestamp'].min()
    initial_phase_events = []
    for phase in range(1, 10):
        base_df = device_id.with_columns([
            pl.lit(min_timestamp).alias('Timestamp'),
            pl.lit(phase).cast(pl.Int8).alias('Parameter')])
        initial_phase_events.append(base_df.with_columns([pl.lit(7).cast(pl.Int8).alias('EventId')]))
        initial_phase_events.append(base_df.with_columns([pl.lit(44).cast(pl.Int8).alias('EventId')]))
    initial_phase_events = pl.concat(initial_phase_events).select(['DeviceId', 'Timestamp', 'EventId', 'Parameter'])
    phase_events = pl.concat([phase_events, initial_phase_events])
    device_detectors = detector_events.select(['DeviceId', 'Detector']).unique()
    expanded_phases = phase_events.join(device_detectors, on='DeviceId', how='inner')
    combined_df = pl.concat([detector_events, expanded_phases]).sort(['DeviceId', 'Detector', 'Timestamp', 'EventId'])
    result_df = combined_df.with_columns([
        pl.when(pl.col('EventId') == 82).then(1.0).when(pl.col('EventId') == 81).then(0.0)
        .otherwise(None).cast(pl.Float32).alias('DetectorState')])
    for phase in range(1, 10):
        result_df = result_df.with_columns([
            pl.when((pl.col('Parameter') == phase) & (pl.col('EventId') == 1)).then(1.0)
            .when((pl.col('Parameter') == phase) & (pl.col('EventId') == 7)).then(0.0)
            .otherwise(None).cast(pl.Float32).alias(f'PhaseGreen{phase}'),
            pl.when((pl.col('Parameter') == phase) & (pl.col('EventId') == 43)).then(1.0)
            .when((pl.col('Parameter') == phase) & (pl.col('EventId') == 44)).then(0.0)
            .otherwise(None).cast(pl.Float32).alias(f'PhaseCall{phase}')])
    fill_columns = ['DetectorState'] + [f'PhaseGreen{p}' for p in range(1, 10)] + \
                   [f'PhaseCall{p}' for p in range(1, 10)]
    result_df = result_df.with_columns([pl.col(c).forward_fill().over(['DeviceId', 'Detector']) for c in fill_columns])
    result_df = (result_df.with_columns([
        (pl.col('Timestamp').diff().over(['DeviceId', 'Detector']) / pl.duration(nanoseconds=1_000_000_000))
        .cast(pl.Float32).alias('Delta')])
        .select(['DeviceId', 'Detector', 'Timestamp', *FEATS]).drop_nulls())
    return result_df.sort(['DeviceId', 'Detector', 'Timestamp'])


def literal_windows(t: pl.DataFrame):
    out = {}
    for detector in t.select('Detector').unique().to_numpy().flatten():
        dd = t.filter(pl.col('Detector') == detector)
        if len(dd) < SEQ:
            continue
        seqs = []
        for s in range(0, len(dd) - SEQ + 1, SEQ):
            seq = dd[s:s + SEQ]
            if seq['DetectorState'].sum() >= MIN_ACT and seq['Delta'].max() <= MAX_DELTA:
                seqs.append(seq.drop(['DeviceId', 'Detector', 'Timestamp']).to_numpy())
        if seqs:
            out[int(detector)] = np.stack(seqs)
    return out


if __name__ == '__main__':
    device = sys.argv[1] if len(sys.argv) > 1 else sorted(
        os.path.basename(d).split('=')[1] for d in glob.glob(os.path.join(CACHE, 'DeviceId=*')))[0]
    files = glob.glob(os.path.join(CACHE, f'DeviceId={device}', 'day=3', '*.parquet'))
    df = pl.read_parquet(files).select(['Timestamp', 'EventId', 'Parameter'])
    df = df.filter((pl.col('Timestamp') >= pl.datetime(2024, 12, 3, 9, 0, 0)) &
                   (pl.col('Timestamp') < pl.datetime(2024, 12, 3, 12, 0, 0)))
    lit = literal_windows(transform_events_literal(df.with_columns(pl.lit(device).alias('DeviceId'))
                                                   .select(['DeviceId', 'Timestamp', 'EventId', 'Parameter'])))

    ph = phase_state_frame(df)
    det_ev = df.filter(pl.col('EventId').is_in([81, 82]))
    fast = {}
    for (d,), grp in det_ev.group_by(['Parameter'], maintain_order=True):
        n, res = detector_windows(ph, grp)
        if res is None:
            continue
        arr, keep = res
        if keep.any():
            fast[int(d)] = arr[keep]

    print('device', device, 'literal dets', len(lit), 'fast dets', len(fast))
    assert set(lit) == set(fast), (sorted(set(lit) ^ set(fast)))
    worst = 0.0
    for d in lit:
        assert lit[d].shape == fast[d].shape, (d, lit[d].shape, fast[d].shape)
        worst = max(worst, float(np.abs(lit[d] - fast[d]).max()))
    print('max abs feature diff:', worst, 'total windows:', sum(v.shape[0] for v in lit.values()))
