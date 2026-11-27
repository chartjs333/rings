#!/usr/bin/env python3
"""Validate attached pinned FlyWire v783 files and export the C++ CSR benchmark.

No download, filtering, sign inference, or precision-reducing ID conversion.
Weights match cpp/main.cpp: aggregate signed counts, sort by (post, pre),
divide by max(1, sum(abs(aggregated counts))) for each receiving row.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import pyarrow.parquet as pq
from scipy import sparse

from download_flywire import FILES, verify


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(completeness: Path, connectivity: Path, output: Path):
    verify(completeness, *FILES['Completeness_783.csv'])
    verify(connectivity, *FILES['Connectivity_783.parquet'])
    with completeness.open(newline='') as handle:
        reader = csv.reader(handle)
        if next(reader) != ['', 'Completed']:
            raise ValueError('Unexpected completeness header')
        rows = list(reader)
    if any(len(row) != 2 or not row[0].isdecimal() or row[1] != 'True' for row in rows):
        raise ValueError('Invalid pinned completeness rows')
    ids = np.array([int(row[0]) for row in rows], dtype='<u8')
    if len(np.unique(ids)) != len(ids):
        raise ValueError('Duplicate node IDs')
    n = len(ids)
    source = pq.ParquetFile(connectivity)
    m = source.metadata.num_rows
    pre = np.empty(m, dtype=np.int32)
    post = np.empty(m, dtype=np.int32)
    signed = np.empty(m, dtype=np.int64)
    positive = np.zeros(n, dtype=bool)
    negative = np.zeros(n, dtype=bool)
    columns = ['Presynaptic_ID', 'Postsynaptic_ID', 'Presynaptic_Index',
               'Postsynaptic_Index', 'Connectivity', 'Excitatory',
               'Excitatory x Connectivity']
    offset = 0
    total_synapses = 0
    for batch in source.iter_batches(batch_size=250_000, columns=columns):
        if any(column.null_count for column in batch.columns):
            raise ValueError('Null source value')
        d = {name: batch.column(name).to_numpy() for name in columns}
        if any(not np.issubdtype(v.dtype, np.integer) for v in d.values()):
            raise ValueError('Noninteger source data')
        p, q, w, sign = (d[key] for key in columns[2:6])
        if np.any(p < 0) or np.any(q < 0) or np.any(p >= n) or np.any(q >= n):
            raise ValueError('Source index out of bounds')
        if not np.array_equal(ids[p], d[columns[0]]) or not np.array_equal(ids[q], d[columns[1]]):
            raise ValueError('ID/index mismatch')
        if np.any(w <= 0) or not np.isin(sign, [-1, 1]).all():
            raise ValueError('Invalid counts/signs')
        if not np.array_equal(w * sign, d[columns[6]]):
            raise ValueError('Signed count mismatch')
        positive[np.unique(p[sign == 1])] = True
        negative[np.unique(p[sign == -1])] = True
        stop = offset + len(p)
        pre[offset:stop], post[offset:stop], signed[offset:stop] = p, q, w * sign
        offset = stop
        total_synapses += int(w.sum())
    if offset != m or np.any(positive & negative):
        raise ValueError('Row count mismatch or inconsistent presynaptic signs')
    # Integer aggregation and row sums are exact for this verified dataset.
    if total_synapses >= 2**53:
        raise ValueError('Counts exceed exact binary64 integer range')
    matrix = sparse.csr_matrix((signed, (post, pre)), shape=(n, n))
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    row_norm = np.maximum(1, np.asarray(abs(matrix).sum(axis=1)).ravel())
    weights = matrix.data.astype('<f8')
    weights /= np.repeat(row_norm, np.diff(matrix.indptr))
    if not np.isfinite(weights).all():
        raise ValueError('Nonfinite normalized weight')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('wb') as handle:
        handle.write(b'RINGSCSR')
        handle.write(struct.pack('<QQ', n, matrix.nnz))
        ids.tofile(handle)
        matrix.indptr.astype('<u8').tofile(handle)
        matrix.indices.astype('<u4').tofile(handle)
        weights.tofile(handle)
    metadata = {
        'format': 'RINGSCSR-v1-little-endian',
        'source_kind': 'real FlyWire v783 full supplied table',
        'source_files': {str(p): {'bytes': p.stat().st_size, 'sha256': sha256(p)}
                         for p in (completeness, connectivity)},
        'matches_repository_pinned_source_hashes': True,
        'neurons': n, 'input_edge_rows': m, 'stored_edges': int(matrix.nnz),
        'synapse_count_sum': total_synapses,
        'orientation': 'W[postsynaptic, presynaptic]',
        'normalization': 'signed aggregate / max(1, row absolute sum)',
        'row_order': 'Completeness source order; pre indices sorted within rows',
        'weight_dtype': 'float64', 'id_dtype': 'uint64', 'pre_index_dtype': 'uint32',
        'no_outgoing_edges': int((~(positive | negative)).sum()),
        'no_incoming_edges': int((np.diff(matrix.indptr) == 0).sum()),
        'csr_bytes': output.stat().st_size, 'csr_sha256': sha256(output),
    }
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completeness', type=Path, required=True)
    parser.add_argument('--connectivity', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.completeness, args.connectivity, args.output), indent=2))
