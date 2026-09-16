import numpy as np

from strawberryfields import dtw_core
from strawberryfields.pitch import center


def dtw(query, reference, subseq=True, backtrack=False, global_constraints=False, band_rad=0.25):
    query = np.asarray(query, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    return dtw_core.dtw(query, reference, subseq, backtrack, global_constraints, band_rad)


def dtw_score(query, reference):
    query = center(np.asarray(query, dtype=np.float64))
    reference = center(np.asarray(reference, dtype=np.float64))
    if len(query) == 0 or len(reference) == 0:
        return float("inf")

    return dtw_core.subsequence_dtw_cost(query, reference)


def search(query_contour, tracks, top_k):
    results = []
    for track_id, contour in tracks:
        if contour is None or len(contour) == 0:
            continue
        score = dtw_score(query_contour, contour)
        results.append((track_id, score))

    results.sort(key=lambda row: row[1])
    return results[:top_k]
