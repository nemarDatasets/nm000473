"""Shared helpers for o5c lane A1 (OSF records) conversions, run inside Voyager Jobs. Builds on b3w10_common / b3w3_common.
write_epochs(): FieldTrip-like trial cells (list of (samples x channels) arrays) -> one BrainVision file, trials back-to-back,
one 'New Segment' marker per trial, events.tsv rows with the trial's position in the file + the authors' trial fields.
h5str(): read a MATLAB char array from an HDF5 (v7.3) file. finalize_common(): dataset_description/README/CHANGES/.bidsignore."""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b3w10_common import write_run, wtsv, wjson, sha256_file, mat_strings, flag_strings, CH_COLS  # noqa: F401
from b3w3_common import write_vhdr, scrub_mat_header, ym01, add_electrodes_na  # noqa: F401

R = os.environ.get("R", "/voyager/ceph/groups/sdp190/bpinto/ieeg-nemar-20261005")
OPEN500 = os.path.join(R, "work", "open500")


def h5str(f, o):
    """MATLAB char dataset (or reference to one) -> str."""
    import h5py
    if not isinstance(o, (h5py.Dataset, h5py.Group)):
        o = f[o]
    a = np.asarray(o[()])
    if a.dtype.kind in "ui" and o.attrs.get("MATLAB_class", b"") == b"char":
        return "".join(chr(int(c)) for c in a.flatten(order="F"))
    if a.dtype.kind in "ui":
        return "".join(chr(int(c)) for c in a.flatten(order="F"))
    return str(a)


def h5cellstr(f, ds):
    return [h5str(f, r) for r in np.asarray(ds[()]).ravel()]


def write_epochs(D, stem, trials, sfreq, names, units, types, sidecar, ev_cols, ev_rows, ch_extra=None, comment="", trial_times=None):
    """trials: list of (n_samples_k x n_ch) arrays. ev_rows[k]: list of extra event columns for trial k (same length as ev_cols).
    Writes <stem>_ieeg.{eeg,vhdr,vmrk,json}, <stem>_channels.tsv, <stem>_events.tsv. Returns report dict."""
    os.makedirs(D, exist_ok=True)
    nch = len(names)
    segs, marks, rows, pos = [], [], [], 0
    maxerr = 0.0
    for k, x in enumerate(trials):
        x = np.asarray(x)
        assert x.ndim == 2 and x.shape[1] == nch, (x.shape, nch)
        x32 = x.astype(np.float32)
        fin = np.isfinite(x)
        if fin.any():
            maxerr = max(maxerr, float(np.max(np.abs(x32[fin].astype(np.float64) - x[fin]))))
        segs.append(x32)
        t0 = float(trial_times[k][0]) if trial_times is not None else "n/a"
        rows.append([round(pos / sfreq, 6), round(x.shape[0] / sfreq, 6), pos, k + 1, t0] + list(ev_rows[k]))
        marks.append(("New Segment", f"trial {k + 1}", pos, 1))
        pos += x.shape[0]
    data = np.concatenate(segs, axis=0)
    base = os.path.join(D, stem + "_ieeg")
    np.ascontiguousarray(data).astype("<f4").tofile(base + ".eeg")
    write_vhdr(base, nch, sfreq, names, units, comment=comment, markers=marks[1:])
    crows = []
    for i, c in enumerate(names):
        r = {"name": c, "type": types[i], "units": units[i], "low_cutoff": "n/a", "high_cutoff": "n/a", "sampling_frequency": sfreq,
             "reference": "n/a", "group": "n/a", "status": "good", "status_description": "n/a", "description": "n/a"}
        if ch_extra:
            r.update(ch_extra[i])
        crows.append([r[k] for k in CH_COLS])
    wtsv(os.path.join(D, stem + "_channels.tsv"), CH_COLS, crows)
    wtsv(os.path.join(D, stem + "_events.tsv"), ["onset", "duration", "sample", "trial_index", "trial_time_start"] + list(ev_cols), rows)
    sc = dict(sidecar)
    sc.setdefault("SamplingFrequency", sfreq)
    sc.setdefault("RecordingDuration", round(data.shape[0] / sfreq, 6))
    sc.setdefault("RecordingType", "epoched")
    sc.setdefault("EpochLength", round(float(np.median([x.shape[0] for x in trials])) / sfreq, 6))
    for key, t in (("ECOGChannelCount", "ECOG"), ("SEEGChannelCount", "SEEG"), ("DBSChannelCount", "DBS"), ("EEGChannelCount", "EEG"),
                   ("EMGChannelCount", "EMG"), ("MiscChannelCount", "MISC"), ("TriggerChannelCount", "TRIG")):
        sc.setdefault(key, sum(1 for x in types if x == t))
    wjson(base + ".json", sc)
    import mne
    rr = mne.io.read_raw_brainvision(base + ".vhdr", preload=False, verbose="error")
    k = min(5000, data.shape[0])
    A = rr.get_data(start=0, stop=k); B = data[:k].T.astype(np.float64)
    ok = rr.n_times == data.shape[0] and len(rr.ch_names) == nch
    if ok:
        for i in range(nch):
            b = B[i]; a = A[i]; m = np.isfinite(b) & (np.abs(b) > 0)
            if not m.any():
                continue
            f = np.median(a[m] / b[m])
            ok = ok and np.allclose(a[np.isfinite(b)], f * b[np.isfinite(b)], rtol=1e-5, atol=1e-12)
    return {"n_trials": len(trials), "n_samples": int(data.shape[0]), "sfreq": sfreq, "n_channels": nch, "float32_max_abs_err": maxerr,
            "roundtrip_ok": bool(ok), "duration_s": data.shape[0] / sfreq}


def events_json(path, desc):
    wjson(path, {"onset": {"Description": "Start of the trial segment in this file (trials are written back-to-back)", "Units": "s"},
                 "duration": {"Description": "Length of the trial segment", "Units": "s"},
                 "sample": {"Description": "First sample of the trial segment in the file (0-based)"},
                 "trial_index": {"Description": "Trial number in the released file (1-based, release order)"},
                 "trial_time_start": {"Description": "Authors' time axis value of the first sample of the trial (s, relative to the trial's event; n/a if not released)", "Units": "s"},
                 **desc})


def finalize_common(OUT, name, dtype, license_, authors, refs, howto, keywords, source, generated_by, readme, changes_note, ethics=None, funding=None, extra=None):
    d = {"Name": name, "BIDSVersion": "1.10.0", "DatasetType": dtype, "License": license_, "Authors": authors,
         "HowToAcknowledge": howto, "ReferencesAndLinks": refs, "Keywords": keywords, "SourceDatasets": source, "GeneratedBy": generated_by}
    if ethics:
        d["EthicsApprovals"] = ethics
    if funding:
        d["Funding"] = funding
    if extra:
        d.update(extra)
    wjson(os.path.join(OUT, "dataset_description.json"), d)
    open(os.path.join(OUT, "README"), "w").write(readme)
    open(os.path.join(OUT, "CHANGES"), "w").write(f"1.0.0 {time.strftime('%Y-%m-%d')}\n  - {changes_note}\n")
    open(os.path.join(OUT, ".bidsignore"), "w").write("code/\n")


def copy_code(OUT, files):
    import shutil
    os.makedirs(os.path.join(OUT, "code"), exist_ok=True)
    H = os.path.dirname(os.path.abspath(__file__))
    for fn in files + ["o5c_A1_common.py", "b3w10_common.py", "b3w3_common.py"]:
        shutil.copy(os.path.join(H, os.path.basename(fn)), os.path.join(OUT, "code", os.path.basename(fn)))


def copy_sourcedata(src_files, SD, scrub=True):
    """Copy release files into sourcedata/ (MAT text-header date day -> 01 when scrub). Returns manifest rows."""
    import hashlib
    rows = []
    for src, rel in src_files:
        dst = os.path.join(SD, rel); os.makedirs(os.path.dirname(dst), exist_ok=True)
        b = open(src, "rb").read()
        ch = False
        if scrub and rel.lower().endswith(".mat") and b[:4] == b"MATL":
            nb, ch = scrub_mat_header(b)
        else:
            nb = b
        open(dst, "wb").write(nb)
        rows.append([rel, len(b), hashlib.sha256(b).hexdigest(), hashlib.sha256(nb).hexdigest(), "MAT text header date -> Mmm 01 yyyy" if ch else "unchanged"])
    wtsv(os.path.join(SD, "DEIDENTIFICATION_MANIFEST.tsv"), ["path", "bytes_original", "sha256_original", "sha256_here", "change"], rows)
    return rows
