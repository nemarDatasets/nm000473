"""src-generic-152 (OSF 4xjg7, CC-BY-4.0; Qin et al. 2023 Cell Reports doi:10.1016/j.celrep.2023.113432, "Predictability alters
information flow during action observation in human electrocorticographic activity") -> BIDS-iEEG DERIVATIVE (re-referenced,
ROI-channel ECoG, continuous 1000 Hz).
Source: _dropbox/ECoGData/Patient<N>LARER_rerefDataInROI.mat (MAT v7.3 FieldTrip struct rerefData: trial{1} ch x samples, fsample,
label (ROI channels), hdr.chanunit uV, elec: label/chanpos (coordsys acpc, mm) for all contacts + elecposMNI for the ROI channels)
and Patient<N>_eventdata.mat (camInfo: 1590 x 7 cell = camera-change triggers of the action movies). Stimuli/: movies, BORIS csv.
Usage: python o5c_A1_convert_gen152.py <open500 record dir> <bids root>"""
import csv, glob, json, os, re, shutil, sys
import numpy as np, h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from o5c_A1_common import write_run, h5str, h5cellstr, wtsv, wjson, finalize_common, copy_code, copy_sourcedata

REC, OUT = sys.argv[1], sys.argv[2]
SRC = os.path.join(REC, "files", "_dropbox")
ED = os.path.join(SRC, "ECoGData"); ST = os.path.join(SRC, "Stimuli")
report = {"recordings": [], "event_checks": []}
# movie lengths from the BORIS csv files (column 'Total length')
movlen = {}
for p in glob.glob(os.path.join(ST, "*.csv")):
    with open(p, newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            movlen[r["Observation id"]] = float(r["Total length"]); break


def camval(f, r):
    o = f[r]
    if o.attrs.get("MATLAB_class", b"") == b"char":
        return h5str(f, o)
    a = np.asarray(o[()]).ravel()
    return float(a[0]) if a.size == 1 else "n/a"


scans = {}
for p in sorted(glob.glob(os.path.join(ED, "Patient*LARER_rerefDataInROI.mat")), key=lambda x: int(re.search(r"Patient(\d+)", x)[1])):
    pn = int(re.search(r"Patient(\d+)", os.path.basename(p))[1])
    sub = f"sub-{pn:02d}"; D = os.path.join(OUT, sub, "ieeg"); stem = f"{sub}_task-actionobservation"
    with h5py.File(p, "r") as f:
        d = f["rerefData"]
        fs = float(np.asarray(d["fsample"][()]).ravel()[0])
        labels = h5cellstr(f, d["label"])
        T = np.asarray(d["trial"][()]).ravel(); assert T.size == 1, T.size
        X = np.asarray(f[T[0]][()])  # (samples, ch)
        if X.shape[1] != len(labels) and X.shape[0] == len(labels):
            X = X.T
        tm = np.asarray(f[np.asarray(d["time"][()]).ravel()[0]][()]).ravel()
        units_h = h5cellstr(f, d["hdr/chanunit"]); alllab = h5cellstr(f, d["elec/label"])
        coordsys = h5str(f, d["elec/coordsys"]); eunit = h5str(f, d["elec/unit"])
        chanpos = np.asarray(d["elec/chanpos"][()]).T  # (N, 3)
        if chanpos.shape[1] != 3:
            chanpos = chanpos.T
        mni = np.asarray(d["elec/elecposMNI"][()]).T
        if mni.shape[1] != 3:
            mni = mni.T
        elecpos = np.asarray(d["elec/elecpos"][()]).T
        if elecpos.shape[1] != 3:
            elecpos = elecpos.T
    assert X.shape[0] == tm.size and abs((tm[1] - tm[0]) - 1 / fs) < 1e-9, (X.shape, tm[:2], fs)
    assert set(units_h) == {"uV"}, set(units_h)
    assert coordsys == "acpc" and eunit == "mm", (coordsys, eunit)
    assert mni.shape[0] == len(labels), (mni.shape, len(labels))
    # events
    ep = os.path.join(ED, f"Patient{pn}_eventdata.mat")
    rows = []; checks = {"flag_matches_S": True, "duration_matches_csv": True, "n": 0}
    with h5py.File(ep, "r") as f:
        C = np.asarray(f["camInfo"][()])  # (7, nrows)
        for i in range(C.shape[1]):
            v = [camval(f, C[j, i]) for j in range(C.shape[0])]
            flag, mid, mnum, samp, iprev, inext, mdur = v
            checks["n"] += 1
            checks["flag_matches_S"] &= (int(flag) == 1) == str(mid).startswith("S")
            checks["duration_matches_csv"] &= (mid in movlen and abs(movlen[mid] - float(mdur)) < 1e-6)
            s0 = int(samp) - 1  # MATLAB 1-based sample -> 0-based
            rows.append([round(s0 / fs, 6), 0.0, "camera_change", s0, mid, int(mnum), int(flag), iprev, inext, mdur])
    report["event_checks"].append({"sub": sub, **checks})
    markers = [("Stimulus", f"{r[4]}", r[3], 1) for r in rows]
    groups = [re.match(r"([A-Za-z]+)", l)[1] for l in labels]
    r = write_run(D, stem, X, fs, labels, ["µV"] * len(labels), ["ECOG"] * len(labels), {
        "TaskName": "actionobservation",
        "TaskDescription": "Passive observation of movies of sequential actions (three runs, RunA/RunB/RunC; 24 movies M01..M22 / S01..S22 listed in Stimuli/StimulusList_*.txt) while ECoG was recorded; the article contrasts predictable and unpredictable action sequences. Camera-change times within the movies (BORIS annotations in stimuli/) are the released events.",
        "Instructions": "n/a (see article)",
        "PowerLineFrequency": 50,
        "SoftwareFilters": {"AuthorsPreprocessing": {"Description": "Release file '<Patient>LARER_rerefDataInROI.mat': FieldTrip data re-referenced by the authors (montage in cfg.previous; scheme not stated in the release, see article) and restricted to the region-of-interest channels. Values written unchanged (µV per hdr.chanunit)."}},
        "HardwareFilters": "n/a",
        "iEEGReference": "re-referenced by the authors (release file name LARER_reref; see article); original reference not stated in the release",
        "iEEGPlacementScheme": "subdural ECoG grids/strips (labels A.., B..); only the region-of-interest channels of the article are released. Contact positions for all implanted contacts (elec.chanpos, ACPC mm) are in sourcedata; the ROI channels have ACPC and MNI positions in electrodes.tsv.",
        "ElectricalStimulation": False, "Manufacturer": "n/a",
    }, ch_extra=[{"description": "ROI channel of the article (release rerefData.label)", "group": groups[i]} for i in range(len(labels))],
        markers=markers, comment=f"o5c_A1_convert_gen152.py: {os.path.basename(p)} rerefData.trial{{1}} float64 µV -> float32")
    ecols = ["onset", "duration", "trial_type", "sample", "movie_id", "movie_number", "movie_type_flag", "interval_before", "interval_after", "movie_duration"]
    wtsv(os.path.join(D, stem + "_events.tsv"), ecols, rows)
    wjson(os.path.join(D, stem + "_events.json"), {
        "trial_type": {"Description": "camera_change: a camera-change trigger of the action movie (camInfo row)", "Levels": {"camera_change": "camera change in the movie (BORIS 'Trigger' annotation)"}},
        "sample": {"Description": "camInfo column 4 minus 1 (0-based sample of the trigger in the ECoG file; interpretation inferred from the value range 1..nSamples)"},
        "movie_id": {"Description": "camInfo column 2: movie identifier (M01..M22, S01..S22; csv/avi in stimuli/)"},
        "movie_number": {"Description": "camInfo column 3: movie number 1..24 (as released)"},
        "movie_type_flag": {"Description": "camInfo column 1: 1 for S-movies, 0 for M-movies in every row (checked in code/conversion_report.json); meaning of S/M per the article (predictable vs unpredictable sequences)"},
        "interval_before": {"Description": "camInfo column 5 (s); matches the interval from the previous camera change (inferred)", "Units": "s"},
        "interval_after": {"Description": "camInfo column 6 (s); matches the interval to the next camera change (inferred)", "Units": "s"},
        "movie_duration": {"Description": "camInfo column 7: total length of the movie (equals 'Total length' in the BORIS csv; checked in code/conversion_report.json)", "Units": "s"}})
    # electrodes: ACPC (from elec.chanpos matched by label) and MNI (elecposMNI, order of rerefData.label)
    idx = [alllab.index(l) for l in labels]
    wtsv(os.path.join(D, f"{sub}_space-ACPC_electrodes.tsv"), ["name", "x", "y", "z", "size", "group"],
         [[l, "%.4f" % chanpos[i, 0], "%.4f" % chanpos[i, 1], "%.4f" % chanpos[i, 2], "n/a", groups[k]] for k, (l, i) in enumerate(zip(labels, idx))])
    wjson(os.path.join(D, f"{sub}_space-ACPC_coordsystem.json"), {"iEEGCoordinateSystem": "ACPC", "iEEGCoordinateUnits": "mm",
          "iEEGCoordinateSystemDescription": "FieldTrip elec.chanpos with elec.coordsys = 'acpc' (subject's ACPC-aligned anatomical space, mm) as released; positions for all %d implanted contacts are in sourcedata (elec.label/chanpos)." % len(alllab),
          "iEEGCoordinateProcessingDescription": "values copied from elec.chanpos for the ROI channels (matched by label)"})
    wtsv(os.path.join(D, f"{sub}_space-Other_electrodes.tsv"), ["name", "x", "y", "z", "size", "group"],
         [[l, "%.4f" % mni[k, 0], "%.4f" % mni[k, 1], "%.4f" % mni[k, 2], "n/a", groups[k]] for k, l in enumerate(labels)])
    wjson(os.path.join(D, f"{sub}_space-Other_coordsystem.json"), {"iEEGCoordinateSystem": "Other", "iEEGCoordinateUnits": "mm",
          "iEEGCoordinateSystemDescription": "MNI coordinates from FieldTrip elec.elecposMNI (one row per ROI channel, taken in the order of rerefData.label). The MNI template version is not stated in the release (hence 'Other'); normalisation method in the article.",
          "iEEGCoordinateProcessingDescription": "values copied from elec.elecposMNI"})
    r.update(sub=sub, stem=stem, file=os.path.basename(p), labels=labels, n_events=len(rows), n_all_contacts=len(alllab))
    report["recordings"].append(r)
    scans.setdefault(sub, []).append([f"ieeg/{stem}_ieeg.vhdr", "n/a"])
    print(stem, json.dumps({k: v for k, v in r.items() if k != "labels"}), checks, flush=True)
for sub, rows in scans.items():
    wtsv(os.path.join(OUT, sub, f"{sub}_scans.tsv"), ["filename", "acq_time"], sorted(rows))
subs = sorted(scans)
wtsv(os.path.join(OUT, "participants.tsv"), ["participant_id", "age", "sex", "source_id"], [[s, "n/a", "n/a", "Patient%d" % int(s[4:])] for s in subs])
wjson(os.path.join(OUT, "participants.json"), {"age": {"Description": "not in the release"}, "sex": {"Description": "not in the release"}, "source_id": {"Description": "patient number in the release file names"}})
wjson(os.path.join(OUT, "space-ACPC_electrodes.json"), {"SpatialReference": "subject's ACPC-aligned anatomical MRI (not released)"})
wjson(os.path.join(OUT, "space-Other_electrodes.json"), {"SpatialReference": "MNI template (version not stated in the release)"})
# stimuli
SM = os.path.join(OUT, "stimuli"); os.makedirs(SM, exist_ok=True)
for p in sorted(glob.glob(os.path.join(ST, "*"))):
    if os.path.isfile(p) and not p.endswith((".fig", ".eps", ".png", ".mat")):
        shutil.copy(p, os.path.join(SM, os.path.basename(p)))
SD = os.path.join(OUT, "sourcedata", "osf-4xjg7")
copy_sourcedata([(p, os.path.relpath(p, SRC)) for p in sorted(glob.glob(os.path.join(ED, "*.mat")) + glob.glob(os.path.join(ST, "*"))) if not p.endswith(".avi")], SD, scrub=False)
open(os.path.join(SD, "NOTE_avi_in_stimuli.txt"), "w").write("The three movie files (Stimuli/Run{A,B,C}_squareAtBeginningOnly.avi, 5.7 GB) are stored once, in stimuli/, not duplicated here.\n")
copy_code(OUT, [__file__])
Rr = report["recordings"]; hrs = sum(r["duration_s"] for r in Rr) / 3600
finalize_common(OUT,
    name="Qin2023 action observation ECoG: re-referenced region-of-interest electrocorticography during observation of predictable and unpredictable action movies",
    dtype="derivative", license_="CC-BY-4.0",
    authors=["Chaoyi Qin", "Frederic Michon", "Yoshiyuki Onuki", "Yohei Ishishita", "Keisuke Otani", "Kensuke Kawai", "Pascal Fries", "Valeria Gazzola", "Christian Keysers"],
    refs=["https://doi.org/10.1016/j.celrep.2023.113432", "https://doi.org/10.1101/2022.12.13.519734", "https://doi.org/10.17605/OSF.IO/4XJG7"],
    howto="Cite Qin C, Michon F, Onuki Y, Ishishita Y, Otani K, Kawai K, Fries P, Gazzola V, Keysers C (2023). Predictability alters information flow during action observation in human electrocorticographic activity. Cell Reports 42(12):113432, doi:10.1016/j.celrep.2023.113432, and the OSF release doi:10.17605/OSF.IO/4XJG7.",
    keywords=["ECoG", "electrocorticography", "action observation", "predictive coding", "beta", "gamma", "epilepsy", "Japan"],
    source=[{"DOI": "doi:10.17605/OSF.IO/4XJG7", "URL": "https://osf.io/4xjg7/", "Version": "OSF node 4xjg7 (files as downloaded 2026-10-11; sha256 in sourcedata/osf-4xjg7/DEIDENTIFICATION_MANIFEST.tsv)"}],
    generated_by=[{"Name": "o5c_A1_convert_gen152.py", "Description": "Repackaging of the FieldTrip rerefData structs (continuous, µV) into BrainVision float32 with camera-change events, ACPC and MNI positions of the ROI channels, and the stimulus movies/annotations. Code in code/."}],
    ethics=["Collected under the source study's approval (see the article, doi:10.1016/j.celrep.2023.113432)."],
    funding=["Japan Society for the Promotion of Science JP17H04305", "Japan Society for the Promotion of Science JP20H03794", "NWO 452-14-015", "NWO 453-15-009"],
    readme=f"""# Qin et al. 2023, "Predictability alters information flow during action observation in human ECoG": ROI channels

Continuous electrocorticography ({len(subs)} patients, {hrs:.2f} h, 1000 Hz) recorded while epilepsy patients watched movies of
sequential actions, released by the authors on OSF (doi:10.17605/OSF.IO/4XJG7, CC-BY-4.0) as FieldTrip structures of the
re-referenced region-of-interest channels used in the article, plus the event tables and stimuli.

## What is converted (DatasetType derivative)
- `sub-XX/ieeg/sub-XX_task-actionobservation_ieeg.vhdr`: `rerefData.trial{{1}}` of `Patient<N>LARER_rerefDataInROI.mat` (µV per
  `hdr.chanunit`, values unchanged, float32), channels = the released ROI contacts (names as released, e.g. A26, B55). The signals
  are re-referenced by the authors (scheme in the article; not stated in the release) and restricted to ROI channels, hence derivative.
- `events.tsv`: the released `camInfo` table of `Patient<N>_eventdata.mat` (camera-change triggers in the movies): sample, movie id,
  movie number, the S/M flag and the intervals before/after. Column meanings were inferred from the values and checked against the
  BORIS annotation files (see `events.json` and `code/conversion_report.json`).
- `electrodes.tsv`: ACPC positions (`elec.chanpos`, FieldTrip `coordsys` 'acpc') and MNI positions (`elec.elecposMNI`, template
  not stated) for the ROI channels. Positions of all implanted contacts are in the sourcedata MAT files.
- `stimuli/`: the three movie runs (avi), the BORIS camera-change annotations per movie (csv), the stimulus lists and the BORIS
  project file, as released. `sourcedata/osf-4xjg7/`: all release files unchanged.
- Age and sex are not in the release. The patient numbers of the release (1,2,3,4,6,8,9,11,12,13) are kept as subject numbers.

## Privacy
The release holds patient numbers only. FieldTrip `cfg` fields in the MAT files contain the authors' analysis paths/usernames and
analysis timestamps (2022-2023), not patient data.
""",
    changes_note="BIDS conversion of the OSF 4xjg7 release (Qin et al. 2023) for NEMAR (o5c lane A1).")
json.dump(report, open(os.path.join(OUT, "code", "conversion_report.json"), "w"), indent=1, default=str)
print(json.dumps({"recordings": len(Rr), "subjects": len(subs), "hours": hrs, "rt_all": all(r["roundtrip_ok"] for r in Rr), "event_checks": report["event_checks"]}, indent=1))
print("CONVERT_DONE")
