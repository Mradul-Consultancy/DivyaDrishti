"""
Quantifies the donor-identity overlap of the identity_primary split.

`identity_primary` keeps the PRIMARY identity of a face swap strictly disjoint
across train/val/test - that is enforced and audited (the embedded
``zero_leakage_audit`` in the shipped manifests). The DONOR identity of a swap
(idX_idY where idY donates their face to idX) is treated as advisory: strict
disjointness is unachievable on Celeb-DF v2 because its chained swap graph
(id0->id16->id17->...) would collapse nearly every identity into one cluster.

Advisory does not mean irrelevant: if a test video's donor face appeared in
training, the model may recognise the donor's anatomy rather than reading
synthesis artefacts, which makes in-domain numbers optimistic by an unknown
margin. This script measures exactly that exposure and writes it to
``reports/donor_overlap_<domain>.json`` so the caveat can be quoted with
numbers instead of hand-waving.

Artefact read : data/processed/<domain>/video_split_manifest.json
Artefact written: reports/donor_overlap_<domain>.json
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

REPORTS = ROOT / "reports"


def identity_keys(video_name: str):
    """Same conventions as IdentitySourceSplitter.extract_identity_or_source_key."""
    stem = Path(video_name).stem
    m = re.match(r"^(id\d+)_(id\d+)", stem, re.IGNORECASE)
    if m:
        return m.group(1).lower(), m.group(2).lower()
    m = re.match(r"^(id\d+)", stem, re.IGNORECASE)
    if m:
        return m.group(1).lower(), None
    m = re.match(r"^(\d{3,4})_(\d{3,4})", stem)
    if m:
        return m.group(1), m.group(2)
    m = re.match(r"^(\d{3,4})$", stem)
    if m:
        return m.group(1), None
    return stem.lower(), None


def source_video_exposure(domain: str, splits, manifest) -> dict:
    """
    DFDC's sanitised mirror renames videos to opaque hashes, so the donor of a
    swap is not recoverable from filenames. What the shipped metadata DOES
    record is ``original`` - the real video a fake was made from - so the
    measurable leakage axis for DFDC is source-video overlap: a test fake
    whose underlying real video sits in train exposes the target's real face
    to the model. The donor identity itself is unmeasurable from the shipped
    data and is reported as such rather than guessed.
    """
    meta_path = ROOT / "data" / "raw_videos" / "dfdc" / "metadata.json"
    if not meta_path.exists():
        return {}
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    split_of_real = {}
    for split, classes in splits.items():
        for class_name, bucket in classes.items():
            if class_name.lower() != "real":
                continue
            for v in bucket.get("videos", []):
                name = v.get("video_file") or v.get("name") or ""
                split_of_real[name] = split

    out = {}
    for fake_split in ("test", "val"):
        rows = []
        for class_name, bucket in (splits.get(fake_split) or {}).items():
            if class_name.lower() != "fake":
                continue
            rows = bucket.get("videos", [])
        n = len(rows)
        exposed = 0
        unknown = 0
        for v in rows:
            name = v.get("video_file") or v.get("name") or ""
            original = (metadata.get(name) or {}).get("original")
            holder = split_of_real.get(original)
            if holder is None:
                unknown += 1
            elif holder != fake_split:
                exposed += 1
        out[f"{fake_split}_source_video_in_other_split"] = {
            "fake_videos_in_split": n,
            "fake_videos_with_source_video_in_other_split": exposed,
            "source_video_split_unknown": unknown,
            "video_share": round(exposed / n, 4) if n else None,
        }
    return out


def measure(domain: str) -> dict:
    manifest_path = ROOT / "data" / "processed" / domain / "video_split_manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = manifest.get("splits") or {}

    # Primary and donor key sets per split, and per-video donor exposure for
    # fake videos (real videos have no donor).
    primaries = defaultdict(set)
    donors = defaultdict(set)
    # (split -> [(video, primary, donor, faces_saved)]) for fake videos only
    fake_rows = defaultdict(list)
    for split, classes in splits.items():
        for class_name, bucket in classes.items():
            for v in bucket.get("videos", []):
                name = v.get("video_file") or v.get("name") or ""
                k1, k2 = identity_keys(name)
                primaries[split].add(k1)
                if k2:
                    donors[split].add(k2)
                if class_name.lower() == "fake" and k2:
                    fake_rows[split].append(
                        (name, k1, k2, int(v.get("faces_saved") or 1))
                    )

    seen_anywhere = primaries | donors  # any trace of an identity in that split
    overlap_pairs = {
        f"{a}_{b}": sorted(primaries[a] & primaries[b]) for a, b in
        (("train", "val"), ("train", "test"), ("val", "test"))
    }
    donor_pairs = {
        f"{a}_{b}": sorted(donors[a] & donors[b]) for a, b in
        (("train", "val"), ("train", "test"), ("val", "test"))
    }

    result = {
        "domain": domain,
        "split_mode": manifest.get("split_mode"),
        "primary_identity_overlap": overlap_pairs,
        "primary_overlap_is_empty": all(not v for v in overlap_pairs.values()),
        "donor_identity_overlap_ids": donor_pairs,
        "donor_exposure": {},
    }

    # Impact: of the fake videos in split B, how many have a donor whose face
    # appears anywhere in split A (as a primary or as a donor)?
    for donor_split, holder_split in (("test", "train"), ("val", "train"), ("test", "val")):
        holder_ids = seen_anywhere[holder_split]
        rows = fake_rows.get(donor_split, [])
        n_videos = len(rows)
        n_faces = sum(r[3] for r in rows)
        exposed_v = sum(1 for _, _k1, k2, _f in rows if k2 in holder_ids)
        exposed_f = sum(f for _, _k1, k2, f in rows if k2 in holder_ids)
        result["donor_exposure"][f"{donor_split}_donors_seen_in_{holder_split}"] = {
            "fake_videos_in_split": n_videos,
            "fake_videos_with_donor_in_other_split": exposed_v,
            "video_share": round(exposed_v / n_videos, 4) if n_videos else None,
            "faces_share": round(exposed_f / n_faces, 4) if n_faces else None,
        }

    if domain == "dfdc":
        result["source_video_exposure"] = source_video_exposure(domain, splits, manifest)

    audit = manifest.get("zero_leakage_audit") or {}
    result["shipped_audit_flags"] = {
        "zero_leakage_guaranteed": audit.get("zero_leakage_guaranteed"),
        "swap_partner_overlap_is_advisory": audit.get("swap_partner_overlap_is_advisory"),
    }

    # Size of the held-out evaluation directory (what the in-domain test
    # columns of the cross-dataset table actually score). The manifest's
    # split bookkeeping and the image directory can differ (e.g. DFDC keeps
    # a fixed 30/30 image test directory even though its manifest test
    # bucket is empty), so both facts are recorded side by side.
    test_dir = ROOT / "data" / "processed" / domain / "test"
    result["test_dir"] = {
        str(sub.name): sum(1 for _ in sub.iterdir() if sub.is_dir())
        for sub in (test_dir / "real", test_dir / "fake")
        if sub.is_dir()
    }
    return result


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    for domain in ("celebdf", "dfdc"):
        result = measure(domain)
        if not result:
            print(f"[skip] {domain}: no manifest")
            continue
        out = REPORTS / f"donor_overlap_{domain}.json"
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\n=== {domain} ===")
        print("  primary overlap empty:", result["primary_overlap_is_empty"])
        for key, val in result["donor_exposure"].items():
            print(f"  {key}: {val['video_share']*100 if val['video_share'] is not None else 0:.1f}% of fake videos "
                  f"({val['fake_videos_with_donor_in_other_split']}/{val['fake_videos_in_split']})")
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
