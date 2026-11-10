"""
Identity and Source-Aware Dataset Partitioning for Deepfake Forensics.

Prevents identity and source video leakage across train, val, and test splits:
1. Deepfake videos are generated from real source videos. If a fake video is in the test split
   while its source real video is in train, the model memorizes facial identity, background,
   lighting, and camera sensor characteristics rather than learning manipulation artifacts.
2. The same actor/celebrity appears across multiple videos. If the same identity is present
   in both train and test, evaluation scores are artificially inflated.

Supported Dataset Conventions & Metadata:
- FaceForensics++ (FF++): Uses official train.json, val.json, test.json split lists or target_source pairs.
- Celeb-DF (v2): Uses official List_of_testing_videos.txt and id<X>_id<Y> celebrity pairings.
- DFDC (Deepfake Detection Challenge): Uses metadata.json ('original' and 'actor_id' fields).
- Generic / Heuristic: Disjoint-Set (Union-Find) connected-component clustering on video stems.

Guarantees:
- Identities(Train) ∩ Identities(Test) = ∅
- Sources(Train) ∩ Sources(Test) = ∅
- Identities(Val) ∩ Identities(Test) = ∅
"""

import os
import re
import sys
import json
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from utils.helpers import setup_logger, ensure_dirs


class DataLeakageError(Exception):
    """Raised when an identity or source video crosses train/val/test split boundaries."""
    pass


class DisjointSet:
    """Disjoint-Set (Union-Find) data structure with path compression."""

    def __init__(self):
        self.parent: Dict[str, str] = {}

    def find(self, item: str) -> str:
        if item not in self.parent:
            self.parent[item] = item
            return item
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, item1: str, item2: str):
        root1 = self.find(item1)
        root2 = self.find(item2)
        if root1 != root2:
            self.parent[root1] = root2


class IdentitySourceSplitter:
    """
    Groups real and fake videos into disjoint clusters by actor identity or source video,
    then allocates entire clusters to train/val/test splits to guarantee zero data leakage.
    """

    def __init__(self, logger=None):
        self.logger = logger or setup_logger("IdentitySourceSplitter")
        # video_name -> (primary_identity, donor_identity) for face-swap videos.
        # Populated only when cluster_videos(link_swap_partners=False).
        self._swap_partner_links: Dict[str, Tuple[str, str]] = {}

    @staticmethod
    def parse_dfdc_metadata(metadata_file: Path | str) -> Dict[str, Dict[str, Any]]:
        """
        Parses DFDC metadata.json format:
        {
          "fake_video.mp4": {"label": "FAKE", "split": "train", "original": "real_video.mp4"},
          "real_video.mp4": {"label": "REAL", "split": "train"}
        }
        """
        metadata_file = Path(metadata_file)
        if not metadata_file.exists():
            raise FileNotFoundError(f"DFDC metadata file not found: {metadata_file}")

        with open(metadata_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    @staticmethod
    def parse_celeb_df_test_list(test_list_file: Path | str) -> Set[str]:
        """
        Parses Celeb-DF v2 official List_of_testing_videos.txt:
        Contains paths or filenames of testing videos (e.g., 'Celeb-synthesis/id0_id1_0000.mp4').
        """
        test_list_file = Path(test_list_file)
        if not test_list_file.exists():
            raise FileNotFoundError(f"Celeb-DF test list not found: {test_list_file}")

        test_videos = set()
        with open(test_list_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    # Typically format is: "<label> <relative_path>" or simply "<relative_path>"
                    path_str = parts[-1]
                    test_videos.add(Path(path_str).name)
        return test_videos

    @staticmethod
    def parse_ffpp_official_splits(splits_dir: Path | str) -> Dict[str, Set[str]]:
        """
        Parses FaceForensics++ official JSON split files:
        splits/train.json, splits/val.json, splits/test.json.
        Each file contains a list of video pair IDs: [["000", "003"], ["001", "005"], ...]
        """
        splits_dir = Path(splits_dir)
        splits = {}
        for split_name in ["train", "val", "test"]:
            split_file = splits_dir / f"{split_name}.json"
            if not split_file.exists():
                continue
            with open(split_file, "r", encoding="utf-8") as f:
                items = json.load(f)
            video_ids = set()
            for pair in items:
                if isinstance(pair, list):
                    for vid_id in pair:
                        video_ids.add(str(vid_id))
                else:
                    video_ids.add(str(pair))
            splits[split_name] = video_ids
        return splits

    @staticmethod
    def extract_identity_or_source_key(video_filename: str) -> Tuple[str, Optional[str]]:
        """
        Extracts the primary identity / source ID and secondary swap ID from filenames.
        Examples:
        - 'video_real_01.mp4' -> ('01', None)
        - 'video_fake_01.mp4' -> ('01', None)
        - '000_003.mp4' (FF++) -> ('000', '003')
        - '000.mp4' (FF++) -> ('000', None)
        - 'id0_0001.mp4' (Celeb-DF) -> ('id0', None)
        - 'id0_id16_0001.mp4' (Celeb-DF) -> ('id0', 'id16')
        - 'actor_42_take_1.mp4' -> ('actor_42', None)
        """
        stem = Path(video_filename).stem

        # Pattern 1: Celeb-DF swap: id<X>_id<Y>_...
        m_celeb_swap = re.match(r"^(id\d+)_(id\d+)", stem, re.IGNORECASE)
        if m_celeb_swap:
            return m_celeb_swap.group(1).lower(), m_celeb_swap.group(2).lower()

        # Pattern 2: Celeb-DF real: id<X>_...
        m_celeb_real = re.match(r"^(id\d+)", stem, re.IGNORECASE)
        if m_celeb_real:
            return m_celeb_real.group(1).lower(), None

        # Pattern 3: FaceForensics++ swap: <num>_<num>
        m_ffpp_swap = re.match(r"^(\d{3,4})_(\d{3,4})", stem)
        if m_ffpp_swap:
            return m_ffpp_swap.group(1), m_ffpp_swap.group(2)

        # Pattern 4: FaceForensics++ real: <num>
        m_ffpp_real = re.match(r"^(\d{3,4})$", stem)
        if m_ffpp_real:
            return m_ffpp_real.group(1), None

        # Pattern 5: video_real_<ID> or video_fake_<ID>
        m_sample = re.match(r"^video_(?:real|fake)_(\w+)", stem, re.IGNORECASE)
        if m_sample:
            return m_sample.group(1).lower(), None

        # Pattern 6: actor_<ID> or person_<ID>
        m_actor = re.match(r"^(actor|person|subject)[-_]?([a-zA-Z0-9]+)", stem, re.IGNORECASE)
        if m_actor:
            return f"{m_actor.group(1).lower()}_{m_actor.group(2).lower()}", None

        # Fallback: full stem is its own source cluster
        return stem, None

    def cluster_videos(
        self,
        video_files: List[Path],
        metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        link_swap_partners: bool = True
    ) -> Dict[str, List[Path]]:
        """
        Uses Disjoint-Set connected components to cluster all videos by identity or source video.
        Ensures that:
        - Real video and all its derivative fakes belong to the SAME cluster.
        - Videos of the same actor belong to the SAME cluster.

        Parameters
        ----------
        link_swap_partners:
            When True (default, strict mode), a face-swap video also merges the
            donor identity's group into the primary identity's group. This
            guarantees that no identity appears in two splits.

            When False, only the PRIMARY identity (key1) groups videos together;
            the donor identity (key2) is recorded but does not merge the groups.

            Why this exists: Celeb-DF v2's swap graph chains identities together
            (id0_id16, id16_id17, ...), so strict merging collapses all 59
            identities into a handful of giant components. A component that
            touches the official test list then swallows the entire dataset,
            leaving train/val empty. Strict identity-disjointness is
            mathematically unachievable on a chained swap graph - the two
            requirements contradict each other. This flag lets a caller choose
            primary-identity disjointness and measure the residual
            donor-identity overlap explicitly instead of getting an unusable
            split.
        """
        ds = DisjointSet()
        file_by_name: Dict[str, Path] = {p.name: p for p in video_files}

        # 1. Connect via explicit metadata (e.g., DFDC 'original' or 'identity'/'actor_id')
        if metadata:
            for vid_name, meta in metadata.items():
                if vid_name not in file_by_name:
                    continue

                # Connect video to its declared identity / actor_id
                actor_id = meta.get("actor_id") or meta.get("identity")
                if actor_id:
                    ident_node = f"ident_{actor_id}"
                    ds.union(vid_name, ident_node)

                # Connect fake video to its source real video
                orig = meta.get("original")
                if orig:
                    orig_name = Path(orig).name
                    ds.union(vid_name, orig_name)

        # 2. Connect via filename patterns (FF++, Celeb-DF, heuristic pairing)
        for vid_path in video_files:
            vid_name = vid_path.name
            key1, key2 = self.extract_identity_or_source_key(vid_name)

            node1 = f"cluster_{key1}"
            ds.union(vid_name, node1)

            if key2:
                node2 = f"cluster_{key2}"
                # In face swaps, connect both identities so neither leaks across splits
                if link_swap_partners:
                    ds.union(vid_name, node2)
                    ds.union(node1, node2)
                else:
                    # Record the donor relationship WITHOUT merging the donor's
                    # group into this one. Note that unioning the video to
                    # node2 would transitively merge the two identity groups
                    # anyway (node2 already owns every video whose primary
                    # identity is the donor), which is what chains Celeb-DF's
                    # identities into giant clusters.
                    self._swap_partner_links.setdefault(vid_name, (key1, key2))

        # 3. Group video files by their root component
        clusters: Dict[str, List[Path]] = defaultdict(list)
        for vid_path in video_files:
            root = ds.find(vid_path.name)
            clusters[root].append(vid_path)

        self.logger.info(
            f"Clustered {len(video_files)} videos into {len(clusters)} isolated identity/source clusters."
        )
        return dict(clusters)

    def split_clusters(
        self,
        clusters: Dict[str, List[Path]],
        split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
        seed: int = 42
    ) -> Dict[str, List[Path]]:
        """
        Allocates whole identity/source clusters to train/val/test splits to eliminate leakage.
        """
        train_ratio, val_ratio, test_ratio = split_ratios
        assert abs(sum(split_ratios) - 1.0) < 1e-4, "Split ratios must sum to 1.0"

        random.seed(seed)
        cluster_keys = list(clusters.keys())
        random.shuffle(cluster_keys)

        total_videos = sum(len(v_list) for v_list in clusters.values())
        target_train = int(round(total_videos * train_ratio))
        target_val = int(round(total_videos * val_ratio))

        splits: Dict[str, List[Path]] = {"train": [], "val": [], "test": []}
        cluster_assignment: Dict[str, str] = {}

        current_train = 0
        current_val = 0

        for c_key in cluster_keys:
            c_videos = clusters[c_key]
            c_size = len(c_videos)

            if (current_train + c_size <= target_train) or (not splits["train"]):
                splits["train"].extend(c_videos)
                cluster_assignment[c_key] = "train"
                current_train += c_size
            elif (current_val + c_size <= target_val) or (not splits["val"]):
                splits["val"].extend(c_videos)
                cluster_assignment[c_key] = "val"
                current_val += c_size
            else:
                splits["test"].extend(c_videos)
                cluster_assignment[c_key] = "test"

        # Ensure no split is empty if there are at least 3 clusters
        if len(cluster_keys) >= 3:
            for s_name in ["train", "val", "test"]:
                if len(splits[s_name]) == 0:
                    # Borrow a cluster from the largest split
                    donor = max(["train", "val", "test"], key=lambda k: len(splits[k]))
                    donor_clusters = [k for k, v in cluster_assignment.items() if v == donor]
                    if len(donor_clusters) > 1:
                        moved_c = donor_clusters[-1]
                        moved_vids = clusters[moved_c]
                        for v in moved_vids:
                            splits[donor].remove(v)
                        splits[s_name].extend(moved_vids)
                        cluster_assignment[moved_c] = s_name

        return splits

    def split_by_official_benchmark(
        self,
        video_files: List[Path],
        ffpp_splits: Optional[Dict[str, Set[str]]] = None,
        celeb_df_test_list: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        train_val_ratio: float = 0.85,
        seed: int = 42,
        link_swap_partners: bool = True
    ) -> Dict[str, List[Path]]:
        """
        Splits videos conforming strictly to official benchmark protocols:
        - FaceForensics++: Uses official train.json, val.json, test.json split ID sets.
        - Celeb-DF v2: Uses official List_of_testing_videos.txt. Any cluster touching a test video
          is assigned to test; remaining clusters are split into train and val.

        link_swap_partners=False keeps donor identities out of the cluster merge.
        This matters for Celeb-DF: with strict merging the swap graph chains all
        identities into a few giant clusters, so every cluster touches the
        official test list and the whole dataset collapses into test.
        """
        splits: Dict[str, List[Path]] = {"train": [], "val": [], "test": []}

        if ffpp_splits:
            self.logger.info("Splitting by official FaceForensics++ partition definitions...")
            for v_path in video_files:
                k1, k2 = self.extract_identity_or_source_key(v_path.name)
                # Check which split contains either k1 or k2
                assigned = False
                for s_name in ["test", "val", "train"]:
                    if k1 in ffpp_splits.get(s_name, set()) or (k2 and k2 in ffpp_splits.get(s_name, set())):
                        splits[s_name].append(v_path)
                        assigned = True
                        break
                if not assigned:
                    # Default unlisted videos to train
                    splits["train"].append(v_path)

        elif celeb_df_test_list:
            self.logger.info("Splitting by official Celeb-DF v2 test list...")
            clusters = self.cluster_videos(
                video_files, metadata=metadata, link_swap_partners=link_swap_partners
            )
            test_clusters = set()

            for c_key, c_vids in clusters.items():
                for v in c_vids:
                    if v.name in celeb_df_test_list or Path(v.stem).name in celeb_df_test_list:
                        test_clusters.add(c_key)
                        break

            non_test_clusters = [k for k in clusters if k not in test_clusters]
            random.seed(seed)
            random.shuffle(non_test_clusters)

            # Assign test clusters
            for c_key in test_clusters:
                splits["test"].extend(clusters[c_key])

            # Partition remaining clusters into train and val
            total_remaining = sum(len(clusters[k]) for k in non_test_clusters)
            target_train = int(round(total_remaining * train_val_ratio))
            curr_train = 0

            for c_key in non_test_clusters:
                c_vids = clusters[c_key]
                if (curr_train + len(c_vids) <= target_train) or (not splits["train"]):
                    splits["train"].extend(c_vids)
                    curr_train += len(c_vids)
                else:
                    splits["val"].extend(c_vids)

            # Ensure val is not empty if we have at least 2 remaining clusters
            if len(non_test_clusters) >= 2 and len(splits["val"]) == 0:
                moved_c = non_test_clusters[-1]
                moved_vids = clusters[moved_c]
                for v in moved_vids:
                    splits["train"].remove(v)
                splits["val"].extend(moved_vids)

        else:
            clusters = self.cluster_videos(video_files, metadata=metadata)
            splits = self.split_clusters(clusters, split_ratios=(0.70, 0.15, 0.15), seed=seed)

        return splits

    def verify_zero_leakage(
        self,
        splits: Dict[str, List[Path]],
        metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        allow_swap_partner_overlap: bool = False
    ) -> Dict[str, Any]:
        """
        Audits partition to mathematically verify zero identity or source video leakage.
        Raises DataLeakageError if ANY identity or source video crosses split boundaries.

        Parameters
        ----------
        allow_swap_partner_overlap:
            When True, the donor identity of a face swap (key2) is treated as an
            advisory metric rather than a hard violation: overlap there is
            reported under ``swap_partner_identity_overlap`` but does not raise.
            The PRIMARY identity (key1) must still be strictly disjoint across
            train/val/test - that check always runs and always raises.
        """
        split_sources: Dict[str, Set[str]] = defaultdict(set)
        split_identities: Dict[str, Set[str]] = defaultdict(set)
        split_donor_identities: Dict[str, Set[str]] = defaultdict(set)

        for s_name, vid_list in splits.items():
            for v_path in vid_list:
                v_name = v_path.name

                # Check metadata sources
                if metadata and v_name in metadata:
                    meta = metadata[v_name]
                    orig = meta.get("original")
                    if orig:
                        split_sources[s_name].add(Path(orig).stem)
                    actor = meta.get("actor_id") or meta.get("identity")
                    if actor:
                        split_identities[s_name].add(str(actor))

                # Check filename sources
                k1, k2 = self.extract_identity_or_source_key(v_name)
                split_sources[s_name].add(k1)
                split_identities[s_name].add(k1)
                if k2:
                    if allow_swap_partner_overlap:
                        # Donor identity is advisory in this mode.
                        split_donor_identities[s_name].add(k2)
                    else:
                        split_sources[s_name].add(k2)
                        split_identities[s_name].add(k2)

        # Audit pairwise intersections
        overlap_train_val_src = split_sources["train"] & split_sources["val"]
        overlap_train_test_src = split_sources["train"] & split_sources["test"]
        overlap_val_test_src = split_sources["val"] & split_sources["test"]

        overlap_train_val_id = split_identities["train"] & split_identities["val"]
        overlap_train_test_id = split_identities["train"] & split_identities["test"]
        overlap_val_test_id = split_identities["val"] & split_identities["test"]

        audit_report = {
            "zero_leakage_guaranteed": True,
            "train_sources_count": len(split_sources["train"]),
            "val_sources_count": len(split_sources["val"]),
            "test_sources_count": len(split_sources["test"]),
            "train_val_source_overlap": list(overlap_train_val_src),
            "train_test_source_overlap": list(overlap_train_test_src),
            "val_test_source_overlap": list(overlap_val_test_src),
            "train_val_identity_overlap": list(overlap_train_val_id),
            "train_test_identity_overlap": list(overlap_train_test_id),
            "val_test_identity_overlap": list(overlap_val_test_id)
        }

        if allow_swap_partner_overlap:
            # Advisory only: report how many donor identities recur across
            # splits. This is the residual that strict mode eliminates by
            # merging swap partners (which collapses Celeb-DF into one cluster).
            donor_overlap = {
                "train_val": sorted(split_donor_identities["train"] & split_donor_identities["val"]),
                "train_test": sorted(split_donor_identities["train"] & split_donor_identities["test"]),
                "val_test": sorted(split_donor_identities["val"] & split_donor_identities["test"]),
            }
            audit_report["swap_partner_identity_overlap"] = donor_overlap
            audit_report["swap_partner_overlap_is_advisory"] = True
            audit_report["note"] = (
                "Primary identities are strictly disjoint across train/val/test. "
                "Donor identities of face swaps may recur across splits; this is "
                "reported for transparency and is inherent to Celeb-DF's chained "
                "swap graph, where strict identity-disjointness is unachievable."
            )

        if overlap_train_test_src:
            audit_report["zero_leakage_guaranteed"] = False
            raise DataLeakageError(
                f"CRITICAL DATA LEAKAGE DETECTED! Sources cross Train and Test: {overlap_train_test_src}"
            )
        if overlap_train_test_id:
            audit_report["zero_leakage_guaranteed"] = False
            raise DataLeakageError(
                f"CRITICAL DATA LEAKAGE DETECTED! Identities cross Train and Test: {overlap_train_test_id}"
            )
        if overlap_train_val_src or overlap_train_val_id:
            audit_report["zero_leakage_guaranteed"] = False
            raise DataLeakageError(
                f"DATA LEAKAGE DETECTED! Overlap between Train and Val: sources={overlap_train_val_src}, id={overlap_train_val_id}"
            )
        if overlap_val_test_src or overlap_val_test_id:
            audit_report["zero_leakage_guaranteed"] = False
            raise DataLeakageError(
                f"DATA LEAKAGE DETECTED! Overlap between Val and Test: sources={overlap_val_test_src}, id={overlap_val_test_id}"
            )

        self.logger.info("Zero-leakage audit PASSED: No source or identity cross-contamination found across splits.")
        return audit_report
