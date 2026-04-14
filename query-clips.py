#!/usr/bin/env python3
"""Query analyzed clip libraries by role, tag, descriptor score, or similarity.

Usage examples:
    # Find dark textures
    python3 query-clips.py --library out-clips/co-production --tags dark --roles texture

    # Find the 10 most vocalish clips
    python3 query-clips.py --library out-clips/co-production --sort vocalish --limit 10

    # Find clips similar to a specific clip
    python3 query-clips.py --library out-clips/co-production --similar-to "chopped/piano-wednesday-empty-room.wav"

    # Find clips that contrast with a specific clip
    python3 query-clips.py --library out-clips/co-production --contrast-with "chopped/piano-wednesday-empty-room.wav"

    # Search across multiple libraries
    python3 query-clips.py --library out-clips/co-production --library out-clips/word-library --tags vocalish --sort vocalish

    # Find words (word library) that sound dark
    python3 query-clips.py --library out-clips/word-library --tags dark --sort dark

    # Filter by descriptor score range
    python3 query-clips.py --library out-clips/co-production --min-score vocalish=0.7 --min-score tonal=0.5

    # Search by filename/path substring
    python3 query-clips.py --library out-clips/word-library --search "love"

    # JSON output for programmatic use
    python3 query-clips.py --library out-clips/co-production --tags vocalish --json
"""

import argparse
import json
import sys
from pathlib import Path


DESCRIPTOR_KEYS = [
    "bright", "dark", "sub", "noise", "tonal", "transient",
    "sustain", "vocalish", "soft_entry", "soft_tail", "hard_cut",
]


def load_library(lib_path):
    """Load a clips_summary.json and return clip list with library tag."""
    lib_path = Path(lib_path).expanduser().resolve()
    summary_path = lib_path / "clips_summary.json"
    if not summary_path.exists():
        print(f"Warning: {summary_path} not found, skipping", file=sys.stderr)
        return [], None

    with summary_path.open() as f:
        data = json.load(f)

    # load similarity index if available
    sim_path = lib_path / "similarity_index.json"
    sim_index = None
    if sim_path.exists():
        with sim_path.open() as f:
            sim_index = json.load(f)

    clips = data.get("clips", [])
    lib_name = lib_path.name
    for clip in clips:
        clip["_library"] = lib_name
    return clips, sim_index


def filter_clips(clips, args):
    """Apply filters and return matching clips."""
    result = clips

    # search by path substring
    if args.search:
        q = args.search.lower()
        result = [c for c in result if q in c.get("relative_path", "").lower() or q in c.get("filename", "").lower()]

    # filter by roles (any match)
    if args.roles:
        roles_set = set(args.roles)
        result = [c for c in result if roles_set & set(c.get("suggested_roles", []))]

    # filter by tags (any match)
    if args.tags:
        tags_set = set(args.tags)
        result = [c for c in result if tags_set & set(c.get("descriptor_tags", []))]

    # filter by minimum descriptor scores
    if args.min_score:
        for spec in args.min_score:
            key, val = spec.split("=", 1)
            val = float(val)
            result = [c for c in result if c.get("descriptor_scores", {}).get(key, 0.0) >= val]

    # filter by maximum descriptor scores
    if args.max_score:
        for spec in args.max_score:
            key, val = spec.split("=", 1)
            val = float(val)
            result = [c for c in result if c.get("descriptor_scores", {}).get(key, 1.0) <= val]

    # filter by duration
    if args.min_duration is not None:
        result = [c for c in result if c.get("duration_s", 0) >= args.min_duration]
    if args.max_duration is not None:
        result = [c for c in result if c.get("duration_s", 999) <= args.max_duration]

    # filter by group
    if args.group:
        result = [c for c in result if clip_group(c) == args.group]

    return result


def clip_group(clip):
    rel = clip.get("relative_path", "")
    parts = rel.split("/")
    return parts[0] if len(parts) > 1 else "(root)"


def sort_clips(clips, sort_key):
    """Sort clips by a descriptor score key or built-in metric."""
    if sort_key in DESCRIPTOR_KEYS:
        return sorted(clips, key=lambda c: c.get("descriptor_scores", {}).get(sort_key, 0.0), reverse=True)
    elif sort_key == "duration":
        return sorted(clips, key=lambda c: c.get("duration_s", 0), reverse=True)
    elif sort_key == "brightness":
        return sorted(clips, key=lambda c: c.get("spectral_centroid_mean_hz", 0), reverse=True)
    elif sort_key == "onsets":
        return sorted(clips, key=lambda c: c.get("onsets_per_second", 0), reverse=True)
    elif sort_key == "path":
        return sorted(clips, key=lambda c: c.get("relative_path", ""))
    else:
        return clips


def format_clip_line(clip, idx):
    """Format a single clip for terminal output."""
    rel = clip.get("relative_path", "?")
    lib = clip.get("_library", "?")
    dur = clip.get("duration_s", 0)
    roles = ", ".join(clip.get("suggested_roles", []))
    tags = ", ".join(clip.get("descriptor_tags", []))
    scores = clip.get("descriptor_scores", {})

    # pick top 3 descriptor scores
    top_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
    score_str = " ".join(f"{k}={v:.2f}" for k, v in top_scores)

    return f"  {idx:3d}. [{lib}] {rel}  ({dur:.2f}s)  roles=[{roles}]  tags=[{tags}]  {score_str}"


def format_clip_json(clip):
    """Format a clip for JSON output."""
    return {
        "relative_path": clip.get("relative_path"),
        "library": clip.get("_library"),
        "duration_s": clip.get("duration_s"),
        "roles": clip.get("suggested_roles", []),
        "tags": clip.get("descriptor_tags", []),
        "descriptor_scores": clip.get("descriptor_scores", {}),
        "source_path": clip.get("path"),
    }


def handle_similarity(clips, sim_indexes, target_path, mode="similar"):
    """Find clips similar to or contrasting with a target."""
    # find target in similarity indexes
    for sim_index in sim_indexes:
        if sim_index and target_path in sim_index:
            neighbors = sim_index[target_path].get(mode, [])
            # resolve neighbor paths to clip objects
            path_to_clip = {c.get("relative_path"): c for c in clips}
            result = []
            for nb in neighbors:
                clip = path_to_clip.get(nb["relative_path"])
                if clip:
                    clip = dict(clip)
                    clip["_similarity_distance"] = nb["distance"]
                    result.append(clip)
            return result

    # fallback: not in precomputed index, compute on the fly
    print(f"Note: '{target_path}' not in precomputed similarity index, computing live", file=sys.stderr)
    import numpy as np
    target_clip = None
    for c in clips:
        if c.get("relative_path") == target_path:
            target_clip = c
            break
    if not target_clip:
        print(f"Error: clip '{target_path}' not found in any loaded library", file=sys.stderr)
        return []

    target_vec = np.array([target_clip.get("descriptor_scores", {}).get(k, 0.0) for k in DESCRIPTOR_KEYS])
    scored = []
    for c in clips:
        if c.get("relative_path") == target_path:
            continue
        vec = np.array([c.get("descriptor_scores", {}).get(k, 0.0) for k in DESCRIPTOR_KEYS])
        dist = float(np.sqrt(np.sum((vec - target_vec) ** 2)))
        entry = dict(c)
        entry["_similarity_distance"] = round(dist, 4)
        scored.append(entry)

    if mode == "similar":
        scored.sort(key=lambda c: c["_similarity_distance"])
    else:
        scored.sort(key=lambda c: c["_similarity_distance"], reverse=True)

    return scored


def main():
    ap = argparse.ArgumentParser(
        description="Query analyzed clip libraries.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--library", action="append", required=True, help="Path to analyzed library output dir (can specify multiple)")
    ap.add_argument("--search", help="Search by path/filename substring")
    ap.add_argument("--roles", nargs="+", help="Filter by roles (any match)")
    ap.add_argument("--tags", nargs="+", help="Filter by descriptor tags (any match)")
    ap.add_argument("--min-score", action="append", help="Minimum descriptor score, e.g. vocalish=0.7")
    ap.add_argument("--max-score", action="append", help="Maximum descriptor score, e.g. noise=0.3")
    ap.add_argument("--min-duration", type=float, help="Minimum duration in seconds")
    ap.add_argument("--max-duration", type=float, help="Maximum duration in seconds")
    ap.add_argument("--group", help="Filter by top-level group name")
    ap.add_argument("--sort", default="path", help=f"Sort by: {', '.join(DESCRIPTOR_KEYS + ['duration', 'brightness', 'onsets', 'path'])}")
    ap.add_argument("--similar-to", help="Find clips similar to this relative path")
    ap.add_argument("--contrast-with", help="Find clips contrasting with this relative path")
    ap.add_argument("--limit", type=int, default=20, help="Max results (default 20)")
    ap.add_argument("--json", action="store_true", help="Output as JSON")
    args = ap.parse_args()

    all_clips = []
    sim_indexes = []
    for lib_path in args.library:
        clips, sim_index = load_library(lib_path)
        all_clips.extend(clips)
        if sim_index:
            sim_indexes.append(sim_index)

    if not all_clips:
        print("No clips found in specified libraries.", file=sys.stderr)
        sys.exit(1)

    # handle similarity/contrast mode
    if args.similar_to:
        results = handle_similarity(all_clips, sim_indexes, args.similar_to, "similar")
        results = results[:args.limit]
    elif args.contrast_with:
        results = handle_similarity(all_clips, sim_indexes, args.contrast_with, "contrast")
        results = results[:args.limit]
    else:
        results = filter_clips(all_clips, args)
        results = sort_clips(results, args.sort)
        results = results[:args.limit]

    if args.json:
        out = [format_clip_json(c) for c in results]
        print(json.dumps(out, indent=2))
    else:
        print(f"Found {len(results)} clips (showing up to {args.limit}):\n")
        for i, clip in enumerate(results):
            line = format_clip_line(clip, i + 1)
            if "_similarity_distance" in clip:
                line += f"  dist={clip['_similarity_distance']:.3f}"
            print(line)


if __name__ == "__main__":
    main()
