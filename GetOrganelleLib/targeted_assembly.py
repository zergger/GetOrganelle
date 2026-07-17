#!/usr/bin/env python
# coding:utf8

from collections import defaultdict, deque
from copy import deepcopy
try:
    from math import inf
except ImportError:
    inf = float("inf")

from GetOrganelleLib.assembly_parser import Assembly


BLAST_OUTFMT_FIELDS = (
    "qseqid", "sseqid", "pident", "length", "qlen",
    "qcovs", "evalue", "bitscore", "sstrand"
)

TARGET_EXTRACTION_LABEL = "label"
TARGET_EXTRACTION_SEED_NEIGHBORHOOD = "seed-neighborhood"
TARGET_EXTRACTION_MODES = (
    TARGET_EXTRACTION_LABEL,
    TARGET_EXTRACTION_SEED_NEIGHBORHOOD,
)


def resolve_target_extraction_mode(requested_mode=None, legacy_general_target=False):
    if requested_mode is not None and requested_mode not in TARGET_EXTRACTION_MODES:
        raise ValueError(
            "--target-extraction must be one of: " +
            ", ".join(TARGET_EXTRACTION_MODES))
    if legacy_general_target:
        if requested_mode not in (None, TARGET_EXTRACTION_SEED_NEIGHBORHOOD):
            raise ValueError(
                "--general-target is an alias for "
                "--target-extraction seed-neighborhood and conflicts with "
                "--target-extraction " + str(requested_mode))
        return TARGET_EXTRACTION_SEED_NEIGHBORHOOD
    if requested_mode is None:
        return TARGET_EXTRACTION_LABEL
    return requested_mode


def parse_graph_hops(raw_value):
    value = str(raw_value).strip().lower()
    if value == "inf":
        return inf
    try:
        parsed = int(value)
    except ValueError:
        raise ValueError("--target-max-graph-hops must be a non-negative integer or inf")
    if parsed < 0:
        raise ValueError("--target-max-graph-hops must be a non-negative integer or inf")
    return parsed


def parse_blast_hits(blast_out_file, min_identity=70., min_query_coverage=20.,
                     max_evalue=1E-5, max_anchor_contigs=50):
    passing_hits = []
    with open(blast_out_file) as blast_handle:
        for line_no, raw_line in enumerate(blast_handle, 1):
            if not raw_line.strip():
                continue
            fields = raw_line.rstrip("\n").split("\t")
            if len(fields) != len(BLAST_OUTFMT_FIELDS):
                raise ValueError(
                    "Unexpected BLAST output at line %i: expected %i fields, observed %i" %
                    (line_no, len(BLAST_OUTFMT_FIELDS), len(fields)))
            hit = dict(zip(BLAST_OUTFMT_FIELDS, fields))
            hit["pident"] = float(hit["pident"])
            hit["length"] = int(hit["length"])
            hit["qlen"] = int(hit["qlen"])
            hit["qcovs"] = float(hit["qcovs"])
            hit["evalue"] = float(hit["evalue"])
            hit["bitscore"] = float(hit["bitscore"])
            if hit["pident"] < min_identity:
                continue
            if hit["qcovs"] < min_query_coverage:
                continue
            if hit["evalue"] > max_evalue:
                continue
            passing_hits.append(hit)

    passing_hits.sort(
        key=lambda hit: (-hit["bitscore"], hit["evalue"], -hit["qcovs"],
                         -hit["pident"], hit["sseqid"], hit["qseqid"]))

    best_by_subject = {}
    for hit in passing_hits:
        if hit["sseqid"] not in best_by_subject:
            best_by_subject[hit["sseqid"]] = hit
    selected_hits = list(best_by_subject.values())
    selected_hits.sort(
        key=lambda hit: (-hit["bitscore"], hit["evalue"], hit["sseqid"]))

    truncated = False
    if max_anchor_contigs and len(selected_hits) > max_anchor_contigs:
        selected_hits = selected_hits[:max_anchor_contigs]
        truncated = True
    return selected_hits, passing_hits, truncated


def find_possible_paralogs(passing_hits, score_ratio=0.9):
    hits_by_query = defaultdict(dict)
    for hit in passing_hits:
        previous = hits_by_query[hit["qseqid"]].get(hit["sseqid"])
        if previous is None or hit["bitscore"] > previous["bitscore"]:
            hits_by_query[hit["qseqid"]][hit["sseqid"]] = hit

    possible_paralogs = {}
    for query_name, subject_hits in hits_by_query.items():
        if len(subject_hits) < 2:
            continue
        ranked_hits = sorted(
            subject_hits.values(),
            key=lambda hit: (-hit["bitscore"], hit["evalue"], hit["sseqid"]))
        best_score = ranked_hits[0]["bitscore"]
        near_best = [
            hit for hit in ranked_hits
            if hit["bitscore"] >= best_score * score_ratio
        ]
        if len(near_best) > 1:
            possible_paralogs[query_name] = near_best
    return possible_paralogs


def collect_subgraph_nodes(graph, anchor_nodes, max_hops=10):
    selected_nodes = set()
    node_hops = {}
    nodes_to_visit = deque()
    for anchor_node in anchor_nodes:
        if anchor_node in graph.vertex_info:
            nodes_to_visit.append((anchor_node, 0))

    while nodes_to_visit:
        current_name, current_hops = nodes_to_visit.popleft()
        if current_name in node_hops and node_hops[current_name] <= current_hops:
            continue
        node_hops[current_name] = current_hops
        selected_nodes.add(current_name)
        if current_hops >= max_hops:
            continue
        current_vertex = graph.vertex_info[current_name]
        for end in (True, False):
            for neighbor_name, neighbor_end in current_vertex.connections[end]:
                if neighbor_name in graph.vertex_info:
                    nodes_to_visit.append((neighbor_name, current_hops + 1))
    return selected_nodes, node_hops


def extract_subgraph(graph, selected_nodes):
    subgraph = Assembly(uni_overlap=graph.uni_overlap())
    for vertex_name in selected_nodes:
        subgraph.vertex_info[vertex_name] = deepcopy(graph.vertex_info[vertex_name])
    for vertex_name in selected_nodes:
        for end in (True, False):
            subgraph.vertex_info[vertex_name].connections[end] = deepcopy(
                type(graph.vertex_info[vertex_name].connections[end])(
                    (connection, overlap)
                    for connection, overlap
                    in graph.vertex_info[vertex_name].connections[end].items()
                    if connection[0] in selected_nodes))
    subgraph.update_vertex_clusters()
    return subgraph


def _connections_within(graph, vertex_name, end, selected_nodes):
    return [
        (connection, overlap)
        for connection, overlap
        in graph.vertex_info[vertex_name].connections[end].items()
        if connection[0] in selected_nodes
    ]


def _extend_path_forward(graph, path, selected_nodes, max_nodes):
    stop_reason = "terminal"
    while len(path) < max_nodes:
        current_name, current_direction = path[-1]
        connections = _connections_within(
            graph, current_name, current_direction, selected_nodes)
        if not connections:
            stop_reason = "terminal"
            break
        if len(connections) != 1:
            stop_reason = "branch"
            break
        (next_name, next_connection_end), overlap = connections[0]
        next_direction = not next_connection_end
        if next_name in {vertex_name for vertex_name, direction in path}:
            stop_reason = "cycle"
            break
        reciprocal = _connections_within(
            graph, next_name, next_connection_end, selected_nodes)
        if len(reciprocal) != 1:
            stop_reason = "merge"
            break
        path.append((next_name, next_direction))
    else:
        stop_reason = "node_limit"
    return stop_reason


def _extend_path_backward(graph, path, selected_nodes, max_nodes):
    stop_reason = "terminal"
    while len(path) < max_nodes:
        current_name, current_direction = path[0]
        connections = _connections_within(
            graph, current_name, not current_direction, selected_nodes)
        if not connections:
            stop_reason = "terminal"
            break
        if len(connections) != 1:
            stop_reason = "branch"
            break
        (previous_name, previous_connection_end), overlap = connections[0]
        previous_direction = previous_connection_end
        if previous_name in {vertex_name for vertex_name, direction in path}:
            stop_reason = "cycle"
            break
        reciprocal = _connections_within(
            graph, previous_name, previous_direction, selected_nodes)
        if len(reciprocal) != 1:
            stop_reason = "merge"
            break
        path.insert(0, (previous_name, previous_direction))
    else:
        stop_reason = "node_limit"
    return stop_reason


def build_unambiguous_candidate_paths(graph, anchor_hits, selected_nodes, max_nodes=100):
    candidate_paths = []
    observed_paths = set()
    for hit in anchor_hits:
        anchor_name = hit["sseqid"]
        if anchor_name not in selected_nodes:
            continue
        anchor_direction = hit.get("sstrand", "plus").lower() != "minus"
        path = [(anchor_name, anchor_direction)]
        left_stop = _extend_path_backward(
            graph, path, selected_nodes, max_nodes)
        right_stop = _extend_path_forward(
            graph, path, selected_nodes, max_nodes)
        path_key = tuple(path)
        reverse_key = tuple(
            (vertex_name, not direction)
            for vertex_name, direction in reversed(path))
        canonical_key = min(path_key, reverse_key)
        if canonical_key in observed_paths:
            continue
        observed_paths.add(canonical_key)
        candidate_paths.append({
            "anchor": anchor_name,
            "query": hit["qseqid"],
            "path": path,
            "left_stop": left_stop,
            "right_stop": right_stop,
            "sequence": graph.export_path(path)
        })
    return candidate_paths


def find_branching_anchors(graph, anchor_nodes, selected_nodes):
    branching = {}
    for anchor_name in anchor_nodes:
        if anchor_name not in graph.vertex_info:
            continue
        end_degrees = {}
        for end in (False, True):
            end_degrees[end] = len(_connections_within(
                graph, anchor_name, end, selected_nodes))
        if end_degrees[False] > 1 or end_degrees[True] > 1:
            branching[anchor_name] = end_degrees
    return branching
