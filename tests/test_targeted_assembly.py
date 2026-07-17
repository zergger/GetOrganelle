#!/usr/bin/env python
# coding:utf8

import os
import tempfile
import unittest
from collections import OrderedDict

from GetOrganelleLib.assembly_parser import Assembly, Vertex
from GetOrganelleLib.targeted_assembly import (
    TARGET_EXTRACTION_LABEL,
    TARGET_EXTRACTION_SEED_NEIGHBORHOOD,
    build_unambiguous_candidate_paths,
    collect_subgraph_nodes,
    extract_subgraph,
    find_branching_anchors,
    find_possible_paralogs,
    parse_blast_hits,
    parse_graph_hops,
    resolve_target_extraction_mode,
)


def make_linear_graph():
    graph = Assembly(uni_overlap=1)
    graph.vertex_info["1"] = Vertex("1", 3, 10., "AAA")
    graph.vertex_info["2"] = Vertex("2", 3, 10., "AAT")
    graph.vertex_info["3"] = Vertex("3", 3, 10., "TGG")
    graph.vertex_info["1"].connections[True] = OrderedDict([(("2", False), 1)])
    graph.vertex_info["2"].connections[False] = OrderedDict([(("1", True), 1)])
    graph.vertex_info["2"].connections[True] = OrderedDict([(("3", False), 1)])
    graph.vertex_info["3"].connections[False] = OrderedDict([(("2", True), 1)])
    graph.update_vertex_clusters()
    return graph


class TargetedAssemblyTests(unittest.TestCase):

    def test_resolve_target_extraction_mode(self):
        self.assertEqual(
            resolve_target_extraction_mode(),
            TARGET_EXTRACTION_LABEL)
        self.assertEqual(
            resolve_target_extraction_mode(TARGET_EXTRACTION_LABEL),
            TARGET_EXTRACTION_LABEL)
        self.assertEqual(
            resolve_target_extraction_mode(
                TARGET_EXTRACTION_SEED_NEIGHBORHOOD),
            TARGET_EXTRACTION_SEED_NEIGHBORHOOD)
        self.assertEqual(
            resolve_target_extraction_mode(legacy_general_target=True),
            TARGET_EXTRACTION_SEED_NEIGHBORHOOD)
        self.assertEqual(
            resolve_target_extraction_mode(
                TARGET_EXTRACTION_SEED_NEIGHBORHOOD,
                legacy_general_target=True),
            TARGET_EXTRACTION_SEED_NEIGHBORHOOD)
        with self.assertRaises(ValueError):
            resolve_target_extraction_mode(
                TARGET_EXTRACTION_LABEL,
                legacy_general_target=True)
        with self.assertRaises(ValueError):
            resolve_target_extraction_mode("unknown")

    def test_parse_graph_hops(self):
        self.assertEqual(parse_graph_hops("0"), 0)
        self.assertEqual(parse_graph_hops("10"), 10)
        self.assertEqual(parse_graph_hops("inf"), float("inf"))
        with self.assertRaises(ValueError):
            parse_graph_hops("-1")

    def test_parse_and_rank_blast_hits(self):
        descriptor, blast_path = tempfile.mkstemp(prefix="target_hits_", suffix=".tsv")
        try:
            with os.fdopen(descriptor, "w") as blast_handle:
                blast_handle.write(
                    "q1\t2\t95\t100\t200\t50\t1e-30\t200\tplus\n"
                    "q1\t3\t92\t90\t200\t45\t1e-20\t180\tminus\n"
                    "q1\t4\t60\t150\t200\t75\t1e-40\t300\tplus\n")
            selected, passing, truncated = parse_blast_hits(
                blast_path, min_identity=70., min_query_coverage=20.,
                max_evalue=1E-5, max_anchor_contigs=1)
            self.assertEqual(len(passing), 2)
            self.assertEqual([hit["sseqid"] for hit in selected], ["2"])
            self.assertTrue(truncated)
        finally:
            os.remove(blast_path)

    def test_bounded_graph_collection(self):
        graph = make_linear_graph()
        nodes_zero, hops_zero = collect_subgraph_nodes(graph, {"2"}, 0)
        nodes_one, hops_one = collect_subgraph_nodes(graph, {"2"}, 1)
        self.assertEqual(nodes_zero, {"2"})
        self.assertEqual(nodes_one, {"1", "2", "3"})
        self.assertEqual(hops_one["1"], 1)

    def test_unambiguous_path_joining(self):
        graph = make_linear_graph()
        hits = [{
            "qseqid": "q1",
            "sseqid": "2",
            "sstrand": "plus"
        }]
        candidates = build_unambiguous_candidate_paths(
            graph, hits, {"1", "2", "3"}, max_nodes=10)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(
            candidates[0]["path"],
            [("1", True), ("2", True), ("3", True)])
        self.assertEqual(candidates[0]["sequence"].seq, "AAAATGG")

    def test_extract_subgraph_removes_external_connections(self):
        graph = make_linear_graph()
        subgraph = extract_subgraph(graph, {"1", "2"})
        self.assertEqual(set(subgraph.vertex_info), {"1", "2"})
        self.assertEqual(
            list(subgraph.vertex_info["2"].connections[True]), [])
        self.assertEqual(
            list(subgraph.vertex_info["2"].connections[False]), [("1", True)])
        descriptor, gfa_path = tempfile.mkstemp(prefix="target_graph_", suffix=".gfa")
        os.close(descriptor)
        try:
            subgraph.write_to_gfa(gfa_path)
            with open(gfa_path) as gfa_handle:
                gfa_text = gfa_handle.read()
            self.assertIn("S\t1\tAAA", gfa_text)
            self.assertIn("S\t2\tAAT", gfa_text)
            self.assertNotIn("S\t3\t", gfa_text)
        finally:
            os.remove(gfa_path)

    def test_path_stops_at_branch(self):
        graph = make_linear_graph()
        graph.vertex_info["4"] = Vertex("4", 3, 10., "ATC")
        graph.vertex_info["2"].connections[True][("4", False)] = 1
        graph.vertex_info["4"].connections[False] = OrderedDict([(("2", True), 1)])
        graph.update_vertex_clusters()
        hits = [{
            "qseqid": "q1",
            "sseqid": "2",
            "sstrand": "plus"
        }]
        candidates = build_unambiguous_candidate_paths(
            graph, hits, {"1", "2", "3", "4"}, max_nodes=10)
        self.assertEqual(candidates[0]["path"], [("1", True), ("2", True)])
        self.assertEqual(candidates[0]["right_stop"], "branch")
        warnings = find_branching_anchors(
            graph, {"2"}, {"1", "2", "3", "4"})
        self.assertEqual(warnings["2"], {False: 1, True: 2})

    def test_minus_strand_anchor_orients_candidate_path(self):
        graph = make_linear_graph()
        hits = [{
            "qseqid": "q1",
            "sseqid": "2",
            "sstrand": "minus"
        }]
        candidates = build_unambiguous_candidate_paths(
            graph, hits, {"1", "2", "3"}, max_nodes=10)
        self.assertEqual(
            candidates[0]["path"],
            [("3", False), ("2", False), ("1", False)])
        self.assertEqual(candidates[0]["sequence"].seq, "CCATTTT")

    def test_possible_paralog_warning(self):
        hits = [
            {"qseqid": "q1", "sseqid": "2", "bitscore": 100., "evalue": 1E-20},
            {"qseqid": "q1", "sseqid": "3", "bitscore": 95., "evalue": 1E-18},
            {"qseqid": "q2", "sseqid": "4", "bitscore": 80., "evalue": 1E-10},
        ]
        warnings = find_possible_paralogs(hits, score_ratio=0.9)
        self.assertEqual(set(warnings), {"q1"})
        self.assertEqual(
            [hit["sseqid"] for hit in warnings["q1"]],
            ["2", "3"])


if __name__ == "__main__":
    unittest.main()
