import numpy as np
import pandas as pd
import pytest

from src.data.protected import intersectional_groups, recorded_attribute
from src.data.splits import stratified_split
from src.data.subgraph import bfs_sample, labelled_lcc
from src.data.target import target_crosswalk, transform_target
from src.utils import ROOT


def test_target_transformation_matches_fairgnn():
    raw = np.array([-1, 0, 1, 2, 3, 4])
    assert transform_target(raw).tolist() == [-1, 0, 1, 1, 1, 1]
    with pytest.raises(ValueError):
        transform_target(np.array([-2]))
    cw = target_crosswalk(raw)
    assert cw.set_index("raw_code").loc[0, "Y"] == 0


def test_protected_attribute_not_imputed():
    with pytest.raises(ValueError):
        recorded_attribute(pd.DataFrame({"gender": [0, np.nan]}), "gender")
    with pytest.raises(ValueError):
        recorded_attribute(pd.DataFrame({"gender": [0, 2]}), "gender")
    assert intersectional_groups(np.array([0, 1]), np.array([1, 0])).tolist() == ["g0_r1", "g1_r0"]


def test_split_disjoint_stratified_reproducible():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 1000); a = rng.integers(0, 2, 1000)
    s1 = stratified_split(y, a, 7); s2 = stratified_split(y, a, 7)
    assert np.array_equal(s1.train, s2.train)
    assert len(set(s1.train) | set(s1.val) | set(s1.test)) == 1000
    for cell in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        m = (y == cell[0]) & (a == cell[1])
        assert abs(np.isin(np.flatnonzero(m), s1.train).mean() - 0.5) < 0.02


def test_bfs_sample_connected_and_sized():
    import networkx as nx
    g = nx.connected_watts_strogatz_graph(300, 4, 0.2, seed=1)
    uid = np.arange(300) * 10
    e = np.array([(uid[a], uid[b]) for a, b in g.edges()])
    lcc = labelled_lcc(uid, np.ones(300, bool), e)
    s = bfs_sample(lcc, 120, seed=3)
    h = nx.Graph(); h.add_nodes_from(range(120)); h.add_edges_from(map(tuple, s.edges))
    assert len(s.user_ids) == 120 and nx.is_connected(h)
    assert np.array_equal(bfs_sample(lcc, 120, seed=3).user_ids, s.user_ids)


@pytest.mark.skipif(not (ROOT / "data/processed/pokec_z_core.npz").exists(), reason="prepared data not present")
def test_prepared_core_integrity():
    from src.data.integrity import check_prepared
    from src.data.prepared import Prepared
    p = Prepared.load(ROOT / "data/processed/pokec_z_core.npz")
    assert 1000 <= p.n <= 3000
    assert all(check_prepared(p.n, p.edges, p.y, p.gender, p.features).values())
    assert "gender" not in p.feature_names and "region" not in p.feature_names
    assert "I_am_working_in_field" not in p.feature_names and "user_id" not in p.feature_names


def test_embeddings_reproducible_and_shaped():
    from src.embeddings.random_walks import WalkConfig, generate_walks, train_embeddings
    import networkx as nx
    g = nx.karate_club_graph()
    e = np.array(g.edges())
    cfg = WalkConfig(dimensions=8, walk_length=10, walks_per_node=3)
    a = train_embeddings(34, e, cfg, 1)["embedding"]
    b = train_embeddings(34, e, cfg, 1)["embedding"]
    assert a.shape == (34, 8) and np.allclose(a, b) and np.all(np.isfinite(a))
    # FairWalk: with groups, every step chooses among groups present -> differs from Node2Vec walks
    grp = np.array([i % 2 for i in range(34)])
    assert generate_walks(34, e, cfg, 1, grp) != generate_walks(34, e, cfg, 1, None)


def test_fairwalk_group_choice_is_uniform_over_present_groups():
    from src.embeddings.random_walks import WalkConfig, generate_walks
    # star: centre 0 has 1 neighbour in group 1 and 9 in group 0. FairWalk picks group 1 ~50% of the time
    e = np.array([(0, i) for i in range(1, 11)])
    grp = np.array([0, 1] + [0] * 9)
    walks = generate_walks(11, e, WalkConfig(walk_length=2, walks_per_node=400), 0, grp)
    second = [w[1] for w in walks if w[0] == 0]
    assert abs(np.mean(np.array(second) == 1) - 0.5) < 0.1
    n2v = generate_walks(11, e, WalkConfig(walk_length=2, walks_per_node=400), 0, None)
    assert np.mean(np.array([w[1] for w in n2v if w[0] == 0]) == 1) < 0.2
