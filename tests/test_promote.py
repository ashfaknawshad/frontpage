import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import promote  # noqa: E402


@pytest.fixture
def tree(tmp_path):
    shutil.copytree(ROOT / "deploy" / "frontpage" / "serving", tmp_path / "deploy" / "frontpage" / "serving")
    return tmp_path


def versions(root):
    s = root / "deploy" / "frontpage" / "serving"
    return (promote.read_version((s / "champion.yaml").read_text()), promote.read_version((s / "challenger.yaml").read_text()))


class FakeMLflow:
    def __init__(self):
        self.aliases = {}
        self.table = {1: {"test_pr_auc": 0.127, "test_roc_auc": 0.707, "test_brier": 0.123},
                      3: {"test_pr_auc": 0.127, "test_roc_auc": 0.707, "test_brier": 0.0417, "test_ece": 0.003}}

    def metrics(self, version):
        return self.table[version], []

    def set_alias(self, alias, version):
        self.aliases[alias] = version


def test_the_real_manifests_match_the_patterns(tree):
    assert versions(tree)[0] >= 1 and versions(tree)[1] >= 1


def test_apply_swaps_champion_and_challenger_and_touches_nothing_else(tree):
    s = tree / "deploy" / "frontpage" / "serving"
    before = {n: (s / n).read_text() for n in ("champion.yaml", "challenger.yaml", "gateway.yaml")}
    old_champion, _ = versions(tree)
    promote.apply(old_champion + 5, show_probability=True, root=tree)
    assert versions(tree) == (old_champion + 5, old_champion)
    for name, text in before.items():
        changed = [(a, b) for a, b in zip(text.splitlines(), (s / name).read_text().splitlines()) if a != b]
        assert len(changed) <= 1 or name == "gateway.yaml", (name, changed)
    assert 'SHOW_PROBABILITY, value: "true"' in (s / "gateway.yaml").read_text()


def test_apply_without_probability_leaves_the_gateway_alone(tree):
    s = tree / "deploy" / "frontpage" / "serving"
    before = (s / "gateway.yaml").read_text()
    promote.apply(versions(tree)[0] + 1, show_probability=False, root=tree)
    assert (s / "gateway.yaml").read_text() == before


def test_promoting_the_current_champion_is_refused(tree):
    with pytest.raises(SystemExit):
        promote.apply(versions(tree)[0], False, root=tree)


def test_show_probability_is_idempotent_and_reversible():
    text = '            - { name: CANARY_PERCENT, value: "0" }\n          readinessProbe:\n'
    once = promote.set_show_probability(text, True)
    assert once.count("SHOW_PROBABILITY") == 1 and once.splitlines()[1].startswith("            - { name: SHOW")
    assert promote.set_show_probability(once, True) == once
    assert 'value: "false"' in promote.set_show_probability(once, False)


def test_plan_shows_both_versions_and_marks_missing_metrics(tree):
    champion = versions(tree)[0]
    fake = FakeMLflow()
    fake.table = {champion: fake.table[1], 3: fake.table[3]}
    text = promote.plan(3, fake, root=tree)
    assert f"champion v{champion}" in text and "candidate v3" in text
    assert "| Calibration error (lower is better) | n/a | 0.0030 |" in text


def test_sync_aliases_follows_git_including_after_a_rollback(tree):
    fake = FakeMLflow()
    old = versions(tree)[0]
    promote.apply(old + 2, False, root=tree)
    assert promote.sync_aliases(fake, root=tree) == {"champion": old + 2, "challenger": old}
    promote.apply(old, False, root=tree)
    promote.sync_aliases(fake, root=tree)
    assert fake.aliases == {"champion": old, "challenger": old + 2}
