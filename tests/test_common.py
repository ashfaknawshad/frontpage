from jobs.common import domain_of, is_live_story, kind_of


def test_kind_of():
    assert kind_of("Show HN: My thing") == "show"
    assert kind_of("Ask HN: Why?") == "ask"
    assert kind_of("Launch HN: Acme (YC W26)") == "launch"
    assert kind_of("A normal title") == "story"


def test_domain_of():
    assert domain_of("https://www.Example.com/a?b=c") == "example.com"
    assert domain_of("http://user@blog.example.org:8080/x") == "blog.example.org"
    assert domain_of(None) is None


def test_is_live_story():
    assert is_live_story({"type": "story", "title": "x", "time": 1})
    assert not is_live_story({"type": "job", "title": "x", "time": 1})
    assert not is_live_story({"type": "story", "title": "x", "time": 1, "dead": True})
    assert not is_live_story({"type": "story", "title": "", "time": 1})
