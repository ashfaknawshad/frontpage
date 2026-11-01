#!/usr/bin/env python3
"""Adds the read-only Postgres role Grafana uses to deploy/platform/data/postgres.yaml. Safe to run twice."""
import sys
from pathlib import Path

path = Path("deploy/platform/data/postgres.yaml")
text = path.read_text()
if "grafana_ro" in text:
    print("postgres.yaml already has grafana_ro; nothing to do")
    sys.exit(0)

anchor = "        passwordSecret:\n          name: pg-mlflow\n"
role = (anchor +
        "      - name: grafana_ro\n"
        "        ensure: present\n"
        "        login: true\n"
        "        inRoles:\n"
        "          - pg_read_all_data\n"
        "        passwordSecret:\n"
        "          name: pg-grafana\n")
if text.count(anchor) != 1:
    sys.exit("Could not find the mlflow role block in postgres.yaml; send me the file and I will adjust the patch.")
path.write_text(text.replace(anchor, role))
print("added grafana_ro role to postgres.yaml")
