from pathlib import Path

ingest = Path("deploy/frontpage/jobs/ingest.yaml")
t = ingest.read_text()
anchor = "                  valueFrom: { secretKeyRef: { name: pg-frontpage, key: password } }\n"
add = (anchor +
       "                - { name: GATEWAY_URL, value: http://gateway.frontpage.svc.cluster.local }\n"
       "                - name: GATEWAY_API_KEY\n"
       "                  valueFrom: { secretKeyRef: { name: gateway-api-keys, key: INGEST_KEY } }\n")
if "GATEWAY_URL" not in t:
    assert t.count(anchor) == 1, "ingest.yaml layout changed; edit by hand"
    ingest.write_text(t.replace(anchor, add))
    print("ingest.yaml: added GATEWAY_URL and GATEWAY_API_KEY")
else:
    print("ingest.yaml: already patched")

app = Path("deploy/argocd/apps/frontpage.yaml")
t = app.read_text()
if "SkipDryRunOnMissingResource" not in t:
    assert "      - CreateNamespace=true\n" in t, "frontpage.yaml layout changed; edit by hand"
    app.write_text(t.replace("      - CreateNamespace=true\n",
                             "      - CreateNamespace=true\n      - SkipDryRunOnMissingResource=true\n", 1))
    print("frontpage.yaml: added SkipDryRunOnMissingResource")
else:
    print("frontpage.yaml: already patched")
