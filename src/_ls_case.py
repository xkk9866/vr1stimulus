import json
import urllib.request

req = urllib.request.Request(
    "https://api.figshare.com/v2/articles/search",
    data=json.dumps({"search_for": "CASE dataset continuously annotated signals emotion", "page_size": 40}).encode(),
    headers={"Content-Type": "application/json"},
)
res = json.load(urllib.request.urlopen(req, timeout=90))
for a in res:
    print("%-9s %s" % (a["id"], a["title"][:90]))
