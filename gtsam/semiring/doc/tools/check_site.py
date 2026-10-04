"""Check every built page of the book for maths errors and leftovers.

Usage, after `myst build --html` in gtsam/semiring/doc:
    python3 gtsam/semiring/doc/tools/check_site.py
"""
import glob, html, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
root = (sys.argv[1] if len(sys.argv) > 1
        else os.path.join(os.path.dirname(HERE), "_build", "html"))
bad = 0
for page in sorted(glob.glob(root + "/**/index.html", recursive=True)) + [root + "/index.html"]:
    name = os.path.relpath(page, root)
    s = open(page).read()
    body = re.sub(r"<script.*?</script>", "", s, flags=re.S)
    body = re.sub(r"<annotation.*?</annotation>", "", body, flags=re.S)
    body = re.sub(r"<(pre|code)[^>]*>.*?</\1>", "", body, flags=re.S)
    text = html.unescape(re.sub(r"<[^>]+>", " ", body))
    raw = sorted(set(re.findall(r"\\[a-zA-Z]{3,}", text)))
    dollars = re.findall(r"\$[^$ ][^$]{0,40}\$", text)
    colons = re.findall(r":::\{?\w*", text)
    errors = s.count("katex-error")
    flag = "" if not (errors or raw or dollars or colons) else "  <-- CHECK"
    bad += bool(flag)
    katex = s.count('class="katex"')
    images = len(re.findall("<img", s))
    print(f"{name:34s} katex {katex:4d} errors {errors} raw {raw[:4]} "
          f"$ {dollars[:2]} ::: {colons[:2]} "
          f"dropdowns {s.count('myst-dropdown ')} img {images}{flag}")
print("pages to check:", bad)
