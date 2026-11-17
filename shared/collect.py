"""Read every solution's info.json and join them into one table. Reads only `solutions/*/info.json`.

    python shared/collect.py                       # table in the terminal
    python shared/collect.py --csv reports/all.csv
    python shared/collect.py --html reports/all.html
    python shared/collect.py --gt public --window all    # filter by ground truth and window

Exits with code 1 and lists the reasons if any solution has no valid info.json: the file is
missing, `name` is empty or `description` is still TODO. Folders starting with `.` or `_` (the
template) are ignored.

This module never writes into solution folders. Reports go to the path it is given.
"""

import argparse
import csv
import html
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared import info  # noqa: E402
from shared.challenge import SOLUTIONS_DIR  # noqa: E402

COLUMNS = ("solution", "submission", "gt", "window", "spell_f1_mean", "macro_f1",
           "n_pred_spells", "n_gt_spells", "date", "description")


def solution_dirs():
    return sorted(os.path.join(SOLUTIONS_DIR, f) for f in os.listdir(SOLUTIONS_DIR)
                  if not f.startswith((".", "_")) and os.path.isdir(os.path.join(SOLUTIONS_DIR, f)))


def collect(gt=None, window=None):
    """(rows, errors). One row per result, sorted by spellF1 descending."""
    rows, errors = [], []
    for folder in solution_dirs():
        name = os.path.basename(folder)
        found = info.problems(folder)
        if found:
            errors.extend("{}: {}".format(name, reason) for reason in found)
            continue
        data = info.read(folder)
        for result in data["results"]:
            if (gt and result["gt"] != gt) or (window and result["window"] != window):
                continue
            rows.append(dict({c: "" for c in COLUMNS}, solution=name,
                             description=data["description"], **{
                                 k: v for k, v in result.items() if k in COLUMNS}))
    rows.sort(key=lambda r: (r["gt"], r["window"], -float(r["spell_f1_mean"])))
    return rows, errors


def to_text(rows):
    head = "{:<26} {:<12} {:<9} {:>8} {:>8} {:>9}".format(
        "submission", "gt", "window", "spellF1", "macroF1", "possessions")
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append("{:<26} {:<12} {:<9} {:>8.4f} {:>8.4f} {:>5}/{:<3}".format(
            r["submission"], r["gt"][:12], r["window"], float(r["spell_f1_mean"]),
            float(r["macro_f1"]), r["n_pred_spells"], r["n_gt_spells"]))
    return "\n".join(lines)


def write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_html(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cells = lambda r, cols: "".join("<td>{}</td>".format(html.escape(str(r[c]))) for c in cols)
    body = "\n".join("<tr>{}</tr>".format(cells(r, COLUMNS)) for r in rows)
    head = "".join("<th>{}</th>".format(c) for c in COLUMNS)
    with open(path, "w") as handle:
        handle.write("""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Results</title>
<style>
:root {{ --bg:#fff; --fg:#1a1a1a; --line:#d8d8d8; --head:#f1f1f1; color-scheme: light dark; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#16181b; --fg:#e8e8e8; --line:#33373d; --head:#22262b; }} }}
body {{ background:var(--bg); color:var(--fg); font:14px/1.4 system-ui, sans-serif; margin:16px; }}
.wrap {{ overflow-x:auto; }}
table {{ border-collapse:collapse; }}
th, td {{ border:1px solid var(--line); padding:4px 10px; text-align:left; white-space:nowrap; }}
td:last-child {{ white-space:normal; min-width:24em; }}
th {{ background:var(--head); }}
</style></head><body>
<h1>Results by solution</h1>
<div class="wrap"><table><thead><tr>{head}</tr></thead><tbody>
{body}
</tbody></table></div></body></html>
""".format(head=head, body=body))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", help="write the table to this CSV")
    parser.add_argument("--html", help="write the table to this HTML file")
    parser.add_argument("--gt", help="only results against this ground truth")
    parser.add_argument("--window", help="only results for this window ('all' or 'start-end')")
    args = parser.parse_args()

    rows, errors = collect(args.gt, args.window)
    print(to_text(rows))
    if args.csv:
        write_csv(args.csv, rows)
        print("\n-> {}".format(args.csv))
    if args.html:
        write_html(args.html, rows)
        print("\n-> {}".format(args.html))
    if errors:
        print("\nINVALID info.json (these solutions are not listed above):")
        for error in errors:
            print("  " + error)
        sys.exit(1)


if __name__ == "__main__":
    main()
