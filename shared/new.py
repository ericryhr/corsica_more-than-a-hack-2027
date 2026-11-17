"""Create a new solution from the hidden template `solutions/.template/`.

    export CORSICA_AUTHOR=fg        # once, your initials
    python shared/new.py hmm-ball -d "HMM over the distance to the ball"
    python shared/new.py hmm-ball -d "..." --name "HMM with ball"
    python shared/new.py hmm-ball --number 7
    python shared/new.py hmm-ball --author jm

Creates `solutions/fg03-hmm-ball/`, copies the template, replaces `NN-name` with the folder name
in every text file and fills in `info.json`. Without `-d` the description stays at TODO and
`score.py` will not score anything until one is written.

The id is your initials plus your own counter, the next free one among the folders that start with
your initials. Each person works on their own branch, so a counter shared by everyone would hand
out the same number twice and the branches would collide when merged. With initials, two people
never pick the same id as long as their initials differ.
"""

import argparse
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared import info  # noqa: E402
from shared.challenge import SOLUTIONS_DIR  # noqa: E402

TEMPLATE_DIR = os.path.join(SOLUTIONS_DIR, ".template")
PLACEHOLDER_NAME = "NN-name"
SLUG = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
AUTHOR = re.compile(r"^[a-z]{2,4}$")
AUTHOR_ENV = "CORSICA_AUTHOR"
TEXT_SUFFIXES = (".py", ".md", ".json", ".txt", ".yaml", ".yml")


def folder_for(author, number, slug):
    return os.path.join(SOLUTIONS_DIR, "{}{:02d}-{}".format(author, number, slug))


def next_number(author):
    used = [int(m.group(1)) for f in os.listdir(SOLUTIONS_DIR)
            if (m := re.match(r"^{}(\d+)-".format(re.escape(author)), f))]
    return max(used, default=0) + 1


def create(author, slug):
    """Create the empty folder with the author's first free number and return its path."""
    number = next_number(author)
    while True:
        folder = folder_for(author, number, slug)
        try:
            os.mkdir(folder)
            return folder
        except FileExistsError:
            number += 1


def fill(folder):
    """Copy the template into `folder` and replace the example name."""
    name = os.path.basename(folder)
    shutil.copytree(TEMPLATE_DIR, folder, dirs_exist_ok=True)
    for base, _, files in os.walk(folder):
        for file in files:
            if not file.endswith(TEXT_SUFFIXES):
                continue
            path = os.path.join(base, file)
            with open(path) as handle:
                text = handle.read()
            if PLACEHOLDER_NAME in text:
                with open(path, "w") as handle:
                    handle.write(text.replace(PLACEHOLDER_NAME, name))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("slug", help="short name in lowercase and hyphens, without the number")
    parser.add_argument("-d", "--description", default=info.PLACEHOLDER,
                        help="what the solution does, in one or two sentences")
    parser.add_argument("--name", help="readable name for reports (default: the folder name)")
    parser.add_argument("--number", type=int, help="force this number instead of the next free one")
    parser.add_argument("--author", default=os.environ.get(AUTHOR_ENV),
                        help="your initials, 2 to 4 lowercase letters (default: ${})".format(AUTHOR_ENV))
    args = parser.parse_args()

    if not SLUG.match(args.slug):
        sys.exit("The name '{}' is not valid. Use lowercase letters, digits and hyphens: "
                 "hmm-ball".format(args.slug))
    if args.author is None:
        sys.exit("Who is creating it? Set your initials once with `export {}=fg` (in ~/.zshrc or "
                 "with `conda env config vars set {}=fg`), or pass --author fg."
                 .format(AUTHOR_ENV, AUTHOR_ENV))
    if not AUTHOR.match(args.author):
        sys.exit("The initials '{}' are not valid. Use 2 to 4 lowercase letters: fg"
                 .format(args.author))
    if not os.path.isdir(TEMPLATE_DIR):
        sys.exit("Template not found: {}".format(os.path.relpath(TEMPLATE_DIR)))

    if args.number is not None:
        folder = folder_for(args.author, args.number, args.slug)
        try:
            os.mkdir(folder)
        except FileExistsError:
            sys.exit("{} already exists".format(os.path.relpath(folder)))
    else:
        folder = create(args.author, args.slug)

    fill(folder)
    data = info.read(folder)
    data["name"] = args.name or os.path.basename(folder)
    data["description"] = args.description
    info.write(folder, data)

    rel = os.path.relpath(folder)
    print("Created {}".format(rel))
    if args.description == info.PLACEHOLDER:
        print("Write the description in {}/info.json before scoring.".format(rel))
    print("Next: fill in the stages of {}/run.py and its README.md".format(rel))


if __name__ == "__main__":
    main()
