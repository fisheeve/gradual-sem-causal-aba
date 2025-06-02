import os
from src.abasp.utils import RelationEnum, Fact
from enum import Enum


class SemanticEnum(str, Enum):
    ST = 'ST'  # Stable semantics
    PR = 'PR'  # Preferred' semantics (maximally complete)
    CO = 'CO'  # Complete semantics


def configure_r(default_rpath='/usr/bin/Rscript'):
    """
    Configure R settings for the cdt package.
    Is necessary for SID metric evaluation.
    """
    import cdt

    rpath = os.environ.get('RPATH', default_rpath)

    cdt.SETTINGS.rpath = rpath

    # Configure R to run non-interactively and disable pagination
    os.environ['R_INTERACTIVE'] = 'FALSE'  # Force non-interactive mode
    os.environ['R_PAPERSIZE'] = 'letter'   # Avoid unnecessary prompts
    os.environ['PAGER'] = 'cat'            # Disable R’s pager (no "q" prompt)

    # Set R options to suppress interactive checks (e.g., package updates)
    os.environ['R_OPTS'] = '--no-save --no-restore --quiet'  # Silent execution


def parse_fact_line(line):
    line = line.strip()
    relation, rest = line.split('(', 1)
    rest = rest.split(')')[0]
    node1, node2, node_set = rest.split(',')
    node1 = int(node1.strip())
    node2 = int(node2.strip())
    node_set = node_set.strip()
    if node_set == 'empty':
        node_set = {}
    else:
        node_set = node_set.strip()[1:].split('y')
        node_set = {int(x) for x in node_set if x.strip()}

    return Fact(
        relation=RelationEnum[relation.strip()],
        node1=node1,
        node2=node2,
        node_set=node_set,
        score=1,
    )

def facts_from_file(filename):
    facts = []
    with open(filename, 'r') as f:
        for line in f:
            facts.append(parse_fact_line(line))
    return facts
