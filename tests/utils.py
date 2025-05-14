from src.abasp.utils import RelationEnum, Fact


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

    if node1 > node2:
        node1, node2 = node2, node1

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
