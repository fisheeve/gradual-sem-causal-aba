# Assumptions and their contraries

def contrary(assumption):
    return f"-{assumption}"


def arr(X, Y):
    return f"arr_{X}_{Y}"


def noe(X, Y):
    # no-edge is symmetric
    if X > Y:
        X, Y = Y, X
    return f"noe_{X}_{Y}"


def edge(X, Y):
    return contrary(noe(X, Y))


def indep(X, Y, S):
    # is symmetric with respect to X and Y
    if X > Y:
        X, Y = Y, X
    S = sorted(list(S))
    return f"indep_{X}_{Y}__" + '_'.join([str(i) for i in S])


def dep(X, Y, S):
    return contrary(indep(X, Y, S))


def blocked_path(source, target, path_id: int, S: set):
    S = sorted(list(S))
    # is symmetric with respect to source and target
    if source > target:
        source, target = target, source
    return f"blocked_path_{source}_{target}__{path_id}__" + '_'.join([str(i) for i in S])


def active_path(source, target, path_id: int, S: set):
    return contrary(blocked_path(source, target, path_id, S))
