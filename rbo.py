"""Rank-biased overlap, a ragged sorted list similarity measure.
See http://doi.acm.org/10.1145/1852102.1852106 for details. All functions
directly taken from the paper are named so that they can be clearly
cross-identified.
The definition of overlap has been modified to account for ties. Without this,
results for lists with tied items were being inflated. The modification itself
is not mentioned in the paper but seems to be reasonable, see function
``overlap()``. Places in the code which diverge from the spec in the paper
because of this are highlighted with comments.
"""

import math
from bisect import bisect_left


def _numtest(floatn):
    return "{:.3f}".format(floatn)


def set_at_depth(lst, depth):
    ans = set()
    for v in lst[:depth]:
        if isinstance(v, set):
            ans.update(v)
        else:
            ans.add(v)
    return ans


def raw_overlap(list1, list2, depth):
    """Overlap as defined in the article.
    """
    set1, set2 = set_at_depth(list1, depth), set_at_depth(list2, depth)
    return len(set1.intersection(set2)), len(set1), len(set2)


def overlap(list1, list2, depth):
    """Overlap which accounts for possible ties.
    This isn't mentioned in the paper but should be used in the ``rbo*()``
    functions below, otherwise overlap at a given depth might be > depth which
    inflates the result.
    There are no guidelines in the paper as to what's a good way to calculate
    this, but a good guess is agreement scaled by depth.
    """
    return agreement(list1, list2, depth) * depth


def agreement(list1, list2, depth):
    """Proportion of shared values between two sorted lists at given depth.
    >>> _numtest(agreement("abcde", "ab", 5)
    '1.000'
    >>> _numtest(agreement("abcde", "abdcf", 1))
    '1.000'
    >>> _numtest(agreement("abcde", "abdcf", 3))
    '0.667'
    >>> _numtest(agreement("abcde", "abdcf", 4))
    '1.000'
    >>> _numtest(agreement("abcde", "abdcf", 5))
    '0.800'
    >>> _numtest(agreement([{1, 2}, 3], [1, {2, 3}], 1))
    '0.667'
    >>> _numtest(agreement([{1, 2}, 3], [1, {2, 3}], 2))
    '1.000'
    """
    len_intersection, len_set1, len_set2 = raw_overlap(list1, list2, depth)
    return 2 * len_intersection / (len_set1 + len_set2)


def cumulative_agreement(list1, list2, depth):
    return (agreement(list1, list2, d) for d in range(1, depth + 1))


def average_overlap(list1, list2, depth=None):
    """Calculate average overlap between ``list1`` and ``list2``.
    >>> _numtest(average_overlap("abcdefg", "zcavwxy", 1))
    '0.000'
    >>> _numtest(average_overlap("abcdefg", "zcavwxy", 2))
    '0.000'
    >>> _numtest(average_overlap("abcdefg", "zcavwxy", 3))
    '0.222'
    >>> _numtest(average_overlap("abcdefg", "zcavwxy", 4))
    '0.292'
    >>> _numtest(average_overlap("abcdefg", "zcavwxy", 5))
    '0.313'
    >>> _numtest(average_overlap("abcdefg", "zcavwxy", 6))
    '0.317'
    >>> _numtest(average_overlap("abcdefg", "zcavwxy", 7))
    '0.312'
    """
    depth = min(len(list1), len(list2)) if depth is None else depth
    return sum(cumulative_agreement(list1, list2, depth)) / depth


def rbo_at_k(list1, list2, p, depth=None):
    # ``p**d`` here instead of ``p**(d - 1)`` because enumerate starts at
    # 0
    depth = min(len(list1), len(list2)) if depth is None else depth
    d_a = enumerate(cumulative_agreement(list1, list2, depth))
    return (1 - p) * sum(p**d * a for (d, a) in d_a)


def rbo_min(list1, list2, p, depth=None):
    """Tight lower bound on RBO.
    See equation (11) in paper.
    >>> _numtest(rbo_min("abcdefg", "abcdefg", .9))
    '0.767'
    >>> _numtest(rbo_min("abcdefgh", "abcdefg", .9))
    '0.767'
    """
    depth = min(len(list1), len(list2)) if depth is None else depth
    x_k = overlap(list1, list2, depth)
    log_term = x_k * math.log(1 - p)
    sum_term = sum(p**d / d * (overlap(list1, list2, d) - x_k)
                   for d in range(1, depth + 1))
    return (1 - p) / p * (sum_term - log_term)


def rbo_res(list1, list2, p):
    """Upper bound on residual overlap beyond evaluated depth.
    See equation (30) in paper.
    NOTE: The doctests weren't verified but seem plausible. In particular, for
    identical lists, ``rbo_min()`` and ``rbo_res()`` should add up to 1, which
    is the case.
    >>> _numtest(rbo_res("abcdefg", "abcdefg", .9))
    '0.233'
    >>> _numtest(rbo_res("abcdefg", "abcdefghijklmnopqrstuvwxyz", .9))
    '0.186'
    """
    S, L = sorted((list1, list2), key=len)
    s, l = len(S), len(L)
    x_l = overlap(list1, list2, l)
    # since overlap(...) can be fractional in the general case of ties and f
    # must be an integer → math.ceil()
    f = math.ceil(l + s - x_l)
    # upper bound of range() is non-inclusive, therefore + 1 is needed
    term1 = s * sum(p**d / d for d in range(s + 1, f + 1))
    term2 = l * sum(p**d / d for d in range(l + 1, f + 1))
    term3 = x_l * (math.log(1 / (1 - p)) - sum(p**d / d for d in range(1, f + 1)))
    return p**s + p**l - p**f - (1 - p) / p * (term1 + term2 + term3)


def rbo_ext(list1, list2, p):
    """RBO point estimate based on extrapolating observed overlap.
    See equation (32) in paper.
    NOTE: The doctests weren't verified but seem plausible.
    >>> _numtest(rbo_ext("abcdefg", "abcdefg", .9))
    '1.000'
    >>> _numtest(rbo_ext("abcdefg", "bacdefg", .9))
    '0.900'
    """
    S, L = sorted((list1, list2), key=len)
    s, l = len(S), len(L)
    x_l = overlap(list1, list2, l)
    x_s = overlap(list1, list2, s)
    # the paper says overlap(..., d) / d, but it should be replaced by
    # agreement(..., d) defined as per equation (28) so that ties are handled
    # properly (otherwise values > 1 will be returned)
    # sum1 = sum(p**d * overlap(list1, list2, d)[0] / d for d in range(1, l + 1))
    sum1 = sum(p**d * agreement(list1, list2, d) for d in range(1, l + 1))
    sum2 = sum(p**d * x_s * (d - s) / s / d for d in range(s + 1, l + 1))
    term1 = (1 - p) / p * (sum1 + sum2)
    term2 = p**l * ((x_l - x_s) / l + x_s / s)
    return term1 + term2


def rbo(list1, list2, p):
    """Complete RBO analysis (lower bound, residual, point estimate).
    ``list`` arguments should be already correctly sorted iterables and each
    item should either be an atomic value or a set of values tied for that
    rank. ``p`` is the probability of looking for overlap at rank k + 1 after
    having examined rank k.
    """
    if not 0 <= p <= 1:
        raise ValueError("The ``p`` parameter must be between 0 and 1.")
    args = (list1, list2, p)
    return dict(min=rbo_min(*args), res=rbo_res(*args), ext=rbo_ext(*args))


def sort_dict(dct):
    scores = []
    items = []
    # items should be unique, scores don't have to
    for item, score in dct.items():
        # sort in descending order, i.e. according to ``-score``
        score = -score
        i = bisect_left(scores, score)
        if i == len(scores):
            scores.append(score)
            items.append(item)
        elif scores[i] == score:
            existing_item = items[i]
            if isinstance(existing_item, set):
                existing_item.add(item)
            else:
                items[i] = {existing_item, item}
        else:
            scores.insert(i, score)
            items.insert(i, item)
    return items


def rbo_dict(dict1, dict2, p):
    """Wrapper around ``rbo()`` for dict input.
    Each dict maps items to be sorted to the score according to which they
    should be sorted.
    """
    list1, list2 = sort_dict(dict1), sort_dict(dict2)
    return rbo(list1, list2, p)

def get_rbo(l1, l2, p=0.9):
    """
        Calculates Ranked Biased Overlap (RBO) score.
        l1 -- Ranked List 1
        l2 -- Ranked List 2
    """
    if l1 == None: l1 = []
    if l2 == None: l2 = []

    sl, ll = sorted([(len(l1), l1), (len(l2), l2)])
    s, S = sl
    l, L = ll
    if s == 0: return 0

    # Calculate the overlaps at ranks 1 through l
    # (the longer of the two lists)
    ss = set([])  # contains elements from the smaller list till depth i
    ls = set([])  # contains elements from the longer list till depth i
    x_d = {0: 0}
    sum1 = 0.0
    for i in range(l):
        x = L[i]
        y = S[i] if i < s else None
        d = i + 1

        # if two elements are same then
        # we don't need to add to either of the set
        if x == y:
            x_d[d] = x_d[d - 1] + 1.0
        # else add items to respective list
        # and calculate overlap
        else:
            ls.add(x)
            if y != None: ss.add(y)
            x_d[d] = x_d[d - 1] + (1.0 if x in ss else 0.0) + (1.0 if y in ls else 0.0)
        # calculate average overlap
        sum1 += x_d[d] / d * pow(p, d)

    sum2 = 0.0
    for i in range(l - s):
        d = s + i + 1
        sum2 += x_d[d] * (d - s) / (d * s) * pow(p, d)

    sum3 = ((x_d[l] - x_d[s]) / l + x_d[s] / s) * pow(p, l)

    # Equation 32
    rbo_ext = (1 - p) / p * (sum1 + sum2) + sum3

    return rbo_ext

def get_rbd_1(l1, l2):
    return(1-get_rbo(l1,l2))

def get_rbd_2(l1, l2):
    return(1-rbo_at_k(l1,l2, 0.9))

def get_rbd_ext(l1, l2):
    return(1-rbo_ext(l1,l2, 0.9))

def get_rbd_res(l1, l2):
    return(1-rbo_res(l1,l2, 0.9))


l1 = [1,2,4,3]
l2 = [1,3,2]

rbd_1 = get_rbd_1(l1, l2)
rbd_2 = get_rbd_2(l1, l2)
rbd_ext = get_rbd_ext(l1, l2)

print(rbd_1)
print(rbd_2)
print(rbd_ext)