class Link:
    def __init__(self, value, rest=None) -> None:
        self.value = value
        self.rest = rest
    empty = ()

def combineLink(link, fn):
    if not link:
        return Link.empty
    restcomb = combineLink(link.rest, fn)
    return fn(link.value, restcomb)
# l1 = Link(1, Link(2, Link(3)))

# print(combineLink(l1, lambda x,y: x + y))
# print(0b111 >> 1)

import itertools
import numpy as np
def nik():
    SET = np.arange(500)
    N = 5
    M = 5
    successes = 0
    for _ in range(M):
        r1, r2 = np.random.choice(SET,size=N,replace=False),np.random.choice(SET,size=N,replace=False)
        print(r1, r2)
        prods = np.arange(N)+1
        maxsofar = -1

        for a,b in itertools.product(itertools.permutations(r1),itertools.permutations(r2)):
            a,b = np.array(a),np.array(b)
            if (temp := np.sum((b-a)*prods)) > maxsofar:
                maxsofar = temp


        soln = np.sum((np.sort(r2)-np.sort(r1)[::-1])*prods)

        print("maxsofar",maxsofar,"soln",soln)

        if maxsofar == soln:
            successes += 1
    print(successes/M)
#nik()

def mything():
    def perms(li):
        if len(li) == 1:
            return [li.copy()]
        results = []
        for _ in li:
            temp = li.pop(0)
            res = perms(li)
            for i in res:
                i.append(temp)
            results.extend(res)
            li.append(temp)
        return results
    # print(perms([i for i in range(3)]))
    set1 = np.arange(500)
    N = 10
    arr2, arr1 = list(np.random.choice(set1, size=N, replace=False)), list(np.random.choice(set1, size=N, replace=False))
    perm2, perm1 = perms(arr2), perms(arr1)
    curmax = -500
    for i2 in perm2:
        for i1 in perm1:
            curmax = max(curmax, sum([(i * (i2[i] - i1[i])) for i in range(len(i2))]))
    sol2 = sum(i*(a-b) for i,(a,b) in enumerate(zip(sorted(arr2),sorted(arr1,reverse=True))))
    print(sol2, curmax)
    return sol2 == curmax

import random
def werewolves():
    lie = (lambda : random.randint(0,1))
    # 0 - w, 1 - v
    players = [1,1,0,1,1,0,1,0,0,1,0,0,1,1,0,1]


    def find(li):
        if len(li) == 1:
            return (li[0], li[0])
        a = li[:len(li) // 2]
        b = li[len(li) // 2:]
        x = find(a)
        y = find(b)
        resx = ask(li, x)
        resy = ask(li, y)
        winner = (0,0)
        if resx[1]:
            winner = resx
        elif resy[1]:
            winner = resy
        return winner


    def ask(li, cand):
        res = [0,0]
        for i in li:
            if i == 0:
                res[lie()] += 1
            else:
                res[cand[0]] += 1
        return (cand[0], int(res[1] > res[0]))

    res = []
    for _ in range(1000):
        np.random.shuffle(players)
        smpl = find(players)
        res.append(smpl[0] == smpl[1])
    return res.count(False)

# print(werewolves())

def binary_search_recursive(lst, of):
    #print(of)
    if len(lst) == 0:
        return -1
    if len(lst) == 1:
        return 0 if of == lst[0] else -1
    left = 0
    right = len(lst) - 1
    mid = (left + right) // 2
    if lst[mid] > of:
        right = mid - 1
        return binary_search_recursive(lst[:right + 1], of)
    elif lst[mid] < of:
        left = mid + 1
        res = binary_search_recursive(lst[left:], of)
        return -1 if res == -1 else left + res
    else:
        return mid

# a = [1,2,3,4,5,6,7,8,9]
# print(binary_search_recursive(a, 9))

def FFT(poly):
    odd = poly[1::2]
    even = poly[0::2]
    res = []
    
import math
def subarray_sum(li):

    def find_sum(left, right, midm = False):
        if left == right:
            return (left, right)
        mid = (left + right) // 2
        lmax = find_sum(left, mid)
        rmax = find_sum(mid + 1, right)
        midmax = findlarger(left, mid, right)
        s1 = sum(li[lmax[0]: lmax[1]])
        s2 = sum(li[rmax[0]: rmax[1]])
        s3 = sum(li[midmax[0]: midmax[1]])
        d = {s1: lmax, s2: rmax, s3: midmax}
        return d[max(s1,s2,s3)]
    
    def findlarger(left, mid, right):
        lmax = (0, mid)
        cur = 0
        for i in range(mid, left - 1, -1):
            cur += li[i]
            if lmax[0] < cur:
                lmax = (cur, i)
        rmax = (0, mid)
        cur = 0
        for i in range(mid + 1, right + 1, 1):
            cur += li[i]
            if rmax[0] < cur:
                rmax = (cur, i)
        return (lmax[1], rmax[1])
    res = find_sum(0, len(li) - 1)
    print(li[res[0]:res[1] + 1])
    return sum(li[res[0]:res[1] + 1])

#print(subarray_sum([-2,1,-3,4,-1,2,1,-5,4]))


def num_to_b(x):
    if int(x) <= 10:
        return bin(int(x))
    x = x[::-1]
    nr, nl = x[0:len(x) // 2], x[len(x) // 2:len(x)]
    x = x[::-1]
    nr = nr[::-1]
    nl = nl[::-1]
    resl = num_to_b(nl)
    resr = num_to_b(nr)
    res_10p = num_to_b(str(10 ** (len(x) // 2))) #left shift O(n)
    resl = bin(int(resl, 2) * int(res_10p, 2)) #karatsuba O(n^1.6)
    return bin(int(resl, 2) + int(resr, 2))

print(num_to_b('420'))
