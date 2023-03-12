#region link
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
#endregion link

#region grabcode
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
#endregion grabcode

#region werewolves
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

#endregion werewolves

#region binsrch
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

#endregion binsrch
    
#region subarraysum
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
#endregion subarraysum

#region num_to_bin
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
#print(num_to_b('420'))
#endregion num_to_bin


#region test-cases
#garbo algo ezz
def cycle(f1, f2, f3):
    def helpern(n):
        def helperx(x):
            a = n
            for i in range(a // 3):
                x = f3(f2(f1(x)))
            a = a % 3
            if a == 1:
                return f1(x)
            if a == 2:
                return f2(f1(x))
            return x
        return helperx
    return helpern

def add1(x):
        return x + 1
def times2(x):
        return x * 2
def add3(x):
    return x + 3
my_cycle = cycle(add1, times2, add3)
identity = my_cycle(0)
# print(identity(5))
# # 5
# add_one_then_double = my_cycle(2)
# print(add_one_then_double(1))
# # 4
# do_all_functions = my_cycle(3)
# print(do_all_functions(2))
# #9
# do_more_than_a_cycle = my_cycle(4)
# print(do_more_than_a_cycle(2))
# #10
# do_two_cycles = my_cycle(6)
# print(do_two_cycles(1))
#19
#endregion test-cases      

#region generator
def hailstone(n):
    """Yields the elements of the hailstone sequence starting at n.
       At the end of the sequence, yield 1 infinitely.

    >>> hail_gen = hailstone(10)
    >>> [next(hail_gen) for _ in range(10)]
    [10, 5, 16, 8, 4, 2, 1, 1, 1, 1]
    >>> next(hail_gen)
    1
    """
    # yield n
    #print(n)
    if n == 1:
        yield 1
            #return
    elif n % 2 == 0:
        yield from [num for num in hailstone(n // 2)]
        #yield from hailstone(n // 2)
    else:
        yield from [num for num in hailstone(n * 3 + 1)]
        #yield from hailstone(n * 3 + 1)

hail_gen = hailstone(10)
# print(next(hail_gen))
# print(next(hail_gen))
# print(next(hail_gen))
# print(next(hail_gen))
a = [1,2,32,3,4,5]
def b(a):
    print(a, "blya")
    if a == 1:
        yield -9000
        return

    # print(a, 2)
    g = b(a - 1)
    def h(i):
        print(i)
        return i
    yield from [h(i) for i in g]
c = b(10)
#print("RETURNED", next(c))
# print(next(c))

r = [1,2,3,4,4,56]
def l(r):
    counter = 0
    while counter < 10:
        print("aaa")
        yield -1
        counter += 1
    yield from r
def p(r):
    yield from [h for h in l(r)]

gen1 = l(r)
gen2 = p(r)

def perms(seq):
    """Generates all permutations of the given sequence. Each permutation is a
    list of the elements in SEQ in a different order. The permutations may be
    yielded in any order.

    >>> p = perms([100])
    >>> type(p)
    <class 'generator'>
    >>> next(p)
    [100]
    >>> try: #this piece of code prints "No more permutations!" if calling next would cause an error
    ...     next(p)
    ... except StopIteration:
    ...     print('No more permutations!')
    No more permutations!
    >>> sorted(perms([1, 2, 3])) # Returns a sorted list containing elements of the generator
    [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
    >>> sorted(perms((10, 20, 30)))
    [[10, 20, 30], [10, 30, 20], [20, 10, 30], [20, 30, 10], [30, 10, 20], [30, 20, 10]]
    >>> sorted(perms("ab"))
    [['a', 'b'], ['b', 'a']]
    """
    if not seq:
        yield []
    else:
        for sub_perm in perms(seq[1:]):
            for i in range(len(seq)):
                #yield p[:i] + [seq[0]] + p[i:]
                print(seq, sub_perm)
                sub_perm.insert(i, seq[0])
                lol = sub_perm.copy()
                sub_perm.pop(i)
                yield lol
                # yield sub_perm[:i] + [seq[0]] + sub_perm[i:]

c = perms([100])
# print(type(c))
# print(sorted(perms([1, 2, 3]))) # Returns a sorted list containing elements of the generator
# print(sorted(perms((10, 20, 30))))
# print(sorted(perms("ab")))
#endregion generator


#region nlenpipe

n = list(range(1, 9))
pr = [1,5,8, 9,10,17,17,20]

def getprice(N, n, pr):
    f = [0] * (N + 1)
    f[0] = 0
    for i in range(1, N+1):
        cand = [j for j in n if j <= i]
        f[i] = max([f[i - cand[j]] + pr[j] for j in range(len(cand))])
    return f[N], f

#print(getprice(8, n, pr))
#endregion nlenpipe

#region striginterpretation
def substr(x, y):
    if len(y) > len(x):
        return None
    x1 = x
    if len(x) > len(y):
        x1 = x[len(x) - 1: len(x) - len(y) - 1: -1][::-1]
    for i in range(len(x1)):
        if x1[i] != y[i]:
            return None
    return x[len(x) - len(y) - 1::-1][::-1] if len(x) != len(y) else ""

s = "101101"
d = {"a": "1", "b": "01", "c": "101"}

def countInterpretation(s, d):
    memo = {}
    def f(s, d):
        if s == None:
            return 0
        if s == "":
            return 1
        if s in memo:
            return memo[s]
        count = 0
        for w in list(d.keys()):
            count += f(substr(s, d[w]), d)
        return count
    return f(s, d)

print(countInterpretation(s, d))
#endregion striginterpretation
