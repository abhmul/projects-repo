import math
import itertools

flatten_iter = itertools.chain.from_iterable


# https://stackoverflow.com/a/6909532/5538273
def factors(n):
    return set(flatten_iter((i, n//i) for i in range(1, int(math.sqrt(n)+1)) if n % i == 0))



def prime_factors(n):
    dividend = n
    prime_nums = primes(n)
    prime_factors = []
    while dividend not in prime_nums:
        for p in prime_nums:
            if dividend % p == 0:
                dividend = dividend // p
                prime_factors.append(p)
                break
    prime_factors.append(dividend)
    return sorted(prime_factors)


# https://stackoverflow.com/a/19498432/5538273
def primes(n):
    odds = range(3, n+1, 2)
    sieve = set(flatten_iter([range(q*q, n+1, q+q) for q in odds]))
    return set([2] + [p for p in odds if p not in sieve])


# Sieve of Eratosthenes
# Code by David Eppstein, UC Irvine, 28 Feb 2002
# http://code.activestate.com/recipes/117119/
def gen_primes():
    """ Generate an infinite sequence of prime numbers.
    """
    # Maps composites to primes witnessing their compositeness.
    # This is memory efficient, as the sieve is not "run forward"
    # indefinitely, but only as long as required by the current
    # number being tested.
    #
    D = {}
    
    # The running integer that's checked for primeness
    q = 2
    
    while True:
        if q not in D:
            # q is a new prime.
            # Yield it and mark its first multiple that isn't
            # already marked in previous iterations
            # 
            yield q
            D[q * q] = [q]
        else:
            # q is composite. D[q] is the list of primes that
            # divide it. Since we've reached q, we no longer
            # need it in the map, but we'll mark the next 
            # multiples of its witnesses to prepare for larger
            # numbers
            # 
            for p in D[q]:
                D.setdefault(p + q, []).append(p)
            del D[q]
        
        q += 1
