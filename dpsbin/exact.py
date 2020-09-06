import numpy as np

def pmf(M: int, T: int) -> np.array:
    """Computes chance of stopping after k rolls of 1dM trying to reach total T

    Args:
        M: positive integer, die roll from 1 -> M (inclusive)
        T: non-negative integer, total we want to meet or exceed

    Returns:
        An array of probabilities with shape (T+1,)
        rtn[k] == <chance of stopping after roll k>

    Notes:
        Argument validation is the responsibility of the caller.

    Design:
        Let X be a random variable with non-negative integer values, and
        let p(k) := Prob(X = k) be its probability mass function. Let
        f(X, T, k) be the chance that the following function outputs k:

            ```
            def simulate(X, T):
                t, n = 0, 0
                while t < T:
                    t, n = t+X(), n+1
                return n
            ```

        To compute f we break the single event (a simulation) into two
        pieces:
            (1) We first sample the random variable X and achieve some
                value less than T.
            (2) We then meet or exceed T only after exactly k-1 additional
                samples.

        Note that event (2) depends on the exact value achieved in (1). In
        general we find the following recursive relationship (we relied on
        k>=1 in our derivation, and that is actually essential for the
        following relationship to hold, so k==0 and k==1 need to be
        separately computed; thankfully those are easy to derive).

            ```
            f(X, T, k) = sum(p(r)*f(X, T-r, k-1) for r in range(T))
            ```

        As far as I can tell this is hard to simplify for an arbitrary pmf,
        but a useful class of random variables is those which are uniform
        from 1->M (a die roll). In such a case, p(r) is just the constant
        1/M, and instead of actually computing each of those convolutions
        we can get away with a rolling window sum.

        The solution then is essentially a dynamic programming problem from
        that recursive definition. Create a T+1 by k+1 table (note that k==T),
        fill in columns 0 and 1 with the base cases, and compute column j
        by looking at column j-1. Your solution is then the last row of the
        table.

        Here we make two additional modifications to take advantage of
        available numpy primitives and to allow for use in certain kinds of
        resource-constrained environments:
            (1) Let f(r) be some function and s(r)=sum(map(f, range(r+1))),
                then sum(map(f, range(z+1, r+1))) == f(r) - f(z). We exploit
                this fact to compute rolling window sums by first calling
                np.cumsum and then subtracting after an appropriate alignment.
            (2) Our solution to this problem has O(T^2) time complexity, but
                strictly speaking we don't ever need the entire O(T^2) table.
                Under the assumption that most callers only care about some
                small subset of all possible T values, we allocate an O(T)
                buffer and over-write it as we conceptually move from one
                column to the next in the dynamic programming problem. Since
                we then don't have the last row of the table to return, we
                also allocate an O(T) array where we store return values.
    """

    #
    # If the total is 0, we always achieve that in 0 rolls.
    #
    if not T:
        return np.array([1.])

    #
    # If the total is positive, we can't possibly get there in
    # 0 rolls, so rtn[0]==0. All other values will be over-written.
    #
    rtn = np.zeros(T+1)
    buffer = np.zeros(T+1)

    #
    # The base case k==1. buffer[i] is the chance that a single roll
    # meets or exceeds i when 0 rolls would not. Note especially that
    # this means buffer[0]==0.
    #
    # Use max(-1, M-T) to make sure we aren't
    # trying to shove something too big into our buffer.
    #
    buffer[1:M+2] = np.arange(M, max(-1, M-T), -1)/M

    #
    # We just computed the base case k==1, and we need to shove its
    # results into our return vector before we move further to the right.
    #
    rtn[1] = buffer[-1]

    #
    # Move right from column 2 to column T in our dynamic programming problem.
    #
    for k in range(2, T+1):
        #
        # Recall that we're computing values like
        #   sum(col[k]/M for k in some_range)
        # It's just as easy to pre-compute the division and give
        # numpy a fighting chance to perform that in-place.
        #
        buffer /= M

        #
        # This is the tricky bit. Assuming M>T we're simply saying the
        # i-th row is the cumulative sum up to row i-1. Afterward though,
        # for every row where we have to start rolling the total off we
        # subtract the same cumulative sum at an appropriate offset.
        #
        # To get an intuition as to the correctness, notice that after the
        # cumsum operation, "new buffer[-1]" will contain
        # "old sum(buffer[:-1])". At that location we then subtract
        # "old sum(buffer[:-M-1])". I.e., "new buffer[-1]" now contains
        # "old sum(buffer[-M-1:-1])" -- the M rows preceding the last row.
        #
        # It's probably also good to work those indices out for yourself
        # and to examine our test cases if anything looks fishy.
        np.cumsum(buffer[:-1], out=buffer[1:])
        buffer[M:] -= buffer[:-M]

        #
        # Buffer is now exactly column k from our dynamic programming problem.
        # Store its last element in our return vector before we over-write it.
        #
        rtn[k] = buffer[-1]

    return rtn
