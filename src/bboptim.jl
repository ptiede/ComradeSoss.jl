using .BlackBoxOptim: bboptimize, best_candidate, best_fitness

"""
    threaded_optimize
Runs a optimizer `nopt` times that are split across how many threads are currently running.
Note this optimizes the unbounded version of the lj

Returns the best parameters and divergences sorted from best to worst fit.
"""
function threaded_bboptimize(nopt, lj, maxevals)
    results = [zeros(lj.transform.dimension) for _ in 1:nopt]
    divs = zeros(nopt)
    srange = [(-5.0,5.0) for i in 1:lj.transform.dimension]
    for i in 1:nopt
        res = bboptimize(x->-lj(x), SearchRange=srange, MaxFuncEvals=maxevals, TraceMode=:compact)
        results[i] = best_candidate(res)
        divs[i] = -best_fitness(res)
    end
    I = sortperm(divs, rev=true)
    return logj.transform.(results[I]), divs[I]
end
