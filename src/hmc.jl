using AdvancedHMC

export AdHMC

"""
    AdHMC
The interface between ComradeSoss and the AdvancedHMC.jl package. To construct this you
should use the function.

```
AdHMC(cm::Soss.ConditionalModel)
```
"""
struct AdHMC{H,P,A,T} <: ComradeSoss.AbstractMCMC
    hamiltonian::H
    proposal::P
    adaptor::A
    transform::T
end

function AdHMC(cm::Soss.ConditionalModel;
              chunksize=5,
              start=nothing,
              minv = nothing,
              Metric=DiagEuclideanMetric,
              Integrator=Leapfrog,
              Treesample=MultinomialTS,
              Nuts=GeneralisedNoUTurn)

    t = xform(cm)
    dim = t.dimension
    function ℓπ(x)
        p, logjac = TV.transform_and_logjac(t, x)
        return logdensity(cm, p) + logjac
    end

    grado = GradObj(ℓπ, rand(dim); chunksize=chunksize)
    function gradℓπ(x)
        res = DiffResults.GradientResult(zeros(dim))
        ForwardDiff.gradient!(res, ℓπ, x, grado.cfg)
        return (DiffResults.value(res), DiffResults.gradient(res))
    end
    gradℓπ(zeros(dim))
    println("A derivative took ")
    @time gradℓπ(zeros(dim))
    println("Adjust your expectations accordingly")

    if isnothing(minv)
        metric = Metric(dim)
    else
        metric = Metric(minv)
    end
    hamiltonian = Hamiltonian(metric, ℓπ, ForwardDiff)
    if isnothing(start)
        x0 = zeros(dim)
    else
        x0 = TV.inverse(t, start)
    end
    initial_ϵ = find_good_stepsize(hamiltonian, x0)
    integrator = Integrator(initial_ϵ)
    proposal = NUTS{Treesample, Nuts}(integrator)
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))
    return AdHMC(hamiltonian, proposal, adaptor, t)
end

function AdHMCRev(cm::Soss.ConditionalModel;
    start=nothing,
    minv = nothing,
    Metric=DiagEuclideanMetric,
    Integrator=Leapfrog,
    Treesample=MultinomialTS,
    Nuts=GeneralisedNoUTurn)

    t = xform(cm)
    dim = t.dimension
    function ℓπ(x)
        p, logjac = TV.transform_and_logjac(t, x)
        return logdensity(cm, p) + logjac
    end

    grado = ReverseDiff.GradientTape(ℓπ, rand(dim))
    ctape = ReverseDiff.compile(grado)
    function gradℓπ(x)
        res = DiffResults.GradientResult(zeros(dim))
        ReverseDiff.gradient!(res, ctape, x)
        return DiffResults.value(res), DiffResults.gradient(res)
    end
    gradℓπ(zeros(dim))
    println("A derivative took ")
    @time gradℓπ(zeros(dim))
    println("Adjust your expectations accordingly")

    if isnothing(minv)
        metric = Metric(dim)
    else
        metric = Metric(minv)
    end
    hamiltonian = Hamiltonian(metric, ℓπ, ForwardDiff)
    if isnothing(start)
        x0 = zeros(dim)
    else
        x0 = TV.inverse(t, start)
    end
    initial_ϵ = find_good_stepsize(hamiltonian, x0)
    integrator = Integrator(initial_ϵ)
    proposal = NUTS{Treesample, Nuts}(integrator)
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))
    return AdHMC(hamiltonian, proposal, adaptor, t)
end



"""
    sample(s::AdHMC, nsamples, nadapt, start, kwargs...)
Sample the posterior using AdvancedHMC with a total of `nsamples` where the first `nadapt`
samples are adaptation. The starting location is given by `start` and this is assumed to
be in model parameter space not the continous one.
"""
function StatsBase.sample(smplr::AdHMC, nsamples, nadapt, θ0::NamedTuple; kwargs...)
    start = TV.inverse(smplr.transform, θ0)
    samples, stats = AdvancedHMC.sample(smplr.hamiltonian, smplr.proposal, start, nsamples, smplr.adaptor, nadapt; kwargs...)
    return TupleVector(TV.transform.(Ref(smplr.transform), samples)), stats
end
