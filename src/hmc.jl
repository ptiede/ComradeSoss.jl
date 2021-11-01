using AdvancedHMC

export AdHMC

"""
    AdHMC
The interface between ROSESoss and the AdvancedHMC.jl package. To construct this you
should use the function.

```
AdHMC(cm::Soss.ConditionalModel)
```
"""
struct AdHMC{H,P,A,T} <: ROSESoss.AbstractMCMC
    hamiltonian::H
    proposal::P
    adaptor::A
    transform::T
end

function AdHMC(cm::Soss.ConditionalModel;
              chunksize=5,
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
    res = DiffResults.GradientResult(zeros(dim))
    function gradℓπ(x)
        ForwardDiff.gradient!(res, ℓπ, x, grado.cfg)
        return (DiffResults.value(res), DiffResults.gradient(res))
    end
    gradℓπ(zeros(dim))
    println("A derivative took ")
    @time gradℓπ(zeros(dim))
    println("Adjust your expectations accordingly")

    metric=  Metric(dim)
    hamiltonian = Hamiltonian(metric, ℓπ, gradℓπ)
    initial_ϵ = find_good_stepsize(hamiltonian, zeros(D))
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
    start = TV.inverse(smplr.t, θ0)
    samples, stats = AdvancedHMC.sample(smplr.hamiltonian, smplr.proposal, start, nsamples, smplr.adaptor, nadapt; kwargs...)
    return TupleVector(TV.transform.(Ref(smplr.transform), samples)), stats
end
