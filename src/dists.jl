import Distributions
using LoopVectorization
using Random

export MvNormalFast
"""
    MvNormalFast
This is a specialized MvNormal that assumes a **diagonal** covariance. This is substantially
faster than the usual Distributions.jl version that for some reason has large overhead.

# Notes
Only logpdf is defined so no calling rand!
"""
struct MvNormalFast{T<:AbstractVector, S<:AbstractVector, N} <: Distributions.AbstractMvNormal
    μ::T
    Σ::S
    lnorm::N
    function MvNormalFast(μ::T, Σ::S) where {T,S}
        lnorm = -0.5*sum(log.(Σ)) - length(μ)/2*log(2π)
        return new{T,S,typeof(lnorm)}(μ, Σ, lnorm)
    end
end

Base.length(d::MvNormalFast) = length(d.μ)
function Distributions.sampler(d::MvNormalFast)
    return sampler(Distributions.MvNormal(d.μ, sqrt.(d.Σ)))
end

Base.eltype(::MvNormalFast{T,N}) where {T,N} = eltype(T)


@inline function Distributions._logpdf(d::MvNormalFast, x::AbstractVector{T}) where {T}
    μ,Σ = d.μ, d.Σ
    acc = zero(eltype(x))
    @inbounds @fastmath for i in eachindex(x)
        acc += -(x[i]-μ[i])^2/Σ[i]/2
    end
    return acc + d.lnorm
end

function Distributions._rand!(rng::Random.AbstractRNG, d::MvNormalFast, x::AbstractArray)
    return Distributions._rand!(rng, Distributions.MvNormal(d.μ, sqrt.(d.Σ)), x)
end

# These are some soss stuff to make Dirichlet and other stuff work.
#Soss.xform(d::Distributions.Dirichlet, _data::NamedTuple=NamedTuple()) = Soss.TV.UnitSimplex(length(d.alpha))
#Soss.xform(d::Distributions.LogNormal, _data::NamedTuple=NamedTuple()) = Soss.TV.as_positive_real
#Soss.TV.as(d::Distributions.LogNormal) = Soss.TV.as_positive_real
