using MLStyle
import Distributions as Dists
using Distributions: quantile
using Soss


hform(m::Soss.ConditionalModel{A, B}, _data::NamedTuple) where {A,B} = hform(m | _data)


"""
    hform(m)
Create the hypercube transform from Soss models that moves from the
unit hypercube to the model parameter space. To transform use the transform function
"""
function hform(m::Soss.ConditionalModel{A,B}) where {A,B}
    return _hform(Soss.getmoduletypencoding(m),
                  Soss.Model(m),
                  Soss.argvals(m),
                  Soss.observations(m)
                  )
end







sourceHform(m::Model) = sourceHform()(m)

function sourceHform(_data=NamedTuple())
    function (_m::Model)
        _datakeys = Soss.getntkeys(_data)
        proc(_m, st::Soss.Assign) = :($(st.x) = $(st.rhs))
        proc(_m, st::Soss.Return) = nothing
        proc(_m, st::Soss.LineNumber) = nothing

        function proc(_m, st::Soss.Sample)
            x = st.x
            xname = QuoteNode(x)
            rhs = st.rhs

            thecode = Soss.@q begin
                _t = hform($rhs, get(_data, $xname, NamedTuple()))
                if !isnothing(_t)
                    _result = merge(_result, ($x=_t,))
                end
            end

            Soss.isleaf(_m, st.x) || pushfirst!(thecode.args, :($x = Soss.testvalue($rhs)))

            return thecode
        end

        wrap(kernel) = Soss.@q begin
            _result=NamedTuple()
            $kernel
            HyperCubeTransform(_result)
        end

        Soss.buildSource(_m, proc, wrap) |> Soss.MacroTools.flatten
    end
end

abstract type AbstractHyperCubeTransform end


struct HyperCubeTransform{T} <: AbstractHyperCubeTransform
    dist::T
    dimension::Int
end

const NCube = Tuple{Vararg{AbstractHyperCubeTransform, N}}

function dimension(h::HyperCubeTransform{T}) where {T}
    return h.dimension
end

function _sum_dimensions(h)
    dim = 0
    for hh in values(h)
        dim += dimension(hh)
    end
    return dim
end


function HyperCubeTransform(h::T) where {T<:NCube}
    dim = _sum_dimensions(h)
    return HyperCubeTransform(h,dim)
end

function HyperCubeTransform(h::T) where {N,S<:NCube, T<:NamedTuple{N,S}}
    dim = _sum_dimensions(h)
    return HyperCubeTransform(h, dim)
end


function hform(d, _data::NamedTuple)
    if hasmethod(Dists.quantile, (typeof(d),))
        return HyperCubeTransform(d,1)
    end

    error("Not implemented:\nhform($d)")
end

hform(d, _data) = nothing


function transform(h::HyperCubeTransform{<:Dists.UnivariateDistribution}, p)
    return quantile(h.dist, p)
end

function transform(h::HyperCubeTransform{<:Dists.Product}, p)
    return quantile.(h.dist.v, p)
end

function transform(h::HyperCubeTransform{<:NamedTuple}, p)
    names = keys(h.dist)
    results = NamedTuple()
    index=1
    for n in names
        stop = start + dimension(h.dist[n]) - 1
        results = merge(results, (n => transform(h.dist[n], @view p[start:stop]),))
        start = stop + 1
    end
    return results
end

function transform_with(h)


Soss.@gg function _hform(M::Type{<:Soss.TypeLevel},
                         _m::Model{Asub,B},
                         _args::A,
                         _data) where {Asub, A,B}

    body = Soss.type2model(_m) |> sourceHform(_data) |> Soss.loadvals(_args, _data)
    Soss.@under_global Soss.from_type(Soss._unwrap_type(M)) Soss.@q let M
        $body
    end
end

function hform(d::Dists.Distribution{Dists.Univariate}, _data::NamedTuple=NamedTuple())
    return HyperCubeTransform(d,1)
end

function hform(d::Dists.Product, _data::NamedTuple=NamedTuple())
    n = length(d)
    return HyperCubeTransform(d,n)
end

function hform(d::ProductMeasure, _data::NamedTuple)
    dist = Dists.Product([d.f(i) for i in d.pars])
    n = length(d)
    return HyperCubeTransform(dist, n)
end
