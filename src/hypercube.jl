using Soss
using Soss: ConditionalModel, getmoduletypencoding, argvals, observations,
             Assign, Return, LineNumber, Sample, testvalue, @q, type2model,
             loadvals, getntkeys, buildSource, isleaf, _unwrap_type
using NestedTuples
using MacroTools
import MeasureTheory
const MT = MeasureTheory
import Distributions
const Dists = Distributions
import HypercubeTransform


HypercubeTransform.ascube(m::ConditionalModel{A, B}, _data::NamedTuple) where {A,B} = HypercubeTransform.ascube(m | _data)
#HypercubeTransform.ascube(::NamedTuple{(), Tuple{}}, _data) = nothing

function HypercubeTransform.ascube(m::ConditionalModel{A, B}) where {A,B}
    return _ascube(getmoduletypencoding(m), Model(m), argvals(m), observations(m))
end

# function ascube(m::Model{EmptyNTtype, B}) where {B}
#     return ascube(m,NamedTuple())
# end

HypercubeTransform.ascube(d, _data) = nothing

HypercubeTransform.ascube(μ::MT.AbstractMeasure,  _data::NamedTuple=NamedTuple()) = HypercubeTransform.ascube(μ)
HypercubeTransform.ascube(μ::Dists.Distribution,  _data::NamedTuple=NamedTuple()) = HypercubeTransform.ascube(μ)

HypercubeTransform.ascube(d::Dists.AbstractMvNormal, _data::NamedTuple=NamedTuple()) = HypercubeTransform.ascube(d)
HypercubeTransform.ascube(::NamedTuple{(), Tuple{}}) = nothing




export sourceascube

sourceascube(m::Model) = sourceascube()(m)

function sourceascube(_data=NamedTuple())
    function(_m::Model)

        _datakeys = getntkeys(_data)
        proc(_m, st::Assign)        = :($(st.x) = $(st.rhs))
        proc(_m, st::Return)     = nothing
        proc(_m, st::LineNumber) = nothing

        function proc(_m, st::Sample)
            x = st.x
            xname = QuoteNode(x)
            rhs = st.rhs

            thecode = @q begin
                _t = HypercubeTransform.ascube($rhs, get(_data, $xname, NamedTuple()))
                if !isnothing(_t)
                    _result = merge(_result, ($x=_t,))
                end
            end

            # Non-leaves might be referenced later, so we need to be sure they
            # have a value
            isleaf(_m, st.x) || pushfirst!(thecode.args, :($x = Soss.testvalue($rhs)))

            return thecode
        end


        wrap(kernel) = @q begin
            _result = NamedTuple()
            $kernel
            $HypercubeTransform.ascube(_result)
        end

        buildSource(_m, proc, wrap) |> MacroTools.flatten

    end
end

Soss.@gg function _ascube(M::Type{<:Soss.GeneralizedGenerated.TypeLevel}, _m::Model{Asub,B}, _args::A, _data) where {Asub,A,B}
    body = type2model(_m) |> sourceascube(_data) |> loadvals(_args, _data)
    Soss.@under_global Soss.from_type(_unwrap_type(M)) @q let M
        $body
    end
end
