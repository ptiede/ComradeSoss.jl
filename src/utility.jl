
"""
    make_ims(model, params; fov=160, npix=256)
Makes images using params. The params must be a vector of NamedTuples or
something of the same nature.
"""
function make_mean(model, params; fov=160.0, npix=256)
    ims = Comrade.StokesImage(zeros(npix,npix), fov, fov)
    for p in params
        m = Soss.predict(model, p)[:img]
        ims += Comrade.intensitymap(m, npix, npix, fov, fov)
    end
    return ims/length(params)
end


"""
    chi2vis(m::Comrade.AbstractModel, obs::Comrade.EHTObservation; kwargs...)
Computes the  chi square for the model given the data. The model
must be a Comrade model object, and obs is a EHTObservation file.

You can optionally pass a named tuple with the best fit gain amplitudes and phases
if you fit those and they'll be applied if applicable
"""
function chi2(m, obs::Comrade.EHTObservation{F,V}, gamps, gphase, f=0) where {F,V<:Comrade.EHTVisibilityDatum}
    u = Comrade.getdata(obs, :u)
    v = Comrade.getdata(obs, :v)
    visr = Comrade.getdata(obs, :visr)
    visi = Comrade.getdata(obs, :visi)
    error = hypot.(Comrade.getdata(obs, :error), f.*hypot.(visr, visi))
    bl = Comrade.getdata(obs, :baseline)

    mvis = Comrade.visibility.(Ref(m), u, v)

    chi2 = 0.0
    ga1 = 1
    ga2 = 1
    gθ1 = 0
    gθ2 = 0
    for i in eachindex(u)
        s1,s2 = bl[i]
        ga1 = gamps[s1]
        ga2 = gamps[s2]
        gθ1 = gphase[s1]
        gθ2 = gphase[s2]
        s,c = sincos(gθ1 - gθ2)
        vr = ga1*ga2*(real(mvis[i])*c + imag(mvis[i])*s)
        vi = ga1*ga2*(-real(mvis[i])*s + imag(mvis[i])*c)
        chi2 += abs2((visr[i] - vr)/error[i])
        chi2 += abs2((visi[i] - vi)/error[i])
    end
    return chi2
end


"""
    chi2vis(m::Comrade.AbstractModel, obs::Comrade.EHTObservation; gamps=nothing, gphase=nothing)
Computes the complex chi square for the model given the data. The model
must be a Comrade model object, and obs is a EHTObservation file.

You can optionally pass a named tuple with the best fit gain amplitudes and phases
if you fit those.
"""
function chi2(m,obs::Comrade.EHTObservation{F,D}, gains, args...) where {F,D<:Comrade.EHTVisibilityAmplitudeDatum}
    u = Comrade.getdata(obs, :u)
    v = Comrade.getdata(obs, :v)
    amp = Comrade.getdata(obs, :amp)
    error = Comrade.getdata(obs, :error)
    bl = Comrade.getdata(obs, :baseline)

    vamp = Comrade.amplitude.(Ref(m), u, v)

    chi2 = 0.0
    for i in eachindex(u)
        s1,s2 = bl[i]
        g1 = gains[s1]
        g2 = gains[s2]
        chi2 += ((amp[i] - g1*g2*vamp[i])/error[i])^2
    end
    return chi2
end

function chi2(m,obs::Comrade.EHTObservation{F,D}, args...) where {F,D<:Comrade.EHTClosurePhaseDatum}
    u1 = Comrade.getdata(obs, :u1)
    v1 = Comrade.getdata(obs, :v1)
    u2 = Comrade.getdata(obs, :u2)
    v2 = Comrade.getdata(obs, :v2)
    u3 = Comrade.getdata(obs, :u3)
    v3 = Comrade.getdata(obs, :v3)
    cp = Comrade.getdata(obs, :phase)
    error = Comrade.getdata(obs, :error)

    mcp = Comrade.closure_phase.(Ref(m),
                               u1, v1,
                               u2, v2,
                               u3, v3
                              )

    chi2 = 0.0
    for i in eachindex(u1)
        dθ = exp(im*cp[i]) - exp(im*mcp[i])
        chi2 += abs2(dθ/error[i])
    end
    return chi2
end

function chi2(m,obs::Comrade.EHTObservation{F,D}, args...) where {F,D<:Comrade.EHTLogClosureAmplitudeDatum}
    u1 = getdata(obs, :u1)
    v1 = getdata(obs, :v1)
    u2 = getdata(obs, :u2)
    v2 = getdata(obs, :v2)
    u3 = getdata(obs, :u3)
    v3 = getdata(obs, :v3)
    u4 = getdata(obs, :u4)
    v4 = getdata(obs, :v4)
    lcamp = getdata(obs, :amp)
    error = getdata(obs, :error)

    mlcamp = Comrade.logclosure_amplitude.(Ref(m),
                               u1, v1,
                               u2, v2,
                               u3, v3,
                               u4, v4
                              )

    chi2 = 0.0
    for i in eachindex(u1)
        dθ = lcamp[i] - mlcamp[i]
        chi2 += abs2(dθ/error[i])
    end
    return chi2
end


function ampres(m, gains, ampobs)
    u = Comrade.getdata(ampobs, :u)
    v = Comrade.getdata(ampobs, :v)
    amp = Comrade.getdata(ampobs, :amp)
    error = Comrade.getdata(ampobs, :error)
    bl = Comrade.getdata(ampobs, :baseline)

    vamp = Comrade.amplitude.(Ref(m), u, v)
    nres = zeros(length(vamp))
    for i in eachindex(u)
        s1,s2 = bl[i]
        g1 = gains[s1]
        g2 = gains[s2]
        nres[i] =  (amp[i] - g1*g2*vamp[i])/error[i]
    end
    return nres

end

function cpres(m, obscp)
    u1 = Comrade.getdata(obscp, :u1)
    v1 = Comrade.getdata(obscp, :v1)
    u2 = Comrade.getdata(obscp, :u2)
    v2 = Comrade.getdata(obscp, :v2)
    u3 = Comrade.getdata(obscp, :u3)
    v3 = Comrade.getdata(obscp, :v3)
    cp = Comrade.getdata(obscp, :phase)
    error = Comrade.getdata(obscp, :error)

    mcp = Comrade.closure_phase.(Ref(m),
                               u1, v1,
                               u2, v2,
                               u3, v3
                              )

    nres = zeros(length(mcp))
    for i in eachindex(u1)
        s,c  = sincosd(mcp[i] - cp[i])
        dθ = atand(s,c)
        nres[i] = dθ/error[i]
    end
    return nres
end

"""
    normalizedresiduals(model, gains, ampobs, cpobs)
Computes the normalized residuals of the amplitudes and closure phases,
using gain amplitude corrections.
"""
function normalizedresiduals(m, gains, ampobs, cpobs)
    return ampres(m, gains, ampobs), cpres(m, cpobs)
end
