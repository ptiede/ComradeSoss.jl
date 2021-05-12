
"""
    make_ims(model, params; fov=160, npix=256)
Makes images using params. The params must be a vector of NamedTuples or
something of the same nature.
"""
function make_mean(model, params; fov=160.0, npix=256)
    ims = ROSE.StokesImage(zeros(npix,npix), fov, fov)
    for p in params
        m = Soss.predict(model, p)[:img]
        ims += ROSE.intensitymap(m, npix, npix, fov, fov)
    end
    return ims/length(params)
end


"""
    chi2vis(m::ROSE.AbstractModel, obs::ROSE.EHTObservation; kwargs...)
Computes the  chi square for the model given the data. The model
must be a ROSE model object, and obs is a EHTObservation file.

You can optionally pass a named tuple with the best fit gain amplitudes and phases
if you fit those and they'll be applied if applicable
"""
function chi2(m, obs::ROSE.EHTObservation{F,V}, gamps, gphase) where {F,V<:ROSE.EHTVisibilityDatum}
    u = ROSE.getdata(obs, :u)
    v = ROSE.getdata(obs, :v)
    visr = ROSE.getdata(obs, :visr)
    visi = ROSE.getdata(obs, :visi)
    error = ROSE.getdata(obs, :error)
    bl = ROSE.getdata(obs, :baselines)

    mvis = ROSE.visibility.(Ref(m), u, v)

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
    chi2vis(m::ROSE.AbstractModel, obs::ROSE.EHTObservation; gamps=nothing, gphase=nothing)
Computes the complex chi square for the model given the data. The model
must be a ROSE model object, and obs is a EHTObservation file.

You can optionally pass a named tuple with the best fit gain amplitudes and phases
if you fit those.
"""
function chi2(m,obs::ROSE.EHTObservation{F,D}, gains, args...) where {F,D<:ROSE.EHTVisibilityAmplitudeDatum}
    u = ROSE.getdata(obs, :u)
    v = ROSE.getdata(obs, :v)
    amp = ROSE.getdata(obs, :amp)
    error = ROSE.getdata(obs, :error)
    bl = ROSE.getdata(obs, :baselines)

    vamp = ROSE.visibility_amplitude.(Ref(m), u, v)

    chi2 = 0.0
    for i in eachindex(u)
        s1,s2 = bl[i]
        g1 = gains[s1]
        g2 = gains[s2]
        chi2 += ((amp[i] - g1*g2*vamp[i])/error[i])^2
    end
    return chi2
end

function chi2(m,obs::ROSE.EHTObservation{F,D}, args...) where {F,D<:ROSE.EHTClosurePhaseDatum}
    u1 = ROSE.getdata(obs, :u1)
    v1 = ROSE.getdata(obs, :v1)
    u2 = ROSE.getdata(obs, :u2)
    v2 = ROSE.getdata(obs, :v2)
    u3 = ROSE.getdata(obs, :u3)
    v3 = ROSE.getdata(obs, :v3)
    cp = ROSE.getdata(obs, :phase)
    error = ROSE.getdata(obs, :error)

    mcp = ROSE.closure_phase.(Ref(m),
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

function ampres(m, gains, ampobs)
    u = ROSE.getdata(ampobs, :u)
    v = ROSE.getdata(ampobs, :v)
    amp = ROSE.getdata(ampobs, :amp)
    error = ROSE.getdata(ampobs, :error)
    bl = ROSE.getdata(ampobs, :baselines)

    vamp = ROSE.visibility_amplitude.(Ref(m), u, v)
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
    u1 = ROSE.getdata(obscp, :u1)
    v1 = ROSE.getdata(obscp, :v1)
    u2 = ROSE.getdata(obscp, :u2)
    v2 = ROSE.getdata(obscp, :v2)
    u3 = ROSE.getdata(obscp, :u3)
    v3 = ROSE.getdata(obscp, :v3)
    cp = ROSE.getdata(obscp, :phase)
    error = ROSE.getdata(obscp, :error)

    mcp = ROSE.closure_phase.(Ref(m),
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
