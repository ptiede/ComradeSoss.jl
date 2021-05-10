Base.@kwdef struct ObsChar
    flag_lsnr=false
    deblur=false
    zblscale=false
    lkgerr=0.02
    floor=0.01
    t_gather=60.0
end



function saveobschar(obschar::ObsChar, model, datafile, file)
    open(file, "w") do io
        println(io, "Fitting Datafile: ", datafile)
        println(io, "Using model: ", model)
        println(io, "flag_lsnr = ", obschar.flag_lsnr)
        println(io, "deblur = ", obschar.deblur)
        println(io, "zblscale = ", obschar.zblscale)
        println(io, "lkgerr = ", obschar.lkgerr)
        println(io, "floor = ", obschar.floor)
        println(io, "t_gather = ", obschar.t_gather)
    end
end


"""
    process_eht_obs(obs, obschar)
Processes an ehtim obsdata object `obs` using the flags
defined in `obschar`

Returns the processed version of the `obs` object
"""
function process_eht_obs(obs, obschar::ObsChar)

    @info "Adding $(obschar.lkgerr) leakage error"
    @info "Adding $(obschar.floor) floor"
    obs_scan = obs.copy()
    obs_fit = obs_scan.copy()
    obs_fit = obs_fit.add_leakage_noise(min_noise=obschar.lkgerr)
    noise = get(obs_fit.data,"sigma")
    noise = sqrt.(noise.^2 .+ obschar.floor^2)
    set!(obs_fit.data, "sigma", noise)
    if obschar.deblur
        sm = ehtim.scattering.stochastic_optics.ScatteringModel()
        @info "Deblurring data (really should blur model)"
        obs_fit = sm.Deblur_obs(obs_fit)
    end

    if obschar.zblscale
        @info "Rescaling zero baseline flux"
        vis = get(obs_fit.data, "vis")
        zbl = median(get(obs.flag_uvdist(uv_max=0.1e9).unpack("amp", debias=true), "amp"))
        set!(obs_fit.data, "vis", vis./zbl)
        set!(obs_fit.data, "sigma", noise./zbl)
    end
    if obschar.flag_lsnr
        @info "Flagging low snr point"
        obs_fit = obs_fit.flag_low_snr()
        if length(obs_fit.data) < 6
            return nothing
        end
    end
    obs_fit.add_cphase()
    obs_fit.add_amp(debias=true)
    return obs_fit
end

function getvisfield(obs)
    obsamps = obs.data::PyObject
    u = deepcopy((get(obsamps, Vector{Float64}, "u")))
    v = deepcopy((get(obsamps, Vector{Float64}, "v")))
    err = deepcopy((get(obsamps, Vector{Float64}, "sigma")))
    vis = deepcopy((get(obsamps, Vector{Complex{Float64}}, "vis")))
    t1 = Symbol.(deepcopy((get(obsamps, Vector{String}, "t1"))))
    t2 = Symbol.(deepcopy((get(obsamps, Vector{String}, "t2"))))
    baselines = tuple.(t1, t2)
    time = deepcopy((get(obsamps, Vector{Float64}, "time")))
    freq = zeros(length(time))
    bw = zeros(length(time))

    return  StructArray{ROSE.EHTVisibilityDatum{Float64}}(
        visr = real.(vis),
        visi = imag.(vis),
        u = u/rad2μas,
        v = v/rad2μas,
        error = err,
        time = time,
        frequency = freq,
        bandwidth = bw,
        baselines = baselines
    )
end


function getampfield(obs)
    obsamps = obs.amp::PyObject
    uamp = deepcopy(get(obsamps, Vector{Float64}, "u"))
    vamp = deepcopy(get(obsamps, Vector{Float64}, "v"))
    erramp = deepcopy(get(obsamps, Vector{Float64}, "sigma"))
    amps = deepcopy(get(obsamps, Vector{Float64}, "amp"))
    t1 = Symbol.(deepcopy(get(obsamps, Vector{String}, "t1")))
    t2 = Symbol.(deepcopy(get(obsamps, Vector{String}, "t2")))
    baselines = tuple.(t1, t2)
    time = deepcopy(get(obsamps, Vector{Float64}, "time"))
    freq = zeros(length(time))
    bw = zeros(length(time))

    return  StructArray{ROSE.EHTVisibilityAmplitudeDatum{Float64}}(
        amp = amps,
        u = uamp/rad2μas,
        v = vamp/rad2μas,
        error = erramp,
        time = time,
        frequency = freq,
        bandwidth = bw,
        baselines = baselines
    )
end

function getcpfield(obs)
    obscp = obs.cphase::PyObject
    u1 = deepcopy((get(obscp, Vector{Float64}, "u1"))./rad2μas)
    v1 = deepcopy((get(obscp, Vector{Float64}, "v1"))./rad2μas)
    u2 = deepcopy((get(obscp, Vector{Float64}, "u2"))./rad2μas)
    v2 = deepcopy((get(obscp, Vector{Float64}, "v2"))./rad2μas)
    u3 = deepcopy((get(obscp, Vector{Float64}, "u3"))./rad2μas)
    v3 = deepcopy((get(obscp, Vector{Float64}, "v3"))./rad2μas)
    cp = deg2rad.(deepcopy((get(obscp, Vector{Float64}, "cphase"))))
    errcp = deg2rad.(deepcopy((get(obscp, Vector{Float64}, "sigmacp"))))

    t1 = Symbol.(deepcopy((get(obscp, Vector{String}, "t1"))))
    t2 = Symbol.(deepcopy((get(obscp, Vector{String}, "t2"))))
    t3 = Symbol.(deepcopy((get(obscp, Vector{String}, "t3"))))
    baselines = tuple.(t1, t2, t3)
    time = deepcopy(get(obscp, Vector{Float64}, "time"))
    freq = zeros(length(time))
    bw = zeros(length(time))

    return StructArray{ROSE.EHTClosurePhaseDatum{Float64}}(
        phase = cp,
        u1 = u1,
        v1 = v1,
        u2 = u2,
        v2 = v2,
        u3 = u3,
        v3 = v3,
        error = errcp,
        time = time,
        frequency = freq,
        bandwidth = bw,
        baselines = baselines
    )

end

function getradec(obs)::Tuple{Float64, Float64}
    return (float(obs.ra), float(obs.dec))
end

function getmjd(obs)::Int
    return Int(obs.mjd)
end

function getsource(obs)::Symbol
    return Symbol(obs.source)
end


"""
    extract_vis(obs)
Extracts the visibility amplitudes from an ehtim observation object

Returns an EHTObservation with visibility amplitude data
"""

function extract_amps(obs)
    data = getampfield(obs)
    ra, dec = getradec(obs)
    mjd = getmjd(obs)
    source = getsource(obs)

    return ROSE.EHTObservation(data = data, mjd = mjd,
                   ra = ra, dec= dec,
                   source = source,
    )
end


"""
    extract_vis(obs)
Extracts the complex visibilities from an ehtim observation object

Returns an EHTObservation with complex visibility data
"""
function extract_vis(obs)
    data = getvisfield(obs)
    ra, dec = getradec(obs)
    mjd = getmjd(obs)
    source = getsource(obs)

    return ROSE.EHTObservation(data = data, mjd = mjd,
                   ra = ra, dec= dec,
                   source = source,
    )
end

"""
    extract_vis(obs)
Extracts the closure phases from an ehtim observation object

Returns an EHTObservation with closure phases datums
"""

function extract_cphase(obs)
    data = getcpfield(obs)
    ra, dec = getradec(obs)
    mjd = getmjd(obs)
    source = getsource(obs)

    return ROSE.EHTObservation(data = data, mjd = mjd,
                   ra = ra, dec= dec,
                   source = source,
    )

end

"""
    scandata(jscan, obs, obschar)
Returns a processed versoin of `jscan` scan of the ehtim obsdata object `obs`.
The processing uses the processing options defined in `obschar`
"""
function scandata(jscan, obs, obschar)
    obs_scan = obs.copy()
    obs_scan.data = obs_scan.tlist(t_gather=obschar.t_gather)[jscan]
    obs_fit = process_eht_obs(obs_scan, obschar)
end
