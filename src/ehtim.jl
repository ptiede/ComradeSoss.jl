Base.@kwdef struct ObsChar
    flag_lsnr=false
    deblur=false
    zblscale=false
    lkgerr=0.00
    floor=0.00
    t_gather=60.0
end



function saveobschar(obschar::ObsChar, model, datafile, file)
    open(file, "w") do io
        println(io, "Fitting Datafile: ", datafile)
        println(io, "Using model: \n", model)
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
    scandata(jscan, obs, obschar)
Returns a processed versoin of `jscan` scan of the ehtim obsdata object `obs`.
The processing uses the processing options defined in `obschar`
"""
function scandata(jscan, obs, obschar)
    obs_scan = obs.copy()
    obs_scan.data = obs_scan.tlist(t_gather=obschar.t_gather)[jscan]
    obs_fit = process_eht_obs(obs_scan, obschar)
end
