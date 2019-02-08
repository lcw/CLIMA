# TODO:
# - Switch to logging
# - Add vtk
# - timestep calculation clean
# - Move stuff to device (to kill transfers back from GPU)
# - Check that Float32 is really being used in all the kernels properly

using CLIMAAtmosDycore
const AD = CLIMAAtmosDycore
using Canary
using MPI

using ParametersType
using PlanetParameters: R_d, cp_d, grav, cv_d
@parameter gamma_d cp_d/cv_d "Heat capcity ratio of dry air"
@parameter gdm1 R_d/cv_d "(equivalent to gamma_d-1)"

const HAVE_CUDA = try
  using CUDAdrv
  using CUDAnative
  true
catch
  false
end

macro hascuda(ex)
  return HAVE_CUDA ? :($(esc(ex))) : :(nothing)
end

# {{{ courant
function courantnumber(runner::AD.VanillaEuler.Runner{DeviceArray};
                       host=false) where {DeviceArray}
  host || error("Currently requires host configuration")
  state = runner.state
  config = runner.config
  params = runner.params
  cpubackend = DeviceArray == Array
  vgeo = cpubackend ? config.vgeo : Array(config.vgeo)
  Q = cpubackend ? state.Q : Array(state.Q)
  courantnumber(Val(params.dim), Val(params.N), vgeo, params.gravity, Q,
                config.mpicomm)
end


function courantnumber(::Val{dim}, ::Val{N}, vgeo, hasgravity, Q,
                       mpicomm) where {dim, N}
  _nstate = 5
  _ρ, _U, _V, _W, _E = 1:_nstate
  _nvgeo = 14
  (_ξx, _ηx, _ζx, _ξy, _ηy, _ζy, _ξz, _ηz, _ζz, _MJ, _MJI,
   _x, _y, _z) = 1:_nvgeo

  DFloat = eltype(Q)

  γ::DFloat       = gamma_d
  R_gas::DFloat   = R_d
  c_p::DFloat     = cp_d
  c_v::DFloat     = cv_d
  gravity::DFloat = hasgravity ? grav : 0

  Np = (N+1)^dim
  (~, ~, nelem) = size(Q)

  dt = [floatmax(DFloat)]

  #Compute DT
  @inbounds for e = 1:nelem, n = 1:Np
    ρ, U, V, W, E = Q[n, _ρ, e], Q[n, _U, e], Q[n, _V, e], Q[n, _W, e], Q[n, _E, e]
    ξx, ξy, ξz = vgeo[n, _ξx, e], vgeo[n, _ξy, e], vgeo[n, _ξz, e]
    ηx, ηy, ηz = vgeo[n, _ηx, e], vgeo[n, _ηy, e], vgeo[n, _ηz, e]
    ζx, ζy, ζz = vgeo[n, _ζx, e], vgeo[n, _ζy, e], vgeo[n, _ζz, e]
    yorz = (dim==2) ? vgeo[n, _y, e] : vgeo[n, _z, e]
    P = (R_gas/c_v)*(E - (U^2 + V^2 + W^2)/(2*ρ) - ρ*gravity*yorz)
    u, v, w = U/ρ, V/ρ, W/ρ
    if dim == 3
    dx=sqrt( (1/(2*ξx))^2 + 0*(1/(2*ηy))^2  + (1/(2*ζz))^2 )
    else
    dx=sqrt( (1/(2*ξx))^2 + 1*(1/(2*ηy))^2)
    end
    vel=sqrt( u^2 + v^2 + w^2)
    wave_speed = (vel + sqrt(γ * P / ρ))
    loc_dt = dx/wave_speed/N

    dt[1] = min(dt[1], loc_dt)
  end
  dt_min=MPI.Allreduce(dt[1], MPI.MIN, mpicomm)
end
# }}}

function meshgenerator(part, numparts, Ne, dim, DFloat)
  if dim == 2
    bbox = (range(DFloat(0); length=Ne+1, stop=1000),
            range(DFloat(0); length=Ne+1, stop=1000))
    periodic = (true, false)
  else
    bbox = (range(DFloat(0); length=Ne+1, stop=1000),
            range(DFloat(0); length=2,    stop=1000),
            range(DFloat(0); length=Ne+1, stop=1000))
    periodic = (true, true, false)
  end

  brickmesh(bbox, periodic, part=part, numparts=numparts)
end

function main()
  MPI.Initialized() || MPI.Init()
  MPI.finalize_atexit()

  mpicomm = MPI.COMM_WORLD
  mpirank = MPI.Comm_rank(mpicomm)
  mpisize = MPI.Comm_size(mpicomm)

  # FIXME: query via hostname
  @hascuda device!(mpirank % length(devices()))

  timeinitial = 0.0
  timeend = 10
  Ne = 10
  N  = 4

  DFloat = Float64
  dim = 3
  backend = HAVE_CUDA ? CuArray : Array

  runner = AD.Runner(mpicomm,
                     #Space Discretization and Parameters
                     :VanillaEuler,
                     (DFloat = DFloat,
                      DeviceArray = backend,
                      meshgenerator = (part, numparts) ->
                      meshgenerator(part, numparts, Ne, dim,
                                    DFloat),
                      dim = dim,
                      gravity = true,
                      N = N,
                      viscosity = 2.134
                     ),
                     # Time Discretization and Parameters
                     :LSRK,
                     (),
                    )

  # Set the initial condition with a function
  AD.initspacestate!(runner, host=true) do (x...)
    DFloat = eltype(x)
    γ::DFloat       = gamma_d
    p0::DFloat      = 100000
    R_gas::DFloat   = R_d
    c_p::DFloat     = cp_d
    c_v::DFloat     = cv_d
    gravity::DFloat = grav

    r = sqrt((x[1] - 500)^2 + (x[dim] - 350)^2)
    rc::DFloat = 250
    θ_ref::DFloat = 300
    θ_c::DFloat = 0.5
    Δθ::DFloat = 0
    if r <= rc
      Δθ = θ_c * (1 + cos(π * r / rc)) / 2
    end
    θ_k = θ_ref + Δθ
    π_k = 1 - gravity / (c_p * θ_k) * x[dim]
    c = c_v / R_gas
    ρ_k = p0 / (R_gas * θ_k) * (π_k)^c
    ρ = ρ_k
    u = zero(DFloat)
    v = zero(DFloat)
    w = zero(DFloat)
    U = ρ * u
    V = ρ * v
    W = ρ * w
    Θ = ρ * θ_k
    P = p0 * (R_gas * Θ / p0)^(c_p / c_v)
    T = P / (ρ * R_gas)
    E = ρ * (c_v * T + (u^2 + v^2 + w^2) / 2 + gravity * x[dim])

    ρ, U, V, W, E
  end

  # Compute a (bad guess) for the time step
  base_dt = courantnumber(runner.spacerunner, host=true)
  mpirank == 0 && @show base_dt
  nsteps = ceil(Int64, timeend / base_dt)
  dt = timeend / nsteps

  mpirank == 0 && @show (nsteps, dt)

  # Set the time step
  AD.inittimestate!(runner, dt)

  eng0 = AD.L2solutionnorm(runner; host=true)
  # mpirank == 0 && @show eng0

  # Setup the info callback
  io = mpirank == 0 ? stdout : open("/dev/null", "w")
  show(io, "text/plain", runner.spacerunner)
  cbinfo =
  AD.GenericCallbacks.EveryXWallTimeSeconds(10, mpicomm) do
    println(io, runner.spacerunner)
  end

  # Setup the vtk callback
  mkpath("viz")
  dump_vtk(step) = AD.writevtk(runner,
                               "viz/RTB"*
                               "_dim_$(dim)"*
                               "_DFloat_$(DFloat)"*
                               "_backend_$(backend)"*
                               "_mpirank_$(mpirank)"*
                               "_step_$(step)")
  step = 0
  cbvtk = AD.GenericCallbacks.EveryXSimulationSteps(10) do
    # TODO: We should add queries back to time stepper for this
    step += 1
    dump_vtk(step)
    nothing
  end

  dump_vtk(0)
  AD.run!(runner; numberofsteps=nsteps, callbacks=(cbinfo, cbvtk))
  dump_vtk(nsteps)

  engf = AD.L2solutionnorm(runner; host=true)

  mpirank == 0 && @show engf
  mpirank == 0 && @show eng0 - engf
  mpirank == 0 && @show engf/eng0
  mpirank == 0 && println()

  nothing
end

main()
