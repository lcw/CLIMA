module JDE

using OrdinaryDiffEq

using ..CLIMAAtmosDycore
AD = CLIMAAtmosDycore
using Base: @kwdef

"""
    Parameters

Data structure containing the low storage RK parameters
"""
# {{{ Parameters
@kwdef struct Parameters # <: AD.AbstractTimeParameter
end
# }}}

"""
    Configuration

Data structure containing the low storage RK configuration
"""
# {{{ Configuration
struct Configuration # <: AD.AbstractTimeConfiguration
  function Configuration(params::Parameters,
                         mpicomm,
                         spacerunner::AD.AbstractSpaceRunner)
    new()
  end
end
#}}}

"""
    State

Data structure containing the low storage RK state
"""
# {{{ State
mutable struct State # <: AD.AbstractTimeState
  function State(config::Configuration, x...)
    new()
  end
end
# }}}

"""
    Runner

Data structure containing the runner for the vanilla DG discretization of
the compressible Euler equations

"""
# {{{ Runner
struct Runner <: AD.AbstractTimeRunner
  params::Parameters
  config::Configuration
  state::State
  function Runner(mpicomm, spacerunner::AD.AbstractSpaceRunner; args...)
    params = Parameters(;args...)
    config = Configuration(params, mpicomm, spacerunner)
    state = State(config, params, spacerunner)
    new(params, config, state)
  end
end
AD.createrunner(::Val{:JDE}, m, s; a...) = Runner(m, s; a...)
# }}}

# {{{ run!
function AD.run!(runner::Runner, spacerunner::AD.AbstractSpaceRunner;
                 ad_timerange=(0.0,1.0),
                 kwargs...) where {SP, T<:State}

  function f(du,u,p,t)
    fill!(du, zero(eltype(du)))
    AD.rhs!(du, spacerunner, Q=u)
  end

  Q = spacerunner[:Q]

  prob = ODEProblem(f,Q,ad_timerange)
  sol = solve(prob, RK4())

  Q .= sol[end]
  spacerunner[:time] = sol.t[end]

end
# }}}

end
