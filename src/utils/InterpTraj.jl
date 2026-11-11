"""
    traj_shifted = InterpTrajTime(traj, dt, delTime, datatime; intermode=AkimaMonotonicInterpolation())
    traj_shifted = InterpTrajTime(traj, dt, delTime; intermode=AkimaMonotonicInterpolation())
# Description
    Interpolates trajectory along the 1st dimension using Akima interpolation.

# Arguments
- `traj::AbstractArray{<:Real, 2}`: trajectory data (2D array, where each row is a point in the trajectory)
- `dt::Float64`: time delay between input samples
- `delTime::Float64`: time delay to be added or subtracted from time
- `datatime`: output sample times in seconds; its precision may differ from `traj`.
  Input knots are `dt * (0:nSample-1) - delTime`; values outside their range are zero.

# Keywords
- `intermode::Interpolations.InterpolationType`: interpolation mode (default is AkimaMonotonicInterpolation). 
        other Monotonic methods from Interpolations.jl are:
    1. LinearMonotonicInterpolation
    2. FiniteDifferenceMonotonicInterpolation
    3. CardinalMonotonicInterpolation
    4. AkimaMonotonicInterpolation
    5. FritschCarlsonMonotonicInterpolation
    6. FritschButlandMonotonicInterpolation
    7. SteffenMonotonicInterpolation

# Returns
- `traj_interp::Matrix{T}`: one row per output time, preserving the trajectory's element type.

# Examples
```julia-repl
julia> traj_shifted = InterpTrajTime(traj, dt, delTime)
```
"""
function InterpTrajTime(
    traj      ::AbstractArray{T, 2}      , 
    dt        ::Real                     , 
    delTime   ::Real                     ,
    datatime  ::AbstractVector{<:Real}   ;
    intermode ::Interpolations.InterpolationType = AkimaMonotonicInterpolation()
    ) where {T<:Real}
    nSample, nTerm = size(traj)
    check_trajectory_interval(dt)
    nSample >= 2 || throw(ArgumentError("Trajectory interpolation needs at least two input samples."))
    isfinite(delTime) || throw(ArgumentError("The trajectory delay must be finite."))
    all(isfinite, datatime) || throw(ArgumentError("Output sample times must be finite."))

    # Calculate adjusted time (delayed)
    trajTim = dt * (0:nSample-1) .- delTime

    # Perform Akima interpolation (assuming 1D interpolation for each row of the trajectory)
    traj_interp = Matrix{T}(undef, length(datatime), nTerm)
    for i in 1:nTerm  # Loop over each column (dimension) of the trajectory
        itp = interpolate(trajTim, traj[:, i], intermode)
        etp = extrapolate(itp, 0)  # Extrapolate to the beginning of the time vector
        traj_interp[:, i] .= etp.(datatime)
    end

    return traj_interp
end

"""
    If dwell time is equal between trajectories and the sampling points.
    You can use this method.
"""
function InterpTrajTime(
    traj      ::AbstractArray{T, 2}      , 
    dt        ::Real                     , 
    delTime   ::Real                     ;
    intermode ::Interpolations.InterpolationType = AkimaMonotonicInterpolation()
    ) where {T<:Real}
    nSample, nTerm = size(traj)
    # Create time vector for input trajectory
    datatime = dt * (0:nSample-1)
    return InterpTrajTime(traj, dt, delTime, datatime; intermode=intermode)
end


"""
    For multishot cases, you can use this method.
    traj 
# Arguments
- `traj::AbstractArray{<:Real, 3}`: [nShot, nSample, nTerm], trajectory data (2D array, where each row is a point in the trajectory)
- `dt::Float64`: time delay between input samples
- `delTime::Float64`: [nShot], time delay to be added or subtracted from time

"""
function InterpTrajTime(
    traj      ::AbstractArray{T, 3}      , 
    dt        ::Real                     , 
    delTime   ::AbstractVector{<:Real}   ,
    datatime  ::AbstractArray{<:Real, 2} ;
    intermode ::Interpolations.InterpolationType = AkimaMonotonicInterpolation()
    ) where {T<:Real}
    @assert size(traj, 1) == length(delTime) "Number of shots in trajectory must match length of delTime vector"
    @assert size(datatime, 1) == length(delTime) "Number of shots in datatime must match length of delTime vector"

    nShot, _, nTerm = size(traj)
    check_trajectory_interval(dt)
    traj_out = Array{T}(undef, nShot, size(datatime, 2), nTerm)
    for i in 1:nShot
        shot_interp = InterpTrajTime(traj[i, :, :], dt, delTime[i], datatime[i, :]; intermode=intermode)
        traj_out[i, :, :] .= shot_interp
    end
    return traj_out
end
