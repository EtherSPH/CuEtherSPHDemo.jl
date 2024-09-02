#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2024/09/01 21:15:47
  @ license: MIT
  @ description:
 =#

@inline function cuAccelerateAndMove!(
    n_particles::IntType,
    delta_t::RealType,
    body_force_x::RealType,
    body_force_y::RealType,
    cu_is_alive,
    cu_type,
    cu_position,
    cu_velocity,
    cu_acceleration,
)::Nothing
    index = (blockIdx().x - IntType(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    delta_t_square = RealType(0.5 * delta_t * delta_t)
    for p in index:stride:n_particles
        @inbounds if p > n_particles
            break
        end
        @inbounds if cu_is_alive[p] == false
            continue
        end
        @inbounds if cu_type[p] == FLUID_TAG
            @inbounds cu_acceleration[p, 1] += body_force_x
            @inbounds cu_acceleration[p, 2] += body_force_y
            @inbounds cu_position[p, 1] += cu_velocity[p, 1] * delta_t + cu_acceleration[p, 1] * delta_t_square
            @inbounds cu_position[p, 2] += cu_velocity[p, 2] * delta_t + cu_acceleration[p, 2] * delta_t_square
            @inbounds cu_velocity[p, 1] += cu_acceleration[p, 1] * delta_t
            @inbounds cu_velocity[p, 2] += cu_acceleration[p, 2] * delta_t
            @inbounds cu_acceleration[p, 1] = RealType(0)
            @inbounds cu_acceleration[p, 2] = RealType(0)
        end
    end
    return nothing
end

@inline function accelerateAndMove!(
    particle_system::ParticleSystem,
    delta_t::RealType,
    body_force_x::RealType,
    body_force_y::RealType;
    numthreads = 256,
)::Nothing
    numblocks = ceil(Int, particle_system.n_particles_ / numthreads)
    CUDA.@sync begin
        @cuda threads = numthreads blocks = numblocks cuAccelerateAndMove!(
            particle_system.n_particles_,
            delta_t,
            body_force_x,
            body_force_y,
            particle_system.cu_is_alive_,
            particle_system.cu_type_,
            particle_system.cu_position_,
            particle_system.cu_velocity_,
            particle_system.cu_acceleration_,
        )
    end
    return nothing
end
