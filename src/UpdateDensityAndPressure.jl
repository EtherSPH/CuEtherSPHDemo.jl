#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2024/09/01 17:51:38
  @ license: MIT
  @ description:
 =#

@inline function cuUpdateDensityAndPressure!(
    n_particles::IntType,
    delta_t::RealType,
    eosFunction::Function,
    cu_is_alive,
    cu_density,
    cu_type,
    cu_pressure,
    cu_density_ratio,
    cu_sum_kernel_weight,
    cu_sum_kernel_weighted_pressure,
)::Nothing
    index = (blockIdx().x - IntType(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for p in index:stride:n_particles
        @inbounds if p > n_particles
            break
        end
        @inbounds if cu_is_alive[p] == false
            continue
        end
        @inbounds if cu_type[p] == FLUID_TAG
            @inbounds cu_density[p] += cu_density_ratio[p] * delta_t
            @inbounds cu_pressure[p] = eosFunction(cu_density[p])
            @inbounds cu_density_ratio[p] = RealType(0)
        elseif cu_type[p] == WALL_TAG
            @inbounds if cu_sum_kernel_weight[p] > RealType(0)
                @inbounds cu_pressure[p] = cu_sum_kernel_weighted_pressure[p] / cu_sum_kernel_weight[p]
            else
                @inbounds cu_pressure[p] = RealType(0)
            end
            @inbounds cu_sum_kernel_weight[p] = RealType(0)
            @inbounds cu_sum_kernel_weighted_pressure[p] = RealType(0)
        end
    end
    return nothing
end

@inline function updateDensityAndPressure!(
    particle_system::ParticleSystem,
    delta_t::RealType,
    eosFunction::Function;
    numthreads = 256,
)::Nothing
    numblocks = ceil(IntType, particle_system.n_particles_ / numthreads)
    CUDA.@sync begin
        @cuda threads = numthreads blocks = numblocks cuUpdateDensityAndPressure!(
            particle_system.n_particles_,
            delta_t,
            eosFunction,
            particle_system.cu_is_alive_,
            particle_system.cu_density_,
            particle_system.cu_type_,
            particle_system.cu_pressure_,
            particle_system.cu_density_ratio_,
            particle_system.cu_sum_kernel_weight_,
            particle_system.cu_sum_kernel_weighted_pressure_,
        )
    end
    return nothing
end
