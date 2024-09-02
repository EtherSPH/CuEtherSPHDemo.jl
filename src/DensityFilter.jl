#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2024/09/01 21:42:52
  @ license: MIT
  @ description:
 =#

@inline function cuDensityFilterInteraction!(
    n_particles::IntType,
    kernel::SPHKernel,
    cu_is_alive,
    cu_type,
    cu_neighbour_count,
    cu_neighbour_index_list,
    cu_neighbour_distance_list,
    cu_mass,
    cu_density,
    cu_sum_kernel_weight,
    cu_sum_kernel_weighted_density,
)::Nothing
    index = (blockIdx().x - IntType(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for p in index:stride:n_particles
        @inbounds if p > n_particles
            break
        end
        @inbounds if cu_is_alive[p] == false || cu_neighbour_count[p] == IntType(0)
            continue
        end
        for i_neighbour in 1:cu_neighbour_count[p]
            @inbounds q = cu_neighbour_index_list[p, i_neighbour]
            if cu_is_alive[q] == true
                if cu_type[p] == FLUID_TAG && cu_type[q] == FLUID_TAG
                    @inbounds w = kernelValue(cu_neighbour_distance_list[p, i_neighbour], kernel)
                    @inbounds kernel_weight = w * cu_mass[q] / cu_density[q]
                    @inbounds cu_sum_kernel_weight[p] += kernel_weight
                    @inbounds cu_sum_kernel_weighted_density[p] += kernel_weight * cu_density[q]
                end
            end
        end
    end
    return nothing
end

@inline function densityFilterInteraction!(
    particle_system::ParticleSystem,
    kernel::SPHKernel;
    numthreads = 256,
)::Nothing
    numblocks = ceil(Int, particle_system.n_particles_ / numthreads)
    CUDA.@sync begin
        @cuda threads = numthreads blocks = numblocks cuDensityFilterInteraction!(
            particle_system.n_particles_,
            kernel,
            particle_system.cu_is_alive_,
            particle_system.cu_type_,
            particle_system.cu_neighbour_count_,
            particle_system.cu_neighbour_index_list_,
            particle_system.cu_neighbour_distance_list_,
            particle_system.cu_mass_,
            particle_system.cu_density_,
            particle_system.cu_sum_kernel_weight_,
            particle_system.cu_sum_kernel_weighted_density_,
        )
    end
    return nothing
end

@inline function cuDensityFilterSelfaction!(
    n_particles::IntType,
    kernel::SPHKernel,
    cu_is_alive,
    cu_mass,
    cu_density,
    cu_type,
    cu_sum_kernel_weight,
    cu_sum_kernel_weighted_density,
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
            @inbounds kernel_weight = kernel.kernel_value_0_ * cu_mass[p] / cu_density[p]
            @inbounds cu_sum_kernel_weight[p] += kernel_weight
            @inbounds cu_sum_kernel_weighted_density[p] += kernel_weight * cu_density[p]
            @inbounds cu_sum_kernel_weight[p] = RealType(0)
            @inbounds cu_sum_kernel_weighted_density[p] = RealType(0)
        end
    end
    return nothing
end

@inline function densityFilterSelfaction!(particle_system::ParticleSystem, kernel::SPHKernel; numthreads = 256)::Nothing
    numblocks = ceil(Int, particle_system.n_particles_ / numthreads)
    CUDA.@sync begin
        @cuda threads = numthreads blocks = numblocks cuDensityFilterSelfaction!(
            particle_system.n_particles_,
            kernel,
            particle_system.cu_is_alive_,
            particle_system.cu_mass_,
            particle_system.cu_density_,
            particle_system.cu_type_,
            particle_system.cu_sum_kernel_weight_,
            particle_system.cu_sum_kernel_weighted_density_,
        )
    end
    return nothing
end
