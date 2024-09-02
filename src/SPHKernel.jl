#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2024/09/01 15:21:00
  @ license: MIT
  @ description:
 =#

abstract type SPHKernel{Dimension} end

const kCubicSplineRadiusRatio::RealType = 2.0
const kCubicSplineSigmaList = RealType[2.0 / 3.0, 10.0 / 7.0 / pi, 1.0 / pi]

struct CubicSpline{Dimension} <: SPHKernel{Dimension}
    h_::RealType
    h_inv_::RealType
    radius_::RealType
    sigma_::RealType
    kernel_value_0_::RealType
end

@inline function CubicSpline{Dimension}(radius::RealType) where {Dimension}
    radius_ratio = kCubicSplineRadiusRatio
    h = radius / radius_ratio
    h_inv = RealType(1.0 / h)
    sigma = kCubicSplineSigmaList[Dimension] / h^Dimension
    kernel_value_0 = sigma
    return CubicSpline{Dimension}(h, h_inv, radius, sigma, kernel_value_0)
end

@inline @fastmath function kernelValue(r::RealType, kernel::CubicSpline{Dimension})::RealType where {Dimension}
    q::RealType = r * kernel.h_inv_
    if q < 1.0
        return kernel.sigma_ * (3 * q * q * (q - 2.0) + 4.0) * 0.25
    elseif q < 2.0
        return kernel.sigma_ * (2.0 - q)^3 * 0.25
    else
        return 0.0
    end
end

@inline @fastmath function kernelGradient(r::RealType, kernel::CubicSpline{Dimension})::RealType where {Dimension}
    q::RealType = r * kernel.h_inv_
    if q < 1.0
        return kernel.sigma_ * kernel.h_inv_ * 0.75 * q * (3 * q - 4.0)
    elseif q < 2.0
        return -kernel.sigma_ * kernel.h_inv_ * 0.75 * (2.0 - q)^2
    else
        return 0.0
    end
end

const kWendlandC2RadiusRatio::RealType = 2.0
const kWendlandC2SigmaList = RealType[0.0, 7.0 / 4.0 / pi, 21.0 / 16.0 / pi]

struct WendlandC2{Dimension} <: SPHKernel{Dimension}
    h_::RealType
    h_inv_::RealType
    radius_::RealType
    sigma_::RealType
    kernel_value_0_::RealType
end

@inline function WendlandC2{Dimension}(radius::RealType) where {Dimension}
    radius_ratio = kWendlandC2RadiusRatio
    h = radius / radius_ratio
    h_inv = 1.0 / h
    sigma = kWendlandC2SigmaList[Dimension] / h^Dimension
    kernel_value_0 = sigma
    return WendlandC2{Dimension}(h, h_inv, radius, sigma, kernel_value_0)
end

@inline @fastmath function kernelValue(r::RealType, kernel::WendlandC2{Dimension})::RealType where {Dimension}
    q::RealType = r * kernel.h_inv_
    if q < 2.0
        return kernel.sigma_ * (2.0 - q)^4 * (1.0 + 2 * q) * 0.0625
    else
        return 0.0
    end
end

@inline @fastmath function kernelGradient(r::RealType, kernel::WendlandC2{Dimension})::RealType where {Dimension}
    q::RealType = r * kernel.h_inv_
    if q < 2.0
        return -kernel.sigma_ * kernel.h_inv_ * 0.625 * q * (2.0 - q)^3
    else
        return 0.0
    end
end

@inline function 𝒲(r::RealType, ker::SPHKernel{Dimension})::RealType where {Dimension}
    return kernelValue(r, ker)
end

@inline function ∇𝒲(r::RealType, ker::SPHKernel{Dimension})::RealType where {Dimension}
    return kernelGradient(r, ker)
end
