#pragma once

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <cmath>

/*
 * This guy handles everything for Kokkos
 * */

using Real = double; //allows for precision switching

static constexpr std::size_t NVAR = 4; // [rho, rho*u, rho*v, E]

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace    = ExecutionSpace::memory_space;

using View1D = Kokkos::View<Real*, Kokkos::LayoutRight, MemorySpace>;
using View2D = Kokkos::View<Real**, Kokkos::LayoutRight, MemorySpace>;
using View3D = Kokkos::View<Real***, Kokkos::LayoutRight, MemorySpace>;
using View4D = Kokkos::View<Real****, Kokkos::LayoutRight, MemorySpace>;

using IntView2D = Kokkos::View<int**, Kokkos::LayoutRight, MemorySpace>;

using HostView4D    = typename View4D::host_mirror_type;
using HostView2D    = typename View2D::host_mirror_type;
using HostIntView2D = typename IntView2D::host_mirror_type;

struct Vec4 {
  Real data[NVAR];
  KOKKOS_INLINE_FUNCTION Real operator[](size_t k) const { return data[k]; }
  KOKKOS_INLINE_FUNCTION Real& operator[](size_t k) { return data[k]; }
};


enum class BCType{Open, Closed, Periodic, Dirichlet};

struct BoundaryCondition{
  BCType type;
  Vec4 Q_fixed; //Dirichlet boundary condition value
};

struct GridBC{
  BoundaryCondition left, right, top, bottom;
};
