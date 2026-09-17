#include "../headers/grid.h"
#include <cmath>
#include <stdexcept>

/*
 Grid BC application for 2D Grids
 Bertalan Szuchovszky 02.03.2026

 Grid, BCType, Cell, GridBC, BoundaryCondition are defined in the grid.h header file

 validateBC: checks if the Periodic BC was set on a pair of opposing walls - if not, error message
  Input:
    GridBC
  Output:
    nothing, throws error message if periodic BC was set incorrectly

 appyBC: applies chosen BCType on a specific wall allowing for different BC at different walls
  -> Open:      what goes out the wall vanishes (Neumann BC: derivative = 0 at walls)
  -> Closed:    what collides with the wall bounces back (normal momentum flipping)
  -> Periodic:  what goes out one wall comes back on the opposing side
  -> Dirichlet: given constant state vector at walls
  Input:
    Grid & GridBC - the whole grid and the boundary conditions
  Output:
    nothing, modifies the state vector at grid walls depending on BC type
*/


void validateBC(const GridBC& bc){ //If periodic, the other wall needs to be set to periodic BC aswell
    if ((bc.left.type == BCType::Periodic) != (bc.right.type == BCType::Periodic))
        throw std::invalid_argument("Periodic BC must be applied to both left and right walls");
    if ((bc.bottom.type == BCType::Periodic) != (bc.top.type == BCType::Periodic))
        throw std::invalid_argument("Periodic BC must be applied to both bottom and top walls");
}





namespace {
 
//apply one BCType along the i (x1) direction at a single physical wall.
//"is_left" picks which wall; ghost layers run for g = 0..nghost-1.
void applyWallX(Grid& grid, const BoundaryCondition& bc,
                 size_t is, size_t ie, size_t js, size_t je, size_t nghost,
                 bool is_left){
  for (size_t j = js - nghost; j <= je + nghost; ++j) {
    for (size_t g = 0; g < nghost; ++g) {
      size_t ghost_i  = is_left ? (is - 1 - g) : (ie + 1 + g);
      switch (bc.type) {
        case BCType::Open:
          grid.copyCell(ghost_i, j, is_left ? is : ie, j);
          break;
        case BCType::Closed: {
          size_t mirror_i = is_left ? (is + g) : (ie - g);
          grid.copyCell(ghost_i, j, mirror_i, j);
          grid(ghost_i, j, 1) *= -1.0; //flip normal momentum rho*u
          break;
        }
        case BCType::Dirichlet:
          grid.setCell(ghost_i, j, bc.Q_fixed);
          break;
        case BCType::Periodic:
          throw std::logic_error(
            "applyBC: wall marked Periodic but has no Mesh neighbor -- "
            "set MeshInputs::periodic_x1/periodic_x2 so Mesh wires this "
            "edge to its neighbor instead of calling applyBC on it.");
      }
    }
  }
}
 
//same as above but along the j (x2) direction; flips the v-momentum (index 2).
void applyWallY(Grid& grid, const BoundaryCondition& bc,
                 size_t is, size_t ie, size_t js, size_t je, size_t nghost,
                 bool is_bottom){
  for (size_t i = is - nghost; i <= ie + nghost; ++i) {
    for (size_t g = 0; g < nghost; ++g) {
      size_t ghost_j = is_bottom ? (js - 1 - g) : (je + 1 + g);
      switch (bc.type) {
        case BCType::Open:
          grid.copyCell(i, ghost_j, i, is_bottom ? js : je);
          break;
        case BCType::Closed: {
          size_t mirror_j = is_bottom ? (js + g) : (je - g);
          grid.copyCell(i, ghost_j, i, mirror_j);
          grid(i, ghost_j, 2) *= -1.0; //flip normal momentum rho*v
          break;
        }
        case BCType::Dirichlet:
          grid.setCell(i, ghost_j, bc.Q_fixed);
          break;
        case BCType::Periodic:
          throw std::logic_error(
            "applyBC: wall marked Periodic but has no Mesh neighbor -- "
            "set MeshInputs::periodic_x1/periodic_x2 so Mesh wires this "
            "edge to its neighbor instead of calling applyBC on it.");
      }
    }
  }
}
 
} //namespace
 
 
void applyBC(Grid& grid, const GridBC& bc,
             size_t is, size_t ie, size_t js, size_t je, size_t nghost,
             bool bnd_left, bool bnd_right, bool bnd_bottom, bool bnd_top){
  validateBC(bc); //check if Periodic BC was requested consistently in the param file
 
  if (bnd_left)   applyWallX(grid, bc.left,   is, ie, js, je, nghost, /*is_left=*/true);
  if (bnd_right)  applyWallX(grid, bc.right,  is, ie, js, je, nghost, /*is_left=*/false);
  if (bnd_bottom) applyWallY(grid, bc.bottom, is, ie, js, je, nghost, /*is_bottom=*/true);
  if (bnd_top)    applyWallY(grid, bc.top,    is, ie, js, je, nghost, /*is_bottom=*/false);
}
