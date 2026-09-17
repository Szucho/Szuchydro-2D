#include <cmath>
#include <iomanip>
#include <ios>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include "../headers/matrix.h"
#include "../headers/grid.h"
#include "../headers/HLLC.h"
#include "../headers/slopelim.h"
#include "../headers/io.h"
#include "../headers/mesh.h"

using namespace VecOps;

/*
 2D Hydrodynamics HLLC Euler Equation solver
 Bertalan Szuchovszky 03.03.2026

 Solves the 2D compressible Euler equations in conservative form
 on a uniform Cartesian grid using the HLLC approximate Riemann solver (see HLLC.cpp)
 with Superbee slope-limited interpolation at cell walls (see slope_limiters.cpp).

 Equation to be solved: dQ/dt + dF/dQ * div Q = 0 with dF/dQ = J Jacobi matrix 
 State vector: Q = [rho, rho*u, rho*v, rho*e_tot]
               e_tot = e_th + 0.5*(u^2+v^2),  e_th = 1/(gamma-1) * k_BT/(mu*m_p) - ideal gas law

 Spatial discretization: 2nd order finite volume (~MUSCL-Hancock, approximate Riemann solver)
 Time integration: Explicit Euler
 => 2nd order FV in space & 1st order in time
 Boundary conditions: Open, Closed, Periodic, Dirichlet (user specified)

 Usage: set init_cond() for your problem (DON'T FORGET THIS), then build and run.
 Build: clang++ -std=c++17 -O3 -march=native -I./headers sources/*.cpp -o builds/szuchydro2d.exe

 No CMake yet as I can't be bothered to write one
 Validated on Sod shock tube (Toro, Chapter 4)



 MODIFIED 08.07.2026 (Mesh/MeshBlock rewrite, see mesh.h)
 The domain used to be exactly one Grid, and this file both owned it and
 knew everything about its physical layout. It's now a Mesh made of one
 or more MeshBlocks (quadtree-addressed, Athena++-style); every routine
 below loops over mesh.blocks() and works on one MeshBlock's local Grid
 + its own index range at a time. cfl_dt/timestep/HLLC_step are
 otherwise the exact same physics as before, they just get handed a
 block instead of "the" grid. The per-block loop below is the only
 place that knows there's more than one block, which is the point:
 later, dispatching each loop iteration to its own device/rank (GPU here) 
 is a change local to that loop, not to the physics.
 
 This build still runs as a single MeshBlock spanning the whole domain
 by default (see nx1_block/nx2_block just below main()'s BC setup)
 numerically identical to the old single-Grid version. To actually
 exercise domain decomposition or static refinement, shrink those and/or
 add entries to mesh_in.refine_regions. Reading multi-block/SMR settings
 from params.txt is a natural next step (touches io.h/SimParams, which
 this pass intentionally left alone) rather than done here.
*/


//FIRST: CFL condition -> need to check if dt > dt_{cfl}
//evaluated per MeshBlock (using that block's own dx1/dx2) and reduced to the global
//min, since a finer SMR block needs a smaller dt than a coarser one.
double cfl_dt(Mesh& mesh, double gamma, double CFL_max=0.5) {
  double dt_min = std::numeric_limits<double>::infinity();
  for (auto& bptr : mesh.blocks()) {
    MeshBlock& b = *bptr;
    double smax_x = 0.0, smax_y = 0.0; //s characteristics
    for (int i = b.is; i <= b.ie; ++i) {
      for (int j = b.js; j <= b.je; ++j) {
        Cell c = b.grid.getCell((size_t)i, (size_t)j); //c contains [rho, rho u, rho v, rho e_tot]
        double rho = c[0];
        double u   = c[1]/rho;
        double v   = c[2]/rho;
        double p   = (gamma-1.0)*(c[3] - 0.5*rho*(u*u+v*v));
        double cs  = std::sqrt(gamma*p/rho); //adiabatic soundspeed
        smax_x = std::max(smax_x, std::abs(u)+cs);
        smax_y = std::max(smax_y, std::abs(v)+cs);
      }
    }
    double dt_b = CFL_max / (smax_x/b.dx1() + smax_y/b.dx2());
    dt_min = std::min(dt_min, dt_b);
  }
  return dt_min;
}

//HLLC step of a single grid point Q_{ij}
Vector HLLC_step(Vector& Qi, 
                 const Vector& FluxXiph, 
                 const Vector& FluxYiph, 
                 const Vector& FluxXimh, 
                 const Vector& FluxYimh,
                 double dx, double dy, double dt){
  Vector Qi_new; //return the new state vector at ij gridpoint
  Qi_new = Qi - dt/dx*(FluxXiph - FluxXimh) - dt/dy*(FluxYiph-FluxYimh); //Euler timestep
  return Qi_new;
}




void timestep_block(MeshBlock& b, double dt, double gamma){
  //advance a single MeshBlock by dt in place. Same HLLC/slope-limiter
  //logic as before, just addressed via the block's own is/ie/js/je and
  //dx1()/dx2() instead of assuming its Grid is the whole domain.
  Grid grid_new(b.grid.rows(), b.grid.cols());

  for (size_t i = 0; i < b.grid.rows(); ++i)
    for (size_t j = 0; j < b.grid.cols(); ++j)
      grid_new.setCell(i, j, b.grid.getCell(i, j));
 
  double dx = b.dx1(), dy = b.dx2();

  for (int i = b.is; i<=b.ie; i++){
    for(int j = b.js; j<=b.je; j++){
      Vector Qi   = CellToVec(b.grid.getCell(i,j)); //bunch of Q_{ij} vals needed for the slope calculations 
      Vector Qim1 = CellToVec(b.grid.getCell(i-1,j));
      Vector Qim2 = CellToVec(b.grid.getCell(i-2,j));
      Vector Qip1 = CellToVec(b.grid.getCell(i+1,j));
      Vector Qip2 = CellToVec(b.grid.getCell(i+2,j));
      Vector Qjm1 = CellToVec(b.grid.getCell(i,j-1));
      Vector Qjm2 = CellToVec(b.grid.getCell(i,j-2));
      Vector Qjp1 = CellToVec(b.grid.getCell(i,j+1));
      Vector Qjp2 = CellToVec(b.grid.getCell(i,j+2));

      //slopes in x direction
      Vector sigma_im1_x, sigma_i_x, sigma_ip1_x;
      sigma_im1_x = sigma_minmod(Qim2, Qim1, Qi,   dx);
      sigma_i_x   = sigma_minmod(Qim1, Qi,   Qip1, dx);
      sigma_ip1_x = sigma_minmod(Qi,   Qip1, Qip2, dx);

      //slopes in the y direction
      Vector sigma_jm1_y, sigma_j_y, sigma_jp1_y;
      sigma_jm1_y = sigma_minmod(Qjm2, Qjm1, Qi,   dy);
      sigma_j_y   = sigma_minmod(Qjm1, Qi,   Qjp1, dy);
      sigma_jp1_y = sigma_minmod(Qi,   Qjp1, Qjp2, dy);
      
      //x interfaces
      Vector QL_imh = Qim1 + 0.5*dx * sigma_im1_x;  //left  state at i-1/2
      Vector QR_imh = Qi   - 0.5*dx * sigma_i_x;    //right state at i-1/2
      Vector QL_iph = Qi   + 0.5*dx * sigma_i_x;    //left  state at i+1/2
      Vector QR_iph = Qip1 - 0.5*dx * sigma_ip1_x;  //right state at i+1/2

      //y interfaces
      Vector QL_jmh = Qjm1 + 0.5*dy * sigma_jm1_y;  //left  state at j-1/2
      Vector QR_jmh = Qi   - 0.5*dy * sigma_j_y;    //right state at j-1/2
      Vector QL_jph = Qi   + 0.5*dy * sigma_j_y;    //left  state at j+1/2
      Vector QR_jph = Qjp1 - 0.5*dy * sigma_jp1_y;  //right state at j+1/2

      //fluxes -> HLLC method (see HLLC.cpp, Toro)
      Vector FXimh = FluxhllcX(QL_imh, QR_imh, gamma);
      Vector FXiph = FluxhllcX(QL_iph, QR_iph, gamma);
      Vector FYjmh = FluxhllcY(QL_jmh, QR_jmh, gamma);
      Vector FYjph = FluxhllcY(QL_jph, QR_jph, gamma); 


      //cache this block's own flux at whichever of its 4 outer faces this cell happens to sit on
      //interior faces don't need it (they telescope exactly regardless of accuracy)
      //only these border values are ever read, by ApplyReflux (mesh.cpp) after
      //every block has finished its own update this step.
      if (i == b.is){
        double* d = b.faceFlux((int)FaceDir::XM, j - b.js); for (int k = 0; k < 4; ++k) d[k] = FXimh[k];
      }
      if (i == b.ie){
        double* d = b.faceFlux((int)FaceDir::XP, j - b.js); for (int k = 0; k < 4; ++k) d[k] = FXiph[k];
      }
      if (j == b.js){
        double* d = b.faceFlux((int)FaceDir::YM, i - b.is); for (int k = 0; k < 4; ++k) d[k] = FYjmh[k];
      }
      if (j == b.je){
        double* d = b.faceFlux((int)FaceDir::YP, i - b.is); for (int k = 0; k < 4; ++k) d[k] = FYjph[k];
      }
       
      //apply HLLC timestep at gridcell Q_{ij}
      Vector Q_new = HLLC_step(Qi, FXiph, FYjph, FXimh, FYjmh, dx, dy, dt);
      grid_new.setCell(i, j, VecToCell(Q_new)); //convert Q_new to Cell and then append it to the new grid
    }
  }
  b.grid = std::move(grid_new); //replace old block data with the new timestep
}


//advance every MeshBlock in the Mesh by dt. This loop, and only this
//loop, knows there's more than one block; per-block work stays a
//self-contained call, ready to be parallelized/offloaded per block.
void timestep(Mesh& mesh, double dt, double gamma) {
  for (auto& bptr : mesh.blocks()) {
    timestep_block(*bptr, dt, gamma);
  }
}



int main(int argc, char*argv[]){
  using std::chrono::high_resolution_clock;
  using std::chrono::duration_cast;
  using std::chrono::duration;
  using std::chrono::milliseconds;
  auto t1 = high_resolution_clock::now();
 
  std::string paramfile = (argc > 1) ? argv[1] : "params.txt";
  SimParams par = readParams(paramfile);
  double cs2 = par.gas_p0/par.gas_rho0;
 
  double Nx = par.Nx, Ny = par.Ny;
  double Nt = par.Nt;
  double t0 = par.t0, tf = par.tf;
  double xmin = par.xmin, xmax = par.xmax;
  double ymin = par.ymin, ymax = par.ymax;
  double gamma = par.gamma;
 
  double dt = (tf-t0)/Nt;
 
  GridBC bc;
  bc.left.type   = parseBC(par.bc_left);
  bc.right.type  = parseBC(par.bc_right);
  bc.top.type    = parseBC(par.bc_top);
  bc.bottom.type = parseBC(par.bc_bottom);
  validateBC(bc);
 
  MeshInputs mesh_in;
  mesh_in.domain.x1min = xmin;
  mesh_in.domain.x1max = xmax;
  mesh_in.domain.x2min = ymin;
  mesh_in.domain.x2max = ymax;
  mesh_in.domain.nx1 = (int)Nx;
  mesh_in.domain.nx2 = (int)Ny;

  //block size and static refinement now come straight from params.txt
  //nx1_block/nx2_block default to the whole domain 
  //one block when not given, see io.h::readParams
  mesh_in.nx1_block = par.nx1_block;
  mesh_in.nx2_block = par.nx2_block;
  mesh_in.nghost = 2;
  mesh_in.periodic_x1 = (bc.left.type == BCType::Periodic);
  mesh_in.periodic_x2 = (bc.bottom.type == BCType::Periodic);
  mesh_in.refine_regions = par.refine_regions; //zero or more `refine` lines from params.txt
 
  Mesh mesh(mesh_in);
  mesh.BuildTree();
  mesh.CreateMeshBlocks();
  mesh.SetupNeighbors();
 
  for (auto& bptr : mesh.blocks()) {
    init_cond(*bptr, par);
  }
  ExchangeGhostZones(mesh);
  ApplyPhysicalBoundaryConditions(mesh, bc);
 
  double cfl = cfl_dt(mesh, gamma);
  if (dt > cfl){
    std::cout << "Warning: dt=" << dt << " exceeds CFL limit=" << cfl << "\n";
    std::cout << "Continue with CFL timestep? (y/n): ";
    std::string ans;
    std::cin >> ans;
    if (ans == "y") {
        dt = cfl;
        Nt = (int)std::ceil((tf - t0) / dt);
        std::cout << "Using dt=" << dt << ", Nt=" << Nt << "\n";
    } else {
        std::cout << "Enter new Nt: ";
        std::cin >> Nt;
        dt = (tf - t0) / Nt;
        if (dt > cfl)
            std::cout << "Still above CFL (" << cfl << "), negative density/pressure/energy is expected...\n";
    }
  }
 
  setupOutputDir(par.outdir);
  writeMetadata(par.outdir, par);
  auto gridfile = openGridFile(par.outdir, mesh);
  double t = t0;
 

  //progress is driven by t/tf, not step count!!! with adaptive per-step
  //CFL timestepping (dt shrinks as the flow accelerates), the real
  //number of steps needed isn't known ahead of time and can drift well
  //past any initial estimate: t/tf is always well-defined in [0,1],
  //a step-count-based fraction is not
  double dt_save = (tf - t0) / 100.0;
  double t_next_save = t0;
 
  int n=0;
  const int barwidth = 40;
  int lastpercentshown = -1;
  auto draw_progress = [&](double t_now, bool force) {
    double progress = std::min(1.0, t_now / tf);
    int percent = (int)(progress * 100.0);
    if (!force && percent == lastpercentshown) return;
    lastpercentshown = percent;
    int filled = (int)(progress * barwidth);
    std::cout << "\r[" << std::string(filled, '#') << std::string(barwidth - filled, '-') << "] "
              << std::setw(3) << percent << "%  t=" << std::fixed << std::setprecision(3) << t_now
              << "  step " << n << std::flush;
  };

  try{
    while (t<tf){
      double dt_cfl = cfl_dt(mesh, gamma, 0.4);
      dt = std::min(dt_cfl, tf - t);
 
      draw_progress(t, n==0);
       
      if (t >= t_next_save) {
        writeFrame(gridfile, mesh, t);
        gridfile.flush();
        t_next_save += dt_save;
      }
 
      timestep(mesh, dt, gamma);
      ApplyReflux(mesh, dt);
      ExchangeGhostZones(mesh);
      ApplyPhysicalBoundaryConditions(mesh, bc);
      n += 1;
      t += dt;
    }
 
    writeFrame(gridfile, mesh, t);
    gridfile.flush();
    draw_progress(tf, true); //final draw
    std::cout << "Simulation finished successfully at t = " << t << std::endl;
 
  } catch (const std::invalid_argument& e){
    std::cerr << "[CRASH] at step: "<< n << ", t = " << t << std::endl;
    throw;
  }
 
  auto t2 = high_resolution_clock::now();
  auto ms_int = duration_cast<milliseconds>(t2 - t1);
  duration<double, std::milli> ms_double = t2 - t1;
  std::cout << ms_int.count() << "ms\n";
  std::cout << ms_double.count() << "ms\n";
  return 0;
}
