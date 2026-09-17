#pragma once
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <map>
#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <cstdint>
#include "grid.h"
#include "mesh.h"
#include <filesystem>

constexpr double PI_IO = 3.141592653589793;


/*
 Input / Output header
 Bertalan Szuchovszky 20.04.2026.

 Only active for one instance, reads a params.txt file containing initial values of particle, gas, BC-s, ...
 Hanldes file writing into a specified folder (in params.txt), creates
  -> meta.txt containing metadata (grid params & physical constants i.e gamma)
  -> grid.bin containing all state vals on every gridpoint at every timestep
  -> particle.bin containing the position, velocity,... of particle
 There are some initial conditions available 
 !!! DO NOT FORGET TO CHANGE IT FOR A SPECIFIC INITIAL CONDITION !!!

*/

//everything the simulation needs to run
struct SimParams {
  //grid
  double Nx, Ny, Nt;
  double t0, tf;
  double xmin, xmax, ymin, ymax;
  //refinement
  int nx1_block = 0, nx2_block = 0;
  std::vector<RefinementRegion> refine_regions;
  //physics
  double gamma;
  //boundary conditions
  std::string bc_left, bc_right, bc_top, bc_bottom;
  //gas initial condition
  std::string gas_profile;
  double gas_rho0, gas_p0, gas_u0, gas_v0; 
  //output
  std::string outdir;
};


//parameter file reader
inline SimParams readParams(const std::string& filename) {
  std::ifstream f(filename);
  if (!f) throw std::runtime_error("Cannot open param file: " + filename);

  std::map<std::string, std::string> kv;
  std::vector<RefinementRegion> refine_regions;
  std::string line;
  while (std::getline(f, line)) {
    auto comment = line.find('#');
    if (comment != std::string::npos) line = line.substr(0, comment);
    if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
    std::istringstream ss(line);
    std::string key;
    ss >> key;
    if (key == "refine") {
      //refine x1min x1max x2min x2max level  (one line per SMR region)
      RefinementRegion r;
      if (!(ss >> r.x1min >> r.x1max >> r.x2min >> r.x2max >> r.level))
        throw std::runtime_error("Malformed 'refine' line (expected: refine x1min x1max x2min x2max level): " + line);
      refine_regions.push_back(r);
      continue;
    }
    std::string val;
    if (ss >> val) kv[key] = val;
  }

  auto has = [&](const std::string& k) { return kv.find(k) != kv.end(); };
  auto get = [&](const std::string& k) -> std::string {
    auto it = kv.find(k);
    if (it == kv.end()) throw std::runtime_error("Missing parameter: " + k);
    return it->second;
  };
  auto getd = [&](const std::string& k) {
    try { return std::stod(get(k)); }
    catch (const std::invalid_argument&) {
      throw std::runtime_error("Parameter '" + k + "' is not a valid number");
    }
  };
  auto geti = [&](const std::string& k) {
    try { return std::stoi(get(k)); }
    catch (const std::invalid_argument&) {
      throw std::runtime_error("Parameter '" + k + "' is not a valid integer");
    }
  };

  SimParams p;

  //grid
  p.Nx   = getd("Nx");    p.Ny  = getd("Ny");   p.Nt  = getd("Nt");
  p.t0   = getd("t0");    p.tf  = getd("tf");
  p.xmin = getd("xmin");  p.xmax = getd("xmax");
  p.ymin = getd("ymin");  p.ymax = getd("ymax");

  //mesh defaults to 1 block: the whole domain
  p.nx1_block = has("nx1_block") ? geti("nx1_block") : (int)p.Nx;
  p.nx2_block = has("nx2_block") ? geti("nx2_block") : (int)p.Ny;
  p.refine_regions = std::move(refine_regions);

  //physics
  p.gamma = getd("gamma");

  //BC
  p.bc_left   = get("bc_left");
  p.bc_right  = get("bc_right");
  p.bc_top    = get("bc_top");
  p.bc_bottom = get("bc_bottom");

  //gas IC
  p.gas_profile = get("gas_profile");
  p.gas_rho0    = getd("gas_rho0");
  p.gas_p0      = getd("gas_p0");
  p.gas_u0      = getd("gas_u0");
  p.gas_v0      = getd("gas_v0");


  //output
  p.outdir = get("outdir");

  //sanity checks
  if (p.Nx <= 0 || p.Ny <= 0 || p.Nt <= 0)
    throw std::invalid_argument("Nx, Ny, Nt must be positive");
  if (p.xmax <= p.xmin || p.ymax <= p.ymin)
    throw std::invalid_argument("xmax <= xmin or ymax <= ymin");
  if (p.tf <= p.t0)
    throw std::invalid_argument("tf must be greater than t0");
  if (p.gas_rho0 <= 0 || p.gas_p0 <= 0)
    throw std::invalid_argument("gas_rho0 and gas_p0 must be positive");
  if (p.nx1_block <= 0 || p.nx2_block <= 0)
    throw std::invalid_argument("nx1_block, nx2_block must be positive");
  if ((int)p.Nx % p.nx1_block != 0 || (int)p.Ny % p.nx2_block != 0)
    throw std::invalid_argument("Nx, Ny must be divisible by nx1_block, nx2_block");
  for (const auto& r : p.refine_regions)
    if (r.level < 0)
      throw std::invalid_argument("refine level must be >= 0");
  return p;
}


//BC string -> BC type
inline BCType parseBC(const std::string& s) {
  if      (s == "Open")      return BCType::Open;
  else if (s == "Closed")    return BCType::Closed;
  else if (s == "Periodic")  return BCType::Periodic;
  else if (s == "Dirichlet") return BCType::Dirichlet;
  else throw std::invalid_argument("Unknown BC type: " + s);
}


//gas initial conditions
//fills one MeshBlock's active zone, using
//that block's own physical coordinates. Called once per block in
//mesh.blocks(); doesn't know or care if there's 1 block or 1000, or
//what level this block is at. It only ever reads block.x1v()/x2v(),
//never par.xmin/par.Nx directly (like before), so it's correct at any refinement
//level without change.
inline void init_cond(MeshBlock& block, const SimParams& par) {
  if (par.gas_profile == "uniform") {
    for (int i = block.is; i <= block.ie; i++) {
      for (int j = block.js; j <= block.je; j++) {
        Cell c;
        c[0] = par.gas_rho0;
        c[1] = par.gas_rho0 * par.gas_u0;
        c[2] = par.gas_rho0 * par.gas_v0;
        c[3] = par.gas_p0 / (par.gamma - 1.0)
             + 0.5 * par.gas_rho0 * (par.gas_u0*par.gas_u0 + par.gas_v0*par.gas_v0);
        block.grid.setCell((size_t)i, (size_t)j, c);
      }
    }
  } else if (par.gas_profile == "Sod_x"){
    //Sod shock tube vals:
    const double rho_L = par.gas_rho0,     p_L = par.gas_p0,     u_L = 0.0, v_L = 0.0;
    const double rho_R = par.gas_rho0/8.0, p_R = 0.1*par.gas_p0, u_R = 0.0, v_R = 0.0;
    const double x_mid = 0.5 * (par.xmin + par.xmax);
 
    for (int i = block.is; i <= block.ie; i++) {
      double x = block.x1v(i); //cell-center x coordinate, this block's own frame
 
      //select Left or Right state based on x location
      double rho = (x < x_mid) ? rho_L : rho_R;
      double p   = (x < x_mid) ? p_L   : p_R;
      double u   = (x < x_mid) ? u_L   : u_R;
      double v   = (x < x_mid) ? v_L   : v_R;
 
      //compute total energy per unit volume: E = p/(gamma-1) + 0.5*rho*(u^2 + v^2)
      double E = p / (par.gamma - 1.0) + 0.5 * rho * (u * u + v * v);
 
      for (int j = block.js; j <= block.je; j++) {
        Cell c;
        c[0] = rho;
        c[1] = rho * u;
        c[2] = rho * v;
        c[3] = E;
        block.grid.setCell((size_t)i, (size_t)j, c);
      }
    } 
  } else {
    throw std::runtime_error("Unknown gas_profile: '" + par.gas_profile + "'. Available: uniform, Sod_x");
  }
}



//output directory
inline void setupOutputDir(const std::string& dir) {
  std::error_code ec;
  // create_directories creates the full path and doesn't error if it exists
  std::filesystem::create_directories(dir, ec); 
  
  if (ec) {
    throw std::runtime_error("Could not create output directory: " + dir + " - " + ec.message());
  }
  std::cout << "Output directory: " << dir << "\n";
}


//metadata file txt containing some important values of the simulation
inline void writeMetadata(const std::string& outdir, const SimParams& par) {
  std::string path = outdir + "/meta.txt";
  std::ofstream f(path);
  if (!f) throw std::runtime_error("Cannot open meta.txt: " + path);
  f << std::setprecision(17) << std::scientific;
  f << "xmin "  << par.xmin  << "\n"
    << "xmax "  << par.xmax  << "\n"
    << "ymin "  << par.ymin  << "\n"
    << "ymax "  << par.ymax  << "\n"
    << "gamma " << par.gamma << "\n"
    << "Nx "    << (int)par.Nx    << "\n"
    << "Ny "    << (int)par.Ny    << "\n"
    << "nx1_block " << par.nx1_block << "\n"
    << "nx2_block " << par.nx2_block << "\n"
    << "cs_iso " << par.gas_p0/par.gas_rho0 <<"\n";
}


//grid output, single binary file, all frames sequential suggested by Claude
//all frames sequential. Block layout is static (SMR only, no AMR yet)
//so the header is written exactly once, before the time loop -- if/when
//AMR lands, blocks can change between frames and this format will need
//a per-frame layout section too (flagged for that future work, not
//needed here).
//
//Header layout:
//  uint64 nblocks
//  per block, in mesh.blocks() (gid) order:
//    int32  level
//    int64  lx1, lx2
//    double x1min, x1max, x2min, x2max
//    uint64 nx1, nx2      (active cell counts)
//Then per frame:
//  double t
//  per block, in the same gid order:
//    nx1*nx2 cells, each NVAR doubles, row-major (i then j) over the
//    block's own active zone -- i.e. exactly grid.cell(i,j) for
//    i in [is,ie], j in [js,je].
inline std::ofstream openGridFile(const std::string& outdir, const Mesh& mesh) {
  std::string path = outdir + "/grid.bin";
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("Cannot open grid output file: " + path);
 
  uint64_t nblocks = mesh.blocks().size();
  f.write(reinterpret_cast<const char*>(&nblocks), sizeof(uint64_t));
  for (const auto& bptr : mesh.blocks()) {
    const MeshBlock& b = *bptr;
    int32_t level = b.loc.level;
    int64_t lx1 = b.loc.lx1, lx2 = b.loc.lx2;
    uint64_t nx1 = (uint64_t)b.block_size.nx1, nx2 = (uint64_t)b.block_size.nx2;
    f.write(reinterpret_cast<const char*>(&level), sizeof(int32_t));
    f.write(reinterpret_cast<const char*>(&lx1), sizeof(int64_t));
    f.write(reinterpret_cast<const char*>(&lx2), sizeof(int64_t));
    f.write(reinterpret_cast<const char*>(&b.block_size.x1min), sizeof(double));
    f.write(reinterpret_cast<const char*>(&b.block_size.x1max), sizeof(double));
    f.write(reinterpret_cast<const char*>(&b.block_size.x2min), sizeof(double));
    f.write(reinterpret_cast<const char*>(&b.block_size.x2max), sizeof(double));
    f.write(reinterpret_cast<const char*>(&nx1), sizeof(uint64_t));
    f.write(reinterpret_cast<const char*>(&nx2), sizeof(uint64_t));
  }
  return f;
}
 
inline void writeFrame(std::ofstream& f, const Mesh& mesh, double t) {
  f.write(reinterpret_cast<const char*>(&t), sizeof(double));
  for (const auto& bptr : mesh.blocks()) {
    const MeshBlock& b = *bptr;
    for (int i = b.is; i <= b.ie; i++)
      for (int j = b.js; j <= b.je; j++)
        f.write(reinterpret_cast<const char*>(b.grid.cell((size_t)i, (size_t)j)), 4 * sizeof(double));
  }
}
