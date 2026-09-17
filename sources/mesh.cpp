#include "../headers/mesh.h"
#include <stdexcept>
#include <algorithm>
#include <map>

/*
 Mesh implementation
 a flat physics-agnostic state array. Everything here is topology only
 domain decomposition, quadtree constructions for SMR, and
 block-to-block neighbor wiring. No NVAR, no Euler equations, no HLLC.
*/

Mesh::Mesh(const MeshInputs& in) : in_(in) {
  if (in_.nx1_block <= 0 || in_.nx2_block <= 0)
    throw std::invalid_argument("Mesh: block size must be positive");
  if (in_.domain.nx1 % in_.nx1_block != 0 || in_.domain.nx2 % in_.nx2_block != 0)
    throw std::invalid_argument("Mesh: root grid cell counts must be divisible by MeshBlock size");
  if (!in_.refine_regions.empty() && (in_.nx1_block % 2 != 0 || in_.nx2_block % 2 != 0))
    throw std::invalid_argument("Mesh: nx1_block and nx2_block must both be even when refine_regions is non-empty "
                                "(2:1 refinement requires each coarse cell to map to exactly 2 fine cells)");
  nx1_blocks_root_ = in_.domain.nx1 / in_.nx1_block;
  nx2_blocks_root_ = in_.domain.nx2 / in_.nx2_block;
}


RegionSize Mesh::BlockSizeFromLocation(const LogicalLocation& loc) const {
  double dom_w1 = in_.domain.x1max - in_.domain.x1min;
  double dom_w2 = in_.domain.x2max - in_.domain.x2min;
  std::int64_t n1 = static_cast<std::int64_t>(nx1_blocks_root_) << loc.level;
  std::int64_t n2 = static_cast<std::int64_t>(nx2_blocks_root_) << loc.level;

  RegionSize sz;
  sz.x1min = in_.domain.x1min + dom_w1 * (double)loc.lx1     / (double)n1;
  sz.x1max = in_.domain.x1min + dom_w1 * (double)(loc.lx1+1) / (double)n1;
  sz.x2min = in_.domain.x2min + dom_w2 * (double)loc.lx2     / (double)n2;
  sz.x2max = in_.domain.x2min + dom_w2 * (double)(loc.lx2+1) / (double)n2;
  sz.nx1 = in_.nx1_block;
  sz.nx2 = in_.nx2_block;
  return sz;
}


void Mesh::BuildTree() {
  roots_.clear();
  roots_.reserve((size_t)nx1_blocks_root_ * (size_t)nx2_blocks_root_);
  for (int j = 0; j < nx2_blocks_root_; ++j) {
    for (int i = 0; i < nx1_blocks_root_; ++i) {
      LogicalLocation loc{0, i, j};
      roots_.push_back(std::make_unique<MeshBlockTree>(this, nullptr, loc));
    }
  }
  //AMR
  //apply every SMR request; this is the entire static-refinement step
  //once done here the tree is frozen (AMR would instead call this, or
  //Refine()/Derefine() directly, mid-run and rebuild the block list but that waits to be implemented)
  for (const auto& reg : in_.refine_regions) {
    for (auto& root : roots_) {
      root->RefineRegion(in_.domain, nx1_blocks_root_, nx2_blocks_root_,
                          reg.x1min, reg.x1max, reg.x2min, reg.x2max, reg.level);
    }
  }
}


void Mesh::CreateMeshBlocks() {
  std::vector<LogicalLocation> leaves;
  for (auto& root : roots_) root->CollectLeaves(leaves);

  blocks_.clear();
  blocks_.reserve(leaves.size());
  int gid = 0;
  for (const auto& loc : leaves) {
    RegionSize sz = BlockSizeFromLocation(loc);
    blocks_.push_back(std::make_unique<MeshBlock>(gid, loc, sz, in_.nghost));
    ++gid;
  }
}


namespace {
//inclusive index range of a block's footprint, expressed in units of
//the finest of the two levels being compared. Lets us test adjacency
//between blocks at different refinement levels with plain integers
struct IndexRange { std::int64_t lo, hi; };

IndexRange RangeAtLevel(std::int64_t lx, int level, int common_level) {
  std::int64_t scale = 1LL << (common_level - level); //common_level >= level always, by construction below
  return { lx * scale, (lx + 1) * scale - 1 };
}

bool Overlap(const IndexRange& a, const IndexRange& b) {
  return a.lo <= b.hi && b.lo <= a.hi;
}
} //namespace


void Mesh::SetupNeighbors() {
  //AMR? Still part of the tree which should be O(N ln N)
  //map every leaf location to its block, for quick existence checks
  //not strictly needed by the O(N^2) adjacency scan below, but this
  //is where interval structure will be needed I guess once block counts get large
  //FLAGGED here rather than in the scan itself
  for (auto& bptr : blocks_) {
    MeshBlock& b = *bptr;
    for (int f = 0; f < 4; ++f) {
      b.is_physical_boundary[f] = true; //assume boundary until a neighbor is found
      b.neighbors[f].clear();
    }
  }

  for (auto& aptr : blocks_) {
    MeshBlock& a = *aptr;
    for (int fi = 0; fi < 4; ++fi) {
      FaceDir face = static_cast<FaceDir>(fi);
      for (auto& bptr : blocks_) {
        MeshBlock& b = *bptr;
        //a block can be its own periodic neighbor (for example a single block
        //spanning the whole domain on a periodic axis), so only skip
        //self-comparison for the non-wrapped test, not the wrapped one
        bool same_block = (&a == &b);

        int lc = std::max(a.loc.level, b.loc.level);
        IndexRange a1 = RangeAtLevel(a.loc.lx1, a.loc.level, lc);
        IndexRange a2 = RangeAtLevel(a.loc.lx2, a.loc.level, lc);

        //try both the direct position and, if that axis is periodic
        //and this face sits on the domain edge, the wrapped position
        //Periodic BC => block connectivity
        std::int64_t domain_w1 = static_cast<std::int64_t>(nx1_blocks_root_) << lc;
        std::int64_t domain_w2 = static_cast<std::int64_t>(nx2_blocks_root_) << lc;

        for (int wrap = 0; wrap < 2; ++wrap) {
          if (wrap == 0 && same_block) continue; //trivial self-adjacency is meaningless
          if (wrap == 1 && !((face == FaceDir::XM || face == FaceDir::XP) ? in_.periodic_x1 : in_.periodic_x2))
            continue;

          IndexRange b1 = RangeAtLevel(b.loc.lx1, b.loc.level, lc);
          IndexRange b2 = RangeAtLevel(b.loc.lx2, b.loc.level, lc);
          if (wrap == 1) {
            if (face == FaceDir::XM || face == FaceDir::XP) { b1.lo -= domain_w1; b1.hi -= domain_w1; }
            else                                            { b2.lo -= domain_w2; b2.hi -= domain_w2; }
            //also try the other wrap direction (neighbor could be on either edge)
          }

          bool adjacent = false;
          switch (face) {
            case FaceDir::XP: adjacent = (b1.lo == a1.hi + 1) && Overlap(a2, b2); break;
            case FaceDir::XM: adjacent = (a1.lo == b1.hi + 1) && Overlap(a2, b2); break;
            case FaceDir::YP: adjacent = (b2.lo == a2.hi + 1) && Overlap(a1, b1); break;
            case FaceDir::YM: adjacent = (a2.lo == b2.hi + 1) && Overlap(a1, b1); break;
          }
          //also test the opposite wrap shift (b shifted the other way)
          if (!adjacent && wrap == 1) {
            IndexRange b1b = RangeAtLevel(b.loc.lx1, b.loc.level, lc);
            IndexRange b2b = RangeAtLevel(b.loc.lx2, b.loc.level, lc);
            if (face == FaceDir::XM || face == FaceDir::XP) { b1b.lo += domain_w1; b1b.hi += domain_w1; }
            else                                            { b2b.lo += domain_w2; b2b.hi += domain_w2; }
            switch (face) {
              case FaceDir::XP: adjacent = (b1b.lo == a1.hi + 1) && Overlap(a2, b2b); break;
              case FaceDir::XM: adjacent = (a1.lo == b1b.hi + 1) && Overlap(a2, b2b); break;
              case FaceDir::YP: adjacent = (b2b.lo == a2.hi + 1) && Overlap(a1, b1b); break;
              case FaceDir::YM: adjacent = (a2.lo == b2b.hi + 1) && Overlap(a1, b1b); break;
            }
          }

          if (adjacent) {
            NeighborLevel lvl = (b.loc.level == a.loc.level) ? NeighborLevel::Same
                               : (b.loc.level <  a.loc.level) ? NeighborLevel::Coarser
                                                               : NeighborLevel::Finer;
            a.neighbors[fi].push_back(NeighborBlock{b.gid, lvl, wrap == 1});
            a.is_physical_boundary[fi] = false;
          }
        }
      }
    }
  }
}


//MeshBlock
MeshBlock::MeshBlock(int gid_, LogicalLocation loc_, RegionSize size_, int nghost_)
  : gid(gid_), loc(loc_), block_size(size_), nghost(nghost_),
    grid(size_.nx1 + 2 * nghost_, size_.nx2 + 2 * nghost_) {
  is = nghost;              ie = nghost + block_size.nx1 - 1;
  js = nghost;              je = nghost + block_size.nx2 - 1;
  for (int f = 0; f < 4; f++) is_physical_boundary[f] = true;
  for (int f = 0; f < 4; f++) flux_face[f].assign((size_t)tangentCount(f)*NVAR, 0.0);
}

double MeshBlock::x1v(int i) const {
  return block_size.x1min + (i - is + 0.5) * dx1();
}
double MeshBlock::x2v(int j) const {
  return block_size.x2min + (j - js + 0.5) * dx2();
}


//physical-domain BC dispatch (applyBC itself lives in grid_setup.cpp)
void ApplyPhysicalBoundaryConditions(Mesh& mesh, const GridBC& bc) {
  for (auto& bptr : mesh.blocks()) {
    MeshBlock& b = *bptr;
    applyBC(b.grid, bc,
            (size_t)b.is, (size_t)b.ie, (size_t)b.js, (size_t)b.je, (size_t)b.nghost,
            b.is_physical_boundary[(int)FaceDir::XM],
            b.is_physical_boundary[(int)FaceDir::XP],
            b.is_physical_boundary[(int)FaceDir::YM],
            b.is_physical_boundary[(int)FaceDir::YP]);
  }
}


//ghost exchange
void ExchangeGhostZones(Mesh& mesh) {
  auto& blocks = mesh.blocks();

  auto copyState = [](Grid& dst, size_t di, size_t dj, const Grid& src, size_t si, size_t sj) {
    const double* s = src.cell(si, sj);
    double* d = dst.cell(di, dj);
    for (size_t k = 0; k < dst.nvar(); ++k) d[k] = s[k];
  };
  auto avgState = [](Grid& dst, size_t di, size_t dj, const Grid& src,
                      size_t si1, size_t sj1, size_t si2, size_t sj2) {
    const double* s1 = src.cell(si1, sj1);
    const double* s2 = src.cell(si2, sj2);
    double* d = dst.cell(di, dj);
    for (size_t k = 0; k < dst.nvar(); ++k) d[k] = 0.5 * (s1[k] + s2[k]);
  };

  for (auto& aptr : blocks) {
    MeshBlock& a = *aptr;
    size_t nghost = (size_t)a.nghost;

    for (int fi = 0; fi < 4; ++fi) {
      if (a.is_physical_boundary[fi]) continue; //filled by ApplyPhysicalBoundaryConditions instead
      FaceDir face = static_cast<FaceDir>(fi);
      bool xface = (face == FaceDir::XM || face == FaceDir::XP);
      int N = xface ? a.block_size.nx2 : a.block_size.nx1; //tangential cell count (same for every block)

      for (const auto& nb : a.neighbors[fi]) {
        MeshBlock& b = *blocks[nb.gid];

        if (nb.level == NeighborLevel::Same) {
          //exact ghost copy
          for (int t = 0; t < N; ++t) {
            for (size_t g = 0; g < nghost; ++g) {
              if (face == FaceDir::XP)      copyState(a.grid, a.ie+1+g, a.js+t, b.grid, b.is+g, b.js+t);
              else if (face == FaceDir::XM) copyState(a.grid, a.is-1-g, a.js+t, b.grid, b.ie-g, b.js+t);
              else if (face == FaceDir::YP) copyState(a.grid, a.is+t, a.je+1+g, b.grid, b.is+t, b.js+g);
              else                          copyState(a.grid, a.is+t, a.js-1-g, b.grid, b.is+t, b.je-g);
            }
          }
        } else if (nb.level == NeighborLevel::Coarser) {
          //AMR: order-0 prolongation, the same coarse cell value
          //is duplicated into both overlapping fine cells and repeated
          //across every ghost layer. Enough to exercise the mesh logic
          //(SMR runs, gets sane ghost data at level jumps) but NOT
          //flux-conservative. Needs to be replaced once the AMR is done.
          int half = xface ? (int)(a.loc.lx2 & 1) : (int)(a.loc.lx1 & 1);
          for (int t = 0; t < N; ++t) {
            int tk = half * (N / 2) + t / 2; //b's local tangential index covering fine index t
            for (size_t g = 0; g < nghost; ++g) {
              if (face == FaceDir::XP)      copyState(a.grid, a.ie+1+g, a.js+t, b.grid, b.is+g, b.js+tk);
              else if (face == FaceDir::XM) copyState(a.grid, a.is-1-g, a.js+t, b.grid, b.ie-g, b.js+tk);
              else if (face == FaceDir::YP) copyState(a.grid, a.is+t, a.je+1+g, b.grid, b.is+tk, b.js+g);
              else                          copyState(a.grid, a.is+t, a.js-1-g, b.grid, b.is+tk, b.je-g);
            }
          }
        } else { //Finer
          //AMR: 2:1 averaging restriction, same caveat as above.
          int half = xface ? (int)(b.loc.lx2 & 1) : (int)(b.loc.lx1 & 1);
          for (int tb = 0; tb < N; tb += 2) {
            int tk = half * (N / 2) + tb / 2; //a's local tangential index this fine pair fills
            for (size_t g = 0; g < nghost; ++g) {
              if (face == FaceDir::XP)
                avgState(a.grid, a.ie+1+g, a.js+tk, b.grid, b.is+g, b.js+tb, b.is+g, b.js+tb+1);
              else if (face == FaceDir::XM)
                avgState(a.grid, a.is-1-g, a.js+tk, b.grid, b.ie-g, b.js+tb, b.ie-g, b.js+tb+1);
              else if (face == FaceDir::YP)
                avgState(a.grid, a.is+tk, a.je+1+g, b.grid, b.is+tb, b.js+g, b.is+tb+1, b.js+g);
              else
                avgState(a.grid, a.is+tk, a.js-1-g, b.grid, b.is+tb, b.je-g, b.is+tb+1, b.je-g);
            }
          }
        }
      }
    }
  }
}


//reflux: flux conservation correction at coarse - fine interface
void ApplyReflux(Mesh& mesh, double dt) {
  auto& blocks = mesh.blocks();
 
  for (auto& aptr : blocks) {
    MeshBlock& a = *aptr;
 
    for (int fi = 0; fi < 4; ++fi) {
      if (a.is_physical_boundary[fi]) continue; //no correction at a real domain edge
      FaceDir face = static_cast<FaceDir>(fi);
      bool xface = (face == FaceDir::XM || face == FaceDir::XP);
      int N = a.tangentCount(fi); //a's own (coarse) resolution along this face
      double d = xface ? a.dx1() : a.dx2();
      //max faces (XP,YP) correct with (F_used - F_eff); min faces (XM,YM)
      //with (F_eff - F_used): opposite sign, matching how each face's
      //flux entered the original Euler update (Q -= dt/dx*(F_right-F_left))
      double sign = (face == FaceDir::XP || face == FaceDir::YP) ? +1.0 : -1.0;
 
      for (const auto& nb : a.neighbors[fi]) {
        if (nb.level != NeighborLevel::Finer) continue; //reflux only ever corrects the coarse side
        MeshBlock& b = *blocks[nb.gid];
        int opp = xface ? (fi == (int)FaceDir::XP ? (int)FaceDir::XM : (int)FaceDir::XP)
                        : (fi == (int)FaceDir::YP ? (int)FaceDir::YM : (int)FaceDir::YP);
        int half = xface ? (int)(b.loc.lx2 & 1) : (int)(b.loc.lx1 & 1); //which half of a's face b covers
 
        for (int tb = 0; tb < N; tb += 2) { //N here is b's own (fine) tangential count == a's, by construction
          int t_a = half * (N / 2) + tb / 2; //a's tangential index this fine pair corrects
 
          const double* f0 = b.faceFlux(opp, tb);
          const double* f1 = b.faceFlux(opp, tb + 1);
          const double* f_used = a.faceFlux(fi, t_a);
 
          size_t ai, aj;
          if (face == FaceDir::XP)      { ai = (size_t)a.ie; aj = (size_t)(a.js + t_a); }
          else if (face == FaceDir::XM) { ai = (size_t)a.is; aj = (size_t)(a.js + t_a); }
          else if (face == FaceDir::YP) { ai = (size_t)(a.is + t_a); aj = (size_t)a.je; }
          else                          { ai = (size_t)(a.is + t_a); aj = (size_t)a.js; }
 
          double* q = a.grid.cell(ai, aj);
          for (size_t k = 0; k < a.grid.nvar(); ++k) {
            double f_eff = 0.5 * (f0[k] + f1[k]); //area-weighted average of the two fine-cell fluxes
            q[k] += sign * (dt / d) * (f_used[k] - f_eff);
          }
        }
      }
    }
  }
}
