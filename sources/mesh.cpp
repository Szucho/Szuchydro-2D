#include "mesh.hpp"
#include <stdexcept>
#include <algorithm>

Mesh::Mesh(const MeshInputs& in) : in_(in){
  if(in_.nx1_block <= 0 || in_.nx2_block <= 0)
    throw std::invalid_argument("Mesh: block size must be positive");
  if(in_.domain.nx1 % in_.nx1_block != 0 || in_.domain.nx2 % in_.nx2_block != 0)
    throw std::invalid_argument("Mesh: root grid cell counts must be divisible by MeshBlock size");

  nx1_blocks_root_ = in_.domain.nx1/in_.nx1_block;
  nx2_blocks_root_ = in_.domain.nx2/in_.nx2_block;

  int total_cells_i = in_.nx1_block + 2*in_.nghost;
  int total_cells_j = in_.nx2_block + 2*in_.nghost;
  int max_tangent = std::max(in_.nx1_block, in_.nx2_block);

  //allocate double-buffered flat device memory pool
  data_view_      = View4D("data_view", in_.max_blocks, total_cells_i, total_cells_j, NVAR);
  data_next_view_ = View4D("data_next_view", in_.max_blocks, total_cells_i, total_cells_j, NVAR);
  flux_view_      = View4D("flux_view", in_.max_blocks, 4, max_tangent, NVAR);
  
  neighbor_gid_   = IntView2D("neighbor_gid", in_.max_blocks, 4);
  neighbor_level_ = IntView2D("neighbor_level", in_.max_blocks, 4);

  host_neighbor_gid_   = Kokkos::create_mirror_view(neighbor_gid_);
  host_neighbor_level_ = Kokkos::create_mirror_view(neighbor_level_);

  neighbor_gid2_ = IntView2D("neighbor_gid2", in_.max_blocks, 4);
  block_parity_  = IntView2D("block_parity",  in_.max_blocks, 2);
  host_neighbor_gid2_ = Kokkos::create_mirror_view(neighbor_gid2_);
  host_block_parity_  = Kokkos::create_mirror_view(block_parity_);
}

RegionSize Mesh::BlockSizeFromLocation(const LogicalLocation& loc) const {
  Real dom_w1 = in_.domain.x1max - in_.domain.x1min;
  Real dom_w2 = in_.domain.x2max - in_.domain.x2min;
  std::int64_t n1 = static_cast<std::int64_t>(nx1_blocks_root_) << loc.level;
  std::int64_t n2 = static_cast<std::int64_t>(nx2_blocks_root_) << loc.level;

  RegionSize sz;
  sz.x1min = in_.domain.x1min + dom_w1*(Real)loc.lx1/(Real)n1;
  sz.x1max = in_.domain.x1min + dom_w1*(Real)(loc.lx1 + 1)/(Real)n1;
  sz.x2min = in_.domain.x2min + dom_w2*(Real)loc.lx2/(Real)n2;
  sz.x2max = in_.domain.x2min + dom_w2*(Real)(loc.lx2 + 1)/(Real)n2;
  sz.nx1 = in_.nx1_block;
  sz.nx2 = in_.nx2_block;
  return sz;
}

void Mesh::BuildTree(){
  roots_.clear();
  roots_.reserve((size_t)nx1_blocks_root_*(size_t)nx2_blocks_root_);
  for(int j = 0; j < nx2_blocks_root_; ++j){
    for (int i = 0; i < nx1_blocks_root_; ++i){
      LogicalLocation loc{0, i, j};
      roots_.push_back(std::make_unique<MeshBlockTree>(this, nullptr, loc));
    }
  }

  for(const auto& reg : in_.refine_regions){
    for(auto& root : roots_){
      root->RefineRegion(in_.domain, nx1_blocks_root_, nx2_blocks_root_,
                         reg.x1min, reg.x1max, reg.x2min, reg.x2max, reg.level);
    }
  }
}


MeshBlock::MeshBlock(int gid_, int slot_id_, LogicalLocation loc_, RegionSize size_, int nghost_)
  : gid(gid_), slot_id(slot_id_), loc(loc_), block_size(size_), nghost(nghost_) {
  is = nghost;
  ie = nghost + block_size.nx1 - 1;
  js = nghost;
  je = nghost + block_size.nx2 - 1;
  for (int i = 0; i < 4; ++i) {
      is_physical_boundary[i] = true;
  }
}

void Mesh::CreateMeshBlocks(){
  std::vector<LogicalLocation> leaves;
  for(auto& root : roots_) root->CollectLeaves(leaves);

  if(leaves.size() > static_cast<size_t>(in_.max_blocks)) {
    throw std::runtime_error("Exceeded maximum allocated block capacity in View4D!");
  }

  blocks_.clear();
  blocks_.reserve(leaves.size());
  int gid = 0;
  for(const auto& loc : leaves){
    RegionSize sz = BlockSizeFromLocation(loc);
    blocks_.push_back(std::make_unique<MeshBlock>(gid, gid, loc, sz, in_.nghost));
    ++gid;
  }
}

namespace{
struct IndexRange{ std::int64_t lo, hi; };

IndexRange RangeAtLevel(std::int64_t lx, int level, int common_level){
  std::int64_t scale = 1LL << (common_level - level);
  return { lx * scale, (lx + 1) * scale - 1 };
}

bool Overlap(const IndexRange& a, const IndexRange& b){
  return a.lo <= b.hi && b.lo <= a.hi;
}
} // namespace

void Mesh::SetupNeighbors(){
  for(auto& bptr : blocks_){
    MeshBlock& b = *bptr;
    for(int f = 0; f < 4; ++f){
      b.is_physical_boundary[f] = true;
      b.neighbors[f].clear();
    }
  }

  for(auto& aptr : blocks_){
    MeshBlock& a = *aptr;
    for(int fi = 0; fi < 4; ++fi){
      FaceDir face = static_cast<FaceDir>(fi);
      for(auto& bptr : blocks_){
        MeshBlock& b = *bptr;
        bool same_block = (&a == &b);

        int lc = std::max(a.loc.level, b.loc.level);
        IndexRange a1 = RangeAtLevel(a.loc.lx1, a.loc.level, lc);
        IndexRange a2 = RangeAtLevel(a.loc.lx2, a.loc.level, lc);

        std::int64_t domain_w1 = static_cast<std::int64_t>(nx1_blocks_root_) << lc;
        std::int64_t domain_w2 = static_cast<std::int64_t>(nx2_blocks_root_) << lc;

        for(int wrap = 0; wrap < 2; ++wrap){
          if(wrap == 0 && same_block) continue;
          if(wrap == 1 && !((face == FaceDir::XM || face == FaceDir::XP) ? in_.periodic_x1 : in_.periodic_x2))
              continue;

          IndexRange b1 = RangeAtLevel(b.loc.lx1, b.loc.level, lc);
          IndexRange b2 = RangeAtLevel(b.loc.lx2, b.loc.level, lc);
          if(wrap == 1){
            if(face == FaceDir::XM || face == FaceDir::XP){ b1.lo -= domain_w1; b1.hi -= domain_w1; }
            else{ b2.lo -= domain_w2; b2.hi -= domain_w2; }
          }

          bool adjacent = false;
          switch(face){
            case FaceDir::XP: adjacent = (b1.lo == a1.hi + 1) && Overlap(a2, b2); break;
            case FaceDir::XM: adjacent = (a1.lo == b1.hi + 1) && Overlap(a2, b2); break;
            case FaceDir::YP: adjacent = (b2.lo == a2.hi + 1) && Overlap(a1, b1); break;
            case FaceDir::YM: adjacent = (a2.lo == b2.hi + 1) && Overlap(a1, b1); break;
          }

          if(!adjacent && wrap == 1){
            IndexRange b1b = RangeAtLevel(b.loc.lx1, b.loc.level, lc);
            IndexRange b2b = RangeAtLevel(b.loc.lx2, b.loc.level, lc);
            if(face == FaceDir::XM || face == FaceDir::XP){ b1b.lo += domain_w1; b1b.hi += domain_w1; }
            else{ b2b.lo += domain_w2; b2b.hi += domain_w2; }
            switch(face){
              case FaceDir::XP: adjacent = (b1b.lo == a1.hi + 1) && Overlap(a2, b2b); break;
              case FaceDir::XM: adjacent = (a1.lo == b1b.hi + 1) && Overlap(a2, b2b); break;
              case FaceDir::YP: adjacent = (b2b.lo == a2.hi + 1) && Overlap(a1, b1b); break;
              case FaceDir::YM: adjacent = (a2.lo == b2b.hi + 1) && Overlap(a1, b1b); break;
            }
          }

          if(adjacent){
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

  SyncNeighborTablesDevice();
}

//changed meshblock sync logic uses gid2 for bookkeeping
void Mesh::SyncNeighborTablesDevice(){
  for(size_t b = 0; b < blocks_.size(); ++b){
    host_block_parity_(b, 0) = (int)(blocks_[b]->loc.lx1 & 1);
    host_block_parity_(b, 1) = (int)(blocks_[b]->loc.lx2 & 1);

    for(int f = 0; f < 4; ++f){
      host_neighbor_gid2_(b, f) = -1;

      if(blocks_[b]->is_physical_boundary[f]){
        host_neighbor_gid_(b, f)   = -1;
        host_neighbor_level_(b, f) = static_cast<int>(NeighborLevel::PhysicalBoundary);
        continue;
      }
      if(blocks_[b]->neighbors[f].empty()) continue;

      NeighborLevel lvl = blocks_[b]->neighbors[f][0].level;
      host_neighbor_level_(b, f) = static_cast<int>(lvl);

      if(lvl == NeighborLevel::Finer){
        bool xface = (f == (int)FaceDir::XM || f == (int)FaceDir::XP);
        for(const auto& nb : blocks_[b]->neighbors[f]){
          const auto& bloc = blocks_[nb.gid]->loc;
          int half = xface ? (int)(bloc.lx2 & 1) : (int)(bloc.lx1 & 1);
          if(half == 0) host_neighbor_gid_(b, f)  = nb.gid;
          else           host_neighbor_gid2_(b, f) = nb.gid;
        }
      } else {
        host_neighbor_gid_(b, f) = blocks_[b]->neighbors[f][0].gid;
      }
    }
  }
  Kokkos::deep_copy(neighbor_gid_,   host_neighbor_gid_);
  Kokkos::deep_copy(neighbor_level_, host_neighbor_level_);
  Kokkos::deep_copy(neighbor_gid2_,  host_neighbor_gid2_);
  Kokkos::deep_copy(block_parity_,   host_block_parity_);
}

Real MeshBlock::x1v(int i) const {
  return block_size.x1min + (i - is + 0.5) * dx1();
}
Real MeshBlock::x2v(int j) const {
  return block_size.x2min + (j - js + 0.5) * dx2();
}

void ExchangeGhostZones(Mesh& mesh){
  int n_active = mesh.num_active_blocks();
  if (n_active == 0) return;

  View4D data = mesh.get_data_view();
  IntView2D n_gid = mesh.get_neighbor_gid_view();
  IntView2D n_level = mesh.get_neighbor_level_view();
  IntView2D n_gid2 = mesh.get_neighbor_gid2_view();   //gid2 was needed and parity
  IntView2D parity = mesh.get_block_parity_view();

  int nx1 = mesh.inputs().nx1_block;
  int nx2 = mesh.inputs().nx2_block;
  int nghost = mesh.inputs().nghost;
  int is = nghost, ie = nghost + nx1 - 1;
  int js = nghost, je = nghost + nx2 - 1;

  Kokkos::parallel_for(
    "ExchangeGhostZones",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {n_active, 4, std::max(nx1, nx2)}),
    KOKKOS_LAMBDA(const int a_slot, const int fi, const int t){
      int lvl = n_level(a_slot, fi);
      if(lvl == static_cast<int>(NeighborLevel::PhysicalBoundary)) return;

      int b_slot = n_gid(a_slot, fi);
      FaceDir face = static_cast<FaceDir>(fi);
      bool xface = (face == FaceDir::XM || face == FaceDir::XP);
      int N = xface ? nx2 : nx1;

      if(t >= N) return;

      if(lvl == static_cast<int>(NeighborLevel::Same)){
        if(face == FaceDir::XP){
          for(int g = 0; g < nghost; ++g)
            for(size_t k = 0; k < NVAR; ++k) data(a_slot, ie + 1 + g, js + t, k) = data(b_slot, is + g, js + t, k);
        }else if(face == FaceDir::XM){
          for(int g = 0; g < nghost; ++g)
              for(size_t k = 0; k < NVAR; ++k) data(a_slot, is - 1 - g, js + t, k) = data(b_slot, ie - g, js + t, k);
        }else if(face == FaceDir::YP){
          for(int g = 0; g < nghost; ++g)
            for(size_t k = 0; k < NVAR; ++k) data(a_slot, is + t, je + 1 + g, k) = data(b_slot, is + t, js + g, k);
        }else if(face == FaceDir::YM){
          for(int g = 0; g < nghost; ++g)
            for(size_t k = 0; k < NVAR; ++k) data(a_slot, is + t, js - 1 - g, k) = data(b_slot, is + t, je - g, k);
        }
      } 
      else if(lvl == static_cast<int>(NeighborLevel::Coarser)){
        int half_N = N/2;
        int half = xface ? parity(a_slot, 1) : parity(a_slot, 0);
        int tk = half*half_N + t/2;
        int dt_sign = (t%2 == 0) ? -1 : 1;
        int tk_neighbor = tk + dt_sign;
        int lo = half * half_N, hi = lo + half_N - 1;
        if (tk_neighbor < lo || tk_neighbor > hi) tk_neighbor = tk;

        //lin interpolation
        Real w1 = 0.75, w2 = 0.25;
        for(int g = 0; g < nghost; ++g){
          if(face == FaceDir::XP){
            for(size_t k = 0; k < NVAR; ++k)
              data(a_slot, ie + 1 + g, js + t, k) = w1*data(b_slot, is + g, js + tk, k) +
                                                    w2*data(b_slot, is + g, js + tk_neighbor, k);
          }else if(face == FaceDir::XM){
            for(size_t k = 0; k < NVAR; ++k)
              data(a_slot, is - 1 - g, js + t, k) = w1*data(b_slot, ie - g, js + tk, k) +
                                                    w2*data(b_slot, ie - g, js + tk_neighbor, k);
          }else if(face == FaceDir::YP){
            for(size_t k = 0; k < NVAR; ++k)
              data(a_slot, is + t, je + 1 + g, k) = w1*data(b_slot, is + tk, js + g, k) +
                                                    w2*data(b_slot, is + tk_neighbor, js + g, k);
          }else if(face == FaceDir::YM){
            for(size_t k = 0; k < NVAR; ++k)
              data(a_slot, is + t, js - 1 - g, k) = w1*data(b_slot, is + tk, je - g, k) +
                                                    w2*data(b_slot, is + tk_neighbor, je - g, k);
          }
        }
      } 
      else if(lvl == static_cast<int>(NeighborLevel::Finer)){
        int half_N = N/2;
        int half = t/half_N;
        int local_t = t%half_N;
        int tb = 2*local_t;
        int b_slot_f = (half == 0) ? b_slot : n_gid2(a_slot, fi);

        for(int g = 0; g < nghost; ++g){
          if(face == FaceDir::XP){
            for(size_t k = 0; k < NVAR; ++k)
              data(a_slot, ie + 1 + g, js + t, k) = 0.5*(data(b_slot_f, is + g, js + tb, k) +
                                                         data(b_slot_f, is + g, js + tb + 1, k));
          }else if(face == FaceDir::XM){
            for(size_t k = 0; k < NVAR; ++k)
              data(a_slot, is - 1 - g, js + t, k) = 0.5*(data(b_slot_f, ie - g, js + tb, k) +
                                                         data(b_slot_f, ie - g, js + tb + 1, k));
          }else if(face == FaceDir::YP){
            for(size_t k = 0; k < NVAR; ++k)
              data(a_slot, is + t, je + 1 + g, k) = 0.5*(data(b_slot_f, is + tb, js + g, k) +
                                                         data(b_slot_f, is + tb + 1, js + g, k));
          }else if(face == FaceDir::YM){
            for(size_t k = 0; k < NVAR; ++k)
              data(a_slot, is + t, js - 1 - g, k) = 0.5*(data(b_slot_f, is + tb, je - g, k) +
                                                         data(b_slot_f, is + tb + 1, je - g, k));
          }
        }
      }
    }
  );
}

void ApplyReflux(Mesh& mesh, Real dt){ //changed
  int n_active = mesh.num_active_blocks();
  if (n_active == 0) return;

  View4D data = mesh.get_data_view();
  View4D flux = mesh.get_flux_view();
  IntView2D n_gid = mesh.get_neighbor_gid_view();
  IntView2D n_level = mesh.get_neighbor_level_view();
  IntView2D n_gid2 = mesh.get_neighbor_gid2_view();   // add
  IntView2D parity = mesh.get_block_parity_view();

  int nx1 = mesh.inputs().nx1_block;
  int nx2 = mesh.inputs().nx2_block;
  int nghost = mesh.inputs().nghost;
  Real dx1 = (mesh.inputs().domain.x1max - mesh.inputs().domain.x1min)/mesh.inputs().domain.nx1;
  Real dx2 = (mesh.inputs().domain.x2max - mesh.inputs().domain.x2min)/mesh.inputs().domain.nx2;

  Kokkos::parallel_for(
    "ApplyReflux",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {n_active, 4, std::max(nx1, nx2)/2}),
    KOKKOS_LAMBDA(const int a_slot, const int fi, const int t_idx) {
      if(n_level(a_slot, fi) != static_cast<int>(NeighborLevel::Finer)) return;

      FaceDir face = static_cast<FaceDir>(fi);
      bool xface = (face == FaceDir::XM || face == FaceDir::XP);
      int N = xface ? nx2 : nx1;
      if (t_idx >= N) return;

      int half_N = N/2;
      int half = t_idx/half_N;
      int local_t = t_idx%half_N;
      int tb = 2*local_t;
      int b_slot = (half == 0) ? n_gid(a_slot, fi) : n_gid2(a_slot, fi);

      Real d = xface ? dx1 : dx2;
      Real sign = (face == FaceDir::XP || face == FaceDir::YP) ? +1.0 : -1.0;
      int opp = xface ? (fi == (int)FaceDir::XP ? (int)FaceDir::XM : (int)FaceDir::XP)
                      : (fi == (int)FaceDir::YP ? (int)FaceDir::YM : (int)FaceDir::YP);

      int is = nghost, ie = nghost + nx1 - 1;
      int js = nghost, je = nghost + nx2 - 1;

      size_t ai = (face == FaceDir::XP) ? ie : (face == FaceDir::XM) ? is : (is + t_idx);
      size_t aj = (face == FaceDir::YP) ? je : (face == FaceDir::YM) ? js : (js + t_idx);

      for(size_t k = 0; k < NVAR; ++k){
        Real f_eff = 0.5*(flux(b_slot, opp, tb, k) + flux(b_slot, opp, tb + 1, k));
        data(a_slot, ai, aj, k) += sign*(dt/d)*(flux(a_slot, fi, t_idx, k) - f_eff);
      }
    }
  );
}
