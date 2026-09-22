#pragma once

#include <vector>
#include <memory>
#include <cstdint>
#include <array>
#include "hydro_config.hpp"

struct LogicalLocation{
  int level = 0;
  std::int64_t lx1 = 0;
  std::int64_t lx2 = 0;

  LogicalLocation Parent() const {
    return LogicalLocation{level - 1, lx1 >> 1, lx2 >> 1};
  }

  bool operator==(const LogicalLocation& o) const {
    return level == o.level && lx1 == o.lx1 && lx2 == o.lx2;
  }

  bool operator<(const LogicalLocation& o) const {
    if(level != o.level) return level < o.level;
    if(lx1 != o.lx1) return lx1 < o.lx1;
    return lx2 < o.lx2;
  }
};

struct RegionSize{
  Real x1min = 0.0, x1max = 0.0;
  Real x2min = 0.0, x2max = 0.0;
  int nx1 = 0;
  int nx2 = 0;
};

struct RefinementRegion{
  Real x1min, x1max, x2min, x2max;
  int level;
};

class Mesh;

class MeshBlockTree{
public:
  MeshBlockTree(Mesh* mesh, MeshBlockTree* parent, LogicalLocation loc);

  bool IsLeaf() const { return children_[0] == nullptr; }
  const LogicalLocation& GetLocation() const { return loc_; }

  void Refine();
  void Derefine();
  void RefineRegion(const RegionSize& root_domain,
                    int nx1_blocks_root, int nx2_blocks_root,
                    Real x1min, Real x1max,
                    Real x2min, Real x2max,
                    int target_level);
  void CollectLeaves(std::vector<LogicalLocation>& out) const;

private:
  Mesh* mesh_;
  MeshBlockTree* parent_;
  LogicalLocation loc_;
  std::array<std::unique_ptr<MeshBlockTree>, 4> children_;
};

enum class FaceDir{ XM = 0, XP = 1, YM = 2, YP = 3 };
enum class NeighborLevel{ Same = 0, Coarser = 1, Finer = 2, PhysicalBoundary = 3 };

struct NeighborBlock{
  int gid;
  NeighborLevel level;
  bool periodic_wrap;
};

class MeshBlock{
public:
  MeshBlock(int gid_, int slot_id_, LogicalLocation loc_, RegionSize size_, int nghost_);

  int gid;
  int slot_id;
  LogicalLocation loc;
  RegionSize block_size;
  int nghost;

  int is, ie, js, je;

  bool is_physical_boundary[4];
  std::vector<NeighborBlock> neighbors[4];

  Real x1v(int i) const;
  Real x2v(int j) const;

  Real dx1() const { return (block_size.x1max - block_size.x1min)/block_size.nx1; }
  Real dx2() const { return (block_size.x2max - block_size.x2min)/block_size.nx2; }

  int tangentCount(int face) const {
    return (face == (int)FaceDir::XM || face == (int)FaceDir::XP) ? block_size.nx2 : block_size.nx1;
  }
};

struct MeshInputs {
    RegionSize domain;
    int nx1_block = 0, nx2_block = 0;
    int nghost = 2;
    int max_blocks = 256;
    bool periodic_x1 = false, periodic_x2 = false;
    std::vector<RefinementRegion> refine_regions;
};

class Mesh{
public:
  explicit Mesh(const MeshInputs& in);

  void BuildTree();
  void CreateMeshBlocks();
  void SetupNeighbors();
  void SyncNeighborTablesDevice();
  void swap_data_views(){ std::swap(data_view_, data_next_view_); }

  std::vector<std::unique_ptr<MeshBlock>>& blocks(){ return blocks_; }
  const std::vector<std::unique_ptr<MeshBlock>>& blocks() const { return blocks_; }

  View4D get_data_view(){ return data_view_; }
  View4D get_data_next_view(){ return data_next_view_; }
  View4D get_flux_view(){ return flux_view_; }
  IntView2D get_neighbor_gid_view(){ return neighbor_gid_; }
  IntView2D get_neighbor_level_view(){ return neighbor_level_; }

  //changed
  IntView2D get_neighbor_gid2_view(){ return neighbor_gid2_; }
  IntView2D get_block_parity_view(){ return block_parity_; }

  const MeshInputs& inputs() const { return in_; }
  int nx1_blocks_root() const { return nx1_blocks_root_; }
  int nx2_blocks_root() const { return nx2_blocks_root_; }
  int num_active_blocks() const { return static_cast<int>(blocks_.size()); }

private:
  MeshInputs in_;
  int nx1_blocks_root_, nx2_blocks_root_;
  std::vector<std::unique_ptr<MeshBlockTree>> roots_;
  std::vector<std::unique_ptr<MeshBlock>> blocks_;

  //double-buffered device memory pool & tables
  View4D data_view_;
  View4D data_next_view_;
  View4D flux_view_;
  IntView2D neighbor_gid_;
  IntView2D neighbor_level_;
  IntView2D neighbor_gid2_;      //second finer-neighbor slot, shape (max_blocks,4), -1 if unused
  IntView2D block_parity_;

  HostIntView2D host_neighbor_gid_;
  HostIntView2D host_neighbor_level_;
  HostIntView2D host_neighbor_gid2_;
  HostIntView2D host_block_parity_;

  RegionSize BlockSizeFromLocation(const LogicalLocation& loc) const;

  friend class MeshBlockTree;
};

void ApplyPhysicalBoundaryConditions(Mesh& mesh, const GridBC& bc);
void ExchangeGhostZones(Mesh& mesh);
void ApplyReflux(Mesh& mesh, Real dt);
