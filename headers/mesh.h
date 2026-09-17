#ifndef MESH_H
#define MESH_H
 
#include <vector>
#include <memory>
#include <cstdint>
#include <array>
#include "grid.h"
 
/*
 Mesh / MeshBlock / MeshBlockTree
 Athena++ style domain decomposition (Stone et al. 2020, ApJS 249, 4)
 
 The idea: separate "where is the physical domain cut up, and how is
 each piece connected to its neighbors" (Mesh + MeshBlockTree, this file)
 from "what physics lives on a piece" (Grid, in grid.h; HLLC/slope
 limiter physics kernels operate on a MeshBlock's Grid same as before).
 Independent mesh blocks will allow us to use a Quadtree for AMR (Athena++)
 
 A MeshBlock owns exactly one Grid plus its own index ranges, physical
 extent and neighbor list. It knows nothing about hydro stuff. 
 Nothing in this file references NVAR, HLLC, or the Euler equations.
 
  - LogicalLocation:  quadtree address (level, lx1, lx2), Athena++ style (public repo)
  - MeshBlockTree:    the quadtree itself, built once at startup and then
                      frozen (this is what makes it "static" mesh
                      refinement. For AMR later, Refine()/Derefine()
                      just gets called mid-run and the tree/block list
                      gets rebuilt)
  - Mesh:             owns the tree, instantiates one MeshBlock per leaf,
                      and works out block-to-block neighbor connectivity
                      (same-level, coarser, finer) purely from
                      LogicalLocations. NO physics involved.
 
*/
 
 
//LogicalLocation: address of a node in the quadtree.
//level 0 = the root grid (the array of MeshBlocks we start with,
//before any refinement). lx1, lx2 are 0-based block indices *at that
//level*, i.e. in [0, nx1_blocks_root << level).
struct LogicalLocation {
  int level = 0;
  std::int64_t lx1 = 0;
  std::int64_t lx2 = 0;
 
  LogicalLocation Parent() const {
    return LogicalLocation{level - 1, lx1 >> 1, lx2 >> 1};
  }
 
  bool operator==(const LogicalLocation& o) const {
    return level == o.level && lx1 == o.lx1 && lx2 == o.lx2;
  }
  //ordering so LogicalLocation can key a std::map 
  //needed for O(log N) leaf lookup (???)
  bool operator<(const LogicalLocation& o) const {
    if (level != o.level) return level < o.level;
    if (lx1 != o.lx1) return lx1 < o.lx1;
    return lx2 < o.lx2;
  }
};
 
 
//RegionSize: purely geometric description of a block 
//physical extent + how many active (non-ghost) cells it has. No physics.
struct RegionSize {
  double x1min = 0.0, x1max = 0.0;
  double x2min = 0.0, x2max = 0.0;
  int nx1 = 0; //active cells in x1
  int nx2 = 0; //active cells in x2
};
 
 
//RefinementRegion: one static-mesh-refinement request 
//"Refine every root block overlapping this box up to 'level'."
//Read from the param file, applied once when the tree is built, never touched again
//that's what makes this SMR and not AMR.
struct RefinementRegion {
  double x1min, x1max, x2min, x2max;
  int level; //target refinement level (0 = root, i.e. no-op)
};
 
 
class Mesh; //fwd decl
 
//MeshBlockTree: one node of the quadtree. A leaf <=> a real MeshBlock
//will be created there. Internal nodes exist only to record how the
//domain was subdivided; they carry no data of their own.
class MeshBlockTree {
public:
  MeshBlockTree(Mesh* mesh, MeshBlockTree* parent, LogicalLocation loc);
 
  bool IsLeaf() const { return children_[0] == nullptr; }
  const LogicalLocation& GetLocation() const { return loc_; }
 
  //AMR
  //split this leaf into 4 children (quadrant order: 0=(lo,lo),
  //1=(hi,lo), 2=(lo,hi), 3=(hi,hi) in (x1,x2))
  void Refine();
 
  //AMR
  //collapse this node's 4 children back into a IsLeaf 
  //(kept for AMR but unused by SMR, since SMR never derefines after setup)
  void Derefine();
 
  //recursively split every leaf whose block overlaps [x1min,x1max] x [x2min,x2max]
  //until it is at 'target_level'. This is the entire
  //SMR algorithm: call it once per RefinementRegion while building.
  void RefineRegion(const RegionSize& root_domain,
                     int nx1_blocks_root, int nx2_blocks_root,
                     double x1min, double x1max,
                     double x2min, double x2max,
                     int target_level);
 
  //append the LogicalLocation of every leaf under this node, in
  //quadrant (Z-order/Morton) order -- gives a spatially-local block
  //ordering for free, which matters once blocks get distributed across
  //ranks/devices
  void CollectLeaves(std::vector<LogicalLocation>& out) const;
 
private:
  Mesh* mesh_;
  MeshBlockTree* parent_;
  LogicalLocation loc_;
  std::array<std::unique_ptr<MeshBlockTree>, 4> children_;
};
 
 
//Neighbor bookkeeping
enum class FaceDir { XM = 0, XP = 1, YM = 2, YP = 3 }; //-x1,+x1,-x2,+x2
enum class NeighborLevel { Same, Coarser, Finer };
 
struct NeighborBlock {
  int gid;              //global id (index into Mesh::blocks()) of the neighbor
  NeighborLevel level;  //neighbor's refinement level relative to this block
  bool periodic_wrap;   //true if this neighbor was found by wrapping across a periodic domain edge
};
 
 
//MeshBlock: one leaf of the tree, now with an actual Grid attached.
//Everything physics-specific (state vector layout, NVAR, HLLC, ...)
//lives outside this class, it only sees Grid, which is a flat
//physics-agnostic array (see grid.h).
class MeshBlock {
public:
  MeshBlock(int gid_, LogicalLocation loc_, RegionSize size_, int nghost_);
 
  int gid;
  LogicalLocation loc;
  RegionSize block_size;
  int nghost;
 
  //active-zone index ranges within "grid" (grid also holds nghost
  //layers of ghost cells on every side, same convention as before)
  int is, ie, js, je;
 
  Grid grid; //the only physics-facing member: this block's local state array
 
  //per-face info, indexed by FaceDir
  bool is_physical_boundary[4];       //true <=> this face has no block neighbor (domain edge, non-periodic)
  std::vector<NeighborBlock> neighbors[4]; //1 entry if same/coarser level, up to 2 if finer
 
  //cell-center physical coordinates (mainly for initial conditions)
  double x1v(int i) const;
  double x2v(int j) const;
 
  double dx1() const { return (block_size.x1max - block_size.x1min) / block_size.nx1; }
  double dx2() const { return (block_size.x2max - block_size.x2min) / block_size.nx2; }
  
  //for reflux
  std::vector<double> flux_face[4];
  //pointer to the NVAR-wide flux stored for tangential index t on the given face
  //t in [0, tangentCount(face)]
  double* faceFlux(int face, int t){ return flux_face[face].data() + (size_t)t*NVAR; }
  const double* faceFlux(int face, int t) const { return flux_face[face].data() + (size_t)t*NVAR; }
  int tangentCount(int face) const {
    return (face == (int)FaceDir::XM || face == (int)FaceDir::XP) ? block_size.nx2 : block_size.nx1;
  }
};
 
 
//Mesh: owns the tree + the flat list of MeshBlocks it produces, and
//wires up their neighbor connectivity. Knows nothing about hydro.
struct MeshInputs {
  RegionSize domain;                 //whole physical domain + total active cells at root level
  int nx1_block = 0, nx2_block = 0;  //active cells per MeshBlock (must divide domain cell counts)
  int nghost = 2;
  bool periodic_x1 = false, periodic_x2 = false; //folds "Periodic" BC into block connectivity
  std::vector<RefinementRegion> refine_regions;  //static refinement requests
};
 
class Mesh {
public:
  explicit Mesh(const MeshInputs& in);
 
  void BuildTree();          //apply SMR refinement regions to the quadtree
  void CreateMeshBlocks();   //instantiate one MeshBlock per leaf
  void SetupNeighbors();     //fill in every MeshBlock's neighbor lists
 
  std::vector<std::unique_ptr<MeshBlock>>& blocks() { return blocks_; }
  const std::vector<std::unique_ptr<MeshBlock>>& blocks() const { return blocks_; }
 
  const MeshInputs& inputs() const { return in_; }
  int nx1_blocks_root() const { return nx1_blocks_root_; }
  int nx2_blocks_root() const { return nx2_blocks_root_; }
 
private:
  MeshInputs in_;
  int nx1_blocks_root_, nx2_blocks_root_; //# of root-level (level 0) MeshBlocks along each axis
  //one quadtree per root-level block (a root block only ever refines
  //**into** its own subtree, so nx1_blocks_root_ * nx2_blocks_root_
  //independent trees is simpler than one shared tree with a fake
  //super-root)
  std::vector<std::unique_ptr<MeshBlockTree>> roots_;
  std::vector<std::unique_ptr<MeshBlock>> blocks_;
 
  RegionSize BlockSizeFromLocation(const LogicalLocation& loc) const;
 
  friend class MeshBlockTree;
};
 
 
void ApplyPhysicalBoundaryConditions(Mesh& mesh, const GridBC& bc);
void ExchangeGhostZones(Mesh& mesh);
void ApplyReflux(Mesh& mesh, double dt);
 
#endif // MESH_H
 
