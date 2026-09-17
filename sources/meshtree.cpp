#include "../headers/mesh.h"

/*
 MeshBlockTree implementation
 Quadtree used for static mesh refinement (SMR). 2D analogue of the
 octree in Athena++ (Stone et al. 2020, Sec. 3.2).

 A leaf's physical extent is never stored on the tree itself; 
 it's recomputed on demand from (root domain, nx*_blocks_root, LogicalLocation)
 by Mesh::BlockSizeFromLocation. That keeps the tree purely topological.
*/

MeshBlockTree::MeshBlockTree(Mesh* mesh, MeshBlockTree* parent, LogicalLocation loc)
  : mesh_(mesh), parent_(parent), loc_(loc) {}


void MeshBlockTree::Refine() {
  if (!IsLeaf()) return; //already refined, nothing to do
  for (int q = 0; q < 4; ++q) {
    //quadrant order: bit0 -> x1 half, bit1 -> x2 half
    std::int64_t clx1 = (loc_.lx1 << 1) | (q & 1);
    std::int64_t clx2 = (loc_.lx2 << 1) | ((q >> 1) & 1);
    LogicalLocation cloc{loc_.level + 1, clx1, clx2};
    children_[q] = std::make_unique<MeshBlockTree>(mesh_, this, cloc);
  }
}


void MeshBlockTree::Derefine() {
  for (auto& c : children_) c.reset();
}


void MeshBlockTree::RefineRegion(const RegionSize& root_domain,
                                  int nx1_blocks_root, int nx2_blocks_root,
                                  double x1min, double x1max,
                                  double x2min, double x2max,
                                  int target_level) {
  if (loc_.level >= target_level) return; //already deep enough here

  //physical extent of this node's block, purely from its LogicalLocation
  double dom_w1 = root_domain.x1max - root_domain.x1min;
  double dom_w2 = root_domain.x2max - root_domain.x2min;
  std::int64_t n1_at_level = static_cast<std::int64_t>(nx1_blocks_root) << loc_.level;
  std::int64_t n2_at_level = static_cast<std::int64_t>(nx2_blocks_root) << loc_.level;
  double bx1min = root_domain.x1min + dom_w1 * (double)loc_.lx1 / (double)n1_at_level;
  double bx1max = root_domain.x1min + dom_w1 * (double)(loc_.lx1 + 1) / (double)n1_at_level;
  double bx2min = root_domain.x2min + dom_w2 * (double)loc_.lx2 / (double)n2_at_level;
  double bx2max = root_domain.x2min + dom_w2 * (double)(loc_.lx2 + 1) / (double)n2_at_level;

  //overlap test against the requested refinement box
  bool overlaps = (bx1min < x1max) && (bx1max > x1min) &&
                   (bx2min < x2max) && (bx2max > x2min);
  if (!overlaps) return;

  if (IsLeaf()) Refine();
  for (auto& c : children_) {
    c->RefineRegion(root_domain, nx1_blocks_root, nx2_blocks_root,
                     x1min, x1max, x2min, x2max, target_level);
  }
}


void MeshBlockTree::CollectLeaves(std::vector<LogicalLocation>& out) const {
  if (IsLeaf()) {
    out.push_back(loc_);
    return;
  }
  for (const auto& c : children_) c->CollectLeaves(out);
}
