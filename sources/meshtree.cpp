#include "mesh.hpp"

/*
 * Adaptive Mesh Refinement (AMR) quadtree implementation
 * 
 * manages recursive quadtree node subdivision, bounding box intersection tests
 * for targeted refinement regions, and leaf collection
*/


//constructs an individual quadtree node with a given logical location, parent link, and mesh context
MeshBlockTree::MeshBlockTree(Mesh* mesh, MeshBlockTree* parent, LogicalLocation loc)
  : mesh_(mesh), parent_(parent), loc_(loc) {}

//subdivides a leaf node into 4 child quadrants (quadtree children),
//computing their refined logical locations using bitwise indexing
void MeshBlockTree::Refine(){
  if(!IsLeaf()) return;
  for(int q = 0; q < 4; ++q){
    std::int64_t clx1 = (loc_.lx1 << 1) | (q & 1);
    std::int64_t clx2 = (loc_.lx2 << 1) | ((q >> 1) & 1);
    LogicalLocation cloc{loc_.level + 1, clx1, clx2};
    children_[q] = std::make_unique<MeshBlockTree>(mesh_, this, cloc);
  }
}

//removes and deallocates all child nodes, reverting this node back to a leaf
void MeshBlockTree::Derefine(){
  for (auto& c : children_) c.reset();
}

//recursively evaluates spatial intersection between the current node's bounding box
//and a specified refinement region, triggering subdivision if the target level is not yet met
void MeshBlockTree::RefineRegion(const RegionSize& root_domain,
                               int nx1_blocks_root, int nx2_blocks_root,
                               double x1min, double x1max,
                               double x2min, double x2max,
                               int target_level) {

  if(loc_.level >= target_level) return;

  //calculate physical coordinate boundaries for the current tree node
  double dom_w1 = root_domain.x1max - root_domain.x1min;
  double dom_w2 = root_domain.x2max - root_domain.x2min;
  std::int64_t n1_at_level = static_cast<std::int64_t>(nx1_blocks_root) << loc_.level;
  std::int64_t n2_at_level = static_cast<std::int64_t>(nx2_blocks_root) << loc_.level;
  double bx1min = root_domain.x1min + dom_w1*(double)loc_.lx1/(double)n1_at_level;
  double bx1max = root_domain.x1min + dom_w1*(double)(loc_.lx1 + 1)/(double)n1_at_level;
  double bx2min = root_domain.x2min + dom_w2*(double)loc_.lx2/(double)n2_at_level;
  double bx2max = root_domain.x2min + dom_w2*(double)(loc_.lx2 + 1)/(double)n2_at_level;

  //check for spatial bounding box overlap with the requested refinement zone
  bool overlaps = (bx1min < x1max) && (bx1max > x1min) &&
                  (bx2min < x2max) && (bx2max > x2min);
  if(!overlaps) return;

  //if overlapping and currently a leaf, perform refinement and recurse into children
  if(IsLeaf()) Refine();
  for (auto& c : children_){
    c->RefineRegion(root_domain, nx1_blocks_root, nx2_blocks_root,
                    x1min, x1max, x2min, x2max, target_level);
  }
}

//recursively traverses the quadtree to gather logical locations of all active leaf nodes
void MeshBlockTree::CollectLeaves(std::vector<LogicalLocation>& out) const {
  if(IsLeaf()){
    out.push_back(loc_);
    return;
  }
  for(const auto& c : children_) c->CollectLeaves(out);
}
