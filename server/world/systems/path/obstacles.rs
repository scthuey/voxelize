use std::collections::HashSet;
use std::sync::Arc;

/// Cells walkers may not pass through that are not voxels: a game's props,
/// say, which it draws and collides as meshes.
///
/// Insert one as a resource and `PathFindingSystem` treats each cell as solid
/// for the space a body needs, but never as ground to stand on - so a path goes
/// around an obstacle, not over it. The physics engine does not see these, so a
/// body routed onto one would fall into it.
///
/// Cloning is cheap: the set is shared, and replaced whole when it changes.
#[derive(Clone, Default)]
pub struct PathObstacles(Arc<HashSet<(i32, i32, i32)>>);

impl PathObstacles {
    pub fn new(cells: HashSet<(i32, i32, i32)>) -> Self {
        Self(Arc::new(cells))
    }

    /// Whether the voxel cell at `(x, y, z)` is blocked.
    pub fn blocks(&self, x: i32, y: i32, z: i32) -> bool {
        self.0.contains(&(x, y, z))
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}
