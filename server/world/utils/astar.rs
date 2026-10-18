use std::sync::atomic::{AtomicI32, Ordering};
use std::time::{Duration, Instant};

use serde::Serialize;

use crate::Vec3;
use pathfinding::prelude::astar;

fn absdiff(a: i32, b: i32) -> u32 {
    if a > b {
        (a - b) as u32
    } else {
        (b - a) as u32
    }
}

#[derive(Debug, Clone, Serialize, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PathNode(pub i32, pub i32, pub i32);

impl PathNode {
    pub fn from_vec3(vec3: &Vec3<i32>) -> Self {
        Self(vec3.0, vec3.1, vec3.2)
    }

    pub fn distance(&self, other: &Self) -> u32 {
        (absdiff(self.0, other.0) + absdiff(self.1, other.1) + absdiff(self.2, other.2)) as u32
    }
}

#[derive(Default)]
pub struct AStar;

impl AStar {
    pub fn calculate(
        start: &Vec3<i32>,
        goal: &Vec3<i32>,
        successors: &dyn Fn(&PathNode) -> Vec<(PathNode, u32)>,
        heuristic: &dyn Fn(&PathNode) -> u32,
    ) -> Option<(Vec<PathNode>, u32)> {
        let start_node = PathNode::from_vec3(start);
        let goal_node = PathNode::from_vec3(goal);

        let mut visited = std::collections::HashSet::new();
        visited.insert(start_node.clone());

        astar(
            &start_node,
            |p| {
                successors(p)
                    .into_iter()
                    .filter(|(s, _)| visited.insert(s.clone()))
                    .collect::<Vec<_>>()
            },
            heuristic,
            |p| *p == goal_node,
        )
    }
}

/// Run the voxel A* search from `start` to `goal`.
///
/// World access is fully abstracted behind `is_passable`, so this is free of
/// any ECS or chunk dependencies and the caller owns the voxel cache. Callers
/// should back `is_passable` with a per-search cache (no shared locking) so
/// concurrent searches never contend.
///
/// `is_ground` is whether a cell can be stood on. It is usually
/// `!is_passable`, but a cell can be neither - an obstacle that blocks a body
/// without being somewhere to stand (see `PathObstacles`).
pub fn find_path(
    start: &Vec3<i32>,
    goal: &Vec3<i32>,
    height: f32,
    max_depth_search: i32,
    max_pathfinding_time: Duration,
    is_passable: &dyn Fn(i32, i32, i32) -> bool,
    is_ground: &dyn Fn(i32, i32, i32) -> bool,
) -> Option<(Vec<PathNode>, u32)> {
    // Whether an entity of `height` can stand on top of the block at (vx, vy, vz).
    let walkable = |vx: i32, vy: i32, vz: i32| -> bool {
        if !is_ground(vx, vy, vz) {
            return false;
        }
        for i in 1..(height.ceil() as i32 + 1) {
            if !is_passable(vx, vy + i, vz) {
                return false;
            }
        }
        true
    };

    let has_wall_nearby = |vx: i32, vy: i32, vz: i32| -> bool {
        for dx in -1..=1 {
            for dz in -1..=1 {
                if dx == 0 && dz == 0 {
                    continue;
                }
                if !is_passable(vx + dx, vy, vz + dz) {
                    return true;
                }
            }
        }
        false
    };

    let start_time = Instant::now();
    let count = AtomicI32::new(0);

    AStar::calculate(
        start,
        goal,
        &|node| {
            let &PathNode(vx, vy, vz) = node;
            let mut successors = vec![];
            let current_count = count.fetch_add(1, Ordering::Relaxed);

            if current_count >= max_depth_search || start_time.elapsed() > max_pathfinding_time {
                return successors;
            }

            // emptiness
            let py = !walkable(vx, vy + 1, vz);
            let px = !walkable(vx + 1, vy, vz);
            let pz = !walkable(vx, vy, vz + 1);
            let nx = !walkable(vx - 1, vy, vz);
            let nz = !walkable(vx, vy, vz - 1);
            let pxpy = !walkable(vx + 1, vy + 1, vz);
            let pzpy = !walkable(vx, vy + 1, vz + 1);
            let nxpy = !walkable(vx - 1, vy + 1, vz);
            let nzpy = !walkable(vx, vy + 1, vz - 1);

            // +X direction
            if walkable(vx + 1, vy - 1, vz) {
                let mut cost = 1;
                if has_wall_nearby(vx + 1, vy, vz) {
                    cost += 1;
                }
                successors.push((PathNode(vx + 1, vy, vz), cost));
            } else if walkable(vx + 1, vy, vz) && py {
                let mut cost = 2;
                if has_wall_nearby(vx + 1, vy + 1, vz) {
                    cost += 1;
                }
                successors.push((PathNode(vx + 1, vy + 1, vz), cost));
            } else if walkable(vx + 1, vy - 2, vz) && px {
                successors.push((PathNode(vx + 1, vy - 1, vz), 2));
            }

            // -X direction
            if walkable(vx - 1, vy - 1, vz) {
                let mut cost = 1;
                if has_wall_nearby(vx - 1, vy, vz) {
                    cost += 1;
                }
                successors.push((PathNode(vx - 1, vy, vz), cost));
            } else if walkable(vx - 1, vy, vz) && py {
                let mut cost = 2;
                if has_wall_nearby(vx - 1, vy + 1, vz) {
                    cost += 1;
                }
                successors.push((PathNode(vx - 1, vy + 1, vz), cost));
            } else if walkable(vx - 1, vy - 2, vz) && nx {
                successors.push((PathNode(vx - 1, vy - 1, vz), 2));
            }

            // +Z direction
            if walkable(vx, vy - 1, vz + 1) {
                let mut cost = 1;
                if has_wall_nearby(vx, vy, vz + 1) {
                    cost += 1;
                }
                successors.push((PathNode(vx, vy, vz + 1), cost));
            } else if walkable(vx, vy, vz + 1) && py {
                let mut cost = 2;
                if has_wall_nearby(vx, vy + 1, vz + 1) {
                    cost += 1;
                }
                successors.push((PathNode(vx, vy + 1, vz + 1), cost));
            } else if walkable(vx, vy - 2, vz + 1) && pz {
                successors.push((PathNode(vx, vy - 1, vz + 1), 2));
            }

            // -Z direction
            if walkable(vx, vy - 1, vz - 1) {
                let mut cost = 1;
                if has_wall_nearby(vx, vy, vz - 1) {
                    cost += 1;
                }
                successors.push((PathNode(vx, vy, vz - 1), cost));
            } else if walkable(vx, vy, vz - 1) && py {
                let mut cost = 2;
                if has_wall_nearby(vx, vy + 1, vz - 1) {
                    cost += 1;
                }
                successors.push((PathNode(vx, vy + 1, vz - 1), cost));
            } else if walkable(vx, vy - 2, vz - 1) && nz {
                successors.push((PathNode(vx, vy - 1, vz - 1), 2));
            }

            // +X+Z direction
            if walkable(vx + 1, vy - 1, vz + 1)
                && px
                && pz
                && is_passable(vx + 1, vy, vz)
                && is_passable(vx, vy, vz + 1)
            {
                let mut cost = 2;
                if has_wall_nearby(vx + 1, vy, vz + 1) {
                    cost += 1;
                }
                successors.push((PathNode(vx + 1, vy, vz + 1), cost));
            } else if walkable(vx + 1, vy, vz + 1) && py && pxpy && pzpy {
                successors.push((PathNode(vx + 1, vy + 1, vz + 1), 3));
            } else if walkable(vx + 1, vy - 2, vz + 1) && px && pz {
                successors.push((PathNode(vx + 1, vy - 1, vz + 1), 3));
            }

            // +X-Z direction
            if walkable(vx + 1, vy - 1, vz - 1)
                && px
                && nz
                && is_passable(vx + 1, vy, vz)
                && is_passable(vx, vy, vz - 1)
            {
                let mut cost = 2;
                if has_wall_nearby(vx + 1, vy, vz - 1) {
                    cost += 1;
                }
                successors.push((PathNode(vx + 1, vy, vz - 1), cost));
            } else if walkable(vx + 1, vy, vz - 1) && py && pxpy && nzpy {
                successors.push((PathNode(vx + 1, vy + 1, vz - 1), 3));
            } else if walkable(vx + 1, vy - 2, vz - 1) && px && nz {
                successors.push((PathNode(vx + 1, vy - 1, vz - 1), 3));
            }

            // -X+Z direction
            if walkable(vx - 1, vy - 1, vz + 1)
                && nx
                && pz
                && is_passable(vx - 1, vy, vz)
                && is_passable(vx, vy, vz + 1)
            {
                let mut cost = 2;
                if has_wall_nearby(vx - 1, vy, vz + 1) {
                    cost += 1;
                }
                successors.push((PathNode(vx - 1, vy, vz + 1), cost));
            } else if walkable(vx - 1, vy, vz + 1) && py && nxpy && pzpy {
                successors.push((PathNode(vx - 1, vy + 1, vz + 1), 3));
            } else if walkable(vx - 1, vy - 2, vz + 1) && nx && pz {
                successors.push((PathNode(vx - 1, vy - 1, vz + 1), 3));
            }

            // -X-Z direction
            if walkable(vx - 1, vy - 1, vz - 1)
                && nx
                && nz
                && is_passable(vx - 1, vy, vz)
                && is_passable(vx, vy, vz - 1)
            {
                let mut cost = 2;
                if has_wall_nearby(vx - 1, vy, vz - 1) {
                    cost += 1;
                }
                successors.push((PathNode(vx - 1, vy, vz - 1), cost));
            } else if walkable(vx - 1, vy, vz - 1) && py && nxpy && nzpy {
                successors.push((PathNode(vx - 1, vy + 1, vz - 1), 3));
            } else if walkable(vx - 1, vy - 2, vz - 1) && nx && nz {
                successors.push((PathNode(vx - 1, vy - 1, vz - 1), 3));
            }

            successors
        },
        // Manhattan distance is admissible and tight for this move set (every
        // move costs at least 1 per voxel of progress), so it keeps paths
        // optimal while exploring far fewer nodes than a scaled-down estimate.
        &|p| p.distance(&PathNode(goal.0, goal.1, goal.2)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Flat ground at y = 0, open air above, and a wall of obstacle cells
    /// across x = 5 from z = -4 to 4, two cells high: blocked, not ground.
    fn search(goal: Vec3<i32>) -> Option<Vec<PathNode>> {
        let wall = |x: i32, y: i32, z: i32| x == 5 && (-4..=4).contains(&z) && (1..=2).contains(&y);
        let is_passable = |x: i32, y: i32, z: i32| y > 0 && !wall(x, y, z);
        let is_ground = |_: i32, y: i32, _: i32| y <= 0;
        find_path(
            &Vec3(0, 1, 0),
            &goal,
            1.0,
            10_000,
            Duration::from_secs(1),
            &is_passable,
            &is_ground,
        )
        .map(|(nodes, _)| nodes)
    }

    #[test]
    fn a_path_goes_around_an_obstacle_and_never_over_it() {
        let nodes = search(Vec3(10, 1, 0)).expect("a way round the wall");
        assert!(nodes
            .iter()
            .all(|PathNode(x, y, z)| !(*x == 5 && (-4..=4).contains(z)) && *y == 1));
        assert!(nodes.iter().any(|PathNode(_, _, z)| z.abs() > 4));
        // The wall's top is not ground, so nothing up there can be reached.
        assert!(search(Vec3(5, 3, 0)).is_none());
    }
}
