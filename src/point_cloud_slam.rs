use crate::lsc_reader;

use std::ops::MulAssign;
#[derive(Debug, Clone)]
struct PointCloud {
    pub matrix: nalgebra::Matrix3xX<f64>,
}

impl PointCloud {
    fn new() -> Self {
        Self {
            matrix: nalgebra::Matrix3xX::from_vec(Vec::new()),
        }
    }

    fn from_x_y_vec(x_vec: Vec<f64>, y_vec: Vec<f64>) -> Self {
        let clen = x_vec.len();
        PointCloud {
            matrix: nalgebra::Matrix3xX::<f64>::from_rows(&[
                nalgebra::RowDVector::from_vec(x_vec),
                nalgebra::RowDVector::from_vec(y_vec),
                nalgebra::RowDVector::from_element(clen, 1.0),
            ]),
        }
    }

    fn from_vec_of_points(points: &Vec<lsc_reader::Point>) -> Self {
        let (xs, ys): (Vec<f64>, Vec<f64>) =
            points.iter().map(|p| (p.x as f64, p.y as f64)).unzip();
        let clen = xs.len();
        PointCloud {
            matrix: nalgebra::Matrix3xX::<f64>::from_rows(&[
                nalgebra::RowDVector::from_vec(xs),
                nalgebra::RowDVector::from_vec(ys),
                nalgebra::RowDVector::from_element(clen, 1.0),
            ]),
        }
    }

    fn transform_by_mat(mut self, transformation_matrix: nalgebra::Matrix3<f64>) -> Self {
        self.matrix = transformation_matrix * self.matrix;
        self
    }

    fn to_h_pointcloud_2d(&self) -> h_analyzer_data::PointCloud2D {
        h_analyzer_data::PointCloud2D::new(
            self.matrix.row(0).iter().map(|&v| v as f64).collect(),
            self.matrix.row(1).iter().map(|&v| v as f64).collect(),
        )
    }
}

struct VoxelGrid {
    x_vec: Vec<f64>,
    y_vec: Vec<f64>,
}

fn voxel_grid_filter(input: &PointCloud, grid_size: f64) -> PointCloud {
    let mut voxels = std::collections::HashMap::new();
    for v in input.matrix.column_iter() {
        if let (Some(&px), Some(&py)) = (v.get(0), v.get(1)) {
            let key = ((px / grid_size) as i32, (py / grid_size) as i32);
            if !voxels.contains_key(&key) {
                voxels.insert(
                    key,
                    VoxelGrid {
                        x_vec: Vec::new(),
                        y_vec: Vec::new(),
                    },
                );
            }
            if let Some(voxel) = voxels.get_mut(&key) {
                voxel.x_vec.push(px);
                voxel.y_vec.push(py);
            }
        }
    }
    let mut x_vec = Vec::new();
    let mut y_vec = Vec::new();
    for (_, voxel) in voxels.iter() {
        let vlen = voxel.x_vec.len() as f64;
        let xmean = voxel.x_vec.iter().sum::<f64>() / vlen;
        let ymean = voxel.y_vec.iter().sum::<f64>() / vlen;
        x_vec.push(xmean);
        y_vec.push(ymean);
    }
    PointCloud::from_x_y_vec(x_vec, y_vec)
}

struct PointCloudMap {
    minimum_time_diff: u64, // in nano seconds
    pub clouds: std::collections::BTreeMap<u64, PointCloud>,
    pub entire_map_cloud: PointCloud,
}

impl PointCloudMap {
    fn new() -> Self {
        Self {
            minimum_time_diff: (1e6 * 1e-1) as u64, // 0.1s
            clouds: std::collections::BTreeMap::new(),
            entire_map_cloud: PointCloud::new(),
        }
    }

    fn add_point_cloud(&mut self, timestamp: f64, cloud: PointCloud) {
        let inner_time = (timestamp * 1e6) as u64;
        if self.clouds.last_key_value().is_some() {
            //log::debug!("ts {} inner {}", timestamp * 1e6, inner_time);
            //log::debug!("last {}", self.clouds.last_key_value().unwrap().0);
        }
        if self.clouds.is_empty()
            || (inner_time - self.clouds.last_key_value().unwrap().0 >= self.minimum_time_diff)
        {
            self.clouds.insert(inner_time, cloud);
        }
        self.entire_map_cloud = voxel_grid_filter(&self.entire_map(), 0.05);
        log::debug!(
            "map point num filtered: {:?}",
            self.entire_map_cloud.matrix.shape()
        );
    }

    fn find_nearest_point(
        &self,
        query_point: &nalgebra::Matrix3x1<f64>,
    ) -> Option<nalgebra::Matrix3x1<f64>> {
        let distances: Vec<f64> = self
            .entire_map_cloud
            .matrix
            .column_iter()
            .map(|v| point_to_point_cost(&(v.into()), query_point))
            .collect();
        let (min_index, &min_dist) = distances
            .iter()
            .enumerate()
            .min_by(|(_, &a), (_, &b)| a.total_cmp(&b))?;
        if min_dist > (0.2 * 0.2) {
            return None;
        }
        Some(self.entire_map_cloud.matrix.column(min_index).into())
    }

    fn to_x_y_vec(&self) -> (Vec<f64>, Vec<f64>) {
        let mut x_vec = Vec::new();
        let mut y_vec = Vec::new();
        for (_, cloud) in self.clouds.iter() {
            x_vec.extend(cloud.matrix.row(0).iter().map(|&v| v).collect::<Vec<f64>>());
            y_vec.extend(cloud.matrix.row(1).iter().map(|&v| v).collect::<Vec<f64>>());
        }
        (x_vec, y_vec)
    }

    fn entire_map(&self) -> PointCloud {
        let (x_vec, y_vec) = self.to_x_y_vec();
        PointCloud::from_x_y_vec(x_vec, y_vec)
    }
}

fn point_to_point_cost(p1: &nalgebra::Matrix3x1<f64>, p2: &nalgebra::Matrix3x1<f64>) -> f64 {
    let dp = p1 - p2;
    let ans = (dp.transpose() * dp).sum();
    if ans > 0.2 {
        //log::debug!("p1 {} p2 {}, ans: {}", p1, p2, ans);
    }
    ans
}

fn calc_cost(
    pointcloud: PointCloud,
    nearests: &Vec<Option<nalgebra::Matrix3x1<f64>>>,
) -> (f64, i32) {
    let mut cost = 0.0;
    let pnum = pointcloud.matrix.shape().1;
    let mut valid_num = 0;
    for i in 0..pnum {
        if let Some(np) = nearests[i] {
            let p = pointcloud.matrix.column(i);
            cost += point_to_point_cost(&np, &p.into());
            valid_num += 1;
        }
    }
    (cost / valid_num as f64, valid_num)
}

struct ScanMatchingCost<'a> {
    initial_transform: nalgebra::Isometry2<f64>,
    derivative: nalgebra::Isometry2<f64>,
    pointcloud: PointCloud,
    nearests: &'a Vec<Option<nalgebra::Matrix3x1<f64>>>,
}

impl<'a> argmin::core::CostFunction for ScanMatchingCost<'a> {
    // one dimensional problem, no vector needed
    type Param = f64;
    type Output = f64;

    fn cost(&self, x: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        let new_x = self.initial_transform.translation.x - x * self.derivative.translation.x;
        let new_y = self.initial_transform.translation.y - x * self.derivative.translation.y;
        let new_a = self.initial_transform.rotation.angle() - x * self.derivative.rotation.angle();
        let new_transform = nalgebra::Isometry2::new(nalgebra::Vector2::new(new_x, new_y), new_a);

        let new_pc = self
            .pointcloud
            .clone()
            .transform_by_mat(new_transform.to_homogeneous());
        let (cost, _) = calc_cost(new_pc, &self.nearests);

        Ok(cost)
    }
}

struct PointCloudMatching {
    pub map: PointCloudMap,
    initial_cost: Option<f64>,
    optimized_cost: Option<f64>,
}

impl PointCloudMatching {
    fn new() -> Self {
        Self {
            map: PointCloudMap::new(),
            initial_cost: None,
            optimized_cost: None,
        }
    }

    pub fn process_current_cloud(
        &mut self,
        timestamp: f64,
        initial_transform: &nalgebra::Isometry2<f64>,
        current_cloud_in_base: PointCloud,
        better: bool,
    ) -> nalgebra::Isometry2<f64> {
        let current_cloud_in_base = voxel_grid_filter(&current_cloud_in_base, 0.05);

        if better {
            log::debug!("\n[PointCloudMatching process]");
        }
        let mut final_transform = initial_transform.clone();

        let mut best_cost = f64::MAX;
        let mut old_cost = f64::MAX;
        if better {
            // iterative optimization
            for i in 0..10 {
                // transform current_cloud by current estimate
                let current_cloud_in_odom = current_cloud_in_base
                    .clone()
                    .transform_by_mat(final_transform.to_homogeneous());
                // data association
                let mut nearests = Vec::new();
                for p in current_cloud_in_odom.matrix.column_iter() {
                    nearests.push(self.map.find_nearest_point(&p.into()));
                }

                let mut local_best_cost = f64::MAX;
                let mut local_old_cost = f64::MAX;
                let mut local_final_transform = final_transform.clone();
                for j in 0..10 {
                    // transform current_cloud by current estimate
                    let current_cloud_in_odom = current_cloud_in_base
                        .clone()
                        .transform_by_mat(local_final_transform.to_homogeneous());
                    // slightly shifted current points
                    let mut cp_transform = local_final_transform.clone();
                    cp_transform.translation.x += 1e-5;
                    let cc_odom_x_shifted = current_cloud_in_base
                        .clone()
                        .transform_by_mat(cp_transform.to_homogeneous());
                    let mut cp_transform = local_final_transform.clone();
                    cp_transform.translation.y += 1e-5;
                    let cc_odom_y_shifted = current_cloud_in_base
                        .clone()
                        .transform_by_mat(cp_transform.to_homogeneous());
                    let cp_transform = nalgebra::Isometry2::new(
                        local_final_transform.translation.vector,
                        (local_final_transform.rotation.angle() + 1e-5) as f64,
                    );
                    let cc_odom_a_shifted = current_cloud_in_base
                        .clone()
                        .transform_by_mat(cp_transform.to_homogeneous());

                    let (ev, vnum) = calc_cost(current_cloud_in_odom.clone(), &nearests);
                    if i == 0 && j == 0 {
                        self.initial_cost = Some(ev);
                    }

                    let dx = (calc_cost(cc_odom_x_shifted, &nearests).0 - ev) / 1e-5;
                    let dy = (calc_cost(cc_odom_y_shifted, &nearests).0 - ev) / 1e-5;
                    let da = (calc_cost(cc_odom_a_shifted, &nearests).0 - ev) / 1e-5;

                    let sm_cost = ScanMatchingCost {
                        initial_transform: local_final_transform,
                        derivative: nalgebra::Isometry2::new(nalgebra::Vector2::new(dx, dy), da),
                        pointcloud: current_cloud_in_base.clone(),
                        nearests: &nearests,
                    };
                    let solver = argmin::solver::brent::BrentOpt::new(-2., 2.);

                    let res = argmin::core::Executor::new(sm_cost, solver)
                        .configure(|state| state.max_iters(40))
                        //.add_observer(SlogLogger::term(), ObserverMode::Always)
                        .run()
                        .unwrap();

                    //println!("Result of brent:\n{res}");
                    let kk = res.state.best_param.unwrap_or_default();
                    //let kk = 1e-3;
                    let new_x = local_final_transform.translation.x - kk * dx;
                    let new_y = local_final_transform.translation.y - kk * dy;
                    let new_a = local_final_transform.rotation.angle() - kk * da;
                    let new_transform =
                        nalgebra::Isometry2::new(nalgebra::Vector2::new(new_x, new_y), new_a);

                    let current_cloud_in_odom_new = current_cloud_in_base
                        .clone()
                        .transform_by_mat(new_transform.to_homogeneous());
                    let (new_cost, _) = calc_cost(current_cloud_in_odom_new, &nearests);
                    log::debug!(
                        "i: {}, j: {}, cost: {} vnum: {}, new cost: {}",
                        i,
                        j,
                        ev,
                        vnum,
                        new_cost
                    );
                    if new_cost < local_best_cost {
                        local_final_transform = new_transform;
                        local_best_cost = new_cost;
                    }
                    if (local_old_cost - new_cost).abs() <= 1e-6 {
                        break;
                    }
                    local_old_cost = new_cost;
                }
                if local_best_cost < best_cost {
                    best_cost = local_best_cost;
                    final_transform = local_final_transform;
                }
                if (old_cost - local_best_cost).abs() <= 1e-6 {
                    break;
                }
                old_cost = local_best_cost;
            }
        }

        // add aligned pointcloud to the map
        self.map.add_point_cloud(
            timestamp,
            current_cloud_in_base.transform_by_mat(final_transform.to_homogeneous()),
        );

        self.optimized_cost = Some(best_cost);

        // print debug
        if better {
            log::debug!("pc map: clouds {}", self.map.clouds.len());
            log::debug!(
                "cost before {:?}, after: {:?}",
                self.initial_cost,
                self.optimized_cost
            );
        }

        final_transform
    }
}

pub struct PointCloudSLAM {
    matching: PointCloudMatching,
    current_pose: nalgebra::Isometry2<f64>,
}

impl PointCloudSLAM {
    pub fn new() -> Self {
        Self {
            matching: PointCloudMatching::new(),
            current_pose: nalgebra::Isometry2::new(
                nalgebra::Vector2::new(0.0, 0.0),
                std::f64::consts::PI,
            ),
        }
    }

    pub fn process_measurement(
        &mut self,
        _: usize,
        current_timestamp: f64,
        measurement: &lsc_reader::Measurement,
        better: bool,
    ) -> h_analyzer_data::Entity {
        if better {
            log::debug!("odom: {:?}", measurement.odometry);
            log::debug!("lidar point num: {}", measurement.lidar.points.len());
        }

        if let Some(rmotion) = measurement.relative_motion {
            self.current_pose.mul_assign(rmotion);
        } else {
            self.current_pose = nalgebra::Isometry2::new(
                nalgebra::Vector2::new(
                    measurement.odometry.x as f64,
                    measurement.odometry.y as f64,
                ),
                (measurement.odometry.theta) as f64,
            );
        }

        let angle_offset = std::f64::consts::PI; // between lidar and ego front
        let baselink_to_odom = nalgebra::Isometry2::from(self.current_pose);
        let lidar_to_baselink = nalgebra::Isometry2::new(nalgebra::Vector2::zeros(), angle_offset);
        let lidar_to_odom = baselink_to_odom * lidar_to_baselink;
        let raw_points_in_odom = PointCloud::from_vec_of_points(&measurement.lidar.points)
            .transform_by_mat(lidar_to_odom.to_homogeneous());
        let int_points_in_base =
            PointCloud::from_vec_of_points(&measurement.lidar.interpolated_points);
        let int_points_in_odom = int_points_in_base
            .clone()
            .transform_by_mat(lidar_to_odom.to_homogeneous());

        let final_transform = self.matching.process_current_cloud(
            current_timestamp,
            &lidar_to_odom,
            int_points_in_base,
            better,
        );

        //log::debug!("current: {:?}", self.current_pose);
        //log::debug!("initial: {:?}", lidar_to_odom);
        //log::debug!("final  : {:?}", final_transform);

        self.current_pose = final_transform * lidar_to_baselink.inverse();

        //log::debug!("new current  : {:?}", self.current_pose);

        // send world frame
        let mut ego = h_analyzer_data::Entity::new();
        ego.add_estimate(
            "odometry".to_string(),
            h_analyzer_data::Estimate::Pose2D(h_analyzer_data::Pose2D::new(
                measurement.odometry.x as f64,
                measurement.odometry.y as f64,
                measurement.odometry.theta as f64,
            )),
        );
        ego.add_measurement(
            "lidar".to_string(),
            h_analyzer_data::Measurement::PointCloud2D(raw_points_in_odom.to_h_pointcloud_2d()),
        );
        ego.add_measurement(
            "lidar_int".to_string(),
            h_analyzer_data::Measurement::PointCloud2D(int_points_in_odom.to_h_pointcloud_2d()),
        );

        ego.add_measurement(
            "reference_map".to_string(),
            h_analyzer_data::Measurement::PointCloud2D(
                self.matching.map.entire_map_cloud.to_h_pointcloud_2d(),
            ),
        );
        ego
    }
}

struct LittleSLAMData {
    measurements: Vec<lsc_reader::Measurement>,
}

impl LittleSLAMData {
    fn new() -> LittleSLAMData {
        let measurements =
            lsc_reader::load_lsc_file("/home/haruki/Works/datasets/little_slam/hall.lsc");
        LittleSLAMData {
            measurements: measurements.unwrap(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use static_init::dynamic;

    #[dynamic(drop)]
    static mut RES: LittleSLAMData = LittleSLAMData::new();

    #[tokio::test]
    async fn scan_matching() {
        env_logger::init();
        std::thread::sleep(std::time::Duration::from_millis(500));
        log::debug!("test");
        let binding = RES.read();

        let mut cl = h_analyzer_client_lib::HAnalyzerClient::new("http://localhost:50051").await;
        cl.register_new_world(&"scan_matching".to_string())
            .await
            .unwrap();

        let mut pm = PointCloudMatching::new();
        for i in 0..2 {
            let measurement = binding.measurements.get(i).clone().unwrap();
            let odom_trfm = nalgebra::Isometry2::new(
                nalgebra::Vector2::new(measurement.odometry.x, measurement.odometry.y),
                measurement.odometry.theta,
            );
            println!("time {}, {}", measurement.time, odom_trfm);
            let int_points_in_base =
                PointCloud::from_vec_of_points(&measurement.lidar.interpolated_points);

            let trfm = pm.process_current_cloud(
                measurement.time,
                &odom_trfm,
                int_points_in_base.clone(),
                true,
            );
            //            let trfm = measurement.relative_motion.unwrap_or_default();

            println!(
                "before {}, after: {}",
                pm.initial_cost.unwrap().sqrt(),
                pm.optimized_cost.unwrap().sqrt()
            );
            println!("trfm: {}", trfm);

            let mut wf = h_analyzer_data::WorldFrame::new(i, measurement.time);
            let mut ego = h_analyzer_data::Entity::new();
            ego.add_measurement(
                "lidar_int".to_string(),
                h_analyzer_data::Measurement::PointCloud2D(
                    int_points_in_base
                        .clone()
                        .transform_by_mat(odom_trfm.to_homogeneous())
                        .to_h_pointcloud_2d(),
                ),
            );
            ego.add_measurement(
                "aligned".to_string(),
                h_analyzer_data::Measurement::PointCloud2D(
                    int_points_in_base
                        .clone()
                        .transform_by_mat(trfm.to_homogeneous())
                        .to_h_pointcloud_2d(),
                ),
            );
            ego.add_measurement(
                "reference_map".to_string(),
                h_analyzer_data::Measurement::PointCloud2D(
                    pm.map.entire_map_cloud.to_h_pointcloud_2d(),
                ),
            );
            wf.add_entity("ego".to_string(), ego);

            cl.send_world_frame(wf).await.unwrap();

            if i > 0 {
                assert_eq!(pm.optimized_cost < pm.initial_cost, true);
            }
        }
    }
}
