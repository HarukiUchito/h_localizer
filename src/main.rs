use std::io::BufRead;
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

fn voxel_grid_filter(input: &PointCloud) -> PointCloud {
    let mut voxels = std::collections::HashMap::new();
    for v in input.matrix.column_iter() {
        if let (Some(&px), Some(&py)) = (v.get(0), v.get(1)) {
            let grid_size = 0.1;
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
            log::debug!("ts {} inner {}", timestamp * 1e6, inner_time);
            log::debug!("last {}", self.clouds.last_key_value().unwrap().0);
        }
        if self.clouds.is_empty()
            || (inner_time - self.clouds.last_key_value().unwrap().0 >= self.minimum_time_diff)
        {
            self.clouds.insert(inner_time, cloud);
        }
        self.entire_map_cloud = voxel_grid_filter(&self.entire_map());
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
            .map(|v| ((v - query_point).transpose() * (v - query_point)).sum())
            .collect();
        let (min_index, &min_dist) = distances
            .iter()
            .enumerate()
            .min_by(|(_, &a), (_, &b)| a.total_cmp(&b))?;
        if min_dist > 0.2 {
            //return None;
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

use h_localizer::lsc_reader;
use tokio::time::{sleep_until, Duration, Instant};

struct PointCloudMatching {
    pub map: PointCloudMap,
}

impl PointCloudMatching {
    fn new() -> Self {
        Self {
            map: PointCloudMap::new(),
        }
    }

    fn point_to_point_cost(p1: nalgebra::Matrix3x1<f64>, p2: nalgebra::Matrix3x1<f64>) -> f64 {
        let dp = p1 - p2;
        (dp.transpose() * dp).sum()
    }

    fn process_current_cloud(
        &mut self,
        timestamp: f64,
        initial_transform: &nalgebra::Isometry2<f64>,
        current_cloud_in_base: PointCloud,
        better: bool,
    ) -> nalgebra::Isometry2<f64> {
        if better {
            log::debug!("\n[PointCloudMatching process]");
        }
        let mut final_transform = initial_transform.clone();

        let calc_cost = |pointcloud: PointCloud,
                         nearests: &Vec<Option<nalgebra::Matrix3x1<f64>>>| {
            let mut cost = 0.0;
            let pnum = pointcloud.matrix.shape().1;
            let mut valid_num = 0;
            for i in 0..pnum {
                if let Some(np) = nearests[i] {
                    let p = pointcloud.matrix.column(i);
                    cost += PointCloudMatching::point_to_point_cost(np, p.into());
                    valid_num += 1;
                }
            }
            (cost, valid_num)
        };

        if better {
            let mut best_cost = f64::MAX;
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

                // slightly shifted current points
                let mut cp_transform = initial_transform.clone();
                cp_transform.translation.x += 1e-5;
                let cc_odom_x_shifted = current_cloud_in_base
                    .clone()
                    .transform_by_mat(cp_transform.to_homogeneous());
                let mut cp_transform = initial_transform.clone();
                cp_transform.translation.y += 1e-5;
                let cc_odom_y_shifted = current_cloud_in_base
                    .clone()
                    .transform_by_mat(cp_transform.to_homogeneous());
                let cp_transform = nalgebra::Isometry2::new(
                    initial_transform.translation.vector,
                    (initial_transform.rotation.angle() + 1e-5) as f64,
                );
                let cc_odom_a_shifted = current_cloud_in_base
                    .clone()
                    .transform_by_mat(cp_transform.to_homogeneous());

                let (ev, vnum) = calc_cost(current_cloud_in_odom, &nearests);
                let dx = (calc_cost(cc_odom_x_shifted, &nearests).0 - ev) / 1e-5;
                let dy = (calc_cost(cc_odom_y_shifted, &nearests).0 - ev) / 1e-5;
                let da = (calc_cost(cc_odom_a_shifted, &nearests).0 - ev) / 1e-5;
                let new_x = initial_transform.translation.x - 1e-6 * dx;
                let new_y = initial_transform.translation.y - 1e-6 * dy;
                let new_a = initial_transform.rotation.angle() - 1e-6 * da;
                /*log::debug!(
                    "before cost: {}, x: {}, y: {}, a: {}",
                    ev,
                    initial_transform.translation.x,
                    initial_transform.translation.y,
                    initial_transform.rotation.angle()
                );*/
                //log::debug!(" after x: {}, y: {}, a: {}", new_x, new_y, new_a);
                let new_transform =
                    nalgebra::Isometry2::new(nalgebra::Vector2::new(new_x, new_y), new_a);

                let current_cloud_in_odom_new = current_cloud_in_base
                    .clone()
                    .transform_by_mat(new_transform.to_homogeneous());
                let (new_cost, _) = calc_cost(current_cloud_in_odom_new, &nearests);
                log::debug!(
                    "i: {}, cost: {} vnum: {}, new cost: {}",
                    i,
                    ev,
                    vnum,
                    new_cost
                );
                if new_cost < best_cost {
                    final_transform = new_transform;
                    best_cost = new_cost;
                    if best_cost - new_cost <= 1e-6 {
                        break;
                    }
                }
            }
        }

        // add aligned pointcloud to the map
        self.map.add_point_cloud(
            timestamp,
            current_cloud_in_base.transform_by_mat(final_transform.to_homogeneous()),
        );
        if better {
            log::debug!("pc map: clouds {}", self.map.clouds.len());
        }
        final_transform
    }
}

struct PointCloudSLAM {
    matching: PointCloudMatching,
}

impl PointCloudSLAM {
    fn new() -> Self {
        Self {
            matching: PointCloudMatching::new(),
        }
    }

    fn process_measurement(
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

        let angle_offset = std::f64::consts::PI; // between lidar and ego front
        let baselink_to_odom = nalgebra::Isometry2::new(
            nalgebra::Vector2::new(measurement.odometry.x as f64, measurement.odometry.y as f64),
            (measurement.odometry.theta + angle_offset) as f64,
        );
        let raw_points_in_odom = PointCloud::from_vec_of_points(&measurement.lidar.points)
            .transform_by_mat(baselink_to_odom.to_homogeneous());
        let int_points_in_base =
            PointCloud::from_vec_of_points(&measurement.lidar.interpolated_points);
        let int_points_in_odom = int_points_in_base
            .clone()
            .transform_by_mat(baselink_to_odom.to_homogeneous());

        self.matching.process_current_cloud(
            current_timestamp,
            &baselink_to_odom,
            int_points_in_base,
            better,
        );

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

#[tokio::main]
async fn main() {
    env_logger::init();
    // open and read lsc file
    let path = "/home/haruki/data/little_slam_dataset/hall.lsc";
    log::debug!("path: {}", path);
    let path = std::path::Path::new(&path);
    let file = match std::fs::File::open(&path) {
        Err(why) => panic!("couldn't open {}: {}", path.display(), why),
        Ok(file) => file,
    };

    let mut cnt = 0;
    let mut measurements: Vec<lsc_reader::Measurement> = Vec::new();
    for line in std::io::BufReader::new(file).lines() {
        match lsc_reader::parse_line(line.unwrap(), measurements.last()) {
            Some(mut measurement) => {
                measurement.lidar.reset_xy();
                measurement.lidar.interpolate();
                measurements.push(measurement);
            }
            None => {
                log::debug!("failed reading line {} as an measurement", cnt);
            }
        }
        cnt = cnt + 1;
        if cnt > 0 {
            //return;
        }
    }

    let mut cl = h_analyzer_client_lib::HAnalyzerClient::new().await;
    cl.register_new_world(&"slam".to_string()).await.unwrap();

    let mut slam = PointCloudSLAM::new();
    let mut slam_better = PointCloudSLAM::new();

    let first_timestamp = measurements[0].time;
    let first_time = Instant::now();
    for i in 0..measurements.len() {
        let measurement = &measurements[i];
        let current_timestamp = measurement.time;
        let time_diff = current_timestamp - first_timestamp;

        //log::debug!("check next {}\nts {}", i, current_timestamp,);
        sleep_until(first_time + Duration::from_secs_f64(time_diff)).await;

        let mut wf = h_analyzer_data::WorldFrame::new(i, current_timestamp);
        wf.add_entity(
            "opt_ego".to_string(),
            slam_better.process_measurement(i, current_timestamp, measurement, true),
        );
        wf.add_entity(
            "ego".to_string(),
            slam.process_measurement(i, current_timestamp, measurement, false),
        );

        cl.send_world_frame(wf).await.unwrap();

        //break;
    }
}
