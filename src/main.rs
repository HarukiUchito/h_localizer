use std::io::BufRead;

#[derive(Debug)]
pub struct Odometry {
    x: f32,
    y: f32,
    theta: f32,
}

#[derive(Clone, Debug, Copy)]
pub struct Point {
    pub x: f32,
    pub y: f32,
    theta: f32,
    range: f32,
}

fn distance(p1: Point, p2: Point) -> f32 {
    let dx = p1.x - p2.x;
    let dy = p1.y - p2.y;
    f32::sqrt(dx * dx + dy * dy)
}

fn angular_mid(p1: Point, p2: Point, theta: f32) -> Point {
    let a = (p1.y - p2.y) / (p1.x - p2.x);
    let b = p1.y - a * p1.x;
    let tant = f32::tan(theta);
    let mx = b / (tant - a);
    let my = tant * mx;
    Point {
        x: mx,
        y: my,
        theta: theta,
        range: f32::sqrt(mx * mx + my * my),
    }
}

#[derive(Debug)]
pub struct LiDAR {
    thetas: Vec<f32>,
    ranges: Vec<f32>,
    pub points: Vec<Point>,
    pub interpolated_points: Vec<Point>,
}

impl LiDAR {
    pub fn reset_xy(self: &mut LiDAR) {
        self.points.clear();
        for i in 0..self.thetas.len() {
            let th = self.thetas[i];
            let rn = self.ranges[i];
            if rn < 0.1 || rn > 6.0 {
                continue;
            }
            let x = rn * th.cos();
            let y = rn * th.sin();
            self.points.push(Point {
                x: x,
                y: y,
                theta: th,
                range: rn,
            });
            //log::debug!("th: {}, rn: {}, x: {}, y: {}", th, rn, x, y);
        }
    }
    pub fn interpolate(self: &mut LiDAR) {
        self.interpolate_verbose(false);
    }
    pub fn interpolate_verbose(self: &mut LiDAR, verbose: bool) {
        self.interpolated_points.clear();
        let mut iter = self.points.iter().peekable();

        let mut last_th;

        // always add first point
        let mut current_p = match iter.next().cloned() {
            Some(firstpoint) => {
                let fp = firstpoint.clone();
                //self.interpolated_points.push(fp);
                last_th = fp.theta;
                fp
            }
            None => return, // zero points
        };
        let mut last_original_p = current_p;

        let dangle = f32::to_radians(-1.0);
        let mut nangle = self.thetas.first().unwrap() + dangle;
        if verbose {
            log::debug!("first nangle: {}", f32::to_degrees(nangle));
        }
        let mut cnt = 0;
        loop {
            let mut next_p: Point;
            let next_p = loop {
                next_p = match iter.peek().cloned() {
                    Some(np) => np.clone(),
                    None => break None, // if no more next point
                };
                if next_p.theta < nangle {
                    break Some(next_p);
                }
                iter.next();
            };
            let next_p = match next_p {
                Some(np) => np,
                None => break,
            };
            let mid_p = angular_mid(current_p, next_p, nangle);
            let mc_dist = distance(mid_p, last_original_p);
            let mn_dist = distance(mid_p, next_p);
            const DIS_THRESHOLD: f32 = 0.05;
            const DEG_THRESHOLD: f32 = 0.5;
            let ddeg = f32::to_degrees(last_th - mid_p.theta).abs();
            if (mc_dist <= DIS_THRESHOLD || mn_dist <= DIS_THRESHOLD) && ddeg >= DEG_THRESHOLD {
                self.interpolated_points.push(mid_p);
                last_th = mid_p.theta;
                current_p = mid_p;
            }

            let lesthan = cnt < 5;
            if verbose && lesthan {
                log::debug!(
                    "\nnangle: {}, cx: {}, cy: {}, cth: {}\n nx: {}, ny: {}, nth: {}",
                    f32::to_degrees(nangle),
                    current_p.x,
                    current_p.y,
                    f32::to_degrees(current_p.theta),
                    next_p.x,
                    next_p.y,
                    f32::to_degrees(next_p.theta)
                );
            }

            last_original_p = next_p;
            nangle += dangle; // update next angle

            if verbose && lesthan {
                log::debug!(
                    "mx: {}, my: {}, mth: {}\n c-dist: {}, n-dist: {}, mth2: {}, ddeg: {}",
                    mid_p.x,
                    mid_p.y,
                    f32::to_degrees(mid_p.theta),
                    mc_dist,
                    mn_dist,
                    f32::to_degrees(mid_p.y.atan2(mid_p.x)),
                    ddeg
                );

                log::debug!("isize: {}", self.interpolated_points.len());
            }
            cnt += 1;
        }
    }
}

#[derive(Debug)]
pub struct Measurement {
    pub time: f64,
    pub odometry: Odometry,
    pub lidar: LiDAR,
}

pub fn parse_line(line: String) -> Option<Measurement> {
    let mut tokens = line.split_ascii_whitespace().collect::<Vec<&str>>();
    //log::debug!("tokens {:?}", tokens);
    log::debug!("tokens len {}", tokens.len());

    let tlen = tokens.len();
    //log::debug!("token num: {}", tlen);
    if tlen < 5 {
        return None;
    }

    let ltype = tokens[0];
    if ltype != "LASERSCAN" {
        return None;
    }
    // let id = tokens[1].parse::<i32>().unwrap();
    let sec = tokens[2].parse::<i32>().unwrap();
    let nsec = tokens[3].parse::<i32>().unwrap();
    let fsec = sec as f64 + (nsec as f64 * 1e-9);

    let odom_x = tokens[tlen - 5].parse::<f32>().unwrap();
    let odom_y = tokens[tlen - 4].parse::<f32>().unwrap();
    let odom_t = tokens[tlen - 3].parse::<f32>().unwrap();

    let pnum = tokens[4].parse::<i32>().unwrap();
    let rest = tlen as i32 - 2 * pnum;
    /*log::debug!(
        "sec: {}, nsec: {}, nsec9: {}, time: {}",
        sec,
        nsec,
        nsec as f32 * 1e-9,
        fsec
    );*/
    if rest < 8 {
        return None;
    }

    let idx = tokens.len() - 5;
    //log::debug!("from: {}", idx);
    tokens.drain(idx..(idx + 5));
    //log::debug!("last token: {}", tokens.last().unwrap());
    tokens.drain(0..5); // remove until pnum
                        //log::debug!("first token: {}", tokens.first().unwrap());
    log::debug!("drained {}, pnum {}", tokens.len(), pnum);

    let thetas = tokens
        .iter()
        .step_by(2)
        .map(|&s| s.parse::<f32>().unwrap().to_radians())
        .collect::<Vec<f32>>();
    //log::debug!("{:?}", thetas);

    // let theta_ds = thetas.windows(2).map(|x| x[0] - x[1]).collect::<Vec<f32>>();

    tokens.drain(0..1); // remove first theta

    let ranges = tokens
        .iter()
        .step_by(2)
        .map(|&s| s.parse::<f32>().unwrap())
        .collect::<Vec<f32>>();

    Some(Measurement {
        time: fsec,
        odometry: Odometry {
            x: odom_x,
            y: odom_y,
            theta: odom_t,
        },
        lidar: LiDAR {
            thetas: thetas,
            ranges: ranges,
            points: Vec::new(),
            interpolated_points: Vec::new(),
        },
    })
}

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

    fn from_vec_of_points(points: &Vec<Point>) -> Self {
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

    fn to_h_pointcloud_2d(&self) -> h_analyzer_data::PointCloud2D {
        let (x_vec, y_vec) = self.to_x_y_vec();
        h_analyzer_data::PointCloud2D::new(x_vec, y_vec)
    }
}

use nalgebra::Matrix3x1;
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

        let calcCost = |pointcloud: PointCloud, nearests: &Vec<Option<Matrix3x1<f64>>>| {
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

                let (ev, vnum) = calcCost(current_cloud_in_odom, &nearests);
                let dx = (calcCost(cc_odom_x_shifted, &nearests).0 - ev) / 1e-5;
                let dy = (calcCost(cc_odom_y_shifted, &nearests).0 - ev) / 1e-5;
                let da = (calcCost(cc_odom_a_shifted, &nearests).0 - ev) / 1e-5;
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
                let (new_cost, _) = calcCost(current_cloud_in_odom_new, &nearests);
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
        measurement: &Measurement,
        better: bool,
    ) -> h_analyzer_data::Entity {
        if better {
            log::debug!("odom: {:?}", measurement.odometry);
            log::debug!("lidar point num: {}", measurement.lidar.points.len());
        }

        let angle_offset = std::f32::consts::PI; // between lidar and ego front
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
    let mut measurements: Vec<Measurement> = Vec::new();
    for line in std::io::BufReader::new(file).lines() {
        match parse_line(line.unwrap()) {
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
