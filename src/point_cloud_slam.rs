use nalgebra::zero;
use nalgebra::{self as na, MatrixXx2};

use crate::lsc_reader;

use std::thread::current;
use std::{any::Any, default, ops::MulAssign};
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

/// Returns 2D rigid transformation matrix 
/// from 2D translation vector and theta angle in radians (x, y, theta)
#[rustfmt::skip]
fn trfm_mat_2d(v: &na::Vector3<f64>) -> na::Matrix3<f64> {
    let x = v[0];
    let y = v[1];
    let t = v[2];
    na::Matrix3::<f64>::from_row_slice(&[
        t.cos(), -t.sin(),   x,
        t.sin(),  t.cos(),   y,
            0.0,      0.0, 1.0,
    ])
}

fn trfm_parameter_vec(m: &na::Matrix3<f64>) -> na::Vector3<f64> {
    na::Vector3::<f64>::from_vec(vec![m[(0, 2)], m[(1, 2)], m[(1, 0)].atan2(m[(0, 0)])])
}

/// Applies rigid transformation specified by t_mat to the specified 2D pointcloud point_mat
/// Each row of point_mat must represensts each point.
fn rigid_transform_2d(
    point_mat: &na::MatrixXx2<f64>,
    t_mat: &na::Matrix3<f64>,
) -> na::MatrixXx2<f64> {
    let (_, r_mat) = extract_matrix(t_mat);
    let mut rotated = (r_mat * point_mat.transpose()).transpose();
    let t_vec = t_mat.fixed_view::<2, 1>(0, 2).transpose();
    rotated.row_iter_mut().for_each(|mut row| row += t_vec);
    rotated
}

fn extract_matrix(t_mat: &na::Matrix3<f64>) -> (na::Matrix2x1<f64>, na::Matrix2<f64>) {
    (
        t_mat.fixed_view::<2, 1>(0, 2).into(), // translation part
        t_mat.fixed_view::<2, 2>(0, 0).into(), // rotation part
    )
}

fn plot_points(rec: &rerun::RecordingStream, name: &str, points: &na::MatrixXx2<f64>) {
    rec.log(
        name,
        &rerun::Points3D::new(
            points
                .row_iter()
                .map(|row| (row[0] as f32, row[1] as f32, 0.0)),
        )
        .with_radii((0..points.shape().0).map(|_| rerun::Radius(rerun::Float32(0.05)))),
    )
    .unwrap();
}

trait ErrorGradient {
    fn number_of_term(&self) -> usize;
    fn residual_length(&self) -> usize;
    fn residual(&self, parameter: &na::Matrix3<f64>, index: usize) -> na::DMatrix<f64>;
    fn jacobian(&self, parameter: &na::Matrix3<f64>, index: usize) -> na::DMatrix<f64>;
}

struct ScanMatchingCost2D {
    source_points: na::MatrixXx2<f64>,
    target_points: na::MatrixXx2<f64>,
}

impl ErrorGradient for ScanMatchingCost2D {
    fn number_of_term(&self) -> usize {
        self.source_points.shape().0
    }
    fn residual_length(&self) -> usize {
        3 // x, y, theta
    }
    fn residual(&self, t_mat: &na::Matrix3<f64>, index: usize) -> na::DMatrix<f64> {
        let src_point = self.source_points.row(index);
        let tgt_point = self.target_points.row(index);

        //println!("tgt {}", tgt_point);
        //println!("src {}", src_point);

        let mut vec = na::MatrixXx2::zeros(1);
        vec.copy_from(&tgt_point);
        let transformed = rigid_transform_2d(&vec, t_mat);

        let mut ret = na::DMatrix::<f64>::zeros(1, 2);
        ret.copy_from(&(transformed.clone() - src_point));

        ret.transpose()
    }
    fn jacobian(&self, t_mat: &na::Matrix3<f64>, index: usize) -> na::DMatrix<f64> {
        let (_, r_mat) = extract_matrix(t_mat);

        let p_vec = self.target_points.fixed_view::<1, 2>(index, 0);
        let mat = na::DMatrix::<f64>::from_row_slice(
            2,
            3,
            &[
                1.0, 0.0, -p_vec[1], //
                0.0, 1.0, p_vec[0],
            ],
        );
        let mut ret = na::DMatrix::<f64>::zeros(2, 3);
        ret.copy_from(&(r_mat * mat));
        ret
    }
}

struct GaussNewton {
    gradient: Box<dyn ErrorGradient>,
}

impl GaussNewton {
    fn optimize(&self, initial_parameter: &na::DMatrix<f64>) -> na::Vector3<f64> {
        let parameter = initial_parameter.to_owned();
        let mut current_t_mat = trfm_mat_2d(&parameter.fixed_view::<3, 1>(0, 0).into());

        let n = self.gradient.residual_length();

        for _ in 0..5 {
            let mut cost = 0.0;
            let mut g = na::MatrixXx1::zeros(n);
            let mut h = na::DMatrix::<f64>::zeros(n, n);
            for i in 0..self.gradient.number_of_term() {
                let r_i = self.gradient.residual(&current_t_mat, i);
                let j_i = self.gradient.jacobian(&current_t_mat, i);
                //println!("i {}", i);
                //println!("r_i {}", r_i);
                //println!("j_i {}", j_i);

                let e2 = (r_i.transpose() * r_i.clone())[(0, 0)];
                let delta = 1e-1;
                let c2 = delta * delta;
                let c2_inv = 1.0 / c2;
                let aux = c2_inv * e2 + 1.0;
                let rho0 = c2 * aux.ln();
                let rho1 = 1.0 / aux;

                g += j_i.transpose() * r_i.clone(); // * rho1;
                h += j_i.transpose() * j_i; // * rho1;
                cost += e2; //rho0;
            }
            //println!("g {}", g.transpose());
            //println!("H {}", h);
            println!("cost {}", cost);
            println!(
                "current T vec {}",
                trfm_parameter_vec(&current_t_mat).transpose()
            );

            let dx = -h.try_inverse().unwrap() * g;
            //println!("dx {}", dx.transpose());
            current_t_mat *= trfm_mat_2d(&(dx.fixed_view::<3, 1>(0, 0).into()));
        }

        trfm_parameter_vec(&current_t_mat)
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
            minimum_time_diff: (1e6 * 1e-3) as u64, // 0.1s
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
        log::info!(
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
        if min_dist > (0.1 * 0.1) {
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
    pub initial_cost: Option<f64>,
    pub optimized_cost: Option<f64>,
}

impl PointCloudMatching {
    fn new() -> Self {
        Self {
            map: PointCloudMap::new(),
            initial_cost: None,
            optimized_cost: None,
        }
    }

    pub fn icp_lie_gauss_newton(
        &mut self,
        initial_transform: &nalgebra::Isometry2<f64>,
        current_cloud_in_base: PointCloud,
    ) -> nalgebra::Isometry2<f64> {
        let current_cloud_in_odom = current_cloud_in_base
            .clone()
            .transform_by_mat(initial_transform.to_homogeneous());
        // data association
        let mut map_points = na::MatrixXx2::<f64>::zeros(current_cloud_in_base.matrix.shape().1);
        let mut idx = 0;
        for p in current_cloud_in_odom.matrix.column_iter() {
            let nearest_opt = self.map.find_nearest_point(&p.into());
            if let Some(nearest) = nearest_opt {
                map_points[(idx, 0)] = nearest[0];
                map_points[(idx, 1)] = nearest[1];
            } else {
                map_points[(idx, 0)] = p[0];
                map_points[(idx, 1)] = p[1];
            }
            idx += 1;
        }

        let scan_2d = ScanMatchingCost2D {
            target_points: current_cloud_in_base
                .matrix
                .transpose()
                .fixed_columns::<2>(0)
                .into(),
            source_points: map_points,
        };

        let param = na::DMatrix::from_vec(
            3,
            1,
            vec![
                initial_transform.translation.x,
                initial_transform.translation.y,
                initial_transform.rotation.angle(),
            ],
        );
        let gn = GaussNewton {
            gradient: Box::new(scan_2d),
        };
        let final_param = gn.optimize(&param);

        self.initial_cost = Some(1.0);
        self.optimized_cost = Some(0.0);

        nalgebra::Isometry2::new(
            nalgebra::Vector2::new(final_param.x, final_param.y),
            final_param.z,
        )
    }

    pub fn icp_newton(
        &mut self,
        initial_transform: &nalgebra::Isometry2<f64>,
        current_cloud_in_base: PointCloud,
    ) -> nalgebra::Isometry2<f64> {
        let mut final_transform = initial_transform.clone();

        let mut best_cost = f64::MAX;
        let mut old_cost = f64::MAX;
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
            for j in 0..1 {
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
                    .configure(|state| state.max_iters(10))
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
        if best_cost != f64::MAX {
            self.optimized_cost = Some(best_cost);
        }
        final_transform
    }

    pub fn process_current_cloud(
        &mut self,
        timestamp: f64,
        initial_transform: &nalgebra::Isometry2<f64>,
        current_cloud_in_base: PointCloud,
    ) -> nalgebra::Isometry2<f64> {
        let current_cloud_in_base = voxel_grid_filter(&current_cloud_in_base, 0.1);
        if self.map.clouds.len() == 0 {
            //println!("map trfm {}", initial_transform);
            self.map.add_point_cloud(
                timestamp,
                current_cloud_in_base.transform_by_mat(initial_transform.to_homogeneous()),
            );
            return initial_transform.clone();
        }

        log::info!("\n[PointCloudMatching process]");
        let final_transform =
            self.icp_lie_gauss_newton(initial_transform, current_cloud_in_base.clone());

        // add aligned pointcloud to the map
        self.map.add_point_cloud(
            timestamp,
            current_cloud_in_base.transform_by_mat(final_transform.to_homogeneous()),
        );

        // print debug
        log::info!("pc map: clouds {}", self.map.clouds.len());
        log::info!(
            "cost before {:?}, after: {:?}",
            self.initial_cost,
            self.optimized_cost
        );

        final_transform
    }
}

struct Odometry2D {
    pub velocity: f64,
    pub yaw_rate: f64,
    pub current_covariance: nalgebra::Matrix3<f64>,
}

impl Odometry2D {
    fn new() -> Self {
        Self {
            velocity: 0.0,
            yaw_rate: 0.0,
            current_covariance: nalgebra::Matrix3::zeros(),
        }
    }
    fn update(&mut self, dt: f64, current_motion: &nalgebra::Isometry2<f64>) {
        let mx = current_motion.translation.x;
        let my = current_motion.translation.y;
        let mt = current_motion.rotation.angle();
        let distance = (mx * mx + my * my).sqrt();

        self.velocity = distance / dt;
        self.yaw_rate = mt / dt;

        log::info!("velocity: {}, yaw rate: {}", self.velocity, self.yaw_rate);
        let vel_ll = 0.001;
        let yaw_ll = 0.01;
        if self.velocity < vel_ll {
            self.velocity = vel_ll;
        }
        if self.yaw_rate < yaw_ll {
            self.yaw_rate = yaw_ll;
        }

        let c_v = 1.0;
        let c_a = 5.0;
        let u_mat = nalgebra::Matrix2::new(
            c_v * self.velocity * self.velocity,
            0.0,
            0.0,
            c_a * self.yaw_rate * self.yaw_rate,
        );

        // rotate covariance
        let cs = mt.cos();
        let sn = mt.sin();

        let jxt_mat = nalgebra::Matrix3::new(
            1.0,
            0.0,
            -distance * sn,
            0.0,
            1.0,
            distance * cs,
            0.0,
            0.0,
            1.0,
        );

        let jut_mat = nalgebra::Matrix3x2::new(dt * cs, 0.0, dt * sn, 0.0, 0.0, dt);

        self.current_covariance = jxt_mat * self.current_covariance * jxt_mat.transpose()
            + jut_mat * u_mat * jut_mat.transpose();

        //log::info!("cov {}", self.current_covariance);
    }
}

pub trait LogHeader<'a> {
    const LOG_HEADER: &'a str;
}

pub struct PointCloudSLAM {
    odometry: Odometry2D,
    matching: PointCloudMatching,
    current_pose: nalgebra::Isometry2<f64>,
    last_pose: Option<nalgebra::Isometry2<f64>>,

    last_timestamp_opt: Option<f64>,
    pub log_str_csv: String,
}

impl LogHeader<'_> for PointCloudSLAM {
    const LOG_HEADER: &'static str = "timestamp[s],cost,velocity[m/s],yaw_rate[rad/s]";
}

fn plot_points_org(rec: &rerun::RecordingStream, name: &str, points: &PointCloud) {
    rec.log(
        name,
        &rerun::Points3D::new(
            points
                .matrix
                .column_iter()
                .map(|p| (p[(0, 0)] as f32, p[(1, 0)] as f32, 0.0)),
        )
        .with_radii((0..points.matrix.shape().1).map(|_| rerun::Radius(rerun::Float32(0.05)))),
    )
    .unwrap();
}

impl PointCloudSLAM {
    pub fn new() -> Self {
        Self {
            odometry: Odometry2D::new(),
            matching: PointCloudMatching::new(),
            current_pose: nalgebra::Isometry2::new(
                nalgebra::Vector2::new(0.0, 0.0),
                std::f64::consts::PI,
            ),
            last_pose: None,
            last_timestamp_opt: None,
            log_str_csv: "".to_string(),
        }
    }

    pub fn process_measurement(
        &mut self,
        _: usize,
        current_timestamp: f64,
        measurement: &lsc_reader::Measurement,
        better: bool,
        rec: Option<&rerun::RecordingStream>,
    ) {
        if better {
            log::info!("odom: {:?}", measurement.odometry);
            log::info!("lidar point num: {}", measurement.lidar.points.len());
        }

        // wheel odometry update
        if let (Some(last_timestamp), Some(last_pose), Some(rmotion)) = (
            self.last_timestamp_opt,
            self.last_pose,
            measurement.relative_motion,
        ) {
            // pose update
            self.current_pose.mul_assign(rmotion);

            let dt = current_timestamp - last_timestamp;
            self.odometry.update(dt, &rmotion);
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

        let final_transform = if better {
            self.matching.process_current_cloud(
                current_timestamp,
                &lidar_to_odom,
                int_points_in_base,
            )
        } else {
            lidar_to_odom.clone()
        };

        //log::debug!("current: {:?}", self.current_pose);
        //log::debug!("initial: {:?}", lidar_to_odom);
        //log::debug!("final  : {:?}", final_transform);

        self.current_pose = final_transform * lidar_to_baselink.inverse();

        //log::debug!("new current  : {:?}", self.current_pose);
        self.log_str_csv.clear();
        self.log_str_csv += format!(
            "{},{},{},{}",
            measurement.time,
            self.matching.optimized_cost.unwrap_or(0.0),
            self.odometry.velocity,
            self.odometry.yaw_rate
        )
        .as_str();

        if let Some(rec) = rec {
            plot_points_org(&rec, "current", &raw_points_in_odom);
            plot_points_org(&rec, "current_int", &int_points_in_odom);
            plot_points_org(&rec, "map", &self.matching.map.entire_map_cloud);
        }

        self.last_timestamp_opt = Some(current_timestamp);
        self.last_pose = Some(self.current_pose);
    }
}

struct LittleSLAMData {
    measurements: Vec<lsc_reader::Measurement>,
}

impl LittleSLAMData {
    fn new() -> LittleSLAMData {
        let measurements = lsc_reader::load_lsc_file(
            homedir::my_home()
                .unwrap()
                .ok_or("home dir not found")
                .unwrap()
                .join(std::path::Path::new("works/datasets/little_slam/hall.lsc"))
                .to_str()
                .unwrap(),
        );
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

            let trfm =
                pm.process_current_cloud(measurement.time, &odom_trfm, int_points_in_base.clone());
            //            let trfm = measurement.relative_motion.unwrap_or_default();

            if i > 0 {
                println!(
                    "before {}, after: {}",
                    pm.initial_cost.unwrap().sqrt(),
                    pm.optimized_cost.unwrap().sqrt()
                );
                println!("trfm: {}", trfm);

                assert_eq!(pm.optimized_cost < pm.initial_cost, true);
            }
        }
    }
}
