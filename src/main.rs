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

#[derive(Debug)]
struct PointCloud {
    pub matrix: nalgebra::Matrix3xX<f64>,
}

impl PointCloud {
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

use tokio::time::{sleep_until, Duration, Instant};

fn process_measurement(
    i: usize,
    current_timestamp: f64,
    measurement: &Measurement,
) -> h_analyzer_data::WorldFrame {
    log::debug!("odom: {:?}", measurement.odometry);
    log::debug!("lidar point num: {}", measurement.lidar.points.len());

    let mut wf = h_analyzer_data::WorldFrame::new(i, current_timestamp);
    let mut ego = h_analyzer_data::Entity::new();
    ego.add_estimate(
        "odometry".to_string(),
        h_analyzer_data::Estimate::Pose2D(h_analyzer_data::Pose2D::new(
            measurement.odometry.x as f64,
            measurement.odometry.y as f64,
            measurement.odometry.theta as f64,
        )),
    );

    let angle_offset = std::f32::consts::PI; // between lidar and ego front
    let baselink_to_odom = nalgebra::Isometry2::new(
        nalgebra::Vector2::new(measurement.odometry.x as f64, measurement.odometry.y as f64),
        (measurement.odometry.theta + angle_offset) as f64,
    );
    let raw_points_in_odom = PointCloud::from_vec_of_points(&measurement.lidar.points)
        .transform_by_mat(baselink_to_odom.to_homogeneous());

    ego.add_measurement(
        "lidar".to_string(),
        h_analyzer_data::Measurement::PointCloud2D(raw_points_in_odom.to_h_pointcloud_2d()),
    );

    let int_points_in_odom = PointCloud::from_vec_of_points(&measurement.lidar.interpolated_points)
        .transform_by_mat(baselink_to_odom.to_homogeneous());

    ego.add_measurement(
        "lidar_int".to_string(),
        h_analyzer_data::Measurement::PointCloud2D(int_points_in_odom.to_h_pointcloud_2d()),
    );
    wf.add_entity("ego".to_string(), ego);
    wf
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

    let first_timestamp = measurements[0].time;
    let first_time = Instant::now();
    for i in 0..measurements.len() {
        let measurement = &measurements[i];
        let current_timestamp = measurement.time;
        let time_diff = current_timestamp - first_timestamp;

        //log::debug!("check next {}\nts {}", i, current_timestamp,);
        sleep_until(first_time + Duration::from_secs_f64(time_diff)).await;

        cl.send_world_frame(process_measurement(i, current_timestamp, measurement))
            .await
            .unwrap();

        //break;
    }
}
