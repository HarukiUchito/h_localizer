use h_localizer::{point_cloud_slam::LogHeader, *};
use tokio::time::{sleep_until, Duration, Instant};

use std::io::Write;

struct LogWriter {
    writer: std::io::BufWriter<std::fs::File>,
}

impl LogWriter {
    fn new(path_str: &str) -> Self {
        let mut writer = std::io::BufWriter::new(
            std::fs::File::create(std::path::Path::new(path_str)).expect("file creation failed"),
        );
        writeln!(writer, "# h_localizer log").expect("write failed");
        writeln!(writer, "# git version: {}", git_version::git_version!()).expect("write failed");

        Self { writer: writer }
    }

    fn write_line(&mut self, line: &str) -> anyhow::Result<()> {
        writeln!(self.writer, "{}", line)?;
        Ok(())
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    let matches = clap::Command::new("h_localizer")
        .arg(
            clap::arg!(--num <VALUE>)
                .default_value("0")
                .value_parser(clap::value_parser!(usize)),
        )
        .get_matches();
    let file_num = matches.get_one::<usize>("num").unwrap();
    let file_path = homedir::my_home()?
        .ok_or("home dir not found")
        .unwrap()
        .join(std::path::Path::new("works/datasets/little_slam/"))
        .to_str()
        .unwrap()
        .to_string()
        + match file_num {
            0 => "hall",
            1 => "corridor",
            _ => panic!("no file"),
        }
        + ".lsc";
    println!("filepath: {}", file_path);
    let measurements = lsc_reader::load_lsc_file(file_path.as_str())?;

    let rec = rerun::RecordingStreamBuilder::new("little_slam").spawn()?;

    let mut slam = point_cloud_slam::PointCloudSLAM::new();
    let mut slam_better = point_cloud_slam::PointCloudSLAM::new();

    // prepare directory for log files
    let _ = std::fs::create_dir("log");
    let now = chrono::Utc::now()
        .format("%Y_%m_%d_%H_%M_%S_%Z")
        .to_string();

    let mut log_writer = LogWriter::new(format!("log/log_{}.csv", now).as_str());
    log_writer.write_line(format!("# generated on: {}", now).as_str())?;
    log_writer.write_line(format!("# data used: {}", file_path).as_str())?;
    log_writer
        .write_line(format!("# {}", point_cloud_slam::PointCloudSLAM::LOG_HEADER).as_str())?;

    let first_timestamp = measurements[0].time;
    let first_time = Instant::now();
    for i in 0..measurements.len() {
        println!("frame {}/{}", i, measurements.len());
        let measurement = &measurements[i];
        let current_timestamp = measurement.time;
        let time_diff = current_timestamp - first_timestamp;

        //log::debug!("check next {}\nts {}", i, current_timestamp,);
        sleep_until(first_time + Duration::from_secs_f64(time_diff)).await;

        rec.set_time_sequence("frame_idx", i as i64);
        rec.set_time_seconds("timestamp", current_timestamp);

        slam_better.process_measurement(i, current_timestamp, measurement, true, Some(&rec));
        //slam.process_measurement(i, current_timestamp, measurement, false, &rec);

        let t = slam_better.log_str_csv.as_str();
        log_writer.write_line(t)?;

        if i == 100 {
            //break;
        }
    }
    Ok(())
}
