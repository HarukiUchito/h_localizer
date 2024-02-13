use h_localizer::*;
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
    let file_path = "/home/haruki/Works/datasets/little_slam/".to_string()
        + match file_num {
            0 => "hall",
            1 => "corridor",
            _ => panic!("no file"),
        }
        + ".lsc";
    println!("filepath: {}", file_path);
    let measurements = lsc_reader::load_lsc_file(file_path.as_str())?;

    let mut cl = h_analyzer_client_lib::HAnalyzerClient::new("http://localhost:50051").await;
    cl.register_new_world(&format!("slam_{}", file_num))
        .await
        .unwrap();

    let mut slam = point_cloud_slam::PointCloudSLAM::new();
    let mut slam_better = point_cloud_slam::PointCloudSLAM::new();

    let mut log_writer = LogWriter::new(format!("log_{}.csv", file_num).as_str());
    log_writer.write_line(format!("# data used: {}", file_path).as_str())?;

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

        let t = slam_better.log_str_csv.as_str();
        log_writer.write_line(t)?;

        //break;
    }
    Ok(())
}
