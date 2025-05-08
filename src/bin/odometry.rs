use itertools::{izip, multiunzip};
use polars::prelude::*;

macro_rules! struct_to_dataframe {
    ($input:expr, [$($field:ident, $( $field_path:ident ).+),+]) => {
        {
            let len = $input.len().to_owned();

            // Extract the field values into separate vectors
            $(let mut $field = Vec::with_capacity(len);)*

            for e in $input.into_iter() {
                $($field.push(e.$($field_path).+);)*
            }
            df! {
                $(stringify!($field) => $field,)*
            }
        }
    };
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    let mut cl = h_analyzer_client_lib::HAnalyzerClient::new("http://localhost:50051").await;

    // citrus odometry
    let mut schema = Schema::new();
    schema.with_column("timestamp[s]".to_string().into(), DataType::Float64);
    schema.with_column("x[m]".to_string().into(), DataType::Float64);
    schema.with_column("y[m]".to_string().into(), DataType::Float64);
    schema.with_column("z[m]".to_string().into(), DataType::Float64);
    schema.with_column("qx[m]".to_string().into(), DataType::Float64);
    schema.with_column("qy[m]".to_string().into(), DataType::Float64);
    schema.with_column("qz[m]".to_string().into(), DataType::Float64);
    schema.with_column("qw[m]".to_string().into(), DataType::Float64);

    let df = CsvReadOptions::default()
        .with_schema(Some(Arc::new(schema)))
        .with_has_header(true)
        .with_skip_rows(2)
        .try_into_reader_with_file_path(Some(
            homedir::my_home()?
                .ok_or("home dir not found")
                .unwrap()
                .join(std::path::Path::new(
                    "works/datasets/Citrus-Farm-Dataset/scripts/ground_truth/01_13B_Jackal/gt.csv",
                ))
                .to_str()
                .unwrap()
                .into(),
        ))?
        .finish()
        .unwrap();
    println!("{}", df);

    let reader = rosbag2_ffi_rs::Rosbag2Reader::new(
        homedir::my_home()?
        .ok_or("home dir not found")
        .unwrap()
        .join(std::path::Path::new(
            "works/datasets/Citrus-Farm-Dataset/scripts/ground_truth/01_13B_Jackal/base_2023-07-18-14-26-48_0",
        ))
        .to_str()
        .unwrap()
        .into(),
    );
    println!("{}", reader);
    cl.register_data_frame("citrus_GT".to_string())
        .await
        .unwrap();
    cl.send_data_frame(df).await.unwrap();

    let topics =
        reader.parse_topic::<r2r::sensor_msgs::msg::NavSatFix>("/piksi/navsatfix_best_fix");

    let t0 = topics[0].0;
    let ts: Vec<f64> = topics.iter().map(|t| t.0 - t0).collect();
    let topics: Vec<r2r::sensor_msgs::msg::NavSatFix> =
        topics.iter().map(|t| t.1.clone()).collect();

    let mut df_horizontal_concat = polars::functions::concat_df_horizontal(&[
        df!(
            "timestamp[s]" => &ts,
        )
        .unwrap(),
        struct_to_dataframe!(
            topics,
            [
                header_stamp_sec,
                header.stamp.sec,
                header_stamp_nanosec,
                header.stamp.nanosec,
                latitude,
                latitude,
                longitude,
                longitude,
                altitude,
                altitude
            ]
        )
        .unwrap(),
    ])
    .unwrap();
    append_enu_column(&mut df_horizontal_concat);

    println!("{}", &df_horizontal_concat);

    cl.register_data_frame("citrus_GNSS".to_string())
        .await
        .unwrap();
    cl.send_data_frame(df_horizontal_concat).await.unwrap();

    // imu
    let topics = reader.parse_topic::<r2r::sensor_msgs::msg::Imu>("/microstrain/imu/data");

    let t0 = topics[0].0;
    let ts: Vec<f64> = topics.iter().map(|t| t.0 - t0).collect();
    let topics: Vec<r2r::sensor_msgs::msg::Imu> = topics.iter().map(|t| t.1.clone()).collect();

    let mut df_horizontal_concat = polars::functions::concat_df_horizontal(&[
        df!(
            "timestamp[s]" => &ts,
        )
        .unwrap(),
        struct_to_dataframe!(
            topics,
            [
                header_stamp_sec,
                header.stamp.sec,
                header_stamp_nanosec,
                header.stamp.nanosec,
                angular_vel_x,
                angular_velocity.x,
                angular_vel_y,
                angular_velocity.y,
                angular_vel_z,
                angular_velocity.z,
                linear_acc_x,
                linear_acceleration.x,
                linear_acc_y,
                linear_acceleration.y,
                linear_acc_z,
                linear_acceleration.z
            ]
        )
        .unwrap(),
    ])
    .unwrap();
    append_enu_column(&mut df_horizontal_concat);

    println!("{}", &df_horizontal_concat);

    cl.register_data_frame("citrus_IMU".to_string())
        .await
        .unwrap();
    cl.send_data_frame(df_horizontal_concat).await.unwrap();

    Ok(())
}

fn append_enu_column(df: &mut DataFrame) -> anyhow::Result<&DataFrame> {
    let lats = df.column("latitude")?.f64()?.into_iter();
    let lons = df.column("longitude")?.f64()?.into_iter();
    let alts = df.column("altitude")?.f64()?.into_iter();

    let ini_llu = geoconv::LLE::<geoconv::Wgs84>::new(
        geoconv::Degrees::new(33.963414),
        geoconv::Degrees::new(-117.346736),
        geoconv::Meters::new(267.865247),
    );
    let enus: Vec<(f64, f64, f64)> = izip!(lats, lons, alts)
        .map(|(lat, lon, alt)| match (lat, lon, alt) {
            (Some(lat), Some(lon), Some(alt)) => {
                let enu = geoconv::LLE::<geoconv::Wgs84>::new(
                    geoconv::Degrees::new(lat),
                    geoconv::Degrees::new(lon),
                    geoconv::Meters::new(alt),
                )
                .enu_to(&ini_llu);
                (enu.east.as_float(), enu.north.as_float(), enu.up.as_float())
            }
            _ => panic!("error"),
        })
        .collect();

    let (es, ns, us): (Vec<_>, Vec<_>, Vec<_>) = multiunzip(enus);
    let es: Float64Chunked = es.into_iter().map(|v| Some(v)).collect();
    let ns: Float64Chunked = ns.into_iter().map(|v| Some(-v)).collect();
    let us: Float64Chunked = us.into_iter().map(|v| Some(v)).collect();

    df.with_column(Series::new("east[m]", es))?;
    df.with_column(Series::new("north[m]", ns))?;
    df.with_column(Series::new("up[m]", us))?;

    Ok(df)
}
