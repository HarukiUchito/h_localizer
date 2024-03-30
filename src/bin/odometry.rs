struct OdometryNode {
    last_timestamp: f64,
    position: nalgebra::Vector3<f64>,
    orientation: nalgebra::UnitQuaternion<f64>,

    client: h_analyzer_client_lib::HAnalyzerClient,
}

impl OdometryNode {
    pub async fn node_callback(&mut self, msg: rosrust_msg::nav_msgs::Odometry) {
        // Callback for handling received messages
        rosrust::ros_info!("Received: {:?}", msg.twist.twist.angular);
        rosrust::ros_info!("{:?}", msg.header.stamp);
        let t_sec = msg.header.stamp.sec;
        let t_nsec = msg.header.stamp.nsec;
        rosrust::ros_info!("{} {}", t_sec, t_nsec);
        let current_timestamp = t_sec as f64 + t_nsec as f64 * 1e-9;
        rosrust::ros_info!("sec {}", current_timestamp);
        let dt = current_timestamp - self.last_timestamp;
        rosrust::ros_info!("dt {}", dt);

        let d_x = msg.twist.twist.angular.x * dt;
        let d_y = msg.twist.twist.angular.y * dt;
        let d_z = msg.twist.twist.angular.z * dt;
        rosrust::ros_info!("d {} {} {}", d_x, d_y, d_z);

        let dq = nalgebra::UnitQuaternion::from_euler_angles(d_x, d_y, d_z);

        self.orientation *= dq;
        let eas = self.orientation.euler_angles();
        rosrust::ros_info!("q {:?}", eas.2.to_degrees());

        let dvec = nalgebra::Vector3::new(msg.twist.twist.linear.x * dt, 0.0, 0.0);
        rosrust::ros_info!("dvec {}", dvec);
        rosrust::ros_info!("rot dvec {}", self.orientation.transform_vector(&dvec));

        self.position += self.orientation.transform_vector(&dvec);

        let mut ego = h_analyzer_data::Entity::new();
        ego.add_estimate(
            "odometry".to_string(),
            h_analyzer_data::Estimate::Pose2D(h_analyzer_data::Pose2D::new(
                self.position.x,
                self.position.y,
                0.0,
            )),
        );
        let mut wf = h_analyzer_data::WorldFrame::new(msg.header.seq as usize, current_timestamp);
        wf.add_entity("ego".to_string(), ego);

        self.client.send_world_frame(wf).await.unwrap();

        self.last_timestamp = current_timestamp;
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    let mut odom = OdometryNode {
        last_timestamp: 0.0,
        position: nalgebra::Vector3::default(),
        orientation: nalgebra::UnitQuaternion::default(),
        client: h_analyzer_client_lib::HAnalyzerClient::new("http://192.168.1.8:50051").await,
    };

    odom.client
        .register_new_world(&"odometry_test".to_string())
        .await
        .unwrap();

    let mut odom_arc = std::sync::Arc::new(std::sync::Mutex::new(odom));

    // Initialize node
    rosrust::init("listener");

    // about passing member function as callback
    // https://github.com/adnanademovic/rosrust/issues/185
    let subscriber_info = rosrust::subscribe("/jackal_velocity_controller/odom", 2, move |v| {
        tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(odom_arc.lock().unwrap().node_callback(v));
    })
    .unwrap();

    // Block the thread until a shutdown signal is received
    rosrust::spin();

    Ok(())
}
