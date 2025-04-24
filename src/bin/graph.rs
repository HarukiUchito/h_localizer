use nalgebra as na;
use polars::series::implementations;

#[derive(Debug)]
struct Vertex {
    id: usize,
    position: f32,
}

trait Edge {
    fn residual(&self, vertices: &Vec<Vertex>) -> f32;
    fn jacobian_col_vec(&self, vertices: &Vec<Vertex>) -> na::DMatrix<f32>;
}

// prior information edge

#[derive(Debug)]
struct PriorEdge {
    id: usize,
    position: f32,
}
impl Edge for PriorEdge {
    fn residual(&self, vertices: &Vec<Vertex>) -> f32 {
        vertices[self.id].position - self.position
    }
    fn jacobian_col_vec(&self, vertices: &Vec<Vertex>) -> na::DMatrix<f32> {
        let mut j = na::DMatrix::zeros(1, vertices.len());
        j[(0, self.id)] = 1.0;
        j
    }
}

// edge of measure relative position of landmark

#[derive(Debug)]
struct MeasureLandmarkEdge {
    from_id: usize,
    to_id: usize,
    relative_position: f32,
}
impl Edge for MeasureLandmarkEdge {
    fn residual(&self, vertices: &Vec<Vertex>) -> f32 {
        let from_position = vertices[self.from_id].position;
        let to_position = vertices[self.to_id].position;
        from_position - to_position - self.relative_position
    }
    fn jacobian_col_vec(&self, vertices: &Vec<Vertex>) -> na::DMatrix<f32> {
        let mut j = na::DMatrix::zeros(1, vertices.len());
        j[(0, self.from_id)] = -1.0;
        j[(0, self.to_id)] = 1.0;
        j
    }
}

#[derive(Debug)]
enum EdgeObject {
    PriorEdge(PriorEdge),
    MeasureLandmarkEdge(MeasureLandmarkEdge),
}

#[derive(Debug)]
struct Graph {
    vertices: Vec<Vertex>,
    edges: Vec<EdgeObject>,
}

impl Graph {
    fn new() -> Self {
        Graph {
            vertices: Vec::new(),
            edges: Vec::new(),
        }
    }

    fn add_vertex(&mut self, id: usize, position: f32) {
        self.vertices.push(Vertex {
            id: id,
            position: position,
        });
    }

    fn add_prior_edge(&mut self, id: usize, position: f32) {
        self.edges
            .push(EdgeObject::PriorEdge(PriorEdge { id, position }));
    }

    fn add_measure_landmark_edge(&mut self, from_id: usize, to_id: usize, relative_position: f32) {
        self.edges
            .push(EdgeObject::MeasureLandmarkEdge(MeasureLandmarkEdge {
                from_id,
                to_id,
                relative_position,
            }));
    }

    fn solve_once(&mut self) {
        let mut score = 0.0;
        let n = self.vertices.len();
        let mut gradient: na::DVector<f32> = na::DVector::zeros(n);
        let mut hessian: na::DMatrix<f32> = na::DMatrix::zeros(n, n);
        for (i, edge) in self.edges.iter().enumerate() {
            let r = match edge {
                EdgeObject::PriorEdge(e) => e.residual(&self.vertices),
                EdgeObject::MeasureLandmarkEdge(e) => e.residual(&self.vertices),
            };
            let j = match edge {
                EdgeObject::PriorEdge(e) => e.jacobian_col_vec(&self.vertices),
                EdgeObject::MeasureLandmarkEdge(e) => e.jacobian_col_vec(&self.vertices),
            };

            score += r * r;

            gradient += (j.transpose() * na::DVector::from_element(1, r));
            hessian += j.transpose() * j.clone();
            println!("Edge: {:?}", edge);
            println!("Residual: {}", r);
            println!("Jacobian: {:?}", j.clone());
        }
        println!("Gradient: {:?}", gradient);
        println!("Hessian: {:?}", hessian);

        let delta = -hessian.cholesky().unwrap().solve(&gradient);
        println!("score: {}", score);
        println!("Delta: {:?}", delta);
        for i in 0..self.vertices.len() {
            self.vertices[i].position += delta[i];
        }
    }

    fn display(&self) {
        println!("Vertices:");
        for vertex in &self.vertices {
            println!("position: {}", vertex.position);
        }
        println!("Edges:");
        for edge in &self.edges {
            println!("Edge: {:?}", edge);
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    let mut graph = Graph::new();
    for i in 0..3 {
        graph.add_vertex(i, 0.0);
    }

    graph.add_measure_landmark_edge(0, 2, 2.0);
    graph.add_measure_landmark_edge(1, 2, -1.0);
    graph.add_measure_landmark_edge(0, 1, 3.1);
    graph.add_prior_edge(0, 0.0);

    println!("graph: {:?}", graph);
    println!("vertices: {:?}", graph.vertices);
    graph.solve_once();

    Ok(())
}
