use std::ffi::CStr;
use std::time::{Duration, Instant};

use onnxruntime::*;
use structopt::{clap, StructOpt};
use std::path::PathBuf;

#[structopt(
    name = "run",
    about = "Run a benchmark on an onnx model. Each worker runs the model in a loop in its own
    thead. Once done it will print the average time to run the model.",
    setting = clap::AppSettings::ColoredHelp
)]
#[derive(StructOpt)]
struct Opt {
    /// The path to the onnx files to benchmark
    onnx: Vec<String>,

    /// A comma separated list of symbolic_dimension=value. If a symbolic dimension is not
    /// specified, 1 will be used.
    #[structopt(long)]
    dims: Option<String>,

    /// The number of worker threads to spawn
    #[structopt(long, default_value = "1")]
    workers: usize,

    /// The number of runs each worker will
    #[structopt(long, default_value = "1")]
    runs: usize,
}

// #[derive(Debug, Clone)]
// struct ModelInfo {
//     pub path: PathBuf,
//     pub name: String,
//     pub description: String,
//     pub version: i64,
//     pub input_shape: Vec<i64>,
//     pub output_shape: Vec<i64>,
//     pub input_name: String,
//     pub output_name: String,
// }

// struct OnnxModelConfig {
//     pub model_path: PathBuf,
//     pub parallel: bool,
//     pub use_cuda: bool,
//     pub use_cpu: bool,
//     pub use_tensorrt: bool,
//     pub cpu_use_arena:bool,
//     pub tensorrt_device_index: i32,
//     pub cuda_device_index: i32,
//     pub logging_level: LoggingLevel,
//     pub logging_id: String,
// }

// struct OnnxModel {
//     info: Arc<ModelInfo>,
//     env: Env,
//     options: SessionOptions,
//     session: Session,
// }

// impl OnnxModel {
//     pub fn new(config: OnnxModelConfig) -> Result<Self, Error> {
//         let env = Env::new(config.logging_level, &config.logging_id)?;
//         let mut options = SessionOptions::new()?;
        
//         if config.parallel {
//             options.set_execution_mode(ExecutionMode::Parallel)?;
//         }

//         if config.use_cpu {
//             options.add_cpu(config.cpu_use_arena);
//         }

//         if config.use_cuda {
//             options.add_cuda(config.cuda_device_index);
//         }

//         if config.use_tensorrt {
//             options.add_tensorrt(config.tensorrt_device_index);
//         }

//         let session = Session::new(&env, config.model_path.to_str().unwrap(), &options)?;
//         let metadata = session.metadata();
//         let inputs = session.inputs();
//         let outputs = session.outputs();

//         if inputs.len() != 1 {
//             return Err(Error::ModelWrongInputsCount);
//         }

//         if outputs.len() != 1 {
//             return Err(Error::ModelWrongOutputsCount);
//         }
        
//         let input = session.input(0);
//         let input_info = input
//             .tensor_info()
//             .or_else(|| Error::ModelWrongInputType)?;
//         let input_name = input.name().to_string();
//         let input_shape: Vec<i64> = input_info
//             .symbolic_dims()
//             .map(|d| match d {
//                 SymbolicDim::Symbolic(_) => -1i64,
//                 SymbolicDim::Fixed(x) => x as i64,
//             })
//             .collect();

//         let output = session.output(0);
//         let output_info = output
//             .tensor_info()
//             .or_else(|| Error::ModelWrongInputType)?;
//         let output_name = output.name().to_string();
//         let output_shape: Vec<i64> = output_info
//             .symbolic_dims()
//             .map(|d| match d {
//                 SymbolicDim::Symbolic(_) => -1i64,
//                 SymbolicDim::Fixed(x) => x as i64,
//             })
//             .collect();
        
//         let info = ModelInfo {
//             path: path.into(),
//             name: format!("{} exported by {}", metadata.graph_name(),  metadata.producer_name()),
//             description: metadata.description().to_string(),
//             version: metadata.version(),
//             input_name,
//             input_shape,
//             output_name,
//             output_shape,
//         };

//         Ok(OnnxModel {
//             info: Arc::new(info),
//             env,
//             options,
//             session,
//         })
//     }
// }


// pub enum ModelCommand {
//     AddOnnxModel(usize, OnnxModelConfig),
//     Inference(usize, )
// }

// struct ModelPool {
//     counter: usize,
//     sender: Sender<ModelCommand>,
// }

// impl ModelPool {
//     pub fn add(&mut self, name: &str, file: Path) -> Result<&AsyncModel, Error> {
        
//     }

//     pub fn get_model(&self, name: &str) -> Option<&AsyncModel> {

//     }
// }


// struct AsyncModel {
    
// }




use std::collections::HashMap;

fn key_val_parse(str: &str) -> HashMap<String, usize> {
    let mut map = HashMap::new();
    if str.is_empty() {
        return map;
    }
    for key_val in str.split(',') {
        let mut iter = key_val.split('=');
        let key = iter.next().expect("no =");
        let val = iter
            .next()
            .expect("nothing after =")
            .parse()
            .expect("parse error");
        assert!(iter.next().is_none(), "more than 1 =");
        map.insert(key.to_owned(), val);
    }
    map
}

/// Get the size of a tensor, substituting symbolic dimentions.
fn tensor_size(
    info: &TensorInfo,
    named_sizes: &mut HashMap<String, usize>,
) -> (OnnxTensorElementDataType, Vec<usize>) {
    let dims = info
        .symbolic_dims()
        .map(|d| match d {
            SymbolicDim::Symbolic(name) => {
                let name = name.to_str().unwrap();
                named_sizes.get(name).cloned().unwrap_or_else(|| {
                    eprintln!("name {} not specified, setting to 1", name);
                    named_sizes.insert(name.to_owned(), 1);
                    1
                })
            }
            SymbolicDim::Fixed(x) => x,
        })
        .collect();
    (info.elem_type(), dims)
}

fn tensor_mut(name: &str, elem_type: OnnxTensorElementDataType, dims: &[usize]) -> Box<dyn AsMut<Val>> {
    use OnnxTensorElementDataType::*;
    
    println!("{:?} {} {:?}", elem_type, name, dims);

    match elem_type {
        Float => Box::new(Tensor::<f32>::init(dims, 0.0).unwrap()),
        Int64 => Box::new(Tensor::<i64>::init(dims, 0).unwrap()),
        Int32 => Box::new(Tensor::<i32>::init(dims, 0).unwrap()),
        t => panic!("Unsupported type {:?}", t),
    }
}

fn load_image(filename: &str, height: usize, width: usize) -> Vec<f32> {
    let img = image::open(filename).unwrap()
        // .resize_exact(width as _,  height as _, image::imageops::FilterType::Triangle)
        .into_rgb();

    img.as_flat_samples()
        .to_vec()
        .samples
        .into_iter()
        .map(|p| (p as f32) / 255.0)
        .collect()
}

fn tensor_with_size(
    name: &str,
    info: &TensorInfo,
    named_sizes: &mut HashMap<String, usize>,
) -> Box<dyn AsRef<Val>> {
    let (ty, dims) = tensor_size(info, named_sizes);
    use OnnxTensorElementDataType::*;

    println!("{:?} {} {:?}", ty, name, dims);
    match ty {
        Float => match name {
            "input" => Box::new(Tensor::<f32>::new(&dims, load_image("/home/andrey/Images/me.jpg", dims[2], dims[3])).unwrap()),
            _ => Box::new(Tensor::<f32>::init(&dims, 0.0).unwrap()),
        },
        Int64 => Box::new(Tensor::<i64>::init(&dims, 0).unwrap()),
        Int32 => Box::new(Tensor::<i32>::init(&dims, 0).unwrap()),
        t => panic!("Unsupported type {:?}", t),
    }
}

fn main() -> Result<()> {
    let env = Env::new(LoggingLevel::Fatal, "test")?;
    let opt = Opt::from_args();

    let mut so = SessionOptions::new()?;

    // so.set_execution_mode(ExecutionMode::Parallel)?;
    // so.add_tensorrt(0);
    // so.add_cuda(0);
    // so.add_cpu(true);

    let mut map = if let Some(dims) = &opt.dims {
        key_val_parse(dims)
    } else {
        HashMap::new()
    };

    let batch_size = 4;

    map.insert("batch_size".into(), batch_size);

    for path in &opt.onnx {
        println!("model {:?}", path);
        let session = match Session::new(&env, path, &so) {
            Ok(sess) => sess,
            Err(err) => {
                eprintln!("error: {}\n", err);
                continue;
            }
        };

        let metadata = session.metadata();
        eprintln!("name: {}", metadata.producer_name());
        eprintln!("graph_name: {}", metadata.graph_name());
        eprintln!("domain: {}", metadata.domain());
        eprintln!("description: {}", metadata.description());

        let mut input_names: Vec<OrtString> = vec![];
        let mut input_tensors: Vec<Box<dyn AsRef<Val>>> = vec![];

        for (i, input) in session.inputs().enumerate() {
            if let Some(tensor_info) = input.tensor_info() {
                input_names.push(input.name());
                input_tensors.push(tensor_with_size(input.name().as_str(), &tensor_info, &mut map));
            } else {
                println!("input {}: {:?} {:?}", i, &*input.name(), input.onnx_type());
            }
        }

        let mut output_names: Vec<OrtString> = vec![];
        let mut output_sizes: Vec<(OnnxTensorElementDataType, Vec<usize>)> = vec![];

        for (i, output) in session.outputs().enumerate() {
            if let Some(tensor_info) = output.tensor_info() {
                output_names.push(output.name());
                output_sizes.push(tensor_size(&tensor_info, &mut map));
            } else {
                println!(
                    "output {}: {:?} {:?}",
                    i,
                    &*output.name(),
                    output.onnx_type()
                );
            }
        }

        let in_names: Vec<&CStr> = input_names.iter().map(|x| x.as_c_str()).collect();
        let in_vals: Vec<&Val> = input_tensors.iter().map(|x| x.as_ref().as_ref()).collect();
        let out_names: Vec<&CStr> = output_names.iter().map(|x| x.as_c_str()).collect();

        let ro = RunOptions::new();

        let before = Instant::now();

        let mut res = session
            .run_raw(&ro, &in_names, &in_vals[..], &out_names)
            .expect("run");

        let tensor = match res.pop().unwrap().as_tensor::<f32>() {
            Ok(t) => t,
            _ => panic!("something went wrong")
        };

        println!("[{:?}] {}", tensor.dims(), before.elapsed().as_millis())

        // println!("{:?}", out_vals[0].as_slice::<f32>());
        // let out_vals[0].into();
        // let total: Duration = times.iter().sum()
        // let avg = total / (times.len() as u32);
        // eprintln!("worker {} avg time: {:.2} ms", i, avg.as_secs_f64() * 1e3);
    }

    Ok(())
}
