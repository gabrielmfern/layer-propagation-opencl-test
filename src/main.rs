use opencl3::command_queue::{CL_BLOCKING, CL_NON_BLOCKING, CL_QUEUE_PROFILING_ENABLE};
use opencl3::device::{cl_float, cl_int, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY};
use opencl3::{
    command_queue::CommandQueue,
    context::Context,
    device::{get_all_devices, Device},
    program::Program,
    types::cl_event,
};
// use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::ptr;
// use std::time::Instant;

const PROPAGATE_KERNEL: &str = include_str!("propagate.cl");
const KERNEL_NAME: &str = "propagate";

fn main() {
    let devices = get_all_devices(CL_DEVICE_TYPE_GPU).expect("Unable to get devices");
    assert!(0 < devices.len());

    // let properties = &Vec::<cl_context_properties>::default();
    let device = Device::new(devices[0]);
    // let max_device_local_size = device.max_work_group_size().expect("Unable to get max local work group sizes");
    let context = Context::from_device(&device)
        .expect("Unable to create context from all of the available devices");

    let queue =
        CommandQueue::create_with_properties(&context, device.id(), CL_QUEUE_PROFILING_ENABLE, 0)
            .expect("Unable to create Queue to first GPU device found");
    // let queues = devices
    //     .into_iter()
    //     .map(|device_id| {
    //         CommandQueue::create_with_properties(&context, device_id, 0, 0)
    //             .expect("Unable to create one of the Queues with one of the devices")
    //     })
    //     .collect::<Vec<CommandQueue>>();

    let program = Program::create_and_build_from_source(&context, PROPAGATE_KERNEL, "")
        .expect("Unable to create and build the program from the kernel source"); let kernel =
        Kernel::create(&program, KERNEL_NAME).expect("Unable to create the kernel for the program");

    let inputs_amount = 14_000;
    let outputs_amount = 401;
    let inputs: Vec<Vec<f32>> = vec![vec![0.31; inputs_amount]; 10_000];
    let weights: Vec<Vec<f32>> = vec![vec![0.123; outputs_amount]; inputs_amount];

    // let cpu_start = Instant::now();
    // let expected_outputs: Vec<f32> = inputs
    //     .par_iter()
    //     .map(|sample_inputs| {
    //         (0..outputs_amount)
    //             .into_iter()
    //             .map(|output_index| {
    //                 (0..inputs_amount)
    //                     .into_iter()
    //                     .map(|input_index| {
    //                         sample_inputs[input_index] * weights[input_index][output_index]
    //                     })
    //                     .sum::<f32>()
    //             })
    //             .collect::<Vec<f32>>()
    //     })
    //     .flatten()
    //     .collect();
    // let cpu_elapsed_nanosceonds = cpu_start.elapsed().as_nanos();
    // println!(
    //     "took {:?} to compute the outputs on the CPU",
    //     cpu_start.elapsed()
    // );
    let samples_amount = inputs.len();
    let flattened_inputs: Vec<f32> = inputs.into_iter().flatten().collect();

    // let expected_outputs: Vec<f32> = Vec::from([
    //     0.0 * 0.53 + 0.0 * 0.32,
    //     0.0 * 0.53 + 1.0 * 0.32,
    //     1.0 * 0.53 + 0.0 * 0.32,
    //     1.0 * 0.53 + 1.0 * 0.32,
    // ]);
    let flattened_weights: Vec<f32> = weights.into_iter().flatten().collect();

    let mut inputs_buffer = Buffer::<cl_float>::create(
        &context,
        CL_MEM_READ_ONLY,
        inputs_amount * samples_amount,
        ptr::null_mut(),
    )
    .expect("Unable to allocate buffer for inputs");
    let mut weights_buffer = Buffer::<cl_float>::create(
        &context,
        CL_MEM_READ_ONLY,
        inputs_amount * outputs_amount,
        ptr::null_mut(),
    )
    .expect("Unable to allocate buffer for weights");
    let outputs_buffer = Buffer::<cl_float>::create(
        &context,
        CL_MEM_WRITE_ONLY,
        samples_amount * outputs_amount,
        ptr::null_mut(),
    )
    .expect("Unable to allocate buffer for outputs");

    let inputs_write_event = queue
        .enqueue_write_buffer(
            &mut inputs_buffer,
            CL_BLOCKING,
            0,
            &flattened_inputs.as_slice(),
            &[],
        )
        .expect("Unable to write the data to the inputs buffer");
    let weights_write_event = queue
        .enqueue_write_buffer(
            &mut weights_buffer,
            CL_BLOCKING,
            0,
            &flattened_weights.as_slice(),
            &[],
        )
        .expect("Unable to write the data to the weights buffer");
    // let local_work_size = (max_device_local_size as f32).sqrt().floor() as usize;

    let arg_inputs_amount: cl_int = inputs_amount as cl_int;

    let samples_work_size = samples_amount; // / local_samples_work_size;
    let outputs_work_size = outputs_amount;

    let propagate_kernel_event = ExecuteKernel::new(&kernel)
        .set_arg(&inputs_buffer)
        .set_arg(&weights_buffer)
        .set_arg(&outputs_buffer)
        .set_arg(&arg_inputs_amount)
        .set_global_work_sizes(&[samples_work_size, outputs_work_size])
        // .set_local_work_sizes(&[max_device_local_size, 1, 1])
        .set_wait_event(&weights_write_event)
        .set_wait_event(&inputs_write_event)
        .enqueue_nd_range(&queue)
        .expect("Unable to execute Kernel in the GPU");

    let mut events: Vec<cl_event> = Vec::default();
    events.push(propagate_kernel_event.get());

    let mut outputs_vec = vec![0.0; samples_amount * outputs_amount];
    let outputs: &mut [cl_float] = outputs_vec.as_mut_slice();
    let read_event = queue
        .enqueue_read_buffer(&outputs_buffer, CL_NON_BLOCKING, 0, outputs, &events)
        .expect("Unable to read the outputs_buffer from the kernel");

    read_event
        .wait()
        .expect("Some error happened while waiting for the events to run");

    let start_time = propagate_kernel_event
        .profiling_command_start()
        .expect("Unable to start profiling the propagate Kernel");
    let end_time = propagate_kernel_event
        .profiling_command_end()
        .expect("Unable to end profilling the propagate Kernel");
    let duration = end_time - start_time;
    println!(
        "propagate kernel execution duration (ns): {}\n",
        duration,
    );
    // println!("being {}x faster than the CPU",cpu_elapsed_nanosceonds / duration as u128);

    // assert_eq!(outputs.to_vec(), expected_outputs);
    // println!("kernel output front {}", outputs[0]);
    // println!("expected_output front {}", expected_outputs[0]);
    // println!("kernel output back {}", outputs.last().unwrap());
    // println!("expected_output back {}", expected_outputs.last().unwrap());
}
