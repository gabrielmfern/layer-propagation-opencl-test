kernel void propagate(
    global float* flattened_inputs,
    global float* flattened_weights,

    global float* flattened_outputs,

    int inputs_amount
) {
    int global_sample_index = get_global_id(0);
    int global_samples_amount = get_global_size(0);

    int global_output_index = get_global_id(1);
    int global_outputs_amount = get_global_size(1);


    int sample_index = global_sample_index;
    int samples_amount = global_samples_amount;
    if (sample_index >= samples_amount) {
        return;
    }

    int output_index = global_output_index;
    int outputs_amount = global_outputs_amount;
    if (output_index >= outputs_amount) {
        return;
    }

    int flattened_output_index = sample_index * outputs_amount + output_index;

    float output = 0.0;

    for (int input_index = 0; input_index < inputs_amount; input_index++) {
        int flattened_input_index = sample_index * inputs_amount + input_index;
        int flattened_weight_index = input_index * outputs_amount + output_index;
    
        output += (float) (flattened_inputs[flattened_input_index] * flattened_weights[flattened_weight_index]);
    }

    flattened_outputs[flattened_output_index] = output;
}
