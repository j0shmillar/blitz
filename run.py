import os
import subprocess

# TODO
# - args
# - cmake
# - lots of EH

def generate_model_h(tflite_model_path, output_header_path):
    with open(tflite_model_path, 'rb') as f:
        model_data = f.read()
    with open(output_header_path, 'w') as f:
        f.write('#ifndef MODEL_H_\n')
        f.write('#define MODEL_H_\n\n')
        f.write('unsigned char model_data[] = {')
        for i, byte in enumerate(model_data):
            if i % 12 == 0:
                f.write('\n ')
            f.write(f'0x{byte:02x}, ')
        f.write('\n};\n\n')
        f.write(f'unsigned int model_data_len = {len(model_data)};\n\n')
        f.write('#endif  // MODEL_H_\n')

def inject_parameters(template_path, main_c_path, params):
    with open(template_path, 'r') as file:
        template = file.read()
    template = template.replace('{', '{{').replace('}', '}}')
    template = template.replace('{{input_size}}', '{input_size}')
    template = template.replace('{{output_size}}', '{output_size}')
    template = template.replace('{{num_runs}}', '{num_inferences}')
    template = template.replace('{{delay_between_inferences}}', '{interval_seconds}')
    code = template.format(
        input_size=params['input_size'],
        output_size=params['output_size'],
        num_runs=params['num_inferences'],
        delay_between_inferences=params['interval_seconds'])
    with open(main_c_path, 'w') as file:
        file.write(code)

def compile_and_flash(output_path, build_dir, device_id):
    # run cmake make etc here
    flash_cmd = ["dfu-util", "-d", device_id, "-a", "0", "-s", "0x08000000:leave", "-D", "build/main.bin"]
    subprocess.run(flash_cmd, check=True, cwd=build_dir)

if __name__ == "__main__":
    # args
    model_path = "models/model.tflite"
    output_path = "src/model.h"
    template_path = "src/main_template.cpp"
    main_c_path = "src/main.cpp"
    build_dir = "build"
    device_id = "your_device_id"  

    generate_model_h(model_path, output_path)

    params = {"input_size": 28*28, "output_size": 10, "num_inferences": 100, "interval_seconds": 1} # args
    inject_parameters(template_path, main_c_path, params)
    compile_and_flash(main_c_path, build_dir, device_id)
