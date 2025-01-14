#!/usr/bin/env python3

# Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import sys
import json
import argparse


def is_json(file):
    try:
        json.load(file)
        file.seek(0) # Reset file pointer after reading
        return True
    except ValueError:
        return False


def verify_args(args):
    # Check if the number of config files matches the number of models
    if args.config and len(args.config) != len(args.models):
        raise ValueError(
            "The number of config files must match the number of model files.")

    # Check existence and validity of model files
    for model_path in args.models:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if '.' not in os.path.basename(model_path):
            raise ValueError("Model file must have an extension.")

    # Check existence of config files
    if args.config:
        for config_path in args.config:
            if not os.path.isfile(config_path):
                raise FileNotFoundError(f"Config file not found: {config_path}")

    # Check IO info files and size of indices
    size_inputs = 0
    size_outputs = 0
    for model_index, io_info_path in enumerate(args.io_info or []):
        with open(io_info_path, "r") as io_json:
            if not is_json(io_json):
                    raise ValueError(f"IO info file is not valid JSON: {io_info_path}")

            model_io = json.load(io_json)
            if model_index == 0:
                size_inputs = len(model_io["org-model-io"]["inputs"]["new-indices"])
                size_outputs = len(model_io["org-model-io"]["outputs"]["new-indices"])
            else:
                if size_inputs != len(model_io["org-model-io"]["inputs"]["new-indices"]):
                    raise ValueError(
                        f"Invalid size of input indices in {io_info_path}. "
                        "Input size differs from previous files."
                    )
                if size_outputs != len(model_io["org-model-io"]["outputs"]["new-indices"]):
                    raise ValueError(
                        f"Invalid size of output indices in {io_info_path}. "
                        "Output size differs from previous files."
                    )


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert model files (tflite, circle or tvn) to nnpkg.',
        usage=''' %(prog)s [options]
  Examples:
      %(prog)s -m add.tflite                           => create nnpkg "add" in current directory
      %(prog)s -o out -m add.tflite                    => create nnpkg "add" in out/
      %(prog)s -o out -p addpkg -m add.tflite          => create nnpkg "addpkg" in out/
      %(prog)s -c add.cfg -m add.tflite                => create nnpkg "add" with add.cfg
      %(prog)s -o out -p addpkg -m a1.tflite a2.tflite -i a1.json a2.json
        => create nnpkg "addpkg" with models a1.tflite and a2.tflite in out/
  ''')
    parser.add_argument(
        "-o", "--outdir",
        type=str,
        default=os.getcwd(),
        help="Output directory for nnpkg."
    )
    parser.add_argument(
        "-p", "--nnpkg-name",
        type=str,
        help="Name of the output nnpkg (default: based on the first model file)."
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        nargs="*",
        default='',
        help="Configuration files for the models."
    )
    parser.add_argument(
        "-m", "--models",
        type=str,
        nargs="+",
        required=True,
        help="Model files to be converted."
    )
    parser.add_argument(
        "-i", "--io-info",
        type=str,
        nargs="+",
        help="IO information JSON files for the models."
    )

    args = parser.parse_args()
    verify_args(args)

    if not args.nnpkg_name:
        args.nnpkg_name = os.path.splitext(os.path.basename(args.models[0]))[0]

    return args


def load_json(path):
    with open(path, "r") as file:
        return json.load(file)


# def generate_manifest(args):
#     io_info_files = args.io_info or []
#     pkg_inputs, pkg_outputs = [], []
#     model_connections = []

#     if io_info_files:
#         base_io = load_json(io_info_files[0])
#         pkg_inputs = list(range(len(base_io["org-model-io"]["inputs"]["new-indices"])))
#         pkg_outputs = list(range(len(base_io["org-model-io"]["outputs"]["new-indices"])))

#         for model_pos, io_path in enumerate(io_info_files):
#             io_data = load_json(io_path)
#             new_inputs = io_data["new-model-io"]["inputs"]["new-indices"]
#             new_outputs = io_data["new-model-io"]["outputs"]["new-indices"]

#             for idx, new_index in enumerate(
#                     io_data["org-model-io"]["inputs"]["new-indices"]):
#                 if new_index != -1:
#                     pkg_inputs[idx] = f"{model_pos}:0:{new_inputs.index(new_index)}"

#             for idx, new_index in enumerate(
#                     io_data["org-model-io"]["outputs"]["new-indices"]):
#                 if new_index != -1:
#                     pkg_outputs[idx] = f"{model_pos}:0:{new_outputs.index(new_index)}"

#             for inp_idx, org_input in enumerate(
#                     io_data["new-model-io"]["inputs"]["org-indices"]):
#                 for out_idx, org_output in enumerate(
#                         io_data["new-model-io"]["outputs"]["org-indices"]):
#                     if org_input == org_output:
#                         model_connections.append({
#                             "from": f"{model_pos}:0:{out_idx}",
#                             "to": f"{model_pos}:0:{inp_idx}"
#                         })

#     return {
#         "major-version": "1",
#         "minor-version": "2",
#         "patch-version": "0",
#         "configs": args.config or [],
#         "models": [os.path.basename(m) for m in args.models],
#         "model-types": [os.path.splitext(m)[1][1:] for m in args.models],
#         "pkg-inputs": pkg_inputs,
#         "pkg-outputs": pkg_outputs,
#         "model-connect": model_connections,
#     }

# import json

def generate_manifest(io_info_paths, major_version="1", minor_version="2", patch_version="0"):
    """
    Generate a manifest file with `model-connect` and additional package information.

    Args:
        io_info_paths (list): List of file paths to the JSON IO info files.
        major_version (str): Major version of the manifest.
        minor_version (str): Minor version of the manifest.
        patch_version (str): Patch version of the manifest.

    Returns:
        dict: Manifest dictionary with `model-connect` and additional metadata.
    """
    model_connect = []
    models = []
    model_types = []
    pkg_inputs = []
    pkg_outputs = []

    # Load all JSON files and store their parsed data
    model_io_data = []
    io_info_files = io_info_paths or []
    for idx, path in enumerate(io_info_files):
        with open(path, "r") as file:
            try:
                model_io_data.append(json.load(file))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON file at {path}: {e}")

        # Add model filenames and types (assume TFLite for now)
        models.append(f"i{idx + 1}.tflite")
        model_types.append("tflite")

    # Generate model-connect data
    for model_idx, model_io in enumerate(model_io_data):
        new_model_outputs = model_io["new-model-io"]["outputs"]

        # Connect outputs to next model's inputs
        for _, output_indices in new_model_outputs.items():
            for output_idx, output_value in enumerate(output_indices):
                current_from = f"{model_idx}:0:{output_idx}"

                connections = []
                for next_model_idx, next_model_io in enumerate(model_io_data[model_idx + 1:], start=model_idx + 1):
                    next_model_inputs = next_model_io["new-model-io"]["inputs"]

                    # Check matching indices in the next model's inputs
                    for _, input_indices in next_model_inputs.items():
                        if output_value in input_indices:
                            next_input_index = input_indices.index(output_value)
                            connection = f"{next_model_idx}:0:{next_input_index}"
                            if connection not in connections:
                                connections.append(connection)

                if connections:
                    model_connect.append({
                        "from": current_from,
                        "to": connections
                    })

    # Define package inputs and outputs
    first_model_inputs = model_io_data[0]["new-model-io"]["inputs"]["new-indices"]
    last_model_outputs = model_io_data[len(io_info_paths) - 1]["new-model-io"]["outputs"]["new-indices"]

    pkg_inputs = [f"0:0:{i}" for i in range(len(first_model_inputs))]
    pkg_outputs = [f"{len(io_info_paths) - 1}:0:{i}" for i in range(len(last_model_outputs))]

    # Build the manifest
    manifest = {
        "major-version": major_version,
        "minor-version": minor_version,
        "patch-version": patch_version,
        "configs": [""],
        "models": models,
        "model-types": model_types,
        "pkg-inputs": pkg_inputs,
        "pkg-outputs": pkg_outputs,
        "model-connect": model_connect
    }
    return manifest


def main():
    try:
        # parse arguments
        args = parse_args()

        # mkdir nnpkg directory
        nnpkg_path = os.path.join(args.outdir, args.nnpkg_name)
        metadata_path = os.path.join(nnpkg_path, 'metadata')
        os.makedirs(metadata_path, exist_ok=True)

        # dump manifest file
        manifest = generate_manifest(args.io_info)
        with open(os.path.join(metadata_path, "MANIFEST"), "w") as manifest_file:
            json.dump(manifest, manifest_file, indent=2)

        # copy models and configurations
        for model in args.models:
            shutil.copy2(model, nnpkg_path)

        if args.config:
            for config in args.config:
                shutil.copy2(config, metadata_path)

        print(f"nnpkg {args.nnpkg_name} generated at {args.outdir}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
