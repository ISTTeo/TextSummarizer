import json
import os
import multiprocessing
from tqdm import tqdm
from utility.llm_general_summarizer import *  # Assuming all summarizer-related functions are in summarizer.py
import argparse

def ensure_directory(directory):
    """Ensure that the specified directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def update_json_file(artifacts, file_path):
    """Helper function to update JSON file after processing each artifact."""
    with open(file_path, 'w') as file:
        json.dump(artifacts, file, indent=4)

def debug_log(message):
    """Helper function to print debug logs if the debug flag is True."""
    if args.debug:
        print(message)

def double_summarize_artifact(model_id,
                              artifact_json,
                              artifact_text_field,
                              first_summmary_field,
                              second_summary_field,
                              sys_summarize_with_context,
                              sys_command_extract_with_context,
                              generation_args,
                              context_field=None):
    """Process a single artifact."""
    if second_summary_field in artifact_json.keys():
        return artifact_json

    # Initialize model and tokenizer
    m, t = model_tokenizer(model_id)  # Initialize model and tokenizer

    context = "" if context_field is None else artifact_json[context_field]

    print("Got model and context")
    artifact_json[first_summmary_field] = get_first_summary(m, t, generation_args, 
                                                            artifact_json,
                                                            artifact_text_field,
                                                            sys_summarize_with_context,
                                                            context)  # Get mini-summary
    print("1st summary")
    artifact_json[second_summary_field] = get_summary_over_summary(m, t, generation_args,
                                                                   artifact_json,
                                                                   first_summmary_field,
                                                                   sys_command_extract_with_context,
                                                                   context)  # Get summary-over-summary
    print("2nd summary")

    return artifact_json

def worker(worker_args):
    """Worker function for multiprocessing."""
    artifacts_subset, partition_id, gpu_id, output_dir, model_id, id_field, artifact_text_field, first_summary_field, summary_over_summary_field, sys_summarize_with_context, sys_command_extract_with_context, generation_args = worker_args
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    processed_artifacts = []
    
    partition_file = os.path.join(output_dir, f'partition_{partition_id}.json')
    
    # Load existing processed artifacts if the partition file exists
    if os.path.exists(partition_file):
        with open(partition_file, 'r') as file:
            processed_artifacts = json.load(file)
        
        # Create a set of processed artifact URLs for quick lookup
        processed_urls = set(artifact[id_field] for artifact in processed_artifacts)
        
        # Filter out already processed artifacts from artifacts_subset
        artifacts_subset = [artifact for artifact in artifacts_subset if artifact[id_field] not in processed_urls]
    
    for i, artifact in enumerate(artifacts_subset):
        try:
            processed_artifact = double_summarize_artifact(model_id,
                                                           artifact, 
                                                           artifact_text_field,
                                                           first_summary_field, 
                                                           summary_over_summary_field,
                                                           sys_summarize_with_context, 
                                                           sys_command_extract_with_context,
                                                           generation_args,
                                                           None) # FIXME add context field
            processed_artifacts.append(processed_artifact)
        except Exception as e:
            print(f"Error processing artifact {artifact[id_field]} in partition {partition_id} on GPU {gpu_id}: {str(e)}")
            processed_artifacts.append(artifact)  # Append original artifact if processing fails
        
        # Save progress after each artifact
        update_json_file(processed_artifacts, partition_file)
        
        # Log progress
        print(f"GPU {gpu_id}, Partition {partition_id}: Processed artifact {i+1}/{len(artifacts_subset)}")
    
    return processed_artifacts, partition_id

def partition_artifacts(artifacts, num_partitions):
    """Partition the artifacts list into num_partitions sublists."""
    partition_size = len(artifacts) // num_partitions
    partitions = [artifacts[i:i + partition_size] for i in range(0, len(artifacts), partition_size)]
    
    # Distribute any remaining artifacts
    for i in range(len(artifacts) % num_partitions):
        partitions[i].append(artifacts[partition_size * num_partitions + i])
    
    return partitions

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpu_devices))
    gpu_list = args.gpu_devices
    num_gpus = len(gpu_list)

    ensure_directory(args.output_dir)

    debug_log("\n===================================")
    debug_log("Getting data")
    debug_log("===================================\n")

    # Set default generation_args
    generation_args = {
        "use_cache": True,
        "max_new_tokens": 1000
    }
    # Override defaults if the generation_args_file is provided
    if args.generation_args_file:
        with open(args.generation_args_file, 'r') as file:
            generation_args = json.load(file)

    with open("default_llm_commands.json", 'r') as file:
        default_sys_commands = json.load(file)
    if args.sys_commands_file:
        with open(args.sys_commands_file, 'r') as file:
            sys_commands = json.load(file)
            
            if "sys_command_extract_with_context" in sys_commands.keys():
                sys_command_extract_with_context = sys_commands['sys_command_extract_with_context']
            else:
                sys_command_extract_with_context = default_sys_commands['sys_command_extract_with_context']
            
            if "sys_summarize_with_context" in sys_commands.keys():
                sys_summarize_with_context = sys_commands['sys_summarize_with_context']
            else:
                sys_summarize_with_context = default_sys_commands['sys_summarize_with_context']
 
    with open(args.input_json, 'r') as file:
        artifacts = json.load(file)

    total_processes = num_gpus * args.processes_per_gpu
    artifact_partitions = partition_artifacts(artifacts, total_processes)

    # Create a multiprocessing Pool with total_processes workers
    with multiprocessing.Pool(processes=total_processes) as pool:
        # Create the main progress bar
        with tqdm(total=len(artifacts), desc="Total Progress") as pbar:
            worker_args = [
                (partition, i, gpu_list[(i // args.processes_per_gpu) % num_gpus],
                args.output_dir,
                args.model_id,
                args.id_field,
                args.artifact_text_field,
                args.first_summary_field,
                args.summary_over_summary_field,
                sys_summarize_with_context,
                sys_command_extract_with_context,
                generation_args) # FIXME use JSON to set some of these variables
                for i, partition in enumerate(artifact_partitions)
            ]
            for processed_partition, partition_id in pool.imap_unordered(worker, worker_args):
                # Update the main artifact list with processed artifacts
                for processed_artifact in processed_partition:
                    for i, artifact in enumerate(artifacts):
                        if artifact[args.id_field] == processed_artifact[args.id_field]:
                            artifacts[i] = processed_artifact
                            break
                    
                    # Update progress bar
                    pbar.update(1)
                
                # Save the complete updated list after each partition
                update_json_file(artifacts, args.input_json)
                print(f"Completed processing partition {partition_id} and saved in original file.")

    print("\n===================================")
    print(f"All artifacts processed. Check your original file {args.input_json}")
    print("===================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process artifacts using GPU acceleration.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--input_json", 
                        required=True, 
                        help="Path to the input JSON file containing artifacts to process.")
    parser.add_argument("--output_dir", 
                        required=True, 
                        help="Directory to store partition JSON files and processed results.")
    parser.add_argument("--debug", 
                        action="store_true", 
                        help="Enable debug output for verbose logging.")
    parser.add_argument("--gpu_devices", 
                        type=int, 
                        nargs="+", 
                        required=True, 
                        help="List of GPU device indices to use (e.g., --gpu_devices 0 1 2).")
    parser.add_argument("--processes_per_gpu", 
                        type=int, 
                        required=True, 
                        help="Number of parallel processes to run per GPU.")
    parser.add_argument("--model_id",
                        required=True,
                        help="Model to be used")
    parser.add_argument("--first_summary_field",
                        required=True,
                        help="First pass summary")
    parser.add_argument("--summary_over_summary_field",
                        required=True,
                        help="Second pass summary")
    parser.add_argument("--artifact_text_field",
                        required=True,
                        help="Field that holds the original text")
    parser.add_argument("--sys_commands_file", 
                        help="Path to the JSON file containing system commands for llm model (optional).")
    parser.add_argument("--generation_args_file", 
                        help="Path to the JSON file containing generation arguments (optional).")
    parser.add_argument("--id_field",
                        required=True,
                        help="Field that ids artifact")

    args = parser.parse_args()
    
    multiprocessing.set_start_method('spawn')
    main(args)