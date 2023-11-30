import os
import subprocess
import time
import sys
import pkg_resources
import io
import logging

def print_red(text):
    red_text = f"\033[0;31m{text}\033[0m"
    print(red_text)

def check_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def check_file(file_path):
    if not os.path.isfile(file_path):
        print_red(f"lack of {file_path}!")
        exit(-1)

def download_file(source, destination):
    print(f"Please Download from {source} to {destination}")
    # Add your code to download the file here

def check_and_create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def check_file_exist(file_path):
    return os.path.isfile(file_path)

def load_txt(module_name, txtfile_path):
    text_data_bytes = pkg_resources.resource_string(module_name, txtfile_path)
    text_data = text_data_bytes.decode('utf-8')
    file_like_object = io.StringIO(text_data)
    data_list = []
    for line in file_like_object:
        data_list.append(line.strip())
    return data_list

def run_cmd(cmd, logger):
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    logger.info(res.stdout)
    if res.stderr:
        logger.warning(res.stderr)

def run_hdr(input_path, hdr_path, logger):
    if check_file_exist(os.path.join(hdr_path, "sky.hdr")):
        logger.info("skip hdr...")
    else:
        debug_save_path = "/all/palette_tmp"
        subprocess.run(["rm", "-rf", debug_save_path])
        palette_cmd =[
            "python",
            "-m",
            "anything_in_anyscene.hdr_sky.palette.run",
            "-p", "test",
            "-t", os.path.join(input_path, "cam2"),
            "-s", debug_save_path
        ]
        run_cmd(palette_cmd, logger)
        
        hdr_cmd = [
            "python", 
            "-m",
            "anything_in_anyscene.hdr_sky.inference", 
            "--inpaint",
            "--indir", os.path.join(input_path, "cam2"), 
            "--outdir", hdr_path
        ]
        run_cmd(hdr_cmd, logger)
            

def run_env_hdr(input_path, hdr_path, logger):
    if check_file_exist(os.path.join(hdr_path, "output_pano.hdr")):
        logger.info("skip env hdr...")
    else:    
        temp_results_path = "/all/tmp"
        subprocess.run(["rm", "-rf", temp_results_path])
        hdr_cmd = [
            "python", 
            "-m",
            "anything_in_anyscene.env_lighting.infer", 
            "--Float_Stack1",
            "--resize",
            "--photomatix_path", "/all/env_lighting/PhotomatixCL",
            "--data_root", input_path, 
            "--save_path", temp_results_path
        ]
        run_cmd(hdr_cmd, logger)
        
        pano_cmd = [
            "python", 
            "-m",
            "anything_in_anyscene.env_lighting.pano", 
            "--input_path", input_path, 
            "--hdr_path", temp_results_path
        ]
        run_cmd(pano_cmd, logger)

def run_placement(input_path, placement_path, logger):
    cfg = "config_highway.yaml"
    path_tail = os.path.basename(input_path)
    track_file = os.path.join(input_path, "../campose", path_tail, "images.txt")
    calibration_file = os.path.join(input_path, "../calibration.json")
    label_dir = os.path.join(input_path, "../pred_json", path_tail)
    lidar_dir = os.path.join(input_path, "../lidar", path_tail)

    if check_file_exist(os.path.join(input_path, "placement_results/points/point0/xyz.txt")):
        logger.info("skip placement...")
    else:
        placement_cmd = [
            "python3",
            "-m",
            "anything_in_anyscene.placement.main",
            "--cfg-file", cfg,
            "--mesh-file", pkg_resources.resource_filename("anything_in_anyscene.core", "models_list.txt"),
            "--src-dir", os.path.join(input_path, "cam0"),
            "--track-file", track_file,
            "--label-dir", label_dir,
            "--lidar-dir", lidar_dir,
            "--calibration", calibration_file,
            "--save-dir", placement_path,
            "--debug-dir", os.path.join(placement_path, "debug"),
            "--device", "0",
            "--check-collision",
            "--fix-pose"
        ]
        run_cmd(placement_cmd, logger)

    
def run_render(input_path, placement_path, hdr_path, logger):
    # Check if placement directory exists
    if not os.path.isdir(os.path.join(placement_path, "points/point0")):
        logger.error(f"{os.path.join(placement_path, 'points/point0')} does not exist.")
        return

    # Check if images_fixby_optical.txt exists
    if not check_file_exist(os.path.join(placement_path, "points/point0/images_fixby_optical.txt")):
        logger.warning(f"{os.path.join(placement_path, 'points/point0/images_fixby_optical.txt')} does not exist. Using old track file.")
        track_file = os.path.join(input_path, "..", "campose", os.path.basename(input_path), "images.txt")
    else:
        logger.info(f"track file use {os.path.join(placement_path, 'points/point0/images_fixby_optical.txt')}")
        track_file = os.path.join(placement_path, "points/point0/images_fixby_optical.txt")

    # Check vulkan_render
    if not check_file_exist("/all/vulkan_render/RaytracingForge/build/RaytracingForge.cpython-39-x86_64-linux-gnu.so"):
        logger.info("build vulkan_render...")
        os.chdir("/all/vulkan_render")
        subprocess.run(["chmod", "a+x", "./build_RaytracingForge.sh"])
        subprocess.run(["./build_RaytracingForge.sh"])

    if not check_file_exist(os.path.join(input_path, "render_results/point0/pre_result/out.mp4")):
        render_cmd = [
            "python",
            "-m",
            "anything_in_anyscene.pytorch_render.apps.render_seq",
            "--calib-file", os.path.join(input_path, "..", "calibration.json"),
            "--mesh-list", pkg_resources.resource_filename("anything_in_anyscene.core", "models_list.txt"),
            "--img-folder", os.path.join(input_path, "cam0"),
            "--track-file", track_file,
            "--sundir-file", os.path.join(hdr_path, "sun_peak_dir.json"),
            "--placement-folder", os.path.join(placement_path, "points/point0"),
            "--save-dir", os.path.join(input_path, "render_results/point0"),
            "--render-first", "10000",
            "--save-shadow",
            "--save4gan",
            "--clear"
        ]
        run_cmd(render_cmd, logger)

        os.chdir(os.path.join(input_path, "render_results/point0/pre_result"))
        # Use FFmpeg to create the final_out.mp4 video
        ffmpeg_cmd = [
            "ffmpeg",
            "-framerate", "12",
            "-pattern_type", "glob",
            "-i", "*.png",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "out.mp4"
        ]
        subprocess.run(ffmpeg_cmd)
        os.chdir("/all")

def run_stytr(input_path, placement_path, hdr_path, logger):
    # Create a symbolic link
    if not os.path.islink(os.path.join(input_path, "render_results/raw")):
        os.symlink(os.path.join(input_path, "cam0"), os.path.join(input_path, "render_results/raw"))

    if not check_file_exist(os.path.join(input_path, "render_results/point0/post_result/final_out.mp4")):
        # Run the 'infer.py' script
        infer_cmd = [
            "python",
            "-m",
            "anything_in_anyscene.style_transfer.infer",
            "--input", os.path.join(input_path, "render_results/point0")
        ]
        run_cmd(infer_cmd, logger)

        # Change directory to /render_results/post_result
        os.chdir(os.path.join(input_path, "render_results/point0/post_result"))

        # Use FFmpeg to create the final_out.mp4 video
        ffmpeg_cmd = [
            "ffmpeg",
            "-framerate", "12",
            "-y",
            "-pattern_type", "glob",
            "-i", "*.png",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "final_out.mp4"
        ]
        subprocess.run(ffmpeg_cmd)
        os.chdir("/all")

def gpu_processor(context: dict, **kwargs):  
    logger = context["logger"]

    video_path = kwargs.get('video_path')

    do_count = 0
    success_count = 0
    all_count = 0
    all_time = 0

    logger.info("--------------------------------------------------")
    logger.info("          checking all 3d model files...")
    
                
    model_files = load_txt("anything_in_anyscene.core", "models_list.txt")
    for model_file in model_files:
        model_file = model_file.strip()
        logger.info(model_file)
        check_file(model_file)

    logger.info("----------------------Done------------------------")

    data_list = [video_path]

    for line in data_list:
        start = time.time()
        
        input_path = line.strip()
        logger.info(f"clip path: {input_path}")
        if "c-" not in input_path:
            logger.warning("video path not valid, skip gpu process")
            return
        
        hdr_path = os.path.join(input_path, "hdr_results")
        placement_path = os.path.join(input_path, "placement_results")
        render_path = os.path.join(input_path, "render_results")
        anno_path = os.path.join(input_path, "annotation_results")

        if not os.path.exists(os.path.join(input_path, "cam2")):
            logger.warning(f"{os.path.join(input_path, 'cam2')} does not exist.")
            all_count += 1
            continue

        if not os.path.exists(os.path.join(input_path, "cam0")):
            logger.warning(f"{os.path.join(input_path, 'cam0')} does not exist.")
            all_count += 1
            continue
        
        if not os.path.exists(hdr_path):
            os.mkdir(hdr_path)
        if not os.path.exists(placement_path): 
            os.mkdir(placement_path)
        if not os.path.exists(render_path):
            os.mkdir(render_path)
        if not os.path.exists(anno_path):
            os.mkdir(anno_path)    
        
        run_hdr(input_path, hdr_path, logger)
                
        run_env_hdr(input_path, hdr_path, logger)
        
        run_placement(input_path, placement_path, logger)
        
        run_render(input_path, placement_path, hdr_path, logger)
        
        run_stytr(input_path, placement_path, hdr_path, logger)

        end = time.time()
        success_count += 1
        do_count += 1
        all_count += 1
        once_time = int(end - start)
        all_time += once_time
        avg_time = int(all_time / do_count / 60)
        logger.info("--------------------------------------------------")
        logger.info(f"done {input_path}, time spent: ({once_time}s) avg: ({avg_time}min)")
        logger.info(f"now SUCCESS clip is {do_count}/{success_count}/{all_count}")
        logger.info("--------------------------------------------------")
        time.sleep(1)

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
  
    context = {
        "local": True,
        "logger": logger
    }
    
    data_list = load_txt("anything_in_anyscene.core", "datalist.txt")
    for line in data_list:
        gpu_processor(context, video_path = line.strip())
