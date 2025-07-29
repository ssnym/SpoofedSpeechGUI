# Spoofed Speech Detection - PyQt6 GUI Application

This repository provides GUI-based desktop application built using **PyQt6** to demonstrate spoofed speech detection. 

This applications uses two spoofed speech detection models :  **[AASIST](https://arxiv.org/abs/2110.01200)**  and **[RawNet](https://arxiv.org/abs/2011.01108)**

The third model One Class Classifier is being trained so in GUI application the value for One-Class will be **N/A** for now

The pretrained models are provided by **[Shilpa](https://github.com/shilpac131)**

---

# Getting Started


## Method 1 : Using Conda Environment

### <b> Prerequisites </b>

* **Conda** need to installed on your system . Please follow the guide to install conda on your system [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

### Steps

1. First, clone the repository locally
```bash
git clone https://github.com/ssnym/SpoofedSpeechGUI

cd SpoofedSpeechGUI
```

2. Create and activate the conda environment

``` bash
# Create a conda environment from the environment.yml file
conda env create -f environment.yml

# Activate the conda environment
conda activate <env_name> 

```

3. Run the main script to launch the GUI

```bash
python main.py
```
---

## Method 2 : Using Docker

### <b> Pre-requisites </b>

Docker need to installed on your system . Please follow the guide to install conda on your system [Docker](https://docs.docker.com/engine/install/)

### **Steps**

1. First, clone the repository locally
```bash
git clone https://github.com/ssnym/SpoofedSpeechGUI

cd SpoofedSpeechGUI
```

2. Start the docker using command

```bash
sudo systemctl start docker
```

3. Build the Docker image. A Dockerfile defines the instructions to build a Docker image with all dependencies and configurations for an application. To build the docker image run

```bash
docker build -t <image_name> .
```

4. The audio files need to be placed in the **'~/audio_files'** directory on your computer first. The app will see these files in its **'/data directory'**.<br>
Run the following command to run the docker image. Make sure to replace `<image_name>` with the name you chose in the previous step.

```bash
xhost +local:docker && docker run -it --rm \
--env="DISPLAY" \
--env="QT_X11_NO_MITSHM=1" \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
--env="XAUTHORITY=$XAUTHORITY" \
--volume="$XAUTHORITY:$XAUTHORITY" \
--env PULSE_SERVER=unix:/run/user/$(id -u)/pulse/native \
--volume /run/user/$(id -u)/pulse:/run/user/$(id -u)/pulse \
--volume ~/.config/pulse/cookie:/root/.config/pulse/cookie \
--volume "$(pwd)":/app \
--volume "$HOME/audio_files":/data \
<image_name>
```




