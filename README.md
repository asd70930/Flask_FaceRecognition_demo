
## Install nvidia driver, cuda, cudnn

very important!!!! Can't skip this step 
* STEP1, install Nvidia-driver
    ```sh
        check out Document/Nvidia GPU Driver 安裝說明.odt
    ``` 
* STEP2, install cuda, cudnn
    ```sh
        check out Document/CUDA_CUDNN安裝說明.odt
    ``` 
    
## Install on Ubuntu 

* STEP1, install python3.6 (can use higher as python3.7).
    ```sh
    sudo add-apt-repository ppa:jonathonf/python-3.6
    sudo apt-get update
    sudo apt-get install python3.6
    sudo apt install python3-pip
    pip3 install --upgrade pip
    ```

`!!! Recommend to use virtual environment !!!`

* STEP2, install virtual environment (venv).
    ```sh
        pip3 install virtualenv
    ```
    
* STEP3, initialize venv, you will create a file venv.
    ```sh
        virtualenv -p /usr/bin/python3.6 venv
    ```

* STEP4, activate venv, make sure always activating venv before next steps!!!
    ```sh
        source venv/bin/activate
    ```  
  
* STEP5, install needed lib.
    ```sh
        pip install -r requirements.txt
    ```   

* STEP6, activate web server.
    ```sh
        python face_web/faces_web.py 
    ``` 
    
* STEP7, When you not using this project, depart from virtualenv.
    ```sh
        deactivate
    ``` 
## Or docker build for testing 

* STEP1, install docker 19.03
    ```sh
        sudo apt-get update
        
        sudo apt-get install \
            apt-transport-https \
            ca-certificates \
            curl \
            gnupg-agent \
            software-properties-common
        
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
        
        sudo add-apt-repository \
           "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
           $(lsb_release -cs) \
           stable"
           
        sudo apt-get update
        
        sudo apt-get install docker-ce docker-ce-cli containerd.io
        
        
        
    ``` 

* STEP2, install Nvidia Container Toolkit
    ```sh
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
        
        curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
        
        curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
        
        sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
        
        sudo systemctl restart docker
    ```
* STEP3, build docker image
    ```sh
        docker build -t face_recognition_web .
    ```
 
* STEP4, run docker image
    ```sh
        docker run --gpus all --rm -it -p 5000:5000 face_recognition_web 
    ``` 
## How to use

Open browser Chrome or Firefox, there two page for testing 
check out more information in face_web file.

127.0.0.1:5000

    Check out the video, demo/demo1.ogv.
    It will show you how to use.  
127.0.0.1:5000/white_black 
   
    Check out the video, demo/demo2.ogv.
    It will show you how to use. 
