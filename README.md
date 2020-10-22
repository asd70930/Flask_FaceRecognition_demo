# README

## Pre-requirement for ubuntu 

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
  
* STEP5, activate web server.
    ```sh
        python face_web/faces_web.py 
    ``` 
    
* STEP6, When you not using this project, depart from virtualenv.
    ```sh
        deactivate
    ``` 
## docker build for testing 
 
 
## How to use

Open browser Chrome or Firefox, there two page for testing 

127.0.0.1:5000

    Upload two iamge from user pc, and click 開始辨識 to start two person is the same or not.
    Check out the video, demo/demo1.ogv.
    It will show you how to use.  
127.0.0.1:5000/white_black 
  
    This page has three part.
    part1: Upload image to database. upload from user pc or IPC.
    part2: Check out all image in database.
    part3: Upload image for face recognition.
    Check out the video, demo/demo2.ogv.
    It will show you how to use. 