#!/bin/zsh

# Update package lists
sudo apt-get update

# Install OpenJDK for python-weka-wrapper3
sudo apt-get install openjdk-11-jdk
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# Install Xvfb for CarRacing Env and swig for gym[box2d]==0.21
sudo apt install xvfb swig

# Create and activate conda environment
conda create -n symbolicRL python=3.10