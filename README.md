# DP-Mix

This repository contains the artifact for the paper titled "DP-Mix: Differentially Private Routing in Mix Networks," accepted for publication at ACSAC 2025.

## Initial setup and dependencies
You can execute the code on any standard laptop or workstation running Ubuntu~18.04 or higher. It is compatible with Python~3.8.10. Importantly, the artifact uses the same configuration as in the paper but with a reduced number of iterations, making it suitable for faster execution so that it can be readily run on a personal laptop or public infrastructure such as Google Colab.

Furthermore, the artifact has been optimized to run on systems with 16\,GB of RAM and 50\,GB of available disk space. These specifications allow users to reproduce results efficiently without requiring access to high-performance computing environments. We have also tested running the Git repository on Google Colab, which users may choose as an alternative.  

If you are interested in running the artifact on a laptop, please ensure that your system satisfies the following requirements: Ubuntu~18.04 or higher, Python~3.8.10, a minimum of 16\,GB of RAM, and at least 50\,GB of available disk space.  

All required dependencies for execution are listed in the `requirements.txt` file included in the repository and are summarized below:  

- matplotlib  
- numpy  
- pandas  
- pulp  
- scipy  
- simpy  
- tabulate  

However, to install all requirements automatically, you only need to run the following command once from the command line or within Google Colab before executing the project:  

<pre> bash install.sh <pre>





