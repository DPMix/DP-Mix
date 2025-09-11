# DP-Mix

This repository contains the artifact for the paper titled "DP-Mix: Differentially Private Routing in Mix Networks," accepted for publication at ACSAC 2025.

## Initial setup and dependencies
You can execute the code on any standard laptop or workstation running Ubuntu 18.04 or higher. It is compatible with Python 3.8.10. Importantly, the artifact uses the same configuration as in the paper but with a reduced number of iterations, making it suitable for faster execution so that it can be readily run on a personal laptop or public infrastructure such as Google Colab.

Furthermore, the artifact has been optimized to run on systems with 16\,GB of RAM and 50\,GB of available disk space. These specifications allow users to reproduce results efficiently without requiring access to high-performance computing environments. We have also tested running the Git repository on Google Colab, which users may choose as an alternative.  

If you are interested in running the artifact on a laptop, please ensure that your system satisfies the following requirements: Ubuntu 18.04 or higher, Python 3.8.10, a minimum of 16\,GB of RAM, and at least 50\,GB of available disk space.  

All required dependencies for execution are listed in the `requirements.txt` file included in the repository and are summarized below:  

- matplotlib  
- numpy  
- pandas  
- pulp  
- scipy  
- simpy  
- tabulate  

However, to install all requirements automatically, you only need to run the following command once from the command line or within Google Colab before executing the project:  

`bash install.sh`

## Hardware Requirements
The code has been tested on standard hardware with 16\,GB of RAM, 8 cores, and 50\,GB of available disk space. Alternatively, the artifact can be executed on Google Colab. To do so, you will need a Google account. Once signed in, you can clone the Git repository and run the code for free by following the instructions provided below.




# Project Structure

```text
.
artifact
├── dataset.pkl                                   # Dataset used in DP-Mix (including NYM and RIPE data)
├── DPMIX_Functions.py                            # This .py file includes all the functions needed to directly run the main
│                                                   experiments of DP-Mix  and generates the corresponding figures.
├── DPMIX.py                                      # DPMIX.py contains a comprehensive set of functions for analyzing DP-Mix.
├── FCP_Functions.py                              # This .py file includes strategies that a mixnode adversary might consider
│                                                    when corrupting mindnodes in mixnets.
├── Main.py                                       # This file provides instructions regarding how to run the experiments described
│                                                    in the main body of the paper.
├── Message_Genartion_and_mix_net_processing_.py  # This Python file, on behalf of clients, generates the messages to be sent
│                                                    to the mixnet.
├── Message_.py                                   # Simulates the messages generated and sent by the clients.
├── Mix_Node_.py                                  # Simulates using discrete event simulation, a mixnode in mixnets.
├── NYM.py                                        # This .py file provides the main simulation components necessary to simulate
│                                                     a mixnet as used in DP-Mix.
├── PLOTTER.py                                    # To plot the figures.
├── Routing.py                                    # This function helps to model the routing approaches in DP-Mix.
└── Sim.py                                        # This .py file also includes the necessary simulation components
                                                    for reproducing simulations of DP-Mix.

Claims
├── Claim1
│   ├── claim.txt            # Text description of Claim 1
│   ├── expected             # Expected results/figures for Claim 1
│   │   ├── Fig_5a.png       # Expected figure 5a
│   │   ├── Fig_5b.png       # Expected figure 5b
│   │   ├── Fig_5c.png       # Expected figure 5c
│   │   └── Fig_5d.png       # Expected figure 5d
│   └── run_E_1.sh           # Script to run experiment for Claim 1
│
├── Claim2
│   ├── claim.txt            # Text description of Claim 2
│   ├── expected             # Expected results/figures for Claim 2
│   │   ├── Fig_6a.png
│   │   ├── Fig_6b.png
│   │   ├── Fig_6c.png
│   │   └── Fig_6d.png
│   └── run_E_2.sh           # Script to run experiment for Claim 2
│
├── Claim3
│   ├── claim.txt            # Text description of Claim 3
│   ├── expected             # Expected results/figures for Claim 3
│   │   ├── Fig_7a.png
│   │   ├── Fig_7b.png
│   │   ├── Fig_7c.png
│   │   └── Fig_7d.png
│   └── run_E_3.sh           # Script to run experiment for Claim 3
│
├── Claim4
│   ├── claim.txt            # Text description of Claim 4
│   ├── expected             # Expected results/figures for Claim 4
│   │   ├── Fig_8a.png
│   │   ├── Fig_8b.png
│   │   ├── Fig_8c.png
│   │   └── Fig_8d.png
│   └── run_E_4.sh           # Script to run experiment for Claim 4
│
├── Claim5
│   ├── claim.txt            # Text description of Claim 5
│   ├── expected             # Expected results/figures for Claim 5
│   │   ├── Fig_9a.png
│   │   ├── Fig_9b.png
│   │   ├── Fig_9c.png
│   │   └── Fig_9d.png
│   └── run_E_5.sh           # Script to run experiment for Claim 5
│
├── Claim_T
│   ├── claim.txt            # Text description of ruining experiments leading to generating Table 1 
│   └── run_E_T.sh           # Script to run Table 1 experiment
│
├── install.sh               # Installation script for environment setup
├── LICENSE                  # License information  
├── README.md                # Project overview & instructions
├── requirements.txt         # Provides the required argument used by install.sh for installing the dependencies at
│                              the beginning of running the artifact.
└── use.txt	                 # Intended use + limitations


```






# Evaluation Workflow

## Major Claims

- **(C1):** The first claim concerns the trend shown in Figure 5. Across all settings and scenarios, when the variable on the x-axis (the privacy parameter \( \varepsilon \)) increases, the corresponding latency decreases. This claim is supported by Experiment E1, which generates Figure 5 and demonstrates this trend. See `./Claims/Claim1/claim.txt` for more details.  

- **(C2):**  This claim concerns the trend shown in Figure 6. Across all settings and scenarios, when the variable on the x-axis (the privacy parameter \( \varepsilon \)) increases, the corresponding entropy decreases. This claim is supported by Experiment E2, which generates Figure 6 and demonstrates this trend. See `./Claims/Claim2/claim.txt` for more details.  

- **(C3):**  This claim concerns the trend shown in Figure 7. Across all settings and scenarios, when the variable on the x-axis (the privacy parameter \( \varepsilon \)) increases, the entropy of messages shown on the y-axis decreases. This claim is supported by Experiment E3, which generates Figure 7 and demonstrates this trend. See `./Claims/Claim3/claim.txt` for more details.  

- **(C4):** This claim concerns the trend shown in Figure 8. Across all settings and scenarios, when the variable on the x-axis (the privacy parameter \( \varepsilon \)) increases, the fraction of corrupted paths (FCP) shown on the y-axis increases. This claim is supported by Experiment E4, which generates Figure 8 and demonstrates this trend. See `./Claims/Claim4/claim.txt` for more details.  

- **(C5):** This claim concerns the trend shown in Figure 9. Across all settings and scenarios, when the variable on the x-axis (the adversary budget \( C/N \)) increases, the fraction of corrupted paths (FCP) shown on the y-axis increases. This claim is supported by Experiment E5, which generates Figure 9 and demonstrates this trend. See `./Claims/Claim5/claim.txt` for more details.







## Experiments

## E1: [Reproducing Fig. 5; verifying Claim C1] [10 min]  

- To run this experiment, it is important to first change the directory to `./Claims/Claim1/`. You can then reproduce this experiment by running:  

`bash ./run_E_1.sh`  

- After execution, the results will be saved as **Fig_5a.png**, **Fig_5b.png**, **Fig_5c.png**, and **Fig_5d.png** in the `artifact/Figures/` directory.  

- Verification:  
You can compare the results with Figure 5 in the paper (shown below). Please note, however, that the experiment may not reproduce exactly the same figures due to adjustments made for execution on a personal laptop or Google Colab (e.g., fewer iterations or modified parameters). For verification, focus on the claim itself, particularly the observed trends—increasing or decreasing values along the x-axis.  

Alternatively, the pre-generated figures provided in `./Claims/Claim1/expected/` offer a more reliable point of comparison, as they were produced using the current artifact with a reduced number of iterations, yielding outputs more consistent with those obtained when running the artifact.


<img width="1419" height="280" alt="image" src="https://github.com/user-attachments/assets/816e5c31-cf2f-40f3-aded-ecf20d85926f" />



## E2: [Reproducing Fig. 6; verifying Claim C2] [10 min]  

- To run this experiment, it is important to first change the directory to `./Claims/Claim2/`. You can then reproduce this experiment by running:  

`bash ./run_E_2.sh`  

- After execution, the results will be saved as **Fig_6a.png**, **Fig_6b.png**, **Fig_6c.png**, and **Fig_6d.png** in the `artifact/Figures/` directory.  

- Verification:  
You can compare the results with Figure 6 in the paper (shown below). Please note, however, that the experiment may not reproduce exactly the same figures due to adjustments made for execution on a personal laptop or Google Colab (e.g., fewer iterations or modified parameters). For verification, focus on the claim itself, particularly the observed trends—increasing or decreasing values along the x-axis.  

Alternatively, the pre-generated figures provided in `./Claims/Claim2/expected/` offer a more reliable point of comparison, as they were produced using the current artifact with a reduced number of iterations, yielding outputs more consistent with those obtained when running the artifact.

<img width="1437" height="267" alt="image" src="https://github.com/user-attachments/assets/f6a174e3-92c4-436c-a6de-36e09235d751" />






## E3: [Reproducing Fig. 7; verifying Claim C3] [20 min]  

- To run this experiment, it is important to first change the directory to `./Claims/Claim3/`. You can then reproduce this experiment by running:  

`bash ./run_E_3.sh`  

- After execution, the results will be saved as **Fig_7a.png**, **Fig_7b.png**, **Fig_7c.png**, and **Fig_7d.png** in the `artifact/Figures/` directory.  

- Verification:  
You can compare the results with Figure 7 in the paper (shown below). Please note, however, that the experiment may not reproduce exactly the same figures due to adjustments made for execution on a personal laptop or Google Colab (e.g., fewer iterations or modified parameters). For verification, focus on the claim itself, particularly the observed trends—increasing or decreasing values along the x-axis.  

Alternatively, the pre-generated figures provided in `./Claims/Claim3/expected/` offer a more reliable point of comparison, as they were produced using the current artifact with a reduced number of iterations, yielding outputs more consistent with those obtained when running the artifact.





<img width="1429" height="249" alt="image" src="https://github.com/user-attachments/assets/4d08e2f0-6f1e-4cf0-ad5d-7e0c76d15a0a" />


## E4: [Reproducing Fig. 8; verifying Claim C4] [15 min]  

- To run this experiment, it is important to first change the directory to `./Claims/Claim4/`. You can then reproduce this experiment by running:  

`bash ./run_E_4.sh`  

- After execution, the results will be saved as **Fig_8a.png**, **Fig_8b.png**, **Fig_8c.png**, and **Fig_8d.png** in the `artifact/Figures/` directory.  

- Verification:  
You can compare the results with Figure 8 in the paper (shown below). Please note, however, that the experiment may not reproduce exactly the same figures due to adjustments made for execution on a personal laptop or Google Colab (e.g., fewer iterations or modified parameters). For verification, focus on the claim itself, particularly the observed trends—increasing or decreasing values along the x-axis.  

Alternatively, the pre-generated figures provided in `./Claims/Claim4/expected/` offer a more reliable point of comparison, as they were produced using the current artifact with a reduced number of iterations, yielding outputs more consistent with those obtained when running the artifact.


<img width="1368" height="256" alt="image" src="https://github.com/user-attachments/assets/c504f2ac-6a33-438c-8cad-a0d55a1c7d32" />





## E5: [Reproducing Fig. 9; verifying Claim C5] [15 min]  

- To run this experiment, it is important to first change the directory to `./Claims/Claim5/`. You can then reproduce this experiment by running:  

`bash ./run_E_5.sh`  

- After execution, the results will be saved as **Fig_9a.png**, **Fig_9b.png**, **Fig_9c.png**, and **Fig_9d.png** in the `artifact/Figures/` directory.  

- Verification:  
You can compare the results with Figure 9 in the paper (shown below). Please note, however, that the experiment may not reproduce exactly the same figures due to adjustments made for execution on a personal laptop or Google Colab (e.g., fewer iterations or modified parameters). For verification, focus on the claim itself, particularly the observed trends—increasing or decreasing values along the x-axis.  

Alternatively, the pre-generated figures provided in `./Claims/Claim5/expected/` offer a more reliable point of comparison, as they were produced using the current artifact with a reduced number of iterations, yielding outputs more consistent with those obtained when running the artifact.



<img width="1412" height="274" alt="image" src="https://github.com/user-attachments/assets/dd04e486-16a7-4290-99ba-b77ded2c4cd6" />




## E*: [Table 1] [20 min]  

Note that this experiment does not support any specific claim but is included for completeness, ensuring that Table 1 in the paper is reproducible. Together with the previous experiments, this makes all experiments in the paper reproducible.  

- To run this experiment, first change the directory to `./Claims/Claim_T/`. You can then reproduce this experiment by running:  

`bash ./run_E_T.sh`  






<img width="1330" height="359" alt="image" src="https://github.com/user-attachments/assets/00ff05ea-68bb-405d-824d-50e6e653a353" />







## Additional Notes

- This artifact has been created to reproduce the results presented in the paper “DP-Mix: Differentially Private Routing in Mix Networks”, accepted for publication at ACSAC 2025, and to provide an environment that makes the code usable for future research. You can execute the code on any standard laptop or workstation running Ubuntu 18.04 or higher. It is compatible with Python 3.8.10. Alternatively, the repository can be cloned and executed in a Google Colab environment.  

- Importantly, the artifact includes precisely the same configurations and settings used in the original DP-Mix evaluation. The only exception is that the number of iterations and some minor parameters have been scaled down to ensure feasibility on standard hardware. These configuration parameters can be reviewed and, where appropriate, modified in the initialization of `DP_Mix_Functions.py`. This adjustment ensures that the artifact remains practical for local execution, while still supporting future extensions where researchers may wish to adapt our approach to scenarios not explicitly tested in DP-Mix.  

- That said, modifying parameters requires a deeper understanding of mixnets. Mixnets are complex systems in which changes to a single parameter may influence others. In many cases, certain parameter combinations may be incompatible or may significantly affect the overall results. Additionally, several parameters are initialized based on prior work, and their default values were chosen to support meaningful comparisons. As such, arbitrary modifications may not yield valid outcomes.  

- However, the following parameters in the initialization of `DP_Mix_Functions.py` can be safely modified within specific intervals:  

1- `self.Iterations`: can be increased up to 30 to improve accuracy; note that this change increases computational cost exponentially.
  
2- `self.num_targets`: specifies the number of target messages in the simulations; can be set to any integer in [20,200].  

3- `self.run`: defines the duration of each simulation time slot; can be set to any real value in [0.05,1.0]. 
 
4- `self.delay1`: represents the average delay imposed on each message upon entering mixnodes; can be set to any real value in [0.01,0.08]. 
 

- Other parameters should not be modified, as they are tied to fixed design assumptions in mixnets. Altering them may lead to execution errors or invalid experimental results. If users wish to modify such parameters, we recommend contacting the authors directly for further guidance. 


 

- Execution notes: If the following warnings appear during execution, they can be safely ignored:  
```text
1) Gdk-CRITICAL **: 13:07:09.758: gdk_cursor_new_for_display: assertion 'GDK_IS_DISPLAY (display)' failed 

2) UserWarning: This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.
```


