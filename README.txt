How create the NVS env:
1) Download the modelnet40v1 data from http://maxwell.cs.umass.edu/mvcnn-data/
2) Put the modelnet40v1 folder inside the nvs/envs/ folder
3) Go to the New-View-Synthesis folder
4) Run 'python nvs/envs/format_data.py'
5) Run 'python test_env.py' to test the NVS env