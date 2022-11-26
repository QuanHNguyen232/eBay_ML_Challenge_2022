import subprocess


subprocess.run('cd testing', shell=True, check=True)
subprocess.run('python train.py', shell=True, check=True)
subprocess.run('cd ..', shell=True, check=True)