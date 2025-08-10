#
# Created/Modified on Sat May 25 2024
#
# Author: KontonGu (Jianfeng Gu)
# Copyright (c) 2024 TUM - CAPS Cloud
# Licensed under the Apache License, Version 2.0 (the "License")
#


import argparse
import os
import sys
import subprocess as sp
import shlex
import signal
import inotify.adapters

args = None

## Record gpu clients' process and record the status with each item: [name, [is_updated, pid]]
gpu_clients = {}
# ## Record gpu clients' process and record the status with each item: [name, [is_updated, pid, configuration_string]]
# gpu_clients_with_config = {}


def make_dirs_not_exist(file_dir):
    if not os.path.exists(file_dir):
        print(file_dir + " created.")
        os.makedirs(file_dir)
        

## Launch the scheduler instance
def launch_scheduler():
    cfg_dir, cfg_file_name = os.path.split(args.sched_resource_file)
    cfg_file_path = args.sched_resource_file
    if cfg_dir == '':
        cfg_dir = os.getcwd()
        cfg_file_path = os.path.join(cfg_dir, cfg_file_name)
    log_dir_tmp = os.path.join(args.log_dir, args.gpu_uuid)
    make_dirs_not_exist(log_dir_tmp)
    log_file_path = os.path.join(log_dir_tmp, "fast_scheduler.log")
    with open(log_file_path, 'w') as log_file: 
        cmd = "{} -f {} -w {} -p {}".format(args.scheduler_exec, cfg_file_path, args.window_size, args.scheduler_port)
        proc = sp.Popen(shlex.split(cmd), universal_newlines=True, bufsize=1, stdout=log_file, stderr=sp.STDOUT)
        return proc


## Prepare the ENV variables for a gpu client
def prepare_env(name, client_port, sched_port):
    client_env = os.environ.copy()
    client_env['SCHEDULER_IP'] = '127.0.0.1'
    client_env['SCHEDULER_PORT'] = str(sched_port)
    client_env['GPU_CLIENT_IP'] = '0.0.0.0'
    client_env['GPU_CLIENT_PORT'] = str(client_port)
    client_env['GPU_CLIENT_NAME'] = name
    return client_env


## Update the gpu clients based on the configuration in the file_path
def update_gpu_clients(file_path):
    with open(file_path) as f:
        lines = f.readlines()
    if not lines:
        return
    clients_num = int(lines[0])
    for _, val in gpu_clients.items():
        val[0] = False
    for i in range(1, clients_num+1):
        cname, port = lines[i].split()
        cname_port = lines[i]
        if cname_port not in gpu_clients:
            sys.stderr.write("[Scheduler] The client, cname_port(id) = \"{}\", port = {}, starts running\n".format(cname_port, port))
            sys.stderr.flush()
            log_dir_tmp = os.path.join(args.log_dir, args.gpu_uuid)
            log_file_path = os.path.join(log_dir_tmp, cname+"_gpu_client.log")
            make_dirs_not_exist(os.path.dirname(log_file_path))
            with open(log_file_path, 'w') as log_file:
                proc = sp.Popen(
                    shlex.split(args.gpu_client_exec),
                    env=prepare_env(cname, port, args.scheduler_port),
                    preexec_fn=os.setpgrp,
                    stdout = log_file,
                    stderr= sp.STDOUT
                )
                gpu_clients[cname_port] = [True, proc]
        else:
            gpu_clients[cname_port][0] = True;
    del_list = []
    for n, val in gpu_clients.items():
        if not val[0]:
            os.killpg(os.getpgid(val[1].pid), signal.SIGKILL)
            sys.stderr.write("[Scheduler] The client, cname_port(id) = {} has been deleted\n".format(n))
            sys.stderr.flush()
            del_list.append(n)
    for n in del_list:
        del gpu_clients[n]
       
            
## Note, TODO: the update is not functional for the resource configuration changes from the same client name
# def update_gpu_clients_with_configs(file_path):

def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('scheduler_exec', help="Scheudler executable file")
    parser.add_argument('gpu_client_exec', help="GPU client executable file")
    parser.add_argument('gpu_uuid', help="The GPU uuid the scheduler is attached to")
    parser.add_argument('sched_resource_file', help="The file of the scheduler resource configuration")
    parser.add_argument('gpu_clients_config_dir', help="The dir path of gpu clients' configuration ")
    parser.add_argument('--scheduler_port', type=int, default=52001, help="The socket port the scheduler is attached to")
    parser.add_argument('--window_size', type=float, default=40, help="The time window size of the scheduler.")
    parser.add_argument('--log_dir', type=str, default="/fastpod/log/", help="The time window size of the scheduler.")
    args = parser.parse_args()
    
    ### sched_resource_file: /fastpod/scheduler/config/$gpu_uuid
    ### gpu_clients_config_dir: /fastpod/scheduler/gpu_clients, specific gpu_client file: /fastpod/scheduler/gpu_clients/$gpu_uuid
    ### log_dir: /fastpod/scheduler/log, specific log files: /fastpod/scheduler/log/$gpu_uuid/<gpu_client_name>_gpu_client.log / fast_scheduler.log
    
    ## start the scheduler
    launch_scheduler()
    sys.stderr.write(f"Starting the scheduler with the port: {args.scheduler_port}\n")
    sys.stderr.flush()
    
    ## listen to the change of the gpu clients
    ino = inotify.adapters.Inotify()
    ino.add_watch(args.gpu_clients_config_dir, inotify.constants.IN_CLOSE_WRITE)
    for event in ino.event_gen(yield_nones=False):
        (_, type_names, path, filename) = event
        try:
            if filename == args.gpu_uuid:
                update_gpu_clients(os.path.join(args.gpu_clients_config_dir, args.gpu_uuid))
        except: # file content may not correct
            sys.stderr.write("[Scheduler] Catch exception in update_gpu_clients: {}\n".format(sys.exc_info()))
            sys.stderr.flush()
    

    
if __name__ == '__main__':
    os.setpgrp()
    try:
        main()
    except:
        sys.stderr.write("Catch exception: {}\n".format(sys.exc_info()))
        sys.stderr.flush()
    finally:
        for _, val in gpu_clients.items():
            os.killpg(os.getpgid(val[1].pid), signal.SIGKILL)
        os.killpg(0, signal.SIGKILL)