
import os
import os.path as op
import shutil
import subprocess
from glob import glob

import pygit2
from loguru import logger


def copy(src, dst):
    if os.path.islink(src):
        linkto = os.readlink(src)
        os.symlink(linkto, dst)
    else:
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy(src, dst)


def copy_repo(src_files, dst_folder, filter_keywords):
    src_files = [
        f for f in src_files if not any(keyword in f for keyword in filter_keywords)
    ]
    dst_files = [op.join(dst_folder, op.basename(f)) for f in src_files]
    for src_f, dst_f in zip(src_files, dst_files):
        logger.info(f"FROM: {src_f}\nTO:{dst_f}")
        copy(src_f, dst_f)





def get_branch():
    return str(pygit2.Repository(".").head.shorthand)


def get_commit_hash():
    return str(pygit2.Repository(".").head.target)



def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def mkdir_p(exp_path):
    os.makedirs(exp_path, exist_ok=True)




def count_files(path):
    """
    Non-recursively count number of files in a folder.
    """
    files = glob(path)
    return len(files)


def get_host_name():
    results = subprocess.run(["cat", "/etc/hostname"], stdout=subprocess.PIPE)
    return str(results.stdout.decode("utf-8").rstrip())

