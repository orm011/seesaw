import os
import subprocess
import filelock

PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.realpath(PROJECT_ROOT_DIR + "/../scripts/")
TMPDIR = os.environ.get("TMPDIR", "/tmp/")
DATA_CACHE_DIR = f"{TMPDIR}/seesaw_data_cache/"
FILE_LOCK = filelock.FileLock(f"{TMPDIR}/seesaw.lock")


def resolve_path(path):
    return os.path.normpath(os.path.realpath(os.path.expanduser(path)))


def parallel_copy(base_dir, cache_dir, rel_path):
    """
    will copy rel_path from base dir to cache dir in parallel, preserving the path,
    will avoid repeated copying if already done before
    """
    ipath = f"{base_dir}/{rel_path}"

    assert os.path.isfile(ipath), "input does not exist"

    opath = f"{cache_dir}/{rel_path}"
    istat = os.stat(ipath)

    with FILE_LOCK:  # global lock for shared path structure
        os.makedirs(os.path.dirname(opath), exist_ok=True)

    with filelock.FileLock(opath + ".lock"):  # only one worker does this
        if os.path.exists(opath):
            ostat = os.stat(opath)
            if istat.st_size == ostat.st_size and istat.st_mtime <= ostat.st_mtime:
                print("path already copied, returning fast")
                return opath

        print(f"copying {ipath} to cache path {opath}... with parallel_copy.bash")
        script_path = f"{SCRIPTS_DIR}/parallel_copy.bash"
        assert os.path.isfile(script_path)
        # path doesn't exist, it is not fully copied, or is stale
        cprun = subprocess.run(
            ["bash", script_path, ipath, opath],
            capture_output=True,
            universal_newlines=True,
        )

        if cprun.returncode == 0:
            print("... done copying")
            print(cprun.stdout)
        else:
            print("script exited with non-zero status: ", cprun.returncode)
            print(cprun.stderr)
            assert False

        assert os.path.exists(opath)
        ostat = os.stat(opath)
        assert ostat.st_size == istat.st_size
        assert istat.st_mtime <= ostat.st_mtime
        return opath
