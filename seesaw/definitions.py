import os
import subprocess
import filelock

def resolve_path(path):
    return os.path.normpath(os.path.realpath(os.path.expanduser(path)))

PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.realpath(PROJECT_ROOT_DIR + "/../scripts/")
TMPDIR = resolve_path(os.environ.get("TMPDIR", "/tmp/"))
DATA_CACHE_DIR = f"{TMPDIR}/seesaw_data_cache/"
os.makedirs(DATA_CACHE_DIR, exist_ok=True)
assert os.path.isdir(DATA_CACHE_DIR)

FILE_LOCK = filelock.FileLock(f"{TMPDIR}/seesaw.lock")


def parallel_copy(ipath, opath):
    """
    will copy rel_path from base dir to cache dir in parallel, preserving the path,
    will avoid repeated copying if already done before
    """
    ipath = resolve_path(ipath)
    opath = resolve_path(opath)
    assert os.path.isfile(ipath), "input does not exist"

    odir = os.path.dirname(opath)
    istat = os.stat(ipath)

    with FILE_LOCK:  # global lock for shared path structure
        os.makedirs(odir, exist_ok=True)

    with filelock.FileLock(opath + ".lock"):  # only one worker does this
        if os.path.exists(opath):
            ostat = os.stat(opath)
            if istat.st_size == ostat.st_size and istat.st_mtime <= ostat.st_mtime:
                print("path already copied, returning fast")
                return opath

        print(f"copying {ipath} to  {opath} with parallel_copy")
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


class FsCache:
    def __init__(self, cache_base):
        self.base_dir = resolve_path(cache_base)

    def _get_cache_path(self, ipath):
        ipath = resolve_path(ipath)
        return f"{self.base_dir}/{ipath}"

    def get(self, path):
        cpath = self._get_cache_path(path)
        parallel_copy(path, cpath)
        return cpath


FS_CACHE = FsCache(DATA_CACHE_DIR)
