import moxing as mox
import os

class OBS:
    class ContextMnger:
        def __init__(self, obs, filename, mode):
            self.filename = obs.pre(obs.abspath(filename))
            self.mode = mode
            self.__file = mox.file.File(self.filename, self.mode)

        def __enter__(self):
            return self.__file

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.__file.close()

        def close(self):
            self.__file.close()

    def __init__(self, bucket_name, base_path='/', ak=None, sk=None):
        self.bucket_name = bucket_name
        self.base_path = self.rmrt(base_path)
        self.cwd = self.base_path # always abspath without s3://bucket_name/
        self.getcwd()
        self.open = lambda filename, mode: self.ContextMnger(self, filename, mode)
        if ak is not None and sk is not None:
            mox.file.set_auth(ak=ak, sk=sk)

    def rmrt(self, path):
        if path.startswith('/'):
            path = path[1:]
        return path

    def pre(self, abspath=None):
        if abspath:
            return os.path.join('s3://', self.bucket_name, self.rmrt(abspath))
        else:
            return os.path.join('s3://', self.bucket_name, self.rmrt(self.cwd))

    def getcwd(self):
        print(self.pre())
        return self.pre()

    def ls(self):
        print(mox.file.list_directory(self.pre()))

    def listdir(self, path):
        path = self.abspath(path)
        return mox.file.list_directory(self.pre(path))

    def abspath(self, path):
        # return does not contain s3://bucket_name/
        if path.startswith('/'):
            new = self.rmrt(path)
        else:
            new = self.cwd
            while path.startswith('../'):
                new = os.path.dirname(new)
                path = path[3:]
            while path.startswith('./'):
                path = path[2:]
            new = os.path.join(new, path)
        return new

    def exists(self, abspathname):
        return mox.file.exists(self.pre(abspathname))

    def cd(self, path:str):
        path = self.abspath(path)
        if not self.exists(path):
            print(path, 'dose not exists!')
            return
        self.cwd = path
        self.getcwd()

    def downloadFile(self, obsfilename, localfilename, quiet=False):
        obsfilename = self.abspath(obsfilename)
        if not self.exists(obsfilename):
            obsfilename = self.pre(obsfilename)
            print(obsfilename, 'dose not exists!')
            return
        obsfilename = self.pre(obsfilename)
        if not quiet:
            print('download from ', obsfilename)
        mox.file.copy(obsfilename, localfilename)

    def uploadFile(self, localfilename, obsfilename):
        if not os.path.exists(localfilename):
            print(localfilename, 'dose not exists!')
            return
        obsfilename = self.pre(self.abspath(obsfilename))
        print('upload to ', obsfilename)
        mox.file.copy(localfilename, obsfilename)

    def downloadDir(self, obspath, localpath):
        obspath = self.abspath(obspath)
        if not self.exists(obspath):
            obspath = self.pre(obspath)
            print(obspath, 'does not exists!')
            return
        obspath = self.pre(obspath)
        print('download from ', obspath)
        mox.file.copy_parallel(obspath, localpath)

    def uploadDir(self, localpath, obspath):
        if not os.path.exists(localpath):
            print(localpath, 'does not exists!')
            return
        obspath = self.pre(self.abspath(obspath))
        print('upload to ', obspath)
        mox.file.copy_parallel(localpath, obspath)

    def mkdir(self, path):
        path = self.abspath(path)
        mox.file.mk_dir(self.pre(path))