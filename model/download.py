import hashlib
import os
import shutil
import sys
import tempfile
from urllib.request import urlopen, Request

# tqdm模拟类（如果没有安装tqdm）
class tqdm:
    def __init__(self, total=None, disable=False):
        self.total = total
        self.n = 0
        self.disable = disable

    def update(self, n):
        if self.disable:
            return
        self.n += n
        progress = 100 * self.n / float(self.total) if self.total else self.n
        sys.stderr.write(f"\r{progress:.1f}%")
        sys.stderr.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.disable:
            sys.stderr.write('\n')

# 下载文件的函数
def download_url_to_file(url, dst, hash_prefix=None, progress=True):
    """
    从URL下载文件并保存到本地路径，同时支持哈希校验和下载进度条。
    
    :param url: 下载文件的URL
    :param dst: 本地保存的路径
    :param hash_prefix: 如果提供，文件的SHA256哈希值应该以该前缀开头（可选）
    :param progress: 是否显示下载进度条，默认显示
    """
    # 请求并获取文件的大小
    req = Request(url, headers={"User-Agent": "python"})
    with urlopen(req) as u:
        file_size = int(u.info().get("Content-Length", 0))

    # 设置临时文件路径
    dst = os.path.expanduser(dst)
    dst_dir = os.path.dirname(dst)
    os.makedirs(dst_dir, exist_ok=True)

    # 临时保存文件
    with tempfile.NamedTemporaryFile(delete=False, dir=dst_dir) as tmp_file:
        try:
            # 初始化进度条（如果需要）
            sha256 = hashlib.sha256() if hash_prefix else None
            with tqdm(total=file_size, disable=not progress, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                while chunk := u.read(8192):
                    tmp_file.write(chunk)
                    if sha256:
                        sha256.update(chunk)
                    pbar.update(len(chunk))

            # 哈希校验
            if sha256 and sha256.hexdigest()[:len(hash_prefix)] != hash_prefix:
                raise RuntimeError(f'Invalid hash value: expected prefix "{hash_prefix}", got "{sha256.hexdigest()[:len(hash_prefix)]}"')

            # 移动临时文件到目标路径
            shutil.move(tmp_file.name, dst)
        except Exception as e:
            if os.path.exists(tmp_file.name):
                os.remove(tmp_file.name)
            raise e

