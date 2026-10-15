"""The Adam store page-cache window, checked against kernel residency through mincore."""

import ctypes
import mmap
import os

import pytest

from miles_plugins.optimizers import nvme_stream


def _resident_pages(fd: int) -> int:
    """Pages of ``fd`` resident in the host page cache, from mincore over a shared mapping."""
    size = os.fstat(fd).st_size
    libc = ctypes.CDLL(None, use_errno=True)
    with mmap.mmap(fd, size, access=mmap.ACCESS_WRITE) as mapping:
        address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
        pages = (ctypes.c_ubyte * ((size + mmap.PAGESIZE - 1) // mmap.PAGESIZE))()
        assert libc.mincore(ctypes.c_void_p(address), ctypes.c_size_t(size), pages) == 0, ctypes.get_errno()
        return sum(page & 1 for page in pages)


_FILE_BYTES = 256 * mmap.PAGESIZE


def require_evictable(directory) -> None:
    """Skip unless files under ``directory`` can leave the page cache, which tmpfs pages, being the storage, cannot."""
    if not hasattr(os, "posix_fadvise"):
        pytest.skip("page-cache control is Linux-only")
    fd = os.open(os.path.join(directory, "evictable-probe.bin"), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        os.pwrite(fd, b"\x01" * mmap.PAGESIZE, 0)
        # Raw syscalls rather than the store's helper, so a broken helper fails the tests instead of skipping them.
        os.fdatasync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        evictable = _resident_pages(fd) == 0
    finally:
        os.close(fd)
    if not evictable:
        pytest.skip(f"{directory} keeps its pages resident (tmpfs), so nothing can be evicted")


@pytest.fixture
def written_file(tmp_path):
    """Opens a file freshly written through the page cache, on a filesystem that can evict it."""
    require_evictable(tmp_path)
    fds = []

    def write(name: str) -> int:
        fd = os.open(str(tmp_path / name), os.O_RDWR | os.O_CREAT, 0o600)
        fds.append(fd)
        os.pwrite(fd, b"\x01" * _FILE_BYTES, 0)
        return fd

    yield write
    for fd in fds:
        os.close(fd)


def test_page_cache_window_drops_the_oldest_file_before_a_release_past_its_cap_returns(written_file):
    window = nvme_stream._PageCacheWindow(cap_bytes=_FILE_BYTES * 3 // 2)
    older, newer = written_file("older.bin"), written_file("newer.bin")
    window.release(older, _FILE_BYTES)
    window.release(newer, _FILE_BYTES)

    assert _resident_pages(older) == 0
    window.drain()
    assert _resident_pages(newer) == 0
    assert os.pread(older, 1, 0) == b"\x01"


def test_zero_cap_drops_each_file_before_its_release_returns(written_file):
    window = nvme_stream._PageCacheWindow(cap_bytes=0)
    fd = written_file("bucket.bin")
    window.release(fd, _FILE_BYTES)

    assert _resident_pages(fd) == 0


def test_page_cache_window_raises_a_failed_writeback_on_the_caller():
    """The writer thread's errors must not vanish with it."""
    if not hasattr(os, "posix_fadvise"):
        pytest.skip("page-cache control is Linux-only")
    window = nvme_stream._PageCacheWindow(cap_bytes=1 << 30)

    # Not an open descriptor, so the writeback fails with EBADF: on this release if the writer is already done, else on
    # the drain.
    with pytest.raises(OSError):
        window.release(1 << 30, 1)
        window.drain()
