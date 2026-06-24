"""测试立绘库管理 API（v2.51）：CRUD + 图片上传/删除/静态访问 + 路径穿越防护。

用 FastAPI TestClient，不起真实 HTTP。立绘库 API 纯文件操作，不依赖 orchestrator。
用临时角色目录隔离（测试角色名 spadmintest，测完清理）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from fastapi.testclient import TestClient
from api import admin

client = TestClient(admin.app)

TEST_CHAR = "spadmintest"


def _cleanup():
    """清理测试角色的立绘目录。"""
    d = Path(__file__).resolve().parent.parent / "data" / "characters" / TEST_CHAR / "sprites"
    if d.exists():
        import shutil
        shutil.rmtree(d, ignore_errors=True)


def test_get_empty():
    """不存在的角色立绘库返回 {emotions:{}}。"""
    _cleanup()
    r = client.get(f"/api/sprites/{TEST_CHAR}")
    assert r.status_code == 200
    assert r.json() == {"emotions": {}}, f"空库应返回 {{emotions:{{}}}}: {r.json()}"
    print("✅ GET 空立绘库")


def test_put_and_get():
    """保存后能读回。"""
    lib = {"emotions": {"得意": {"image": "smug.png"}, "neutral": {"image": "neutral.png"}}}
    r = client.put(f"/api/sprites/{TEST_CHAR}", json={"lib": lib})
    assert r.status_code == 200 and r.json()["ok"]
    r = client.get(f"/api/sprites/{TEST_CHAR}")
    assert r.json()["emotions"]["得意"]["image"] == "smug.png"
    print("✅ PUT/GET 立绘库")


def test_image_upload_get_delete():
    """上传图片 → 静态访问 → 删除。"""
    # 上传（造一张最小 PNG）
    png_bytes = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
        "890000000d49444154789c63000100000005000101b380e3230000000049454e44ae426082"
    )
    r = client.post(
        f"/api/sprites/{TEST_CHAR}/image",
        files={"file": ("test.png", png_bytes, "image/png")},
    )
    assert r.status_code == 200, f"上传失败: {r.status_code} {r.text}"
    fname = r.json()["filename"]
    assert fname == "test.png", f"filename 错误: {fname}"
    print("✅ 上传图片")

    # 静态访问
    r = client.get(f"/api/sprites/{TEST_CHAR}/image/{fname}")
    assert r.status_code == 200 and r.headers["content-type"].startswith("image"), "应返回图片"
    print("✅ 静态访问图片")

    # 删除
    r = client.delete(f"/api/sprites/{TEST_CHAR}/image/{fname}")
    assert r.status_code == 200 and r.json()["ok"]
    r = client.get(f"/api/sprites/{TEST_CHAR}/image/{fname}")
    assert r.status_code == 404, "删后应 404"
    print("✅ 删除图片")


def test_path_traversal():
    """路径穿越防护：filename 含 ../ 或路径分隔符应被拒（400）。"""
    png = b"\x89PNG\r\n\x1a\n"
    # filename 含 ../
    r = client.post(
        f"/api/sprites/{TEST_CHAR}/image",
        files={"file": ("../../evil.png", png, "image/png")},
    )
    assert r.status_code == 400, f"穿越文件名应 400: {r.status_code}"
    # filename 含反斜杠
    r = client.post(
        f"/api/sprites/{TEST_CHAR}/image",
        files={"file": ("..\\evil.png", png, "image/png")},
    )
    assert r.status_code == 400, f"含反斜杠文件名应 400: {r.status_code}"
    print("✅ 路径穿越防护")


if __name__ == "__main__":
    print("=== 测试立绘库管理 API ===\n")
    _cleanup()
    test_get_empty()
    test_put_and_get()
    test_image_upload_get_delete()
    test_path_traversal()
    _cleanup()
    print("\n=== 全部测试通过 ===")
