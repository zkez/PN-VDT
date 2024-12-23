import torch

def rearrange_3dct_to_2d(ct3d: torch.Tensor) -> torch.Tensor:
    # 检查输入形状
    c, d, h, w = ct3d.shape
    assert c == 1, "通道数必须为1"
    assert d == 16, f"深度 D=16 才能按4x4正确拼接,当前 D={d}"
    assert h == 64 and w == 64, "H=W=64 才能按4x4正确拼接"

    ct3d_5d = ct3d.reshape(1, 4, 4, 64, 64)  # shape: [1,4,4,64,64]
    
    rows = []
    for i in range(4):
        row_slices = []
        for j in range(4):
            slice_2d = ct3d_5d[0, i, j, :, :]
            row_slices.append(slice_2d)
        row_cat = torch.cat(row_slices, dim=1)
        rows.append(row_cat)

    full_2d = torch.cat(rows, dim=0) 

    ct2d = full_2d.unsqueeze(0) 
    return ct2d


if __name__ == "__main__":
    dummy = torch.randn(1, 16, 64, 64)  # [1,16,64,64]
    result = rearrange_3dct_to_2d(dummy)
    print(result.shape)  # 期望输出: [3,256,256]
    