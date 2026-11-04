import torch

data = torch.load(
    "fashion_mnist_mlp.pth",
    map_location="cpu",
    weights_only=True
)

print("CÁC LỚP TRỌNG SỐ TRONG MÔ HÌNH:")

for name, weight in data.items():
    print("\nTên lớp:", name)
    print("Kích thước:", weight.shape)
    print("Giá trị mẫu:")
    print(weight[:2])