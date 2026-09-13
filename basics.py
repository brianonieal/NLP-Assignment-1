"""basics.py -- PyTorch fundamentals for HW1 Section 5.1.

Covers tensor creation, tensor operations, mathematical operations, and the
PyTorch/NumPy bridge. Each function prints what it does so the output can be
read alongside the assignment text.

Run via:  python main.py basics
"""

import numpy as np
import torch


def _banner(title):
    """Print a section header so the console output maps to the assignment."""
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# 5.1.2  Tensors
# ---------------------------------------------------------------------------
def tensor_creation():
    """Demonstrate multiple ways to create and initialize PyTorch tensors."""
    _banner("5.1.2  TENSOR CREATION")

    # From a Python list. dtype is inferred from the contents.
    from_list = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print("from list          :", from_list.tolist())
    print("  shape", tuple(from_list.shape), " dtype", from_list.dtype)

    # A single float literal promotes the whole tensor to float32.
    from_list_float = torch.tensor([[1.0, 2, 3], [4, 5, 6]])
    print("with a float       : dtype", from_list_float.dtype)

    # From a NumPy array. NumPy defaults to float64 and torch.tensor copies that
    # dtype over, which is a common source of mismatches against float32 weights.
    from_numpy = torch.tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    print("from numpy         : dtype", from_numpy.dtype, "(float64, not float32)")

    # Shape-based constructors.
    print("zeros(2, 3)        :", torch.zeros(2, 3).tolist())
    print("ones(2, 3)         :", torch.ones(2, 3).tolist())
    print("eye(3)             :", torch.eye(3).tolist())
    print("arange(0, 10, 2)   :", torch.arange(0, 10, 2).tolist())
    print(
        "linspace(0, 1, 5)  :",
        [round(v, 3) for v in torch.linspace(0, 1, 5).tolist()],
    )

    # Random constructors. Seed first so output is reproducible.
    torch.manual_seed(0)
    print(
        "rand(2, 3)         :",
        [[round(v, 3) for v in r] for r in torch.rand(2, 3).tolist()],
    )
    print(
        "randn(2, 3)        :",
        [[round(v, 3) for v in r] for r in torch.randn(2, 3).tolist()],
    )

    # "_like" constructors copy shape and dtype from an existing tensor.
    print("zeros_like(x)      :", torch.zeros_like(from_list).tolist())

    # Explicit dtype and device.
    typed = torch.zeros(2, 2, dtype=torch.long, device="cpu")
    print("explicit dtype     :", typed.dtype, " device", typed.device)

    return from_list


# ---------------------------------------------------------------------------
# 5.1.3  Tensor operations
# ---------------------------------------------------------------------------
def tensor_operations():
    """Demonstrate indexing, slicing, reshaping, and reduction, as in NumPy."""
    _banner("5.1.3  TENSOR OPERATIONS")

    x = torch.arange(12).reshape(3, 4)
    print("x =", x.tolist(), " shape", tuple(x.shape))

    # Indexing and slicing follow NumPy rules.
    print("x[0]               :", x[0].tolist())
    print("x[:, 1]            :", x[:, 1].tolist())
    print("x[1:, 2:]          :", x[1:, 2:].tolist())

    # view() needs a contiguous tensor and shares storage.
    # reshape() falls back to a copy when it has to.
    print("x.view(4, 3)       :", x.view(4, 3).tolist())
    print("x.reshape(2, 6)    :", x.reshape(2, 6).tolist())
    print("x.view(-1)         :", x.view(-1).tolist(), "(-1 infers the dimension)")

    # Transpose.
    print("x.T shape          :", tuple(x.T.shape))

    # Adding and removing singleton dimensions, which is how the batch
    # dimension gets built when a DataLoader is not doing it for you.
    v = torch.tensor([1.0, 2.0, 3.0])
    print("v.unsqueeze(0)     :", tuple(v.unsqueeze(0).shape))
    print("v.unsqueeze(1)     :", tuple(v.unsqueeze(1).shape))
    print("squeeze back       :", tuple(v.unsqueeze(0).squeeze().shape))

    # Reductions. dim= picks the axis that collapses.
    xf = x.float()
    print("x.sum()            :", xf.sum().item())
    print("x.sum(dim=0)       :", xf.sum(dim=0).tolist(), "(collapses rows)")
    print("x.mean(dim=1)      :", xf.mean(dim=1).tolist(), "(collapses columns)")
    print(
        "x.max(dim=1)       : values",
        xf.max(dim=1).values.tolist(),
        " indices",
        xf.max(dim=1).indices.tolist(),
    )
    print("x.argmax(dim=1)    :", xf.argmax(dim=1).tolist())

    # cat joins along an existing axis, stack creates a new one.
    # create_tensor_dataset uses stack.
    a, b = torch.ones(2, 3), torch.zeros(2, 3)
    print("cat(dim=0) shape   :", tuple(torch.cat([a, b], dim=0).shape))
    print("stack(dim=0) shape :", tuple(torch.stack([a, b], dim=0).shape))

    return x


# ---------------------------------------------------------------------------
# 5.1.4  Mathematical operations
# ---------------------------------------------------------------------------
def math_operations():
    """Demonstrate element-wise arithmetic, broadcasting, and matmul."""
    _banner("5.1.4  MATHEMATICAL OPERATIONS")

    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    # Function form and operator form are the same thing.
    print("torch.add(a, b)    :", torch.add(a, b).tolist())
    print("a + b              :", (a + b).tolist(), "(identical)")
    print(
        "torch.mul(a, b)    :",
        torch.mul(a, b).tolist(),
        "(element-wise, NOT matmul)",
    )
    print("a * b              :", (a * b).tolist())
    print("a - b              :", (a - b).tolist())
    print(
        "a / b              :",
        [[round(v, 3) for v in r] for r in (a / b).tolist()],
    )

    # Matrix multiplication. This is the operation nn.Linear performs.
    print("torch.matmul(a, b) :", torch.matmul(a, b).tolist())
    print("a @ b              :", (a @ b).tolist(), "(identical)")

    # The classifier computes xW^T + b, so shapes are
    # [batch, in_features] @ [in_features, out_features] -> [batch, out_features].
    torch.manual_seed(0)
    batch = torch.randn(4, 8)  # 4 examples, 8 features each
    weight = torch.randn(8, 2)  # 8 features in, 2 classes out
    print("[4,8] @ [8,2] shape:", tuple((batch @ weight).shape))

    # Broadcasting: the bias row vector is stretched across the batch.
    bias = torch.randn(2)
    print(
        "adding bias [2]    :",
        tuple((batch @ weight + bias).shape),
        "(broadcast over the batch)",
    )

    # In-place operations carry a trailing underscore and overwrite the input.
    c = torch.ones(2, 2)
    c.add_(5)
    print("in-place add_(5)   :", c.tolist())

    # Common element-wise math.
    print(
        "exp, log, sqrt     :",
        round(torch.exp(torch.tensor(1.0)).item(), 4),
        round(torch.log(torch.tensor(float(np.e))).item(), 4),
        round(torch.sqrt(torch.tensor(9.0)).item(), 4),
    )
    print("sigmoid(0)         :", torch.sigmoid(torch.tensor(0.0)).item())
    print(
        "softmax([1,2,3])   :",
        [
            round(v, 4)
            for v in torch.softmax(torch.tensor([1.0, 2.0, 3.0]), dim=0).tolist()
        ],
    )

    return a @ b


# ---------------------------------------------------------------------------
# 5.1.5  PyTorch and NumPy bridge
# ---------------------------------------------------------------------------
def torch_numpy():
    """Convert between torch.Tensor and numpy.ndarray in both directions."""
    _banner("5.1.5  PYTORCH / NUMPY BRIDGE")

    # tensor -> ndarray
    t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    arr = t.numpy()
    print(
        "t.numpy()          :",
        arr.tolist(),
        " type",
        type(arr).__name__,
        " dtype",
        arr.dtype,
    )

    # ndarray -> tensor
    arr2 = np.array([[5.0, 6.0], [7.0, 8.0]])
    t2 = torch.from_numpy(arr2)
    print("from_numpy(arr)    :", t2.tolist(), " dtype", t2.dtype)

    # Both directions SHARE memory. Mutating one mutates the other.
    shared = torch.ones(3)
    shared_np = shared.numpy()
    shared_np[0] = 99.0
    print(
        "shared memory      : tensor is now",
        shared.tolist(),
        "after editing the ndarray",
    )
    print("  use .clone() or np.copy() when you need an independent buffer")

    # NumPy defaults to float64 while torch layers default to float32, so the
    # .float() cast in featurize() is not optional.
    f64 = torch.from_numpy(np.array([1.0, 2.0]))
    print("float64 -> float32 :", f64.dtype, "->", f64.float().dtype)

    # .item() pulls a Python scalar out of a one-element tensor.
    print(".item()            :", torch.tensor([3.5]).item())

    # A tensor that requires grad must be detached before .numpy() works.
    g = torch.ones(2, requires_grad=True)
    print("detach().numpy()   :", g.detach().numpy().tolist())

    return arr


def run_all():
    """Run every Section 5.1 demonstration in order."""
    tensor_creation()
    tensor_operations()
    math_operations()
    torch_numpy()
    print()
    print("basics.py complete.")


if __name__ == "__main__":
    run_all()
