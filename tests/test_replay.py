from src.agent.replay import ReplayBuffer
from src.consts import ACTION_BUFFER_DIM
from src.consts import FRAME_STACK
from src.consts import IMAGE_HEIGHT
from src.consts import IMAGE_WIDTH
from src.consts import RAM_DIM
import torch

from conftest import rand_acts
from conftest import rand_img
from conftest import rand_ram


# from src.agent.network import DQN


_BATCH = 4
_CAP = 20


def _filled(n: int = _CAP) -> ReplayBuffer:
    buf = ReplayBuffer(capacity=_CAP)
    for _ in range(n):
        buf.add(img=rand_img(), ram=rand_ram(), acts_buf=rand_acts(), action=0, reward=1.0, done=False)

    return buf


def test_size_grows_to_capacity():
    buf = ReplayBuffer(capacity=_CAP)
    for i in range(_CAP + 5):
        buf.add(img=rand_img(), ram=rand_ram(), acts_buf=rand_acts(), action=0, reward=0.0, done=False)
        assert buf.size == min(i + 1, _CAP)


def test_sample_returns_none_when_not_enough():
    buf = ReplayBuffer(capacity=_CAP)
    for _ in range(_BATCH):  # need batch_size + 1
        buf.add(img=rand_img(), ram=rand_ram(), acts_buf=rand_acts(), action=0, reward=0.0, done=False)

    assert buf.sample(_BATCH) is None


def test_sample_tensor_shapes():
    batch = _filled().sample(_BATCH)
    assert batch is not None

    imgs, next_imgs, rams, next_rams, acts, next_acts, actions, rewards, dones = batch

    assert imgs.shape == (_BATCH, FRAME_STACK, IMAGE_HEIGHT, IMAGE_WIDTH)
    assert next_imgs.shape == (_BATCH, FRAME_STACK, IMAGE_HEIGHT, IMAGE_WIDTH)

    assert rams.shape == (_BATCH, RAM_DIM)
    assert next_rams.shape == (_BATCH, RAM_DIM)

    assert acts.shape == (_BATCH, ACTION_BUFFER_DIM)
    assert next_acts.shape == (_BATCH, ACTION_BUFFER_DIM)

    assert actions.shape == (_BATCH,)
    assert rewards.shape == (_BATCH,)
    assert dones.shape == (_BATCH,)


def test_sample_dtypes():
    batch = _filled().sample(_BATCH)
    assert batch is not None

    imgs, _, rams, _, acts, _, actions, rewards, dones = batch

    assert imgs.dtype == torch.uint8
    assert rams.dtype == torch.float32
    assert acts.dtype == torch.float32

    assert actions.dtype == torch.int64
    assert rewards.dtype == torch.float32
    assert dones.dtype == torch.bool


# def test_shapes_match_network_forward():
#     net = DQN()
#     net.eval()
#
#     batch = _filled().sample(_BATCH)
#     assert batch is not None
#
#     imgs, _, rams, _, acts, _, _, _, _ = batch
#
#     with torch.no_grad():
#         q = net(imgs, rams, acts)
#
#     assert q.shape == (_BATCH, NUM_ACTIONS)
