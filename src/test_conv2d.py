from src.improved_model import BinarizingConv2dBatchNorm


def test_eval_applies_to_child_network():
    conv = BinarizingConv2dBatchNorm(
        in_channels=1,
        out_channels=8,
        kernel_size=4,
        stride=2,
        padding=0,
        binarize_parameters=False,
    )
    assert conv.training
    assert conv.batch_norm.training
    conv.eval()
    assert not conv.training
    assert not conv.batch_norm.training
    conv.train()
    assert conv.training
    assert conv.batch_norm.training
