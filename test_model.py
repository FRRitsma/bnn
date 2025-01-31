from src.improved_model import BinarizingNetwork, ModelMode


def test_child_mode_transfer_model_mode():
    # Create a parent and a child network
    parent = BinarizingNetwork()
    child = BinarizingNetwork()

    # Assign the child to the parent
    parent.child = child  # type: ignore

    # Initially, both should be in train mode
    assert parent.model_mode == ModelMode.train
    assert child.model_mode == ModelMode.train
    assert parent.scramble
    assert child.scramble

    # Switch parent to evaluation mode
    parent.eval_mode()

    # Check if both parent and child are in evaluation mode
    assert parent.model_mode == ModelMode.evaluation
    assert child.model_mode == ModelMode.evaluation
    assert not parent.scramble
    assert not child.scramble

    # Switch parent back to train mode
    parent.train_mode()

    assert parent.model_mode == ModelMode.train
    assert child.model_mode == ModelMode.train
    assert parent.scramble
    assert child.scramble


def test_child_transfer_scramble_distance():
    # Create a parent and a child network
    parent = BinarizingNetwork()
    child = BinarizingNetwork()

    # Assign the child to the parent
    parent.child = child  # type: ignore

    assert parent.scramble_distance == 0.0
    assert child.scramble_distance == 0.0

    parent.set_scramble_distance(1.0)

    assert parent.scramble_distance == 1.0
    assert child.scramble_distance == 1.0
