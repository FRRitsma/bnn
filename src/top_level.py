from src.improved_model import BinarizingBase


class BinarizingTopLevel:
    @property
    def _child_binary_networks(self) -> list[BinarizingBase]:
        return [
            getattr(self, attr)
            for attr in dir(self)
            if not attr.startswith("_")
            and isinstance(getattr(self, attr, None), BinarizingBase)
        ]

    def set_scramble_distance(
        self, add_scramble_distance: float, decay_rate: float
    ) -> None:
        assert decay_rate <= 1
        for child_network in self._child_binary_networks:
            add_scramble_distance = child_network.inner_set_scramble_distance(
                add_scramble_distance, decay_rate
            )

    def binary_mode(self):
        for network in self._child_binary_networks:
            network.binary_mode()

    def train_mode(self):
        for network in self._child_binary_networks:
            network.train_mode()

    def clean_mode(self):
        for network in self._child_binary_networks:
            network.clean_mode()
