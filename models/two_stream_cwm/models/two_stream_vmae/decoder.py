from models.two_stream_cwm.models.two_stream_vmae.two_stream.two_stream import TwoStreamTransformer


class TwoStreamDecoder(TwoStreamTransformer):
    def reset_classifier(
        self,
        num_classes_primary: int = 0,
        num_classes_secondary: int | None = None,
        output_primary_stream: bool = True,
        output_secondary_stream: bool = False,
    ) -> None:
        """Reset the output heads depending on which streams are outputting and output
        dim"""
        self.num_classes_primary = num_classes_primary
        self.num_classes_secondary = num_classes_secondary or self.num_classes_primary

        self.output_layers = self._build_output_config_and_layers(
            output_primary_stream=output_primary_stream,
            output_secondary_stream=output_secondary_stream,
            num_classes_primary=self.num_classes_primary,
            num_classes_secondary=self.num_classes_secondary,
        )
