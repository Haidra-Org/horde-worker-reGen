from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopSkippedStatus


def test_skipped_status_handles_unknown_fields() -> None:
    """Test that the ImageGenerateJobPopSkippedStatus model can handle unknown fields."""
    # This test is to ensure that the model can handle unknown fields without throwing an error and that it can be
    # printed without error.
    skipped_status = ImageGenerateJobPopSkippedStatus(
        max_pixels=100,
        testing_field=1,  # type: ignore
    )

    assert skipped_status.max_pixels == 100

    dumped_skipped_status = skipped_status.model_dump()
    assert dumped_skipped_status["max_pixels"] == 100
    assert dumped_skipped_status["testing_field"] == 1

    print(skipped_status)
